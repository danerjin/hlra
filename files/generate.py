"""
generate.py
===========
Use a trained checkpoint: tokenize an input prompt, run it through the model,
and output text. Two things happen, matching the architecture:

  1. READ  -- the prompt is chunked, each chunk encoded to a latent and run
     through the HRM inner loop, building up the gestalt memory and the running
     H-state ("thought"). This is the model reading the prompt as self-content.
  2. GENERATE -- for each new chunk: predict the next latent in encoder space
     (gen_predictor; notes §15.1), form the next thought via the HRM loop, and let the
     Talker autoregressively decode tokens for that thought (conditioned on the
     thought + gestalt memory). The generated chunk is re-encoded to a latent to
     seed the next step. Decoded to text with the gpt2 tokenizer.

NOTE: the shipped checkpoint is a tiny smoke model (~1.5M tokens); output will
NOT be coherent. This script demonstrates the *inference path*, not quality.

Run:  python generate.py "Your prompt here"
      python generate.py --score "text to score perplexity on"
      python generate.py --ckpt runs/scaled/model.pt "prompt"   # scaled-run checkpoint
"""
from __future__ import annotations

import os
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(PROJECT, ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
TOKENIZER_DIR = os.path.join(PROJECT, "gpt2_tok")

import sys
import math

import torch

from config import ModelConfig
from model import LatentThoughtModel, SELF
from gestalt_memory import GestaltMemoryBank
from data import build_regex_gpt2_chunker, PAD
from losses import grounded_nll_loss

CKPT = os.path.join(PROJECT, "runs", "model.pt")


# ModelConfig fields renamed after some checkpoints were saved (§20.2 rename:
# the Parcae-named gate became DiagonalDecayGate). Only the *config field names*
# changed -- module/state_dict keys are identical -- so old checkpoints stay
# loadable by mapping the saved names forward here.
_LEGACY_CFG_FIELDS = {"parcae_min_decay": "decay_min", "parcae_max_decay": "decay_max"}


def load(ckpt_path: str = CKPT):
    import dataclasses
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # carries non-tensor RNG/cfg state
    raw_cfg = dict(ckpt["model_cfg"])
    for old, new in _LEGACY_CFG_FIELDS.items():
        if old in raw_cfg and new not in raw_cfg:
            raw_cfg[new] = raw_cfg.pop(old)
    known = {f.name for f in dataclasses.fields(ModelConfig)}
    dropped = sorted(k for k in raw_cfg if k not in known)
    if dropped:
        print(f"[generate] WARNING: ignoring unknown checkpoint config fields {dropped} "
              f"(older/newer code version); defaults will be used where they mattered.")
    cfg = ModelConfig(**{k: v for k, v in raw_cfg.items() if k in known})
    tok_src = ckpt.get("tokenizer_path") if os.path.isdir(ckpt.get("tokenizer_path", "")) else TOKENIZER_DIR
    chunker, _ = build_regex_gpt2_chunker(cfg, tok_src)
    model = LatentThoughtModel(cfg, chunker)
    # strict=False so checkpoints predating the gen_predictor head still load;
    # report anything missing so degraded generation is attributable.
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:
        mods = sorted({k.split(".")[0] for k in missing})
        print(f"[generate] WARNING: checkpoint predates module(s) {mods}; they are "
              f"randomly initialized -- latent prediction will be untrained. "
              f"(Pre-§25 checkpoints trained a different predictor head and cannot "
              f"drive the HRM-loop predictor.)")
    if unexpected:
        raise SystemExit(f"checkpoint has unexpected keys (wrong model?): {unexpected[:5]}")
    model.eval()
    return model, chunker, cfg, ckpt


def _decode(tok, ids):
    ids = [int(i) for i in ids if int(i) != PAD]
    return tok.decode(ids) if ids else ""


@torch.no_grad()
def read_prompt(model, chunker, cfg, prompt):
    """Chunk + encode the prompt through the HRM loop; return (memory, h, l, last_latent)."""
    ct, cm = chunker.chunk_batch([prompt])           # (1, C, L)
    memory = GestaltMemoryBank(cfg.memory_capacity, cfg.d_model)
    h_state = l_state = last_latent = None
    for t in range(ct.shape[1]):
        if not bool(cm[0, t]):
            continue
        chunk_ids = ct[:, t, :]
        latent = model.chunk_encoder(chunk_ids, chunk_ids != 0)
        h_state, _ = model.hrm_loop(latent, memory, None, h_state=h_state, l_state=l_state,
                                    grad_window=5, use_act=False)
        l_state = h_state
        memory.write(h_state.detach(), SELF)
        last_latent = latent
    if last_latent is None:                          # empty/too-short prompt
        last_latent = torch.zeros(1, cfg.d_model)
    return memory, h_state, l_state, last_latent


@torch.no_grad()
def talker_decode(model, thought, memory, max_len, temperature=0.9, greedy=False):
    """
    Autoregressively decode one chunk's tokens from a thought. PAD (id 0) is
    the trained end-of-chunk signal (model.forward_grounded supervises the
    first pad position of every short chunk), so emitting PAD stops the chunk.
    It is only banned at position 0 to rule out degenerate empty chunks --
    e.g. under a checkpoint that predates the end-of-chunk supervision, where
    the PAD logit is untrained noise.
    """
    ids = []
    for _ in range(max_len):
        inp = torch.zeros(1, max_len, dtype=torch.long)
        for j, g in enumerate(ids):
            inp[0, j] = g
        logits = model.talker(inp, thought, memory)[0, len(ids)]   # (vocab,)
        if not ids:
            logits[PAD] = -1e9
        if greedy:
            nxt = int(logits.argmax())
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = int(torch.multinomial(probs, 1))
        if nxt == PAD:
            break                                                   # trained end-of-chunk
        ids.append(nxt)
        if len(ids) >= max_len:
            break
    return ids


@torch.no_grad()
def generate(model, chunker, cfg, prompt, n_chunks=3, temperature=0.9, greedy=False):
    tok = chunker.tokenizer
    memory, h_state, l_state, last_latent = read_prompt(model, chunker, cfg, prompt)
    out_chunks = []
    for _ in range(n_chunks):
        # Next-latent prediction in ENCODER space: the HRM loop itself produces it
        # (run the loop on last_latent, read the next latent off the thought via
        # pred_head -- the same map forward_self_supervised trains).
        pred_latent, _ = model.predict_next_latent(last_latent, memory, h_state=h_state,
                                                   l_state=l_state, grad_window=5, use_act=False)
        h_state, _ = model.hrm_loop(pred_latent, memory, None, h_state=h_state, l_state=l_state,
                                    grad_window=5, use_act=False)
        l_state = h_state
        # Write BEFORE decoding, matching training (model.forward_grounded) and
        # score(): the Talker is always trained with its own thought as the
        # newest memory slot, so decoding must see the same memory state.
        memory.write(h_state.detach(), SELF)
        ids = talker_decode(model, h_state, memory, cfg.max_chunk_len, temperature, greedy)
        # re-encode the produced chunk to seed the next step
        gen = torch.zeros(1, cfg.max_chunk_len, dtype=torch.long)
        for j, g in enumerate(ids[:cfg.max_chunk_len]):
            gen[0, j] = g
        last_latent = model.chunk_encoder(gen, gen != 0)
        out_chunks.append(_decode(tok, ids))
    return " ".join(c for c in out_chunks if c).strip()


@torch.no_grad()
def score(model, chunker, cfg, text):
    """Teacher-forced perplexity of `text` under the model."""
    ct, cm = chunker.chunk_batch([text])
    memory = GestaltMemoryBank(cfg.memory_capacity, cfg.d_model)
    h_state = l_state = None
    total_nll, n = 0.0, 0
    for t in range(ct.shape[1]):
        if not bool(cm[0, t]):
            continue
        chunk_ids = ct[:, t, :]
        latent = model.chunk_encoder(chunk_ids, chunk_ids != 0)
        h_state, _ = model.hrm_loop(latent, memory, None, h_state=h_state, l_state=l_state,
                                    grad_window=5, use_act=False)
        l_state = h_state
        memory.write(h_state.detach(), SELF)
        logits = model.talker(chunk_ids, h_state, memory)
        # Token-weighted accumulation: grounded_nll_loss returns the per-token
        # MEAN within this chunk, so weight it by the chunk's real-token count.
        # (A plain mean-of-chunk-means over-weights short chunks and is not
        # comparable with baseline_gpt.eval_ppl's token-weighted number.)
        n_tok = int((chunk_ids != 0).sum())
        total_nll += grounded_nll_loss(logits, chunk_ids, chunk_ids != 0).item() * n_tok
        n += n_tok
    avg = total_nll / max(n, 1)
    return avg, math.exp(min(avg, 20))


def main():
    args = sys.argv[1:]
    ckpt_path = CKPT
    if "--ckpt" in args:                       # e.g. --ckpt runs/scaled/model.pt
        i = args.index("--ckpt")
        ckpt_path = args[i + 1]
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(PROJECT, ckpt_path)
        args = args[:i] + args[i + 2:]
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"no checkpoint at {ckpt_path} -- run train_real.py first "
                         f"(or pass --ckpt runs/scaled/model.pt)")
    do_score = args and args[0] == "--score"
    if do_score:
        args = args[1:]
    prompt = " ".join(args) if args else "The history of science shows that"

    model, chunker, cfg, ckpt = load(ckpt_path)
    print(f"[generate] checkpoint stage={ckpt.get('stage_reached')} "
          f"vocab={ckpt.get('vocab_size')}  ({ckpt.get('note','')})\n")

    if do_score:
        nll, ppl = score(model, chunker, cfg, prompt)
        print(f"prompt: {prompt!r}\n  avg NLL/token = {nll:.3f}   perplexity = {ppl:.1f}")
        return

    print(f"prompt:     {prompt!r}")
    cont = generate(model, chunker, cfg, prompt, n_chunks=3, temperature=0.9)
    print(f"generated:  {cont!r}")
    print("\n(reminder: smoke-scale model -- output is not expected to be coherent)")


if __name__ == "__main__":
    main()
