"""
generate.py
===========
Use a trained checkpoint: tokenize an input prompt, run it through the model,
and output text. Two things happen, matching the architecture:

  1. READ  -- the prompt is chunked, each chunk encoded to a latent and run
     through the HRM inner loop, building up the gestalt memory and the running
     H-state ("thought"). This is the model reading the prompt as self-content.
  2. GENERATE -- for each new chunk: predict the next latent (JEPA
     latent_predictor), form the next thought via the HRM loop, and let the
     Talker autoregressively decode tokens for that thought (conditioned on the
     thought + gestalt memory). The generated chunk is re-encoded to a latent to
     seed the next step. Decoded to text with the gpt2 tokenizer.

NOTE: the shipped checkpoint is a tiny smoke model (~1.5M tokens); output will
NOT be coherent. This script demonstrates the *inference path*, not quality.

Run:  python generate.py "Your prompt here"
      python generate.py --score "text to score perplexity on"
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


def load():
    ckpt = torch.load(CKPT, map_location="cpu")
    cfg = ModelConfig(**ckpt["model_cfg"])
    tok_src = ckpt.get("tokenizer_path") if os.path.isdir(ckpt.get("tokenizer_path", "")) else TOKENIZER_DIR
    chunker, _ = build_regex_gpt2_chunker(cfg, tok_src)
    model = LatentThoughtModel(cfg, chunker)
    model.load_state_dict(ckpt["model_state"])
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
    """Autoregressively decode one chunk's tokens from a thought."""
    ids = []
    for _ in range(max_len):
        inp = torch.zeros(1, max_len, dtype=torch.long)
        for j, g in enumerate(ids):
            inp[0, j] = g
        logits = model.talker(inp, thought, memory)[0, len(ids)]   # (vocab,)
        logits[PAD] = -1e9                                          # never emit PAD
        if greedy:
            nxt = int(logits.argmax())
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = int(torch.multinomial(probs, 1))
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
        pred_latent = model.latent_predictor(last_latent)              # JEPA: next-segment latent
        h_state, _ = model.hrm_loop(pred_latent, memory, None, h_state=h_state, l_state=l_state,
                                    grad_window=5, use_act=False)
        l_state = h_state
        ids = talker_decode(model, h_state, memory, cfg.max_chunk_len, temperature, greedy)
        memory.write(h_state.detach(), SELF)
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
        total_nll += grounded_nll_loss(logits, chunk_ids, chunk_ids != 0).item()
        n += 1
    avg = total_nll / max(n, 1)
    return avg, math.exp(min(avg, 20))


def main():
    args = sys.argv[1:]
    if not os.path.exists(CKPT):
        raise SystemExit(f"no checkpoint at {CKPT} -- run train_real.py first")
    do_score = args and args[0] == "--score"
    if do_score:
        args = args[1:]
    prompt = " ".join(args) if args else "The history of science shows that"

    model, chunker, cfg, ckpt = load()
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
