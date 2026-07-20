"""
chat_core.py
============
Shared inference helpers for the interactive testers (chat.py CLI, web_chat.py
server). Thin wrappers over files/generate.py -- the SAME inference path the
project's generate.py uses -- so a checkpoint behaves identically here and there.

Nothing model-specific is hard-coded: load_checkpoint() reads the model config
out of the checkpoint itself, so this works for any preset (smoke/small/base/...)
the final run produces.
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.join(HERE, "files")
if FILES not in sys.path:
    sys.path.insert(0, FILES)

import torch  # noqa: E402
from generate import (load, generate as _generate, score as _score,  # noqa: E402
                      _decode, talker_decode as _talker_decode)

# Rare control char used to split generate()'s joined output back into a list of
# per-chunk strings without re-implementing the (subtle) generation loop.
_SENT = "\x1f"


def load_checkpoint(ckpt_path):
    """Resolve + load a checkpoint. Returns (model, chunker, cfg, ckpt_meta)."""
    path = os.path.expanduser(ckpt_path.strip())
    if not os.path.isabs(path):
        # try as-given (cwd) first, then project-relative
        cand = path if os.path.exists(path) else os.path.join(HERE, path)
        path = cand
    if not os.path.exists(path):
        raise FileNotFoundError(f"no checkpoint at {ckpt_path!r} (resolved {path!r})")
    return load(path)


def input_chunks(chunker, text):
    """How the SaT-Capped chunker splits `text` into the chunk-level 'thoughts'
    the model reads -- each chunk decoded back to a string. This is exactly the
    segmentation read_prompt/score operate on."""
    ct, cm = chunker.chunk_batch([text])
    tok = chunker.tokenizer
    parts = [_decode(tok, ct[0, t]).strip()
             for t in range(ct.shape[1]) if bool(cm[0, t])]
    return [p for p in parts if p]


def generate_chunks(model, chunker, cfg, text, n_chunks=3, temperature=0.9, greedy=False):
    """Generate a continuation, returned as a LIST of per-chunk strings (one per
    generated 'thought')."""
    joined = _generate(model, chunker, cfg, text, n_chunks=n_chunks,
                       temperature=temperature, greedy=greedy, separator=_SENT)
    return [c for c in joined.split(_SENT) if c]


def score_text(model, chunker, cfg, text):
    """Teacher-forced reconstruction perplexity of `text`. Returns (nll, ppl)."""
    return _score(model, chunker, cfg, text)


@torch.no_grad()
def autoencode(model, chunker, cfg, text, temperature=0.9, greedy=True):
    """Pure encoder->Talker autoencoder round-trip, per chunk. This is exactly the
    `forward_grounded` reconstruction anchor's inference form: encode a chunk to its
    ENCODER-space latent, decode THAT SAME latent back to tokens with the codec
    Talker (empty memory, NO HRM loop). It isolates the codec from the reasoning
    loop, so it answers "does the autoencoder round-trip?" without prediction in the
    way.

    Defaults to greedy decode -- for checking codec fidelity you want the argmax
    reconstruction, not a sampled one. Returns a list of per-chunk dicts:
        {"original": str, "recon": str, "latent": Tensor(d_latent,)}
    where `latent` is the RAW encoder latent vector (detached, on CPU)."""
    ct, cm = chunker.chunk_batch([text])          # (1, C, L)
    tok = chunker.tokenizer
    out = []
    for t in range(ct.shape[1]):
        if not bool(cm[0, t]):
            continue
        chunk_ids = ct[:, t, :]                    # (1, L)
        original = _decode(tok, chunk_ids[0]).strip()
        if not original:
            continue
        latent = model.chunk_encoder(chunk_ids, chunk_ids != 0)   # (1, d_latent)
        ids = _talker_decode(model, latent, cfg, temperature=temperature, greedy=greedy)
        recon = _decode(tok, ids).strip()
        out.append({"original": original,
                    "recon": recon,
                    "latent": latent[0].detach().cpu()})
    return out


def autoencode_json(model, chunker, cfg, text, temperature=0.9):
    """JSON-serializable form of autoencode() for the web UI: the raw latent tensor
    is rendered to a plain float list, with norm/std precomputed so the frontend
    (which never imports torch) can display them directly."""
    rows = autoencode(model, chunker, cfg, text, temperature=temperature, greedy=True)
    out = []
    for r in rows:
        lat = r["latent"]
        out.append({
            "original": r["original"],
            "recon": r["recon"],
            "exact": r["recon"] == r["original"],
            "dim": int(lat.numel()),
            "norm": float(lat.norm()),
            "std": float(lat.std()),
            "latent": [round(float(v), 6) for v in lat.tolist()],
        })
    return out


def load_dialogue_checkpoint(ckpt_path):
    """Load a Stage-F chatbot: (model, adapter, chunker, cfg, ckpt_meta). Reuses
    load_checkpoint for the model/chunker/cfg (the config is read from the
    checkpoint, so the Stage-F flags it was trained with are honored), then loads
    the DialogueAdapter (the learned response seed). Works on a plain A→E
    checkpoint too -- the adapter is then zero-init, i.e. UNTRAINED dialogue."""
    from dialogue import DialogueAdapter
    model, chunker, cfg, ckpt = load_checkpoint(ckpt_path)
    adapter = DialogueAdapter(cfg.d_latent)
    if ckpt.get("adapter_state") is not None:
        # Non-strict: a Stage-F checkpoint written before the turn-end gate has no
        # `end_head.*` and would otherwise raise "Missing key(s)" here and refuse to
        # serve at all. Missing end_head just means an untrained gate, which never
        # fires (bias -4.0) -- i.e. the old fixed-length behavior. Anything else
        # missing IS a real mismatch and must not pass silently.
        missing, unexpected = adapter.load_state_dict(ckpt["adapter_state"], strict=False)
        # ALL of end_head.* missing == a genuine pre-gate checkpoint; a PARTIAL miss is
        # corruption and must not be waved through. Exact names, not a prefix.
        end_keys = {"end_head.weight", "end_head.bias"}
        miss_end = end_keys & set(missing)
        pre_gate = miss_end == end_keys
        other = [k for k in missing if k not in end_keys] + (
            sorted(miss_end) if miss_end and not pre_gate else [])
        if other or unexpected:
            raise RuntimeError(f"adapter state mismatch: missing={other} "
                               f"unexpected={list(unexpected)}")
        if pre_gate:
            print("[chat_core] NOTE: checkpoint predates the turn-end gate; replies "
                  "will run to max_chunks.")
    adapter.eval()
    return model, adapter, chunker, cfg, ckpt


def new_dialogue_session(model, adapter, chunker, cfg, ckpt=None):
    """A fresh DialogueSession (the full Stage-F two-lane serving: input lane +
    response seed + cross-turn gestalt memory). One session = one conversation.

    Pass the `ckpt` dict from load_dialogue_checkpoint so serving matches training:

      * the turn-end gate is enabled only when that checkpoint actually TRAINED it
        (`end_gate_trained`). Otherwise it stays off -- an untrained end_head fires
        at P=0.018 per CHUNK (8.7% of 6-chunk replies), so switching it on blindly
        would truncate replies at random.
      * ACT is run the way training ran it (`stage_f_use_act`). Serving with ACT on
        a --no-act checkpoint (or vice versa) changes the loop depth, hence h_t,
        hence what the gate sees. Defaults True for checkpoints written before this
        was recorded -- which is what they all trained with."""
    from dialogue import DialogueSession
    ck = ckpt or {}
    return DialogueSession(model, adapter, chunker, cfg,
                           use_act=bool(ck.get("stage_f_use_act", True)),
                           use_end_head=bool(ck.get("end_gate_trained", False)))


def dialogue_reply(session, text, n_chunks=6, temperature=0.9, greedy=False,
                   use_end_head=None, end_threshold=0.5):
    """One chatbot turn through the persistent session (memory carries across
    calls). Returns (reply_chunks: list[str], read_chunks: list[str]).

    `n_chunks` is a CAP. `use_end_head=None` inherits the session's setting, which
    new_dialogue_session turns on only for a checkpoint whose `end_gate_trained`
    flag is set; otherwise the reply runs to n_chunks exactly as before the gate
    existed. Forcing it True on an untrained gate stops 8.7% of 6-chunk replies at
    random (P=0.018 per chunk x 5 chances to stop early) -- don't."""
    read = input_chunks(session.chunker, text)
    joined = session.reply(text, max_chunks=n_chunks, temperature=temperature,
                           greedy=greedy, separator=_SENT,
                           use_end_head=use_end_head, end_threshold=end_threshold)
    reply = [c for c in joined.split(_SENT) if c]
    return reply, read


def ckpt_summary(cfg, ckpt):
    """Small dict of checkpoint facts for display/debug."""
    return {
        "stage_reached": ckpt.get("stage_reached"),
        "global_step": ckpt.get("global_step"),
        "vocab_size": ckpt.get("vocab_size", getattr(cfg, "vocab_size", None)),
        "d_model": getattr(cfg, "d_model", None),
        "max_chunk_len": getattr(cfg, "max_chunk_len", None),
        "max_chunks_per_doc": getattr(cfg, "max_chunks_per_doc", None),
        "note": ckpt.get("note", ""),
    }
