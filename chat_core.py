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

from generate import load, generate as _generate, score as _score, _decode  # noqa: E402

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
