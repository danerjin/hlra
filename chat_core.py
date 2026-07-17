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


def load_dialogue_checkpoint(ckpt_path):
    """Load a Stage-F chatbot: (model, adapter, chunker, cfg, ckpt_meta). Reuses
    load_checkpoint for the model/chunker/cfg (the config is read from the
    checkpoint, so the Stage-F flags it was trained with are honored), then loads
    the DialogueAdapter (the learned response seed). Works on a plain A→E
    checkpoint too -- the adapter is then zero-init, i.e. UNTRAINED dialogue."""
    from dialogue import DialogueAdapter
    model, chunker, cfg, ckpt = load_checkpoint(ckpt_path)
    adapter = DialogueAdapter(cfg.d_latent)
    if "adapter_state" in ckpt:
        adapter.load_state_dict(ckpt["adapter_state"])
    adapter.eval()
    return model, adapter, chunker, cfg, ckpt


def new_dialogue_session(model, adapter, chunker, cfg):
    """A fresh DialogueSession (the full Stage-F two-lane serving: input lane +
    response seed + cross-turn gestalt memory). One session = one conversation."""
    from dialogue import DialogueSession
    return DialogueSession(model, adapter, chunker, cfg, use_act=True)


def dialogue_reply(session, text, n_chunks=6, temperature=0.9, greedy=False,
                   use_end_head=True, end_threshold=0.5):
    """One chatbot turn through the persistent session (memory carries across
    calls). Returns (reply_chunks: list[str], read_chunks: list[str]).

    `n_chunks` is a CAP: a checkpoint trained with StageFConfig.end_weight > 0
    ends its own turn when P(end) > `end_threshold` (STAGE_F.md §2.1). An
    untrained end head effectively never fires, so this degrades to exactly
    n_chunks chunks -- the old behavior."""
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
