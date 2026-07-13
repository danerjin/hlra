"""
data_prep.py
============
Offline pre-chunking: run the (expensive) SaT-Capped chunker + tokenizer over a
text corpus ONCE and write sharded chunk tensors to disk, so training does zero
chunking per epoch and can use DataLoader workers. This is the main data-side
scaling change -- on-the-fly chunking in the training loop does not scale.

Output layout (config.DataConfig.cache_dir):
    manifest.json            # shard list, counts, and the chunk-dim config
    shard_00000.pt ...       # dict of int32/bool tensors per shard

The real path uses the SAME chunker train.py runs on the fly: Thought Gestalt's
"SaT Capped" preprocessing (§3.1/§5.1) -- SaT-predicted sentence boundaries plus
punctuation-aware length capping -- over the gpt2 tokenizer, so the cached corpus
matches the intended pipeline exactly (not a regex approximation). Needs
`datasets wtpsplit transformers` and the local gpt2 tokenizer in ../gpt2_tok; the
SaT model (DataConfig.sat_model_name, default sat-3l-sm) downloads from the HF hub
on first run and is cached under HF_HOME thereafter.

Run (FULL-RUN cache -- the real Stages A-E mixture, config.DataConfig.sources,
interleaved by weight and shuffled):
    python data_prep.py --mixture --preset small-w3 --max-tokens 3000000000 --out chunk_cache

Run (single-corpus cache, e.g. a quick smoke; --name for multi-config datasets so
`datasets` doesn't silently pick the DEFAULT config -- for fineweb-edu the full
multi-TB corpus rather than sample-10BT):
    python data_prep.py --dataset HuggingFaceFW/fineweb-edu --name sample-10BT \
        --streaming --preset small-w3 --max-tokens 100000000 --out chunk_cache_smoke

Run (offline synthetic, no downloads / no wtpsplit -- stub SaT-Capped chunker):
    python data_prep.py --offline --preset smoke --docs 2000
"""
from __future__ import annotations

import os
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Keep the HF cache in-project. NOTE: we deliberately do NOT force
# TRANSFORMERS_OFFLINE/HF_HUB_OFFLINE here -- the SaT model (sat-3l-sm) must be
# fetched from the hub on first use. The gpt2 tokenizer is loaded from the local
# gpt2_tok dir (below), so it stays offline/deterministic regardless.
os.environ.setdefault("HF_HOME", os.path.join(PROJECT, ".hf_cache"))

import argparse
import json

import torch

from config import model_config, MODEL_PRESETS, DataConfig
from data import (
    MANIFEST, build_offline_chunker, chunk_text_example,
    iter_hf_single, iter_hf_mixture, SyntheticTextCorpus,
)

TOKENIZER_DIR = os.path.join(PROJECT, "gpt2_tok")


def prepare(text_iter, chunker, model_cfg, out_dir, shard_size, min_chunks,
            vocab_size, max_examples=None, max_tokens=None):
    # Refuse to prep into a non-empty directory: mixing shards from two prep
    # runs leaves a stale manifest next to a blend of old and new shards, and
    # the load-time consistency check only catches it when the totals differ.
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        raise SystemExit(f"cache dir {out_dir} is not empty -- data_prep must write into a "
                         f"FRESH directory (delete it or pass a new --out).")
    os.makedirs(out_dir, exist_ok=True)
    window = model_cfg.recent_token_window
    buf, shard_files, counts = [], [], []
    total, tokens, shard_idx = 0, 0, 0

    def flush():
        nonlocal shard_idx
        if not buf:
            return
        shard = {
            "chunk_tensor": torch.stack([b[0] for b in buf]).to(torch.int32),
            "chunk_mask": torch.stack([b[1] for b in buf]),
            "raw_ids": torch.stack([b[2] for b in buf]).to(torch.int32),
            "raw_mask": torch.stack([b[3] for b in buf]),
        }
        name = f"shard_{shard_idx:05d}.pt"
        torch.save(shard, os.path.join(out_dir, name))
        shard_files.append(name); counts.append(len(buf))
        shard_idx += 1
        buf.clear()

    for text in text_iter:
        ex = chunk_text_example(text, chunker, window)
        if int(ex[1].sum()) < min_chunks:
            continue
        buf.append(ex)
        total += 1
        tokens += int((ex[0] != 0).sum())
        if len(buf) >= shard_size:
            flush()
        if total % 500 == 0:
            print(f"  prepared {total} examples, ~{tokens} tokens", flush=True)
        if max_examples is not None and total >= max_examples:
            break
        if max_tokens is not None and tokens >= max_tokens:
            break
    flush()

    manifest = {
        "shards": shard_files, "counts": counts, "total": total, "tokens": tokens,
        "config": {
            "max_chunk_len": model_cfg.max_chunk_len,
            "max_chunks_per_doc": model_cfg.max_chunks_per_doc,
            "recent_token_window": model_cfg.recent_token_window,
            "vocab_size": vocab_size,
        },
    }
    with open(os.path.join(out_dir, MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[data_prep] wrote {total} examples (~{tokens} tokens) in "
          f"{len(shard_files)} shards to {out_dir}")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small", choices=list(MODEL_PRESETS))
    ap.add_argument("--offline", action="store_true",
                    help="synthetic text + stub SaT-Capped chunker (no downloads, no wtpsplit) -- CI/smoke only")
    ap.add_argument("--mixture", action="store_true",
                    help="pre-chunk the full config.DataConfig.sources mixture (the real Stages A-E "
                         "corpus), interleaved by weight and shuffled; ignores --dataset/--name. "
                         "This is the full-run cache.")
    ap.add_argument("--dataset", default="NeelNanda/pile-10k",
                    help="single HF dataset id, used when --mixture is NOT set (e.g. a smoke cache)")
    ap.add_argument("--name", default=None,
                    help="HF config name for the single-dataset path (e.g. sample-10BT for fineweb-edu); "
                         "REQUIRED for multi-config datasets. The mixture carries its own per-source names.")
    ap.add_argument("--streaming", action="store_true",
                    help="stream the single dataset instead of a full download (the mixture always streams)")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--out", default=None, help="cache dir (default DataConfig.cache_dir)")
    ap.add_argument("--docs", type=int, default=None, help="cap #documents (single-dataset path)")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--min-chunks", type=int, default=None)
    args = ap.parse_args()

    if args.mixture and args.max_tokens is None and args.docs is None:
        raise SystemExit("--mixture needs a cap: pass --max-tokens (the run's token budget) or "
                         "--docs, otherwise it streams the entire multi-TB mixture without stopping.")

    data_cfg = DataConfig()
    out_dir = os.path.join(PROJECT, args.out or data_cfg.cache_dir)
    min_chunks = args.min_chunks if args.min_chunks is not None else data_cfg.min_chunks

    # vocab_size is set from the chunker's tokenizer; the model built later must
    # match it, so we record it in the manifest.
    if args.offline:
        model_cfg = model_config(args.preset, vocab_size=8000)
        chunker = build_offline_chunker(model_cfg)
        vocab_size = model_cfg.vocab_size
        text_iter = iter(SyntheticTextCorpus(n_docs=args.docs or 2000, seed=0))
        src, chunker_name = "offline-synthetic", "stub-SaT-Capped"
    else:
        # REAL path: the true SaT-Capped chunker (Thought Gestalt preprocessing,
        # §3.1/§5.1) -- byte-for-byte the same chunker train.py builds on the fly
        # (build_pipeline), so the cache matches the intended pipeline instead of
        # a regex approximation. Lazy-imported so --offline needs no wtpsplit.
        # Pin the tokenizer to the local gpt2 dir: offline + deterministic ids.
        from train import build_sat_chunker
        data_cfg.tokenizer_name = TOKENIZER_DIR
        model_cfg = model_config(args.preset)  # vocab fixed after chunker build
        chunker, vocab_size = build_sat_chunker(model_cfg, data_cfg)
        model_cfg.vocab_size = vocab_size
        chunker_name = f"SaT({data_cfg.sat_model_name})"
        if args.mixture:
            text_iter = iter_hf_mixture(data_cfg)   # streams config.DataConfig.sources by weight
            src = "mixture[" + ", ".join(
                s.hf_id + (f":{s.name}" if s.name else "") + f"@{s.weight}"
                for s in data_cfg.sources) + "]"
        else:
            text_iter = iter_hf_single(args.dataset, args.text_field, name=args.name,
                                       streaming=args.streaming, max_docs=args.docs)
            src = args.dataset + (f":{args.name}" if args.name else "") + \
                  (" (streaming)" if args.streaming else " (full download)")

    print(f"[data_prep] source={src}")
    print(f"[data_prep] preset={args.preset} chunker={chunker_name} vocab={vocab_size} out={out_dir} "
          f"chunk_dims=({model_cfg.max_chunk_len},{model_cfg.max_chunks_per_doc},{model_cfg.recent_token_window})")
    prepare(text_iter, chunker, model_cfg, out_dir, data_cfg.shard_size, min_chunks,
            vocab_size, max_examples=args.docs, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
