"""
make_manifest.py
================
Reconstruct `manifest.json` for a chunk cache whose prep DIED or STALLED before
finishing. `data_prep.py` writes the manifest only at the very end, so a killed
prep leaves completed `shard_*.pt` files but no manifest -> the cache is unusable.
This scans the completed shards and writes a valid manifest so you can TRAIN on
the PARTIAL cache -- e.g. salvage a 0.6 B-token cache when a 1.2 B prep stalled.

    python make_manifest.py chunk_cache small-w3

The token budget has slack (STRIX_HALO/TRAINING call 1.0-1.5 B a soft estimate),
so a 0.5-1 B-token partial cache is a perfectly good run. A trailing shard that
was mid-write when the prep was killed is skipped automatically.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch

from chunker import CHUNKER_VERSION
from config import model_config
from data import MANIFEST


def main():
    cache = sys.argv[1] if len(sys.argv) > 1 else "chunk_cache"
    preset = sys.argv[2] if len(sys.argv) > 2 else "small-w3"
    cfg = model_config(preset)

    shards = sorted(glob.glob(os.path.join(cache, "shard_*.pt")))
    if not shards:
        raise SystemExit(f"no shard_*.pt files in {cache} -- nothing to salvage.")

    kept, counts, tokens = [], [], 0
    for s in shards:
        try:
            d = torch.load(s, map_location="cpu")
            n = int(d["chunk_tensor"].shape[0])
            tok = int((d["chunk_tensor"] != 0).sum())
        except Exception as e:  # a shard killed mid-write -> drop it
            print(f"  SKIP (partial/corrupt) {os.path.basename(s)}: {e}")
            continue
        kept.append(os.path.basename(s))
        counts.append(n)
        tokens += tok
        print(f"  {os.path.basename(s)}: {n} examples")

    total = sum(counts)
    manifest = {
        "shards": kept, "counts": counts, "total": total, "tokens": tokens,
        "chunker_version": CHUNKER_VERSION,   # matches the current chunker, so the loader accepts it
        "chunker_name": "salvaged (make_manifest.py)",
        "config": {
            "max_chunk_len": cfg.max_chunk_len,
            "max_chunks_per_doc": cfg.max_chunks_per_doc,
            "recent_token_window": cfg.recent_token_window,
            "vocab_size": 50258,              # gpt2 + 1 (the real SaT/regex prep tokenizer)
        },
    }
    with open(os.path.join(cache, MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[make_manifest] wrote {cache}/{MANIFEST}: {total} examples, ~{tokens} tokens, "
          f"{len(kept)} shards -- ready to train.")


if __name__ == "__main__":
    main()
