"""
prep_proverbs.py
================
Pre-chunk the Book of Proverbs (proverbs.jsonl, one prose chapter per line) into
the shard cache the trainer reads. Reuses the repo's real-text chunker
(gpt2 tokenizer + regex sentence boundaries + SaT-Capped length capping) and the
same prepare() writer that data_prep.py uses -- no training semantics are
touched, this only swaps the text source (local chapters instead of an HF stream).

Each chapter is treated as an independent document (independent gestalt memory).

Usage:
    python prep_proverbs.py --preset smoke --out chunk_cache
"""
import os, sys, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.join(os.path.dirname(HERE), "files")
sys.path.insert(0, FILES)  # import the repo modules

from config import model_config, MODEL_PRESETS, DataConfig          # noqa: E402
from data import build_regex_gpt2_chunker                            # noqa: E402
from data_prep import prepare, TOKENIZER_DIR                         # noqa: E402


def chapter_iter(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="smoke", choices=list(MODEL_PRESETS))
    ap.add_argument("--jsonl", default=os.path.join(HERE, "proverbs.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "chunk_cache"))
    # Proverbs chapters are short prose; keep the default min_chunks (4) so a
    # chapter must yield >= 4 thoughts to be kept (exercises the memory).
    ap.add_argument("--min-chunks", type=int, default=None)
    args = ap.parse_args()

    data_cfg = DataConfig()
    min_chunks = args.min_chunks if args.min_chunks is not None else data_cfg.min_chunks

    model_cfg = model_config(args.preset)
    chunker, vocab_size = build_regex_gpt2_chunker(model_cfg, TOKENIZER_DIR)
    model_cfg.vocab_size = vocab_size

    out_dir = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    print(f"[prep_proverbs] preset={args.preset} vocab={vocab_size} out={out_dir}")
    print(f"[prep_proverbs] chunk_dims=(L={model_cfg.max_chunk_len}, "
          f"C={model_cfg.max_chunks_per_doc}, win={model_cfg.recent_token_window}) "
          f"min_chunks={min_chunks}")

    prepare(chapter_iter(args.jsonl), chunker, model_cfg, out_dir,
            data_cfg.shard_size, min_chunks, vocab_size,
            max_examples=None, max_tokens=None)


if __name__ == "__main__":
    main()
