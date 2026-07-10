"""
run_small.py
============
A tiny end-to-end test on ~1M tokens of REAL text, meant to confirm the whole
Stages A->E curriculum runs and learns *something* on real prose -- not to
produce a good model.

Dataset: NeelNanda/pile-10k (10k diverse Pile documents, one whole doc per row,
~33MB single download, ungated). We cap to ~1M real tokens.

Dependencies: only `datasets` (pip install datasets). This run uses the OFFLINE
stub chunker (regex sentence split + hashing whitespace tokenizer) on the real
text, so there is NO SaT / HF-tokenizer download. For real subword tokenization
+ real SaT boundaries, install `wtpsplit transformers` and swap
`build_offline_chunker` for `train.build_sat_chunker` (see comments below).

Run:  python run_small.py
"""
from __future__ import annotations

import os
# Keep the dataset cache inside the project (alongside the venv) rather than
# ~/.cache, so the whole test is self-contained. Set before `datasets` imports.
os.environ.setdefault(
    "HF_HOME", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".hf_cache"))

import torch

from config import ModelConfig, TrainConfig, DataConfig
from model import LatentThoughtModel
from ema_target import EMATargetEncoder
from curriculum import Curriculum
from data import build_offline_chunker, iter_hf_single
from utils import set_seed
import train as T

# ~1M-token smoke settings -------------------------------------------------
DATASET = "NeelNanda/pile-10k"
TEXT_FIELD = "text"
MAX_TOKENS = 1_000_000
MAX_DOCS = 600            # generous upper bound; MAX_TOKENS is the real stop
MAX_GLOBAL_STEPS = 120   # hard cap so the smoke always terminates (~1s/step)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    device = pick_device()

    # Small-but-real config: real chunk sizes, modest width so a CPU/MPS run
    # finishes quickly. vocab_size is the stub tokenizer's hashing range.
    model_cfg = ModelConfig(
        vocab_size=32000, d_model=192, n_heads=6, d_ff=768,
        max_chunk_len=64, max_chunks_per_doc=16, recent_token_window=128,
        memory_capacity=64,
    )
    # Fast curriculum: force stage transitions on any plateau so A->E is walked
    # through within the token/step budget (this is a smoke, not a real schedule).
    train_cfg = TrainConfig(
        batch_size=4, lr=3e-4, max_steps_per_stage=40, plateau_patience=3,
        plateau_min_delta=1e9, log_every=10, device=device,
    )
    data_cfg = DataConfig(min_chunks=3)
    set_seed(train_cfg.seed)

    print(f"[run_small] device={device} dataset={DATASET} budget~{MAX_TOKENS} tokens")

    # Offline stub chunker over REAL text (only `datasets` needed). For real SaT:
    #   chunker, vocab = T.build_sat_chunker(model_cfg, data_cfg)
    #   model_cfg.vocab_size = vocab
    chunker = build_offline_chunker(model_cfg)

    # Non-streaming: download the 33MB dataset once (cached), then iterate
    # in-memory. Streaming row-by-row is ~7s/doc for this dataset -- far slower
    # than the ~1s/step compute, so it would dominate wall-clock.
    train_factory = lambda: iter_hf_single(DATASET, TEXT_FIELD, streaming=False, max_docs=MAX_DOCS)
    val_factory = lambda: iter_hf_single(DATASET, TEXT_FIELD, streaming=False, max_docs=32)

    model = LatentThoughtModel(model_cfg, chunker).to(device)
    ema = EMATargetEncoder(model.chunk_encoder, momentum=model_cfg.ema_momentum).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                                  weight_decay=train_cfg.weight_decay)
    curriculum = Curriculum(model_cfg, train_cfg)

    train_loader = T.make_loader(train_factory, chunker, model_cfg, train_cfg, data_cfg,
                                 max_tokens=MAX_TOKENS)
    val_loader = T.make_loader(val_factory, chunker, model_cfg, train_cfg, data_cfg,
                               max_examples=32)

    T.train_stages_a_to_e(model, ema, curriculum, model_cfg, train_cfg, optimizer,
                          train_loader, val_loader, max_global_steps=MAX_GLOBAL_STEPS)
    print(f"[run_small] done. final stage reached: {curriculum.stage.name}")


if __name__ == "__main__":
    main()
