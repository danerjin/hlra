"""
train_real.py
=============
Small training run on ~1.5M tokens of real text (NeelNanda/pile-10k) using the
REAL gpt2 subword tokenizer + regex sentence chunking, so the resulting
checkpoint produces *decodable* text at inference (no SaT download needed --
only `transformers`). Records per-log-step metrics and saves a checkpoint.

Outputs (in ../runs/):
  model.pt      -- checkpoint (state_dict + config + tokenizer/chunker info)
  metrics.json  -- per-step {step, stage, val_loss, nll, ssl, ponder}

This is a *smoke-scale* model: tiny width, ~1.5M tokens, ~250 steps. It is NOT
gpt2-quality and won't produce coherent text -- it exists to exercise the full
architecture end-to-end and to feed generate.py. Scale up (width, tokens,
steps, real SaT chunker) for anything real.

Run:  python train_real.py
"""
from __future__ import annotations

import os
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(PROJECT, ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")  # load tokenizer from local dir, no HEAD stalls
TOKENIZER_DIR = os.path.join(PROJECT, "gpt2_tok")   # gpt2 tokenizer files fetched locally

import json
from dataclasses import asdict

import torch

from config import ModelConfig, TrainConfig, DataConfig
from model import LatentThoughtModel
from ema_target import EMATargetEncoder
from curriculum import Curriculum
from data import build_regex_gpt2_chunker, iter_hf_single
from utils import set_seed
import train as T

RUN_DIR = os.path.join(PROJECT, "runs")
DATASET, TEXT_FIELD = "NeelNanda/pile-10k", "text"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    device = pick_device()
    model_cfg = ModelConfig(
        d_model=192, n_heads=6, d_ff=768,
        max_chunk_len=48, max_chunks_per_doc=12, recent_token_window=96, memory_capacity=64,
    )
    train_cfg = TrainConfig(
        batch_size=4, lr=3e-4, max_steps_per_stage=45, plateau_patience=3,
        plateau_min_delta=1e9, log_every=10, device=device,
        # Reconstruction (grounded) is the anti-collapse anchor -- keep it on
        # every step from Stage D, rather than thinning it below SSL.
        grounded_loss_min_frequency=1.0,
    )
    data_cfg = DataConfig(min_chunks=3)
    set_seed(train_cfg.seed)

    chunker, vocab_size = build_regex_gpt2_chunker(model_cfg, TOKENIZER_DIR)
    model_cfg.vocab_size = vocab_size
    print(f"[train_real] device={device} tokenizer=gpt2(local) vocab={vocab_size}")

    model = LatentThoughtModel(model_cfg, chunker).to(device)
    ema = EMATargetEncoder(model.chunk_encoder, momentum=model_cfg.ema_momentum,
                           online_proj=model.ssl_proj).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                                  weight_decay=train_cfg.weight_decay)
    curriculum = Curriculum(model_cfg, train_cfg)

    train_factory = lambda: iter_hf_single(DATASET, TEXT_FIELD, streaming=False, max_docs=800)
    val_factory = lambda: iter_hf_single(DATASET, TEXT_FIELD, streaming=False, max_docs=32)
    train_loader = T.make_loader(train_factory, chunker, model_cfg, train_cfg, data_cfg,
                                 max_tokens=1_500_000)
    val_loader = T.make_loader(val_factory, chunker, model_cfg, train_cfg, data_cfg,
                               max_examples=32)

    metrics = []
    T.train_stages_a_to_e(model, ema, curriculum, model_cfg, train_cfg, optimizer,
                          train_loader, val_loader, max_global_steps=250, metrics=metrics)

    os.makedirs(RUN_DIR, exist_ok=True)
    model.to("cpu")
    torch.save({
        "model_state": model.state_dict(),
        "model_cfg": asdict(model_cfg),
        "tokenizer_name": "gpt2",
        "tokenizer_path": TOKENIZER_DIR,
        "chunker": "regex_gpt2",
        "vocab_size": vocab_size,
        "stage_reached": curriculum.stage.name,
        "note": "toy smoke checkpoint -- NOT gpt2 quality; exercises the architecture only",
    }, os.path.join(RUN_DIR, "model.pt"))
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[train_real] saved checkpoint + metrics to {RUN_DIR} "
          f"(final stage {curriculum.stage.name}, {len(metrics)} log points)")


if __name__ == "__main__":
    main()
