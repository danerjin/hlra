"""
train_scaled.py
===============
Scale-oriented training entry point. Trains Stages A-E from a *pre-chunked*
cache (data_prep.py) using trainer.Trainer -- no tokenizer or SaT work at train
time, so it's fast and DataLoader-worker friendly.

Typical flow:
    python data_prep.py --dataset NeelNanda/pile-10k --preset small --max-tokens 100000000
    python train_scaled.py --preset small --cache chunk_cache --device cuda --amp

Nothing here downloads anything: it only reads the cache directory. Resume with
--resume runs/scaled/checkpoint.pt.
"""
from __future__ import annotations

import os
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse

import torch
from torch.utils.data import DataLoader, Subset

from config import model_config, TrainConfig, DataConfig, MODEL_PRESETS
from model import LatentThoughtModel
from ema_target import EMATargetEncoder
from curriculum import Curriculum
from data import CachedChunkDataset, collate_chunked
from trainer import Trainer
from utils import set_seed

# Default per-stage optimizer-step budgets (A,B,C,D,E,F). Tune per compute.
DEFAULT_STAGE_STEPS = (2000, 2000, 2000, 2000, 4000, 0)


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small", choices=list(MODEL_PRESETS))
    ap.add_argument("--cache", default="chunk_cache")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--stage-steps", default=None, help="comma list A,B,C,D,E,F")
    ap.add_argument("--lr-schedule", default="per-stage", choices=["per-stage", "global"],
                    help="per-stage: warmup+cosine within each stage's budget (fixes D/E LR "
                         "starvation); global: one cosine across A..E (legacy)")
    ap.add_argument("--out", default="runs/scaled")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    stage_steps = (tuple(int(x) for x in args.stage_steps.split(","))
                   if args.stage_steps else DEFAULT_STAGE_STEPS)
    total_steps = sum(stage_steps)
    max_steps = args.max_steps if args.max_steps is not None else total_steps

    model_cfg = model_config(args.preset)
    cache_dir = os.path.join(PROJECT, args.cache)
    ds = CachedChunkDataset(cache_dir, expect={
        "max_chunk_len": model_cfg.max_chunk_len,
        "max_chunks_per_doc": model_cfg.max_chunks_per_doc,
        "recent_token_window": model_cfg.recent_token_window,
    })
    model_cfg.vocab_size = ds.vocab_size
    print(f"[train_scaled] preset={args.preset} device={device} vocab={ds.vocab_size} "
          f"examples={len(ds)} stage_steps={stage_steps}")

    val_n = min(256, max(8, len(ds) // 20))
    val_ds = Subset(ds, range(val_n))
    train_ds = Subset(ds, range(val_n, len(ds)))

    train_cfg = TrainConfig(
        batch_size=args.batch_size, lr=args.lr, device=device,
        grad_accum_steps=args.grad_accum, amp=args.amp, amp_dtype=args.amp_dtype,
        num_workers=args.num_workers, log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        warmup_steps=max(100, total_steps // 50), total_steps=total_steps,
        grounded_loss_min_frequency=1.0,   # reconstruction stays the always-on anchor
        stage_steps=stage_steps,
        per_stage_lr=(args.lr_schedule == "per-stage"),
    )
    set_seed(train_cfg.seed)

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True,
                              num_workers=train_cfg.num_workers, collate_fn=collate_chunked,
                              drop_last=True, persistent_workers=train_cfg.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_chunked)

    model = LatentThoughtModel(model_cfg, chunker=None).to(device)  # chunker unused (data pre-chunked)
    ema = EMATargetEncoder(model.chunk_encoder, momentum=model_cfg.ema_momentum,
                           online_proj=model.ssl_proj).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                                  weight_decay=train_cfg.weight_decay)
    curriculum = Curriculum(model_cfg, train_cfg)

    trainer = Trainer(model, ema, optimizer, curriculum, model_cfg, train_cfg,
                      train_loader, val_loader, ckpt_dir=os.path.join(PROJECT, args.out))
    if args.resume:
        trainer.load(args.resume)

    trainer.train(max_steps=max_steps)
    trainer.save("model.pt")
    print(f"[train_scaled] done. final stage {curriculum.stage.name}, step {trainer.global_step}")


if __name__ == "__main__":
    main()
