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
from rocm_compat import maybe_apply_rocm_workarounds

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
    ap.add_argument("--heartbeat-every", type=int, default=10,
                    help="cheap '[step N] stage=X (heartbeat)' liveness ping every N steps (no eval/metrics); "
                         "0 = off. Lets you see progress between the pricier --log-every metric lines.")
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--archive-every", type=int, default=0,
                    help="also keep a numbered checkpoint_{step}.pt every N steps "
                         "(rollback depth for long runs; 0 = off)")
    ap.add_argument("--stage-steps", default=None, help="comma list A,B,C,D,E,F")
    ap.add_argument("--lr-schedule", default="per-stage", choices=["per-stage", "global"],
                    help="per-stage: warmup+cosine within each stage's budget (fixes D/E LR "
                         "starvation); global: one cosine across A..E (legacy)")
    ap.add_argument("--ssl-weight", type=float, default=None,
                    help="override ssl_loss_weight (default 1.0, co-equal with reconstruction; "
                         "the on-loop SSL that trains the HRM loop to predict forward)")
    ap.add_argument("--pred-var-weight", type=float, default=0.0,
                    help="anti-collapse weight on the PREDICTIONS (losses.prediction_variance_loss). "
                         "The cosine SSL objective's degenerate optimum is emitting one constant "
                         "vector; --var-weight guards only the ENCODER. 0.0 (default) = today's "
                         "behaviour; try 3.0 (mirroring --var-weight) on a NEW run. Watch "
                         "pred_collapse in the log: ~1.0 means the predictor went constant.")
    ap.add_argument("--pred-contrastive-weight", type=float, default=0.0,
                    help="InfoNCE weight on the next-latent prediction (in-batch negatives). More "
                         "targeted than --pred-var-weight: the hinge stops CONSTANT output, InfoNCE "
                         "requires INFORMATIVE output. Use alongside the cosine term, never instead "
                         "of it. 0.0 = today's behaviour; try 1.0 on a NEW run.")
    ap.add_argument("--pred-head-hidden", type=int, default=0,
                    help="make pred_head an MLP: Linear(d_latent,H)->GELU->Linear(H,d_latent). "
                         "0 (default) = the plain Linear, byte-identical to existing checkpoints. "
                         "CHANGES THE STATE_DICT -- to load an old checkpoint into it, add "
                         "--reinit-pred-head.")
    ap.add_argument("--reinit-pred-head", action="store_true",
                    help="on resume, discard pred_head and start it fresh on top of the restored "
                         "encoder/loop/Talker. The rescue for a mean-collapsed predictor.")
    ap.add_argument("--var-weight", type=float, default=None,
                    help="override ssl_var_weight (default 2.0, the VICReg-style per-dim variance "
                         "floor that resists latent collapse). At wider d_latent the latent starts "
                         "closer to the floor -- recommended ~3.0 for the -w3 presets (small-w3).")
    ap.add_argument("--norm", default="layer", choices=["layer", "rms"],
                    help="token-level normalization. 'layer' (default) = nn.LayerNorm, byte-identical to "
                         "every existing checkpoint. 'rms' = RMSNorm, which SIDESTEPS the broken gfx1151 "
                         "native_layer_norm_backward kernel entirely -- so you can drop "
                         "LATENT_MANUAL_LAYERNORM=1 and its speed/memory tax (see STRIX_HALO.md §2/§4). "
                         "WARNING: changes the architecture (RMSNorm has no bias), so a 'rms' run CANNOT "
                         "resume a 'layer' checkpoint or vice versa -- choose it at the START of a run.")
    ap.add_argument("--halt-mode", default="ponder", choices=["ponder", "supervised"],
                    help="ACT depth training (experiments.md #2). 'ponder' (default) = the "
                         "validated Graves/PonderNet soft cost, BYTE-IDENTICAL to every prior run. "
                         "'supervised' = the TRM-style per-row BCE halt gate (post-run experiment; "
                         "only affects Stage D+ where ACT is on).")
    ap.add_argument("--halt-target", default="marginal", choices=["marginal", "best_relative"],
                    help="BCE halt target when --halt-mode supervised. 'marginal' (default) halts when "
                         "the next cycle's improvement < halt_epsilon (halts early on gentle slopes); "
                         "'best_relative' halts when within halt_epsilon of the chunk's best achievable "
                         "cos_dist (keeps thinking on steadily-improving chunks).")
    ap.add_argument("--out", default="runs/scaled")
    ap.add_argument("--resume", default=None,
                    help="resume from a specific checkpoint path. If omitted and "
                         "<out>/checkpoint.pt exists, that one is auto-resumed (see --fresh).")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing <out>/checkpoint.pt and start from step 0 "
                         "(without this, re-running the same command resumes automatically).")
    ap.add_argument("--progress", default="auto", choices=["auto", "on", "off"],
                    help="tqdm progress bar: auto (bar on a terminal, plain log lines when "
                         "redirected to a file), on (force), off (never).")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    maybe_apply_rocm_workarounds()   # opt-in gfx1151 kernel workarounds (LATENT_MANUAL_LAYERNORM=1)

    device = pick_device(args.device)
    stage_steps = (tuple(int(x) for x in args.stage_steps.split(","))
                   if args.stage_steps else DEFAULT_STAGE_STEPS)
    if len(stage_steps) != 6:
        raise SystemExit(f"--stage-steps needs 6 comma-separated values (A,B,C,D,E,F), "
                         f"got {len(stage_steps)}: {stage_steps}")
    total_steps = sum(stage_steps)
    max_steps = args.max_steps if args.max_steps is not None else total_steps

    model_cfg = model_config(args.preset, pred_head_hidden=args.pred_head_hidden,
                             halt_mode=args.halt_mode,
                             halt_target=args.halt_target,
                             norm=args.norm)                # defaults == unchanged
    cache_dir = os.path.join(PROJECT, args.cache)
    ds = CachedChunkDataset(cache_dir, expect={
        "max_chunk_len": model_cfg.max_chunk_len,
        "max_chunks_per_doc": model_cfg.max_chunks_per_doc,
        "recent_token_window": model_cfg.recent_token_window,
    })
    model_cfg.vocab_size = ds.vocab_size
    print(f"[train_scaled] preset={args.preset} device={device} vocab={ds.vocab_size} "
          f"examples={len(ds)} stage_steps={stage_steps}")

    # Seeded random split rather than "first N docs": the cache preserves corpus
    # order, so a head slice can be topically clustered (one dump/source) and
    # make val unrepresentative. Deterministic across resumes (fixed seed,
    # computed before any RNG-state restore).
    val_n = min(256, max(8, len(ds) // 20))
    perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(0)).tolist()
    val_ds = Subset(ds, perm[:val_n])
    train_ds = Subset(ds, perm[val_n:])

    train_cfg = TrainConfig(
        batch_size=args.batch_size, lr=args.lr, device=device,
        grad_accum_steps=args.grad_accum, amp=args.amp, amp_dtype=args.amp_dtype,
        num_workers=args.num_workers, log_every=args.log_every,
        heartbeat_every=args.heartbeat_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_archive_every=args.archive_every,
        warmup_steps=max(100, total_steps // 50), total_steps=total_steps,
        ssl_pred_var_weight=args.pred_var_weight,
        ssl_contrastive_weight=args.pred_contrastive_weight,
        reinit_pred_head=args.reinit_pred_head,
        grounded_loss_min_frequency=1.0,   # reconstruction stays the always-on anchor
        stage_steps=stage_steps,
        per_stage_lr=(args.lr_schedule == "per-stage"),
        ssl_loss_weight=(args.ssl_weight if args.ssl_weight is not None else TrainConfig.ssl_loss_weight),
        ssl_var_weight=(args.var_weight if args.var_weight is not None else TrainConfig.ssl_var_weight),
    )
    set_seed(train_cfg.seed)

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True,
                              num_workers=train_cfg.num_workers, collate_fn=collate_chunked,
                              drop_last=True, persistent_workers=train_cfg.num_workers > 0)
    if len(train_loader) == 0:
        raise SystemExit(f"train split ({len(train_ds)} examples) yields zero batches at "
                         f"--batch-size {train_cfg.batch_size} (drop_last). Use a bigger "
                         f"cache or a smaller batch.")
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_chunked)

    model = LatentThoughtModel(model_cfg, chunker=None).to(device)  # chunker unused (data pre-chunked)
    ema = EMATargetEncoder(model.chunk_encoder, momentum=model_cfg.ema_momentum).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                                  weight_decay=train_cfg.weight_decay)
    curriculum = Curriculum(model_cfg, train_cfg)

    trainer = Trainer(model, ema, optimizer, curriculum, model_cfg, train_cfg,
                      train_loader, val_loader, ckpt_dir=os.path.join(PROJECT, args.out),
                      data_fingerprint={"examples": len(ds),
                                        "tokens": ds.manifest.get("tokens"),
                                        "shards": len(ds.manifest.get("shards", []))})
    # Resume resolution: an explicit --resume wins; otherwise auto-resume from
    # <out>/checkpoint.pt if it exists (so `stop then re-run the same command`
    # just works), unless --fresh forces a clean start.
    default_ckpt = os.path.join(PROJECT, args.out, "checkpoint.pt")
    if args.resume:
        resume = args.resume if os.path.isabs(args.resume) else os.path.join(PROJECT, args.resume)
    elif not args.fresh and os.path.exists(default_ckpt):
        resume = default_ckpt
        print(f"[train_scaled] auto-resuming from {default_ckpt} "
              f"(pass --fresh to start over, or --resume PATH for a specific checkpoint).")
    else:
        resume = None
    if resume:
        trainer.load(resume)

    trainer.train(max_steps=max_steps, progress=args.progress)
    trainer.save("model.pt")
    print(f"[train_scaled] done. final stage {curriculum.stage.name}, step {trainer.global_step}")


if __name__ == "__main__":
    main()
