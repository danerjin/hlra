"""
bench_throughput.py
===================
Measure training throughput (tokens/sec) of the *actual* grounded path on the
target GPU, across batch sizes, and extrapolate the wall-clock time for a token
budget. This is the go/no-go number for whether a run is days or months on a
given box -- run it AFTER rocm_smoke.py passes, BEFORE any data prep.

Why this matters here specifically: forward_grounded walks chunks in a
sequential Python loop (up to max_chunks_per_doc per doc), each running the
8-step HRM recurrence + the Talker. That's many small ops, so throughput is
launch-overhead-bound and *rises with batch size* until the GPU saturates --
which is exactly what this sweep reveals, and why the 128 GB of unified memory
(large batch) is the main lever. Everything is synthetic (no data needed).

Run (on the ROCm box, from files/):
    python bench_throughput.py --preset small --batch-size 4,16,32,64 --amp
    python bench_throughput.py --preset small --batch-size 64 --stage E   # ACT path

Reads:
  * `step time`      : seconds per optimizer step (fwd+bwd+step), GPU-synced.
  * `dense tok/s`    : batch*chunks*chunk_len / step_time -- raw hardware
                       throughput over ALL positions (upper bound).
  * `real tok/s`     : dense * --fill-frac -- an estimate of *non-pad* tokens/s
                       for run planning (real docs are ~50-70% non-pad; tune
                       --fill-frac to your prepared cache's actual ratio).
  * `peak GB`        : peak device memory for that batch (pick the largest batch
                       that fits your 128 GB with headroom).
  * `budget ETA`     : --token-budget / real_tok_s, in days.
"""
from __future__ import annotations

import argparse
import time

import torch

from config import model_config
from model import LatentThoughtModel, StageFlags, SELF
from ema_target import EMATargetEncoder
from gestalt_memory import GestaltMemoryBank


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def synth_batch(cfg, batch, device):
    N, L, W = cfg.max_chunks_per_doc, cfg.max_chunk_len, cfg.recent_token_window
    ct = torch.randint(1, cfg.vocab_size, (batch, N, L), device=device)
    cm = torch.ones(batch, N, dtype=torch.bool, device=device)   # all chunks valid = worst case
    ri = torch.randint(1, cfg.vocab_size, (batch, W), device=device)
    rm = torch.ones(batch, W, dtype=torch.bool, device=device)
    return ct, cm, ri, rm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small")
    ap.add_argument("--batch-size", default="4,16,32,64", help="comma list to sweep")
    ap.add_argument("--vocab", type=int, default=50258)
    ap.add_argument("--stage", default="C", choices=["C", "E"], help="C=fixed depth, E=ACT")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (recommended on GPU)")
    ap.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--warmup", type=int, default=3, help="untimed steps (kernel autotune/compile)")
    ap.add_argument("--steps", type=int, default=10, help="timed steps averaged")
    ap.add_argument("--fill-frac", type=float, default=0.6, help="est. non-pad token fraction of real docs")
    ap.add_argument("--token-budget", type=float, default=1.2e9, help="tokens for the ETA extrapolation")
    args = ap.parse_args()

    device = pick_device()
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    batches = [int(x) for x in args.batch_size.split(",")]

    cfg = model_config(args.preset, vocab_size=args.vocab)
    model = LatentThoughtModel(cfg, chunker=None).to(device)
    ema = EMATargetEncoder(model.chunk_encoder, momentum=cfg.ema_momentum).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    n_params = sum(p.numel() for p in model.parameters())

    if args.stage == "C":
        flags = StageFlags(use_hrm_loop=True, detach_memory=False, inner_loop_grad_window=5,
                           memory_grad_window=5, use_act=False, use_input_lanes=False)
    else:
        flags = StageFlags(use_hrm_loop=True, detach_memory=False, inner_loop_grad_window=5,
                           memory_grad_window=5, use_act=True, use_input_lanes=False)

    amp_on = args.amp and device == "cuda"     # autocast is CUDA-only here
    autocast = (torch.autocast(device_type="cuda", dtype=dtype) if amp_on
                else torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False))
    N, L = cfg.max_chunks_per_doc, cfg.max_chunk_len

    if args.amp and not amp_on:
        print(f"NOTE: --amp requested but device={device}; timing full precision.")
    print(f"device={device}  preset={args.preset}({n_params/1e6:.0f}M)  stage={args.stage}  "
          f"amp={args.amp_dtype if amp_on else 'off'}  chunks/doc={N} chunk_len={L}")
    print(f"{'batch':>6} {'step s':>9} {'dense tok/s':>13} {'real tok/s':>12} {'peak GB':>9} "
          f"{'ETA @'+f'{args.token_budget/1e9:g}'+'B (days)':>18}")

    def run_step(batch_data):
        ct, cm, ri, rm = batch_data
        opt.zero_grad(set_to_none=True)
        memory = GestaltMemoryBank(cfg.memory_capacity, cfg.d_latent)
        with autocast:
            # Full step, mirroring trainer._loss_on exactly: ONE shared online
            # encoder pass reused by both branches (omitting chunk_vecs= here
            # double-ran the encoder fwd+bwd and inflated step time & peak GB),
            # then the cheap autoencoder anchor + the SEQUENTIAL on-loop SSL
            # (the loop reading/writing memory) -- the latter dominates cost.
            chunk_vecs = model.encode_chunks(ct)
            nll = model.forward_grounded(ct, cm, chunk_vecs=chunk_vecs)
            ssl, ponder = model.forward_self_supervised(
                ct, cm, ri, rm, memory, SELF, flags, ema,
                cos_weight=1.0, var_weight=2.0, ponder_weight=cfg.act_ponder_cost,
                chunk_vecs=chunk_vecs)
            loss = nll + ssl + ponder
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model.chunk_encoder)   # part of every real optimizer step

    for B in batches:
        try:
            data = synth_batch(cfg, B, device)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            for _ in range(args.warmup):
                run_step(data)
            sync(device)
            t0 = time.perf_counter()
            for _ in range(args.steps):
                run_step(data)
            sync(device)
            step_s = (time.perf_counter() - t0) / args.steps

            dense_tok = B * N * L
            dense_tps = dense_tok / step_s
            real_tps = dense_tps * args.fill_frac
            peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else float("nan")
            eta_days = args.token_budget / real_tps / 86400
            print(f"{B:>6} {step_s:>9.3f} {dense_tps:>13,.0f} {real_tps:>12,.0f} "
                  f"{peak_gb:>9.2f} {eta_days:>18.1f}")
        except RuntimeError as e:
            print(f"{B:>6}  OOM/err: {str(e)[:70]}")
            break

    print("\nRead: pick the largest batch whose 'peak GB' leaves headroom under your 128 GB, and\n"
          "use its 'real tok/s' for planning. If real tok/s is still low at large batch, the\n"
          "sequential chunk loop -- not the GPU -- is the bottleneck (see STRIX_HALO.md).")


if __name__ == "__main__":
    main()
