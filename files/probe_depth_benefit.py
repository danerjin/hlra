"""
probe_depth_benefit.py
======================
The question that gates every halt mechanism (ACT ponder cost AND the TRM
supervised gate, experiments.md #2): **does running the HRM loop DEEPER actually
improve the next-latent prediction on REAL data?**

Both halt mechanisms already collapse to the min-depth floor, and that is *correct*
if extra cycles don't help -- the experiments.md #2 smoke test measured a FLAT
per-cycle cos_dist (0.6023 -> 0.6010, ~0.0003 over the whole depth range), but on
smoke data with "no depth signal". This probe runs the SAME per-cycle measurement on
a real checkpoint + real cache, so we can tell which world we're in:

  * cos(pred, true next) RISES with depth  -> depth helps. The loop reasons; the halt
    gate is worth fixing (best_relative target may already escape the floor). Then a
    halt prototype is the right next step.
  * cos is FLAT across depth (like smoke)  -> depth does NOT help. The RECURRENCE is
    the bug, not the halt -- the H-update is (near-)idempotent / converges to a fixed
    point after the first cycle, so no halt mechanism can matter. The fix is upstream
    (make the loop refine), not another halt gate.

It reuses `HRMInnerLoop.forward_halt_trace` (returns the H-state after every cycle) and
enters the loop FRESH per chunk with accumulated gestalt memory -- the same
methodology as probe_predictor.collect (no cross-chunk h carry), so the two probes are
comparable. The memory write uses the deployed min-depth (n_cycles) H-state.

Run (CPU fine):
    python files/probe_depth_benefit.py --ckpt runs/scaled/anticollapse/model.pt \
        --cache chunk_cache --batches 8
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
PROJECT = os.path.dirname(_HERE)

from config import ModelConfig                      # noqa: E402
from data import CachedChunkDataset                 # noqa: E402
from ema_target import EMATargetEncoder             # noqa: E402
from gestalt_memory import GestaltMemoryBank        # noqa: E402
from model import LatentThoughtModel, SELF          # noqa: E402


def _resolve(p: str) -> str:
    p = os.path.expanduser(p)
    return p if os.path.isabs(p) or os.path.exists(p) else os.path.join(PROJECT, p)


@torch.no_grad()
def collect_by_depth(model, ema, batch, device, min_depth, cap):
    """For each chunk transition, run the loop to the cap and record cos(pred, true
    next) at EVERY cycle depth. Returns a list (indexed by depth 1..cap) of 1-D
    tensors of per-pair cosines pooled over the batch."""
    ct, cm, _ri, _rm = (t.to(device) for t in batch)
    batch_n, n_chunks = ct.shape[0], ct.shape[1]

    flat = ct.reshape(batch_n * n_chunks, -1)
    chunk_vecs = model._encode_real_rows(flat, model.chunk_encoder).reshape(batch_n, n_chunks, -1)
    tgt = model._encode_real_rows(flat, ema.encode).reshape(batch_n, n_chunks, -1)

    memory = GestaltMemoryBank(model.cfg.memory_capacity, model.cfg.d_latent)
    per_depth = [[] for _ in range(cap)]
    hstep = [[] for _ in range(cap)]   # cos(h_k, h_{k-1}): does the STATE converge?
    for t in range(n_chunks):
        valid = cm[:, t]
        if not bool(valid.any()):
            continue
        # (cap, batch, d): the H-state after every cycle. Enter FRESH (h/l=None),
        # matching probe_predictor's per-chunk methodology.
        trace = model.hrm_loop.forward_halt_trace(chunk_vecs[:, t], memory, None, grad_window=5)
        if t + 1 < n_chunks:
            pair = valid & cm[:, t + 1]
            if bool(pair.any()):
                target = F.normalize(tgt[:, t + 1][pair], dim=-1)
                for k in range(cap):
                    pred = F.normalize(model.pred_head(trace[k])[pair], dim=-1)
                    per_depth[k].append((pred * target).sum(-1))   # cosine per pair
                    if k > 0:  # how much did the H-state move from the previous cycle?
                        a = F.normalize(trace[k][pair], dim=-1)
                        b = F.normalize(trace[k - 1][pair], dim=-1)
                        hstep[k].append((a * b).sum(-1))
        # Continue the sequence on the DEPLOYED min-depth H-state (index min_depth-1).
        memory.write(trace[min_depth - 1], SELF)
    return per_depth, hstep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to model.pt / checkpoint.pt")
    ap.add_argument("--cache", default="chunk_cache", help="chunk cache dir (real training data)")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    ckpt = torch.load(_resolve(args.ckpt), map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_cfg"]) if isinstance(ckpt.get("model_cfg"), dict) else ckpt["model_cfg"]
    model = LatentThoughtModel(cfg, chunker=None).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    ema = EMATargetEncoder(model.chunk_encoder, momentum=cfg.ema_momentum).to(device)
    if isinstance(ckpt.get("ema"), dict):
        try:
            ema.target_encoder.load_state_dict(ckpt["ema"])
        except Exception as e:
            print(f"[depth] NOTE: could not restore EMA weights ({e}); using a fresh EMA copy.")

    loop = model.hrm_loop
    min_depth = loop.h_updates_per_thought          # the fixed-depth floor (Stages A-C)
    cap = loop.act_max_ponder_steps                 # ACT / halt-trace ceiling
    print(f"[depth] {args.ckpt}: d_latent={cfg.d_latent} min_depth(n_cycles)={min_depth} cap={cap}")

    ds = CachedChunkDataset(_resolve(args.cache))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    pooled = [[] for _ in range(cap)]
    hpooled = [[] for _ in range(cap)]
    for i, batch in enumerate(loader):
        if i >= args.batches:
            break
        pd, hs = collect_by_depth(model, ema, batch, device, min_depth, cap)
        for k in range(cap):
            pooled[k].extend(pd[k])
            hpooled[k].extend(hs[k])
    if not pooled[0]:
        raise SystemExit("no valid (chunk_t, chunk_t+1) pairs found -- try more --batches")

    print(f"\n[depth] cos(pred_head(h_at_cycle_k), EMA(true next)) vs loop depth, and "
          f"cos(h_k, h_k-1) = how much the H-STATE moved ({sum(x.numel() for x in pooled[0])} pairs):\n")
    print(f"  {'depth':>5}  {'pred_cos':>9}  {'Δ vs prev':>10}  {'h moved (cos h_k,h_k-1)':>24}   note")
    base = None
    prev = None
    for k in range(cap):
        cos_k = float(torch.cat(pooled[k]).mean())
        if base is None:
            base = cos_k
        d_prev = "" if prev is None else f"{cos_k - prev:+.4f}"
        hmoved = "" if k == 0 else f"{float(torch.cat(hpooled[k]).mean()):.4f}"
        note = ""
        if k + 1 == min_depth:
            note = "<- min-depth floor (deployed)"
        print(f"  {k+1:>5}  {cos_k:>9.4f}  {d_prev:>10}  {hmoved:>24}   {note}")
        prev = cos_k

    total_gain = float(torch.cat(pooled[cap - 1]).mean()) - float(torch.cat(pooled[min_depth - 1]).mean())
    print(f"\n[depth] gain from min-depth ({min_depth}) to cap ({cap}) = {total_gain:+.4f}")
    if total_gain < 0.005:
        print("  VERDICT: FLAT -- extra cycles do NOT improve prediction on this data. The halt")
        print("           mechanisms correctly collapse to the floor; the bug is the RECURRENCE")
        print("           (near-idempotent H-update), not the halt gate. Fixing the halt won't help.")
    else:
        print("  VERDICT: DEPTH HELPS -- prediction improves with cycles. The loop reasons, and a")
        print("           halt mechanism is worth it (try --halt-mode supervised --halt-target")
        print("           best_relative). Fixing/keeping the halt gate is the right next step.")


if __name__ == "__main__":
    main()
