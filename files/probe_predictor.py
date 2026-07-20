"""
probe_predictor.py -- is the next-latent predictor INFORMATIVE, or predicting the mean?
======================================================================================
The SSL objective is `k * (1 - cos(pred, target))`. A **degenerate optimum** is to
emit (near) the dataset-mean latent: the mean has decent cosine against everything,
so it scores a respectable loss while carrying ZERO information about which chunk
comes next. The variance floor in the loss guards the *encoder* from collapse
(`variance_regularization`, target_std=0.1) -- **nothing guards the predictor from
mean-prediction**. So a plateaued `ssl` is ambiguous on its own: it could be a
predictor that learned all it can, or one that gave up and outputs a constant.

This measures it directly. It replays `forward_self_supervised`'s exact path
(online encoder -> HRM loop over chunks, writing gestalt memory as it goes ->
pred_head(h_t) vs the EMA target of chunk t+1) and reports cosine similarity of the
predictions against four references:

  MATCHED   pred_i  vs  target_i            <- what SSL actually optimizes
  SHUFFLED  pred_i  vs  target_perm(i)      <- chance: a real latent, wrong position
  MEAN-BASE mean(targets) vs target_i       <- what the DEGENERATE strategy scores
  PRED-SELF pred_i  vs  mean(preds)         <- are the predictions all the same vector?

How to read it
--------------
  MATCHED >> SHUFFLED                  -> informative. The predictor knows WHICH
                                          chunk comes next, not just the average shape.
  MATCHED ~= SHUFFLED                  -> content-free. It is not resolving position.
  MATCHED ~= MEAN-BASE                 -> it is doing no better than emitting the mean.
  PRED-SELF ~= 1.0                     -> it IS emitting a constant (mean collapse).

The headline number is the GAP (matched - shuffled). A large gap with a modest
absolute matched score is FINE and expected: next-chunk prediction is genuinely
uncertain (many sentences could follow), so cos_sim 1.0 is impossible in principle
and a plateau near the task's entropy ceiling is not a failure.

Run (CPU is fine; a few hundred chunks is plenty):
    python files/probe_predictor.py --ckpt runs/scaled/model.pt --cache chunk_cache --batches 8
"""
from __future__ import annotations

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
from model import LatentThoughtModel, StageFlags, SELF  # noqa: E402


def _resolve(p: str) -> str:
    p = os.path.expanduser(p)
    return p if os.path.isabs(p) or os.path.exists(p) else os.path.join(PROJECT, p)


@torch.no_grad()
def collect(model, ema, batch, flags, device):
    """Replay forward_self_supervised's pred/target pairing (no grad, no loss)."""
    ct, cm, _ri, _rm = (t.to(device) for t in batch)
    batch_n, n_chunks = ct.shape[0], ct.shape[1]

    flat = ct.reshape(batch_n * n_chunks, -1)
    chunk_vecs = model._encode_real_rows(flat, model.chunk_encoder).reshape(batch_n, n_chunks, -1)
    tgt = model._encode_real_rows(flat, ema.encode).reshape(batch_n, n_chunks, -1)

    memory = GestaltMemoryBank(model.cfg.memory_capacity, model.cfg.d_latent)
    preds, targets = [], []
    for t in range(n_chunks):
        valid = cm[:, t]
        if not bool(valid.any()):
            continue
        h_state, _ponder = model.hrm_loop(
            chunk_vecs[:, t], memory, None,
            grad_window=flags.inner_loop_grad_window, use_act=flags.use_act,
        )
        memory.write(h_state, SELF)
        if t + 1 < n_chunks:
            pair = valid & cm[:, t + 1]
            if bool(pair.any()):
                preds.append(model.pred_head(h_state)[pair])
                targets.append(tgt[:, t + 1][pair])
    if not preds:
        return None, None
    return torch.cat(preds, 0), torch.cat(targets, 0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to model.pt / checkpoint.pt")
    ap.add_argument("--cache", default="chunk_cache", help="chunk cache dir (real training data)")
    ap.add_argument("--batches", type=int, default=8, help="how many batches to pool")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--act", action="store_true", help="run the loop with ACT on (stage D/E behaviour)")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    ckpt = torch.load(_resolve(args.ckpt), map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_cfg"]) if isinstance(ckpt.get("model_cfg"), dict) else ckpt["model_cfg"]
    model = LatentThoughtModel(cfg, chunker=None).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    # EMATargetEncoder is deliberately NOT an nn.Module: it keeps its inner encoder in
    # eval() from __init__ (deterministic SSL targets, dropout off) and exposes no
    # .eval()/.load_state_dict(). Restore its weights by loading into target_encoder.
    ema = EMATargetEncoder(model.chunk_encoder, momentum=cfg.ema_momentum).to(device)
    if isinstance(ckpt.get("ema"), dict):
        try:
            ema.target_encoder.load_state_dict(ckpt["ema"])
        except Exception as e:                      # shape/key drift -> fall back to the online copy
            print(f"[probe] NOTE: could not restore EMA weights ({e}); using a fresh EMA copy "
                  f"of the online encoder. Targets differ slightly from training.")
    print(f"[probe] {args.ckpt}: d_model={cfg.d_model} d_latent={cfg.d_latent} "
          f"k={cfg.cosine_loss_k} act={args.act}")

    ds = CachedChunkDataset(_resolve(args.cache))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    flags = StageFlags(use_hrm_loop=True, detach_memory=False, inner_loop_grad_window=5,
                       memory_grad_window=5, use_act=args.act, use_input_lanes=False)

    P, T = [], []
    for i, batch in enumerate(loader):
        if i >= args.batches:
            break
        p, t = collect(model, ema, batch, flags, device)
        if p is not None:
            P.append(p); T.append(t)
    if not P:
        raise SystemExit("no valid (chunk_t, chunk_t+1) pairs found -- try more --batches")
    pred, target = torch.cat(P, 0), torch.cat(T, 0)
    n = pred.shape[0]

    perm = torch.randperm(n)
    mean_t = target.mean(0, keepdim=True)
    mean_p = pred.mean(0, keepdim=True)

    matched  = F.cosine_similarity(pred, target, dim=-1)
    shuffled = F.cosine_similarity(pred, target[perm], dim=-1)
    meanbase = F.cosine_similarity(mean_t.expand_as(target), target, dim=-1)
    predself = F.cosine_similarity(pred, mean_p.expand_as(pred), dim=-1)

    k = cfg.cosine_loss_k
    f = lambda x: f"{x.mean().item():+.4f} (sd {x.std().item():.4f})"
    print(f"\n[probe] {n} (pred, target) pairs\n")
    print(f"  MATCHED   pred vs true next    : {f(matched)}   -> ssl = k*(1-cos) = {k*(1-matched.mean().item()):.3f}")
    print(f"  SHUFFLED  pred vs wrong next   : {f(shuffled)}   <- chance")
    print(f"  MEAN-BASE mean(target) vs true : {f(meanbase)}   <- the degenerate strategy")
    print(f"  PRED-SELF pred vs mean(pred)   : {f(predself)}   <- 1.0 means constant output")

    gap = (matched - shuffled).mean().item()
    lift = matched.mean().item() - meanbase.mean().item()
    print(f"\n  GAP  matched - shuffled = {gap:+.4f}")
    print(f"  LIFT matched - meanbase = {lift:+.4f}")
    print()
    if predself.mean().item() > 0.98:
        print("  VERDICT: predictions are ~a CONSTANT vector -> mean-collapse. The SSL number is")
        print("           meaningless as a capability measure; fix the predictor before tuning it.")
    elif gap < 0.05:
        print("  VERDICT: matched ~= shuffled -> the predictor is NOT resolving which chunk comes")
        print("           next. Pushing ssl lower will not help; it is optimizing a content-free term.")
    elif lift < 0.05:
        print("  VERDICT: no better than emitting the dataset mean. Same conclusion as above.")
    else:
        print("  VERDICT: INFORMATIVE -- it beats both chance and the mean baseline. A plateaued ssl")
        print("           here is likely near the task's entropy ceiling, so 'lower ssl' is the wrong")
        print("           goal; look at the train/serve gap (Talker sees REAL latents in training,")
        print("           PREDICTED ones at generation) and at the data budget instead.")


if __name__ == "__main__":
    main()
