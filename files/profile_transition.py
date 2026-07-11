"""
profile_transition.py
=====================
Decision tool: what fraction of a training step is spent in the L-module's
DiagonalDecayGate? This is the number that says whether the loop-constant-e
caching rewrite (precompute B·e and the e-half of the residual once per L-group
instead of every L-step -- see decay_gate.py / the design notes) is worth the
truncated-BPTT risk it introduces.

Reuses bench_throughput's synthetic harness. Times *forward* only, via module
hooks. The per-hook device sync inflates absolute times, so read the SHARES,
not the seconds. Backward roughly mirrors the forward split.

Run (from files/):
    python profile_transition.py --preset small --batch-size 16
    python profile_transition.py --preset small --batch-size 16 --stage E   # ACT path
"""
from __future__ import annotations

import argparse
import collections
import time

import torch

from config import model_config
from model import LatentThoughtModel, StageFlags, SELF
from ema_target import EMATargetEncoder
from gestalt_memory import GestaltMemoryBank
from decay_gate import DiagonalDecayGate
from hrm_loop import HRMInnerLoop
from bench_throughput import pick_device, sync, synth_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--vocab", type=int, default=50258)
    ap.add_argument("--stage", default="C", choices=["C", "E"], help="C=fixed depth, E=ACT")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    device = pick_device()
    cfg = model_config(args.preset, vocab_size=args.vocab)
    model = LatentThoughtModel(cfg, chunker=None).to(device)
    ema = EMATargetEncoder(model.chunk_encoder, momentum=cfg.ema_momentum).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    flags = StageFlags(use_hrm_loop=True, detach_memory=False, inner_loop_grad_window=5,
                       memory_grad_window=5, use_act=(args.stage == "E"), use_input_lanes=False)

    # --- forward-time accounting via module hooks --------------------------
    # Category per module: the L-gate and H-gate (both DiagonalDecayGate, told
    # apart by attribute name) and the whole reasoner (HRMInnerLoop).
    totals = collections.defaultdict(float)
    starts: dict[int, float] = {}

    def make_hooks(cat):
        def pre(mod, inp):
            sync(device)
            starts[id(mod)] = time.perf_counter()
        def post(mod, inp, out):
            sync(device)
            totals[cat] += time.perf_counter() - starts[id(mod)]
        return pre, post

    for name, m in model.named_modules():
        if isinstance(m, DiagonalDecayGate):
            cat = "l_gate" if name.endswith("l_transition") else "h_gate"
        elif isinstance(m, HRMInnerLoop):
            cat = "reasoner"
        else:
            continue
        pre, post = make_hooks(cat)
        m.register_forward_pre_hook(pre)
        m.register_forward_hook(post)

    data = synth_batch(cfg, args.batch_size, device)

    def run_step():
        opt.zero_grad(set_to_none=True)
        memory = GestaltMemoryBank(cfg.memory_capacity, cfg.d_model)
        # The L-gate runs inside the on-loop SSL (the loop is here now, not in
        # reconstruction). data = (ct, cm, ri, rm).
        ssl, ponder = model.forward_self_supervised(*data, memory, SELF, flags, ema,
                                                    cos_weight=1.0, var_weight=2.0,
                                                    ponder_weight=cfg.act_ponder_cost)
        (ssl + ponder).backward()
        opt.step()

    for _ in range(args.warmup):
        run_step()
    totals.clear()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        run_step()
    sync(device)
    wall = (time.perf_counter() - t0)

    step_ms = wall / args.steps * 1e3
    def per_step_ms(cat):
        return totals[cat] / args.steps * 1e3

    print(f"device={device}  preset={args.preset}  batch={args.batch_size}  "
          f"stage={args.stage}  steps={args.steps}")
    print(f"full step (fwd+bwd+opt):        {step_ms:8.2f} ms")
    print(f"{'module (forward)':<24}{'ms/step':>10}{'% of step':>11}{'% of reasoner':>15}")
    reasoner_ms = per_step_ms("reasoner") or float("nan")
    for cat in ("reasoner", "l_gate", "h_gate"):
        ms = per_step_ms(cat)
        print(f"{cat:<24}{ms:>10.2f}{100*ms/step_ms:>10.1f}%{100*ms/reasoner_ms:>14.1f}%")

    # Upper bound on what the caching rewrite could save: it removes ~1/3 of the
    # L-gate's per-step matmuls (B·e + the e-half of the residual first layer),
    # amortized over l_steps_per_h_update; only the L-gate benefits.
    l_ms = per_step_ms("l_gate")
    est_fwd_save = 0.33 * l_ms
    print(f"\nest. forward saved by loop-constant-e caching (~1/3 of L-gate): "
          f"{est_fwd_save:.2f} ms/step  (~{100*est_fwd_save/step_ms:.1f}% of a step, forward only)")
    print("Rule of thumb: if that is under ~2-3% of a step, the caching is not "
          "worth perturbing the just-fixed truncated-BPTT path before the scaled run.")


if __name__ == "__main__":
    main()
