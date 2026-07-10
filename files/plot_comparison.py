"""
plot_comparison.py
==================
Rebuilds runs/comparison.png (notes §13.3): teacher-forced page perplexity vs
training step, memorizing one Wikipedia page, latent-thought vs a standard GPT at
two matched scales. Same tokenizer / optimizer / schedule / step budget; only the
architecture differs.

Curves (all read from runs/*/metrics.json):
  * Latent-thought grounded-only reconstruction  (wiki_overfit_grounded.py)
  * GPT same-params  (44.7M, d512x6)   causal      (baseline_gpt.py)
  * GPT same-compute (14.1M, d192x10)  causal      (baseline_gpt.py)
Reference lines:
  * Latent under the FULL A->E curriculum -- recomputed from runs/wiki_overfit
  * chance = vocab-uniform perplexity

Run:  python plot_comparison.py
"""
from __future__ import annotations

import os
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = os.path.join(PROJECT, "runs")

# Which baseline dirs to plot. Prefer the fresh *_v2 re-run if present, else the
# original.
def _pick(*names):
    for n in names:
        p = os.path.join(RUNS, n, "metrics.json")
        if os.path.exists(p):
            return p, n
    raise SystemExit(f"none of {names} found under runs/")


def _curve(path, key="page_ppl"):
    m = json.load(open(path))
    xs = [d["step"] for d in m if key in d]
    ys = [d[key] for d in m if key in d]
    return xs, ys


def full_curriculum_page_ppl():
    """Recompute the FULL-curriculum latent page ppl from the saved wiki_overfit
    checkpoint, so the reference line is consistent with this session's numbers
    (maps the legacy parcae_* config fields forward, per generate.py §22.3)."""
    import torch
    from dataclasses import fields
    from config import ModelConfig
    from model import LatentThoughtModel
    from data import CachedChunkDataset
    from wiki_overfit_grounded import page_ppl, pick_device

    ckpt_path = os.path.join(RUNS, "wiki_overfit", "model.pt")
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = dict(ckpt["model_cfg"])
    for old, new in {"parcae_min_decay": "decay_min", "parcae_max_decay": "decay_max"}.items():
        if old in raw and new not in raw:
            raw[new] = raw.pop(old)
    known = {f.name for f in fields(ModelConfig)}
    cfg = ModelConfig(**{k: v for k, v in raw.items() if k in known})
    device = pick_device()
    model = LatentThoughtModel(cfg, chunker=None).to(device)
    # strict=False: this checkpoint predates the gen_predictor head (§15.1),
    # which page_ppl (chunk_encoder -> hrm_loop -> talker) never touches anyway.
    model.load_state_dict(ckpt["model_state"], strict=False)
    ds = CachedChunkDataset(os.path.join(PROJECT, "wiki_cache"))
    _, ppl = page_ppl(model, ds, cfg, device)
    return ppl


def main():
    latent_path, _ = _pick("wiki_overfit_grounded")
    sp_path, sp_name = _pick("baseline_same_params_v2", "baseline_same_params")
    sc_path, sc_name = _pick("baseline_same_compute_v2", "baseline_same_compute")

    lx, ly = _curve(latent_path)
    spx, spy = _curve(sp_path)
    scx, scy = _curve(sc_path)

    # chance = uniform over the vocab (id 0 = PAD reserved, so 50257 real gpt2 ids).
    chance = 50257.0
    full = full_curriculum_page_ppl()

    fig, ax = plt.subplots(figsize=(1425 / 150, 870 / 150), dpi=150)
    ax.plot(lx, ly, "-D", color="#d62728", ms=6, lw=1.8,
            label="Latent-thought (43.1M, d192) — grounded-only, batch 16", zorder=5)
    ax.plot(spx, spy, "-o", color="#1f77b4", ms=6, lw=1.8,
            label="GPT same-params (44.7M, d512×6) — causal", zorder=4)
    ax.plot(scx, scy, "-s", color="#2ca02c", ms=6, lw=1.8,
            label="GPT same-compute (14.1M, d192×10) — causal", zorder=4)
    if full is not None:
        ax.axhline(full, color="#ff7f0e", ls="--", lw=1.8,
                   label=f"Latent under the FULL curriculum (A→E) = {full:.0f}")
    ax.axhline(chance, color="#7f7f7f", ls=":", lw=1.5, label=f"chance = {chance:.0f}")

    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("teacher-forced page perplexity (log)")
    ax.set_title("Memorizing one Wikipedia page — same tokenizer, optimizer, schedule")
    # Honest caveat baked into the figure: the GPT baselines memorize at batch 4;
    # the latent model does NOT at batch 4 (plateaus ~2000) and needs batch 16 to
    # get here, where it still floors ~20x above the same-params GPT.
    ax.text(0.5, -0.13, "GPT baselines batch 4; latent needs batch 16 to memorize and still plateaus "
                        "~20× above same-params GPT",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#6b7280")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="center right", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    out = os.path.join(RUNS, "comparison.png")
    fig.savefig(out)
    print(f"[plot_comparison] wrote {out}")
    print(f"  latent grounded-only: {ly[0]:.0f} -> {ly[-1]:.1f} ppl over {lx[-1]} steps")
    print(f"  GPT {sp_name}: -> {spy[-1]:.1f} ppl")
    print(f"  GPT {sc_name}: -> {scy[-1]:.1f} ppl")
    print(f"  full-curriculum ref: {full:.0f} ppl" if full else "  full-curriculum ref: (skipped)")


if __name__ == "__main__":
    main()
