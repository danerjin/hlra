"""
plot_metrics.py
===============
Reads <run_dir>/metrics.json (written by train_real.py or trainer.Trainer) and
renders training curves with matplotlib to <run_dir>/loss_curves.png:

  * panel 1: validation loss over steps, with stage boundaries shaded/labelled
  * panel 2: training grounded NLL and self-supervised (SSL) loss over steps
  * panel 3: the latent-std collapse monitor

Run:  python plot_metrics.py                 # default ../runs/
      python plot_metrics.py runs/shakedown_small   # any run dir (rel to project or absolute)
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_dir() -> str:
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        return arg if os.path.isabs(arg) else os.path.join(PROJECT, arg)
    return os.path.join(PROJECT, "runs")


RUN_DIR = _run_dir()


def _series(metrics, key):
    """(steps, values) for entries where `key` is present and not None."""
    xs, ys = [], []
    for m in metrics:
        v = m.get(key)
        if v is not None:
            xs.append(m["step"]); ys.append(v)
    return xs, ys


def _stage_spans(metrics):
    """List of (stage_name, start_step, end_step) contiguous spans."""
    spans, cur, start = [], None, None
    last = metrics[0]["step"] if metrics else 0
    for m in metrics:
        if m["stage"] != cur:
            if cur is not None:
                spans.append((cur, start, m["step"]))
            cur, start = m["stage"], m["step"]
        last = m["step"]
    if cur is not None:
        spans.append((cur, start, last))
    return spans


def main():
    path = os.path.join(RUN_DIR, "metrics.json")
    with open(path) as f:
        metrics = json.load(f)
    if not metrics:
        raise SystemExit("no metrics to plot -- run train_real.py first")

    spans = _stage_spans(metrics)
    palette = {"A": "#dbeafe", "B": "#dcfce7", "C": "#fef9c3",
               "D": "#fee2e2", "E": "#ede9fe", "F": "#f3e8ff"}

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 9.5), sharex=True)

    def shade(ax):
        for name, a, b in spans:
            ax.axvspan(a, b, color=palette.get(name, "#eeeeee"), alpha=0.6, zorder=0)
            ax.text((a + b) / 2, ax.get_ylim()[1], f" {name}", va="top", ha="center",
                    fontsize=9, color="#374151")

    # Panel 1: validation loss
    vx, vy = _series(metrics, "val_loss")
    ax1.plot(vx, vy, "-o", color="#111827", ms=3, lw=1.5, label="val_loss", zorder=3)
    ax1.set_ylabel("validation loss")
    ax1.set_title(f"Latent-Thought model — {os.path.basename(RUN_DIR.rstrip('/'))} "
                  f"({'->'.join(s[0] for s in spans)})")
    shade(ax1)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.25)

    # Panel 2: training NLL + SSL
    nx, ny = _series(metrics, "nll")
    sx, sy = _series(metrics, "ssl")
    if nx:
        ax2.plot(nx, ny, "-o", color="#2563eb", ms=3, lw=1.4, label="grounded NLL (train)", zorder=3)
    if sx:
        ax2.plot(sx, sy, "-s", color="#dc2626", ms=3, lw=1.4, label="self-supervised loss", zorder=3)
    ax2.set_ylabel("training loss")
    shade(ax2)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.25)

    # Panel 3: collapse monitor -- mean per-dim std of the shared chunk latent.
    lx, ly = _series(metrics, "latent_std")
    if lx:
        ax3.plot(lx, ly, "-o", color="#059669", ms=3, lw=1.4, label="latent std (collapse monitor)", zorder=3)
        ax3.axhline(0.1, color="#9ca3af", ls="--", lw=1, label="collapse floor")
    ax3.set_ylabel("latent std")
    ax3.set_xlabel("step")
    ax3.set_ylim(bottom=0)
    shade(ax3)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.25)

    fig.tight_layout()
    out = os.path.join(RUN_DIR, "loss_curves.png")
    fig.savefig(out, dpi=150)
    print(f"[plot_metrics] wrote {out}  ({len(metrics)} points, stages: "
          f"{'->'.join(s[0] for s in spans)})")


if __name__ == "__main__":
    main()
