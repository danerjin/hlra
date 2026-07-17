#!/usr/bin/env python
"""poster_figs.py — render the two poster figures for the "Results" section.

Writes PNGs sized to the EXACT placeholder rects in poster.py (4.8 x 2.55 in on the
48x36 board), so figsize inches map 1:1 to printed inches and font points are literal
printed points. poster.py picks them up automatically if they exist.

    python poster_figs.py                      # default run dir: runs/
    python poster_figs.py runs/scaled          # the A->E run
    python poster_figs.py runs/scaled --out poster_figs

Outputs:
    poster_figs/fig_loss.png   <- runs/<dir>/metrics.json   (automatic)
    poster_figs/fig_arc.png    <- poster_data/arc_c.json    (you fill this in)

Design notes (from the dataviz review):
  * NO dual axis. val_loss and latent_std are two stacked panels sharing the step
    axis -- never two y-scales on one plot.
  * Stage boundaries are hairline rules + letters, not 5 shaded bands: at 4.8in wide
    the bands are noise, and the story is one boundary (B = prediction turns on).
  * Single series per panel => no legend box; the panel title names the series.
  * Poster TEAL 0E7C6B is kept verbatim even though it measures chroma 0.094 vs the
    0.10 floor: this print sits inches from a dozen other TEAL elements and the
    floor's purpose (don't read as gray) is met -- dE 33.5 from the actual gray.
    TEAL<->MUTE is dE 10.0 (deutan), inside the 8-12 band, which is legal ONLY with
    secondary encoding -- hence the mandatory direct labels on fig_arc.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PROJECT = os.path.dirname(os.path.abspath(__file__))

# ---- poster palette (must match poster.py exactly) -------------------------
INK = "#10151F"; MUTE = "#53606E"; FAINT = "#93A0AE"
TEAL = "#0E7C6B"; BLUE = "#1D4ED8"; AMBER = "#B4560A"; NAVY = "#0B3B6F"
SOFT = "#C2CAD6"
SURFACE = "#FFFFFF"

# ---- placeholder geometry (poster.py: RW=10.0 -> (RW-0.4)/2 x 2.55) --------
FIG_W, FIG_H = 4.8, 2.55
DPI = 600

# ---- type scale, in PRINTED points ----------------------------------------
FS_TITLE = 11.5
FS_LABEL = 8.5
FS_TICK = 7.5
FS_NOTE = 7.0
FS_STAGE = 8.0

SANS = ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": SANS,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": SOFT,
    "axes.linewidth": 0.6,
    "xtick.color": MUTE,
    "ytick.color": MUTE,
    "text.color": INK,
    "axes.labelcolor": MUTE,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
})


def _run_dir(arg=None):
    if arg:
        return arg if os.path.isabs(arg) else os.path.join(PROJECT, arg)
    return os.path.join(PROJECT, "runs")


def _series(metrics, key):
    xs, ys = [], []
    for m in metrics:
        v = m.get(key)
        if v is not None:
            xs.append(m["step"]); ys.append(v)
    return xs, ys


def _stage_spans(metrics):
    """[(stage, start_step, end_step)] contiguous spans, in order."""
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


def _tidy(ax):
    """Recessive chrome: hairline solid y-grid, no top/right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color=SOFT, lw=0.4, ls="-", alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=FS_TICK, pad=1.5)


def _stage_marks(ax, spans, letters=True):
    """Hairline boundary rules + stage letters along the top. Not shaded bands."""
    for _, a, b in spans[1:]:
        ax.axvline(a, color=SOFT, lw=0.5, ls="-", zorder=1)
    if not letters:
        return
    for name, a, b in spans:
        ax.text((a + b) / 2, 1.03, name, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=FS_STAGE, color=MUTE, weight="bold")


def fig_loss(run_dir, out_path):
    path = os.path.join(run_dir, "metrics.json")
    with open(path) as f:
        metrics = json.load(f)
    if not metrics:
        raise SystemExit("[poster_figs] %s is empty" % path)

    spans = _stage_spans(metrics)
    vx, vy = _series(metrics, "val_loss")
    lx, ly = _series(metrics, "latent_std")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(FIG_W, FIG_H), sharex=True, dpi=DPI,
        gridspec_kw={"height_ratios": [1.5, 1], "hspace": 0.18})

    # -- panel 1: validation loss (single series -> no legend) --------------
    ax1.plot(vx, vy, "-", color=INK, lw=1.5, solid_capstyle="round",
             solid_joinstyle="round", zorder=3)
    if vx:
        # endpoint dot with a surface ring, + one selective direct label
        ax1.plot(vx[-1], vy[-1], "o", ms=3.4, mfc=INK, mec=SURFACE, mew=1.0, zorder=4)
        ax1.annotate("%.2f" % vy[-1], (vx[-1], vy[-1]), textcoords="offset points",
                     xytext=(-2, 6), ha="right", fontsize=FS_NOTE, color=INK, weight="bold")
    ax1.set_ylabel("validation loss", fontsize=FS_LABEL, labelpad=2)
    # title lives at figure level: an axes title would collide with the stage letters
    fig.text(0.012, 0.972, "Training curves, staged curriculum",
             fontsize=FS_TITLE, color=INK, weight="bold", ha="left", va="top")
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    _tidy(ax1)
    _stage_marks(ax1, spans, letters=True)

    # emphasis: the one boundary the poster's claim rests on
    b = next((a for name, a, _ in spans if name == "B"), None)
    if b is not None:
        ax1.annotate("prediction on", xy=(b, 0.06), xycoords=ax1.get_xaxis_transform(),
                     xytext=(6, 0), textcoords="offset points", fontsize=FS_NOTE,
                     color=AMBER, weight="bold", ha="left", va="bottom")
        ax1.axvline(b, color=AMBER, lw=0.9, ls="-", zorder=2, alpha=0.85)

    # -- panel 2: collapse monitor (single series -> no legend) -------------
    ax2.plot(lx, ly, "-", color=TEAL, lw=1.5, solid_capstyle="round",
             solid_joinstyle="round", zorder=3)
    if lx:
        ax2.plot(lx[-1], ly[-1], "o", ms=3.4, mfc=TEAL, mec=SURFACE, mew=1.0, zorder=4)
    # the floor is a real threshold -> dashed is meaningful here (grids are not)
    ax2.axhline(0.1, color=AMBER, lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax2.text(0.995, 0.1, "collapse floor", transform=ax2.get_yaxis_transform(),
             ha="right", va="bottom", fontsize=FS_NOTE, color=AMBER)
    ax2.set_ylabel("latent std", fontsize=FS_LABEL, labelpad=2)
    ax2.set_xlabel("training step", fontsize=FS_LABEL, labelpad=2)
    ax2.set_ylim(bottom=0)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))
    _tidy(ax2)
    _stage_marks(ax2, spans, letters=False)
    if b is not None:
        ax2.axvline(b, color=AMBER, lw=0.9, ls="-", zorder=2, alpha=0.85)

    fig.subplots_adjust(left=0.135, right=0.985, top=0.795, bottom=0.145)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print("[poster_figs] wrote %s  (%d pts, stages %s)"
          % (out_path, len(metrics), "->".join(s[0] for s in spans)))


def fig_arc(data_path, out_path):
    """ARC-C vs FLOPs. Reads poster_data/arc_c.json -- NO numbers are invented here.

    Schema: {"ours": {...}, "reference": [{...}]} with each entry
        {"label": str, "flops": float, "arc_c": float, "source": str}
    Every reference point needs a `source` you can defend at the board.
    """
    if not os.path.exists(data_path):
        print("[poster_figs] %s not found -> skipping fig_arc (poster keeps its "
              "'data coming soon' placeholder)" % data_path)
        return False
    with open(data_path) as f:
        d = json.load(f)

    ours = d.get("ours") or {}
    ref = [r for r in d.get("reference", []) if r.get("flops") and r.get("arc_c") is not None]
    if not ref or ours.get("arc_c") is None:
        print("[poster_figs] %s has no usable points yet -> skipping fig_arc "
              "(fill in ours.arc_c and >=1 reference entry)" % data_path)
        return False

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)

    # emphasis pattern: highlight ours, gray the rest.
    # TEAL<->MUTE is dE 10.0 (deutan) -> inside the 8-12 floor band, so the direct
    # labels below are MANDATORY secondary encoding, not decoration.
    rx = [r["flops"] for r in ref]; ry = [r["arc_c"] for r in ref]
    ax.plot(rx, ry, "o", ms=4.5, mfc=MUTE, mec=SURFACE, mew=1.0, ls="none", zorder=3)
    for r in ref:
        ax.annotate(r["label"], (r["flops"], r["arc_c"]), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=FS_NOTE, color=MUTE)

    ax.plot([ours["flops"]], [ours["arc_c"]], "o", ms=6.5, mfc=TEAL, mec=SURFACE,
            mew=1.2, zorder=4)
    ax.annotate(ours.get("label", "HLRA (ours)"), (ours["flops"], ours["arc_c"]),
                textcoords="offset points", xytext=(0, 8), ha="center",
                fontsize=FS_NOTE + 0.5, color=INK, weight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("training compute (FLOPs)", fontsize=FS_LABEL, labelpad=2)
    ax.set_ylabel("ARC-C accuracy (%)", fontsize=FS_LABEL, labelpad=2)
    ax.set_title("ARC-Challenge vs. training compute", fontsize=FS_TITLE,
                 color=INK, weight="bold", pad=9, loc="left")
    _tidy(ax)
    fig.subplots_adjust(left=0.135, right=0.975, top=0.845, bottom=0.165)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print("[poster_figs] wrote %s  (%d reference pts + ours)" % (out_path, len(ref)))
    return True


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_dir = os.path.join(PROJECT, "poster_figs")
    if "--out" in sys.argv:
        out_dir = sys.argv[sys.argv.index("--out") + 1]
        out_dir = out_dir if os.path.isabs(out_dir) else os.path.join(PROJECT, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    run_dir = _run_dir(args[0] if args else None)
    fig_loss(run_dir, os.path.join(out_dir, "fig_loss.png"))
    fig_arc(os.path.join(PROJECT, "poster_data", "arc_c.json"),
            os.path.join(out_dir, "fig_arc.png"))


if __name__ == "__main__":
    main()
