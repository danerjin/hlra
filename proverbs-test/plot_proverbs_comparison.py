"""
plot_proverbs_comparison.py
===========================
Side-by-side of the Proverbs run: latent-thought (A->E) vs a plain causal GPT at
two matched scales. Same tokenizer / optimizer / schedule / 1800-step budget /
batch 8 / MPS / same seed-0 train-val chapter split -- only the architecture differs.

Two panels vs training step (log-perplexity):
  LEFT  memorization    -> perplexity on the 23 TRAIN chapters
  RIGHT generalization  -> perplexity on the 8 HELD-OUT chapters

Honest asymmetry: the latent curves are AUTOENCODER reconstruction (lookahead
through a 192-d bottleneck); the GPT curves are pure CAUSAL next-token. Absolute
nats are NOT directly comparable across the two objectives -- read the SHAPE
(does it memorize train? how far does held-out sit above train?).
"""
import os, json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")


def latent_curves():
    m = json.load(open(os.path.join(RUNS, "proverbs", "metrics.json")))
    xs = [d["step"] for d in m]
    train = [math.exp(min(d["nll"], 20)) for d in m]        # recon NLL on train
    val = [math.exp(min(d["val_loss"], 20)) for d in m]     # recon NLL on held-out
    return xs, train, val


def gpt_curves(name):
    d = json.load(open(os.path.join(RUNS, name, "metrics.json")))
    m = d["metrics"]
    xs = [r["step"] for r in m]
    return xs, [r["train_ppl"] for r in m], [r["val_ppl"] for r in m], d


def main():
    lx, ltr, lva = latent_curves()
    spx, sptr, spva, spd = gpt_curves("baseline_same_params")
    scx, sctr, scva, scd = gpt_curves("baseline_same_compute")
    chance = math.exp(10.825)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=140, sharey=True)
    series = [
        ("Latent-thought A→E (43M, d192) — autoencoder recon", "#d62728", "-D", lx, ltr, lva),
        ("GPT same-params (44.7M, d512×6) — causal",           "#1f77b4", "-o", spx, sptr, spva),
        ("GPT same-compute (14.1M, d192×10) — causal",         "#2ca02c", "-s", scx, sctr, scva),
    ]
    for label, c, mk, xs, tr, va in series:
        axL.plot(xs, tr, mk, color=c, ms=4, lw=1.6, label=label, markevery=6)
        axR.plot(xs, va, mk, color=c, ms=4, lw=1.6, label=label, markevery=6)

    for ax, title in ((axL, "Memorization — 23 TRAIN chapters"),
                      (axR, "Generalization — 8 HELD-OUT chapters")):
        ax.axhline(chance, color="#7f7f7f", ls=":", lw=1.3, label=f"chance ≈ {chance:.0f}")
        ax.set_yscale("log")
        ax.set_xlabel("training step")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
    axL.set_ylabel("teacher-forced perplexity (log)")
    axR.legend(loc="upper right", fontsize=8, framealpha=0.95)

    fig.suptitle("Book of Proverbs — latent-thought vs plain GPT (same data, optimizer, budget; "
                 "architecture differs)", fontsize=12)
    fig.text(0.5, 0.005,
             "Latent curves are autoencoder reconstruction (lookahead through a 192-d bottleneck); "
             "GPT curves are pure causal next-token. Absolute nats not directly comparable across "
             "objectives — compare the shape.",
             ha="center", va="bottom", fontsize=8, color="#6b7280")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    out = os.path.join(RUNS, "comparison_proverbs.png")
    fig.savefig(out)
    print(f"[plot] wrote {out}")

    # ---- text summary table ----
    print("\n=== final perplexities (lower = better under each model's own task) ===")
    print(f"{'model':46s} {'TRAIN ppl':>11s} {'HELD-OUT ppl':>13s}")
    print(f"{'Latent-thought A→E (d192, autoencoder recon)':46s} {ltr[-1]:>11.1f} {lva[-1]:>13.1f}")
    print(f"{'GPT same-params (d512×6, causal)':46s} {sptr[-1]:>11.1f} {spva[-1]:>13.1f}")
    print(f"{'GPT same-compute (d192×10, causal)':46s} {sctr[-1]:>11.1f} {scva[-1]:>13.1f}")

    print("\n=== probe-sentence perplexity (each model's own scoring objective) ===")
    latent_probe = {"Proverbs 1:7 (verbatim, in TRAIN)": 1.0,
                    "reworded 15:1 (held-out/paraphrase)": 729.3,
                    "out-of-domain (modern finance)": 7647.0}
    print(f"{'probe':38s} {'Latent':>9s} {'GPT-sp':>9s} {'GPT-sc':>9s}")
    sp_p = {p["label"]: p["ppl"] for p in spd["probes"]}
    sc_p = {p["label"]: p["ppl"] for p in scd["probes"]}
    for label, lp in latent_probe.items():
        print(f"{label:38s} {lp:>9.1f} {sp_p.get(label, float('nan')):>9.1f} "
              f"{sc_p.get(label, float('nan')):>9.1f}")


if __name__ == "__main__":
    main()
