# Proverbs — a small MPS training run + GPT baseline comparison

A self-contained, small-scale exercise of the full latent-thought A→E curriculum on
one tiny corpus (the Book of Proverbs), plus a matched-scale plain-GPT baseline. This
directory is isolated from the main project: it does **not** touch the validated
training path or the upcoming big run — it only *uses* the repo's code (`../files/`)
with a local text source.

> Everything here runs on **Apple MPS** (the Mac `.venv`, torch 2.2.2), fp32 (AMP is a
> CUDA-only path and auto-disables off-CUDA).

## Contents

| File | What it is |
|---|---|
| `fetch_proverbs.py` | Download Proverbs (World English Bible, public domain) from bible-api.com; strip verse numbers/titles; format each **chapter as one prose document**. |
| `proverbs.jsonl` | The canonical data: one `{"text": <chapter prose>}` per line, 31 chapters (~14.5k words). |
| `proverbs_prose.txt` | Human-readable copy (blank line between chapters). |
| `prep_proverbs.py` | Pre-chunk `proverbs.jsonl` into the shard cache the trainer reads (reuses the repo's gpt2 chunker + `data_prep.prepare`). |
| `chat_proverbs.py` | Interactive tester: loads a checkpoint once, then `input()`-loops to generate / `:score` / `:chunks`. Shows chunk borders with `|`. |
| `baseline_proverbs.py` | Plain causal GPT baseline on the **same setup** (data, tokenizer, optimizer, schedule, 1800-step budget, batch 8, same seed-0 split); only the architecture differs. |
| `plot_proverbs_comparison.py` | Two-panel memorization/generalization figure + summary tables. |
| `runs/proverbs/` | Latent A→E run: `metrics.json`, `loss_curves.png` (checkpoints `*.pt` are local-only / git-ignored). |
| `runs/baseline_same_*/` | GPT baseline runs: `metrics.json`. |
| `runs/comparison_proverbs.png` | The head-to-head figure. |

Checkpoints (`model.pt`, `checkpoint.pt`) are **git-ignored** (450 MB latent, 171/54 MB
GPT) — regenerate them with the commands below.

## Reproduce

```bash
source ../.venv/bin/activate            # Mac venv, torch 2.2.2 (MPS)

python fetch_proverbs.py                # -> proverbs.jsonl (+ raw_chapters/ cache)
python prep_proverbs.py --preset smoke  # -> chunk_cache/ (31 examples, 6,623 tokens)

# Full A->E curriculum on MPS (fp32; ~13 min):
cd ../files
python train_scaled.py --preset smoke --cache proverbs-test/chunk_cache --device mps \
  --batch-size 8 --stage-steps 300,300,300,300,600,0 --num-workers 0 \
  --log-every 25 --checkpoint-every 500 --out proverbs-test/runs/proverbs
python plot_metrics.py proverbs-test/runs/proverbs
cd ../proverbs-test

python chat_proverbs.py                 # interactive; type text, :score <t>, :chunks <t>, :sep, :q

# GPT baselines (same data/budget/split; both fair scales):
python baseline_proverbs.py --preset same-params  --out runs/baseline_same_params
python baseline_proverbs.py --preset same-compute --out runs/baseline_same_compute
python plot_proverbs_comparison.py      # -> runs/comparison_proverbs.png
```

## Setup choices

- **Data formatting.** Each chapter → one prose document (verses joined, all whitespace
  collapsed). Verse numbers and section titles are absent from the source's `text` field,
  so nothing but the poetic line breaks needed removing. **Chapters are independent
  documents** — each gets its own gestalt memory (no cross-chapter context).
- **Preset `smoke`** (d192, ~43M): the right fit for 31 documents on MPS and the model
  the full curriculum was designed to exercise quickly. Bumping to `--preset small`
  (d512, ~153M) is a one-flag change in both `prep_proverbs.py` and the train command.
- **Split.** `train_scaled.py` derives a seed-0 split from the 31-example cache: **23
  train / 8 held-out** chapters (`[3, 9, 11, 15, 19, 20, 23, 30]`). `baseline_proverbs.py`
  reproduces this *exact* split so both architectures see the same train/held-out sets.

## Findings — latent-thought A→E run

The run is **healthy in every architectural sense** (see `runs/proverbs/loss_curves.png`):

- **No latent collapse.** The primary signal — `val_loss` across the A→B boundary where
  the predictor turns on — went **down, not up**: 5.076 (end of A) → 5.065 (start of B),
  and kept falling. `latent_std` *rose* 0.29 → 1.05 (healthy, far above the collapse
  floor), and `ssl` fell 4.0 → 0.06 (the loop learns to predict forward). `ponder` turned
  on at D and stayed near 0.
- **Memorizes the tiny train set** (expected at 23 documents): reconstruction `nll` →
  0.014 (train perplexity ~1.0).
- **Overfits on held-out chapters** (also expected at this size): `val_loss` bottomed
  ~5.0 around step 300, then drifted up to 5.63 (held-out perplexity ~280) as training
  continued. This is the tiny-corpus regime, **not** collapse.

Probe scoring (`--score`, autoencoder reconstruction perplexity):

| Text | Perplexity |
|---|---:|
| Proverbs 1:7 (verbatim, in train) | 1.0 |
| "A soft answer turns away wrath…" (reworded 15:1) | 729 |
| Modern finance sentence (out-of-domain) | 7,647 |

## Findings — GPT baseline comparison

Same data, tokenizer, optimizer, schedule, 1800-step budget, batch 8, MPS, and the same
seed-0 split — **only the architecture differs**. Two fair scales (total params vs width):

**Final perplexity** (lower = better under each model's own task):

| model | train ppl | held-out ppl |
|---|---:|---:|
| Latent-thought A→E (d192, autoencoder recon) | 1.0 | 279.9 |
| GPT same-params (d512×6, 44.7M, causal) | 1.1 | 1057.8 |
| GPT same-compute (d192×10, 14.1M, causal) | 309.2 | 594.5 |

**Probe-sentence perplexity** (each model's own scoring objective):

| probe | Latent | GPT-sp | GPT-sc |
|---|---:|---:|---:|
| Prov 1:7 (verbatim, train) | 1.0 | 1.4 | 220 |
| reworded 15:1 (held-out) | 729 | 602 | 1327 |
| out-of-domain finance | 7,647 | 158,764 | 24,580 |

**Reading it:**

- ⚠️ **Task asymmetry (governs all absolute numbers).** The latent model's reconstruction
  is an **autoencoder** — it conditions on an encoding of the chunk it decodes (lookahead
  through a 192-d bottleneck + FIFO memory). The GPT does **pure causal next-token** (no
  lookahead). Different objectives; the absolute nats are **not** directly comparable.
  Compare the *shape*, not the gap.
- **Both large models memorize** the 23 train chapters (ppl → ~1). The same-params GPT
  greedily generates fluent scripture (*"he who tills his land shall have plenty of bread,
  but he who chases fantasies is void of understanding"* — Prov 12:11).
- **At matched width (d192), the plain GPT underfits** — it never memorizes in this budget
  (train ppl 309, degenerate output), while the latent model at the same width drove train
  recon to ~1.0. Partly the latent task's lookahead advantage, not purely architecture.
- **Everyone overfits** on 31 chapters: held-out perplexity rises for all three after
  ~step 250. The reworded-verse probe is close between latent (729) and same-params GPT
  (602); the models diverge most on out-of-domain text, where the causal GPT is far more
  surprised (159k) than the lookahead autoencoder (7.6k).

## Honest limitations

- 31 chapters is a **memorization/overfit** regime, not a generalization benchmark. There
  is no claim of one architecture being "better" — the objectives differ and the corpus is
  tiny.
- `smoke`-scale output is not coherent by design; the point is to exercise the full
  pipeline end-to-end on real text and produce a like-for-like baseline.
- The latent chunker caps each chapter at `max_chunks_per_doc`×`max_chunk_len` tokens; the
  GPT sees full chapters. This is an architecture-driven data-capacity difference, noted
  rather than artificially equalized.
