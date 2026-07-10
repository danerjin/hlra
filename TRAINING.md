# TRAINING.md — How to Run the Big Training Run (step by step)

This is a **copy-paste guide**. If you can paste commands into a terminal and
compare numbers against the tables below, you can run this successfully. You do
**not** need to understand the model.

It covers the **A→E scaled run** (`train_scaled.py`), which is the only path
meant for a real training run. Stage F (chatbot fine-tuning) is **not** part of
this and is intentionally skipped.

> Read this whole page once before starting. Then do the steps in order. Do not
> skip the pre-flight checks (Steps 2–3) — they exist to stop you from wasting
> days of compute on a broken box.

---

## 0. The only two numbers you must watch

While training runs, every log line prints many numbers. **Two of them decide
success or failure.** If you only remember one thing from this page, remember
this table:

| Number in the log | Name | ✅ Healthy | 🚨 Abort / investigate |
|---|---|---|---|
| `val_loss=` | reconstruction quality (lower = better) | goes **down**, then flattens | **rises and keeps rising**, especially right after Stage D starts |
| `lstd=` | latent health (collapse monitor) | stays **above 0.15** (normal range 0.2–0.7) | **drops below 0.1** and stays there |

Everything else is diagnostic. As long as `val_loss` is trending down (or flat)
and `lstd` is above ~0.1, the run is healthy. The rest of this document is about
getting to that point and keeping it there.

`val_loss` measures the **autoencoder-like reconstruction loss**: the model
encodes a chunk of text into a latent "thought," then decodes that same chunk
back out, and `val_loss` is how well the decoded text matches the original. It's
autoencoder-*like* (not a plain autoencoder) because the encode→decode path runs
through the reasoning loop and a latent bottleneck. This loss is the run's
**anti-collapse anchor** — a model can't reconstruct varied text from a
collapsed (constant) latent, so keeping it healthy is what keeps the whole run
healthy.

A "chance" (untrained) `val_loss` is about **10.8**. A good run drives it well
below that over time. It will never reach GPT-2 quality at this scale — that is
expected and fine (see `README.md`); the goal is a **healthy, non-collapsed**
run, not a specific perplexity.

---

## 1. Prerequisites: GPU and PyTorch

> ⚠️ **Before starting this guide, read and follow `STRIX_HALO.md`** (at the repo
> root). It has the ROCm setup and the exact `pip install` command for PyTorch
> on your GPU hardware. Come back here once PyTorch is installed and
> `torch.cuda.is_available()` returns `True` on your GPU box.

Once that's done, you need:

- A machine with the **AMD ROCm GPU** (Strix Halo / Radeon 8060S) — or any CUDA
  GPU. The AMD GPU shows up to the code as "cuda"; that is normal.
- Python with **PyTorch (GPU build)**, plus `datasets`, `transformers`,
  `matplotlib`.
- The local GPT-2 tokenizer folder must exist at the project root: **`gpt2_tok/`**
  (it should already be there — check below).
- Disk space: roughly **~10 GB** for the prepared data cache, plus a few GB per
  checkpoint.

Throughout this guide, **`PROJECT`** means the top folder of this repository
(the one that contains `README.md`, `notes.md`, and the `files/` folder). Set it
once and reuse it:

```bash
# EDIT this line to the real path on your machine, then paste it:
export PROJECT=/path/to/ucsc
cd "$PROJECT"
```

For AMD gfx1151, some ROCm builds also need this (harmless if not needed). Paste
it once per terminal session:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1
```

See `STRIX_HALO.md` for the full ROCm setup notes if the GPU isn't detected.

---

## 2. One-time environment setup

Activate the project's Python environment and install the remaining training
dependencies (PyTorch must already be installed from Step 1 / `STRIX_HALO.md`):

```bash
cd "$PROJECT"
source .venv/bin/activate          # activates the project environment

# Install the other training deps:
pip install datasets transformers matplotlib
```

> Do **not** run `pip install -r requirements.txt` — that file is unrelated to
> this project and will not install what you need.

**Verify the environment is ready.** Paste this exactly. It must print a line
ending in `ENV OK`:

```bash
cd "$PROJECT/files"
python - <<'PY'
import torch, os
print("torch:", torch.__version__)
print("GPU visible (want True on the GPU box):", torch.cuda.is_available())
tok = os.path.join(os.path.dirname(os.getcwd()), "gpt2_tok")
print("gpt2_tok present (want True):", os.path.isdir(tok))
assert os.path.isdir(tok), "MISSING gpt2_tok/ — see README 'gpt2 tokenizer download workaround'"
print("ENV OK")
PY
```

- If `GPU visible` is `False` on the GPU box → **stop**. Your PyTorch is not the
  GPU build, or ROCm isn't set up. Fix this before continuing (`STRIX_HALO.md`).
- If `gpt2_tok present` is `False` → download the five GPT-2 tokenizer files into
  `gpt2_tok/` (see `README.md`, section "gpt2 tokenizer download workaround").

**From here on, all commands are run from inside the `files/` folder:**

```bash
cd "$PROJECT/files"
```

---

## 3. Pre-flight check #1 — does the GPU run the model? (2 minutes)

This runs the model on fake data and checks nothing breaks. **No downloads, no
real data.** Run it:

```bash
python rocm_smoke.py --preset small
```

**What you want to see:** several lines, then a final line starting with
**`PASS:`**. Specifically every check should say `finite: True`:

```
[2] bf16 matmul finite: True
[3] built small model: ... M params on cuda
    grounded loss finite: True (nll=...)   grads finite: True
[4] SSL+gen loss finite: True (...)
[5] ACT (stage E) loss finite: True (...)
================================================================
PASS: training path runs and stays finite on this GPU under bf16 autocast...
```

| Result | What it means | What to do |
|---|---|---|
| Ends with `PASS:` | GPU + mixed precision work | ✅ go to Step 4 |
| Ends with `FAIL:` | a number came out NaN/Inf | 🚨 **stop.** Do not train. Note which `[n]` check failed and see Troubleshooting. Try `--amp-dtype bf16` (it is the default and recommended). |
| Exits saying device is not `cuda` | GPU not detected | 🚨 fix PyTorch/ROCm first (`STRIX_HALO.md`) |

---

## 4. Pre-flight check #2 — how long will it take? (5 minutes)

This measures speed at different batch sizes so you can (a) pick the biggest
batch that fits your memory and (b) see the estimated wall-clock time. **Still
no real data.**

```bash
python bench_throughput.py --preset small --batch-size 16,32,64,128 --amp --token-budget 1200000000
```

Read the output table. For each batch size it prints `step time`, `real tok/s`,
`peak GB`, and a `budget ETA` (in days).

**How to choose your batch size (write it down — you'll reuse it):**

1. Look at the `peak GB` column. **Pick the largest batch size whose `peak GB`
   comfortably fits your GPU memory** (leave ~20% headroom; on the 128 GB Strix
   Halo almost anything fits).
2. Check that batch's `budget ETA`. That's roughly how many days a ~1.2B-token
   run will take. If it's unacceptably long, a smaller model or fewer tokens is
   the lever — but the numbers are what they are; better to know now.

Call your chosen batch size **`BATCH`** below (e.g. `BATCH=64`).

---

## 5. Prepare the data (one-time; can take a few hours)

Training reads from a **pre-chunked cache**, not raw text. You build that cache
once with `data_prep.py`. This is the slow part but you only do it once.

### 5a. First, a tiny timed dry-run (do NOT skip)

This makes sure the dataset downloads/streams and chunks correctly before you
commit hours. It prepares only 1000 documents:

```bash
python data_prep.py \
  --dataset HuggingFaceFW/fineweb-edu --name sample-10BT --streaming \
  --preset small --docs 1000 --out chunk_cache_dryrun
```

**What you want to see:** a source line, then progress, then a final
`[data_prep] wrote N examples (~T tokens) ...`:

```
[data_prep] source=HuggingFaceFW/fineweb-edu:sample-10BT (streaming)
[data_prep] preset=small vocab=50258 out=.../chunk_cache_dryrun chunk_dims=(64,32,256)
  prepared 500 examples, ~... tokens
[data_prep] wrote ... examples (~... tokens) in ... shards to .../chunk_cache_dryrun
```

- **Time this dry-run.** If 1000 docs take `M` minutes, the full run below scales
  roughly linearly with the number of documents. Use that to sanity-check the
  full prep won't take absurdly long.
- If it errors on the dataset name, the most common cause is a missing `--name`
  (see Troubleshooting). Delete the dry-run folder and retry.

Delete the dry-run cache when satisfied:

```bash
rm -rf "$PROJECT/chunk_cache_dryrun"
```

### 5b. The real prep (~1.2 billion tokens)

> ⚠️ Always prep into a **fresh, empty directory**. Re-prepping into an existing
> cache directory can silently mix old and new data. If you need to redo it,
> delete the folder first.

```bash
python data_prep.py \
  --dataset HuggingFaceFW/fineweb-edu --name sample-10BT --streaming \
  --preset small --max-tokens 1200000000 --out chunk_cache
```

This writes the cache to `PROJECT/chunk_cache/`. When it finishes it prints the
total examples and tokens. **Write down the two numbers** from the final line —
you'll use them in Step 6:

```
[data_prep] wrote  <EXAMPLES>  examples (~ <TOKENS>  tokens) in ... shards to .../chunk_cache
```

> **Important:** `--preset small` here **must match** `--preset small` in
> training (Step 7). The chunk dimensions are baked into the cache; using a
> different preset later will error out (that's a safety check, not a bug).

---

## 6. Choose how long to train (the stage budget)

Training runs in five stages, A→B→C→D→E, each for a fixed number of **optimizer
steps** you set with `--stage-steps A,B,C,D,E,F`. (The last number, F, stays
`0` — Stage F is not trained here.)

**If you do nothing, the built-in default is far too short** (it under-trains by
~5×). You must pass `--stage-steps` explicitly.

Your prep in Step 5b capped the cache at ~1.2 billion tokens (the recommended
budget for the `small` model). So the right amount of training is **one pass
("epoch") over that cache**, which is simply `examples ÷ batch` steps. You don't
have to guess — this command computes the exact `--stage-steps` for you.

**Paste this after prep finishes.** Set `BATCH` to the value you chose in Step 4,
and make sure `--cache` matches what you prepped (`chunk_cache`):

```bash
export BATCH=64        # <-- your batch size from Step 4
python - <<'PY'
import json, os
cache = os.path.join(os.path.dirname(os.getcwd()), "chunk_cache", "manifest.json")
m = json.load(open(cache)); ex = m["total"]; B = int(os.environ["BATCH"])
total = max(5, ex // B)          # ~1 epoch over the prepped ~1.2B-token cache
unit  = max(1, total // 6)       # split A:B:C:D:E = 1:1:1:1:2 (E is double)
print(f"examples={ex}  batch={B}  ->  ~1 epoch = {total} steps")
print(f"--stage-steps {unit},{unit},{unit},{unit},{2*unit},0")
PY
```

It prints a line like `--stage-steps 2500,2500,2500,2500,5000,0`. **Copy that
value** — that is your `STAGE_STEPS` for Step 7. (The A=B=C=D, E=double pattern
gives adaptive-depth the most settling time.)

Want to train longer (more passes) for a stronger model? Multiply every number
by 2 (two epochs), 3 (three epochs), etc. More is generally better here, at the
cost of proportionally more wall-clock time.

Call the value you copied **`STAGE_STEPS`** below.

---

## 7. Launch training

Now start the run. Two ways: a quick foreground test, then the real background
run.

### 7a. A 5-minute sanity launch (recommended before the real run)

Start a **tiny** run just to confirm everything is wired up and the first log
lines look healthy. This uses tiny stage budgets so it finishes fast:

```bash
python train_scaled.py --preset small --cache chunk_cache --device cuda --amp \
  --batch-size 16 --stage-steps 5,5,5,5,5,0 --log-every 1 --checkpoint-every 0 \
  --out runs/scaled_sanity
```

Confirm you see stages advancing (`>>> curriculum advanced to stage B`, `C`, …)
and that `val_loss` and `lstd` print as normal numbers (not `nan`). Then delete
the sanity output and move on:

```bash
rm -rf "$PROJECT/runs/scaled_sanity"
```

### 7b. The real run (background, survives disconnects)

Replace `BATCH` and `STAGE_STEPS` with your values from Steps 4 and 6. This runs
in the background and writes everything to `train.log`:

```bash
cd "$PROJECT/files"
nohup python train_scaled.py \
  --preset small --cache chunk_cache --device cuda --amp --amp-dtype bf16 \
  --batch-size BATCH \
  --stage-steps STAGE_STEPS \
  --num-workers 8 \
  --log-every 50 \
  --checkpoint-every 1000 \
  --archive-every 5000 \
  --out runs/scaled \
  > train.log 2>&1 &
```

- `--checkpoint-every 1000` saves a resumable checkpoint every 1000 steps (so a
  crash loses at most ~1000 steps).
- `--archive-every 5000` also keeps permanent numbered snapshots
  (`checkpoint_0005000.pt`, …) so you can roll back if a problem is noticed late.
- `--num-workers 8` speeds up data loading; lower it if you see worker/memory
  errors.

**Watch it live.** These two commands are your dashboard:

```bash
# See everything as it happens (Ctrl-C to stop watching; training keeps running):
tail -f "$PROJECT/files/train.log"

# Or watch ONLY the two numbers that matter:
tail -f "$PROJECT/files/train.log" | grep --line-buffered -o 'val_loss=[0-9.]*\|lstd=[0-9.]*'
```

A healthy log line looks like this:

```
[step 2000] stage=B lr=3.00e-04 logs={'nll': 6.9, 'ponder': 0.0} val_loss=6.85 lstd=0.42
```

---

## 8. Monitor the run (check a few times a day)

Every logged step prints a line. Compare the numbers to this reference. You are
looking for **trends**, not exact values.

### The full number guide

| Field | What it is | Healthy behavior | Warning sign |
|---|---|---|---|
| `stage=` | current stage (A→E) | advances A→B→C→D→E over the run | stuck far past its budget (shouldn't happen with `--stage-steps`) |
| `val_loss=` | **reconstruction quality** (autoencoder-like: encode→reason→decode the same chunk) | **decreasing**, then flat; starts ~10.8 | **rising** over many logs (esp. after Stage D begins) |
| `lstd=` | **latent health** | stays **> 0.15** (0.2–0.7 normal) | **< 0.1** and staying there |
| `nll` | training reconstruction loss | decreasing | rising steadily |
| `ssl` | self-supervised loss (**starts at Stage D**) | small, wobbles ~0.02–0.5 | racing to ~0.0 **while `val_loss` rises** = collapse |
| `gen` | generation-head loss (**starts at Stage D**) | decreasing | doesn't matter for run health; ignore unless `nan` |
| `ponder` | compute cost (**starts at Stage E**) | small (~0.01–0.05) | `nan` |
| `lr` | learning rate | rises during warmup, then eases down each stage | — |

### What "collapse" looks like (the one failure to catch early)

The known failure mode of this architecture is **latent collapse** at Stage D,
when the self-supervised loss turns on. It looks like this:

- `ssl` drops quickly toward **0.0**, **and at the same time**
- `val_loss` **rises**, **and**
- `lstd` **falls below ~0.1**.

**All three together = collapse. Stop the run** (see Step 9 to kill it) and see
Troubleshooting → "Latent collapse". If `ssl` is small but `val_loss` is flat or
falling and `lstd` is healthy, that is **normal** — `ssl` is a secondary signal
and a low value alone is fine.

### The single most important comparison

Look at `val_loss` and `lstd` **at the last step of Stage C** vs **during Stage
D/E**. Grab them quickly:

```bash
grep 'stage=C' "$PROJECT/files/train.log" | tail -1
grep 'stage=D' "$PROJECT/files/train.log" | tail -1
grep 'stage=E' "$PROJECT/files/train.log" | tail -1
```

Across those three lines: `val_loss` should be **flat or lower**, and `lstd`
should **stay above ~0.1**. If `val_loss` jumps up a lot and `lstd` dives at
Stage D, that's the collapse pattern.

---

## 9. Stopping, resuming, and crashes

### Stop the run on purpose

```bash
# Find the process:
pgrep -af train_scaled.py
# Stop it (replace 12345 with the PID printed above):
kill 12345
```

### Resume after a crash, a stop, or a reboot

The run checkpoints itself. To continue from where it left off, launch the
**exact same command** as Step 7b but add `--resume`:

```bash
cd "$PROJECT/files"
nohup python train_scaled.py \
  --preset small --cache chunk_cache --device cuda --amp --amp-dtype bf16 \
  --batch-size BATCH \
  --stage-steps STAGE_STEPS \
  --num-workers 8 --log-every 50 --checkpoint-every 1000 --archive-every 5000 \
  --out runs/scaled \
  --resume "$PROJECT/runs/scaled/checkpoint.pt" \
  >> train.log 2>&1 &
```

**Critical:** the flags after `--resume` (especially `--stage-steps`,
`--batch-size`, `--lr`) **must be identical** to the original launch. If they
differ, the trainer prints a loud multi-line `WARNING: resume schedule differs
from checkpoint` block. If you see that warning and you did **not** intend to
change anything, **stop and fix your command** — otherwise the run's schedule
will silently drift.

A clean resume prints, with no warning block:

```
[trainer] resumed from .../checkpoint.pt at step NNNN (stage X)
```

---

## 10. When it finishes

The run stops on its own after Stage E and prints:

```
[trainer] checkpoint -> .../runs/scaled/model.pt (step ...)
[train_scaled] done. final stage F, step ...
```

("final stage F" just means "finished E and would start F" — F is not trained.
That's expected.)

### Plot the curves (optional, needs matplotlib)

The metrics are saved in `runs/scaled/metrics.json`. To see the curves, **pass
the run directory** (otherwise it looks in the wrong place):

```bash
python plot_metrics.py runs/scaled     # writes runs/scaled/loss_curves.png
```

Open `runs/scaled/loss_curves.png` and check the same two things visually: the
`val_loss` curve trends down/flat (no big rise at the D/E bands), and the
`latent_std` panel stays above 0.1.

### Try the model (optional)

Because you trained from a **real GPT-2** cache, the checkpoint is decodable:

```bash
python generate.py --ckpt "$PROJECT/runs/scaled/model.pt" "The history of science shows that"
python generate.py --ckpt "$PROJECT/runs/scaled/model.pt" --score "a sentence to measure perplexity on"
```

Output will be real words but not coherent at this scale — that is expected and
documented; it is not a sign the run failed.

> Do **not** point `generate.py` at a checkpoint that was trained from an
> `--offline` (synthetic) cache — those aren't real GPT-2 tokens and the output
> is meaningless.

---

## 11. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `rocm_smoke.py` prints `FAIL` or a `nan` | GPU kernel / mixed-precision issue | Ensure you're on `--amp-dtype bf16` (default). If bf16 also fails, note which `[n]` check failed; the GPU/ROCm build is suspect — see `STRIX_HALO.md`. **Do not train until this passes.** |
| "device resolved to 'cpu'/'mps', not cuda" | PyTorch isn't the GPU build, or ROCm not visible | Install a ROCm/CUDA torch wheel; set `HSA_OVERRIDE_GFX_VERSION` (Strix Halo). `STRIX_HALO.md`. |
| `data_prep` starts a huge multi-TB download | Missing `--name sample-10BT`, or missing `--streaming` | Always pass **both** `--name sample-10BT` and `--streaming` for fineweb-edu. Delete the partial cache and retry. |
| `cache/model mismatch on max_chunk_len ...` at training start | Cache was prepped with a different `--preset` than training | Re-prep with the **same** preset, or train with the preset the cache was built for. |
| `cache inconsistent: manifest says X but shards hold Y` | You re-prepped into an existing cache dir | Delete the cache folder and prep again into a **fresh, empty** directory. |
| Out-of-memory (OOM) at training start | `--batch-size` too big | Lower `--batch-size`. To keep a large *effective* batch, add e.g. `--grad-accum 4` (does 4 small batches per step). |
| `val_loss` rising + `ssl` → 0 + `lstd` < 0.1 (all three) | **Latent collapse** | Stop the run. This path already keeps the anti-collapse anchor on every step (`train_scaled.py` hardcodes `grounded_loss_min_frequency=1.0` — don't lower it). If collapse still happens at scale, lower the SSL weight in `config.py` (`ssl_loss_weight` from `0.1` toward `0.05`), then relaunch from the last healthy `--archive-every` snapshot (no re-prep needed). Re-tuning anti-collapse settings at scale is expected (see `notes.md` §9). |
| Loud `WARNING: resume schedule differs from checkpoint` | `--resume` flags don't match the original launch | Make the command identical to the first launch (same `--stage-steps`, `--batch-size`, `--lr`). Only ignore if you *intended* to change the schedule. |
| Everything is `nan` from the very first step | mixed-precision or data issue | Re-run `rocm_smoke.py`. If that passes but training NaNs, try without `--amp` once to isolate; report which stage it first appears in. |
| Training extremely slow / GPU underused | batch too small (this model is launch-overhead-bound) | Increase `--batch-size` (the 128 GB unified memory is meant for this); re-check `bench_throughput.py`. |
| DataLoader worker errors / memory spikes at start | `--num-workers` too high for the box | Lower `--num-workers` (e.g. to `2`, or `0`). |

---

## 12. Full command cheat-sheet

Fill in `PROJECT`, `BATCH`, and `STAGE_STEPS` once, then these are all you need.

```bash
# ---- setup (once per terminal) ----
export PROJECT=/path/to/ucsc
export HSA_OVERRIDE_GFX_VERSION=11.5.1      # AMD gfx1151 only; harmless otherwise
cd "$PROJECT" && source .venv/bin/activate && cd files

# ---- pre-flight ----
python rocm_smoke.py --preset small
python bench_throughput.py --preset small --batch-size 16,32,64,128 --amp --token-budget 1200000000

# ---- prepare data (dry-run, then real; fresh dir each time) ----
python data_prep.py --dataset HuggingFaceFW/fineweb-edu --name sample-10BT --streaming \
  --preset small --docs 1000 --out chunk_cache_dryrun          # dry-run; delete after
python data_prep.py --dataset HuggingFaceFW/fineweb-edu --name sample-10BT --streaming \
  --preset small --max-tokens 1200000000 --out chunk_cache     # real prep

# ---- compute STAGE_STEPS (~1 epoch); copy the printed --stage-steps value ----
export BATCH=64
python - <<'PY'
import json, os
m=json.load(open(os.path.join(os.path.dirname(os.getcwd()),"chunk_cache","manifest.json")))
ex=m["total"]; B=int(os.environ["BATCH"]); total=max(5,ex//B); unit=max(1,total//6)
print(f"--stage-steps {unit},{unit},{unit},{unit},{2*unit},0")
PY

# ---- train (background) ----
nohup python train_scaled.py --preset small --cache chunk_cache --device cuda --amp --amp-dtype bf16 \
  --batch-size BATCH --stage-steps STAGE_STEPS \
  --num-workers 8 --log-every 50 --checkpoint-every 1000 --archive-every 5000 \
  --out runs/scaled > train.log 2>&1 &

# ---- monitor ----
tail -f train.log
grep 'stage=C' train.log | tail -1 ; grep 'stage=D' train.log | tail -1 ; grep 'stage=E' train.log | tail -1

# ---- resume after a stop/crash (same flags + --resume) ----
nohup python train_scaled.py --preset small --cache chunk_cache --device cuda --amp --amp-dtype bf16 \
  --batch-size BATCH --stage-steps STAGE_STEPS \
  --num-workers 8 --log-every 50 --checkpoint-every 1000 --archive-every 5000 \
  --out runs/scaled --resume "$PROJECT/runs/scaled/checkpoint.pt" >> train.log 2>&1 &

# ---- when done ----
python plot_metrics.py runs/scaled
python generate.py --ckpt "$PROJECT/runs/scaled/model.pt" --score "a sentence to score"
```

---

## 13. Glossary (plain English)

- **Stage (A–E):** the run trains in five phases; each turns on one more part of
  the model. You don't manage them — `--stage-steps` sets how long each lasts.
- **`val_loss` / reconstruction loss:** the **autoencoder-like** loss — how well
  the model rebuilds a chunk of text after squeezing it through a latent
  "thought" (encode → reason → decode the same chunk). Lower is better.
- **`lstd` / latent_std:** a health check for "did the model's internal
  representation go flat/dead." Above ~0.1 = alive.
- **Collapse:** the specific failure where the model cheats the self-supervised
  loss by making every internal vector identical. Caught by `lstd` dropping and
  `val_loss` rising together.
- **Checkpoint:** a saved snapshot you can resume from. `checkpoint.pt` is the
  latest; `checkpoint_XXXXXXX.pt` are numbered backups.
- **Preset (`small`):** the model size. The data cache and training must use the
  **same** preset.
- **AMP / bf16:** mixed precision — runs faster on the GPU using a compact number
  format. Use `bf16` (the default).
- **Token budget (~1.2B):** roughly how much text the model sees. More tokens =
  longer run. The `small` model targets ~1.2 billion.

---

**If in doubt:** a run is healthy as long as **`val_loss` trends down or flat**
and **`lstd` stays above ~0.1**. Watch those two, keep checkpoints
(`--checkpoint-every` / `--archive-every`), and you're in good shape.
