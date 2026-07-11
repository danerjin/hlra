# TRAINING.md — How to run the big training run

A copy-paste guide for the **A→E scaled run** (`train_scaled.py`) — the only path meant for a real
run. Stage F (dialogue fine-tuning) is intentionally skipped. Do the steps in order; don't skip the
pre-flight checks.

---

## 0. The two numbers that decide success

| Log field | What it is | ✅ Healthy | 🚨 Abort / investigate |
|---|---|---|---|
| `val_loss` | reconstruction quality (**the primary collapse signal**) | falls, then flattens; **keeps falling through the Stage-B boundary** | **jumps up right when Stage B starts** and keeps rising |
| `lstd` | latent health (secondary, **width-dependent**) | holds its own Stage-A band | craters *well below* that band and stays there |

`val_loss` is the **autoencoder reconstruction**: encode a chunk → the Talker decodes that same chunk
(a pure codec, no reasoning loop). A constant latent can't reconstruct varied text, so this is the
run's anti-collapse anchor. Chance (untrained) is ~10.8; a good run drives it well below.

The reasoning loop is trained by the **separate predictive loss `ssl`**, which turns on at **Stage B**.
Collapse, if it happens, shows as `val_loss` rising *right when `ssl` turns on*. It will never reach
gpt2 quality at this scale — that's expected; the goal is a healthy, non-collapsed run.

> ⚠️ **Do not abort on an absolute `lstd`.** The old "collapse < 0.1" rule was calibrated on the 192-d
> smoke model; at `small` (512-d) the natural `lstd` is lower and *rises* over a healthy run (a
> validated run went 0.14 → 0.79). Note the `lstd` band in Stage A and only worry if it craters below
> it. **`val_loss` at the Stage-B boundary is the signal that matters.**

---

## 1. Prerequisites

Follow [`STRIX_HALO.md`](STRIX_HALO.md) first for the ROCm/GPU PyTorch install. Then you need: the GPU
box (AMD ROCm shows up as "cuda", which is normal) with a GPU PyTorch build + `datasets transformers
matplotlib`; the local gpt2 tokenizer at `gpt2_tok/`; ~10 GB disk for the cache.

```bash
export PROJECT=/path/to/ucsc          # edit to your path
export HSA_OVERRIDE_GFX_VERSION=11.5.1 # AMD gfx1151 only; harmless otherwise
cd "$PROJECT" && source .venv/bin/activate
pip install datasets transformers matplotlib
cd "$PROJECT/files"
python - <<'PY'
import torch, os
print("GPU visible (want True):", torch.cuda.is_available())
assert os.path.isdir(os.path.join(os.path.dirname(os.getcwd()),"gpt2_tok")), "MISSING gpt2_tok/"
print("ENV OK")
PY
```

If `GPU visible` is `False` on the GPU box, stop and fix PyTorch/ROCm (`STRIX_HALO.md`).

---

## 2. Pre-flight checks

**Does the GPU run the model?** (synthetic, ~2 min) — every line must say `finite: True`, ending in
`PASS:`:

```bash
python rocm_smoke.py --preset small
# [3] autoencoder loss finite: True ...   [4] on-loop SSL (loop+memory) finite: True ...
# [5] ACT (loop) loss finite: True ...    [6] eval-mode (fused) val path finite: True ...
# PASS: ...
```

If it ends `FAIL:`, do not train — note which `[n]` failed and see Troubleshooting.

**How long will it take?** (synthetic, ~5 min):

```bash
python bench_throughput.py --preset small --batch-size 16,32,64,128 --amp --token-budget 1200000000
```

Pick the largest batch whose `peak GB` fits with ~20% headroom; note its `budget ETA`. The expensive
path is the **sequential predictor loop**, so this times a full step. Call your chosen batch `BATCH`.

---

## 3. Prepare the data (one-time, hours)

Training reads a **pre-chunked cache**. Always prep into a **fresh, empty** directory.

```bash
# tiny timed dry-run first (do NOT skip) -- confirms streaming + chunking work:
python data_prep.py --dataset HuggingFaceFW/fineweb-edu --name sample-10BT --streaming \
  --preset small --docs 1000 --out chunk_cache_dryrun
rm -rf "$PROJECT/chunk_cache_dryrun"

# the real prep (~1.2B tokens; --name and --streaming are REQUIRED for fineweb-edu):
python data_prep.py --dataset HuggingFaceFW/fineweb-edu --name sample-10BT --streaming \
  --preset small --max-tokens 1200000000 --out chunk_cache
```

`--preset small` here **must match** training. Note the final `wrote <EXAMPLES> examples` count.

---

## 4. Choose the stage budget

Five stages A→E, each a fixed number of optimizer steps via `--stage-steps A,B,C,D,E,F` (F stays 0).
One pass over the ~1.2B-token cache is `examples ÷ BATCH` steps; split A:B:C:D:E = 1:1:1:1:2:

```bash
export BATCH=64
python - <<'PY'
import json, os
m=json.load(open(os.path.join(os.path.dirname(os.getcwd()),"chunk_cache","manifest.json")))
ex=m["total"]; B=int(os.environ["BATCH"]); total=max(5,ex//B); unit=max(1,total//6)
print(f"--stage-steps {unit},{unit},{unit},{unit},{2*unit},0")
PY
```

Copy the printed value as `STAGE_STEPS`. Multiply all numbers by N for N epochs (more is better).

---

## 5. Launch

Sanity launch first (tiny budgets, confirms stages advance and numbers aren't `nan`):

```bash
python train_scaled.py --preset small --cache chunk_cache --device cuda --amp \
  --batch-size 16 --stage-steps 5,5,5,5,5,0 --log-every 1 --out runs/scaled_sanity
rm -rf "$PROJECT/runs/scaled_sanity"
```

Then the real run (background, survives disconnects):

```bash
cd "$PROJECT/files"
nohup python train_scaled.py \
  --preset small --cache chunk_cache --device cuda --amp --amp-dtype bf16 \
  --batch-size BATCH --stage-steps STAGE_STEPS \
  --num-workers 8 --log-every 50 --checkpoint-every 1000 --archive-every 5000 \
  --out runs/scaled > train.log 2>&1 &
```

Watch it: `tail -f train.log`. A healthy line:

```
[step 2000] stage=B lr=3.00e-04 logs={'nll': 6.9, 'ssl': 0.7, 'ponder': 0.0} val_loss=6.85 lstd=0.42
```

`nll` = reconstruction; `ssl` = the predictor (starts at B, decreasing = the loop learning);
`ponder` = ACT cost (starts at D).

---

## 6. Monitor

The one comparison that matters — `val_loss` across the A→B boundary (where the predictor turns on):

```bash
grep 'stage=A' train.log | tail -1 ; grep 'stage=B' train.log | tail -1 ; grep 'stage=C' train.log | tail -1
```

`val_loss` should be **flat or lower** across those (no jump at A→B), and `lstd` should hold its band.
**Collapse** = `val_loss` rises at Stage B **and** `ssl` races to ~0 **and** `lstd` craters below its
Stage-A band, all together. A low `ssl` alone is normal (the loop is just predicting well).

---

## 7. Stop / resume / crashes

```bash
pgrep -af train_scaled.py ; kill <PID>          # stop
```

Resume with the **exact same flags** plus `--resume`:

```bash
nohup python train_scaled.py --preset small --cache chunk_cache --device cuda --amp --amp-dtype bf16 \
  --batch-size BATCH --stage-steps STAGE_STEPS \
  --num-workers 8 --log-every 50 --checkpoint-every 1000 --archive-every 5000 \
  --out runs/scaled --resume "$PROJECT/runs/scaled/checkpoint.pt" >> train.log 2>&1 &
```

If the flags differ, the trainer prints a loud `WARNING: resume schedule differs` — stop and fix
unless you intended it. A clean resume prints `resumed from ... at step NNNN` with no warning.

---

## 8. When it finishes

It stops after Stage E ("final stage F" just means "finished E"; F is not trained).

```bash
python plot_metrics.py runs/scaled                                   # runs/scaled/loss_curves.png
python generate.py --ckpt "$PROJECT/runs/scaled/model.pt" --score "a sentence to score"
```

Output is real gpt2 subwords but not coherent at this scale — expected. Check the curves: `val_loss`
trends down with no rise at the Stage-B band, and `latent_std` holds/rises.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `rocm_smoke.py` prints `FAIL`/`nan` | Ensure `--amp-dtype bf16` (default). If bf16 also fails, the ROCm kernel for one op is suspect (`STRIX_HALO.md`). Don't train until it passes. |
| device resolves to cpu/mps not cuda | PyTorch isn't the GPU build, or ROCm not visible — see `STRIX_HALO.md`. |
| `data_prep` starts a huge download | Missing `--name sample-10BT` or `--streaming` — pass both; delete the partial cache. |
| `cache/model mismatch` at start | Cache prepped with a different `--preset` than training — match them. |
| `cache inconsistent: manifest says X but shards hold Y` | Re-prepped into an existing dir — delete the folder and prep fresh. |
| OOM at start | Lower `--batch-size`; keep the effective batch with `--grad-accum N`. |
| **`val_loss` rises at Stage B + `ssl`→0 + `lstd` craters** | **Latent collapse.** Lower `--ssl-weight` (from 1.0 toward 0.5) and relaunch from the last healthy `--archive-every` snapshot. Re-tuning at scale is expected. |
| `WARNING: resume schedule differs` | Make the resume flags identical to the launch (`--stage-steps`, `--batch-size`, `--lr`). |
| Everything `nan` from step 1 | Re-run `rocm_smoke.py`; try without `--amp` to isolate; report which stage it first appears in. |

---

**In doubt:** a run is healthy as long as **`val_loss` trends down with no jump when Stage B turns the
predictor on**, and `lstd` holds its Stage-A band. Watch `val_loss` above all; keep checkpoints.
