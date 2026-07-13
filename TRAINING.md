# TRAINING.md — How to run the big training run

A copy-paste guide for the **A→E scaled run** (`train_scaled.py`) — the pretraining path. Do the steps
in order; don't skip the pre-flight checks. **Stage F (chatbot fine-tuning) is a separate, optional,
still-unvalidated phase that runs off the finished A→E checkpoint — see §10.**

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

Follow [`STRIX_HALO.md`](STRIX_HALO.md) first for the ROCm/GPU PyTorch install. Then install the rest
of the training deps from [`training.txt`](training.txt) (`datasets`, `transformers`, **`wtpsplit`**,
`tqdm`, `matplotlib` — `torch>=2.2` is already satisfied by the ROCm build so pip won't clobber it).
You also need: the GPU box (AMD ROCm shows up as "cuda", which is normal); the local gpt2 tokenizer at
`gpt2_tok/`; ~10-15 GB disk for the cache (it also loads fully into RAM at train start, with ~2x that
transiently while shards concatenate). Budget disk for checkpoints too: each snapshot (model + AdamW
moments + EMA) is ~2 GB at `small`, and `--archive-every 5000` keeps them all — tens of GB over a long
run.

> **SaT chunker:** `data_prep.py` segments with the real **SaT** model (`sat-3l-sm`), which `wtpsplit`
> downloads from the HF hub on the **first** prep run (a few hundred MB, cached after). This is why
> `wtpsplit` is required; the gpt2 tokenizer itself is read locally from `gpt2_tok/`.

```bash
export PROJECT=/path/to/ucsc          # edit to your path
export HSA_OVERRIDE_GFX_VERSION=11.5.1 # AMD gfx1151 only; harmless otherwise
cd "$PROJECT" && source .venv/bin/activate
pip install -r training.txt           # datasets/transformers/wtpsplit/tqdm/matplotlib (torch stays as installed)
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
# [3] autoencoder loss finite: True ... grads finite: True
# [4] on-loop SSL (loop+memory) finite: True ... grads finite: True
# [5] ACT (loop) loss finite: True ... grads finite: True
# [6] eval-mode (fused) val path finite: True ...
# PASS: ...
```

(Checks [3]-[5] verify the **gradients** too, not just the loss — the sequential loop's
*backward* kernels are the untested ROCm/bf16 path, and a backward NaN leaves the loss finite.)

If it ends `FAIL:`, do not train — note which `[n]` failed and see Troubleshooting.

**How long will it take?** (synthetic, ~5 min):

```bash
python bench_throughput.py --preset small --batch-size 16,32,64,128 --amp --token-budget 1200000000
```

Pick the largest batch whose `peak GB` fits with ~20% headroom; note its `budget ETA`. The expensive
path is the **sequential predictor loop**, so this times a full step. Call your chosen batch `BATCH`.

After the cache is prepped (§3), re-check the ETA with the cache's *actual* non-pad fraction.
Expect roughly **0.4-0.5** on prose (measured 0.38-0.47 with the post-fix chunker): chunks are
sentence-granularity, so most sit well under the 64-token cap — the old "0.6" guess was
optimistic, and more fill means fewer real tokens per step than the bench assumed. The snippet
below computes the true number from the manifest; trust it over any guess. (The bench ETA is
also mildly *pessimistic* on the other side: it charges every step at full Stage-C/E cost,
while Stage A — ~1/6 of the plan — runs the cheap codec only.)

```bash
python - <<'PY'
import json, os
m = json.load(open(os.path.join(os.path.dirname(os.getcwd()), "chunk_cache", "manifest.json")))
c = m["config"]
print("--fill-frac %.2f" % (m["tokens"] / (m["total"] * c["max_chunks_per_doc"] * c["max_chunk_len"])))
PY
```

---

## 3. Prepare the data (one-time, hours)

Training reads a **pre-chunked cache**, segmented with the real **SaT** chunker (§0 of the design
doc). Always prep into a **fresh, empty** directory (`data_prep.py` refuses a non-empty one). **Never
re-prep or touch the cache dir once the run has started** — the val/train split is derived from the
cache size, and resume hard-fails if it changed (see §7). The **first** prep run downloads the
`sat-3l-sm` SaT model (needs `wtpsplit`, see §1).

```bash
# tiny timed dry-run first (do NOT skip) -- confirms the SaT download, streaming + chunking work:
python data_prep.py --dataset HuggingFaceFW/fineweb-edu --name sample-10BT --streaming \
  --preset small --docs 1000 --out chunk_cache_dryrun
rm -rf "$PROJECT/chunk_cache_dryrun"
```

Then the real prep. **Recommended: the full Stages A-E mixture** (`--mixture` streams
`config.DataConfig.sources` — fineweb-edu + pg19 + wikipedia + arxiv + open-web-math + code,
interleaved by weight; it carries its own per-source `--name`s, and `--max-tokens` is required):

```bash
python data_prep.py --mixture --preset small --max-tokens 1200000000 --out chunk_cache
```

Or a **single corpus** (simpler; `--name` + `--streaming` are REQUIRED for fineweb-edu so you don't
pull the multi-TB default config):

```bash
python data_prep.py --dataset HuggingFaceFW/fineweb-edu --name sample-10BT --streaming \
  --preset small --max-tokens 1200000000 --out chunk_cache
```

`--preset` here **must match** training. Note the final `wrote <EXAMPLES> examples` count.

> **Wide-thought (`-w3`) run:** swap `--preset small` → `--preset small-w3` here **and** in every
> training command below, and add `--var-weight 3.0` at launch (§5). The retuned `cosine_loss_k` rides
> along in the preset automatically; `--var-weight` does not, so it must be passed. `small` and
> `small-w3` share chunk dims, so one cache serves either — but a cache's `--preset` must still match
> the model you train on it.

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
print(f"export STAGE_STEPS={unit},{unit},{unit},{unit},{2*unit},0")
PY
```

Run the printed `export` line (the later commands read `$STAGE_STEPS`). Multiply all numbers by N
for N epochs (more is better).

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
  --batch-size "$BATCH" --stage-steps "$STAGE_STEPS" \
  --num-workers 8 --log-every 50 --checkpoint-every 1000 --archive-every 5000 \
  --out runs/scaled > train.log 2>&1 &
```

Watch it: `tail -f train.log`. A healthy line:

```
[step 2000] stage=B lr=3.00e-04 logs={'nll': 6.9, 'ssl': 0.7, 'ponder': 0.0} val_loss=6.85 lstd=0.42
```

`nll` = reconstruction; `ssl` = the predictor (starts at B, decreasing = the loop learning);
`ponder` = ACT cost (starts at D).

> The `nohup ... > train.log` run writes these plain log lines (tqdm auto-detects the redirect
> and stays out of the file). Run it in a **foreground terminal** instead and you get a live
> `tqdm` progress bar with `stage / lr / val / lstd` in the postfix. Force either way with
> `--progress on|off`.

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

**Stop gracefully** — one `kill` (or `Ctrl-C` on a foreground run):

```bash
pgrep -af train_scaled.py ; kill <PID>          # stop
```

The trainer catches the signal, finishes the current step, writes `checkpoint.pt` at that
clean boundary, prints `stopped at step NNNN; checkpoint saved`, and exits. You lose **at most
one step**, not up to `--checkpoint-every`. (Send the signal a second time to force-quit
without the final save.)

**Resume** — just re-run the **same launch command** (same flags, same untouched cache). It
finds `runs/scaled/checkpoint.pt` and prints `auto-resuming from ...`; no `--resume` needed:

```bash
nohup python train_scaled.py --preset small --cache chunk_cache --device cuda --amp --amp-dtype bf16 \
  --batch-size "$BATCH" --stage-steps "$STAGE_STEPS" \
  --num-workers 8 --log-every 50 --checkpoint-every 1000 --archive-every 5000 \
  --out runs/scaled >> train.log 2>&1 &
```

- Pass `--resume "$PROJECT/runs/scaled/checkpoint_0025000.pt"` to rewind to a specific
  `--archive-every` snapshot instead of the latest checkpoint.
- Pass `--fresh` to ignore an existing checkpoint and start from step 0.

Two guards fire on resume: a hard error if the cache changed size since launch (val/train split
reshuffle → val leakage; `LATENT_ALLOW_DATA_CHANGE=1` overrides), and a printed note that the
data order re-shuffles (iterator position isn't checkpointed — the continuation is statistically
equivalent, not sample-exact). If the **flags** differ from the checkpoint, the trainer prints a
loud `WARNING: resume schedule differs` — stop and fix unless you intended it. A clean resume
prints `resumed from ... at step NNNN` with no warning.

---

## 8. When it finishes

It stops after Stage E ("final stage F" just means "finished E"; F is trained separately — §10).

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
| `WARNING: non-finite grad norm ... skipping optimizer step` | A NaN/Inf gradient appeared; the trainer skips that step so weights stay finite (one-off spikes are survivable). It hard-fails after 25 consecutive — if that happens, the run is numerically dead: check the last checkpoint, re-run `rocm_smoke.py`, consider a lower LR or no `--amp`. |
| **`val_loss` rises at Stage B + `ssl`→0 + `lstd` craters** | **Latent collapse.** Lower `--ssl-weight` (from 1.0 toward 0.5) and relaunch from the last healthy `--archive-every` snapshot. Re-tuning at scale is expected. |
| `WARNING: resume schedule differs` | Make the resume flags identical to the launch (`--stage-steps`, `--batch-size`, `--lr`). |
| Everything `nan` from step 1 | Re-run `rocm_smoke.py`; try without `--amp` to isolate; report which stage it first appears in. |

---

**In doubt:** a run is healthy as long as **`val_loss` trends down with no jump when Stage B turns the
predictor on**, and `lstd` holds its Stage-A band. Watch `val_loss` above all; keep checkpoints.

---

## 10. Stage F — chatbot fine-tuning (optional, **UNVALIDATED**)

A **separate** phase off the finished A→E `runs/scaled/model.pt`, run by its own driver
(`train_dialogue.py`, **not** `train_scaled.py`, which stops at E). The full design + mechanism map is
in [`STAGE_F.md`](STAGE_F.md). Everything here is **additive and opt-in** — with no `--`feature flags
the model is byte-identical to A→E — and **smoke-only: never trained on real dialogue data.** Treat
this as "the path is wired and reviewed," not "this produces a good chatbot."

It trains four losses on assistant (SELF) turns only: the **reconstruction anchor** (keeps the codec
from drifting), the **cosine** predictor (predict the assistant's next thought — latent-space SFT), a
**generative token NLL** (decode the *true* assistant tokens from the *predicted* latent — the piece
A→E never trains), and an **anti-sycophancy** contrastive term. Weights live in `config.StageFConfig`.

### 10.1 Sanity (offline, no downloads, ~1 min)

```bash
cd "$PROJECT/files"
python train_dialogue.py --offline --preset smoke --multi-turn --persona \
  --steps 20 --batch-size 2 --out runs/dialogue_sanity
rm -rf "$PROJECT/runs/dialogue_sanity"
```

Confirms the path runs and the loss components aren't `nan`. (With no `--ckpt` it inits a **fresh**
model — fine for a smoke; a real fine-tune **must** pass `--ckpt`.)

### 10.2 Real fine-tune off the A→E checkpoint

**Chat dataset** (a `messages`-format instruct/chat set):

```bash
python train_dialogue.py --ckpt runs/scaled/model.pt --hf-chat <HF_DATASET_ID> --hf-name <subset> \
  --multi-turn --soft-tags --content-tags --trust-gate --persona --gestalt-readout \
  --batch-size 8 --steps <N> --out runs/dialogue
```

**Transcript dataset** (debate / courtroom / socratic — long cross-turn deps + adversarial assertions;
you choose **who is SELF**: the reasoner vs. an advocate — that decides what capability you train):

```bash
python train_dialogue.py --ckpt runs/scaled/model.pt --hf-transcript <HF_DATASET_ID> \
  --text-field text --target-speaker "SOCRATES" --system-speakers "NARRATOR,MODERATOR" \
  --multi-turn --soft-tags --trust-gate --persona --gestalt-readout \
  --batch-size 8 --steps <N> --out runs/dialogue
```

Add `--rag` for latent RAG (adds the `RETRIEVED` role; a 3-role A→E checkpoint is auto-padded to 4).

| Flag | Effect |
|---|---|
| `--multi-turn` | prior turns → role+persona-tagged aged gestalts in memory (cross-turn context). Implied by `--hf-chat`/`--hf-transcript`. |
| `--soft-tags` | soft learned role tags (shared codebook + learned temperature) |
| `--content-tags` | content-condition the soft tags (the *dynamic* per-turn shift; implies `--soft-tags`) |
| `--trust-gate` / `--vector-gate` | anti-sycophancy trust gate; scalar / per-dimension (vector discounts a polarity subspace, keeps topic) |
| `--persona` | per-speaker embedding (distinguishes >3 speakers) |
| `--gestalt-readout` | homogenize self-thoughts + external content into one memory space |
| `--rag` | add the `RETRIEVED` role for source injection |

### 10.3 Watch

```
[step 500] nll=6.9 cos=3.8 gen=7.1 var=0.0 ponder=0.5 syco=6.2 trust=USER:0.71/SELF:0.95/SYSTEM:0.93
```

`nll` = reconstruction anchor (should hold, not rise); `cos`/`gen` = the SFT signals (should fall —
`gen` is the response-quality proxy); `syco` = anti-sycophancy; `trust=…` = the learned per-role trust
(watch **trust(USER) fall relative to SELF** as anti-sycophancy trains). Then chat with it:

```bash
python chat.py runs/dialogue/model.pt          # or: python web_chat.py runs/dialogue/model.pt
```

### 10.4 Honest caveats

- **UNVALIDATED** — no real dialogue run has happened; treat outputs as a plumbing check.
- **Anti-sycophancy uses *synthetic* contrastive pairs** by default (real minimal pairs are hard to
  source). Swap your own in via `dialogue_data.ContrastiveDataset` for a real Layer-3 signal.
- **RAG is mechanism-only** — the loop's read of `RETRIEVED` slots and the decode-time Talker grounding
  are untrained until you fine-tune on retrieval-augmented dialogue.
- The `--var-weight`/collapse watch-items from the A→E run still apply; the reconstruction anchor
  (`nll`) is your Stage-F collapse signal.
