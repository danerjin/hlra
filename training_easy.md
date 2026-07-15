# training_easy.md — two pastes, start to finish

The **absolute no-brainer** path: one command to verify the box, one command to run
the whole thing (data prep → auto-started A→E training). For the *why* behind each
step, the monitoring commands, and troubleshooting, see **[`TRAINING.md`](TRAINING.md)**
(and [`STRIX_HALO.md`](STRIX_HALO.md) for gfx1151-specific setup).

**Assumes** the box is already set up per [`STRIX_HALO.md`](STRIX_HALO.md) §1–2: torch
installed in `~/hlra/.venv-rocm`, the repo at `~/hlra`, and the corpus available
(local parquet or a reachable HF dataset). If your venv/paths differ, edit the two
lines at the top of each block.

---

## ① Preflight — verify the box (one paste, ~4 min)

Runs the GPU kernel check, the full data→train path (offline synthetic, no downloads),
and the Stage-F path — cleaning up after itself. It **fail-fasts**: if anything is
wrong it stops with a non-zero step and prints nothing after it. Runs in a subshell,
so a failure can't close your login shell.

```bash
bash <<'PREFLIGHT'
set -eo pipefail
cd ~/hlra/files
source ~/hlra/.venv-rocm/bin/activate          # <-- edit if your venv path differs
export LATENT_MANUAL_LAYERNORM=1               # gfx1151 LayerNorm-backward workaround
DRY=$(mktemp -d)

echo "== [1/4] torch sees the GPU =="
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, '· cuda OK')"

echo "== [2/4] rocm_smoke — forward/backward, grads finite (THE kernel check) =="
python rocm_smoke.py --preset small-w3

echo "== [3/4] data_prep + train_scaled path (offline synthetic, no downloads) =="
python data_prep.py --offline --preset small-w3 --docs 120 --out "$DRY/cache" >/dev/null
python train_scaled.py --preset small-w3 --cache "$DRY/cache" --device cuda --amp --amp-dtype bf16 \
  --batch-size 4 --stage-steps 2,2,2,2,2,0 --var-weight 3.0 --log-every 2 --out "$DRY/run" | tail -3

echo "== [4/4] Stage-F dialogue path (offline smoke) =="
python train_dialogue.py --offline --preset small-w3 --multi-turn --persona --trust-gate --vector-gate \
  --steps 6 --batch-size 2 --device cuda --out "$DRY/dlg" | tail -2

rm -rf "$DRY"
echo "======================================================"
echo "  PREFLIGHT PASSED — the box is ready for the real run"
echo "======================================================"
PREFLIGHT
```

If it stops early: fix what it complained about (usually the venv path, a missing
`LATENT_MANUAL_LAYERNORM=1`, or GPU access) — see [`STRIX_HALO.md`](STRIX_HALO.md) §8.

---

## ② Real run — prep → auto-started training (one paste)

Edit the **CONFIG** block (four lines), then paste the whole thing. It launches data
prep in the background and queues training to **auto-start the moment prep finishes**,
in the same venv, with the LayerNorm workaround — all `nohup`'d, so you can close SSH.

```bash
bash <<'RUN'
cd ~/hlra/files
source ~/hlra/.venv-rocm/bin/activate

#==================== CONFIG — edit these ====================
PRESET=small-w3
BATCH=32                       # size to GPU memory (32 fits ~64GB GTT; 64 OOMs)
MAX_TOKENS=1200000000          # 1.2B soft target (0.5-1B is a fine first run)
# how to read your corpus — pick ONE (arrays keep the glob intact):
PREP_ARGS=(--local-glob "/home/daniel/hlra/fineweb_local/**/*.parquet")   # local parquet (Xet escape)
# PREP_ARGS=(--dataset HuggingFaceFW/fineweb-edu --name sample-10BT --streaming)   # or stream from HF
# want it in minutes? add --regex to PREP_ARGS (fast approximate chunker, fine for a first run)
#============================================================

export LATENT_MANUAL_LAYERNORM=1

# 1) launch prep (background, survives logout):
nohup python data_prep.py "${PREP_ARGS[@]}" --preset "$PRESET" --max-tokens "$MAX_TOKENS" \
  --out chunk_cache > prep.log 2>&1 &
PREP_PID=$!
echo "PREP started (PID $PREP_PID)"

# 2) write the venv-aware auto-start watcher:
cat > ~/run_pipeline.sh <<'WATCHER'
#!/bin/bash
PY="__PY__"; BATCH="__BATCH__"; PRESET="__PRESET__"
export LATENT_MANUAL_LAYERNORM=1
PREP_PID="$1"
cd ~/hlra/files
log(){ echo "[$(date '+%F %T')] $*"; }
log "STATUS: python=$PY torch=$("$PY" -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>&1)"
log "STATUS: waiting for prep (PID $PREP_PID)..."
while kill -0 "$PREP_PID" 2>/dev/null; do sleep 60; done
if [ ! -f ~/hlra/chunk_cache/manifest.json ]; then
  log "STATUS: ABORTED — prep died without manifest.json (half cache). NOT training."
  log "STATUS: salvage the partial cache with:  python make_manifest.py ~/hlra/chunk_cache $PRESET"; exit 1
fi
EX=$("$PY" -c "import json,os;print(json.load(open(os.path.expanduser('~/hlra/chunk_cache/manifest.json')))['total'])")
STAGE_STEPS=$("$PY" -c "u=max(1,($EX//$BATCH)//6);print(f'{u},{u},{u},{u},{2*u},0')")
log "STATUS: prep done — $EX examples; launching training BATCH=$BATCH STAGE_STEPS=$STAGE_STEPS"
"$PY" train_scaled.py --preset "$PRESET" --cache chunk_cache --device cuda --amp --amp-dtype bf16 \
  --batch-size "$BATCH" --stage-steps "$STAGE_STEPS" --var-weight 3.0 --lr-schedule per-stage \
  --num-workers 8 --log-every 50 --checkpoint-every 1000 --archive-every 5000 --out runs/scaled
log "STATUS: train_scaled.py exited (rc=$?)"
WATCHER

# 3) bake the exact interpreter + config into the watcher, then queue it:
PY="$(command -v python)"                       # the venv python we activated above
sed -i "s|__PY__|$PY|; s|__BATCH__|$BATCH|; s|__PRESET__|$PRESET|" ~/run_pipeline.sh
chmod +x ~/run_pipeline.sh
nohup ~/run_pipeline.sh "$PREP_PID" > ~/hlra/files/pipeline.log 2>&1 &
sleep 2
echo "QUEUED training to auto-start when prep finishes (python=$PY)."
echo "--- pipeline.log head (verify torch=...True) ---"; head -2 ~/hlra/files/pipeline.log
echo "Monitor:  tail -f ~/hlra/files/pipeline.log      (prep: tail -f ~/hlra/files/prep.log)"
RUN
```

The last lines print `STATUS: python=… torch=…rocm… True` — that's your proof the
queue will train on the GPU in the right venv. **Now you can disconnect SSH.**

---

## ③ Watch it (optional)

Prep takes hours, then training auto-starts and runs A→E over days. All output — prep
*and* training — is in the two logs above. Quick health check (the one that matters:
`val_loss` must not jump when Stage B turns the predictor on):

```bash
grep -E 'stage=(A|B|C|D|E)' ~/hlra/files/pipeline.log | tail -5
```
Full monitoring one-liners (ETA, %, throughput): [`TRAINING.md`](TRAINING.md) §7.5.

## ④ When it's done

`runs/scaled/model.pt` is your trained model. Get it off the box (HuggingFace push or
rsync) and chat with it: [`TRAINING.md`](TRAINING.md) §6.3.

---

**More control / what each flag means / troubleshooting →
[`TRAINING.md`](TRAINING.md)** (structured guide) and
**[`STRIX_HALO.md`](STRIX_HALO.md)** (gfx1151 setup + gotcha matrix).
