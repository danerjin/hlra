# training_easy.md — paste-to-run, start to finish

The **no-brainer** path. Each step is one paste. For the *why*, the flags, and
monitoring, see **[`TRAINING.md`](TRAINING.md)** (and [`STRIX_HALO.md`](STRIX_HALO.md)
for gfx1151 setup).

**The pipeline in two halves:**
- **Foundation (A→E)** — `data_prep.py` → `train_scaled.py`. Steps ① ②. Produces
  `runs/scaled/model.pt`. **This is the base model; it is NOT a chatbot.**
- **Chatbot (Stage-F)** — `train_dialogue.py`, a *separate* fine-tune that loads the
  foundation read-only and writes to `runs/dialogue/` (**never touches `runs/scaled`**).
  Step ③. Optional, and still **experimental** (see [`STAGE_F.md`](STAGE_F.md)).

Run ① ② first. Let A→E finish and confirm it's healthy **before** ③ — don't fine-tune a
collapsed foundation.

**Assumes** the box is set up per [`STRIX_HALO.md`](STRIX_HALO.md) §1–2: torch in
`~/hlra/.venv-rocm`, repo at `~/hlra`, corpus available. Edit the venv path at the top
of each block if yours differs.

---

## ① Preflight — verify the box (one paste, ~4 min)

Asks your model size, then runs the GPU kernel check, the full data→train path
(offline synthetic, no downloads), and the Stage-F path — cleaning up after itself.
**Fail-fasts** in a subshell, so a failure can't close your login shell.

```bash
bash <<'PREFLIGHT'
set -eo pipefail
cd ~/hlra/files
source ~/hlra/.venv-rocm/bin/activate          # <-- edit if your venv path differs
export LATENT_MANUAL_LAYERNORM=1               # gfx1151 LayerNorm-backward workaround

read -rp "Model size to smoke [small-w3] (small-w3 base-w3 large-w3 xl-w3 …): " PRESET </dev/tty || true
PRESET=${PRESET:-small-w3}
case " smoke small small-w3 base base-w3 large large-w3 xl xl-w3 " in
  *" $PRESET "*) echo "preset: $PRESET" ;;
  *) echo "unknown preset '$PRESET'"; exit 1 ;;
esac
DRY=$(mktemp -d)

echo "== [1/4] torch sees the GPU =="
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, '· cuda OK')"

echo "== [2/4] rocm_smoke — forward/backward, grads finite (THE kernel check) =="
python rocm_smoke.py --preset "$PRESET"

echo "== [3/4] data_prep + train_scaled path (offline synthetic, no downloads) =="
python data_prep.py --offline --preset "$PRESET" --docs 120 --out "$DRY/cache" >/dev/null
python train_scaled.py --preset "$PRESET" --cache "$DRY/cache" --device cuda --amp --amp-dtype bf16 \
  --batch-size 4 --stage-steps 2,2,2,2,2,0 --var-weight 3.0 --log-every 2 --out "$DRY/run" | tail -3

echo "== [4/4] Stage-F chatbot path (offline smoke) =="
python train_dialogue.py --offline --preset "$PRESET" --multi-turn --persona --trust-gate --vector-gate \
  --steps 6 --batch-size 2 --device cuda --out "$DRY/dlg" | tail -2

rm -rf "$DRY"
echo "======================================================"
echo "  PREFLIGHT PASSED ($PRESET) — the box is ready"
echo "======================================================"
PREFLIGHT
```

---

## ② Foundation run — prep → auto-started A→E training (one paste)

Asks your model size, launches data prep in the background, and queues training to
**auto-start when prep finishes** — same venv, LayerNorm workaround, all `nohup`'d
(close SSH freely). Edit the CONFIG block first.

```bash
bash <<'RUN'
cd ~/hlra/files
source ~/hlra/.venv-rocm/bin/activate

read -rp "Model size to TRAIN [small-w3] (small-w3 base-w3 large-w3 xl-w3): " PRESET </dev/tty || true
PRESET=${PRESET:-small-w3}
case " smoke small small-w3 base base-w3 large large-w3 xl xl-w3 " in
  *" $PRESET "*) : ;; *) echo "unknown preset '$PRESET'"; exit 1 ;;
esac
# batch auto-sized to the model (64GB GTT); edit if you OOM or have headroom:
case "$PRESET" in small*|smoke) BATCH=32;; base*) BATCH=16;; large*) BATCH=8;; xl*) BATCH=4;; esac

#==================== CONFIG — edit these ====================
MAX_TOKENS=1200000000          # 1.2B soft target (0.5-1B is a fine first run)
# how to read your corpus — pick ONE (arrays keep the glob intact):
PREP_ARGS=(--local-glob "/home/daniel/hlra/fineweb_local/**/*.parquet")   # local parquet (Xet escape)
# PREP_ARGS=(--dataset HuggingFaceFW/fineweb-edu --name sample-10BT --streaming)   # or stream from HF
# want it in minutes? add --regex to PREP_ARGS (fast approximate chunker, fine for a first run)
#============================================================
echo "preset=$PRESET batch=$BATCH"

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
echo "QUEUED A→E training to auto-start when prep finishes (python=$PY)."
echo "--- pipeline.log head (verify torch=...True) ---"; head -2 ~/hlra/files/pipeline.log
echo "Monitor:  tail -f ~/hlra/files/pipeline.log      (prep: tail -f ~/hlra/files/prep.log)"
RUN
```

`STATUS: … torch=…rocm… True` is your proof the queue will train on the GPU in the
right venv. **Now you can disconnect SSH.** Output → `pipeline.log`. The result is
`runs/scaled/model.pt` (the foundation).

---

## ③ Chatbot Stage-F — real fine-tune off the foundation (one paste)

Run this **only after** ② finished and A→E looked healthy (`val_loss` never jumped at
Stage B). It loads `runs/scaled/model.pt` **read-only**, fine-tunes on a real dialogue
dataset, and writes to **`runs/dialogue/`** — a **separate** directory, so your
foundation is never overwritten.

Default dataset: **`HuggingFaceH4/no_robots`** (10k high-quality instruct dialogues,
standard `messages` schema, downloads cleanly — verified). Scale up later with
`HuggingFaceH4/ultrachat_200k --hf-name default --split train_sft`, or use a
transcript corpus (debate/courtroom/socratic) via the `--hf-transcript` variant in
[`TRAINING.md`](TRAINING.md) §6.2.

```bash
bash <<'STAGEF'
cd ~/hlra/files
source ~/hlra/.venv-rocm/bin/activate
export LATENT_MANUAL_LAYERNORM=1

#==================== CONFIG — edit these ====================
FOUNDATION=runs/scaled/model.pt          # finished A→E model (loaded READ-ONLY; config inherited)
OUT=runs/dialogue                         # Stage-F checkpoints -> SEPARATE dir (never touches the foundation)
HF_CHAT=HuggingFaceH4/no_robots           # real chat dataset (messages schema)
SPLIT=train
STEPS=3000
BATCH=8
#============================================================

test -f ~/hlra/"$FOUNDATION" || { echo "no foundation at $FOUNDATION — finish ② first"; exit 1; }

# NOTE: do NOT pass --preset with --ckpt — the checkpoint's config wins.
nohup python train_dialogue.py --ckpt "$FOUNDATION" \
  --hf-chat "$HF_CHAT" --split "$SPLIT" \
  --multi-turn --soft-tags --content-tags --trust-gate --vector-gate --persona --gestalt-readout \
  --steps "$STEPS" --batch-size "$BATCH" --out "$OUT" > dialogue.log 2>&1 &
echo "STAGE-F fine-tune started (PID $!) -> tail -f ~/hlra/files/dialogue.log"
echo "   foundation (read-only): $FOUNDATION"
echo "   Stage-F checkpoint:     $OUT/model.pt   (SEPARATE from the foundation)"
STAGEF
```

Watch `nll` (should hold), `cos`/`gen` (should fall — `gen` = response quality), and
`trust=USER:…/SELF:…`. **Honest caveat:** the anti-sycophancy / trust-gate behavior is
**unproven** — a 2026-07 review found the loss doesn't reliably train the gate. Stage-F
gives you a working conversational checkpoint; treat the trust-gate as experimental
([`STAGE_F.md`](STAGE_F.md)).

---

## ④ Watch it (optional)

The one health check that matters (`val_loss` must not jump when Stage B turns the
predictor on):
```bash
grep -E 'stage=(A|B|C|D|E)' ~/hlra/files/pipeline.log | tail -5   # foundation
grep -E 'nll|gen|trust' ~/hlra/files/dialogue.log | tail -5       # Stage-F
```
Full monitoring one-liners (ETA, %, throughput): [`TRAINING.md`](TRAINING.md) §7.5.

## ⑤ When it's done

- Foundation → `runs/scaled/model.pt` · Chatbot → `runs/dialogue/model.pt` (separate).
- Get either off the box and chat with it: [`TRAINING.md`](TRAINING.md) §6.3
  (`push_to_hf.py` / `rsync`, then `dialogue_chat.py` or `web_chat.py`).

---

**More control / flag meanings / troubleshooting →
[`TRAINING.md`](TRAINING.md)** and **[`STRIX_HALO.md`](STRIX_HALO.md)**.
