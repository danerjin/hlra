# training_easy.md — paste-to-run, start to finish

The **no-brainer** path. Each step is one paste. For the *why*, the flags, and
monitoring, see **[`TRAINING.md`](TRAINING.md)** (and [`STRIX_HALO.md`](STRIX_HALO.md)
for gfx1151 setup).

**The pipeline in two halves:**
- **Foundation (A→E)** — `data_prep.py` → `train_scaled.py`. Produces
  `runs/scaled/model.pt`. **This is the base model; it is NOT a chatbot.**
- **Chatbot (Stage-F)** — `train_dialogue.py`, a *separate* fine-tune that loads the
  foundation read-only and writes to `runs/dialogue/` (**never touches `runs/scaled`**).
  Still **experimental** (see [`STAGE_F.md`](STAGE_F.md)).

Step ② runs the **whole chain in one paste**: prep → A→E → (auto) Stage-F. The chain
only advances to Stage-F if A→E exits cleanly (a crash won't fine-tune junk), but it
does **not** judge `val_loss` health for you — if you'd rather inspect the foundation
first, set `RUN_STAGE_F=0` in ②'s config and run Stage-F later with step ③.

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
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1   # gfx1151 flash attention (optional; validated by rocm_smoke)

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
python train_dialogue.py --offline --preset "$PRESET" --multi-turn --persona --trust-gate --vector-gate --trust-prior \
  --steps 6 --batch-size 2 --device cuda --out "$DRY/dlg" | tail -2

rm -rf "$DRY"
echo "======================================================"
echo "  PREFLIGHT PASSED ($PRESET) — the box is ready"
echo "======================================================"
PREFLIGHT
```

---

## ② The whole run — prep → A→E → Stage-F (one paste)

Asks your model size, launches data prep in the background, and queues the full chain
to **auto-run when prep finishes**: A→E foundation, then (if `RUN_STAGE_F=1`) the
Stage-F chatbot fine-tune — same venv, LayerNorm workaround, all `nohup`'d (close SSH
freely). Edit the CONFIG block first.

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

# ---- Stage-F chatbot fine-tune, auto-chained after A→E (set RUN_STAGE_F=0 to stop at the foundation) ----
RUN_STAGE_F=1
HF_CHAT=HuggingFaceH4/ultrachat_200k   # 100% multi-turn (what Stage F needs)
SPLIT=train_sft
STAGEF_STEPS=3000
STAGEF_BATCH=8
#============================================================
echo "preset=$PRESET batch=$BATCH  stage-F=$RUN_STAGE_F ($HF_CHAT)"

export LATENT_MANUAL_LAYERNORM=1
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1   # flash attention (optional)

# 1) launch prep (background, survives logout):
nohup python data_prep.py "${PREP_ARGS[@]}" --preset "$PRESET" --max-tokens "$MAX_TOKENS" \
  --out chunk_cache > prep.log 2>&1 &
PREP_PID=$!
echo "PREP started (PID $PREP_PID)"

# 2) write the venv-aware auto-start watcher:
cat > ~/run_pipeline.sh <<'WATCHER'
#!/bin/bash
PY="__PY__"; BATCH="__BATCH__"; PRESET="__PRESET__"
RUN_STAGE_F="__RUN_STAGE_F__"; HF_CHAT="__HF_CHAT__"; SPLIT="__SPLIT__"; STAGEF_STEPS="__STAGEF_STEPS__"; STAGEF_BATCH="__STAGEF_BATCH__"
export LATENT_MANUAL_LAYERNORM=1
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
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
log "STATUS: prep done — $EX examples; launching A→E training BATCH=$BATCH STAGE_STEPS=$STAGE_STEPS"
"$PY" train_scaled.py --preset "$PRESET" --cache chunk_cache --device cuda --amp --amp-dtype bf16 \
  --batch-size "$BATCH" --stage-steps "$STAGE_STEPS" --var-weight 3.0 --lr-schedule per-stage \
  --num-workers 2 --log-every 50 --checkpoint-every 1000 --archive-every 5000 --out runs/scaled
RC=$?
log "STATUS: train_scaled.py exited (rc=$RC) -> runs/scaled/model.pt"
# --- auto-chain Stage-F, but ONLY if A→E exited cleanly (a crash must not fine-tune junk) ---
if [ "$RC" -ne 0 ]; then
  log "STATUS: A→E FAILED (rc=$RC) — NOT starting Stage-F. See the log above."; exit "$RC"
fi
if [ "$RUN_STAGE_F" = "1" ]; then
  log "STATUS: A→E done — starting Stage-F fine-tune on $HF_CHAT (foundation READ-ONLY -> runs/dialogue)"
  "$PY" train_dialogue.py --ckpt runs/scaled/model.pt --amp --amp-dtype bf16 \
    --hf-chat "$HF_CHAT" --split "$SPLIT" \
    --multi-turn --soft-tags --content-tags --trust-gate --vector-gate --trust-prior --persona --gestalt-readout --end-weight 0.5 \
    --steps "$STAGEF_STEPS" --batch-size "$STAGEF_BATCH" --out runs/dialogue
  log "STATUS: train_dialogue.py exited (rc=$?) -> runs/dialogue/model.pt"
else
  log "STATUS: RUN_STAGE_F=0 — stopping at the foundation. Fine-tune later with step ③."
fi
WATCHER

# 3) bake the exact interpreter + config into the watcher, then queue it:
PY="$(command -v python)"                       # the venv python we activated above
sed -i "s|__PY__|$PY|; s|__BATCH__|$BATCH|; s|__PRESET__|$PRESET|; s|__RUN_STAGE_F__|$RUN_STAGE_F|; s|__HF_CHAT__|$HF_CHAT|; s|__SPLIT__|$SPLIT|; s|__STAGEF_STEPS__|$STAGEF_STEPS|; s|__STAGEF_BATCH__|$STAGEF_BATCH|" ~/run_pipeline.sh
chmod +x ~/run_pipeline.sh
nohup ~/run_pipeline.sh "$PREP_PID" > ~/hlra/files/pipeline.log 2>&1 &
sleep 2
echo "QUEUED: prep → A→E → $([ "$RUN_STAGE_F" = 1 ] && echo 'Stage-F' || echo '(stop at foundation)')  (python=$PY)."
echo "--- pipeline.log head (verify torch=...True) ---"; head -2 ~/hlra/files/pipeline.log
echo "Monitor:  tail -f ~/hlra/files/pipeline.log      (prep: tail -f ~/hlra/files/prep.log)"
RUN
```

`STATUS: … torch=…rocm… True` is your proof the queue will train on the GPU in the
right venv. **Now you can disconnect SSH.** Output → `pipeline.log`. Results:
`runs/scaled/model.pt` (foundation) and, if `RUN_STAGE_F=1`, `runs/dialogue/model.pt`
(chatbot) — separate files.

> **gfx1151 must-knows** (full detail: [`STRIX_HALO.md`](STRIX_HALO.md) §7.5):
> - **`LATENT_MANUAL_LAYERNORM=1` is REQUIRED** — the stock LayerNorm-backward kernel
>   writes NaN grads (`rocm_smoke` `[3]` FAILs without it).
>   `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` (flash attention) is **optional** — the
>   smoke validates it. `--num-workers 2` (not 8): the cache is already fully in RAM, so
>   extra workers just fork a multi-GB process for nothing.
> - **Healthy startup**: `[data] LOADING cache … 356 shards` → `[data] LOADED … in ~3s` →
>   `[trainer] training loop starting …` → `[trainer] first optimizer step done in ~3min`
>   (one-off kernel warmup) → `[step N … (heartbeat)]` every 10 steps.
> - **If the log looks frozen, check the CHECKPOINT, not the GPU:**
>   ```bash
>   ls -l --time-style=+%H:%M:%S ~/hlra/runs/scaled/checkpoint.pt   # advancing = training fine
>   ```
>   A `tqdm.write()` buffering bug used to hide ~1300 steps of output under `nohup`
>   (fixed 2026-07-16) — the run was always healthy, the log lied. GPU% and disk I/O are
>   **worthless** here; `checkpoint.pt` mtime and `py-spy` are the only honest signals.

---

## ③ Stage-F standalone — fine-tune off an existing foundation (one paste)

**You usually don't need this** — ② auto-runs Stage-F. Use this to fine-tune again
with **different data**, after setting `RUN_STAGE_F=0`, or if A→E finished earlier. It
loads `runs/scaled/model.pt` **read-only**, fine-tunes on a real dialogue dataset, and
writes to **`runs/dialogue/`** — a **separate** directory, so the foundation is never
overwritten.

Default dataset: **`HuggingFaceH4/ultrachat_200k`** (`--split train_sft`) — **measured
100% multi-turn**, which is what Stage F's cross-turn memory needs. (**Not** `no_robots`:
measured only **8%** multi-turn, so `--multi-turn`/`--persona`/`--gestalt-readout` would
train on empty context for ~92% of it.) Or use a transcript corpus
(debate/courtroom/socratic) via the `--hf-transcript` variant in
[`TRAINING.md`](TRAINING.md) §6.2. The loader accepts `messages` as a native list **or
a JSON string**, and both `role`/`content` and ShareGPT `from`/`value` schemas.

```bash
bash <<'STAGEF'
cd ~/hlra/files
source ~/hlra/.venv-rocm/bin/activate
export LATENT_MANUAL_LAYERNORM=1
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1   # flash attention (optional)

#==================== CONFIG — edit these ====================
FOUNDATION=runs/scaled/model.pt          # finished A→E model (loaded READ-ONLY; config inherited)
OUT=runs/dialogue                         # Stage-F checkpoints -> SEPARATE dir (never touches the foundation)
HF_CHAT=HuggingFaceH4/ultrachat_200k      # 100% multi-turn (what Stage F needs)
SPLIT=train_sft
STEPS=3000
BATCH=8
#============================================================

test -f ~/hlra/"$FOUNDATION" || { echo "no foundation at $FOUNDATION — finish ② first"; exit 1; }

# NOTE: do NOT pass --preset with --ckpt — the checkpoint's config wins.
nohup python train_dialogue.py --ckpt "$FOUNDATION" --amp --amp-dtype bf16 \
  --hf-chat "$HF_CHAT" --split "$SPLIT" \
  --multi-turn --soft-tags --content-tags --trust-gate --vector-gate --trust-prior --persona --gestalt-readout --end-weight 0.5 \
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
