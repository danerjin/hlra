#!/usr/bin/env bash
# probe_granularity_sweep.sh
# ==========================
# Tests the JEPA-Reasoner hypothesis: does the next-latent predictor's mean-collapse
# (LIFT<0 on the step-500 clean-experiment probe) go away at SMALLER thought
# granularity -- i.e. is it a GRANULARITY wall, not a loss problem?
#
# JEPA-Reasoner's segment latents are near-token-sized, so its next-latent target is
# low-entropy and its cosine SSL works. We widened the thought to a multi-token
# sentence chunk (higher-entropy, multimodal target) -- which is where the centroid
# wins. This sweep trains a SHORT, FRESH A+B run at several max_chunk_len values and
# probes each, so encoder AND loop are native to each granularity (a clean one-variable
# test). Read GAP (matched-shuffled) and LIFT (matched-meanbase) vs L:
#
#   * GAP grows / LIFT turns POSITIVE as L shrinks  -> granularity IS the wall.
#     Single-point cosine SSL only survives near JEPA-Reasoner's token scale; our
#     sentence-thoughts are past it. Fix = smaller granularity, token grounding
#     (--pred-token-weight), or a distributional predictor -- NOT loss tuning.
#   * GAP/LIFT flat across L -> granularity is NOT it; the SSL is broken some other
#     way (a more urgent bug than the multimodality story).
#
# Cost: each L runs data_prep + (A+B) training + probe. Defaults are deliberately
# SHORT -- the probe's GAP/LIFT are RELATIVE to each run's own encoder, so a rough
# encoder still gives a valid verdict. Tune L_LIST / STAGE_A / STAGE_B / MAX_TOKENS.
#
#   ./probe_granularity_sweep.sh                       # default sweep: L = 8 16 32 64
#   L_LIST="16 64" STAGE_B=1000 ./probe_granularity_sweep.sh
#
# Run from the repo root on the box. LATENT_MANUAL_LAYERNORM=1 is REQUIRED on gfx1151
# (broken native LayerNorm-backward kernel); harmless elsewhere.
set -euo pipefail

# Python interpreter: default `python`, but the ROCm torch lives in a venv, so under
# nohup (no shell activation) set PYTHON to the venv binary, e.g.
#   PYTHON=~/hlra/.venv/bin/python ./probe_granularity_sweep.sh
PY="${PYTHON:-python}"
PRESET="${PRESET:-small-w3}"
L_LIST="${L_LIST:-8 16 32 64}"
STAGE_A="${STAGE_A:-1500}"          # codec (encoder+Talker) warmup
STAGE_B="${STAGE_B:-1500}"          # loop + SSL (the part the probe reads)
MAX_TOKENS="${MAX_TOKENS:-20000000}"
BATCHES="${BATCHES:-8}"             # probe pooling
DEVICE="${DEVICE:-cuda}"
EXTRA_TRAIN="${EXTRA_TRAIN:-}"      # e.g. EXTRA_TRAIN="--amp --amp-dtype bfloat16"
OUTROOT="${OUTROOT:-runs/gran_sweep}"

export LATENT_MANUAL_LAYERNORM="${LATENT_MANUAL_LAYERNORM:-1}"
mkdir -p "$OUTROOT"
SUMMARY="$OUTROOT/summary.txt"
touch "$SUMMARY"   # preserve prior L rows across a resumed/partial sweep (rm it for a clean slate)

echo "[sweep] preset=$PRESET  L_LIST='$L_LIST'  stages A=$STAGE_A B=$STAGE_B  device=$DEVICE"
echo "[sweep] results -> $OUTROOT   summary -> $SUMMARY"

for L in $L_LIST; do
  CACHE="chunk_cache_L${L}"
  OUT="$OUTROOT/L${L}"
  echo ""
  echo "=================== L = $L ==================="

  echo "[sweep] (1/3) building cache $CACHE at max_chunk_len=$L ..."
  rm -rf "$CACHE"   # data_prep insists on a FRESH dir; clear any partial cache from a crashed run
  "$PY" files/data_prep.py --preset "$PRESET" --max-chunk-len "$L" \
      --max-tokens "$MAX_TOKENS" --out "$CACHE"

  echo "[sweep] (2/3) training fresh A+B at L=$L -> $OUT ..."
  "$PY" files/train_scaled.py --preset "$PRESET" --max-chunk-len "$L" \
      --cache "$CACHE" --stage-steps "${STAGE_A},${STAGE_B},0,0,0,0" \
      --device "$DEVICE" --out "$OUT" $EXTRA_TRAIN

  echo "[sweep] (3/3) probing $OUT/model.pt ..."
  PROBE_OUT="$OUT/probe.txt"
  "$PY" files/probe_predictor.py --ckpt "$OUT/model.pt" --cache "$CACHE" \
      --batches "$BATCHES" --device "$DEVICE" | tee "$PROBE_OUT"

  # Pull the two verdict numbers (probe prints "GAP ... = +0.0458" / "LIFT ... = -0.0148").
  GAP=$(grep -E "^\s*GAP"  "$PROBE_OUT" | grep -oE "[+-]?[0-9]+\.[0-9]+" | tail -1)
  LIFT=$(grep -E "^\s*LIFT" "$PROBE_OUT" | grep -oE "[+-]?[0-9]+\.[0-9]+" | tail -1)
  printf "L=%-4s  GAP=%-9s  LIFT=%-9s  (%s)\n" "$L" "${GAP:-NA}" "${LIFT:-NA}" "$OUT" | tee -a "$SUMMARY"
done

echo ""
echo "=================== SWEEP SUMMARY ==================="
cat "$SUMMARY"
echo ""
echo "[sweep] Read it: LIFT crossing 0 to POSITIVE as L shrinks == granularity is the wall"
echo "        (single-point cosine SSL only works near JEPA-Reasoner's token scale)."
echo "        Flat GAP/LIFT across L == granularity is NOT the cause; look elsewhere."
