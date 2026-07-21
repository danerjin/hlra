#!/usr/bin/env bash
# probe_depth_sweep.sh
# ====================
# The HRM-Text question, done as a TRAINING sweep: does more FIXED recurrent depth
# (h_updates_per_thought) help next-latent prediction? probe_depth_benefit.py showed
# that extending a 2-cycle-TRAINED loop deeper AT INFERENCE (ACT) does nothing -- but
# that's off-distribution. This trains fresh short runs at n_cycles = 2,3,4 and probes
# each, so each loop is TRAINED at its depth. Read MATCHED cos and LIFT vs n_cycles:
#
#   * MATCHED / LIFT rise with n_cycles -> more TRAINED recurrent depth helps (HRM-Text
#     benefit is real here); 2 cycles is under-powered. Then n_cycles is a real knob.
#   * flat across n_cycles -> 2 cycles is already enough; more depth doesn't help this
#     task, and the fixed HRM-Text structure is doing all it can. (Combined with the
#     dead ACT result: drop the adaptive-depth machinery, keep fixed shallow depth.)
#
# The cache is built ONCE (n_cycles doesn't change chunking). val_loss is NOT the metric
# (reconstruction has no loop, so it's identical across n_cycles) -- the probe is.
#
#   ./probe_depth_sweep.sh                        # n_cycles = 2 3 4
#   N_LIST="2 4" STAGE_B=1500 ./probe_depth_sweep.sh
#
# Run from repo root on the box. Set PYTHON to the ROCm venv under nohup.
set -euo pipefail

PY="${PYTHON:-python}"
PRESET="${PRESET:-small-w3}"
N_LIST="${N_LIST:-2 3 4}"
STAGE_A="${STAGE_A:-1200}"
STAGE_B="${STAGE_B:-2000}"          # loop + SSL -- the part depth affects; give it room
MAX_TOKENS="${MAX_TOKENS:-20000000}"
BATCHES="${BATCHES:-8}"
DEVICE="${DEVICE:-cuda}"
EXTRA_TRAIN="${EXTRA_TRAIN:-}"
CACHE="${CACHE:-chunk_cache_depthsweep}"
OUTROOT="${OUTROOT:-runs/depth_sweep}"

export LATENT_MANUAL_LAYERNORM="${LATENT_MANUAL_LAYERNORM:-1}"
mkdir -p "$OUTROOT"
SUMMARY="$OUTROOT/summary.txt"; touch "$SUMMARY"

echo "[depth-sweep] preset=$PRESET  N_LIST='$N_LIST'  stages A=$STAGE_A B=$STAGE_B  device=$DEVICE"

# Build the shared cache once (n_cycles does not affect chunking).
if [ ! -f "$CACHE/manifest.json" ]; then
  echo "[depth-sweep] building cache $CACHE ..."
  rm -rf "$CACHE"
  "$PY" files/data_prep.py --preset "$PRESET" --max-tokens "$MAX_TOKENS" --out "$CACHE"
else
  echo "[depth-sweep] reusing cache $CACHE"
fi

for N in $N_LIST; do
  OUT="$OUTROOT/n${N}"
  echo ""
  echo "=================== n_cycles = $N ==================="
  rm -rf "$OUT"
  "$PY" files/train_scaled.py --preset "$PRESET" --n-cycles "$N" \
      --cache "$CACHE" --stage-steps "${STAGE_A},${STAGE_B},0,0,0,0" \
      --device "$DEVICE" --out "$OUT" $EXTRA_TRAIN

  if [ ! -f "$OUT/model.pt" ]; then
    printf "n=%-3s  MATCHED=%-9s  LIFT=%-9s  (%s)\n" "$N" "TRAIN-FAIL" "TRAIN-FAIL" "$OUT" | tee -a "$SUMMARY"
    continue
  fi

  PROBE_OUT="$OUT/probe.txt"
  "$PY" files/probe_predictor.py --ckpt "$OUT/model.pt" --cache "$CACHE" \
      --batches "$BATCHES" --device "$DEVICE" | tee "$PROBE_OUT"
  MATCHED=$(grep -E "^\s*MATCHED" "$PROBE_OUT" | grep -oE "[+-]?[0-9]+\.[0-9]+" | head -1)
  LIFT=$(grep -E "^\s*LIFT" "$PROBE_OUT" | grep -oE "[+-]?[0-9]+\.[0-9]+" | tail -1)
  printf "n=%-3s  MATCHED=%-9s  LIFT=%-9s  (%s)\n" "$N" "${MATCHED:-NA}" "${LIFT:-NA}" "$OUT" | tee -a "$SUMMARY"
done

echo ""
echo "=================== DEPTH SWEEP SUMMARY ==================="
cat "$SUMMARY"
echo ""
echo "[depth-sweep] MATCHED/LIFT rising with n_cycles == more TRAINED recurrent depth helps."
echo "              Flat == 2 cycles is enough; keep fixed shallow depth, drop ACT/halt."
