# Latent-Thought Reasoning Architecture — Reference Implementation

A PyTorch reference implementation of the architecture in
[`latent-thought-architecture.md`](latent-thought-architecture.md), combining **JEPA-Reasoner**,
**HRM-Text**, **Thought Gestalt**, and **Parcae** into a model that thinks in latent "thoughts"
(chunk-level vectors), each decoded into tokens by a separate **Talker**, using a bounded recurrent
**HRM loop** as the reasoning mechanism.

It runs end-to-end (Stages A→F) both **offline** (synthetic text, no downloads) and on **real text**
(a HuggingFace corpus), checkpoints, plots training curves, and generates.

> This README is the map; [`latent-thought-architecture.md`](latent-thought-architecture.md) is the
> design. The full engineering history (every revision and dead end) is in
> [`archive/`](archive/) — read it only if you want the story of how the design got here.

## The design in one screen

Two objectives share the chunk encoder but touch the rest of the model **disjointly**:

- **Reconstruction = a pure autoencoder codec.** `encode chunk t → Talker decodes chunk t`, no loop,
  no memory. Trains the **encoder + Talker**. It's the always-on anti-collapse anchor (a constant
  latent can't reconstruct varied chunks) and is what `val_loss` / `--score` measure.
- **Prediction = the HRM loop, run sequentially with memory.** Per chunk *t*: `h_t = loop(z_t,
  memory)` (the loop reasons while reading its accumulating gestalt memory) → write `h_t` → `pred_head
  (h_t)` predicts the next chunk's EMA-target latent. Trains the **loop + encoder + pred_head +
  memory**. This is where reasoning and cross-thought memory live.

The loop is in *prediction only* on purpose: making one thought both decode the current chunk and
predict the next pulls it two ways. Removing it from reconstruction leaves a clean codec and frees the
loop to reason forward. See the design doc §2.3.

## File map

| File | Contents |
|---|---|
| `config.py` | `ModelConfig` / `TrainConfig` / `DataConfig`, size presets (`smoke`/`small`/`base`/`large`/`xl`). |
| `chunker.py` | `SegmentAnyTextChunker` ("SaT Capped": sentence boundaries + punctuation-aware length capping). |
| `ema_target.py` | `ChunkEncoder` (shared latent producer) + `EMATargetEncoder` (momentum copy; encoder-space target). |
| `decay_gate.py` | `DiagonalDecayGate` — the per-channel `exp(-softplus·dt)` carry path (Parcae's diagonal case). |
| `norm.py` | MagicNorm: Pre-LN wrapper + `hard_normalize` (the ‖h‖=√d shell that bounds the loop at any depth). |
| `hrm_loop.py` | `HRMInnerLoop`: fast L / slow H modules, decay gate, hard-norm, memory + input cross-attention, ACT halting, in-loop truncated BPTT. |
| `gestalt_memory.py` | FIFO memory of thoughts with role tags; truncated-gradient cross-attention reader. |
| `talker.py` | The Talker: causal decoder reconstructing a chunk from a latent (internal right-shift so it can't copy). |
| `input_lane.py` | Read-only bidirectional input-lane encoder (Stage F). |
| `model.py` | `LatentThoughtModel`: `forward_grounded` (autoencoder codec, the anchor), `forward_self_supervised` (on-loop sequential predictor + memory), `predict_next_latent` (generation), `latent_collapse_metric`. |
| `losses.py` | Scaled cosine (prediction), grounded NLL (reconstruction), ACT ponder cost, variance floor. |
| `curriculum.py` | Stages A→F: A autoencoder-only, B loop+SSL (memory detached), C un-detach memory, D ACT, E consolidate, F dialogue. |
| `data.py` / `data_prep.py` | Real-text pipeline (HF streams) + offline synthetic fallback; offline pre-chunking into a shard cache. |
| `trainer.py` / `train_scaled.py` | Scale-ready trainer (AMP, grad-accum, LR schedule, checkpoint/resume, per-stage budgets) + its entry point. |
| `train.py` / `train_real.py` / `run_small.py` | Smoke entry points (offline; ~1M-token real run). |
| `generate.py` | Load a checkpoint → read prompt → the loop predicts the next latent (rescaled onto the encoder-latent shell; the cosine loss trains direction, not scale) → the codec Talker decodes it. `--score` = reconstruction perplexity. |
| `baseline_gpt.py` | Standard GPT baseline for a matched-scale comparison. |
| `rocm_smoke.py` / `bench_throughput.py` / `profile_transition.py` | GPU finiteness check, throughput/ETA sweep, L-gate profiler. |
| `plot_metrics.py` / `plot_comparison.py` | Render training curves / the baseline comparison. |

## Running it

```bash
pip install torch                # offline synthetic path needs only torch
python train.py                  # offline: synthetic text + stub chunker, walks A→F

pip install torch datasets transformers matplotlib
python train_real.py             # ~1.5M-token real run (gpt2 tokenizer) -> runs/model.pt
python plot_metrics.py           # -> runs/loss_curves.png
python generate.py "The history of science shows that"
python generate.py --score "some text to score"
```

**This is smoke-scale — not gpt2 quality, and can't be at this scale.** Its purpose is to exercise the
full architecture end-to-end. For a real run, see [`TRAINING.md`](TRAINING.md) (the copy-paste guide
for the A→E scaled run) and [`STRIX_HALO.md`](STRIX_HALO.md) (the ROCm/GPU setup).

## Scaling up

```bash
# 1. Pre-chunk the corpus ONCE into a shard cache (needs datasets + local gpt2 tokenizer in ../gpt2_tok).
python data_prep.py --dataset HuggingFaceFW/fineweb-edu --name sample-10BT --streaming \
    --preset small --max-tokens 1200000000
# 2. Train from the cache (no tokenizer/SaT at train time; worker-friendly).
python train_scaled.py --preset small --cache chunk_cache --device cuda --amp \
    --batch-size 32 --stage-steps <~1 epoch, see TRAINING.md>
```

Size presets (`config.MODEL_PRESETS`): `smoke` (~43M) → `small` (512-d, ~152M) → `base`/`large`/`xl`.
`data_prep.py` and `train_scaled.py` must use the **same** preset.

## Status and honest limits

- **Verified at smoke and `small` (512-d) scale**, on offline synthetic and real gpt2 text: full A→E
  runs healthy, `val_loss` falls through the Stage-B predictor boundary, no latent collapse. **No large
  training run has been done.**
- **Collapse monitoring:** the reliable signal is a `val_loss` rise when prediction turns on at Stage
  B. `latent_std` is a secondary monitor and is **width-dependent** — do not abort on an absolute
  threshold (see `TRAINING.md`).
- **ACT adaptive depth** trains the loop's executed depth but not the halting policy (the soft ponder
  cost gives no compute-vs-quality gradient; halting degenerates to minimum depth). A real ACT
  accumulator is future work.
- **`ssl_loss_weight`** (co-equal with reconstruction) is validated collapse-free but may want tuning
  at full scale.
- **Stage F** (two-lane dialogue, anti-sycophancy loss) is designed but not yet exercised.
- The `--amp` path is implemented; sanity-check it on the first CUDA run (`rocm_smoke.py`,
  6 checks covering the training-mode and eval-mode/monitoring paths — it must end `PASS`).
- A 2026-07-10 pre-flight review (gradient-routing audit, truncation severance, A→E walk,
  resume equivalence) found the training path clean; see `notes.md` for the three
  inference/tooling fixes it landed and the items it flagged (notably: fix the input-lane
  target leak before ever training Stage F on generic documents).
- A second, comprehensive 2026-07-10 review (full spec+code pass, three independent adversarial
  audits, all invariants re-verified) landed a chunker rewrite (`_cap_span` no longer explodes
  long sentences into one-word chunks or deletes delimiters — **re-prep any cache built before
  it**), an ACT ponder/halt fix for ended documents, a resume dataset-fingerprint guard, and a
  bit-exact pad-row encoder skip (~30-45% off both encoder passes); see `notes.md` for the full
  list and the flagged-not-changed items.
- A third review (2026-07-11, four independent adversarial audits) again found the A→E training
  semantics clean and landed run-robustness + data-quality fixes: a **non-finite gradient guard**
  in the trainer (a single NaN grad no longer destroys all weights via the global clip norm — the
  step is skipped, with a hard-fail after 25 consecutive), a **splitter-fragment merge** in the
  chunker ('Dr.'/'2.'-style 1-3-token chunks: 17% of the old cache → 0.3%) and a
  character-boundary hard fallback (no more U+FFFD corruption or over-cap chunks on unicode) —
  **re-prep any cache built before it (again)** — an aligned input-lane raw window, a
  generation-path fix (no double loop pass on the last prompt chunk), gradient-finiteness gates
  in `rocm_smoke.py` [4]/[5], and a `bench_throughput.py` step that mirrors the real trainer.
  Details in `notes.md`.
