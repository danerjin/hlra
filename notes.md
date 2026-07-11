# Engineering Notes — current state

A concise record of **what the design is and why**, plus the results that validate it. This replaces
the old blow-by-blow log; the full history — every review, bug, fix, and dead end — is preserved in
[`archive/notes.md`](archive/notes.md).

---

## The current design, in brief

Two objectives share the chunk encoder but are otherwise disjoint:

- **Reconstruction** = a pure autoencoder codec: `encode chunk t → Talker decodes chunk t` (no loop,
  no memory). Trains encoder + Talker. The always-on anti-collapse anchor; `val_loss` / `--score`.
- **Prediction** = the HRM loop run *sequentially with memory*: `h_t = loop(z_t, memory)` → write
  `h_t` → `pred_head(h_t)` predicts the next chunk's EMA-target latent. Trains loop + encoder +
  pred_head + memory. Where reasoning and cross-thought memory live.

Curriculum: **A** autoencoder-only → **B** loop + prediction (memory detached) → **C** un-detach
memory → **D** ACT → **E** consolidate → **F** dialogue. The predictor turns on at **B**.

## Key decisions and why (the non-obvious ones)

- **The loop is in prediction, not reconstruction.** One thought can't both decode the current chunk
  *and* predict the next without being pulled two ways (reconstruction trains the loop toward an
  identity pass-through, since the encoder already represents the current chunk; prediction wants a
  forward shift). Splitting them gives a clean encoder↔Talker codec and a purely predictive loop.
- **Prediction is sequential *with memory*, not parallel.** The loop reading a *populated*,
  accumulating memory is what trains the gestalt memory (Thought Gestalt's cross-thought reasoning). A
  parallel/empty-memory predictor leaves the memory readers/writers untrained.
- **Reconstruction is the anti-collapse anchor.** A constant latent can't reconstruct varied chunks,
  so keeping this always-on holds the shared encoder informative. Backed by a variance floor + a slow
  EMA target (momentum 0.996). There is *no* separate SSL projection head — an A/B showed the on-loop
  loss is more collapse-robust without one.
- **Collapse is watched via `val_loss`, not absolute `latent_std`.** The reliable signal is a
  `val_loss` rise when the predictor turns on (Stage B). `latent_std`'s healthy band is
  width-dependent (lower at larger width) — judge it against its own Stage-A band, never an absolute
  0.1.
- **Boundedness is MagicNorm's hard-norm, not the decay gate.** The decay gate shapes on-shell
  dynamics; the hard norm (‖h‖=√d at every step) is what bounds the loop at arbitrary depth.
- **The Talker right-shifts teacher forcing** (learned start vector) so it can't trivially copy the
  input — without this, reconstruction NLL goes to ~0 by copying and the latent does no work.

## Validated results

- **Full A→E runs healthy** offline (synthetic) and on real gpt2 text, at both `smoke` (~43M) and
  `small` (512-d, ~152M) presets: every stage fires, the predictor turns on at B and its loss
  decreases (the loop learning to predict), `val_loss` falls monotonically, no collapse.
- **512-d collapse check (real pile-10k, small preset):** `val_loss` fell straight through the
  Stage-B predictor boundary (9.78 → 8.84 → … → 7.66) and `latent_std` *rose* (0.14 → 0.79) — the
  opposite of collapse. This is the width where collapse risk is highest.
- **Gradient audit:** reconstruction trains encoder + Talker only; prediction trains the loop's L/H
  transitions + `memory_reader` + encoder + `pred_head` (not the Talker); the EMA target is grad-free.
- **Baseline (memorizing one Wikipedia page):** a matched-*compute* GPT (14M, width-matched) fails
  (~487 ppl); the latent model beats it; a matched-*params* GPT (44.7M) memorizes more efficiently
  (~2 ppl). Reconstruction alone floors ~38 ppl at smoke scale (the 192-d thought bottleneck). One
  data point, not the architecture's value proposition.

## Pre-flight review before the big run (2026-07-10)

A full spec + code review with fresh smoke audits: per-module gradient routing,
inner-loop truncation severance (fixed depth and ACT, incl. the ponder term),
memory grad-window checks, a full A→E walk of `train_scaled.py` on a tiny
offline cache, and a kill/resume equivalence check (the resumed run reproduces
the uninterrupted trajectory to ~1e-3; the schedule-drift warning fires on
changed flags). The A→E training path came through clean — every audit matched
the design doc. Three fixes landed, none touching the training path:

- **`predict_next_latent` rescales the predicted latent onto the encoder-latent
  norm shell** (model.py). The SSL objective is a scaled *cosine* — scale-
  invariant — so `pred_head` learns the target's direction but its output norm
  is unconstrained (measured ~0.6x the encoder-latent norm on a trained
  checkpoint), while the Talker consumes latents as *unnormalized*
  cross-attention K/V. Generation now rescales the prediction to the incoming
  latent's norm (fallback √d for a zero prompt). Inference-only.
- **`generate.py` no longer hard-exits on pre-restructure checkpoints.** Known
  legacy modules (`ssl_proj`, `latent_predictor`, `gen_predictor`) are ignored
  with a warning, so `--score` (the codec path) works on old checkpoints.
  Truly unknown state-dict keys still abort.
- **`rocm_smoke.py` gained check [6]: the eval-mode monitoring path.**
  `Trainer.evaluate()` and the `lstd` collapse metric run an eval-mode encoder
  (the fused BetterTransformer kernel, no autocast) every `log_every` steps —
  a different ROCm code path than the training-mode forwards, and the family
  that misbehaved before (§18.1). The GPU pre-flight now covers it.

Flagged, deliberately not changed:

- **Stage F would leak the SSL target through the input lane.**
  `encode_recent` takes the raw window from the document *tail*, so with
  `use_input_lanes=True` the loop could cross-attend to chunk t+1's own tokens
  while predicting it. Irrelevant to A→E (lanes off) and to real dialogue data
  (the lane is the fixed user turn), but fix before training Stage F on
  generic documents.
- `CachedChunkDataset` holds the whole cache in RAM (~10 GB at 1.2B tokens,
  ~2x transient at init). Fine on the 128 GB Linux box with fork workers;
  don't move the run to a spawn-based platform without rethinking it.
- AdamW weight-decays LayerNorm gains, embeddings, and the decay-gate
  `theta`/`log_dt` (no param groups). Left as-is: the validated smoke/512-d
  runs trained this way, and re-grouping the optimizer right before the big
  run would invalidate that validation. Optional experiment for later.

## Open items before a large run

- **Re-confirm at full scale (~1.2B tokens):** watch `val_loss` at the Stage-B predictor boundary; the
  512-d check used a modest budget.
- **Re-tune `ssl_loss_weight`** (currently 1.0, co-equal with reconstruction) once reconstruction has
  room to converge before B.
- **ACT halting doesn't learn** (soft ponder cost, no compute-vs-quality gradient) — a real ACT
  accumulator is needed to make adaptive depth a learned dial.
- **Stage F** (two-lane dialogue, anti-sycophancy loss) is designed but not exercised.
- **`--amp`** validated only on synthetic tensors; run `rocm_smoke.py` on the GPU box first
  (now 6 checks — it must end `PASS`, incl. the eval-mode monitoring path added in the
  2026-07-10 pre-flight review).

---

*Full history, including all superseded designs and the reasoning at each step, is in
[`archive/`](archive/).*
