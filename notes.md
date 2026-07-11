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

## Second pre-flight review (2026-07-10, comprehensive: spec + full code + 3 adversarial audits)

A second full review (gradient-routing re-audit, trainer/resume/AMP audit, data-pipeline +
inference audit, all invariants re-verified empirically) before the big run. The training-loss
semantics again came through clean (reconstruction trains encoder+Talker only; SSL trains
loop+encoder+pred_head+memory reader, never the Talker; EMA grad-free; inner-loop truncation
exact in fixed and ACT modes). Fixes landed, each verified:

- **Chunker `_cap_span` rewrite (the important one — affects the data you're about to prep).**
  The old cap split at the finest delimiter present and emitted every fragment as a chunk,
  deleting the delimiters: on the real pile-10k shakedown cache 9.1% of chunks were single-token
  (17.2% ≤3 tokens), long comma sentences lost every comma, and unpunctuated sentences became
  one word per chunk with token inflation. Now pieces keep their delimiter and are greedily
  re-packed up to `max_chunk_len` (recursing to finer delimiters only for oversize pieces), so
  emitted chunks concatenate back to the original text verbatim and pack near the cap.
  **Re-prep any existing cache; do not train the big run on a pre-fix cache.**
- **ACT ponder cost + halt vote now mask ended documents** (`active_mask` threaded from
  `forward_self_supervised` into the loop). Before, rows whose doc had ended kept evolving on
  pad-chunk latents and (a) polluted the ponder gradient into the halting head/h_transition and
  (b) voted on the whole batch's depth. SSL was verified exactly invariant; ponder was not.
  All-rows-active batches are bit-identical to the old behavior.
- **Pad-row encoder skip** (`model._encode_real_rows`): the online encoder and the EMA target
  only encode chunk rows containing a real token (absent chunks get exact-zero latents).
  Verified bit-exact on every loss, grads intact; saves ~30-45% of both encoder passes at
  realistic fill.
- **Trainer/tooling:** numbered archive checkpoints no longer require `--archive-every` to be a
  multiple of `--checkpoint-every`; checkpoints fsync before the atomic rename; a **dataset
  fingerprint** (examples/tokens/shards) is saved and resume **hard-fails if the cache changed
  size** (the seeded val/train split reshuffles → val leaks into train and val_loss goes
  optimistic; `LATENT_ALLOW_DATA_CHANGE=1` overrides); resume prints that data order re-shuffles
  (iterator position is not checkpointed — statistically equivalent, not sample-exact);
  `--resume` accepts project-relative paths; empty-train-loader and wrong-length
  `--stage-steps` now fail fast; `data_prep` refuses a non-empty output dir; the dead
  `run_grounded` RNG burn was removed; `grad_window<=0` now detaches the ponder cost too
  (latent contract bug, unreachable in the shipped curriculum).
- **`generate --score` now calls `forward_grounded` directly**, so it includes the end-of-chunk
  PAD supervision and is literally val_loss-comparable (the old hand-rolled mask under-counted
  by ~0.4 nats).

Flagged, deliberately not changed:

- **Decay-gate clamp zeroes `theta`/`log_dt` gradient while a channel is clamped** (comment now
  says so honestly). Carry-path gradient w.r.t. h still flows; AdamW decay is the escape.
  Validated runs trained this way — revisit with the optimizer-grouping experiment, not now.
- **`hard_normalize` gradient spikes (~1e6) if a state ever lands within 1e-6 of the origin** —
  never observed; grad-clip bounds the blast radius; bf16 has fp32 range.
- **Memory credit chains transitively past `memory_grad_window`** (in-window slots carry their
  own reads' graphs) — this is the *documented* §3.6 design ("activation graph spans the
  document"), not a bug; utils.py's docstring now states it plainly.
- **`CachedChunkDataset` RAM:** ~2x transient at init (shard list + concat); with the post-fix
  chunker expect roughly 10-15 GB disk/resident at 1.2B tokens. Fine on the 128 GB box.
- ACT at inference uses fixed depth (2 cycles) while D/E trained adaptive — degenerate halting's
  minimum equals the fixed depth, so identical in practice.
- The `LATENT_USE_HF=1 train.py` mixture path likely needs `trust_remote_code` for pg19/
  RedPajama with modern `datasets` — not on the big-run path (data_prep uses `iter_hf_single`).

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
