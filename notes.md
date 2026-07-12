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
  The raw window covers the *end* of the kept text (since the 2026-07-11
  review it is the kept chunks' trailing ids, no longer `encode_recent` on the
  doc tail — better aligned, same leak), so with `use_input_lanes=True` the
  loop could cross-attend to chunk t+1's own tokens while predicting it.
  Irrelevant to A→E (lanes off) and to real dialogue data (the lane is the
  fixed user turn), but fix before training Stage F on generic documents.
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

## Third pre-flight review (2026-07-11, four independent adversarial audits)

A third full pass before the big run: one audit each on gradient-routing/truncation, the
trainer/LR/resume/AMP machinery, the data pipeline + chunker, and the inference/monitoring
tooling — plus an offline A→F walk and a fixed-vs-resumed `train_scaled` equivalence re-check.
**The A→E training-loss semantics came through clean for the third time** (routing matrix,
truncation cuts at windows 0-6, ended-doc isolation, PAD supervision, SSL target alignment all
re-verified empirically, down to float64 finite-difference checks on pred_head/halting-head
gradients). Fixes landed, each verified; the offline A→F walk is *numerically identical*
before/after (the training path is untouched on healthy steps):

- **Trainer non-finite gradient guard (the important one for a multi-day run).**
  `clip_grad_norm_` computes ONE global norm, so a single NaN/Inf grad element made the clip
  coefficient NaN, scaled EVERY gradient to NaN, and one `optimizer.step()` destroyed all
  weights — after which the run kept training and overwriting checkpoints with the corpse
  (bf16 has no GradScaler to filter it; the §21.2 "grad-clip bounds the blast radius" argument
  holds for finite spikes but *globalizes* non-finite ones). The trainer now checks the norm
  it already computed: non-finite → skip the step (weights bit-unchanged, verified with
  poisoned grads), warn, and hard-fail after 25 consecutive so an unattended run can't spin
  dead. Healthy steps are bit-identical.
- **Chunker: splitter-fragment merge (`min_chunk_tokens=4`).** The 2026-07-10 `_cap_span`
  rewrite killed the capper's fragment spray, but the *regex sentence splitter* still emitted
  abbreviations and list markers ('Dr.', 'on Jan.', '2.') as standalone 1-3-token chunks —
  measured 53% of chunks ≤3 tokens on numbered lists, 11% on a realistic mix: degenerate
  thoughts that burn chunk slots and pollute SSL targets across fineweb-edu. Tiny "sentences"
  are now glued to their neighbor before capping (a repair of the stub splitter's
  approximation — real SaT wouldn't produce these boundaries; sentences ≥4 tokens are
  untouched). Post-fix realistic mix: ≤1-token 0.00% (old cache 9.09%), ≤3-token 0.28%
  (17.18%), fill 0.31 → 0.38. **Prep the big-run cache with this chunker.**
- **Chunker: character-boundary hard fallback.** The no-punctuation fallback sliced the token
  *id* stream and decoded windows — not lossless under byte-level BPE: window edges split
  multi-byte characters, yielding U+FFFD corruption (emoji/CJK) and windows that re-encode
  past the cap (65 > 64, silently truncated in `chunk_batch`). Now splits on character
  boundaries (binary split, recursing until every piece fits): verbatim round-trip, exact cap,
  zero corruption on the CJK/emoji adversarial set. Whitespace-only accumulators are also no
  longer dropped at flush boundaries (fold-forward; trailing whitespace attaches only if the
  merged chunk still re-encodes within the cap).
- **Input-lane raw window now comes from the kept chunks** (`chunk_text_example`): the old
  `encode_recent`-on-truncated-text-tail was provably disjoint from the kept chunks for any
  doc past the chunk capacity (the 8-chars/token budget deliberately overshoots; measured 0%
  overlap on 8k+-char docs) — the cache baked in `raw_ids` that were noise. The window is now
  the trailing `recent_token_window` ids of the kept chunks, aligned by construction. Inert
  for A→E (lanes are Stage F), but the cache no longer violates its own contract. The Stage-F
  SSL-target-leak flag from the first review still stands: the window covers the *last* kept
  chunks, so a mid-doc prediction could still see its target through the lane — fix before
  training Stage F on generic documents.
- **Generation no longer runs the loop twice on the last prompt chunk.** `read_prompt` runs
  the loop on every prompt chunk (writing each thought), then `generate()` ran
  `predict_next_latent` on the last chunk's latent *again* — re-ingesting it from a state that
  already contained it, reading a memory holding its own thought (a configuration training
  never produces), and writing a duplicate thought carried for the whole continuation
  (~15% state shift on a non-converged model). The first continuation chunk now reads
  `pred_head` straight off the prompt's final thought (`reuse_thought=`), exactly the training
  convention; loop passes happen only on re-encoded generated chunks.
- **`rocm_smoke.py` checks [4]/[5] now verify gradient finiteness** (they gated on the loss
  only, letting a backward-kernel NaN print PASS — and the sequential loop's *backward* is
  precisely the untested ROCm/bf16 path). `bench_throughput.py` now mirrors the real trainer
  step (shared `chunk_vecs` encoder pass — it was double-running the encoder — plus
  `ema.update`), so step time and peak-GB read true.
- **Curriculum:** a `0` entry in `--stage-steps` now *skips* the stage instead of training one
  stray step under its flags; the plateau detector's state rides in the checkpoint (fixed-
  budget runs never consult it; old checkpoints load fine). Trainer checkpoints now record
  `stage_reached`; per-micro-batch `.item()` syncs only happen on log steps (three fewer
  host-device syncs per step on a launch-overhead-bound workload).
- **Docs/honesty:** TRAINING.md's launch/resume commands are now copy-pasteable
  (`export STAGE_STEPS=...`, `"$BATCH"`) — they previously passed the literal string `BATCH`
  and died in argparse inside `nohup`, visible only in train.log. Expected cache fill
  corrected to ~0.4-0.5 (the ">0.6" guess was wrong — sentence-granularity chunks sit under
  the cap; the manifest snippet computes the true number). `wiki_overfit_grounded.page_ppl`'s
  "exactly val_loss-comparable" claim was false (it omits the end-of-chunk PAD supervision:
  measured 0.20 nats / ~20% ppl below `--score` on the same text) — docstring now states the
  offset; the mask is kept matched to the GPT baselines, which have no EOS term either.
  `talker.memory_reader` is documented as untrained dead weight (every live caller passes an
  empty bank post-§27; kept for checkpoint compatibility). `generate.py` warns when input
  overflows the chunk budget (silent head-truncation) and refuses to score empty text
  (previously printed perplexity 1.0).

Flagged, deliberately not changed:

- **Ponder/cosine losses weight documents by length**: the ponder averages over *active* rows
  (a batch's longest doc's tail gets up-weighted as others end), the cosine over concatenated
  pairs (long docs contribute more pairs). Internally consistent, deterministic; noted so the
  loss composition isn't re-derived mid-run.
- **The pad-row encoder skip is bit-exact on losses; grads match to fp32 reduction-order
  noise (~1e-8)** — re-verified, claim stands with that precision caveat.
- **`data_prep` is not resumable** — a crash hours into the 1.2B prep restarts from zero
  (and requires deleting the partial dir). Known cost; the streaming prep is ~4-5 h single
  process at measured 70-93k kept-tokens/s.
- **fp16-only edges**: `torch.cuda.amp.GradScaler` is the deprecated namespace on torch ≥2.4
  (FutureWarning; only constructed under fp16), and a fully-degenerate accumulation window
  would crash `scaler.step` — both unreachable on the planned bf16 run.
- **`--lr-schedule global` (legacy) can spend a short run entirely inside warmup**
  (`warmup_steps = max(100, total//50)` can exceed a tiny `total_steps`); the default
  per-stage schedule caps warmup at `budget//10` and is immune.
- **`metrics.json` flushes only on checkpoint saves** — the plot lags the log by up to
  `--checkpoint-every` steps; `tail -f train.log` is the live view.

## Fourth pre-flight review (2026-07-11, three independent adversarial audits + full smoke suite)

A fourth full pass, run fresh against commit 261de64b: one audit each on gradient-routing/
truncation (86 float64 checks incl. finite differences and an adversarial garbage-token
active-mask probe), the chunker/data pipeline (real gpt2 tokenizer, 27 adversarial cases +
400-doc unicode fuzz), and the trainer/resume/curriculum machinery (87 checks incl. a real
SIGKILL-and-resume of `train_scaled.py`). **The A→E training semantics came through clean for
the fourth time** — every routing/truncation/masking/guard claim in this file re-verified
empirically, the chunker holds all its invariants (cap exact, verbatim, zero U+FFFD, ≤3-token
chunks 0.0% on the realistic mix), the non-finite guard leaves weights bit-identical on skipped
steps and hard-fails exactly at 25, and kill/resume reproduces the uninterrupted trajectory
(LR/stage boundaries exact; val_loss within 0.1% reshuffle noise). Post-fix integration walk:
A→E on a fresh offline cache, all stages fire, `val_loss` 8.95→8.18 with no Stage-B jump.

Three small fixes landed, none touching training semantics (the ACT edit is verified
**bit-identical** on losses and every gradient in float64, including an ended-doc row):

- **`trainer.py`: `--log-every 0` no longer crashes.** The eval branch divided by
  `log_every` unguarded (ZeroDivisionError at step 1); anyone "disabling logging" with 0
  killed the run instantly. One-line guard; `--log-every 50` behavior unchanged.
- **`hrm_loop.py`: the ACT halt-vote host sync is only paid when a break is possible.**
  `float(halt_vote)` ran every ACT step, but the break requires `step+1 >= n_cycles` — so the
  step-0 sync (one per chunk per optimizer step in Stages D/E, up to 32/step at `small`) was
  pure launch-overhead waste. The vote is only consumed by that break, so moving the read
  inside the condition is provably decision-identical.
- **`data.py`: an empty cache (0 shards) now raises a clear ValueError** instead of the opaque
  `torch.cat(): expected a non-empty list of Tensors`.

Flagged, deliberately not changed:

- **Over-cap pure-whitespace runs are silently dropped by the chunker's hard fallback**
  (`_cap_span("hello" + " "*200 + ...)` loses the spaces; threshold ~100-200 spaces). Only
  whitespace is ever lost, never text; a 64-token all-whitespace chunk would be a garbage
  thought anyway. Accepted.
- **The fp16 GradScaler path has no non-finite streak counter** — it skips bad steps silently
  forever, so an `--amp-dtype fp16` run could spin unattended with no 25-strike hard-fail.
  The big run is bf16 (guard active); do not switch to fp16 unattended.
- Two unreachable-config nits: `act_max_ponder_steps < h_updates_per_thought` would silently
  cap ACT depth below the fixed-depth minimum (shipped 6 > 2); a `chunk_mask=True` chunk with
  zero real tokens (a data-contract violation the pipeline prevents) yields a degenerate SSL
  pair against a zero target. Neither is reachable from the shipped pipeline/config.
- Cosmetic: a crash mid-save can leave a stale `checkpoint.pt.tmp` (overwritten next save);
  at step 5000 `--checkpoint-every 1000` and `--archive-every 5000` both fire (two
  back-to-back saves, seconds wasted); per-stage warmup legitimately starts below the cosine
  floor (standard warmup, not a bug — noted so nobody "fixes" it).

## Fifth pre-flight review (2026-07-11, three independent adversarial audits + post-audit-delta review)

A fifth full pass, run fresh against commit bb45050f: one audit each on model/loss/gradient
semantics (CPU probes: routing disjointness, SSL target alignment elementwise vs `EMA(z_{t+1})`,
truncation severance at windows 0/2/5/100 incl. the ACT ponder term, ended-doc ponder masking,
in-place-op safety), the trainer/curriculum/resume machinery (a real kill-at-step-8-and-resume
run reproduced the uninterrupted run's LR/stage sequence exactly and final weights
**bit-identically**; the schedule-drift and fingerprint guards fire; every TRAINING.md /
STRIX_HALO.md command parses against current argparse), and the data pipeline (real gpt2
tokenizer, 21 adversarial docs + 200-trial unicode fuzz + a shard-boundary prep→load round-trip:
cap exact, verbatim, masks contiguous, no degenerate examples, no int32 overflow at 1.2B tokens).
**The A→E training semantics came through clean for the fifth time.** The post-fourth-audit
deltas were reviewed too: the chat testers (`chat.py`/`chat_core.py`/`web_chat.py`) and
`generate.py`'s `separator` kwarg are inference-only and clean.

Changes landed — docs/process only, **zero code changes** (deliberate: five clean audits are
worth more than any pre-run "optimization"):

- **Spec corrected on Pre-LN.** `latent-thought-architecture.md` claimed MagicNorm = "Pre-LN
  internally" + hard-norm and that "Pre-LN keeps the truncated-BPTT gradient well-conditioned";
  in the current L/H cells `norm.PreNormWrapper` is dead code (never instantiated — only
  `decay_gate.py`'s comment admitted it). §0/§3 and the README file map now say the hard-norm
  half carries the stability argument, with `R`'s inputs conditioned by the loop's invariants.
- **Spec ACT stage label fixed:** §1 said "(Stage E+)"; the §5 table and `curriculum.py` turn
  ACT on at Stage **D**.
- **The stale shakedown cache is renamed** to
  `chunk_cache_shakedown.STALE-pre-0711-chunker-DO-NOT-TRAIN/`. It was built 2026-07-09 (before
  both chunker rewrites) yet its manifest dims (`64/32/256`, vocab 50258) are identical to what
  a fresh cache will say — the manifest records no chunker version, so a one-word `--cache` typo
  would have passed every guard and trained the big run on pre-fix data. Now it fails loudly.
  Nothing referenced the old path.

Flagged, deliberately not changed:

- **The manifest still has no chunker-version/prep-commit stamp** — cache freshness remains
  procedural (the rename above + TRAINING.md's fresh-prep flow). Adding a stamp to
  `data_prep.py`/`data.py` is the right post-run hardening; not worth touching the audited
  loader right before the run.
- **`ema.update()` still runs on steps skipped by the non-finite guard** — the target takes an
  extra momentum step toward *unchanged* online weights; equivalent to marginally lower
  effective momentum on skip steps, negligible.
- `_nonfinite_streak` is not checkpointed (resets on resume) — the hard-fail kills the process
  anyway, so no realistic behavior change. The fp16 silent-skip flag from the fourth review
  stands: bf16 only for unattended runs.
- **Train-time RAM at 1.2B tokens:** measured ~9.5 KB/example → ~11-14 GiB resident,
  ~2x transient at load (`torch.cat`). Fine on the 128 GB box with fork workers; do not point
  `--num-workers > 0` at this cache on a spawn platform (macOS).
- Segmentation blind spots, coherence-only: text with no ASCII spaces or `[.!?]` (newline-only
  layouts, pure CJK) falls through to the character-boundary split — tensors stay valid; trace
  exposure on fineweb-edu.
- `generate.score()` raises `SystemExit` on zero-chunk text; the chat testers can't reach it
  (both reject empty/whitespace input first) and `web_chat`'s `except Exception` wouldn't catch
  it if they could. Inference-tooling nit only.
- TRAINING.md §9's collapse remediation (lower `--ssl-weight`, relaunch from an archive
  snapshot) will *intentionally* fire the "resume schedule differs" warning — expected in that
  recovery, not a stop signal.

## Open items before a large run

- **Re-confirm at full scale (~1.2B tokens):** watch `val_loss` at the Stage-B predictor boundary; the
  512-d check used a modest budget.
- **Re-tune `ssl_loss_weight`** (currently 1.0, co-equal with reconstruction) once reconstruction has
  room to converge before B.
- **ACT halting doesn't learn** (soft ponder cost, no compute-vs-quality gradient) — needs either a
  real ACT accumulator or the simpler TRM-style supervised halt gate (see `experiments.md` #2).
- **Stage F** (two-lane dialogue, anti-sycophancy loss) is designed but not exercised.
- **`--amp`** validated only on synthetic tensors; run `rocm_smoke.py` on the GPU box first
  (now 6 checks — it must end `PASS`, incl. the eval-mode monitoring path added in the
  2026-07-10 pre-flight review and the gradient-finiteness gates on the SSL/ACT backwards
  added 2026-07-11).
- **Re-prep the cache with the post-2026-07-11 chunker** (splitter-fragment merge +
  character-boundary fallback): any cache built earlier — including one built right after the
  2026-07-10 `_cap_span` fix — has the tiny-chunk pathology and, on unicode-heavy docs,
  corrupted fallback chunks.

## Post-run experiments

See [`experiments.md`](experiments.md) — TRM-inspired ablations (arXiv:2510.04871) mapped onto this
architecture: full-thought grad window, supervised halt gate, shared L/H transition, cheap no-grad
depth. All post-run only; nothing there touches the validated A→E training semantics.

---

*Full history, including all superseded designs and the reasoning at each step, is in
[`archive/`](archive/).*
