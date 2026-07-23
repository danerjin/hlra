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
  loader right before the run. **(DONE 2026-07-13 — see below.)**
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

## Stage F implementation + 4-agent review (2026-07-13)

Stage F (chatbot fine-tuning, §4) — previously designed-only — was **implemented**,
plus the tagging/RAG/persona extensions. All **additive and opt-in**: with every
Stage-F flag off the model is **byte-identical to the validated A→E model**
(confirmed against commit 261de64b), and no A→E path (`forward_grounded` /
`forward_self_supervised` / `trainer.py`) is touched. **Smoke-only, UNVALIDATED —
no real dialogue run.** Full map in [`STAGE_F.md`](STAGE_F.md).

What landed (new files `dialogue.py`, `dialogue_data.py`, `train_dialogue.py`,
`lm_eval_adapter.py`; additions to `model.py`, `gestalt_memory.py`, `losses.py`,
`config.py`, `hrm_loop.py`, `talker.py`):
- **SFT**: `forward_dialogue` (SELF-masked cosine + a NEW generative token loss —
  decode the true assistant tokens from the *predicted* latent via `score_tokens`,
  the gap A→E never trained — + var + ponder); learned `response_seed` in a
  separate `DialogueAdapter` (keeps the base state_dict identical).
- **Three-layer input/self separation**: structural (lane never writes state),
  informational (leak-free data contract), behavioral (`anti_sycophancy_loss` +
  the trust gate, memory-routed).
- **Tagging**: soft learned role tags (shared codebook + learned temperature),
  content-conditioned tags, scalar/vector **trust gate**, **gestalt-readout**
  projection (homogenizes the bank, §Q2), per-batch role/persona ids.
- **Personalized/dynamic tags**: per-speaker `persona_embed` (conversation-local
  id), `tag_trajectory()` to observe a speaker's per-turn mixture shift.
- **Latent RAG (§Q3, mechanism only)**: `RETRIEVED` role, `inject_source`,
  `DialogueSession.add_source` + decode-time Talker grounding.
- **Real data**: transcript parser (debate/courtroom/socratic `SPEAKER:`),
  chat-messages + HF streamers, speaker→role map (pick who is SELF), multi-turn
  `tensorize_dialogue_sft`; offline synthetic corpora as fallback.
- **Eval**: `lm_eval_adapter.py` scores via the predictive chain (NOT the leaking
  reconstruction path).

**Review** — four independent adversarial audits (model/loss/gradient,
gestalt_memory reader/bank, data pipeline/driver, serving/lm-eval/config). The
model/loss and reader audits came through **clean of HIGH-severity defects**
(gradient routing, teacher-forcing with no off-by-one leak, the leak-free
contract, and A→E-byte-identical-when-off all re-verified). Fixes landed:
- **HIGH**: the two HF turn-iterators still unpacked 2-tuples after the turn
  format became `(role, persona, text)` — every real-data run would have crashed
  (the offline smoke structurally couldn't catch it). Fixed to 3-tuple unpack.
- **MED**: persona ids now clamped to `n_personas` (a transcript with more distinct
  speakers than the table no longer index-errors); `inject_source`/`add_source`
  raise a clear error on a &lt;4-role model instead of a far-away IndexError; the
  serving SELF write now goes through `_gestalt` (+SELF persona) to match training;
  the lm-eval adapter forces CPU (the shared inference path is CPU-only);
  anti-sycophancy masks to active answer rows.
- **LOW**: dependent-flag config warnings; a clear error in `_reconcile_role_tables`
  for the reverse (higher→lower role) load; transcript-regex false-positive caveat
  documented; scalar-vs-vector trust-gate limitation noted (a scalar gate can't
  discount polarity while keeping topic — use the vector gate).
Post-fix: all smokes + the lm-eval self-test pass; A→E smokes numerically unchanged.

## Stage-F input-lane SSL-target-leak fix (2026-07-13)

The leak flagged in the 1st/2nd/3rd pre-flight reviews (see the "Stage F would leak the
SSL target through the input lane" items above) is **fixed**. It lived only in
`forward_self_supervised` with `use_input_lanes=True` on *generic-document* data:
`input_lane_kv` was built once from `raw_token_ids` — which `data.chunk_text_example`
fills with the document's **trailing** tokens (the last kept chunks) — and reused at every
chunk `t`, so predicting chunk `t+1` could cross-attend to `t+1`'s own tokens through the
lane. Inert for A→E (lanes off in every A→E stage) and for real dialogue
(`forward_dialogue`'s lane is the disjoint user turn), but it would have corrupted a
Stage-F run on plain documents (the `curriculum.py` Stage F path, which the Trainer would
hit if ever advanced to F on document data).

**Fix (`model.forward_self_supervised`):** the raw-token document lane is dropped. There is
no causal single-window form — any static document window contains future chunks — and a
self-supervised document has no external "input turn" to legitimately place in the lane
(that is `forward_dialogue`'s user turn). The only causally-safe lane content is prior-turn
aged gestalts (USER/SYSTEM) already in memory, snapshotted before the loop writes any
current thought. When there are none (the A→E-shaped case: memory holds only SELF) the lane
stays `None` — **exactly equivalent to `use_input_lanes=False`**, and leak-free.

Verified (CPU, smoke preset): lanes-on document == lanes-off **bit-for-bit** (ssl+ponder
identical to the last digit); the pre-fix raw lane *did* shift the cosine prediction (~2%),
so the equality is meaningful, not vacuous; a legitimate prior USER gestalt still enters the
lane (holding only the aged gestalt, never raw tokens); the offline `train_dialogue.py`
smoke (`forward_dialogue`, untouched) still runs. The change is guarded entirely inside
`if stage.use_input_lanes:`, so the A→E path is untouched by construction.

## Cache manifest chunker-version stamp (2026-07-13)

Closes the "no chunker-version stamp" flag from the fifth review. Before this, a cache
built by an *older* chunker had a manifest byte-indistinguishable from a fresh one (same
`max_chunk_len/max_chunks_per_doc/recent_token_window/vocab_size`), so a one-word `--cache`
typo pointing at a stale cache passed every guard and would train the big run on pre-fix
data — the exact footgun the renamed `chunk_cache_shakedown.STALE-...` cache embodied.

- **`chunker.CHUNKER_VERSION`** (currently `3`) is the source-of-truth version of the
  chunk-boundary policy; bump it whenever a change makes existing caches stale. History in
  the constant's comment (1 = original, 2 = 0710 `_cap_span` rewrite, 3 = 0711 fragment
  merge + char-boundary fallback).
- **`data_prep.py`** stamps `chunker_version`, `chunker_name`, and a best-effort
  `prep_commit` (git HEAD) into every manifest.
- **`data.CachedChunkDataset`** hard-checks `chunker_version` on load: a **missing** stamp
  (a pre-guard/pre-0711 cache, unknown chunker → presumed stale) or a **mismatch** raises a
  clear `ValueError` telling you to re-prep. `LATENT_ALLOW_STALE_CHUNKER=1` overrides for a
  known-good pre-stamp cache.

The guard is universal, so old analysis caches (`wiki_cache`, offline smoke caches built
before today) will now also demand a re-prep or the override — intended fail-loud behavior.
Verified: a freshly-prepped offline cache stamps `v3` and loads; a stamp-stripped manifest
and a `v2` manifest are both refused; the override loads them. A→E training path untouched
(the check is load-time only, additive).

## Stage-F shakedown (2026-07-13, offline CPU/MPS — no GPU)

The untested Stage-F path (STAGE_F.md: "smoke-only, UNVALIDATED, real HF loaders
coded but not run") driven end-to-end in-process on the Mac. **33 cases, all
green** across four surfaces; the A-E training path (`forward_grounded` /
`forward_self_supervised` / `trainer.py` / `model.py`) was **not touched** — the
one fix is inference/driver-only.

- **A-E byte-identity re-confirmed**: with every Stage-F flag off the model's
  state_dict is byte-identical to a plain A-E model (137 tensors), and a saved
  A-E checkpoint round-trips through `load_base_model` byte-for-byte. Each opt-in
  flag only *adds* params, except `--soft-tags`, which by design *swaps* the
  discrete `role_embed` for the soft codebook (verified: the only removed keys).
- **Full flag matrix** (soft / content / trust / vector-gate / gestalt-readout /
  rag / persona, plus the full stack, single- **and** multi-turn): `forward_dialogue`
  + `forward_anti_sycophancy` + backward all produce finite losses and finite
  grads; `response_seed` receives gradient; the non-finite clip guard holds.
- **Real-data loaders exercised** (the path the 4-agent review's 3-tuple crash
  lived in) by mocking `datasets.load_dataset`: `parse_transcript`,
  `transcript_to_turns`, `messages_to_turns` all emit correct 3-tuples;
  `iter_hf_chat_turns` / `iter_hf_transcript_turns` → `DialogueTurnsDataset` →
  `collate_dialogue_sft` → `forward_dialogue` runs; `max_docs` honored, no-SELF
  docs skipped, >`n_personas` speakers clamp (no IndexError).
- **Serving + eval**: `lm_eval_adapter._score_continuation` (predictive chain, CPU)
  returns a valid log-lik; `DialogueSession.reply` multi-turn accumulates memory
  across turns; `add_source` (RAG, 4-role model) injects RETRIEVED slots and replies.

**One fix landed (`train_dialogue.load_base_model`, driver-only):** the
"checkpoint missing modules" warning collapsed parameter paths to their top-level
module (`k.split(".")[0]`), so loading a *valid* A-E checkpoint with `--soft-tags`
printed **"WARNING: checkpoint missing modules ['hrm_loop', 'talker'] (randomly
initialized)"** — false (only 6 small tag params are fresh; the whole L/H loop and
Talker load fine) and, worse, byte-identical to what a genuinely truncated
checkpoint would print. Now it separates a module *entirely* absent (real WARNING)
from one with a few opt-in Stage-F params added (informational note). Verified on
three cases: benign soft-tags load → note; talker-stripped checkpoint → WARNING;
clean flags-off load → silent. No other change; the fix is off the A-E path.

Still not covered (needs the box / network): a real HF dialogue dataset actually
streamed (only the loader *logic* is exercised, via a mocked stream), and any
GPU/bf16/ROCm Stage-F run. Stage F remains **unvalidated as training** — this
shakes out the plumbing, not the learning.

## TRM supervised halt gate — prototype (2026-07-13, offline CPU)

`experiments.md` #2 (the TRM-style supervised halt gate, the fix for "ACT halting
doesn't learn") **implemented as an opt-in alternative to the ponder cost**, off by
default. With `halt_mode="ponder"` (the default, what the big run uses) the A→E path
is **byte-for-bit identical to pre-change HEAD** — verified by running the same
fixed-seed `forward_self_supervised` in a HEAD worktree and the new tree: ssl,
second term, and grad-norm matched to the last digit in both `use_act` modes.

- **New config** (`ModelConfig`): `halt_mode` ("ponder"|"supervised"), `halt_epsilon`
  (marginal cosine-distance threshold), `supervised_halt_weight`; `__post_init__`
  rejects a bad mode and an inverted depth range (`act_max_ponder_steps <
  h_updates_per_thought`). `train_scaled.py` gains `--halt-mode` (default "ponder" =
  unchanged).
- **New code, all additive**: `losses.supervised_halt_loss` (self-calibrating BCE:
  halt when one more cycle cuts the SSL cosine distance by < ε; cos_dist is a detached
  label); `HaltingHead.logit` + `HRMInnerLoop.forward_halt_trace` (a *separate* method
  that runs to the ACT cap and returns the per-cycle H-states — `forward` is left
  byte-identical); `model.forward_self_supervised_halt` (a parallel predictor reached
  only via a guarded early-return in `forward_self_supervised`).
- **Design**: per chunk it (a) SELECTS a per-row halt depth (first cycle ≥ min-depth
  floor with prob>0.5, else cap) and drives the primary SSL prediction + memory write
  from the *selected* thought — train/test depth match, and per-row depth is the gain
  over the ponder path's batch-mean vote; (b) SUPERVISES the halt head with the BCE on
  a **detached** H-state, so the halt objective trains only the head, never reshaping
  the reasoning (the unchanged primary losses do that). Returns the same
  `(ssl, second_term)` 2-tuple, so `trainer.py`/`curriculum.py` are untouched — the
  second term flows in exactly where the ponder did.
- **Verified (smoke, CPU)**: default path bit-identical to HEAD (above); supervised
  path trains (halt BCE 0.98→0.04 on an overfit batch), the halt head receives
  gradient, and the selected depth adapts cap(6)→floor(2) as marginal improvement
  vanishes. Stage-F harnesses + config validation still green.
- **Not done / honest limits**: unvalidated at scale (a runnable A/B, not a result);
  no "think harder on hard chunks" pressure (needs an accumulator/REINFORCE); a depth
  *spread* needs harder/varied data than the smoke overfit; the trainer logs the second
  term under the `ponder` key in supervised mode too (a label only). Post-run
  experiment — does **not** land before the A→E run.

## Stage-F + halt-gate review (2026-07-14, 4 independent adversarial audits)

The first review aimed squarely at the two **additive, un-hardened** surfaces (Stage F and the
TRM halt gate) rather than the A→E path — the A→E semantics are frozen for the run. Four parallel
audits with distinct lenses (halt-gate correctness; Stage-F gradient routing / three separations;
Stage-F state-dict/tagging/checkpoint; Stage-F data pipeline + lm-eval), each verifying claims with
CPU smoke probes. **Both surfaces are structurally sound: no target leak, no garbage-training, no
A→E perturbation, and the halt gate is fully clean.**

Probe-verified clean: halt gate — all 6 claims (ponder path bit-identical incl. Stages A–C
mode-independent; BCE trains only the head, zero grad on encoder/loop/pred_head; target sign
correct; per-row depth train/test match, no off-by-one; truncation severance; trainer untouched).
Stage F — the 2026-07-13 raw-lane target-leak drop is present and correct; `score_tokens` can't copy
(NLL pinned at chance); `forward_dialogue` SELF-masked; all-flags-off byte-identity (0 extra params,
137-key state_dict); `_reconcile_role_tables` 3→4 preserves the first 3 rows bit-exactly; 8-tuple
field order matches end-to-end; SFT lane/target strings disjoint at runtime; lm-eval scores via the
predictive chain (no answer leak); structural lane isolation holds.

**Fixes landed (all off the frozen A→E path, all verified):**
- **[#1 moderate] Stage-F resume silently dropped trained state.** `train_dialogue.save()` wrote
  `adapter_state` (the response seed) + `ema` + `optimizer`, but `load_base_model` never read them
  back — a resumed run re-zeroed the seed and restarted Adam/EMA cold. `load_base_model` now returns
  a `resume` payload (only for a `stage_reached=='F'` checkpoint — an A→E checkpoint's optimizer is
  over model-only params and must NOT be loaded), and `main()` restores seed/EMA/optimizer and the
  step. Verified: response seed round-trips bit-identical, optimizer moments present, run continues
  from the saved step.
- **[#3 low] lm-eval zero-chunk continuation returned logprob 0.0 = the max score**, so any
  empty/whitespace-only candidate outranked every real answer. `_score_continuation` now returns
  `-1e30` (can't win, not greedy) when `total_tok == 0`.
- **[#5 low] `--soft-tags` on an A→E checkpoint silently discarded the trained discrete
  `role_embed`** (dropped tensors landed in an uncaptured `unexpected` list). Now warned — surfaces
  `hrm_loop.memory_reader.role_embed.weight` + the Talker's copy.

**Flagged, NOT auto-fixed (design decisions):**
- **[#2 moderate/design] The anti-sycophancy loss does not train the trust gate as wired.** Routing
  is correct (grad reaches the USER trust params) but SGD reduces the loss via the response seed /
  encoder instead (probe: `trust_proj` grad ≈0.03 vs `response_seed` ≈57), and the *scalar* gate is
  self-defeating (discounts topic+polarity together). The load-bearing Layer-3 separation is
  therefore unproven as built. Options and a recommendation (require+log the vector gate; freeze the
  escape routes; add an explicit trust objective; separate question from premise) in
  [`antisycophancy_trust_gate_note.md`](antisycophancy_trust_gate_note.md).
- Low-severity, left as documented heuristics: train/serve latent-rescale delta on `score_tokens`;
  `acc_norm` numerator/denominator tokenization mismatch; preceding turn → lane regardless of role
  (adjacent SELF turns); `parse_transcript` drops pre-first-marker text and `SPEAKER:body`
  (no-space) lines; `_reconcile_role_tables` is a shape-heuristic; `>n_personas` speakers collapse.

Smoke suites re-run on CPU this session (offline, post-fix chunker cache): the Stage-F dialogue smoke
(`--multi-turn --persona`) and the halt A/B (`train_scaled`, ponder vs supervised, explicit stage
budgets) both walk a clean A→F — Stages A–C bit-identical across halt modes, the gate diverges only
at Stage D as designed, `val_loss` falls through the Stage-B boundary with no collapse. The offline
`train.py` plateau-gated walk is a poor regression harness (no hard step cap; Stage A's autoencoder
loss keeps creeping down so the plateau gate never fires — it sat in Stage A past step 2250); use
`train_scaled` with explicit `--stage-steps` for an A→F smoke.

## Modern-architecture upgrades — opt-in, default-off byte-identical (2026-07-14)

Brought the transformer stack up to the current decoder-only conventions (RMSNorm, RoPE, QK-norm,
SwiGLU, GQA) as **opt-in flags that all default to the exact legacy path**, so the frozen A→E
semantics and every existing checkpoint are untouched unless a flag is explicitly set. New file
[`modern.py`](files/modern.py) holds the primitives (`RMSNorm`, `RoPE`, `ModernAttention` with
QK-norm + GQA over `F.scaled_dot_product_attention`, `SwiGLU`, `ModernEncoder`); the flags live on
`ModelConfig`, bundled into the `cfg.arch` property (an `ArchConfig`).

**Scope — token-level only, by design.** The `cfg.arch` flags reach only the four TOKEN-SEQUENCE
transformers: the baseline GPT, the Talker (RoPE on self-attn only — the cross-attn reads a length-2
`[thought, memory]` set with no token order), the input-lane encoder, and the chunk encoder
(`ema_target`). The loop's recurrence is deliberately excluded: RoPE is meaningless for a gated
recurrence over pooled thoughts, and the loop's bounded-state discipline is MagicNorm `hard_normalize`
(a fixed-norm shell, §3.3), not a LayerNorm to swap. When RoPE is on, a module drops its learned
`pos_embed` table. RoPE cos/sin are non-persistent buffers (out of the state_dict).

**How back-compat holds.** Flags default to their legacy values → `arch.is_legacy` → each wired module
builds the *exact* stock `nn.MultiheadAttention` / `nn.TransformerEncoderLayer` / `nn.LayerNorm` /
GELU path, so the state_dict is byte-identical. The flags round-trip through `model_cfg =
asdict(cfg)`: a modern-trained checkpoint rebuilds its modern arch on load; an old checkpoint lacks
the fields → dataclass defaults → legacy. Verified: a smoke `LatentThoughtModel` built default
strict-loads the pre-edit 137-key reference; every modern variant (full stack / rms-only / rope+gqa /
modern+w3) runs `forward_grounded` + on-loop SSL + backward finite; GQA cuts params; modern cfg
round-trips `asdict → ModelConfig → strict self-load`. Baseline A/B harness added
(`baseline_gpt.py --modern`, plus granular `--arch-*` to bisect which upgrade moves the metric).

**`core_qk_norm` — the one core upgrade (separate flag, NOT in `cfg.arch`).** QK-norm on the three
cross-attention readers (the loop's `input_reader` + `memory_reader`, and the Talker's dead-weight
`memory_reader`). Off → stock reader, core byte-identical (the pre-edit reference still strict-loads).
On → `ModernAttention(qk_norm=True, bias=True)`; the query pre-norm stays LayerNorm (the flag is
QK-norm *only*). **Safe for the core because QK-norm lives entirely inside the attention op** (per-head
RMSNorm on Q,K before the dot product) — it does not touch `_TruncationSchedule`, `hard_normalize`,
the decay gates, or the h/l state chain, so the §3.5/§3.6 credit-assignment is provably unchanged.
`ModernAttention` gained a `value=` arg for the reader's trust-gated value (`value = kv * g`, key
untouched) and a `bias` arg (`bias=True` on the core so its projections are a structural *superset* of
`nn.MultiheadAttention`).

**Importing old checkpoints into a QK-normed core — exact remap.**
`modern.remap_legacy_core_readers(state_dict, model)` transfers the readers' learned projections with
zero loss: it slices the packed MHA `in_proj_weight`/`in_proj_bias` (3E) into `q/k/v_proj`, copies
`out_proj` verbatim (same key name), and handles the two-width input reader's separate
`q/k/v_proj_weight` at `latent_mult>1`. The only params without an old counterpart — the new QK-norm
scales — init to 1. Scoped to the three core readers via their parent's `core_qk_norm` flag, so a
legacy Talker self-attn is never touched; a no-op on an already-modern checkpoint. Wired into
`generate.load(ckpt, core_qk_norm=True)` (load-time arch override via `dataclasses.replace` +
auto-remap). Verified exact + clean strict load for `latent_mult` 1 (packed) and 3 (packed + bridged).
Caveat: the *projection transfer* is exact, but QK-norm is a genuinely new op (rescales Q,K to unit
RMS), so it is a maximal warm-start, not a byte-for-byte continuation — enabling it does **not**
strict-*resume* an A→E run without this remap.

**Anti-collapse validated** (offline, real machinery: `build_offline_chunker` → reconstruction anchor
+ on-loop cosine SSL + VICReg variance floor + EMA target, synthetic varied text, smoke, 160 steps).
`latent_collapse_metric` (mean per-dim std; collapse floor 0.1) stays healthy and *grows* in every
config — final std: legacy 0.995, rms-only 1.010, full-modern 1.015, core-qk-only 0.999,
core-qk+modern 1.016; none approaches the floor. **Notable:** RMSNorm at the encoder `out_norm` (no
mean-centering) puts the latent on a *wider* shell — full-modern init std 0.964 vs legacy 0.686 — which
is collapse-favorable but means the width-dependent `cosine_loss_k` / variance-floor margin should be
re-checked at scale, same caveat as the w3 widening. This is smoke-scale evidence of *no collapse
pathology*, **not** a claim that the tuned constants are optimal through the real Stage-B boundary.

**Guidance:** highest-value flag on the Strix Halo box is `norm="rms"` — it removes every token-level
LayerNorm, sidestepping the broken gfx1151 LayerNorm-backward kernel that `LATENT_MANUAL_LAYERNORM=1`
works around. **None of these are enabled for the pending A→E run** (flags off = the validated path);
re-validate anti-collapse at width before turning any on.

## Open items before a large run

- **Re-confirm at full scale (~1.2B tokens):** watch `val_loss` at the Stage-B predictor boundary; the
  512-d check used a modest budget.
- **Re-tune `ssl_loss_weight`** (currently 1.0, co-equal with reconstruction) once reconstruction has
  room to converge before B.
- **ACT halting doesn't learn** (soft ponder cost, no compute-vs-quality gradient) — needs either a
  real ACT accumulator or the simpler TRM-style supervised halt gate (see `experiments.md` #2).
- **Stage F** (two-lane dialogue, anti-sycophancy loss) is designed but not exercised.
- **Turn-end gate (2026-07-16, Stage-F only, off by default).** Audited termination and
  found the model had no end-of-*turn* at all: PAD ends a chunk (§19.2), nothing ended a
  reply, and `DialogueSession.reply`'s only exit was dead code (`_decode_chunk` bans PAD at
  position 0, so `if not ids` never fires). Added a learned gate on `DialogueAdapter`
  (`STAGE_F.md` §2.1) — label read free off `resp_mask`, no data-format change, the
  truncation-ambiguous final label of a filled row masked out. Safe for the in-flight run by
  construction *and* by measurement: A→E `forward_grounded`/`forward_self_supervised` are
  **byte-identical** (losses + every per-param grad norm, float64), and `end_weight=0`
  reproduces pre-change Stage F byte-identically (**only after the `skip_init` fix** —
  see the review below). **No evidence it works:** the "BCE 0.455 vs 0.598 base-rate
  entropy = real signal" claim was WITHDRAWN — a null head on pure noise with random
  labels scores 0.008 and beats the base rate 20/20, so beating `H(p)` on ~14 points in
  192-d proves nothing (and 0.455 is *worse* than the null, i.e. underfit). Needs a
  held-out split. Try `--end-grad` first on real data. Document-level end-of-text is not
  done — but the "truncation-poisoned label" blocker was also wrong (real presets are 32,
  not 12; masking costs 1.88% of labels, no cache change needed); see `experiments.md` #5.

- **Turn-end review (2026-07-16, 3 independent adversarial audits) — the A→E claim held,
  three of my own claims did not.** Audits: (1) A→E safety, (2) objective correctness,
  (3) docs/claims honesty. **A→E survived** — a stronger probe than mine (3 real optimizer
  steps with dropout live across A–E + the halt path + input lanes, over 5 arch configs
  incl. `norm=rms`, full `modern`, `latent_mult=3`, `core_qk_norm`; 1471 lines) diffed
  empty, and `forward_grounded`/`forward_self_supervised`/`forward_self_supervised_halt`
  are textually identical via `inspect.getsource`. Refuted and fixed:
  - **`nn.Linear.__init__` consumes global RNG**, so merely constructing `end_head` shifted
    every later dropout mask: 130/137 base tensors differed after 3 Stage-F steps at
    `end_weight=0`. Fixed with `torch.nn.utils.skip_init` (no draw on any device, unlike
    CPU-only `get/set_rng_state`). A→E was never affected — it never builds the adapter.
  - **`end_n` was a lying metric.** The truncation mask drops a filled row's *only
    positive* and keeps all its negatives, so an all-filled batch reports `end_n=44`,
    `positives=0`, BCE→0.000, `end_acc`→1.000 — a head that learned "never end" with a
    perfect scorecard. Added `end_pos` + a 50-dry-batch warning.
  - **A serving regression I introduced and then understated:** sigmoid(−4) = 0.018 is
    *per chunk*, so an untrained gate stops 8.7% of 6-chunk replies (5 chances to stop
    early, not 6) — and `reply()`
    defaulted `use_end_head=True` without consulting `end_weight`. Now off unless the
    checkpoint's new `end_gate_trained` flag says otherwise.
  - **ACT's halt vote is a batch mean**, so train (B=batch) and serve (B=1) *could* take
    different loop depths — the audit forced a 1.36 end-logit gap straddling the threshold.
    **But that used a synthetic discriminative halting head, which training does not
    produce:** measured on the trained checkpoint, P(halt) over 64 varied thoughts spans
    [0.554, 0.674] (mean 0.619, std 0.031) — entirely above 0.5, so every row votes halt and
    batch-mean == per-row. Inert today, courtesy of the documented ACT degeneracy. **ACT
    stays ON in Stage F** (curriculum.py's Stage-F setting; D/E consolidated with it;
    adaptive depth is a central claim). A first draft recommended `--no-act` — an
    over-correction from a worst-case hypothetical, withdrawn; it is a diagnostic only. The
    real asymmetry: at serve B=1 the "batch mean" IS that row's own vote, so **serving
    already halts per-row — it is TRAINING that lets batchmates decide a row's depth**. If
    the head ever discriminates at scale, the fix is per-row halting (`experiments.md` #2),
    not disabling adaptive depth.
  - Pre-gate Stage-F checkpoints could not resume *or serve* (strict load; optimizer group
    138→140) — both sites now load non-strictly with a clear note.
  - `StageFConfig.end_threshold` was dead config (serving never sees StageFConfig) —
    removed rather than left settable-but-ignored.
- **`--amp`** validated only on synthetic tensors; run `rocm_smoke.py` on the GPU box first
  (now 6 checks — it must end `PASS`, incl. the eval-mode monitoring path added in the
  2026-07-10 pre-flight review and the gradient-finiteness gates on the SSL/ACT backwards
  added 2026-07-11).
- **Re-prep the cache with the post-2026-07-11 chunker** (splitter-fragment merge +
  character-boundary fallback): any cache built earlier — including one built right after the
  2026-07-10 `_cap_span` fix — has the tiny-chunk pathology and, on unicode-heavy docs,
  corrupted fallback chunks. **Now enforced** (2026-07-13): `data.CachedChunkDataset` refuses
  any cache not stamped `chunker_version == CHUNKER_VERSION` (currently 3), so a stale cache
  can't be trained by mistake — re-prep, or `LATENT_ALLOW_STALE_CHUNKER=1` to override.

- **Phantom memory slots (2026-07-16, pre-existing, Stage-F only, FIXED).** A
  `memory.write` writes one slot for the WHOLE batch, so `_write_context`'s batch-level
  `.any()` guard forced slots onto rows with no context there — they got
  `_encode_real_rows`' exact-zero latent tagged role 0 (**USER**), a fully attendable
  "the user said nothing" memory (`kv = stacked + tags`). It violated
  `_encode_real_rows`' own documented invariant ("pad-row latents feed only dead
  paths" — true until you write them to memory). Measured on the real corpus: **~45%
  of context memory fabricated across the batch, ~28% of rows 100% phantom** (two draws:
  45.7/28.0 and 45.4/27.5); a row's `h_t` depended on
  its batchmates' context LENGTH; it degraded `cos`/`gen`, not just the turn-end gate;
  and `--no-act` did NOT mitigate it (the ACT skew is a different, benign coupling).
  Fix: `GestaltMemoryBank.write(valid=)` + `valid_mask()` → a reader `key_padding_mask`,
  returning None when no slot carries a validity so **A→E keeps its original unmasked
  attention, byte-identical**. Applied at all three Stage-F writers, but **only
  `_write_context` was a live bug** — `inject_source` is unreachable (B=1 serving only)
  and `forward_dialogue`'s SELF write is semantically inert (`resp_mask` is left-packed,
  so a row never reactivates and its junk slots are per-row); both are kept as defensive
  applications of the pattern. NB the justification first given for the SELF write was
  **wrong**: an inactive row does NOT keep a stale `h` — `active_mask` gates only the
  ponder cost and halt vote, so those rows *keep evolving on pad-chunk latents*
  (`hrm_loop.py:320`) and write fresh garbage. Gotcha: attention over a fully-masked row
  is **NaN, not zero** — those rows are zeroed explicitly. Guarded by `files/dialogue.py` check
  [5], verified sensitive (6.6e-2 drift without the
  fix vs 4.8e-7 float32 noise with it) — but it guards **`_write_context` only**; check
  [6] guards the round-3/4 fixes, which shipped with none. Two hazards the mask cannot
  close are now **enforced** instead of relied upon: FIFO eviction is batch-coupled
  (`valid` marks a slot dead, it does not protect it from `pop(0)`, and the pop is
  driven by the *batch's* write count) — the real headroom is **2×, not the 4–8× the
  capacity ratio suggests**, since `forward_dialogue` writes context **plus** SELF; both
  512-d-class presets sit at exactly 2.0×, including **`small-w3`, which is what the A→E
  run uses**, so `train_dialogue` now refuses to start below `2 × max_chunks_per_doc`.
  And `filtered_stacked` cannot express validity (its return has no per-row mask), so it
  now raises rather than handing a masked slot back for every row.

- **Serving mis-tagged the aged USER turn with SELF's persona (2026-07-16, FIXED).**
  `_age_user_turn` passed no `persona_id`; `persona_id_tensor` maps None→0, and 0 is
  SELF's persona — so memory asserted the user's own turn was spoken by the model.
  Training tags it 1 (`dialogue_data._ROLE_MAP`). Inert while `persona_embed` is
  zero-init, live under `--persona`, which is in TRAINING.md's recommended command.
  Found independently by two auditors.

- **Stage F could not run on real data at all (2026-07-16, found by driving the
  DOCUMENTED commands end-to-end; both FIXED).** Three prior audit rounds missed both
  because the preflight only runs `train_dialogue.py --offline` with **no `--ckpt`** —
  it exercises neither the checkpoint handoff nor the HF loader.
  - **The A→E→F handoff crashed on every completed run.** `train_dialogue` used
    `stage_reached == "F"` to mean "my own Stage-F checkpoint" — but `F` is ALSO the
    A→E curriculum's terminal stage (`curriculum.Stage.F`), and `trainer.py` saves
    `curriculum.stage.name`. So a finished A→E `model.pt` is stamped `"F"`, was misread
    as a resume, and loaded the A→E optimizer (model params only) into the Stage-F
    optimizer (model + adapter) → `ValueError: param group ... doesn't match`. The
    guard's own comment stated the invariant it broke. Now keys on `adapter_state`,
    which only `save()` writes. `runs/model.pt` (stage_reached='F', no adapter) was the
    counterexample sitting in the repo the whole time. `dialogue_chat.py` had the same
    confusion (an A→E foundation passed its "is a chatbot" check).
  - **`TRANSFORMERS_OFFLINE=1` at import made the real corpus unreachable.**
    `huggingface_hub` honours it as a legacy alias for `HF_HUB_OFFLINE` and `datasets`
    inherits it, so every `--hf-chat` run died "Offline mode is enabled" — the data this
    driver exists to train on, unreachable by construction. Copied from
    `generate.py`/`train_real.py`, which only load the LOCAL `gpt2_tok` and so were
    unaffected; `data_prep.py` omits it deliberately and says why. Removed.

## Eval-tooling dry-run (2026-07-16)

Exercised the whole post-run eval path CPU-only against `runs/model.pt`, before the A→E
run lands, so nothing is discovered on results day. `generate.py --score` / generate,
`chat_core` (both testers), and `plot_metrics.py` are green.

- **`lm_eval` is NOT installed** — the ARC-C path needs `pip install lm_eval`. Nothing
  warns you; the adapter's core is deliberately dependency-free, so only the harness
  wrapper needs it.
- **The `lm_eval_adapter` self-test had been failing since `999b6d3b`.** That review
  correctly changed a zero-chunk continuation's score from `0.0` to a large-negative
  sentinel (0.0 is the *maximum* log-likelihood and would win every multiple-choice
  ranking) but left the self-test asserting the old `0.0` contract. Repaired the test,
  not the scorer, and it now asserts the *property* — a zero-chunk continuation must rank
  below every real one and cannot be `is_greedy` — so it will not rot again if the
  sentinel changes.
- **ARC-C is this adapter's documented worst case.** Its own module docstring: multiple
  choice whose options differ by a token is "the least reliable use of this adapter";
  cloze/continuation tasks (LAMBADA, HellaSwag) "sit much better on the chunking". The
  model has no token-level conditional logprob, so scoring granularity is the chunk and a
  short option is a single `pred_head`→Talker decode. At `small` scale ARC-C will also sit
  near chance. If a headline benchmark is wanted, LAMBADA/HellaSwag is the honest choice.

## Predictor mean-collapse investigation (2026-07-20/21) — IMPORTANT

The first big A→E run reached a near-lossless codec (`val_loss` ~0.0067) but **generated mush.**
Root cause and the reframe that followed — the load-bearing findings:

**1. The failure is the PREDICTOR, not the codec.** Good codec (round-trips), but the on-loop
predictor (`pred_head(h_t) → next latent`) mean-collapsed: on the foundation, `probe_predictor`
showed PRED-SELF **0.98** (predictions ~constant) while HSTATE was 0.88 (the loop's states were
still diverse). So the loop reasons, but `pred_head` crushes its diverse states into a near-constant
vector — which decodes to generic tokens → mush. `pred_collapse` / `hstate_collapse` were added to the
train log + `metrics.json` precisely because `ssl` and `latent_std` **cannot see this** (a collapsed
predictor and a good one score the same `ssl`; `latent_std` only watches the encoder).

**2. The "mean-collapse is fatal / a scale wall" verdict was largely a FROZEN-ENCODER ARTIFACT.**
The clean experiment (frozen foundation encoder + fresh loop+head + contrastive) probed at LIFT
**−0.015** (predictor *worse* than emitting the target centroid) — read at the time as "the loop can't
predict, H1 scale wall." But a **granularity sweep** (fresh A+B per `L`, encoder CO-TRAINED with the
loop, plain cosine) told a different story:

| max_chunk_len | GAP (matched−shuffled) | LIFT (matched−meanbase) |
|---|---|---|
| 8  | +0.080 | +0.041 |
| 16 | +0.109 | +0.054 |
| 32 | +0.116 | +0.055 |
| 64 | +0.152 | **+0.078** |

LIFT is **positive at every granularity and rises as thoughts get coarser** — so (a) **granularity is
NOT the wall** (the sentence-sized-thought worry is unfounded; L=64 predicts best), and (b) a
co-trained predictor is **informative** (beats the centroid). The I-JEPA "condition the predictor on
the target position" idea was considered and **retracted** — it doesn't map: we always predict t+1
(one target, no position to disambiguate), and our entropy is *generative* (many valid futures), not
*positional* like a masked image patch.

**3. What flipped the sign was UNFREEZING THE ENCODER, not the loss or granularity.** The tell: the
clean experiment used **contrastive** (the *better* anti-collapse loss) + frozen → collapsed; the sweep
used plain **cosine** (the mean-seeker) + co-trained → informative. The loss points the *wrong* way, so
encoder co-training dominates it. Caveat: co-training lets the encoder shape its own targets to be
predictable (a possible shortcut), so "informative" still needs the downstream meaning check.

**4. The remaining bottleneck is the TRAIN/SERVE GAP.** Even informative, MATCHED cos is only ~0.50 —
the predicted latent is half-aligned to the true next. The Talker is trained on REAL latents
(reconstruction, cos~1.0) but at generation decodes PREDICTED (~0.5-cos) latents it never practiced on.
Measured directly by the new **token-grounding** loss (`--pred-token-weight`, `ssl_token_weight`):
decode the PREDICTED latent through the Talker vs the real next tokens (`model.score_tokens`, rescaled
onto the encoder norm shell). On a short run the `tok_nll − nll` gap sat at **~1.5 nats** and did not
close — likely near an information floor (a 0.5-cos latent genuinely carries less about the next tokens).
NOTE: `tok_nll` is **teacher-forced**, so it closes the *latent* exposure gap but is blind to the
*token* one (autoregressive decode from the Talker's own samples) — `web_chat` free-running generation
stays the final arbiter.

**5. THE OPEN QUESTION — collapse is a LATE-TRAINING ATTRACTOR.** The foundation (long, co-trained,
good codec) collapsed to PRED-SELF 0.98; the short co-trained runs (sweep, token-grounding test) sit at
~0.76–0.87 and do NOT collapse. The only systematic difference is **training length + data**. So the
short runs likely just haven't fallen into the attractor yet, and a naive big run would reproduce the
mush. Crucially, the *only* training-level difference between the proposed big run and the foundation
is the added losses — so those losses must be a **real, well-placed** difference, not a sprinkle.

**The anti-collapse regime (what the next big run actually changes):**
- **Token-grounding from Stage B** (moved from D+; `curriculum.loss_plan`). Its gradient into
  `pred_head` is centroid-PROOF (a constant prediction decodes to generic tokens → high token NLL), so
  it must run *during* the whole collapse-prone window as a PREVENTATIVE, not arrive in D as a fix. A
  D+ gate is too late if the attractor forms in B/C.
  **[SUPERSEDED 2026-07-23 — see the anisotropy section below. The centroid-proof claim was false as
  wired: the gradient also reached the encoder and loop, which can lower `tok_nll` by making latents
  easy to decode rather than predictions correct. `tok_nll` is now detached to the Talker only and is
  NOT an anti-collapse term.]**
- **Hard-negative InfoNCE from Stage B** (`--pred-contrastive-weight --pred-contrastive-hard`). The
  *most direct* anti-collapse pressure: a centroid predictor gives a uniform softmax = the worst
  InfoNCE loss. Hard negatives (same-document chunks) force the fine next-chunk distinction rather than
  cross-doc topic separation. Kept alongside the cosine anchor so it can't drift to an undecodable
  scale. (Was at chance in the frozen clean experiment — plausibly because frozen refined targets
  weren't rankable; co-training should give it traction. Unproven.)
- Both hit the confirmed collapse point (`pred_head`) via complementary mechanisms (decodability vs
  ranking margin). All default-off; the A→E semantics are byte-identical without the flags.

**Decision criterion for the big run** (instrument `pred_collapse` over time, probe intermediate
checkpoints — do NOT wait for the end): `pred_collapse` stays down through long training → the attractor
is broken (then ablate which lever mattered). `pred_collapse` climbs back toward 0.98 with BOTH losses
on → it is **not** a loss problem (architectural / scale) and we stop tuning losses.

Tooling added this round: autoencoder round-trip in the chat interfaces (`chat.py` `:auto`/`:latent`,
`web_chat.py` Codec mode) to separate codec failures from predictor failures; `--max-chunk-len`
override on `data_prep`/`train_scaled` + `probe_granularity_sweep.sh` for the sweep.

## Latent anisotropy is upstream of predictor collapse (2026-07-22/23)

The anti-collapse regime above treats collapse as a **loss** problem. The measurements below
locate it one level earlier, in the **geometry of the latent space itself**, and refute two of
the claims made there. Everything here is measured on `small-w3` / `chunk_cache` unless noted.

**1. The cone. `probe_latent_semantics` / `probe_predictability` measure random-pair cosine** — how
aligned two *unrelated* chunks are. A healthy space is near 0; the reconstruction-only codec measured
**0.495–0.515**, i.e. every latent crammed into a narrow cone. This is the mechanism under collapse:
in a narrow cone the centroid is already close to every target, so a cosine predictor that ignores its
input and emits the mean is *nearly optimal*. The narrower the cone, the better that cheat pays.
`latent_std` cannot see this (it is a per-dimension variance floor, not an angular spread).

**1b. THE AUTOENCODER IS WHAT CLOSES THE CONE — the two objectives are in direct tension.**
Reconstruction needs each latent to be *decodable*, and a tight huddle is perfectly decodable; nothing
in `L_rec` rewards angular separation. So the anchor that prevents *encoder* collapse is the same force
that drives the anisotropy causing *predictor* collapse. It also wins by default on scale: `nll` is
unbounded (~0.4) while `distill` is bounded (1−cos ≤ 1), which is why weights of 1 and 3 were never
competitive and weight **10** was required. Observed directly: with the distill hinge dormant, the cone
re-closed 0.110 → 0.220 over ~7k steps with no other change.

**1c. The codec is BRITTLE — an exact-latent decoder.** `probe_latent_use`: NLL under the true latent
**0.0042**, under a shuffled latent **41.4**. Good news (the Talker genuinely uses the latent; it is
not memorizing the token distribution), but it decodes only *near-exact* latents. At generation it
receives a **predicted** latent at ~0.5 cosine to the truth, which it has never practiced on — the
train/serve exposure gap, and the direct explanation for a near-lossless codec (`val_loss` 0.0067)
producing mush. This is what the (now detached) token-grounding term is for, and why `--recon-weight`
exists: reconstruction sharpness and decoder tolerance genuinely trade off.

**2. SBERT distillation opens the cone; it is the load-bearing anti-collapse lever.** Pointwise
distillation (`--sbert-distill-weight`, `losses.sbert_distill_loss`) pulls a *projection* of the latent
toward a frozen MiniLM embedding. Crucially the constraint sits on the **projection**, not the latent,
so the latent stays free to be *more* isotropic than the teacher:

| run | random cos | adjacent cos | lift | vs SBERT |
|---|---|---|---|---|
| `distill_test` (w=10, no floor, ~fresh encoder) | **0.1103** | 0.1898 | **+0.0795** | 1.46× |
| `distill_full` (w=5, floor 0.8) | 0.2203 | 0.2832 | +0.0629 | 1.15× |
| after rewind (w=10, no floor, 2k steps) | 0.1999 | 0.2694 | +0.0695 | 1.24× |
| SBERT (MiniLM) itself | 0.2082 | 0.2615 | +0.0561 | — |

`distill_test` is the only configuration that reached an open cone, and it is also the only run whose
predictor escaped collapse (`pred_collapse` 0.9986 → **0.7967** within ~200 steps of Stage B) — with
**no token-grounding and no contrastive term at all**. That co-occurrence is the central evidence that
geometry, not the predictor-side losses, is what breaks collapse.

**3. Over-imitation is real: distillation has an optimum, not a maximum.** Driving distill cosine to
convergence converges our geometry *onto* MiniLM's and gives back the task-specific advantage:

| distill cos to SBERT | lift | vs SBERT |
|---|---|---|
| 0.81 (partial) | +0.0713 | 1.30× |
| 0.994 (converged) | +0.0619 | 1.15× |

So the teacher must act as a **prior, not a target**. That is what `--sbert-distill-floor` is for
(hinge `clamp(floor − cos, 0)`, dormant once cos ≥ floor).

**4. The floor failed because of its VALUE, not its mechanism — and the failure is instructive.**
A floor is dormant whenever cos ≥ floor. Setting `--sbert-distill-floor 0.8` while cos was already
**0.98** made it inert from the first step: `distill` contributed ~0.016 against `nll` ~0.4 (~25×
weaker), reconstruction won unopposed, and the cone re-closed **0.110 → 0.220** over ~7k steps
(observed drifting: 0.2064 → 0.2108 → 0.2183 → 0.2203). A floor only acts if it is set *above* current
cos. **Rule: pick the floor from a measured cos, and verify `distill > 0` after applying it.**

**5. Token-grounding's gradient path was wrong (fixed, `b4d61d51`).** `tok_nll` decoded `pred_head`'s
forecast through the Talker with gradient flowing into the Talker **and** the loop **and** the encoder.
That breaks the centroid-proof property claimed above: the encoder/loop can lower `tok_nll` by making
latents **low-information and easy to decode** instead of making predictions correct — the same leak as
the "encoder degradation" risk. Observed directly: `tok_nll` rose monotonically (4.66 → 4.79 → 4.90)
while the predictor spread out. Fix: detach the predicted latent before `score_tokens`, so the gradient
reaches **only the Talker**. After the detach `tok_nll` reversed (4.90 → 4.78 → 4.69) and `val_loss`
improved. **`tok_nll` is now a decoder-robustness term (the train/serve exposure fix) and NOT a
collapse diagnostic** — a detached Talker will happily learn to decode a collapsed predictor's output.
Collapse is read from `pred_collapse` / `hstate_collapse` and the probe's LIFT, full stop.

**6. The cone cannot be reopened once the encoder hardens.** `distill_test` reached 0.110 from a
*fresh* encoder. Retrofitting the same setting (w=10, no floor) onto a checkpoint with only **5k steps**
of reconstruction behind it, for 2k steps, moved the cone just **0.2203 → 0.1999 (~19% of the way)**
while lift recovered ~40%. The anisotropy is established early and is far cheaper to prevent than to
undo. **Implication: distillation must be on from step 0, not added or restored later.**

**7. The escape is learning-rate-gated.** `pred_collapse` is flat during Stage-B warmup and moves only
as LR approaches its ~3e-04 peak — in both the escaping run and ours. Two earlier "it has stalled"
readings were taken at 3.7e-05 during warmup and were meaningless. With `--lr-schedule per-stage` the
warmup scales with stage length, so a long Stage B delays the escape window by ~750 steps. Judge
collapse **only at peak LR**, and over a ≥4-reading window: single-point noise on `pred_collapse` is
about ±0.02, so nothing under ~0.05 of movement is interpretable.

**8. Where it stands (PENDING).** The rewound run (fresh loop into a half-open cone, 0.1999) sat at
`pred_collapse` mean **0.906** over 250 steps at full peak LR — versus `distill_test`, which completed
its entire descent to 0.797 within 150 steps of peak. `hstate_collapse` fell cleanly to ~0.69 in both
this run and the abandoned one, i.e. **the loop differentiates and the head does not follow** —
collapse is localized in `pred_head`, not the recurrence. Verdict window 8200–8500 not yet read. If it
confirms the plateau, the indicated next run is `distill_test`'s recipe at full scale — pointwise
distill at weight 10, no floor, **from step 0** — rather than any further loss tuning on this lineage.

**Relational distillation is the wrong tool for this problem** (`--sbert-distill-mode relational`,
`losses.relational_distill_loss`). It matches pairwise-similarity structure on the **raw latents** with
no projection, so a perfect fit converges our random-pair cosine to the teacher's **0.208** — the very
cone we are trying to escape. Only the pointwise form can land *below* the teacher's isotropy (0.110 vs
0.208). Relational remains well-motivated for its stated purpose (freeing latent capacity from MiniLM's
arbitrary basis) but only after the cone question is settled.

## Post-run experiments

See [`experiments.md`](experiments.md) — TRM-inspired ablations (arXiv:2510.04871) mapped onto this
architecture: full-thought grad window, supervised halt gate, shared L/H transition, cheap no-grad
depth. All post-run only; nothing there touches the validated A→E training semantics.

---

*Full history, including all superseded designs and the reasoning at each step, is in
[`archive/`](archive/).*
