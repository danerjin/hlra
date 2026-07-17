# Post-run experiments — TRM-inspired changes

Changes suggested by **"Less is More: Recursive Reasoning with Tiny Networks"** (TRM,
Jolicoeur-Martineau, [arXiv:2510.04871](https://arxiv.org/abs/2510.04871)), mapped onto this
architecture in a 2026-07-11 review. **None of these land before the A→E run** — the training
semantics are validated (four pre-flight reviews); everything here is a post-run ablation.

Standing caveat: TRM's evidence comes from ~1k-example exact-match puzzle benchmarks
(Sudoku/ARC/Maze). Its *gradient-flow* findings should transfer; its *shrink-the-network*
findings are substantially a small-data regularization story and may not.

## Where we already match TRM (no change needed)

- **No 1-step gradient.** HRM's biggest weakness (backprop through only the final L/H step,
  justified by a fixed-point argument TRM refutes) was never used here — `_TruncationSchedule`
  in `files/hrm_loop.py` is windowed truncated BPTT, i.e. TRM's "no-grad prefix, full-grad tail".
- **The (y, z) reinterpretation.** TRM reads HRM's two states as y = current answer,
  z = latent scratchpad: refine z several times, update y once. That is structurally our loop:
  `l_state` refined 3× per cycle, one `h_state` update, `h_state` is the output thought.

## Experiments, in priority order

### 1. Full-thought grad window (config-only — strongest TRM evidence)

TRM's largest ablation win over HRM: backprop through the *entire* final recursion.
Shipped config leaves 3 of 8 loop steps outside the gradient
(`l_steps=3`, `n_cycles=2` → 8 steps; `inner_loop_grad_window_end=5`).

- **Change:** warm `inner_loop_grad_window_end` 5 → 8. `window >= total` is already supported —
  the cut moves to step 0 and still severs the entering cross-thought states, so cross-thought
  truncation semantics (§3.5/§3.6) are untouched.
- **Cost:** 3 more steps of graph memory per thought.
- **Compare on:** small preset, Stage B/C metrics (nll, ssl cosine) vs. window=5 baseline.

### 2. TRM-style supervised halt gate (Stage D variant) — **PROTOTYPED 2026-07-13**

Replace the Graves/PonderNet ponder cost with TRM's simplification of ACT: a per-row halting
probability trained with BCE against "is the output good enough now". Directly addresses the
documented weakness (soft ponder cost has no compute-vs-quality gradient; halting degenerates
toward minimum depth) and is simpler than a full ACT accumulator or REINFORCE.

- **Target problem:** TRM's BCE target is exact-match correctness, which doesn't exist in a
  latent LM. Preferred proxy: **marginal-improvement** — halt-target = 1 when one more cycle
  improves the SSL cosine loss by less than ε (self-calibrating; needs the SSL head evaluated
  per ponder step). Fallback: thresholded quality (cosine error < τ), cheaper but τ is
  arbitrary and non-stationary.
- **Keep:** the min-depth floor (`n_cycles`), `act_max_ponder_steps` cap, active-row masking.
- **Gain over current:** quality-grounded learning signal for the halt head; per-row halting
  instead of the batch-mean vote.
- **Not fixed:** no pressure to think *harder* on hard chunks — TRM sidesteps this by running
  max steps at eval. A learned compute dial still needs an accumulator/REINFORCE.
- **Compare on:** small preset Stage D, vs. ponder-cost baseline: depth distribution
  (does it escape min-depth?), nll/ssl at matched compute.

**Implementation (prototype, opt-in, off by default — the A→E path is byte-identical).**
Chose the **marginal-improvement** target. Gated entirely by `ModelConfig.halt_mode`
(`"ponder"` default = the validated soft cost; `"supervised"` = this gate); `--halt-mode
supervised` on `train_scaled.py` flips it. It only diverges at Stage D+ (ACT on); fixed-depth
stages A–C are identical in both modes. Pieces:
- `losses.supervised_halt_loss` — BCE(halt_logit_c, target_c) with a self-calibrating
  target: `target_c = 1` iff `cos_dist_c − cos_dist_{c+1} < halt_epsilon` (the next cycle
  barely helps); the cap cycle's target is 1. cos_dist is a **detached label**.
- `hrm_loop.HRMInnerLoop.forward_halt_trace` — a *separate* method (leaves `forward`
  byte-identical) that runs the loop to the `act_max_ponder_steps` cap and returns the H-state
  after every cycle `(cap, batch, d)`, reusing the same rolling `_TruncationSchedule` as ACT.
- `model.forward_self_supervised_halt` — a parallel predictor reached only via a guarded
  dispatch in `forward_self_supervised`. Per chunk it (a) **selects** a per-row halt depth
  (first cycle ≥ floor with prob > 0.5, else cap) and drives the primary SSL prediction +
  memory write from the *selected* thought (train/test depth match, per-row depth = the gain
  over the batch-mean vote), and (b) **supervises** the halt head with the BCE above, reading a
  **detached** H-state so the BCE trains only the head — the primary losses still shape the
  reasoning. Returns the same `(ssl, second_term)` 2-tuple, so `trainer.py` is untouched.
- **Verified:** the ponder path is bit-for-bit identical to pre-change HEAD (ssl/second/gradnorm
  to the last digit, both `use_act` modes); the supervised path trains (halt BCE 0.98 → 0.04 on
  an overfit smoke batch) and the selected depth adapts from cap→floor as marginal improvement
  vanishes. Config guards a bad mode / an inverted depth range.
- **Still open (as flagged above):** no "think harder on hard chunks" pressure (needs an
  accumulator/REINFORCE); a depth *spread* needs varied/harder data than the smoke overfit; the
  trainer logs the second term under the `ponder` key in both modes (a label only). Unvalidated
  at scale — this is a runnable A/B, not a result.

**Measured 2026-07-15 (box-free CPU smoke on the offline chunker-v3 cache) + a target fix.**
Ran the supervised gate through Stage D and probed the selected depth. Two findings:
- **It trains clean but sits at the min-depth floor — and here that is *correct*.** A→D→F walks
  cleanly (`val_loss` 7.50→5.69 through the Stage-B boundary, no collapse). But the trained
  `halting_head` emits a near-constant P(halt)≈0.95 at every cycle → **100% of rows at the floor**.
  Root cause is not the gate: the loop's per-cycle cos_dist is **flat** on this data (0.6023 →
  0.6013 → 0.6011 → … → 0.6010, ~0.0003 total floor→cap). Extra cycles genuinely don't help, so
  halting at the floor is the right call and `halt_epsilon` is moot (the whole improvement is below
  any sane epsilon). Smoke has **no depth signal**, so it cannot demonstrate depth-escape either way.
- **A real design bug in the `marginal` target, now fixed (opt-in `halt_target`).** The marginal
  target halts at cycle *c* when the *next* cycle's improvement < epsilon — the local slope. On a
  gently-but-steadily improving curve (each step < epsilon, but large cumulative gain) it halts at
  the floor even though many more cycles would substantially improve — the opposite of "think harder
  on hard chunks". Added `ModelConfig.halt_target`: `"marginal"` (default, byte-identical) vs
  `"best_relative"` — halt when within epsilon of the chunk's **best** achievable cos_dist over its
  legal cycles ("keep going until you're about as good as you'll get"). Proven on synthetic curves:
  gently-improving (0.008/cyc) → marginal picks depth 1, best_relative picks depth 4; flat → both
  pick the floor (no forced compute); steep-then-plateau → both halt right after the drop. Marginal
  BCE is byte-identical to the pre-change loss (verified to 1e-9). `--halt-target best_relative` on
  `train_scaled.py`. Still needs real/harder data to show end-to-end depth-escape — the synthetic
  proof isolates the target, not the loop's flatness.

  **A per-step-gain "frugality floor" was prototyped to cap best_relative over-think, then removed —
  no benefit.** A min_gain sweep on 3000 synthetic curves found no beneficial regime: on realistic
  diminishing-returns (exp-decay) curves best_relative already halts at the convergence knee (a small
  floor is inert; a large one just trades quality for compute linearly, same as lowering the depth
  budget); on a linear curve (the only genuine over-think regime) it is all-or-nothing (inert below the
  slope, halt-at-floor above); and it is myopic (halts before a late drop on a non-monotonic curve,
  depth 7→1). The over-think worry was largely unfounded — best_relative self-limits at the knee — so
  the knob was pulled rather than carried unused. **Kept the resume guard** (`trainer.load`):
  halt_mode/halt_target/halt_epsilon are CLI-sourced, so resuming without re-passing them silently
  reverts the halt policy; the guard warns loudly on any halt-config drift (weights still load; a
  warning like the schedule guard, since switching may be intentional). Verified: fires on drift, no
  false-positive on match.

### 3. Shared L/H transition network (ablation, lowest confidence)

TRM collapses HRM's two networks into one tiny net used for both the z-updates and the
y-update. Analogue: one shared `DiagonalDecayGate` instead of separate
`l_transition`/`h_transition` — halves loop transition params.

- **Skepticism:** TRM's "less is more" is largely small-data regularization (~1k examples);
  at Wikipedia scale that pressure is absent, and our H-transition sees a genuinely different
  input distribution (memory + input-lane injections).
- **Run only if** experiments 1–2 move the needle and loop capacity looks like the bottleneck
  in neither direction.

### 4. Cheap extra depth via no-grad cycles (pairs with #1)

TRM runs T−1 recursions without gradient + one fully-backpropped recursion, buying depth at
the memory cost of one recursion. Our fixed-depth mode has the same property: raising
`h_updates_per_thought` while keeping the grad window fixed adds deliberation depth with no
extra graph memory (pre-cut step graphs are freed as carried states detach).

- **Change:** e.g. `h_updates_per_thought` 2 → 3–4 with the grad window covering only the
  trailing thought's-worth of steps. Optionally wrap pre-cut steps in `torch.no_grad()` for
  compute savings (currently they build then free their graphs).
- **Watch:** wall-clock cost per step; the workload is launch-overhead-bound on MPS/ROCm.

### 5. Document-level end-of-text (the A→E half of the turn-end)

Found 2026-07-16 while auditing termination. **The Stage-F half of this shipped**
(`STAGE_F.md` §2.1: a learned turn-end head on `DialogueAdapter`, off by default) —
Stage F had not started, so it was free to change. **This entry is the A→E half,
which is post-run by the usual rule.**

The model has a trained end-of-**chunk** stop (PAD, §19.2) and now a Stage-F
end-of-**turn** gate, but still no end-of-**document**: there is no EOS token
anywhere, and `generate.generate` emits a caller-supplied `n_chunks` with no break.
For A→E pretraining this is mostly benign — the objective is next-chunk latent
prediction and docs truncate at `max_chunks_per_doc` regardless — so it is not a
defect in the run, just a missing capability.

- **Change:** mirror the Stage-F head at document level — a binary "no chunk
  follows" head off `h_t` in `forward_self_supervised`, supervised from the chunk
  mask (`chunk_mask[:, t+1]`), and a `break` in `generate.generate`.
- **This entry originally claimed a truncation blocker. That claim was wrong and is
  withdrawn** (2026-07-16 review). It asserted `max_chunks_per_doc` is 12 and that
  truncation is therefore the common case, so masking filled rows "would mask nearly
  the whole corpus." Every number in that was off:
  - **12 is the `smoke` preset only.** The real presets are **32** (`small`,
    `small-w3`, `base`) and 48 (`large`); the in-flight run is `small-w3` → 32.
  - **Truncation is the minority case.** Measured on the repo's own real-text cache
    (1401 docs at `max_chunks_per_doc=32`): **39.8%** of docs fill every slot, so
    **60.2% end naturally**. (That 39.8% is an *upper bound*: the cache was built
    with the pre-0711 chunker, whose tiny-chunk pathology inflates chunks/doc. The
    v3 chunker makes fewer, larger chunks → fewer filled docs → more clean labels.)
  - **Masking costs ~2%, not "nearly the whole corpus."** `_turn_end_labels` drops
    one label per filled row, not the row: 557 of 29568 labels = **1.88% masked,
    98.12% retained**, leaving ~844 clean document-end positives.
  - **No cache change is needed.** `_turn_end_labels` reads `chunk_mask` alone, and
    that is already in every shard. No truncation flag, no `CHUNKER_VERSION` bump, no
    re-prep. (The sub-claim that truncation is not *currently recorded* is true — it
    is just not load-bearing.)
- **So this is feasible; it is a cost/benefit call, not a blocked one.** The honest
  reasons to still not do it: end-of-document only matters for free-running
  generation, which is a serving concern Stage F already covers; and it touches
  `forward_self_supervised`, the most heavily validated function in the repo, for a
  capability nothing currently needs. Post-run regardless.
- **Compare on:** whether the Stage-F gate trains at all first (§2.1's honest
  limits). There is currently **no evidence** the detached head extracts signal even
  on clean dialogue labels — settle that before spending a re-validation here.

## Rejected (don't transfer)

- **Tiny 2-layer network / aggressive downsizing** — small-data regularization; our regime
  doesn't reward it.
- **Attention-free (MLP-Mixer) variant** — TRM used it only for fixed small grids; memory and
  input-lane reads need attention.
- **Weight EMA for evaluation** — small-data stabilizer; we already have the JEPA
  target-encoder EMA (different purpose, momentum 0.996).
- **Deep supervision on the same input** (TRM's N_sup ≤ 16 passes per puzzle) — our
  supervision is a stream of chunks, each already a supervised thought with state carry-over;
  re-running the loop on the same chunk multiplies cost for unclear benefit.
