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

### 2. TRM-style supervised halt gate (Stage D variant)

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
