# Design note — the anti-sycophancy loss does not train the trust gate as wired

**Status:** flagged by the 2026-07-14 Stage-F review (finding #2). Nothing here is a
coding bug — the objective is non-vacuous and the gradient *reaches* the trust-gate
parameters. The problem is that gradient descent has no reason to *use* them, so the
load-bearing behavioral separation (STAGE_F.md §1, "Layer 3 is the load-bearing one") is
unproven as built.

**Implemented (2026-07-14, all opt-in, off = byte-identical A→E/Stage-F):** options 1–3
below are now code, meant to be run as an escalating A/B on the box — nothing here is
*validated*, only wired and smoke-checked.
- **#1 measurement** — `train_dialogue` warns when the syco loss runs without a gate or
  with only a scalar gate, and logs `trust(USER)` mean + across-dim `min/std` (vector
  gate) so the gate's movement is observable. *(commit 54857966)*
- **#2 freeze escape routes** — `--syco-freeze` detaches the response seed + premise
  encoder for the contrastive term (probe: seed grad 59→0, encoder 6→0, trust gate stays
  live; the loop's H-transition still carries ~14.6 — full isolation still needs the
  loop change in §4). *(commit fbdc406e)*
- **#3 explicit prior** — `--trust-prior`: a hinge driving `trust(USER)` a `margin` below
  `trust(SELF)`, trained every step, **plus a lower-floor safety** (`trust_prior_floor`,
  default 0.2). The floor exists because the one-sided hinge has no restoring force: a
  probe at aggressive lr crushed `trust(USER)` to ~0.001 (a fully-zeroed slot the loop
  can't read at all); with the floor it settles at ~0.259. *(commit 31829523)*

Still unbuilt: the H-transition isolation (§4 option 2's "optionally") and option 4
(separate question from premise in the data). Run #1→#2→#3 and read `trust(USER)` before
reaching for those.

---

## 1. What the design claims

STAGE_F.md §3: two user turns differ *only* in an asserted premise (asserts X vs. not-X);
the correct answer is the same. Each premise is compressed into a **USER gestalt in
memory** (not the lane) and read through the **trust-gated memory read**, so
`anti_sycophancy_loss` (both variants match the role-invariant truth **and** each other)
"drives `trust(USER)` down." The trust gate scales an untrusted slot's *value* — attend to
it, but incorporate less of its content — and is the mechanism that makes the model robust
to a confidently-wrong user.

## 2. What actually happens

The routing is correct (verified): the premise enters as a USER gestalt, the loss gradient
reaches **only** the USER row of the role table and `trust_proj` is non-None. But *where the
loss chooses to reduce itself* is the issue. Polarity-invariance of the opening stance can
be achieved anywhere on the path — the memory-reader attention, the H-transition, or the
**response seed** — without ever moving the trust gate.

Probe (smoke model, scalar `trust_gate`, one contrastive batch, gradient magnitudes at init):

| Parameter          | grad magnitude |
|--------------------|----------------|
| `response_seed`    | ~57–194        |
| USER `role_embed`  | ~0.52          |
| `trust_proj.weight`| **~0.033**     |

SGD overwhelmingly moves the seed/encoder; `trust(USER)` stays ≈ its open init (~0.98).
So the model learns to *ignore the premise's polarity in its opening move* without ever
learning *to distrust user assertions* — the general disposition the trust gate is meant to
represent. On any premise the contrastive pair didn't cover, nothing has changed.

## 3. Why the *scalar* gate makes it worse

In `forward_anti_sycophancy` the premise is the **only** per-example signal — there is no
separate question tensor; topic must be read out of the premise gestalt itself. A scalar
gate discounts a slot's whole value (topic **and** polarity together), so driving it down
destroys the topic signal the per-variant terms (`la`/`lb`) still need. The scalar gate is
therefore self-defeating for this loss: it cannot be reduced without hurting the very terms
that would justify reducing it. Only a gate that can discount **polarity while keeping
topic** can express the intended behavior — i.e. the per-dimension `trust_gate_vector`,
which discounts a subspace rather than the whole slot. Even the vector gate is not *forced*
to be the thing that moves; it just makes the correct solution representable.

## 4. Options (roughly ascending cost)

1. **Require `trust_gate_vector` for any anti-sycophancy run, and log it.** Make the scalar
   gate an error (or a loud warning) when `syco_weight > 0`; log `trust_by_role(USER)` (and
   the per-dim gate norm on the polarity subspace) every N steps. Cheapest, and turns "is it
   working?" from a hope into an observable. Prediction to falsify: under the scalar gate the
   USER trust barely moves; under the vector gate a polarity subspace is discounted while
   topic survives. **Do this first regardless of what else you choose** — it is the
   measurement that tells you whether any of the rest is needed.

2. **Freeze the cheaper escape routes during the sycophancy step.** If the seed/encoder
   soak up the gradient, stop them: run `forward_anti_sycophancy` with the response seed and
   (optionally) the H-transition detached / `requires_grad=False`, so the *only* parameter
   that can reduce the loss is the memory read + trust gate. Cheap, surgical, and directly
   targets the mechanism — at the risk of a degenerate solution if the gate alone is too
   weak to carry it (watch the loss floor).

3. **Add an explicit trust-gate objective, not just an emergent one.** Supervise the gate
   directly: a small hinge/regularizer that pushes `trust(USER) < trust(SELF)` (a provenance
   prior), or a contrastive target on the *gated readout* of the premise subspace rather than
   on the opening stance. Makes "distrust user assertions" a first-class training signal
   instead of hoping it falls out of stance-invariance. Most robust, most design work, and
   needs a decision about how much distrust is correct (a prior, not a fact).

4. **Separate the question from the premise in the data.** Give `forward_anti_sycophancy` a
   neutral question tensor distinct from the asserted premise, so "answer the question" and
   "discount the premise's polarity" are different tensors and the loss can only satisfy both
   by using the gate. Requires a data-format change (`tensorize_contrastive`) but removes the
   topic/polarity entanglement at the source rather than papering over it with a vector gate.

## 5. Recommendation

Ship **#1 as a guard/measurement now** (it is a safe, small change and makes the rest
decidable), and treat **#2** as the first thing to try if the measurement confirms the gate
isn't moving. **#3/#4** are the real fix if you want the trust gate to be a trained
disposition rather than a per-batch artifact — but they change the objective/data contract,
so they belong to a deliberate Stage-F design pass, not a pre-run patch. Until one of these
lands, describe the behavioral separation in STAGE_F.md as *an affordance the current loss
does not yet reliably train*, not as achieved.
