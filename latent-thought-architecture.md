# A Latent-Thought Reasoning Architecture

A model that thinks in **latent "thoughts"** — chunk-level vectors, each decoded into tokens by a
separate Talker — using a bounded recurrent **HRM loop** as the thinking mechanism. It fuses four
ideas: **JEPA-Reasoner** (decouple latent reasoning from token generation), **HRM-Text** (a
dual-timescale recurrence for the reasoner), **Thought Gestalt** (a persistent memory of past
thoughts with un-detached gradient), and **Parcae** (a diagonal-decay-gated, stable looped update).

> This is the design spec for the **current** implementation. The full engineering history — every
> revision, bug, and dead end — lives in [`archive/`](archive/). This file describes only where the
> design landed.

---

## 0. What each source contributes

- **JEPA-Reasoner** decouples reasoning from expression: a reasoner works in latent space and a
  separate **Talker** turns latents into tokens. It trains a self-supervised objective that predicts
  the latent of the *next* segment against an **EMA target encoder**, via a scaled cosine loss. The
  Talker is a pure readout — it cannot produce meaningful text without good latents.
- **HRM-Text** replaces the flat transformer with a **dual-timescale recurrence**: a fast **L-module**
  does local refinement, a slow **H-module** carries strategic context. It is stabilized with
  **MagicNorm** (Pre-LN internally, a hard norm at each recurrent module's exit — in this
  implementation only the hard-norm half is used; see §3) and **warmup credit
  assignment** (backprop through the last 2 steps early, expanding to 5).
- **Thought Gestalt** generates one chunk at a time while cross-attending to a **memory of prior
  chunk vectors** ("gestalts"). Crucially, gradient from later losses flows *back through* that memory
  into how earlier gestalts were written — the memory is never detached.
- **Parcae** stabilizes looped models by constraining the looped update; here we use its diagonal
  case — a per-channel **decay gate** `exp(-softplus·dt) ∈ (0,1)` — as the loop's linear carry path.

---

## 1. Core design

A **thought** is a chunk-level latent vector — the representational size Thought Gestalt uses for
sentences, generalized to variable-length semantic chunks. Text flows:

```
        chunk t's tokens
              │
              ▼
   ┌──────────────────┐        RECONSTRUCTION (anchor)          PREDICTION (reasoning)
   │  Chunk encoder    │─── z_t ──┬─────────────► Talker         z_t ──► HRM loop ──► h_t ──► pred_head ──► ẑ_{t+1}
   │ (bidir. + pool)   │          │            decodes chunk t          (reads/writes             ≈ EMA(z_{t+1})
   └──────────────────┘          │            (a pure codec)            gestalt memory)
                                  │
                          the always-on              the HRM loop lives ONLY here, run sequentially
                          anti-collapse anchor       so it reads its accumulating gestalt memory
```

The components:

- **Chunk encoder** — a small bidirectional transformer + masked mean-pool: chunk tokens → one
  latent `z_t`. Shared: `z_t` feeds the Talker (reconstruction), the HRM loop (prediction), and is the
  EMA target's input.
- **HRM inner loop** — the reasoner. Per thought it runs a few cycles of `L-module` × `l_steps` then
  one `H-module`; each step re-projects the state onto the fixed-norm shell (MagicNorm hard-norm). It
  reads the gestalt memory (and, in dialogue, the input lane) by cross-attention, and (Stage D+)
  varies its depth with an **ACT** halting head. The looped update is
  `h_{n+1} = a⊙h_n + B·ê + R(h_n, e)` with `a` the diagonal decay gate.
- **Gestalt memory** — a per-example FIFO of finished thoughts, each with a role tag (USER/SELF/
  SYSTEM). Read by the loop and (via the input lane) for context; written un-detached so credit
  reaches back into earlier thoughts, subject to a truncation window.
- **Talker** — a small causal decoder that reconstructs a chunk's tokens from a latent, teacher-forced
  with an internal right-shift (a learned start vector) so it can't trivially copy the input. It is a
  clean readout of the encoder's latent space.
- **Input lane** — a read-only bidirectional encoder over raw recent tokens + aged gestalts, cross-
  attended by the loop and Talker but never written into the recurrent state (the §4 self/input
  boundary). Used only in Stage F.

### 1.1 Two widths: a thought is wider than a token

Tokens and thoughts do **not** share a width. A token carries one word; a thought is a whole chunk —
a clause or sentence of many tokens — that the Talker must decode back into all of them. Forcing a
thought through the token width would bottleneck that decode, so the thought/chunk-latent width is a
multiple of the token width:

```
d_latent = latent_mult · d_model          # thought width = multiple · token width
```

Everything that *carries a thought* lives at **`d_latent`**: the chunk encoder's transformer body and
its pooled output `z_t`, the gestalt memory, the whole HRM loop, `pred_head`, and the EMA target.
Everything that *handles individual tokens* stays at the word-level **`d_model`**: the token embedding
tables, the Talker's token stream (self-attention, FFN, LM head), and the input lane's raw tokens —
these cross-attend *into* the thought space rather than living in it. The encoder runs its body at
`d_latent` (not merely a projection after a `d_model` mean-pool, which would confine the latent to a
`d_model` subspace and forfeit the capacity). The FFN of the `d_latent` modules scales with the width
(`latent_d_ff = latent_mult · d_ff`).

This is a **deliberate departure from JEPA-Reasoner**, whose token embeddings and segment latents
share one "Latent Dim" (its analyzed latents are near-token-sized, empirically linear combinations of
vocabulary embeddings). Our thoughts are genuinely multi-token chunks, for which that identity does
not hold — so we decouple the two. (This is a *different* axis from JEPA-Reasoner's "Attention Dim >
Latent Dim", which widens the reasoner's **internal** width while the latent flowing between
components stays narrow; here it is the flowing latent itself that widens.) `latent_mult = 1` recovers
token == thought exactly — the validated baseline and an exact no-op — and is what the five baseline
presets ship; the `small-w3` / `base-w3` / `large-w3` / `xl-w3` rungs are `latent_mult = 3`, each
rebalanced to its baseline tier's parameter budget by trading token width for thought width. Widening the thought
moves the anti-collapse machinery (§2.4) into the wider space, so `cosine_loss_k` and the variance
floor are re-tuned at `d_latent`.

---

## 2. The two losses, split by role

The model is trained by two objectives that share the encoder but touch the rest of the model
**disjointly**. This split is the load-bearing design decision.

### 2.1 Reconstruction — a pure autoencoder codec (the anchor)

`encode chunk t → the Talker decodes chunk t` — masked next-token NLL against the chunk's own tokens,
run in parallel over all chunks. **No HRM loop, no memory.** It trains the **encoder + Talker**.

Because a constant latent cannot reconstruct varied chunks, this objective cannot be satisfied by a
collapsed representation — so it is the always-on **anti-collapse anchor** for the shared encoder, and
the Talker it trains is a faithful codec of the latent space. This is what `val_loss` and
`generate --score` measure.

**Two costs of this term, both measured (2026-07-22/23), both load-bearing:**

- **It anchors the encoder but does not spread it.** Reconstruction needs each latent to be
  *decodable*, and a tight huddle of latents is perfectly decodable — nothing in the objective rewards
  angular separation. So it drives the latent space **anisotropic** (measured random-pair cosine
  0.495–0.515, where a healthy space is ~0), and that narrow cone is what makes the predictor's
  centroid solution nearly optimal. Reconstruction is therefore an anti-collapse anchor for the
  *encoder* and simultaneously the upstream *cause* of collapse in the *predictor*. It stands in
  direct tension with the distillation term (§2.4), and it wins by default: `L_rec` is unbounded
  (~0.4) while `L_dist` is bounded (≤1), so an under-weighted distill term is simply overrun.
- **It makes the Talker a rigid exact-latent decoder.** `probe_latent_use` measures NLL under the
  true latent vs. a shuffled one: **0.0042 vs 41.4**. The codec genuinely *uses* the latent (no
  memorization) — but it decodes only near-exact ones. At generation the Talker is handed a
  *predicted* latent, empirically ~0.5 cosine to the truth, which it has never practiced on. This is
  the **train/serve exposure gap**, and it is why a good codec (`val_loss` 0.0067) still generated
  mush. The token-grounded term (§2.4) exists for exactly this and is deliberately detached to the
  Talker, so it buys decoder tolerance without letting the encoder pay for it by emitting
  low-information latents.

### 2.2 Prediction — the HRM loop, sequential, with memory (the reasoning)

For each chunk *t*, in order:

```
h_t = loop(z_t, memory)     # the loop reasons, reading its accumulating gestalt memory
memory.write(h_t)           # detached in Stage B; un-detached from Stage C (cross-thought credit)
pred_head(h_t) ≈ EMA(z_{t+1})   # predict the next chunk's latent (scaled cosine, k=4), EMA stop-grad
```

This trains the **loop + encoder + pred_head + the memory readers/writers** to reason forward. It is
JEPA-Reasoner's self-supervised objective, run *on the reasoner* (the loop) exactly as JEPA-Reasoner
runs it on its reasoner transformer — and *sequentially with memory* so it is also Thought Gestalt's
un-detached cross-thought reasoning. Gradients use the inner-loop 2→5 truncation; cross-thought credit
through memory is bounded by its own window. Generation uses the same `pred_head`, its output rescaled
onto the encoder-latent norm shell at inference — the cosine objective trains the prediction's
*direction*, not its scale, and the Talker consumes latents unnormalized.

### 2.3 Why the loop is in prediction and not reconstruction

If the loop's output had to *both* decode the current chunk (reconstruction) *and* predict the next
(prediction), it would be pulled two ways: "preserve chunk *t*" (and since the encoder already
produced `z_t`, that trains the loop toward an identity pass-through) versus "shift to chunk *t+1*."
Removing the loop from reconstruction frees it to be purely predictive, leaves a clean encoder↔Talker
codec, and — because prediction is now sequential — is the only thing that actually trains the gestalt
memory. Reconstruction anchors the encoder; prediction is where reasoning and memory live.

### 2.4 Anti-collapse

A shared encoder under a predictive self-distillation loss can collapse to a constant (predict a
constant from a constant is a stable fixed point). **Collapse has two distinct seats, and the
encoder-side defenses do not reach the second one.**

**Encoder-side** (the latent space itself going constant):

1. **The reconstruction anchor** runs every step and cannot be satisfied by a constant latent — the
   load-bearing defense.
2. **A variance floor** (VICReg-style hinge) on the shared latent's per-dimension variance — dormant
   in normal operation, active only near collapse. A floor, never a target.
3. **A slow EMA target** (momentum 0.996) — harder to chase into a constant than a fast one.

**Predictor-side** (`pred_head` going constant while the latent space is fine). This is the failure
that actually occurred: a finished run measured `pred_collapse` **0.98** with `hstate_collapse` 0.88 —
the loop's states stayed diverse and the head crushed them into one vector. None of the three defenses
above can see or prevent it. Two more, in order of measured importance:

4. **Semantic distillation to open the cone** (`--sbert-distill-weight`, §2.1). A narrow cone is
   *why* the centroid is attractive: if unrelated latents already sit at cosine 0.5, the mean is close
   to every target and predicting it is nearly optimal. Distilling a frozen sentence encoder through a
   **learned projection** (the constraint binds the projection, not the latent) opened the cone to
   random-pair cosine **0.110** — below the teacher's own 0.208 — and the run that achieved this is the
   only one whose predictor escaped (`pred_collapse` 0.9986 → 0.797), *with no predictor-side loss at
   all*. Two caveats, both measured: driving distill cosine to convergence (0.994) **over-imitates**
   and gives back the task-specific advantage, so the teacher must act as a prior, not a target; and
   the cone cannot be reopened once the encoder hardens (~19% recovery after only 5k steps of
   reconstruction), so **distillation must be on from step 0.**
5. **Hard-negative InfoNCE** on the prediction — a constant forecast is maximally uninformative and
   pays the worst possible ranking loss. Same-document negatives force the next-chunk distinction
   rather than cross-document topic separation.

**Signals.** `val_loss` and `latent_std` **cannot see predictor collapse** — a collapsed head and a
good one score the same, because `forward_grounded` is encoder→Talker and never touches the loop. Read
`pred_collapse` / `hstate_collapse` (logged every eval) and the probes' LIFT. `tok_nll` is *not* a
collapse signal either: it is detached to the Talker, which will happily learn to decode a collapsed
predictor's output. Two further cautions: the escape is **learning-rate-gated** (it moves only near
peak LR, so a long stage's warmup delays it), and single-reading noise on `pred_collapse` is ~±0.02, so
judge over a ≥4-reading window at peak LR.

---

## 3. Why the non-obvious technical choices

**Decay gate *and* hard-norm, not one or the other.** The looped update is `a⊙h + B·ê + R(h,e)`.
The decay gate `a` is contractive, but `R` is an unconstrained nonlinear term, so `a` alone does not
bound the map. **Boundedness at any depth comes from MagicNorm's hard normalization** (re-project onto
the ‖h‖=√d shell at every step); the decay gate is the linear carry path that *shapes* the on-shell
dynamics toward convergence (the mechanism behind Parcae's predictable test-time-depth scaling).
There is **no explicit Pre-LN inside the L/H cells** (`norm.PreNormWrapper` exists but is unused):
the `R` sublayer's inputs are kept well-conditioned by the loop's invariants instead — `h` enters
hard-normalized from the previous step's exit, and `ê` is normalized before injection.

**Chunk boundaries are sentence/clause-aware, not fixed windows.** A thought should be a semantically
complete unit; a fixed window bisects clauses and reintroduces compounding fragility at the chunk
level. We use "SaT Capped" (sentence boundaries + punctuation-aware length capping).

**The 3-fast : 1-slow L:H ratio is an empirical HRM-Text hyperparameter**, not derived, and is *not*
subsumed by ACT (which sets total depth, not the L:H interleave). Held fixed.

**Truncated credit assignment at two levels.** Inner loop: only the trailing `k` L/H steps carry
gradient (warmup 2→5), with the entering state severed so cross-thought credit never flows through the
raw recurrent chain. Memory: only the trailing `w` gestalt slots carry gradient (warmup 1→5). Distant
credit still reaches back *transitively* through memory (attenuated per hop), so the activation graph
spans the document once memory is un-detached — budget GPU memory accordingly.

**Two losses, not one.** Pure self-prediction admits degenerate, self-consistent-but-inexpressible
latents; the reconstruction codec anchors them to language. Pure reconstruction never learns to reason
forward. Each covers the other's failure mode.

---

## 4. Chatbot context: the input/self boundary (Stage F)

The current turn's user input goes in as **raw tokens with full bidirectional attention**, not
compressed into thoughts — self-generation compounds errors (the reason for chunking), but user input
is a fixed external artifact with no such process, and compressing it costs fidelity (exact quotes,
code, numbers). Aged input is compressed into role-tagged gestalts for bounded-cost recall.

Two lanes feed the memory: an **input lane** (raw tokens + aged gestalts) that is only ever
*cross-attended to*, and a **self lane** (the loop's own thoughts) that is the only thing allowed to
write the recurrent state. Role tags (USER/SELF/SYSTEM) let attention weight sources differently — the
substrate for representing "the user asserted X" distinctly from "I concluded X." This is an
*affordance*: it only pays off if a training signal (e.g. an anti-sycophancy contrastive loss)
exploits it. That loss and the two-lane fine-tuning are Stage F, deferred and not yet exercised.

---

## 5. Training curriculum

Nothing trains end-to-end from scratch: the Talker needs good latents; the loop can't be deepened
until near a stable fixed point; the memory gradient is the same problem one level up; the EMA target
must be meaningful before prediction chases it. So each stage's stability is the precondition for the
next. The autoencoder anchor runs **every** stage; the predictor turns on at B.

| Stage | Adds | Loop | Memory | ACT |
|---|---|---|---|---|
| **A** | autoencoder codec (encoder + Talker) — grounds the codec and makes the EMA target meaningful | off | — | — |
| **B** | the HRM loop + on-loop SSL predictor; inner-loop grad warmup 2→5 | on, fixed depth | detached | — |
| **C** | un-detach the gestalt memory (cross-thought reasoning), memory warmup 1→5 | on | un-detached | — |
| **D** | adaptive depth (ACT) | on | un-detached | on |
| **E** | consolidation at full config | on | un-detached | on |
| **F** | two-lane input/self separation, cross-turn memory (dialogue fine-tuning) | on | spans dialogue | on |

Stage transitions gate on fixed per-stage step budgets (a validation-plateau gate is also available).
`val_loss` is the loop-independent autoencoder reconstruction, so it is comparable across every
boundary — a rise when the predictor turns on at B is the collapse signal.

---

## 6. Honest limits and open questions

- **ACT depth is not a learned compute dial yet.** The soft ponder cost gives the halting head no
  compute-vs-quality gradient (task loss can't see a non-differentiable halt), so it degenerates toward
  minimum depth. A real ACT accumulator or REINFORCE is needed to make "think harder" learnable. The
  loop's *executed* depth is trained (prediction flows through it); *how much* depth to spend is not.
- **`ssl_loss_weight` is co-equal (1.0) with reconstruction**, validated collapse-free at smoke and
  512-d, but the balance may shift with scale — re-tune on the full run.
- **The interleave/weighting of the two losses** and how it should scale is open (neither source paper
  combines them).
- **What gets written to memory** — raw H-state vs. a separately-projected gestalt readout — is open.
- **The anti-sycophancy loss and Stage F two-lane training** are designed but not implemented/exercised.
- **No large training run has been done.** Everything is verified at smoke and small-preset scale.
