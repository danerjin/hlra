# A Latent-Thought Reasoning Architecture

Combining **JEPA-Reasoner**, **HRM-Text**, **Thought Gestalt**, and **Parcae** into a single model
that thinks in latent thoughts — each decoded into multiple tokens — using HRM-style looping
(a Parcae-stabilized looped transformer) as the "thinking" mechanism.

---

## 0. Source material — what each paper actually contributes

None of these four papers fully overlaps with the others. Each fixes a different, specific
problem, and the combination is only justified because the problems compose.

- **JEPA-Reasoner** ("JEPA-Reasoner: Decoupling Latent Reasoning from Token Generation")
  decouples reasoning from expression using a Joint-Embedding Predictive Architecture for pure
  latent-space reasoning and a separate **Talker** module for linguistic reconstruction. Training
  has two phases: ordinary next-token pretraining, then a self-supervised phase where the model
  predicts the latent representation of the next sequence segment via a **scaled cosine distance
  loss** against an **EMA target encoder**. Ablations confirm the Talker is a pure readout head —
  it cannot generate meaningful content without proper latent representations from the Reasoner.

- **HRM-Text** ("HRM-Text: Efficient Pretraining Beyond Scaling") replaces the flat transformer
  with a **dual-timescale recurrence**: a fast **L-module** performs local iterative refinement
  while a slow **H-module** maintains stable semantic context across cycles — concretely, two
  high-level cycles, each executing three fast L-module updates followed by one slow H-module
  update. It is stabilized with **MagicNorm** (PreNorm internally, hard normalization at the exit
  of each recurrent module) and **warmup deep credit assignment** (gradients initially
  backpropagated through only the final two recurrent steps, expanding to the final five as
  training progresses). It also uses a **PrefixLM mask**: full bidirectional attention across
  instruction tokens, causal generation for the response.

- **Thought Gestalt** ("Modeling Language as a Sequence of Thoughts") operates at two levels of
  abstraction: it generates the tokens of one sentence at a time while cross-attending to a
  **working memory of prior sentence representations** ("gestalts"), with token and sentence
  representations produced by a shared transformer stack and trained with a single next-token
  loss. Critically, gradients from future token losses flow **backward through cross-attention**
  into the parameters that generated earlier sentence vectors — the memory is never detached.

- **Parcae** ("Parcae: Scaling Laws For Stable Looped Language Models") fixes the actual
  pathology in looped transformers: instability arises from **large spectral norms in the
  injection parameters** of the looped residual update. Parcae constrains the spectral norm via
  **discretization of a negative diagonal parameterization**. Empirically, looping and data
  should scale together under a fixed FLOP budget at train time, and at test time, looping scales
  compute following a **predictable, saturating exponential decay** — i.e., more loop iterations
  is a real, tunable test-time-compute dial.

---

## 1. Core design: three nested loops

A **thought** is a chunk-level latent vector — not a token (too fine-grained), not a whole
document (too coarse to decode faithfully). It sits at the representational size Thought Gestalt
uses for sentences, generalized to variable-length semantic chunks.

Three timescales of computation, each borrowed from a different paper:

```
 THOUGHT LOOP (JEPA-Reasoner / Thought Gestalt cadence)
 ────────────────────────────────────────────────────────────►  time
   Thought₁          Thought₂          Thought₃
 ┌──────────┐      ┌──────────┐      ┌──────────┐
 │ INNER    │      │ INNER    │      │ INNER    │
 │ HRM LOOP │─────►│ HRM LOOP │─────►│ HRM LOOP │───►  ...
 │ (Parcae- │  z₁  │          │  z₂  │          │  z₃
 │ stable)  │      │          │      │          │
 └────┬─────┘      └────┬─────┘      └────┬─────┘
      │  write            │  write            │  write
      ▼                   ▼                   ▼
 ┌─────────────────────────────────────────────────┐
 │        GESTALT MEMORY (FIFO, ungated grad)       │◄── cross-attend
 └─────────────────────────────────────────────────┘
      │ z₁                 │ z₂                 │ z₃
      ▼                    ▼                    ▼
   TALKER              TALKER               TALKER
 "tokens for     "tokens for          "tokens for
  thought 1"       thought 2"          thought 3"
```

### 1.1 Producing a thought: the inner HRM/Parcae loop

Instead of one forward pass producing a thought (vanilla JEPA-Reasoner), each thought is the
output of a **bounded recurrent deliberation**, structurally identical to HRM-Text's L/H split:

- **L-module (fast)**: several inner steps of local refinement per thought.
- **H-module (slow)**: updates once per thought, carrying the "strategic" state forward — this is
  the role Thought Gestalt's sentence-vector plays, generalized from sentence boundaries to
  arbitrary thought-chunks.
- **Stability**: the loop's residual update is parameterized the Parcae way —
  `h_{n+1} = A h_n + B·e + R(h_n, e)`, with `A` a spectral-norm-constrained negative-diagonal
  matrix (discretized) and the injection `B·e` normalized. MagicNorm is kept as the intra-block
  norm; Parcae's constraint is applied specifically to the state-transition matrix governing
  loop-to-loop recurrence. These are not redundant — see §3 for why both are needed.
- **Adaptive depth (test-time compute dial)**: the *number of L/H cycles per thought* is adaptive
  (ACT-style halting), not fixed. A filler word gets a shallow pass; a load-bearing inference step
  gets many inner iterations. (The L:H *ratio within* a cycle is a separate quantity that a single
  halting head does not set — see §3.2.) This turns "more inner iterations" into a real test-time
  compute knob; and because Parcae's constraint makes the loop *converge* rather than merely stay
  bounded (§3.3), that knob follows the predictable, saturating scaling Parcae reports instead of
  producing noise.

### 1.2 The persistent gestalt memory

When a thought's inner loop finishes, the H-module's final state is written into a fixed-capacity
FIFO memory bank (Thought Gestalt's mechanism). Two readers:

- The **next thought's** inner loop cross-attends into it — this is how context persists across
  thoughts without reprocessing raw tokens, at O(1) cost per thought rather than growing with
  sequence length the way raw KV-cache attention does.
- The **Talker** also cross-attends into it, so it can reference prior gestalts (coreference,
  long-range grounding) rather than seeing only the current thought vector.

The memory is **not detached**: gradients from later losses can flow back through it into the
H/L states that produced earlier thoughts, subject to truncation (§4).

### 1.3 The Talker

A separate, lightweight decoder-only stack (JEPA-Reasoner's ablation-verified design). It takes a
finished latent thought (post-loop, post-H-module) plus gestalt-memory cross-attention, and
autoregressively emits the tokens for that chunk. Its causal token-sampling noise cannot leak
backward into the Reasoner's next thought, because the next thought conditions on the *latent*
and the memory, never on the Talker's sampled tokens — JEPA-Reasoner's "error containment"
property, preserved.

---

## 2. Training objective: two losses, two granularities

Neither source paper hands us this combination directly — JEPA-Reasoner's SST phase and Thought
Gestalt's end-to-end phase are alternatives *within their own papers*, not something either paper
combines.

1. **Self-supervised latent loss** (cheap, parallelizable, JEPA-style): predict the EMA target
   encoder's representation of the next chunk via the scaled cosine distance loss
   `L(θ,θ′) = k·(1 − cos(h_pred(θ), h_target(θ′)))`, with the same EMA-momentum trick (momentum
   0.98 on the target embeddings, to prevent rank collapse while allowing angular adjustment).
   Trains raw predictive competence without running the Talker or unrolling through memory — fully
   parallel across chunks, like JEPA-Reasoner's SST phase. *(Implementation note: the naive form of
   this — SSL on the shared chunk latent, equal weight, every step — collapses the latent. The
   reference implementation runs the prediction in a **separate projection space**, down-weights it,
   raises EMA momentum, and adds a variance floor. See §2.4.)*

2. **Grounded end-to-end loss** (expensive, Thought-Gestalt-style): periodically, run the Talker,
   take its NLL loss on realized tokens, and backprop through the latent thought, through the
   inner HRM loop, and back through the un-detached gestalt memory into earlier thought-steps.
   This is what keeps latents *decodable*, not just self-predictive — pure self-distillation
   objectives can drift toward representations that are self-consistent but not linguistically
   expressible. Concretely this is a **reconstruction (autoencoder) loss** — encode a chunk, run
   the loop, decode that same chunk's tokens — which is *why* it resists collapse: a constant latent
   cannot reconstruct varied chunks. It is therefore the anti-collapse **anchor** for the shared
   encoder, not merely a regularizer (§2.4).

3. **Credit-assignment truncation** (HRM-Text, applied at two levels): warm up the backward
   horizon — start by backpropagating through only the last two steps, expand to the last five as
   training progresses — applied both at the **inner-loop level** (how many L/H iterations get
   gradient) and at the **outer thought-memory level** (how many past thoughts get gradient
   through the FIFO). This is what makes Thought Gestalt's "ungated gradient through memory" trick
   tractable at scale instead of becoming a full-sequence BPTT graph.

The interleave ratio and relative weighting between the two losses is an open empirical
question — neither source paper actually combines them, so there's no principled answer to lift.

### 2.4 Empirical addendum — objective interference between the two losses

*(Added after building and running the reference implementation; see the implementation notes for
the run that produced this.)*

Combining the two losses naively — one **shared** chunk encoder feeding both the reconstruction
path and the self-supervised prediction, with the SSL loss at equal weight and running every
step — **collapses the shared latent**. Observed directly in a small run: once the self-supervised
loss switches on, the encoder learns to emit a near-constant vector for every chunk (cosine between
predicted and target latents → ~0.996), which *perfectly* satisfies the self-predictive objective
while carrying no information — and because the encoder is shared, decodability regressed at exactly
the step the SSL loss turned on.

Three properties make this severe and easy to miss:
- **Silent.** The SSL loss falling toward zero *looks* like success; only a separate
  reconstruction/validation signal reveals the damage.
- **Propagating.** A shared encoder means SSL collapse flattens the representation the Talker and
  the inner loop depend on, not just the SSL objective.
- **Absorbing.** The EMA target is a copy of the (collapsing) online encoder, so "predict a constant
  from a constant" is a stable fixed point with no gradient pressure to escape.

The fix that removed it (verified: no collapse, no reconstruction regression):
1. **Reconstruction is the always-on anchor.** The grounded loss (§2.2) is an autoencoder and
   *cannot* be satisfied by a constant latent, so keeping it dominant — not thinned below SSL from
   Stage D — holds the shared encoder informative. This is the load-bearing part, and it validates
   the framing in §3.7: reconstruction is an anchor, not a taperable regularizer.
2. **Separate SSL projection head.** The self-supervised loss operates on a projection of the shared
   latent (BYOL-style), with its own EMA copy, so if it still wants to collapse it collapses *its
   own head* rather than the shared encoder — this resolves the §6 open question toward separate
   heads.
3. **SSL demoted.** Cosine term down-weighted (≈0.1×) and EMA momentum raised (0.98 → 0.996: a
   slower target is harder to chase into a constant).
4. **Variance safety floor.** A VICReg-style hinge penalizes the shared latent's per-dimension
   variance falling below a *low* floor — dormant in normal operation, active only as the latent
   approaches collapse. The floor must sit *below* the encoder's natural scale; setting it too high
   forces a rescale that itself hurts reconstruction (learned the hard way).

Two methodological lessons fell out of this: (a) the validation metric must be
**reconstruction-only** — adding the SSL term to eval at a different weight than training makes the
Stage-D transition look like a regression when it isn't; (b) an anti-collapse regularizer is a
*floor*, not a *target* — it must never drive the latent's scale, only prevent collapse to zero.

One structural consequence of the separate-head fix deserves stating: once the SSL prediction
lives in its own projection space, the SSL predictor **no longer provides an encoder-space
next-latent map** — but that map is exactly what free-running generation needs (predict the next
chunk's latent, feed it to the inner loop as the next injection). The fix therefore has to come
with a dedicated generation head trained in the *shared encoder* space; to keep it from
reintroducing the collapse pressure this section removed, it is trained with both its input and
its target detached, so its loss reaches only the head itself — a pure readout, exactly like the
Talker's relationship to the latent.

---

## 3. Why every non-obvious technical choice is justified

### 3.1 Sentence/clause chunk boundaries, not fixed token windows
A thought vector is supposed to be a semantically complete unit — the whole motivation for
chunking is that the Talker can reconstruct it faithfully and the Reasoner can condition on it as
a coherent proposition. A fixed-length window will sometimes bisect a clause mid-thought, forcing
one "thought" to encode half a proposition — reintroducing the exact compounding-fragility problem
chunking was meant to remove, just at the chunk level instead of the token level. Variable-length,
boundary-aware chunking costs variable compute per chunk, but this composes for free with adaptive
loop depth — both are already "spend more compute where it's needed."

### 3.2 The 3-fast : 1-slow L:H ratio is *not* principled
This is an empirical hyperparameter HRM-Text found to work for its own objective and data mix, not
a derived quantity. It's tempting to say "let ACT learn it too," but that conflates two different
quantities: ACT-style halting decides *total depth* — when to stop adding cycles — whereas the L:H
ratio is a structural choice about how the fast and slow updates interleave *within* that depth. A
single halting head gives you the former, not the latter. Making the ratio itself adaptive needs
its own mechanism — e.g. a separate halting/gate decision for "take another L-step" vs. "commit an
H-update," so the model can spend a variable number of fast steps between slow ones — not the same
signal that sets depth. So the honest status is: the ratio is unprincipled and *plausibly*
learnable, but it is not automatically subsumed by turning ACT on. Absent that extra gate the ratio
stays a fixed hyperparameter, which is exactly what Stage E does — ACT there varies the number of
cycles per thought while the L:H ratio inside each cycle remains fixed (§5.5).

### 3.3 Parcae's spectral-norm stabilization *alongside*, not instead of, MagicNorm
These fix different failure modes, and it's worth being precise about which mechanism supplies
which guarantee — because the update actually run is `h_{n+1} = A h_n + B·e + R(h_n, e)`, and a
spectral-norm constraint on `A` alone does **not** bound that map. `R` is an unconstrained
nonlinear sublayer, so a contractive linear part does nothing to stop the residual from growing
the state without limit as depth increases. Three complementary pieces, not two:

- **Boundedness at arbitrary depth comes from MagicNorm's hard normalization, not from Parcae.**
  Projecting the state back onto a fixed-norm shell (norm = √d) at the exit of every L- and H-step
  makes ‖h‖ depth-independent regardless of what `A` or `R` do — this is the guarantee that
  actually holds when the loop is run far deeper at test time than it ever was in training.
- **Parcae's spectral-norm constraint on `A` then shapes the dynamics on that shell.** Bounding the
  state only says the iterate stays in a compact set; it does not say the deep behavior is
  *well-behaved*. Constraining the linear part to be contractive (eigenvalues strictly inside the
  unit circle) makes the recurrence settle toward a fixed point rather than orbiting or wandering
  as depth grows — which is precisely what makes Parcae's "test-time compute scales as a
  predictable, saturating exponential" hold. Without it you could have a bounded but
  non-convergent loop, and "more iterations" would be noise rather than a monotone compute dial.
- **MagicNorm's PreNorm half is the training-time guarantee.** Its stability argument depends on the
  asymmetry between the forward horizon `N` and the truncated backward horizon `K` — module-level
  norms bound forward variance across all `N` steps while gradients only ever see `K` of them, so it
  behaves like stable PreNorm during optimization. This is a statement about **training dynamics
  under truncated BPTT**, distinct from either guarantee above.

So the three are not redundant: hard-norm bounds the forward state at any depth, Parcae's
constraint makes those bounded dynamics *converge* (which is what buys predictable test-time
scaling), and PreNorm keeps the truncated-BPTT gradient well-conditioned. The one framing to avoid
is the tempting shorthand "spectral-norm constraint → bounded forward dynamics at any depth": it
credits Parcae with a boundedness property the nonlinear residual actually voids, and it's
hard-norm — not `A` — doing that work.

### 3.4 Scaled cosine loss with k=4, and EMA momentum 0.98
The scaling factor exists because standard cosine distance yields insufficient gradients when the
loss is small — the same saturating-gradient problem that motivates label smoothing or
focal-loss-style rescaling elsewhere, not something specific to latent reasoning. The exact value
(k=4, chosen by sweeping 1–6) is a property of a particular embedding dimensionality and should be
re-tuned if model width changes, not assumed to transfer. The EMA momentum defends against a
known self-distillation failure mode (BYOL/DINO-style collapse, where predictor and target
converge to a trivial constant if the target moves too fast): momentum trades off target staleness
(too high) against collapse risk (too low), and 0.98 is whatever the source paper's sweep found for
its own setup — not a universal constant. In this design specifically, 0.98 turned out to be
*insufficient*: the shared-encoder/two-loss coupling collapses under it, and the implementation
raised momentum to 0.996 and added a separate projection head, a variance floor, and an always-on
reconstruction anchor (§2.4). Momentum alone is not enough here.

### 3.5 Warmup credit-assignment schedule (2 → 5 steps)
Early in training, the recurrent map is nowhere near a well-behaved fixed point, so
backpropagating through many steps of an effectively-random nonlinear recurrence gives noisy,
high-variance gradients — the classic reason recurrent models are hard to train from scratch with
full BPTT. Starting shallow lets the network first learn a recurrent map that's closer to
well-behaved (roughly contractive) before gradients are asked to travel through more of it. This
mirrors the logic behind implicit-differentiation / deep-equilibrium approaches that justify a
1-step gradient approximation once the map is near its fixed point — the warmup schedule is a
practical bridge from "assume near-fixed-point behavior" (invalid early) to "actually approximating
it" (valid later).

### 3.6 Un-detached memory with truncation, at the thought level
The memory can't be detached because that's Thought Gestalt's core point — relational /
reversal-curse fixes require gradient to reach back and reshape how *earlier* gestalts were
written, not just how the current one is read. But un-truncated BPTT through an entire
conversation's worth of thoughts is intractable and hits the same instability problem as the inner
loop. So the same asymmetric trick (long forward horizon, short backward horizon) is reapplied one
level up: unbounded forward reads from memory, bounded backward credit assignment into it.

One honest caveat about what the window actually bounds: it bounds *direct* credit — a loss at
thought *t* reads at most the trailing *k* memory slots with gradient. But slot *t−1*'s own
computation read slots *t−2…t−k−1* in-graph when it was written, so credit still reaches
arbitrarily far back **transitively**, attenuated per hop (measured ~30× per hop in the reference
implementation). Two consequences: the effective credit horizon is soft rather than hard, and the
autograd graph — hence activation memory — still spans the whole document whenever memory is
un-detached (Stages C+). The truncation makes distant credit *cheap and weak*, not absent; only
per-hop reach and gradient magnitude are bounded, not graph depth. Budget GPU memory accordingly
(the throughput bench runs with memory un-detached, so its peak-memory numbers reflect this).

### 3.7 Two losses, not one
Pure self-supervised latent prediction is cheap and parallelizable, but self-distillation
objectives are known to admit degenerate solutions that satisfy the training signal without being
useful downstream — the objective only constrains latents to be *self-consistent*, not
*language-groundable*. The grounded NLL loss (full Talker unroll) anchors the latents to actually
be decodable, but is expensive — hence it should run at lower frequency/weight as a regularizer
rather than replace the cheap loss. The interleave ratio is a real empirical trade with no
principled answer from either source paper, since neither paper combines the two.

**Correction from the implementation (§2.4):** calling the grounded loss "a regularizer to run at
lower frequency/weight" was backwards for the collapse problem. Because it is a *reconstruction*
objective, it is the one loss that cannot be satisfied by a degenerate latent, so it must be the
**always-on anchor**; it is the *self-supervised* loss that has to be the secondary, down-weighted
term. Thinning reconstruction below SSL (as the naive frequency-floor reading suggested) is exactly
what let the latent collapse. The expense of the grounded loss is real, but the resolution is a
separate SSL projection head so SSL cannot collapse the shared encoder — not demoting the anchor.

---

## 4. Chatbot-context refinements: input handling and the input/self boundary

### 4.1 Should the input be tokens or thoughts?

**Raw tokens, not thoughts — at least for the current turn's input.**

Thought-chunking exists to solve a specific problem: autoregressive *self*-generation compounds
errors, because the model conditions on its own sampled output (JEPA-Reasoner's core
justification — any localized token-selection error pollutes the context window and corrupts all
subsequent reasoning). That failure mode is intrinsic to self-generation. The user's input isn't
self-generated — it's a fixed, externally-given artifact. There is no compounding-error process to
contain, so chunking it into thoughts buys nothing on the problem it was designed for, while
costing something real: irreversible lossy compression of the one part of the context where
maximum fidelity is wanted (exact quotes, numbers, code, formatting, "fix line 47" references).

The input should go in as tokens, processed with full (bidirectional) attention — which is already
what HRM-Text's PrefixLM mask does (full bidirectional attention across instruction tokens,
causal generation for the response). Keep that split, generalized from "response tokens" to "the
Reasoner's thought-loop + Talker."

**Caveat — this doesn't mean unbounded raw tokens forever.** Unbounded full attention over an
entire long conversation history doesn't scale. There's a legitimate second tier: once input
content ages out of a recent window, compress it into a gestalt and write it into the memory bank
— reusing Thought Gestalt's compression mechanism, but repurposed. Thought Gestalt originally
compresses the model's *own* output sentence-by-sentence as it talks; here the same mechanism is
applied to *old input* for bounded-cost recall, while recent input stays raw. **Two-tier context:
raw tokens for recency/fidelity, gestalt summaries for long-range compressed recall.**

### 4.2 Separating input-processing from self-generation architecturally

This is a stronger claim than what HRM-Text's PrefixLM mask already provides, worth being precise
about the gap. PrefixLM separates input and output by *masking* — different attention patterns —
but both still flow through the same weights and, critically, both can end up written into the
same recurrent state (the H-module's `z_H`). That means there's no representational boundary
between "the user asserted X" and "I concluded X" — they can collapse into the same latent
subspace, which is exactly the substrate sycophancy could exploit: no structural way to represent
"I am tracking that you believe X" as distinct from "X is my belief."

**Proposal: two lanes feeding the memory bank, not one.**

- **Input lane**: raw tokens (this turn) plus aged-out gestalt summaries (prior turns), encoded by
  a stack that never writes into the Reasoner's H/L recurrent state directly — it only gets to be
  *cross-attended to*.
- **Self lane**: the H/L thought loop's own output — the only thing allowed to write into the
  recurrent state that represents "what I currently believe / am reasoning toward." This persists
  across turns (the gestalt memory doesn't reset per turn), so the model's own prior reasoning
  carries forward as *self*-content, while a fresh user turn always arrives through the input lane.

Every memory slot carries a **role tag** (USER / SELF, plausibly SYSTEM as a third), so at read
time — both for the Reasoner's inner loop and for the Talker — attention can learn
source-dependent weighting instead of being forced to blend everything into an undifferentiated
context. This is the mechanism that would let the model represent "user is pushing back and
asserting Y" without that assertion automatically becoming the content of the model's own next
thought.

**What happens to the model's own previous turns** on the next call: treat them as **self-lane**
content, not input-lane — rehydrate the persisted gestalt-memory state rather than re-encoding the
assistant's past turns as if they were external text. This is a nontrivial engineering commitment
(memory has to survive across calls, not just within one generation), but it's the only way the
self/input boundary stays coherent across a multi-turn conversation instead of resetting every
turn.

### 4.3 Two honest limits on the sycophancy claim

1. **Architecture is an affordance, not a guarantee.** Nothing stops the Reasoner's inner loop
   from learning to just copy the input-lane content into its own thought verbatim — the
   separation makes it *possible* to represent disagreement, it doesn't force the model to do so.
   Sycophancy is largely a training-signal problem (reward models that score agreement highly),
   and the architectural separation only pays off if the training objective actually exploits it —
   e.g., an auxiliary loss penalizing the Reasoner's thought-stream for being a high-similarity
   copy of the input gestalt when independent verification was possible, or contrastive examples
   during the grounded-loss phase where correct behavior requires the thought-stream to diverge
   from the input's framing. Without that, this is a nicer variable left unused.

2. **Fidelity cost is real**, and it's the same tension as §4.1: if the input lane only exposes
   gestalt *summaries* to the reasoning loop, token-level grounding needed for precise quoting or
   code edits is at risk. So the input lane needs **both** the raw-token cross-attention path (for
   the Talker, mainly — "repeat back exactly what they said") **and** the gestalt path (for the
   Reasoner's higher-level reasoning) — not a single compressed representation standing in for
   both.

---

## 5. Training curriculum

### 5.0 Why this can't be trained end-to-end from a random init

Every component in this architecture depends on some other component already being
"reasonably good" before it can receive a useful gradient:

- The **Talker** can't produce anything useful without good latents (JEPA-Reasoner's own
  ablation), but latents don't become *decodable* without Talker feedback (Thought Gestalt's
  point — pure self-supervision drifts toward self-consistent-but-inexpressible representations).
- The **inner HRM loop** can't be safely trained deep until it's already near a stable fixed
  point, because deep BPTT through an effectively-random recurrence is just noise (§3.5).
- The **gestalt memory's** ungated gradient is the same problem one level up: unrolling gradient
  through many past thoughts isn't safe before the *single-thought* loop is stable.
- The **EMA target encoder** needs to already produce meaningful representations, or the
  self-supervised loss is chasing a moving target that starts out as garbage.
- **Adaptive depth (ACT halting)** needs a loss signal that already reflects "more compute → a
  better thought" before the halting policy has anything sensible to learn from.

So nothing can be turned on simultaneously from scratch. Training has to be staged, with each
stage's stability as the precondition for turning on the next mechanism.

### 5.1 Stage A — Ground the Talker on a working (simple) latent

Strip out the loop, the memory, and both fancy losses. Use a fixed heuristic chunker
(off-the-shelf sentence/clause segmenter, not learned yet), a *shallow, fixed-depth* Reasoner
(one H-update, no L-iteration — essentially JEPA-Reasoner's own first, plain pretraining phase),
and train Talker + Reasoner jointly on ordinary next-chunk NLL.

**Goal**: get "latent → decodable text" working at all, cheaply, before anything recurrent or
adaptive is layered on. This mirrors why JEPA-Reasoner itself does ordinary pretraining before its
self-supervised phase — the initial competence of the Talker and the initial semantics of the
latent shouldn't be solved simultaneously from noise.

### 5.2 Stage B — Turn on the inner HRM loop, fixed depth, no cross-thought memory gradient

Introduce the L/H split (fixed 3:1 ratio, fixed number of cycles — no ACT yet) with Parcae's
spectral-norm-constrained recurrence and MagicNorm. Apply HRM's warmup deep-supervision schedule
(backprop through the last 2 steps, expanding to the last 5) here, since this is the first point
where there's a nontrivial recurrence to warm up. Memory writes can exist, but stay **detached**:
thought *t* sees thought *t-1*'s output only as a fixed input, no gradient flows back into it yet.

**Why detach memory here**: this isolates "is the inner loop stable and useful" from "is
cross-thought credit assignment stable," so a training failure is attributable to one mechanism,
not a tangle of two.

### 5.3 Stage C — Un-detach the gestalt memory, with its own truncation warmup

Apply Thought Gestalt's core trick: gradients from thought *t*'s loss are allowed to flow back
into the state that produced thought *t-k*. Start with a very small window (*k*=1, barely more
than Stage B) and expand it, mirroring the same "start shallow, deepen once stable" logic as the
inner-loop warmup — just one level up.

**Staggering, not simultaneity**: running two independent truncation warmups (inner-loop steps,
outer thought-memory window) at once is more moving parts than either source paper handles
individually. Deepen the inner-loop schedule first (it gates whether a single thought is even
trustworthy), then the outer memory-window schedule.

### 5.4 Stage D — Bring in the self-supervised JEPA loss, alongside (not before) the grounded loss

This reverses the phase order used in vanilla JEPA-Reasoner, deliberately. The obvious move would
be to introduce the cheap, parallel self-supervised loss *first*, since JEPA-Reasoner's own two
phases run in that order. But by this point the EMA target encoder is one more untrained
component, and pretraining it against a Reasoner whose latents don't yet have any grounded meaning
risks the model settling into a self-consistent-but-meaningless latent space early — which the
grounded loss then has to fight to undo later. JEPA-Reasoner's Reasoner has no separate
memory-bank/loop dynamics to destabilize in the way this design does, so the situations aren't
equivalent, and the safer order here is: let the grounded loss give the latents *some* linguistic
grounding first, then bring the self-supervised loss in and run both together from this point on.

### 5.5 Stage E — Turn on adaptive depth (ACT), and optionally a learned chunk boundary

Only once fixed-depth dynamics are stable does the halting policy have a meaningful signal ("would
more iterations have helped this specific thought") to learn from. Add a small ponder-cost penalty
so the model is pushed toward the cheapest depth that doesn't hurt the loss. This is also the
first point where Parcae's "predictable test-time scaling" claim actually starts to matter — before
this stage, depth isn't a free variable at all.

**Implementation caveat — "doesn't hurt the loss" needs a gradient path, and a hard halting
branch doesn't provide one.** If the halting decision is taken as a non-differentiable branch
(continue/stop on a thresholded probability), the task loss has *no* gradient into the halting
head: depth affects the loss only through discrete control flow, which gradients cannot see. The
ponder cost is then the head's *only* training signal, and it points one way — halt sooner — so
the learned policy provably degenerates to "always halt at the minimum depth" (verified in the
reference implementation: NLL gradient on the halting head is exactly zero). Getting a genuine
compute-vs-quality trade requires the task loss to see depth differentiably — a real ACT
accumulator (output = Σₖ p(halt=k)·h_k) or a REINFORCE-style estimator — which is the mechanism
this stage should eventually use; the soft expected-value ponder cost alone is not it.

### 5.6 Stage F — Chatbot fine-tuning: two-lane input/self separation, cross-turn persistence

Everything above can be trained on generic long-document text (chunk boundaries within a document,
no notion of "speaker"). The USER/SELF/SYSTEM role-tagging and the two-lane memory-write
restriction (§4.2) is a *fine-tuning* phase on multi-turn dialogue data, introduced only after the
base dynamics are already stable. Introducing a whole new structural constraint — some content may
only be cross-attended to, never written into self-state — at the same time as the recurrence
itself is still stabilizing would confound two separate sources of instability. This is also the
natural place to add the anti-sycophancy auxiliary loss / contrastive data flagged in §4.3, since
that loss only becomes measurable once the lanes exist to make "did the thought-stream just copy
the input lane" a well-defined, penalizable quantity.

One consequence of deferring the lanes to here is worth stating plainly: the input-lane encoder
(§4.2) is **cold-started at Stage F**. Nothing in Stages A–E exercises it — those stages have no
input/self split, and both losses operate purely on chunks — so it receives its first gradient only
once the Reasoner's inner loop and the Talker begin cross-attending into it under the grounded loss.
Stage F therefore has to bring an *untrained* encoder up to usefulness, not merely adapt an
already-good one, and needs enough dialogue data to do so. If that proves to be too little signal,
the fallback is to pretrain the input lane on raw text during Stages A–E with a plain bidirectional
denoising/MLM objective — kept strictly read-only w.r.t. the recurrent state throughout, per §4.2 —
so that Stage F only has to learn the *routing* (what to attend to, and when to diverge from it),
not the encoder from scratch.

### 5.7 Two infrastructure issues that follow from this staging

1. **Asymmetric training cost between the two losses.** The self-supervised loss is embarrassingly
   parallel across chunks (the EMA target for every chunk in a document can be computed in one
   pass, no sequential dependency). The grounded loss is inherently sequential — thought *t*
   depends on the finalized state from thought *t-1* through both the loop and the memory. This is
   a real asymmetry in wall-clock cost, not just an "interleave ratio" nicety: whatever ratio is
   chosen, the grounded loss will dominate training time per token touched. This pushes toward
   running it less often per token, but a **floor** on grounded-loss frequency should be enforced
   from Stage D onward rather than letting it taper arbitrarily low, since dropping too low is
   exactly the ungrounded-drift failure mode §3.7 warns about. *(In practice, §2.4: the floor should
   be high — reconstruction is the always-on anchor and should run every step; the empirical collapse
   happened precisely when reconstruction was thinned to a low floor while SSL ran every step. The
   compute cost is instead managed by keeping SSL — the cheap, parallel loss — as the frequent one
   and reconstruction as the always-on-but-not-the-only signal.)*

2. **Gating stage transitions on loss plateau, not fixed iteration counts.** HRM-Text's own
   "2→5 steps" schedule is presumably tuned to its own compute budget and iteration count. Here,
   two independent warmup schedules (inner-loop, outer-memory) are being stacked, plus a third
   effective one (when to introduce the self-supervised loss). Rather than fixed step counts
   copied from the source papers, stage transitions should be gated on a **validation-loss-plateau
   signal**, so the curriculum isn't silently miscalibrated if compute budget or model scale
   changes. This is a deliberate deviation from how the source papers describe their own
   schedules, worth flagging as such rather than presenting as settled.

---

## 6. Summary of open questions (no principled answer available from source material)

- Exact chunk-boundary policy: fixed budget vs. learned segmenter vs. punctuation heuristic, and
  how a learned boundary policy should interact with adaptive inner-loop depth.
- What exactly gets written to memory: raw H-state vs. a separately-projected "gestalt readout"
  (decoupling "state needed to keep computing" from "state useful for other thoughts to attend to").
- Interleave ratio and relative weighting between the self-supervised and grounded losses.
  *(Partially resolved in the implementation (§2.4): reconstruction must dominate as the always-on
  anchor and SSL must be secondary/down-weighted, or the latent collapses. The exact ratio and how
  it should shift with scale remain open.)*
- Whether the self-supervised and grounded losses should share projection heads on top of the
  shared latent, or use separate heads to reduce objective interference. *(Resolved toward
  **separate heads** (§2.4): a shared head collapsed the latent; a separate SSL projection head plus
  an always-on reconstruction anchor fixed it.)*
- Whether an explicit anti-sycophancy auxiliary loss (or contrastive training data) is added on
  top of the input/self lane separation, since the separation alone only creates the *possibility*
  of independent judgment, not the incentive for it.
