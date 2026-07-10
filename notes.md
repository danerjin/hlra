# Engineering Notes — Latent-Thought Reasoning Architecture

The full log of what was done to this project: every review finding, every bug
and its fix (with the evidence that confirmed it), every training run and its
numbers, and the theory/observations behind the decisions. The README is the
map; this is the story. Nothing is omitted.

---

## 0. Context & environment

- **Project**: a reference implementation of the design in
  `latent-thought-architecture.md` — combine JEPA-Reasoner, HRM-Text, Thought
  Gestalt, and Parcae into a model that thinks in chunk-level latent "thoughts",
  each decoded by a separate Talker, with a Parcae-stabilized HRM loop.
- **Canonical spec** lives at repo root (`../files/latent-thought-architecture.md`); Code + docs live in `files/`.
- **Interpreters**:
  - `/usr/bin/python3` → torch 2.1.2, **no** `datasets`/`transformers` (used for
    early static checks only).
  - `.venv/` (created for this work) → Python 3.9.6, **torch 2.2.2, datasets
    2.21.0, transformers 4.57.6, numpy 1.26.4, matplotlib 3.9.4**. MPS available.
    All real runs use this.
- **Caches, kept inside the project so it's self-contained**: `.hf_cache/`
  (HuggingFace datasets, `HF_HOME`), `gpt2_tok/` (gpt2 tokenizer files fetched
  manually — see §4.3), `runs/` (checkpoints, metrics, plots).
- **No large training run was ever performed.** Everything below is
  smoke/offline scale, on CPU or Apple MPS.

---

## 1. Review findings

### 1.1 Spec issues (design doc)

1. **§3.3 overstated the Parcae stability guarantee.** The doc claimed Parcae's
   spectral-norm constraint on `A` gives "bounded forward dynamics at any depth."
   False for the actual update `h_{n+1} = A h_n + B·e + R(h_n,e)`: `R` is an
   unconstrained nonlinear MLP, so a contractive `A` does not bound the map.
   What actually bounds the state at arbitrary depth is MagicNorm's
   **hard_normalize** (projects onto a fixed-norm shell each step). Parcae's `A`
   contributes *convergence* (contraction toward a fixed point → predictable
   test-time scaling), not boundedness. Three complementary guarantees, not two.

2. **§3.2 / §1.1 conflated ACT depth with the L:H ratio.** The doc said ACT
   should "learn the 3:1 L:H ratio." A halting head decides *total depth* (when
   to stop), not the *interleaving* of fast/slow updates. Those need separate
   mechanisms; ACT alone doesn't subsume the ratio.

3. **§2.1 / §4 input-lane cold-start gap.** The input lane only turns on at
   Stage F and is never exercised A–E (both losses are chunk-only), so it's
   cold-started — the doc didn't acknowledge this.

4. **§3.4 minor**: k=4 (cosine scale) is said to be tuned for a specific width,
   but the width is never stated, so "re-tune if width changes" isn't actionable.
   *Not fixed* — fixing it "properly" would mean inventing a width the source
   doesn't give, which is less honest, not more.

### 1.2 Code bugs — found by reading, then confirmed empirically

Confirmation used tiny targeted scripts (not full training). Verdicts:

- **C1 — Talker teacher-forcing off-by-one (critical).** `talker.py` was
  documented to take input "shifted right by caller," but `model.forward_grounded`
  passed `chunk_ids` as **both** the Talker input and the NLL target with no
  shift. With a causal mask that includes the diagonal, position *i* is fed token
  *i* and must predict token *i* → a trivial identity copy that drives NLL→0
  **without using the thought at all**, gutting the one loss meant to keep latents
  decodable.
  - *Evidence*: standalone Talker, thought held at **zero**, trained 2000 steps.
    Current wiring: train NLL **0.0000**, NLL on **unseen** tokens **0.87**
    (chance ln(48)=3.87) — it copies the input and generalizes the copy. Correct
    (shifted) wiring: unseen NLL **8.55** (can't cheat). Decisive.

- **C2 — the SSL branch was a disconnected island (critical).** The
  self-supervised loss used a *separate* `online_chunk_encoder` + a `Linear`
  predictor, none of which are used by `forward_grounded`. So SSL trained an
  encoder the Talker/reasoner never read.
  - *Evidence*: backprop of `forward_self_supervised`, tensors receiving gradient:
    `online_chunk_encoder` 28/28, `latent_predictor` 2/2, **`hrm_loop` 0/31,
    `reasoner_chunk_embed` 0/1, `talker` 0/48, `input_lane` 0/27**. The JEPA
    signal never reached the reasoning path.

- **C3 — input-lane recent-token slice grabbed padding → NaN (high).**
  `build_raw_lane_inputs` took `token_ids[:, -window:]`, but documents are
  *right*-padded, so the slice landed entirely in the PAD region. All-pad →
  `key_padding_mask` all-True → attention softmax over a fully-masked row → NaN,
  propagating through the HRM cross-attention in Stage F.
  - *Evidence*: seq_len 192, content 10, window 64 → 0 real tokens in the slice →
    input-lane output `NaN`.

- **C4 — ACT ponder cost inverted (high).** `hrm_loop` accumulated
  `ponder += halt_prob.mean()` and the loss penalized it. `halt_prob` is the prob
  of *stopping*, so penalizing it taught the model to **lower** halt prob → never
  halt → run to the cap. Backwards from ACT's intent.

- **M1** — the reasoner used an **order-free** masked-mean-pool of token
  embeddings, while the order-aware `ChunkEncoder` existed but was spent only on
  the disconnected SSL branch.
- **M2** — dead code `e = chunk_embed + h_state.detach()*0.0 + h_state` (middle
  term identically zero).
- **M3** — the grounded-frequency floor `max(min_freq, 0.5)` hardcoded 0.5,
  making the configurable `0.2` floor inert.
- **M4** — ACT halting is batch-coupled (`halt_prob.mean() > 0.5`), not
  per-thought as §1.1 wants.
- **M5** — `ChunkEncoder` (and the input lane) NaN on all-pad rows (same
  masked-softmax issue as C3).

---

## 2. Fixes applied

### 2.1 Spec fixes (`latent-thought-architecture.md`)

- **§3.3 rewritten** as three complementary guarantees: hard-norm →
  boundedness at any depth; Parcae's `A` → convergence (hence predictable
  test-time scaling); PreNorm → training-time conditioning under truncated BPTT.
  Flagged the exact wrong shorthand to avoid.
- **§3.2 / §1.1** — un-conflated: ACT sets total depth; the L:H ratio needs its
  own gate; noted Stage E's ACT varies cycle count while holding the ratio fixed
  (consistent with the code).
- **§5.6** — added the input-lane cold-start caveat + a fallback (pretrain it
  with a bidirectional denoising objective during A–E, kept read-only).
- Left the §3.4 k-width nit alone (see §1.1.4 above).

### 2.2 Code fixes (with verification numbers)

- **C1** — Talker now shifts teacher forcing internally: a learned start vector
  at position 0, then `tokens[:-1]`, so position *i* only ever sees tokens `< i`.
  *Verified*: zero-thought unseen NLL **9.05** (chance 3.87) — can no longer
  copy; output now depends on the thought (mean |Δlogit| between two random
  thoughts = 0.30).
- **C2 + M1** — introduced **one shared `chunk_encoder`** (the order-aware
  transformer) used by *both* the reasoner (HRM injection) and the SSL branch;
  deleted `online_chunk_encoder` and the bag-of-words `reasoner_chunk_embed`.
  *Verified*: SSL gradient reaches `chunk_encoder` 28/28 and `latent_predictor`
  2/2, and the grounded path also trains `chunk_encoder` 28/28 (genuinely
  shared, not an island).
- **C3 + M5** — `chunker.encode_recent` takes the last *non-pad* tokens; added
  all-masked-row guards in `input_lane.py` and `ema_target.ChunkEncoder`.
  *Verified*: real tokens captured, finite output, all-pad row guarded.
- **C4** — ponder now penalizes *continuing* (`1 − halt_prob`). *Verified*:
  ponder(halt≈1)=**0.013** < ponder(halt≈0)=**5.96**.
- **M2** removed; **M3** honors the configured floor.
- *Integration*: all six stage-flag configs run finite forward+backward; the
  full A→F harness runs on synthetic data with decreasing loss.

---

## 3. Real-text data pipeline

Rewrote `data.py` and `config.py` from a synthetic *integer* toy corpus to a
real-text pipeline:

- **Chunking moved into the data pipeline** (not the training loop) — enables
  length bucketing by chunk count and moves SaT cost off the hot path.
- **Two tiers behind one interface**: real (`iter_hf_mixture` streams the
  weighted `DataConfig` mixture, real SaT chunker) and offline (synthetic *text*
  + stub `RegexSentenceSegmenter` / `WhitespaceStubTokenizer`, running the exact
  same SaT-Capped code path with no downloads).
- **`ReservePadTokenizer`** — offsets real tokenizer ids by +1 so id 0 stays
  reserved for PAD. Critical: gpt2's native id 0 is a real token (`"!"`), which
  would silently corrupt the model's `id != 0` pad mask.
- **Real chunk sizes**: `max_chunk_len` 16→64 (paper L=64), `max_chunks_per_doc`
  12→32, `recent_token_window` 64→128.
- `train.py` now consumes pre-chunked batches, defaults offline, `LATENT_USE_HF=1`
  opts into the real mixture; device-aware.
- *Verified offline*: correct shapes, PAD-only padding, real ids in `[1,vocab)`,
  `min_chunks` bucketing enforced, full A→F finite.

---

## 4. Small real run + graphs + inference

### 4.1 Dataset choice

Requirement: real prose, **whole-document** examples (long enough to exercise the
gestalt memory), small download, only `datasets` needed. Checked via HF API:
- **`NeelNanda/pile-10k`** ✓ — 10k diverse Pile docs, ~3k tokens each, **33 MB**,
  ungated, `text` column. Chosen.
- `stas/openwebtext-10k` — datasets-server couldn't parse (legacy loading
  script); skipped.
- `roneneldan/TinyStories` — 1 GB total, docs too short (~175 tokens); skipped.

### 4.2 The two bugs the run surfaced

1. **Streaming was the bottleneck.** Streaming pile-10k row-by-row = **57.5 s
   for 8 docs (~7 s/doc)** — at that rate the *data* alone would take ~45 min.
   Fix: `streaming=False` (one 33 MB download, then in-memory iteration).
2. **`float("-inf")` causal mask NaNs on MPS.** The first run produced `nan`
   from Stage A step 1 on MPS. Diagnosed by comparing devices on the same batch:
   **CPU finite (nll 10.48), MPS NaN**, while `chunk_encoder` was finite on both →
   the NaN was downstream, in the Talker's `-inf` causal mask (a known MPS
   footgun). Fix: **boolean causal mask** (`talker.py`), backend-safe. Verified
   MPS then matches CPU (10.48), and the C1 leak-fix still holds. Also learned:
   **MPS is no faster than CPU here** (~1.0 s/step both) — many tiny ops, MPS
   dispatch overhead cancels the gain.

### 4.3 gpt2 tokenizer download workaround

For decodable output we need the real gpt2 tokenizer (the stub hashes words and
can't be inverted). `transformers`' Hub HEAD/etag check kept timing out (10 s,
flaky) even though `curl` to the Hub worked. Fix: `curl` the five tokenizer files
(`vocab.json`, `merges.txt`, `tokenizer.json`, `tokenizer_config.json`,
`config.json`) into `gpt2_tok/`, load from that local dir with
`TRANSFORMERS_OFFLINE=1`. Verified round-trip through `ReservePadTokenizer`
(vocab 50258, no id 0 in real ids).

### 4.4 Results (`train_real.py`, gpt2 tokenizer, ~1.5M tokens, MPS)

Chance NLL = ln(50258) = **10.82**. This run (pre-collapse-fix) walked A→F:

| stage | step | grounded val_loss | SSL |
|---|---|---|---|
| A | 10 | 10.58 | — |
| C | 120 | **7.77** | — |
| D | 130 | 8.48 | 0.86 |
| D | 160 | 7.87 | 0.10 |
| E | 200 | **8.26** | **0.016** |

→ Val dropped to 7.77 through C, then **regressed to ~8.26 at D/E** as the SSL
loss **collapsed** (SSL 0.86 → 0.016; cosine 0.996). Reconstruction perplexity
(via `generate.py --score`) ≈ **3360**. This regression is what §5 fixes.

### 4.5 Graphs & inference

- `plot_metrics.py` → `runs/loss_curves.png`: val loss (top), train NLL + SSL
  (middle), and — after the fix — the `latent_std` collapse monitor (bottom),
  all with stage bands.
- `generate.py`: tokenize prompt → read it through the HRM loop (build memory +
  running thought) → predict next latent (JEPA head) → Talker autoregressively
  decodes → gpt2 detokenizes. Output is **real subword text but incoherent** —
  expected at this scale. `--score` gives teacher-forced perplexity.
- **Honest bar**: this is not gpt2-quality and can't be (ppl ~3k vs random 50k;
  gpt2 is 124M params over ~10B tokens with real BPE). It exists to exercise the
  architecture and feed `generate.py`. The checkpoint (`runs/model.pt`, ~172 MB —
  vocab×d_model embeddings dominate) is saved regardless because the inference
  script needs it. Aside: perplexity depends on prompt length (a short prompt
  scored ~19.7k vs ~3.1k for a longer one — less context, higher per-token NLL).

---

## 5. The SSL collapse — full theory and fix

The centerpiece finding. See design-doc §2.4 for the spec-level writeup.

### 5.1 The role of the chunk encoder

A "thought" is a chunk-level latent. The chunk encoder is the **only** map from a
chunk's tokens → that latent. After the C2 fix it is **shared** across three
consumers: (1) the reasoner's front door (its output is the HRM injection, so
*everything generated flows through it*), (2) the SSL loss's input space, (3)
via EMA copy, the SSL target. That confluence is the whole reason a problem in
the SSL loss doesn't stay contained.

### 5.2 Why the SSL loss collapses

SSL: `pred = predictor(encoder(chunk_t))`, `target = EMA_encoder(chunk_{t+1})`,
`loss = k·(1 − cos(pred, target))`. The **easiest** way to drive this to 0 is not
"predict the future well" — it's to make the encoder output the **same vector for
every chunk**. Then pred∥target always, cos=1, loss=0, forever, while the latent
carries zero information. This is BYOL/DINO/JEPA representational collapse; the
usual defenses (stop-grad target, EMA momentum, predictor asymmetry) are what we
had — and at this scale (tiny model, 1M tokens, momentum 0.98) they weren't
enough (cosine reached 0.996).

### 5.3 Why it's severe *here* — four compounding axes

1. **Propagating** — shared encoder → collapse flattens the representation the
   Talker and loop depend on, not just SSL. (This is why grounded val *rose* at
   Stage D.)
2. **Silent** — SSL→0 *looks* like success; only a separate reconstruction
   signal reveals the damage.
3. **Absorbing** — the EMA target is a copy of the collapsing encoder, so
   "predict a constant from a constant" is a stable fixed point with no gradient
   to escape it.
4. **Schedule-amplified** — our curriculum thinned grounded to a 0.2 frequency
   floor while SSL ran every step at weight 1.0, right after Stages A–C had made
   the encoder good. We handed the collapse-prone loss the wheel at the worst
   moment.

### 5.4 The user's insight: reconstruction is the anti-collapse anchor

The grounded loss **is** an autoencoder: encode chunk → HRM → Talker decodes the
*same* chunk → NLL. Reconstruction **cannot** be satisfied by a constant latent
(a constant can't reconstruct varied chunks) — this is an information lower
bound. That's precisely why Stages A–C (grounded only) never collapsed, and it's
the load-bearing fix: keep reconstruction the always-on anchor.

Caveat noted: because the Talker is autoregressive (post-C1), *some* tokens
reconstruct from the prefix, so the anti-collapse pressure on the latent is real
but partial. A latent-only reconstruction term (e.g. bag-of-tokens from the
latent) would be a *hard* guarantee; we used a variance floor instead (below).

### 5.5 The fix (now the default)

1. **Reconstruction always-on** (grounded frequency 1.0 from Stage D) — the
   anchor.
2. **Separate SSL projection head** (`model.ssl_proj`, with its own EMA copy in
   `ema_target.py`, BYOL-style) — SSL can only collapse *its own* head, not the
   shared encoder. (Resolves design-doc §6's shared-vs-separate-head question
   toward **separate**.)
3. **SSL demoted** — cosine weight 0.1; EMA momentum **0.98 → 0.996** (slower
   target is harder to chase into a constant).
4. **Variance safety floor** (`losses.variance_regularization`, VICReg-style
   hinge on the shared latent's per-dim std) — dormant in normal operation,
   active only near collapse.
5. **`latent_std` collapse monitor** logged every eval.

### 5.6 The two wrong turns while fixing it (kept as lessons)

- **Variance floor overshoot.** First attempt used `target_std=1.0`, which forced
  the latent's per-dim std from its natural ~0.25 up to ~1.0 — a 4× rescale that
  *disrupted the Talker* and made val look worse. Lesson: **an anti-collapse
  regularizer is a floor, not a target** — it must never drive the scale. Fixed
  to `target_std=0.1` (below natural scale → dormant).
- **Contaminated eval metric.** `evaluate()` added the SSL term at *full* weight
  (1.0) while training weighted it 0.1, so val jumped at Stage D as a
  *measurement artifact*, not real regression. Lesson: **val must be
  reconstruction-only** and comparable across the stage boundary. Fixed.

### 5.7 Before/after (both ~1.5M tokens, gpt2 tokenizer, MPS)

| metric | BROKEN (naive) | FIXED |
|---|---|---|
| val_loss end of C | 7.77 | 7.84 |
| val_loss D/E | **7.77 → 8.26 (regressed)** | **~7.75–7.86 (held flat)** |
| SSL loss at E | **0.016 (collapsed)** | ~0.02–0.14 (stable, secondary) |
| latent cosine / std | cos 0.996 (≈collapse) | latent std ~0.24–0.56 (healthy, floor 0.1 never touched) |
| reconstruction ppl | ~3360 | ~3136 |

The point was never a perplexity win — it was **removing the pathology**: no
collapse, no regression, and a monitor to catch any recurrence.

---

## 6. Scaling infrastructure

The smoke path chunks on-the-fly every epoch, loads single-process, checkpoints
only at the end, and gates the curriculum with a forced-plateau hack — none of
which scale. Added (no training run — verified offline):

- **`data_prep.py`** — offline pre-chunking → sharded int32 tensors + manifest.
  Chunking/tokenization paid **once**; training does zero chunking.
- **`data.CachedChunkDataset`** — map-style over the shard cache → DataLoader
  workers + shuffling. Asserts cache chunk-dims match the model. (Loads shards
  into RAM; memmap is the next step for very large corpora — format unchanged.)
- **`trainer.Trainer`** — AMP autocast (bf16/fp16 + GradScaler on CUDA fp16),
  gradient accumulation, warmup→cosine LR schedule, periodic checkpoint/resume
  (model + optimizer + EMA + curriculum + step), fixed per-stage step-budget
  gating, and the `latent_std` monitor.
- **`train_scaled.py`** — entry: preset + cache + Trainer + `--resume`. Needs no
  tokenizer at train time (data pre-chunked).
- **Size presets** (`config.MODEL_PRESETS`): `smoke` (192d, current), `small`
  (512d, ~100M+ params), `base` (768d). Configurable `chunk_encoder_layers`.
- **Checkpoint state** added to `EMATargetEncoder` and `Curriculum`.
- *Verified offline* (tiny synthetic cache, CPU): prepare → train → checkpoint →
  resume (continued from a mid-run checkpoint) → multi-worker (`num_workers=2`).
  Fixed-budget A→F gating, LR warmup (6e-6→4.8e-5), `latent_std` ~0.68 all
  observed working.
- Minor bugs fixed while building this: a garbled import line in `data_prep.py`,
  a docstring that broke `ema_target.py` syntax, and `str | None` annotations
  needing `from __future__ import annotations` on Python 3.9.

---

## 7. Consolidated theory notes

- **Reconstruction vs. self-distillation.** Reconstruction (autoencoder) cannot
  collapse — it lower-bounds mutual information between input and latent.
  Contrastive/self-distillation objectives *can* — a constant satisfies them.
  When both act on one shared encoder, reconstruction must be the anchor.
- **Shared encoder = shared fate.** Connecting SSL to the reasoning path (the C2
  fix) was correct, but it means SSL collapse propagates. The resolution is
  isolation (separate head), not disconnection.
- **Parcae vs. MagicNorm division of labor** (corrected §3.3): hard-norm bounds
  the state at any depth; Parcae's contraction shapes the dynamics (→ predictable
  test-time scaling); PreNorm handles truncated-BPTT training stability. A
  spectral-norm constraint alone does **not** bound a map with an unconstrained
  nonlinear residual.
- **ACT sets depth, not the L:H ratio** — different mechanisms.
- **The input lane is cold-started at Stage F** — nothing A–E trains it.
- **Autoregressive decoders dilute reconstruction's anti-collapse pressure** —
  the prefix can carry some of the reconstruction, so the latent isn't strictly
  forced to hold everything. A latent-only reconstruction term is the hard
  guarantee if ever needed.
- **A regularizer is a floor, not a target.** (Variance-floor overshoot lesson.)
- **Watch the metric, not just the loss.** A contaminated eval metric faked a
  regression; the loss the model minimizes can be the one being gamed (SSL→0).
- **Isolating a loss into its own projection space can amputate a capability.**
  The §5 separate-head fix quietly removed the only trained encoder-space
  next-latent map — exactly what free-running generation needs. The resolution
  mirrors the Talker pattern: a dedicated head trained as a pure readout
  (input AND target detached), so it restores the capability without
  reintroducing the collapse pressure (§15.1).
- **A hard halting branch gives the task loss no gradient into the halting
  head.** Depth chosen via a thresholded `.item()` is invisible to autograd, so
  the ponder cost is the head's only signal and the policy degenerates to
  always-halt-at-minimum-depth. A compute-vs-quality trade needs the loss to
  see depth differentiably (ACT accumulator) or via REINFORCE (§15.5).
- **Truncation windows bound *direct* credit, not *transitive* credit.** Slot
  t−1's graph contains its own in-graph reads of t−2…, so credit chains
  arbitrarily far back (attenuating ~30×/hop) and the activation graph spans
  the whole document whenever memory is un-detached (Stages C+) (§15.5).

## 8. Consolidated empirical/infra notes

- MPS: `float("-inf")` attention masks NaN → use boolean masks. MPS is **not**
  faster than CPU for many small ops (dispatch overhead).
- Streaming a small dataset row-by-row can be far slower (~7 s/doc) than a single
  full download — download once for anything under a few hundred MB.
- gpt2 token id 0 is a real token → reserve PAD by offsetting ids +1
  (`ReservePadTokenizer`).
- `transformers` Hub HEAD/etag checks can hang on flaky networks → fetch
  tokenizer files locally + `TRANSFORMERS_OFFLINE=1`.
- Checkpoints are dominated by vocab×d_model embedding tables (~172 MB at gpt2
  vocab, d_model 192).
- Perplexity is prompt-length sensitive; compare on fixed text.
- Everything is pinned inside the project (`.venv`, `.hf_cache`, `gpt2_tok`) so
  the setup is reproducible and self-contained.
- **Checkpoint writes must be atomic** (write `*.tmp`, then `os.replace`) — a
  crash mid-save must never destroy the only checkpoint of a long run (§15.4).
- **Save curriculum state AFTER `advance_step`**, or resume replays one extra
  step-in-stage and the per-stage LR schedule drifts off by one (§15.3).
- **Pre-truncate documents before chunking** to a char budget covering what's
  kept — otherwise long docs are segmented/tokenized in full for ~2k kept
  tokens, and the input-lane window lands on text disjoint from the kept
  chunks (§15.4).
- **Val split by seeded permutation, not a head slice** — caches preserve
  corpus order, so "first N docs" can be topically clustered (§15.4).
- **`hard_normalize` makes ‖h‖ constant — a `h.pow(2).mean()` probe loss has
  ~zero gradient.** Gradient-flow tests on normalized states need a
  direction-sensitive loss (e.g. `<h, v>` with random v); learned during the
  §15.2 verification.

## 9. Open items / not done / next

- **No large training run.** The obvious next step: real `data_prep` (100M+
  tokens) + a short `train_scaled --preset small` shakedown on a GPU box to
  confirm throughput and that `latent_std` stays healthy at scale, *before* a
  long run.
- **AMP untested** — implemented, default off; sanity-check on the first CUDA run.
- **`CachedChunkDataset` is in-RAM** — switch to memmap for very large corpora.
- **Real SaT chunker** — `data_prep`'s real path uses regex+gpt2 (no download);
  swap in `build_sat_chunker` (needs `wtpsplit`) for true SaT boundaries.
- **Stage F is synthetic** — real multi-turn dialogue + the anti-sycophancy
  contrastive data (§4.3) are deferred; the two-lane machinery is wired but
  cold-started at F.
- **ACT is per-batch, soft** (M4) — **and one-sided** (§15.5): the halting head
  gets gradient only from the ponder cost, so the Stage-E policy degenerates to
  always-halt-at-minimum-depth. Per-thought adaptive depth that actually trades
  compute against quality needs a real ACT accumulator (output = Σₖ
  p(halt=k)·h_k) or REINFORCE — one change fixes both. Deliberately not landed
  before the big run.
- **`data_prep.py` is single-dataset and single-process** (§15.6) — the
  DataConfig mixture (`iter_hf_mixture`) is not wired into offline prep, and
  pile-10k (~20M usable tokens) cannot feed the ~1.2B budget; pick a big corpus
  (e.g. fineweb-edu `sample-10BT`) or wire the mixture in, and time a 1k-doc
  dry run.
- **Weight-decay param groups** — AdamW currently decays everything, including
  LayerNorms, embeddings, and Parcae's `theta`/`log_dt` (a mild pull toward
  decay ≈ 0.62). Conventional no-decay groups recommended, but it changes
  optimizer state/dynamics, so not applied pre-run.
- **Talker batching across chunks** — the main throughput lever if large-batch
  isn't enough (STRIX_HALO.md §4); must be tested before landing.
- **Anti-collapse hyperparameters** (cos weight 0.1, var weight 2.0, var floor
  0.1, momentum 0.996) were tuned on the smoke run — **re-tune at scale**.
- **k=4 cosine scale** should be re-tuned per model width (§3.4).
- **Model sizing** for a real run is unsettled (`small`/`base` presets are
  starting points, not validated).

---

## 10. Key numbers, one place

- Chance NLL, gpt2 vocab: ln(50258) = **10.82**.
- C1 confirm: zero-thought unseen NLL, broken **0.87** vs chance 3.87; fixed **9.05**.
- C2 confirm: SSL grad to reasoning path, broken **0/107** tensors; fixed shared encoder **28/28**.
- C4 confirm: ponder(halt≈1) **0.013** < ponder(halt≈0) **5.96**.
- Broken run: val **7.77 (C) → 8.26 (D/E)**; SSL **0.86 → 0.016**; cos **0.996**; ppl **~3360**.
- Fixed run: val **7.84 (C) → ~7.8 (D/E, flat)**; latent std **~0.24–0.56**; ppl **~3136**.
- Streaming pile-10k: **~7 s/doc**; compute **~1.0 s/step** (CPU ≈ MPS).
- Env: `.venv` torch **2.2.2**, datasets **2.21.0**, transformers **4.57.6**, numpy **1.26.4**, matplotlib **3.9.4**.
- Small-preset shakedown (MPS): A→E in ~12 min, **~33 s/step** (batch 4); latent_std **0.48–0.67** across the D boundary (§13.1).
- Wiki-page overfit, grounded-only: reconstruction **59,013 → 1.0 ppl** — the architecture *does* memorize (§13.2).
- Fair-scale baselines, same page: GPT same-params (44.7M) **ppl 1.1** (verbatim); GPT same-compute (14.1M) **484**; latent grounded-only **1.0** (§13.3).
- Chinchilla-equivalent for `small` (153M): compute-active N **75.4M → ~1.2B tokens** (naive-total would say 3.06B) (§14.1).
- §15 halting-head gradient: from NLL **0** (exactly), from ponder **0.27** — one-sided, Stage E degenerates to min depth (§15.5).
- §15 transitive memory credit: window 2, thought-9 loss → thought-8 grad **7.3e-4**, thought-0 **2.5e-8** (~30×/hop; graph spans the doc) (§15.5).
- §15 M11 confirm: cross-thought leak at grad_window 8/12 **6.5e-8** pre-fix, **0** at 5; post-fix **0 at every window**, forwards bit-identical (§15.2).
- §15 M12 confirm: resumed vs uninterrupted per-stage LR at same step **1.23e-4 vs 5.58e-5** pre-fix; identical post-fix, final val 8.3104 vs 8.3108 (§15.3).
- §15 gen head: isolation **4/4** own tensors with grad, **0** elsewhere (both directions); gen loss **2.97 → 1.78** over the 24-step toy (§15.1).
- pile-10k usable tokens after the 32×64 cap: **~20M** — vs the ~1.2B budget (§15.6).

---

## 11. Second pre-scale review (2026-07-08) — before the first big run

A fresh read of spec + code with targeted empirical checks, done specifically
because the big training run was about to start. One critical bug and several
medium issues found and fixed; all verified.

### 11.1 C5 — inner-loop gradient truncation was a **no-op** (critical)

`hrm_loop.forward` recorded every L/H state in a `step_history` list and, after
the loop finished, replaced older entries with `.detach()`ed copies
(`utils.truncate_gradient_window`), then re-bound `h_state = step_history[-1]`.
**Detaching recorded tensors after the fact does not cut the graph**: the final
state was computed from the *originals*, so its autograd graph still reached
back through every step. (The same helper is *valid* for the memory bank,
because there all future reads consume the detached list.)

- *Evidence*: 2 cycles (8 steps), backward from the returned thought — L-module
  / H-module / chunk-embedding gradients **byte-identical for
  grad_window = 1, 2, 5, and 8**. Full BPTT, always; §3.5's 2→5 warmup never
  happened. Worse, the raw `h/l` state chain carried across chunks
  (`l_state = h_state` in `model.forward_grounded`) was also never cut, so the
  *whole document* was one BPTT graph (up to 32 chunks × 8 steps at the
  `small` preset) — bypassing both truncation windows and exactly the
  "full-sequence BPTT" §2.3/§3.6 exist to avoid.
- *Fix*: cut the **carried** `(h, l)` states *during* the loop
  (`hrm_loop._TruncationSchedule`): fixed depth → one exact cut `window` steps
  before the end (HRM's "backprop through only the final K steps", exactly);
  ACT (depth unknown) → rolling cut every `window` steps (backward horizon
  ≤ window). Because the entering states sit before the cut, the raw
  cross-thought chain is severed each thought, leaving the gestalt memory —
  with its own §3.6 window — as the *only* cross-thought gradient path, as
  designed. `grad_window <= 0` detaches the returned thought entirely (old
  behavior).
- *Verified*: window=1 → L-module grad **0** (only the final H-step in graph),
  monotonically growing gradient with window, window=8 reproduces the old
  full-BPTT numbers exactly; forward values **identical** for every window;
  with detached memory, chunk *t-1* gets **zero** gradient from thought *t*
  (chain cut) but **nonzero** through un-detached memory (the intended path);
  ACT mode finite with halting-head gradient flowing.
- *Consequence for scale*: activation memory per document no longer grows as
  one full-doc graph, and Stage B's warmup now actually protects the fresh
  recurrence. Note the smoke baselines (§5.7) were trained under accidental
  full BPTT, so per-stage loss curves may shift slightly on the next run —
  that is the *corrected* behavior, not a regression.

### 11.2 M6 — validation metric included the ACT ponder cost

`evaluate()` (both `train.py` and `trainer.py`) returned `nll + ponder` while
claiming to be "reconstruction only". Ponder is a training-time compute
penalty, not a quality signal — including it bumps val at the Stage-E boundary,
repeating the §5.6 contaminated-eval lesson. **Fixed**: val = reconstruction
NLL only.

### 11.3 M7 — ponder cost scaled with document length

`forward_grounded` summed ponder over chunks but averaged NLL over chunks, so
the effective ACT penalty varied ~8× between a `min_chunks=4` doc and a
32-chunk doc, and would have been ~2.7× stronger at the `small` preset than in
the smoke run that tuned it. **Fixed**: ponder is now also a per-thought mean.
⚠ `act_ponder_cost=0.01` was tuned against the summed form — if Stage E now
halts too late, scale it up (an ~order-of-magnitude bump reproduces the old
per-thought pressure at smoke chunk counts).

### 11.4 M8 — warmup windows ignored the fixed stage budgets

`Curriculum.stage_flags()` ramped the 2→5 inner and 1→5 memory windows over
`max_steps_per_stage` (default 200) even when `stage_steps` budgets (e.g.
2000/stage in `train_scaled.py`) drive the curriculum — the warmup would have
finished in the first 10% of Stage B. **Fixed**: the ramp horizon is the
current stage's own budget when `stage_steps` is set
(`Curriculum._stage_budget`).

### 11.5 M9 — input-lane cross-attention attended to pad slots

`hrm_loop.input_reader` had no `key_padding_mask`, so the Reasoner attended to
the input lane's pad-position outputs (contextualized noise). **Fixed**:
`InputLaneEncoder.forward` now returns `(kv, mask)` and the HRM read masks pad
slots (with the same all-masked-row NaN guard as everywhere else). Stage-F-only
path; verified in the A→F integration run.

### 11.6 M10 — checkpoints didn't actually save RNG state

`trainer.save`'s docstring promised RNG in the checkpoint; it wasn't there.
**Fixed**: python/numpy/torch/cuda RNG states saved and restored on resume
(older checkpoints without the key still load).

### 11.7 Optimization — batched chunk encoding in the grounded path

Chunk latents don't depend on loop state, so `forward_grounded` now encodes
all chunks of the batch in **one** `chunk_encoder` call up front instead of
once per chunk inside the sequential loop (identical math, ~`n_chunks`× fewer
encoder launches; matters at 32 chunks/doc).

### 11.8 Re-verification after the changes

- Gradient-reach audit (the C2 methodology): Stage C grounded → `chunk_encoder`
  28/28, `talker` 49/49, `hrm_loop` 23/31 (the 8 without grad = halting head +
  input-lane reader, correctly inactive pre-E/F); Stage D adds `ssl_proj` 4/4 +
  `latent_predictor` 2/2; Stage F reaches **31/31** + `input_lane` 27/27 +
  halting head 2/2. All losses finite.
- Offline A→F integration: all six stages run, loss decreases, Stage E val now
  comparable across the D/E boundary (no ponder contamination).
- Scaled path: `data_prep --offline` → `train_scaled` (workers=2, fixed
  budgets, mid-run checkpoint) → `--resume` from step 16 continues through E
  with val within noise of the uninterrupted run.
- `generate.py` (generation + `--score`) still works against the existing
  `runs/model.pt` checkpoint (no state-dict changes).

### 11.9 Known gaps left open (deliberately)

- The **Talker still doesn't cross-attend to the input lane** (§4.3.2 wants the
  raw-token path "for the Talker, mainly"); Stage-F concern, deferred with the
  rest of Stage F's real-data work.
- When both losses run, the batch is chunk-encoded twice (once in
  `forward_grounded`, once in `forward_self_supervised`) — a possible future
  dedup, left alone to keep the two loss paths independent.
- ACT remains per-batch (M4), and in mixed batches finished documents' rows
  still contribute to the batch-mean halt probability.
- `CachedChunkDataset` remains in-RAM (memmap when corpora get big).

---

## 12. Investigating the "D/E regresses reconstruction" claim (2026-07-09)

Prompted by a full-curriculum overfit run on one Wikipedia page (~5.8k gpt2
tokens, smoke preset) where held-out val rose across Stages D/E (7.3 -> 7.9),
looking like SSL/ACT were damaging reconstruction. Dug in with a controlled
ablation; **the SSL/ACT hypothesis was largely wrong.**

### 12.1 First ablation was contaminated (a methodology lesson)
Branched four arms (control / +SSL / +ACT / +both) from a common grounded-only
init, each with a **fresh** AdamW. The *control itself diverged* (ppl
458 -> 14000), which is impossible if grounded-only is stable (it drives ppl ->
1.0 given a continuous run). Cause: a fresh optimizer with zero second-moment
estimates takes huge first steps at lr 3e-4 on an already-trained model.
**Lesson: to model a curriculum stage transition you must carry the optimizer
state** (as `Trainer` does -- one optimizer across all stages). Re-ran carrying
`opt.state_dict()`.

### 12.2 Corrected ablation (held LR 3e-4, continuous optimizer, common init)
All arms *improve* smoothly from ppl 458 over 300 steps:

| arm | final recon ppl | vs control |
|---|---|---|
| control (grounded only) | **11.6** | -- |
| +SSL (Stage D) | 14.0 | ~20% worse |
| +ACT (Stage E) | 12.0 | ~neutral |
| +SSL +ACT | 14.6 | ~25% worse |
| +SSL, cos_w 0.02 | 12.4 | penalty halved |
| +SSL, encoder-detached | 11.9 | penalty gone |

Findings: **ACT is innocent** (12.0 ≈ 11.6, and ACT-mode eval matches
fixed-depth). **SSL costs a small ~20%**, and it is exactly the shared-encoder
coupling: `forward_self_supervised` does *not* stop-grad `chunk_encoder`, so the
cosine term trains the shared encoder that reconstruction decodes from.
Stop-grad'ing it (or cos_w 0.02) removes the penalty. Neither loss produces the
big "regression".

### 12.3 The actual cause: LR starvation + overfitting confound
The dramatic D/E rise was two artifacts, not model damage:
1. **Global cosine LR starves late stages.** One cosine over the whole A..E
   horizon means D/E run at the tail. Measured on the wiki curriculum: Stage E
   saw lr **5.5e-5** (global) vs the intended **3.0e-4** (per-stage) -- ~5x
   starved, so D/E couldn't keep learning.
2. **Held-out val on 8 paragraphs is overfitting-dominated.** On a 45-paragraph
   train set the model memorizes; held-out val rising is overfitting, not a
   reconstruction regression. Full-page (train) reconstruction never regressed
   under a held LR.

### 12.4 The fix (implemented, opt-in)
`trainer._lr` + `TrainConfig.per_stage_lr` + `train_scaled --lr-schedule`:
a **per-stage warmup->cosine over each stage's own budget**, so every stage
starts with a usable LR and still anneals. Default is `per-stage` on the scaled
path; `--lr-schedule global` reverts. Verified it delivers 3e-4 to D/E (vs
5.5e-5). **Honest caveat: on this toy the fix is ~neutral** (final recon ppl
407 global vs 425 per-stage -- within noise), because reconstruction quality
here is dominated by total grounded exposure, not per-stage LR. Its benefit is a
**scale hypothesis**: with the real DEFAULT_STAGE_STEPS (2000+/stage) the global
cosine starves E far harder than at 200 steps/stage, so per-stage should matter
more on the big run. Watch full-page/train reconstruction (not held-out val) at
the D/E boundaries to confirm.

### 12.5 Left as decisions, not silently changed
- **SSL shared-encoder coupling.** Genuine tension: C2 (§1.2 fix) deliberately
  *connected* SSL to the shared encoder; §2.4's collapse-fix framing says SSL
  should touch only its own head. Fully stop-grad'ing reverts C2. Recommend
  leaving cos_w=0.1 (its payoff is generalization at scale, which a memorization
  toy can't show) but flag cos_w / stop-grad as knobs if reconstruction is hurt
  at scale.
- **Stage budgets.** For memorizing one page the dominant lever is grounded
  exposure (bigger A-D), not the schedule -- but that trades against SSL/ACT
  time and is scale-dependent; not retuned off the toy.

---

## 13. Scaling & baseline experiments (2026-07-08/09)

Three experiments run after the §11 fixes, to answer: does the pipeline work at
real scale, does the architecture actually *learn*, and how does it compare to a
plain transformer at matched scale. All on MPS/CPU (no GPU yet).

### 13.1 Small-preset shakedown (pipeline at 153M)
Pre-chunked 1,401 pile-10k docs (`small` preset, gpt2) into a cache, then walked
A→E via `train_scaled` on MPS in ~12 min (**~33 s/step**, batch 4). Confirmed:
all stages fire (SSL at D, ACT at E), `latent_std` healthy **0.48–0.67** across
the D boundary, val monotone with **no artificial jumps at D/E** (the M6 eval-fix
holding), and checkpoint→resume works (RNG restore, 1.7 GB ckpt). Purpose was
*plumbing*, not learning: 22 steps sits at chance, and — a useful lesson — with
`warmup=100` > total steps the LR never left warmup, so the flat loss measured
the schedule, not the model.

### 13.2 Single-page overfit — the architecture *does* learn
Built a tiny high-quality corpus: the Wikipedia "Solar System" article via the
API, ~5.8k gpt2 tokens, split into 53 paragraph docs (`wiki_cache`, smoke
preset). Full A→E drove reconstruction hard once warmup completed (val 10.95 →
~7.2 in Stage A). The decisive test — **grounded-only** (loop+memory, no SSL/ACT,
held LR 3e-4, 1500 steps) — drove page reconstruction **59,013 → 1.0 ppl**: the
architecture memorizes the page. The earlier ~936 "plateau" was a curriculum
artifact (§12), not an architecture wall. Caveat: at ppl 1.0 teacher-forced,
free-running greedy decode is still only *partially* coherent (some chunks
verbatim, others degenerate) — the autoregressive-dilution-through-bottleneck
caveat (§5.4, §7).

### 13.3 Fair-scale baseline vs a vanilla GPT (`baseline_gpt.py`)
New file `baseline_gpt.py`: a standard pre-LN decoder GPT, same gpt2 tokenizer /
text / optimizer / steps. Two presets, because "same scale" is ambiguous when
67–90% of the latent model's params are duplicate embedding tables:
`same-params` (44.7M, d512×6, matches *total* params) and `same-compute` (14.1M,
d192×10, matches *width* + ~4.5M non-embedding compute). **Bug caught+fixed:**
no GPT-style weight init → default `N(0,1)` embeddings made init CE 112–320
(should be ~10.8), wasting ~300 steps and unfairly handicapping the *wider*
model; post-fix init CE ≈ 10.86.

Results (teacher-forced page ppl): `same-params` **1.1** (generates the page
verbatim); `same-compute` **484** (degenerate); latent grounded-only **1.0**
(§13.2). Honest reading: (a) at the same *total* size a vanilla GPT memorizes
trivially and is far more parameter-efficient; (b) the latent's ppl is on an
*easier* task (reconstruction *with* lookahead) than the GPT's causal next-token,
so it's not a tie in the latent's favour; (c) this is a *memorization* test — not
the architecture's value proposition (latent reasoning, test-time-compute
scaling), which it cannot measure. One sobering data point, not a verdict.
Artifact: `runs/comparison.png`.

---

## 14. Scale & hardware planning (2026-07-09)

Planning the first real run: `small` (~153M), A→E only, on a single AMD GPU.

### 14.1 A Chinchilla-equivalent token budget (embedding-corrected)
Naive 20:1 on *total* params overcounts, because most params here are input
embedding **lookups** (~0 FLOP). Measured for `small` (152.8M total): input token
embeds **77.2M** (~0 FLOP), output head 25.7M (FLOP-active), core 49.7M. So the
compute-active N (head+core) = **75.4M → ~1.5B tokens**; core-only 49.7M →
~1.0B; the naive total would say 3.06B. The recurrence ("etc.") raises
FLOPs/token but **not** the optimal token/param ratio: the compute-optimal
D-for-given-N falls out of the loss-curve exponents, and the FLOP constant
cancels in the Lagrange condition `αA/N^α = βB/D^β` (independent of the per-token
FLOP multiplier) — consistent with Parcae's "looping and data trade at fixed
FLOPs". So the loop costs GPU-hours, not tokens. **Working budget: ~1.0–1.5B
tokens (center ~1.2B), ≈ ⅓ of naive**; `base` ≈ 3.0–3.8B. Caveats: the 20×
constant is borrowed from next-token LM, so the true number for this
reconstruction+SSL objective needs a token/IsoFLOP sweep; and ~1.2B tokens ≈
~9 GB cache, which eases the in-RAM `CachedChunkDataset` concern.

### 14.2 Target hardware: single AMD Strix Halo (Radeon 8060S, ROCm), 128 GB
Decisions (user): ROCm GPU + bf16 AMP, `small` preset to start, A→E only. Full
ops detail in **`STRIX_HALO.md`**; summary:
- **Compatibility:** ROCm exposes the GPU as `torch.cuda`, so the code needs no
  change and the bf16 AMP path works (prefer bf16 over fp16). gfx1151 may need
  `HSA_OVERRIDE_GFX_VERSION`.
- **Memory is an advantage, not a limit:** model+opt ~2.5 GB and a ~10 GB cache
  fit the 128 GB unified memory easily → in-RAM `CachedChunkDataset` is fine
  (memmap optional), and large batches are possible.
- **Throughput is the real question:** `forward_grounded` is a sequential
  per-chunk loop of many small ops → launch-overhead-bound, underutilizes any
  GPU (why `small` was ~33 s/step on MPS). Lever: **large batch** (the 128 GB
  enables it). The thought recurrence across chunks is inherently sequential;
  the Talker (teacher-forced) is the one real batch-across-chunks optimization,
  but needs testing — deferred.
- **New deliverables (all synthetic, no data needed):** `rocm_smoke.py`
  (validate the training path stays finite under bf16 on the GPU — the **first
  real execution of the AMP path**, so PASS is necessary not sufficient),
  `bench_throughput.py` (tokens/sec + wall-clock ETA sweep across batch sizes),
  and `STRIX_HALO.md` (setup + go/no-go). Recommended pre-flight order on the
  box: `rocm_smoke.py` → `bench_throughput.py` → read the ETA → short real-data
  shakedown watching `latent_std` → launch.

---

## 15. Third pre-scale review (2026-07-09) — external pass before the big run

A fresh independent read of spec + all code, with the same methodology as §1/§11
(read → hypothesize → confirm with tiny targeted scripts → fix → re-verify).
One critical inference-path bug, one latent training footgun, several run-ops
hardening fixes; plus two findings **documented but deliberately not changed**.
All fixes verified; full A→E (`train_scaled`, offline cache, resume) and A→F
(`train.py`) integration re-run clean, and `generate.py` still works against
the old `runs/model.pt`.

### 15.1 C6 — generation used the wrong latent space (critical, inference path)

`generate.py` seeded each new thought with `latent_predictor(last_latent)`. But
after the §5 collapse fix, `latent_predictor` lives on top of `ssl_proj` — it
maps *SSL-projection space → SSL-projection space*. Generation fed it a raw
**encoder-space** latent and pushed its output into the HRM loop as if it were
an encoder-space chunk latent: wrong input space, wrong output space. Deeper
problem: the separate-head fix removed the only trained encoder-space
next-latent map, so **free-running generation had no trained pathway at all** —
the architecture's own §2.4 fix quietly amputated its generation mechanism.

- *Fix*: new `model.gen_predictor` (2-layer MLP head) + `forward_gen_predictor`
  loss: predict chunk t+1's **shared encoder latent** from chunk t's, with BOTH
  input and target detached (encoder runs under `no_grad`), scaled-cosine (the
  HRM injection LayerNorms the latent, so cosine alignment suffices). Runs
  alongside SSL from Stage D (`TrainConfig.gen_loss_weight`, 0 disables),
  logged separately as `gen` so the `ssl` metric stays comparable.
  `generate.py` now uses it; checkpoints predating the head load with
  `strict=False` + an explicit warning. `rocm_smoke.py` covers it under bf16.
- *Why it cannot hurt the run*: gradient-isolated by construction. Audit
  (the §1.2/C2 methodology): gen loss → grads on exactly 4/4 `gen_predictor`
  tensors, **zero** on every other parameter; grounded+SSL losses → **zero**
  grad on `gen_predictor`, `chunk_encoder` unaffected (28/28 as before).
- *Verified learning*: in the 24-step offline integration run the gen loss
  fell 2.97 → 1.78 while nll/ssl/latent_std matched pre-change behavior.

### 15.2 M11 — truncation silently off when grad_window ≥ inner steps (latent footgun)

`_TruncationSchedule` (fixed-depth mode) cut at `step == total - window`, with
a `step > 0` guard. For `window >= total` (e.g. window 8 with 2×(3+1)=8 inner
steps) the cut never fired, so the **entering states were never detached and
the raw cross-thought chain came back** — the exact C5 full-document-BPTT bug,
one config tweak away. Confirmed empirically: nonzero gradient on the previous
thought at window 8/12, zero at 5. Current defaults (2→5) were safe; anyone
raising `inner_loop_grad_window_end` or lowering the cycle counts would have
silently reverted C5. **Fixed**: cut at `max(0, total - window)` (cut at step 0
= whole thought keeps gradient, entering chain still severed). Re-verified:
chain severed at every window, forward values bit-identical, per-window
gradient monotone and saturating at full depth (reproduces the §11.1 numbers).

### 15.3 M12 — checkpoint/resume off-by-one in the per-stage LR schedule

`Trainer.train` saved checkpoints *before* `curriculum.advance_step`, so a
resume replayed one extra `step_in_stage` and the per-stage LR drifted off the
uninterrupted run (measured on the toy: lr 1.23e-4 resumed vs 5.58e-5
uninterrupted at the same global step — large there because the toy stage was 6
steps; small but nonzero at real budgets). **Fixed**: advance before save;
resume now schedule-exact (identical lr at the same step, val within data-order
noise).

### 15.4 Run-ops hardening (small, each verified)

- **Atomic checkpoints** (`trainer.save`): write to `*.tmp` + `os.replace`, for
  both the checkpoint and `metrics.json` — a crash mid-save can no longer
  destroy the only checkpoint of a multi-day run.
- **Document pre-truncation** (`data.chunk_text_example`): docs are cut to a
  char budget generously covering what survives (`max_chunks × max_chunk_len ×
  8 chars/token`) before SaT/tokenization. Previously a PG-19-length book was
  segmented and tokenized in FULL (10–100× wasted prep work) and — worse —
  `encode_recent` took the input-lane window from the *end* of the document,
  text disjoint from the kept chunks; now the raw window covers the kept
  region's tail. Verified: byte-identical tensors to manual truncation; short
  docs unaffected.
- **Seeded random val split** (`train_scaled.py`): was "first N cache docs"
  (corpus-ordered → potentially topically clustered); now a seed-0 permutation,
  deterministic across resumes.
- **Tokenizer length-warning silenced** (`model_max_length = 1e12` on the
  wrapped tokenizer): whole-doc encodes of >1024 tokens flooded (and slowed)
  data prep with per-doc warnings.

### 15.5 Documented, deliberately NOT changed

- **ACT's halting head is trained by the ponder cost alone.** Confirmed
  empirically: |grad| on the halting head from the NLL is exactly **0** (the
  halting decision is a thresholded `.item()` branch — depth is invisible to
  autograd), from the ponder cost **0.27**. The only signal says "halt sooner",
  so Stage E's policy must converge to halt-at-minimum-depth (= fixed 2-cycle
  depth): benign for stability (E ≈ D plus a tiny loss term) but the
  test-time-compute dial *learns nothing*, and §11.3's "bump `act_ponder_cost`
  if it halts too late" can never be needed in this form — there is no
  counter-pressure to trade against. The real fix is an ACT accumulator
  (output = Σₖ p(halt=k)·h_k, giving the NLL leverage on the head) or
  REINFORCE; that is an untested training-dynamics change and exactly the
  wrong thing to land unvalidated days before a long run. Spec §5.5 now
  carries the caveat. **Expect `halt_prob → 1` in Stage E logs; that is this
  mechanism working as (currently) built, not a bug appearing.**
- **Memory truncation is transitively leaky, and the notes overstated §11.1's
  consequence.** With `memory_grad_window=2`, gradient from thought 9's loss
  still reaches thought 0 (~30× attenuation per hop; 7e-4 → 2e-8). The window
  bounds *direct* reads; slot t−1's graph contains its own reads of t−2…, so
  credit chains transitively and the autograd graph **still spans the whole
  document in Stages C+** — §11.1's "activation memory no longer grows as one
  full-doc graph" is true only for the raw state chain (and Stage B, where
  writes are detached). Not a correctness bug (it is Thought Gestalt's
  un-detached-memory semantics, softly truncated), but it is the memory-footprint
  reality: trust `bench_throughput.py`'s peak-GB (it runs Stage-C flags with
  memory un-detached), not a mental model of a 5-step graph. Spec §3.6 now
  carries the caveat. Related: in ACT mode the *ponder* term also leaks a small
  bounded gradient across thoughts (early-cycle halt probs sit before the
  rolling cut) — weight `act_ponder_cost`=0.01, Stage E only, left alone.

### 15.6 Planning note surfaced by the review (not a code change)

The §14.1 budget (~1.2B tokens) cannot come from the README's example dataset:
pile-10k holds ~10k docs ≈ 20M usable tokens after the 32×64 per-doc cap.
`data_prep.py` also only supports a **single** HF dataset (`iter_hf_single`) —
the §5.6/DataConfig mixture was never wired into the offline prep path. For the
big run either point `--dataset` at one big corpus (e.g.
`HuggingFaceFW/fineweb-edu` `sample-10BT`, which `--text-field text` handles,
noting `data_prep` downloads non-streaming) or wire `iter_hf_mixture` into
`data_prep` first. Prep is single-process; with the truncation fix it should be
hours for ~1.2B tokens, but do a timed 1k-doc dry run before committing.

### 15.7 Re-verification summary

- Gradient audits: gen-head isolation both directions (15.1); truncation cut at
  windows {1,2,5,8,12} with identical forwards (15.2); halting-head gradient
  sources (15.5); transitive memory reach (15.5).
- Offline A→E via `data_prep --offline` → `train_scaled` (smoke preset, fixed
  budgets 4/4/4/6/6): all stages fire, `gen` logged and decreasing, ckpt at
  step 20 → `--resume` continues schedule-exact to the same final val (8.3104
  vs 8.3108) — the M12 fix holding.
- Offline A→F via `train.py`'s loops (incl. Stage F dialogue path): finite,
  loss decreasing.
- `generate.py` against the existing pre-review `runs/model.pt`: loads with the
  explicit missing-module warning, `--score` matches expectations (ppl ~1.3k on
  the test sentence), generation path runs end-to-end.

---

## 16. Fourth pre-scale review (2026-07-09) — final read before the big run

Another independent pass over spec + all code, same methodology (read →
hypothesize → confirm by running → fix → re-verify), specifically to catch
anything the §11/§15 passes missed before committing to an expensive run. The
structural bugs those passes fixed (inner-loop truncation no-op §11.1, gen-head
latent space §15.1, resume LR off-by-one §15.3) were re-confirmed **present and
working**. One real latent bug found and fixed; two minor items surfaced and
deliberately left for the owner to decide.

### 16.1 C7 — `grounded_loss_min_frequency` default contradicted the §2.4 anti-collapse fix (latent bug, fixed)

`config.TrainConfig.grounded_loss_min_frequency` still defaulted to **0.2** — a
stale value from before the §2.4/§5 collapse fix. At that value, from Stage D
the grounded (reconstruction) *anchor* loss runs only 20% of steps while SSL
runs every step. That is **exactly** the collapse recipe §2.4 exists to remove
and that §5 verified empirically flattens the shared encoder (cosine → ~0.996,
reconstruction regresses the moment SSL turns on).

- The **big-run path was already safe**: `train_scaled.py` and `train_real.py`
  both explicitly override to `1.0` ("reconstruction stays the always-on
  anchor").
- But `train.py` — the README-documented real-data entry point
  (`LATENT_USE_HF=1 python train.py`) — uses the bare `TrainConfig()` default,
  so a real run launched that way would have re-triggered the collapse.

Fix: changed the default to **1.0** in `config.py` (the stated single source of
truth), with a comment tying it to the §2.4 correction, so *every* entry point
is safe-by-default. The knob still exists for deliberate thinning; it just no
longer defaults into the failure mode. Verified `TrainConfig().grounded_loss_min_frequency == 1.0`.

### 16.2 Documented, deliberately NOT changed

- **EMA target encoder runs with dropout active.** `EMATargetEncoder` never
  puts its `target_encoder`/`target_proj` in `.eval()`, so the SSL cosine
  target is stochastic per call. Low impact (target noise is mildly
  anti-collapse and `cos_weight` is only 0.1), but a deterministic momentum
  target is the standard BYOL/DINO choice. Left unchanged because it shifts
  training dynamics and is not a correctness failure — flagged for the owner.
- **`--resume` path asymmetry (UX).** `train_scaled.py` resolves `--cache` and
  `--out` against the project root but passes `--resume` verbatim
  (`trainer.load(args.resume)`), i.e. relative to cwd. A relative `--resume`
  threw `FileNotFoundError` during re-verification until an absolute path was
  used. Cosmetic; worth normalizing or documenting.

### 16.3 Re-verification (what was actually run this pass)

- **Full A→E via the exact big-run path** (`data_prep.py --offline --preset
  smoke` → `train_scaled.py`, fixed budgets 4/4/4/4/4): all stage transitions
  fire, `ssl`/`gen`/`ponder` log and move correctly (ponder activates at E),
  checkpoints written, `latent_std` healthy (~0.43, no collapse).
- **Checkpoint → `--resume`**: continues schedule-exact (LR picks up mid-cosine,
  losses/`latent_std` continuous) — the §15.3 fix holding.
- **bf16 autocast numerics on CPU** (`torch.autocast`, mirroring `rocm_smoke`'s
  checks since the trainer forces AMP off on CPU): `forward_grounded` for stages
  C / E-ACT / F-lanes, plus `forward_self_supervised` + `forward_gen_predictor`
  — **all losses and grads finite**. The `--amp` path the ROCm run uses stays
  finite through the ops that can NaN (hard_normalize division, Parcae
  exp/softplus, masked softmax, CE).
- **Multi-worker DataLoader** (`--num-workers 2`) and **`generate.py`**: both run
  end-to-end.

Net: only `config.py` changed. The scaled path was already collapse-safe; this
closes the one remaining footgun in the shared default. No large training run
was started.

---

## 17. Fifth pre-scale review (2026-07-09) — independent read + two hardening fixes

Another full pass over spec + all code before the run, same methodology
(read → hypothesize → confirm by running → fix → re-verify). The structural
fixes from §11/§15/§16 were re-confirmed present and working. The scaled
big-run path (`data_prep.py` → `train_scaled.py` → `trainer.py`) is clean: a
full offline A→E smoke fired every stage, `ssl`/`gen`/`ponder` moved correctly
(ponder activates at E), `latent_std` held ~0.43 (no collapse), and
checkpoint → `--resume` continued schedule-exact. Two real-but-narrow issues
found and fixed; no critical bug in the scaled path.

### 17.1 Stage F device bug (fixed)

`train.py:train_stage_f` initialized `total_loss = torch.zeros(())` — a CPU
scalar — then added the per-turn `nll`/`ponder` (on the model's device). On CPU
(the default, and what all prior A→F verification used) this is silent; on a
GPU run it raises "expected all tensors on the same device" the moment Stage F
starts. Fixed to `torch.zeros((), device=device)`. Narrow (Stage F via
`train.py` only; the scaled A→E path is unaffected) but a real latent crash for
any GPU fine-tuning run.

### 17.2 `torch.load` weights_only portability (hardened)

`trainer.load` and `generate.load` called `torch.load` without
`weights_only=`. Checkpoints carry Python/NumPy/torch RNG state (non-tensor
pickled objects); torch ≥ 2.6 defaults `weights_only=True` and would refuse to
unpickle them, breaking `--resume` — exactly the thing a multi-day run cannot
afford, and easy to hit if the GPU box runs a newer torch than the dev box
(2.2.2). Added an explicit `weights_only=False` at both sites (no-op on 2.2.2,
correct on newer torch). Shard loads in `CachedChunkDataset` are plain-tensor
dicts and were left as-is.

### 17.3 Two efficiency changes applied at owner request

Both items §16.2 had deferred were pulled in (owner: "I don't really care about
dropout", which removes the one reason the second was non-trivial):

- **EMA target now deterministic.** `EMATargetEncoder.__init__` puts
  `target_encoder`/`target_proj` in `.eval()` (dropout off) — the standard
  BYOL/DINO choice. Nothing flips them back (the wrapper is not an nn.Module, so
  `model.train()` never reaches them).
- **Shared encoder pass deduplicated.** New `model.encode_chunks()` computes the
  order-aware chunk latents ONCE per step; `forward_grounded`,
  `forward_self_supervised`, and `forward_gen_predictor` each take an optional
  `chunk_vecs=` and reuse it (methods still self-encode when called standalone —
  Stage F, eval, generate). Wired through `trainer._loss_on` and
  `train.train_stages_a_to_e`. Online `chunk_encoder` forwards per D–E step drop
  **3 → 1** (verified by forward-hook count); the EMA target still runs its own
  single pass. `gen_predictor` isolation is preserved by detaching the shared
  vecs (verified: gen loss reaches no `chunk_encoder` param). Because the three
  branches previously drew independent dropout masks, D+ numerics shift slightly
  vs the pre-change run (A–C are bit-identical — grounded's single encode was
  already shared); `latent_std` stays ~0.43, no collapse, losses still decrease.

### 17.4 Standing recommendation before launch

`--amp`/bf16 is still only validated by `rocm_smoke.py`, never a real GPU
training step. Run the STRIX_HALO.md go/no-go (`rocm_smoke.py` +
`bench_throughput.py`) on the actual box first; that autocast path is the
largest remaining untested surface.

### 17.5 Re-verification (what was actually run this pass)

- **Full A→E via the exact big-run path** (`data_prep.py --offline --preset
  smoke` → `train_scaled.py`, budgets 3/3/3/3/3): every stage transition fires,
  `ssl`/`gen`/`ponder` move correctly (ponder activates at E), checkpoints
  write, `latent_std` ~0.43 (no collapse). Re-run after the §17.3 changes: A–C
  bit-identical, D+ shifts only by the shared-dropout-mask/deterministic-target
  amount, still healthy.
- **Checkpoint → `--resume`**: continues schedule-exact (LR mid-cosine,
  losses/`latent_std` continuous) — the §15.3 fix still holding.
- **Encoder-pass dedup**: forward-hook count confirms **1** online
  `chunk_encoder` forward per D–E step (was 3) + 1 EMA target forward; EMA
  target confirmed in eval mode; `gen_predictor` still isolated (gen loss
  reaches zero `chunk_encoder` params).
- **`generate.py`**: still loads the existing `runs/model.pt` (with the expected
  missing-`gen_predictor` warning for that pre-§15 checkpoint) and scores.
- All files byte-compile. No large training run was started.

Files changed this session: `files/train.py` (17.1 device fix, 17.3 wiring),
`files/trainer.py` (17.2 load flag, 17.3 wiring), `files/generate.py` (17.2
load flag), `files/ema_target.py` (17.3 eval target), `files/model.py` (17.3
`encode_chunks` + `chunk_vecs=` reuse), `notes.md` (this section).

---

## 18. Sixth pre-scale review (2026-07-09) — one critical AMP-path bug

Another independent read of spec + all code before the run. The structural
fixes from §11/§15/§16/§17 were re-confirmed present and working (offline A→E
via the exact big-run path fires every stage; `ssl`/`gen`/`ponder` move
correctly; `latent_std` ~0.47–0.64, no collapse; checkpoint → `--resume`
schedule-exact, val 8.5737 vs 8.5735 uninterrupted). One **critical bug on the
`--amp` path** found and fixed — the single largest untested surface, and the
exact one the big run flips on.

### 18.1 C8 — EMA target encoder crashes under bf16 autocast (critical, AMP path)

`EMATargetEncoder.target_encoder` is kept in `.eval()` (§17.3, for a
deterministic momentum target) and `encode()` is `@torch.no_grad()`. Under AMP
autocast, an **eval-mode `nn.TransformerEncoder` with grad disabled takes the
fused BetterTransformer fast path** (`torch._transformer_encoder_layer_fwd`),
which feeds autocast-cast **bf16 activations into the encoder's fp32 weights**
and raises `RuntimeError: mat1 and mat2 must have the same dtype, but got
BFloat16 and Float`. This fires inside `forward_self_supervised` → `ema.encode`
**the moment the SSL loss turns on (Stage D)** on any `--amp` run — i.e. it
would kill the big run at the D transition, and would fail `rocm_smoke.py` step 4.

- *Why every prior AMP check missed it*: the trainer forces AMP **off on CPU**
  (`device.type != "cpu"`), `rocm_smoke.py` **exits at step 1 without a GPU**
  (its autocast is hardcoded `device_type="cuda"`), and the one explicit
  bf16-autocast-on-CPU check (§16.3) predates §17.3's `.eval()` change — so the
  combination *(eval target encoder + autocast + no_grad)* had never actually
  executed anywhere. The online `chunk_encoder`/`input_lane` encoders are in
  `.train()` during the step (no fast path → fine); `evaluate()` runs eval-mode
  encoders but is **not** autocast-wrapped (fp32 → fine). The EMA target was the
  sole eval+autocast site.
- *Confirmed*: minimal repro — `nn.TransformerEncoder` in `eval()` +
  `no_grad()` + `autocast(cpu, bf16)` raises the dtype error; `train()` mode or
  no-autocast are both fine. Full model: pre-fix `forward_self_supervised`
  under bf16 autocast throws at `ema.encode`; post-fix all of stages C / E-ACT /
  F-lanes give finite `nll`/`ponder`/`ssl`/`gen` and finite grads.
- *Fix* (`ema_target.encode`): compute the detached momentum target with
  autocast **disabled** (`torch.autocast(device_type=dev, enabled=False)`, guarded
  to cpu/cuda; `nullcontext` elsewhere). The target is a stop-grad momentum copy,
  so full precision is *correct and more accurate*, not a compromise; the online
  path is unchanged. The fp32 target vs bf16 prediction in the subsequent
  scaled-cosine loss is handled by eager type promotion (verified finite).
  No-op on the non-AMP path (A–C bit-identical; C2 gradient reach preserved —
  SSL still trains `chunk_encoder` 28/28, `gen_predictor` still isolated 4/4).

### 18.2 Documented, deliberately NOT changed

- **ACT-mode truncation has the §15.2 footgun un-hardened.** §15.2 fixed the
  *fixed-depth* cut (`max(0, total-window)`); the **ACT rolling cut**
  (`in_graph >= window`) has the same latent issue — if `window` ever exceeds
  the executed step count the entering cross-thought chain would survive. **Not
  reachable at current config** (`inner_loop_grad_window_end=5` < the 8-step ACT
  minimum of `n_cycles*(l_steps+1)=2*4`), so left as-is to avoid an unvalidated
  gradient-path change days before the run; harden it (mirror the `max(0,…)`
  guard) only if the cycle counts or the end-window are changed.
- **AdamW still decays `theta`/`log_dt`/LayerNorms** (§9). Over a long run the
  weight decay on Parcae's generator params pulls the per-channel decay toward
  its init ~0.62; benign but worth a no-decay param group if ever revisited.

### 18.3 Re-verification (what was actually run this pass)

- Offline A→E via `data_prep.py --offline --preset smoke` → `train_scaled.py`
  (budgets 3/3/3/3/3 and 2/2/2/2/2): every stage fires, SSL at D, ponder at E,
  `gen` logged, `latent_std` healthy, checkpoints written; `--resume` from the
  step-12 checkpoint continues schedule-exact.
- bf16-autocast finiteness at `smoke` **and** `small` presets across stages
  C / E-ACT / F-lanes (`forward_grounded` + `forward_self_supervised` +
  `forward_gen_predictor`): all losses and grads finite **after** the C8 fix
  (all crash pre-fix at `ema.encode`).
- Gradient-isolation audit re-run post-fix: SSL → `chunk_encoder` 28/28 +
  `ssl_proj` 4/4, `gen_predictor` 0/4; gen → `gen_predictor` 4/4 only.
- All files byte-compile. **No large training run was started.**

Standing recommendation (unchanged from §17.4): still run the STRIX_HALO.md
go/no-go (`rocm_smoke.py` → `bench_throughput.py`) on the actual box first —
that autocast path is now expected to *pass* rather than crash at step 4.

Files changed this session: `files/ema_target.py` (18.1 autocast-safe target),
`notes.md` (this section).

---

## 19. Seventh pre-scale review (2026-07-09) — independent pass, new model family

Full review of spec + all 25 source files before the big run, run as three
parallel review passes (core model math; training/data pipeline; spec
conformance + inference) plus an independent line-by-line read, each finding
re-verified against the code and empirically where possible. **No critical bug
in the A→E scaled training path** — the §11–§18 passes genuinely cleaned it.
What this pass found lives at the edges the earlier passes under-weighted: the
prep CLI, the *inference* contract of the checkpoint being trained, and the
observability of the collapse defense.

### 19.1 C9 — `data_prep.py` could not select a HF dataset config (prep blocker, fixed)

`iter_hf_single` has a `name` parameter, but `data_prep.py` never passed it and
had no `--name` flag — so §15.6's own big-run plan ("point `--dataset` at
`HuggingFaceFW/fineweb-edu` `sample-10BT`") was **unexecutable**: `name=None`
resolves to fineweb-edu's *default* config (the full multi-TB corpus) and,
because prep was hard-coded non-streaming, would begin downloading all of it;
`--max-tokens` only caps iteration, not the download. §15.6 asserted this route
"handles" it — it didn't. Fixed: `--name` and `--streaming` flags wired through
(`data_prep.py`), source echoed at startup. Big-run prep is now:
`python data_prep.py --dataset HuggingFaceFW/fineweb-edu --name sample-10BT
--streaming --preset small --max-tokens 1200000000`.

### 19.2 C10 — no end-of-chunk supervision: generation could never terminate a chunk (training-side, fixed)

The grounded NLL masked to real tokens only, so no position past a chunk's true
length ever received gradient — there was no EOS in the vocab and PAD-after-end
was unsupervised. `talker_decode` additionally hard-banned PAD, so **every**
generated chunk ran to the full `max_chunk_len` tokens; everything past the
natural end was sampled from an *untrained* distribution, then the garbage tail
was re-encoded (`gen != 0` all-True) to seed the next latent — contaminating
every subsequent thought. Distinct from the documented "smoke-scale output is
incoherent" caveat: this is structural, would persist at any scale, and an
inference-side patch cannot retroactively add missing supervision to a finished
checkpoint — which is why it had to land *before* the big run.

Fix (both sides, verified):
- `model.forward_grounded` now includes the **first pad position** of every
  active shorter-than-max chunk in the NLL target mask — one supervised
  end-of-chunk step, PAD (reserved id 0, never a real token) acting as EOS.
  Full-length chunks get no end mark (a capped span legitimately continues).
  Cost: ~1 extra supervised token per short chunk (~2–5% of the loss mass);
  NLL values shift accordingly vs pre-§19 runs (same-config comparisons remain
  valid, cross-§19 comparisons are off by the EOS term).
- `generate.talker_decode` treats a sampled PAD as stop (banned only at
  position 0, so a pre-§19 checkpoint with untrained PAD logits can't emit a
  degenerate empty chunk); the re-encode of a stopped chunk automatically
  masks the tail.
- Verified: PAD row of `talker.lm_head` receives gradient (norm 0.54 on the
  micro-test; exactly 0 before the fix); full/short/absent-chunk mask cases
  checked independently; A→E offline run trains normally with the new term.

### 19.3 M13 — `generate.py --score` was chunk-weighted, not token-weighted (fixed)

`score()` averaged per-chunk *mean* NLLs with equal weight per chunk, while
`baseline_gpt.eval_ppl` token-weights — so a 3-token chunk counted as much as a
48-token one and short chunks (systematically higher per-token NLL) biased the
latent model's number relative to the baseline's. Any non-saturated post-run
perplexity comparison would have been invalid (§13.3's ppl≈1 regime masked it).
Fixed: accumulate `mean_nll × n_real_tokens`, divide by total real tokens.
Labeled "avg NLL/token" is now true. (Real-token positions only — the EOS term
of 19.2 is deliberately excluded so the number stays comparable with the
baseline, which has no EOS.)

### 19.4 M14 — collapse monitor measured dropout noise, not collapse (fixed); variance floor attenuation (documented)

`latent_collapse_metric` ran with the encoder in train mode (the trainer calls
it right after `evaluate()` restores `.train()`), and dropout noise alone reads
as per-dim std ≈0.05–0.08 — the same order as the 0.1 floor. Verified: a fully
collapsed encoder (zeroed embeddings) read `latent_std = 0.078` in train mode
vs the true `0.0` in eval mode. At the `small` preset a Stage-D collapse could
have logged ≈0.1 and looked healthy for days. Fixed: the monitor now switches
the chunk encoder to eval for the measurement and restores train mode
(monitor-only; training dynamics untouched; healthy readings barely move —
0.65 vs 0.62 on the micro-test).

**Documented, deliberately NOT changed:** the VICReg variance floor itself is
computed on train-mode (dropout-noised) latents, so at full collapse its hinge
fires at ≈`0.1−0.08`, i.e. ~4–5× weaker than nominal. Changing where the floor
is measured would alter training dynamics days before the run; the grounded
anchor at `grounded_loss_min_frequency=1.0` remains the primary defense, the
floor is the last-resort net. Operational rule: **at scale, treat
`latent_std ≲ 0.1` as collapse even though it's nonzero.**

### 19.5 M15 — checkpoints carried no training schedule; resume trusted the CLI (hardened)

`Trainer.save` stored model/optimizer/EMA/curriculum/RNG but nothing from
`train_cfg` — a resume with a drifted command line (different `--stage-steps`,
`--lr`, `--lr-schedule`, or edited defaults) would reinterpret the restored
`stage_idx`/`step_in_stage` against different budgets and silently diverge —
the §15.3 drift class reintroduced through the config side door (wrong
`--preset` fails loudly; wrong schedule flags failed silently). Now: the
checkpoint embeds the resolved schedule (stage budgets, LR family, accum,
batch, loss weights, AMP) and `load` prints a loud field-by-field warning on
any mismatch. Verified: clean resume silent; drifted resume prints the diff.

### 19.6 Run-ops hardening (small, each verified)

- **Numbered checkpoint archives**: `--archive-every N` keeps
  `checkpoint_{step}.pt` snapshots alongside the rolling `checkpoint.pt` —
  rollback depth for slow pathologies noticed hours after onset (the rolling
  file alone cannot rewind past onset). Off by default; recommended ~2000 for
  the big run (each snapshot is model+optimizer+EMA sized).
- **Stale/mixed cache guard**: `CachedChunkDataset` now errors if the manifest
  `total` disagrees with the loaded shard rows (a crashed re-prep into an
  existing dir leaves the old manifest next to a mix of old/new shards; it
  previously trained on the blend silently). Always prep into a fresh dir.
- **`--amp` off-CUDA**: `Trainer` now runs full precision with a printed note
  instead of crashing at step 1 on MPS (torch 2.2 raises on mps autocast; on
  newer torch the EMA-target C8 pattern would resurface). CUDA/ROCm unchanged.
  `bench_throughput.py` similarly stops printing `amp=bf16` while silently
  timing fp32 off-CUDA.
- **`generate.py --ckpt`**: the big run's checkpoint (`runs/scaled/model.pt`)
  is now reachable without copying files around.

### 19.7 Spec/docs corrections (no behavior change)

- Spec §5.1 "next-chunk NLL" — stale pre-§2.2 wording, now corrected in place
  (the grounded loss is same-chunk reconstruction; the code was right).
- `parcae.py` claimed "PreNorm is applied by the caller" — false
  (`norm.PreNormWrapper` is instantiated nowhere); comment now states the real
  invariant (hard-normalized state + normalized injection bound the residual's
  inputs).
- README "What's simplified" now records: Stage A skips the H-update entirely
  (spec says one H-update); the scaled prep path chunks with regex boundaries,
  not SaT (only `train.py LATENT_USE_HF=1` uses real SaT); and the Stage F
  loop trains user turns through the grounded/self path, so the §4.2 two-lane
  separation is wired but **not exercised** by Stage F as written — fix before
  Stage F fine-tuning is treated as real (irrelevant to the A→E run).
- `data_prep.py` docstring no longer claims SaT.

### 19.8 Re-verification (what was actually run this pass)

- Micro-tests: EOS mask cases (short/full/absent chunks, per-row), PAD-row
  lm_head gradient present after fix, eval-mode collapse monitor reads 0.0000
  at forced collapse / 0.65 healthy, train-mode restored after measurement.
- Offline end-to-end at smoke preset: `data_prep --offline` → `train_scaled`
  budgets 3/3/3/3/3 — every stage fires, SSL+gen at D, ponder at E,
  `latent_std` healthy (0.62→0.40), rolling + numbered checkpoints written.
- Interrupted-resume: stop at step 7 → resume → stage transitions land on the
  same steps, final val 8.0965 vs 8.0987 uninterrupted (known data-order
  noise); drifted-schedule resume prints the field-by-field warning.
- `generate.py` against the pre-§19 `runs/model.pt`: `--score` (now
  token-weighted) and generation both run end-to-end; `--ckpt` flag works.
- All files byte-compile. **No large training run was started.**

Standing recommendation (unchanged from §17.4/§18.3): run the STRIX_HALO.md
go/no-go (`rocm_smoke.py` → `bench_throughput.py`) on the actual box first.
Two planning reminders surfaced by this pass: `DEFAULT_STAGE_STEPS` (12k steps
≈ 230M tokens at batch 16) is far short of the §14.1 ~1.2B-token budget — set
`--stage-steps` explicitly; and `trainer.save` labels every checkpoint
`tokenizer_name: gpt2` even for a stub-tokenizer cache, so don't point
`generate.py` at a checkpoint trained from an `--offline` cache.

Files changed this session: `files/data_prep.py` (19.1, 19.7), `files/model.py`
(19.2 EOS supervision, 19.4 eval-mode monitor), `files/generate.py` (19.2
PAD-stop, 19.3 token-weighting, 19.6 --ckpt), `files/trainer.py` (19.5 schedule
guard, 19.6 archives + amp gate), `files/config.py` + `files/train_scaled.py`
(19.6 archive knob), `files/data.py` (19.6 cache guard), `files/parcae.py` +
`files/bench_throughput.py` + `README.md` + `latent-thought-architecture.md`
(19.7), `notes.md` (this section).

---

## 20. "Is Parcae actually relevant?" — honest rename + a deferred optimization (2026-07-10)

Prompted by a plain question: is the "Parcae" state transition actually doing
what the doc credits it with, and is it *necessarily* Parcae at all?

### 20.1 The finding (no code behavior at stake — a naming/claim audit)

The looped state transition (`h_{n+1} = a⊙h + B·e + R(h,e)`, then hard_normalize)
is a **generic per-channel diagonal decay gate** — the S4/Mamba discretization
`a = exp(-softplus(theta)·dt) ∈ (0,1)`. There is no Parcae-specific machinery in
it: a per-channel scalar in (0,1) satisfies "spectral norm < 1" trivially, by
construction. The Parcae paper's actual contribution (a spectral-norm constraint
on a full looped-*transformer* transition) is **not implemented**; we use only
its trivial diagonal case. So the primitive is not "necessarily Parcae" — it is a
gated residual / leaky-integrator cell that Parcae happens to be one citation for.

What the gate does and does not do (sharpens §1.1's earlier correction):
- **Does not** bound the state at depth — `R` is an unconstrained nonlinear MLP,
  so a contractive `a` bounds nothing. Boundedness is entirely MagicNorm's
  hard_normalize (re-projection onto ‖h‖=√d each step). Already logged in §1.1.
- **Does** two real jobs: (1) it is the only *linear carry path* propagating
  state across steps (remove `a⊙h` and all carry-over must route through `R`);
  (2) keeping the linear part contractive *shapes* the on-shell dynamics toward
  convergence rather than orbiting — which is the mechanism we *hope* buys
  predictable, saturating test-time-depth scaling. That last part is an
  **empirical claim to measure** (the §5.5 depth sweep), not a guarantee the
  gate provides. The clean ablation that would settle it: full gate vs. `a`
  frozen at a constant vs. `a` removed — if frozen==full, the "spectral" story
  is really just "a leaky-integrator carry path" and can be dropped from the
  narrative.

### 20.2 Rename (behavior-preserving — forward left byte-identical)

Renamed the primitive to describe what it is. **The `forward()` was not touched**,
so training numerics are unchanged:
- `files/parcae.py` → `files/decay_gate.py`; class `ParcaeStateTransition` →
  `DiagonalDecayGate`. Docstring rewritten with the honesty note above.
- Config `parcae_min_decay`/`parcae_max_decay` → `decay_min`/`decay_max`
  (`config.py`), updated at the single read site (`model.py`).
- Import + comments in `hrm_loop.py`; mechanism references in `norm.py`,
  `rocm_smoke.py`, `STRIX_HALO.md`.
- Docs: `README.md` (file-map row, intro), `latent-thought-architecture.md`
  (intro + a §3.3 bridge note mapping the doc's "Parcae" to `DiagonalDecayGate`).
  The **Parcae paper citation is kept** as prior art (§0 bullet, §3.3) — it is a
  real influence; only the claim that our primitive *is* Parcae's mechanism was
  removed. Verified: no dangling `parcae` symbols in code (only two intentional
  paper citations in `decay_gate.py`), all files byte-compile.

### 20.3 A real but deferred optimization: loop-constant-`e` caching

Verified at `hrm_loop.py` (the L-group): inside the fast loop,
`e = chunk_embed + h_state` is **constant across all `l_steps_per_h_update`
iterations** (neither term changes until the H-update). Yet `DiagonalDecayGate`
recomputes the e-dependent matmuls — `B·e` and the e-half of the residual's first
layer — on every L-step. Only the h-dependent half genuinely changes. Splitting
the cell into `precompute(e)` (once per group) + `step(h)` (per iteration) would
remove ~1/3 of the L-gate's per-step matmuls, algebraically exact
(`W·[h;e] = W_h·h + W_e·e`), zero quality change. Only the L-loop benefits (the
H-transition runs once per group).

**Not applied yet, on purpose.** Precomputing `e` freezes its autograd attachment
for the whole group, whereas the current code re-derives it each step *after*
`trunc.maybe_detach`. If a truncated-BPTT detach boundary falls mid-L-loop, the
gradient reaching `h_state` through the e-path changes — and the truncation path
was just fixed (§19, f74b82e9 / cba8804f). So this is a real change with a real
regression surface, not a free lunch. Gated behind a measurement.

### 20.4 `profile_transition.py` — the go/no-go for 20.3

New script (reuses `bench_throughput.py`'s synthetic harness): hooks the L-gate /
H-gate / reasoner, prints each one's forward share of a step, and estimates what
the caching would save (~1/3 of L-gate time). Rule of thumb baked into the
output: if the estimated saving is under ~2–3% of a step, the caching is **not**
worth perturbing the truncated-BPTT path before the scaled run. Run it on the
ROCm box (torch not installed on the dev box), like bench:
`python profile_transition.py --preset small --batch-size 16`.

### 20.5 Deliberately left alone

- `build_deck.py` — the slide deck still lists "Parcae" as one of the four
  ingredient chips (presentation branding; rename only if the deck should match).
- Earlier `notes.md` entries (incl. the §19.7 `parcae.py` filename references) —
  historical log, not rewritten. This §20 is the forward pointer.

Files changed this session: `files/parcae.py` → `files/decay_gate.py` (rename +
honest docstring), `files/hrm_loop.py` (import/comments), `files/config.py`
(field rename), `files/model.py` (read site), `files/norm.py` +
`files/rocm_smoke.py` + `STRIX_HALO.md` (mechanism wording),
`files/profile_transition.py` (new, 20.4), `README.md` +
`latent-thought-architecture.md` (20.2 docs), `notes.md` (this section). No
training run started; forward numerics unchanged.
