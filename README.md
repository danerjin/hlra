# Latent-Thought Reasoning Architecture — Reference Implementation

A PyTorch implementation of the architecture described in
`latent-thought-architecture.md`, combining JEPA-Reasoner, HRM-Text,
Thought Gestalt, and Parcae into a single model that thinks in latent
"thoughts" (chunk-level vectors), each decoded into tokens by a separate
Talker module, using a diagonal-decay-gated HRM loop as the thinking
mechanism (the decay gate is the Parcae paper's diagonal case; the
depth-stability is MagicNorm's hard-norm, not the gate — see §3.3).

This is a from-scratch reference implementation meant to be readable and
traceable back to the design doc, not a production training pipeline. It runs
end-to-end (Stages A→F) both **offline** (synthetic text, no downloads) and on
**real text** (a HuggingFace corpus), can **save a checkpoint**, plot training
curves, and **generate** from the model.

> 🔴 **HIGHLY MAJOR CHANGE (notes §27) — the two losses were split by role.**
> **Reconstruction is now a pure autoencoder** (encoder → Talker, **no HRM
> loop**). **The HRM loop lives only in the predictive self-supervised loss**,
> which runs **sequentially** so the loop reads its accumulating gestalt memory
> (this is what finally trains the cross-thought memory). The loop is trained
> *only* by prediction; the Talker *only* by reconstruction. **Curriculum
> reshuffled** — SSL now starts at **Stage B** (not D); `val_loss` = autoencoder
> reconstruction. This supersedes the loss/curriculum descriptions throughout
> this README and the design doc; §27 is the source of truth. Verified at smoke
> scale (no collapse); **not yet at `small`+.**

> **See [`notes.md`](notes.md) for the full engineering log** — every bug found
> and fixed, all training-run results, and the theory/observation notes behind
> the decisions here. This README is the map; `notes.md` is the story.

## File map

| File | Design-doc section | Contents |
|---|---|---|
| `config.py` | throughout | `ModelConfig`/`TrainConfig`/`DataConfig`, each field commented with its source section. `MODEL_PRESETS` + `model_config()` give size presets (`smoke`/`small`/`base`). `TrainConfig` also holds the scaling knobs (AMP, grad-accum, LR schedule, checkpointing, per-stage step budgets). |
| `utils.py` | §3.6, §5.7.2 | Seeding, the post-hoc detach helper used by the *memory* truncation (the inner loop cuts its graph mid-loop instead — see `hrm_loop._TruncationSchedule` and notes §11.1), plateau detector. |
| `decay_gate.py` | §0, §3.3 | Per-channel **diagonal decay gate** — the discretized negative-diagonal state transition (S4/Mamba `exp(-softplus·dt)` form), shared by the L- and H-modules. Depth-boundedness comes from `norm.py`'s hard-norm, not from this gate; the gate is the state carry path and shapes on-shell dynamics. |
| `profile_transition.py` | §3.3, §5.5 | Measures the L-gate's share of a training step, to decide whether the loop-constant-`e` caching rewrite is worth doing. Reuses `bench_throughput.py`'s synthetic harness. |
| `norm.py` | §0, §3.3 | MagicNorm: Pre-LN wrapper + hard normalization at recurrent-module exit. |
| `chunker.py` | §3.1, §5.1 | `SegmentAnyTextChunker` — "SaT Capped" (SaT sentence boundaries + punctuation-aware length capping), matching Thought Gestalt's preprocessing. `encode_recent` slices recent raw tokens for the input lane. `TokenIdBoundaryChunker` is a legacy token-id fallback. |
| `gestalt_memory.py` | §1.2, §3.6, §4.2 | FIFO memory bank with role tags (USER/SELF/SYSTEM) and truncated-gradient cross-attention reader. |
| `hrm_loop.py` | §1.1, §3.2, §3.5, §5.5 | The inner HRM loop: fast L-module, slow H-module, the diagonal decay gate, ACT adaptive depth (ponder cost rewards halting), and in-loop truncated BPTT (`_TruncationSchedule` — the §3.5 grad window, cut mid-loop on the carried states). |
| `talker.py` | §1.3 | Lightweight causal decoder reconstructing a chunk's tokens from a finished thought. Teacher forcing is right-shifted internally (learned start vector) so it can't trivially copy the input; boolean causal mask (MPS/AMP-safe). |
| `input_lane.py` | §4.1, §4.2 | Bidirectional input-lane encoder; read-only via cross-attention, never writes recurrent self-state. All-masked-row guard against NaN. |
| `ema_target.py` | §2.1, §3.4 | `ChunkEncoder` (the shared latent producer) + `EMATargetEncoder`, a momentum copy of the encoder (encoder-space `encode`, no projection since §26), with `state_dict`/`load_state_dict`. All-pad-row guard. |
| `losses.py` | §2, §5.5 | Scaled cosine loss (the SSL prediction term), grounded NLL (autoencoder reconstruction), ACT ponder cost, VICReg-style `variance_regularization` anti-collapse floor. |
| `model.py` | §1-§4 (notes §27) | `LatentThoughtModel`: shared `chunk_encoder`, HRM reasoner, Talker, two-lane input, `pred_head` (the loop's next-latent head). **`forward_grounded`** = pure autoencoder (encoder→Talker, **no loop/memory**) — the anchor. **`forward_self_supervised`** = the on-loop predictor run **sequentially with the gestalt memory** (`h_t=loop(z_t,memory)`→write→`pred_head(h_t)` predicts chunk t+1's EMA latent + variance floor + ACT ponder). `latent_collapse_metric`. |
| `curriculum.py` | §5 (notes §27) | Stage A→F flags/loss-plan: **A** autoencoder-only, **B** loop+SSL (memory detached), **C** un-detach memory, **D** ACT, **E** consolidate, **F** dialogue. Gates on plateau *or* fixed per-stage budgets; `state_dict`/`load` for resume. |
| `data.py` | §3.1, §5.6 | Real-text pipeline: HF mixture stream (`iter_hf_mixture`), single-dataset stream (`iter_hf_single`), offline synthetic-text + stub-SaT fallback, `ReservePadTokenizer` (id-0 = PAD), in-pipeline chunking + length bucketing, and `CachedChunkDataset` (map-style over a pre-chunked shard cache). |
| `train.py` | §5, §5.7 | Smoke training entry (offline by default; `LATENT_USE_HF=1` = real streaming mixture). Device-aware; grounded/SSL loss orchestration. |
| `run_small.py` | — | ~1M-token smoke run on real text (pile-10k) via the offline stub chunker (only `datasets` needed). |
| `train_real.py` | — | ~1.5M-token run with the **real gpt2 tokenizer** (decodable output); writes `runs/model.pt` + `runs/metrics.json`. |
| `plot_metrics.py` | — | Renders `runs/metrics.json` to `runs/loss_curves.png` (val loss, train NLL/SSL, and the latent-std collapse monitor, with stage bands). |
| `generate.py` | §1.3 (notes §27) | Use a checkpoint: read the prompt through the HRM loop (building its gestalt memory) → the loop predicts the next encoder-space latent (`predict_next_latent`) → the **codec Talker** decodes that latent → detokenize. `--score` reports **autoencoder reconstruction** perplexity (no loop). Pre-§26 checkpoints load with a warning. |
| `data_prep.py` | scaling | Offline pre-chunking: run chunking + tokenization once, write sharded chunk tensors + manifest to a cache dir. |
| `trainer.py` | scaling | `Trainer`: AMP autocast, gradient accumulation, warmup→cosine LR schedule, atomic checkpoint/resume (schedule-exact since notes §15.3), fixed-budget curriculum gating, collapse monitor. Loss per step: autoencoder anchor (always) + on-loop SSL from Stage B. |
| `train_scaled.py` | scaling | Scale-oriented entry point: trains from the pre-chunked cache via `Trainer`, using a `MODEL_PRESETS` size preset. `--lr-schedule per-stage` (default) gives each stage its own warmup→cosine (the notes §12 curriculum fix); `global` reverts. |
| `baseline_gpt.py` | comparison | Standard decoder-only GPT baseline (two presets: `same-params` / `same-compute`) for a fair-scale memorization comparison vs the latent model. See notes §13.3. |
| `rocm_smoke.py` | scaling | Validate the training path stays finite under bf16 on an AMD ROCm GPU (Strix Halo, gfx1151) — synthetic tensors, no data. First real execution of the AMP path. See `STRIX_HALO.md`. |
| `bench_throughput.py` | scaling | Tokens/sec + wall-clock ETA sweep across batch sizes, to size a run on a given GPU before prepping data. Synthetic, no data. |
| `STRIX_HALO.md` | scaling | Ops guide for the ROCm / Strix Halo run: setup, the two scripts above, throughput→ETA table, vectorization analysis, and a go/no-go checklist. |

## Running it

```bash
pip install torch
python train.py                  # offline: synthetic text + stub SaT-Capped chunker, no downloads
```

This trains a small model through the full curriculum and prints per-stage
losses and stage transitions. For real text, install the extra deps and opt in:

```bash
pip install torch datasets wtpsplit transformers
LATENT_USE_HF=1 python train.py  # streams the config.DataConfig mixture, real SaT chunker
```

The real path streams a mixture of general prose (FineWeb-Edu), long documents
(PG-19, Wikipedia, arXiv), and a reasoning slice (OpenWebMath, code) — see
`DataConfig` in `config.py` to adjust sources/weights. `vocab_size` is set
automatically from the tokenizer on the real path (ids are offset by +1 so id 0
stays reserved for PAD).

## Small real run: checkpoint, graphs, inference

A ~1.5M-token run on real text (`NeelNanda/pile-10k`) with the real gpt2
tokenizer, producing a checkpoint, training curves, and a usable inference
script. Only `datasets` + `transformers` are needed (no SaT download — regex
sentence boundaries + gpt2 subwords):

```bash
pip install torch datasets transformers matplotlib
python train_real.py     # writes runs/model.pt + runs/metrics.json  (walks A→F)
python plot_metrics.py   # writes runs/loss_curves.png
python generate.py "The history of science shows that"   # generate
python generate.py --score "some text to score"          # perplexity
```

The gpt2 tokenizer is loaded from a local `../gpt2_tok/` dir with
`TRANSFORMERS_OFFLINE=1` (avoids flaky Hub HEAD timeouts); download the five
tokenizer files there once if it's missing.

**This is a smoke-scale model — it is NOT gpt2-quality and can't be at this
scale** (perplexity ~3k vs. random 50k; output is real subword text but
incoherent). Its purpose is to exercise the full architecture end-to-end and
feed `generate.py`. See `notes.md` for the numbers and the honest read.

## The two losses, split by role (notes §27)

- **Reconstruction = a pure autoencoder codec.** `forward_grounded`: encode chunk
  *t* → the Talker decodes chunk *t*, parallel over chunks, **no HRM loop, no
  memory**. Trains encoder + Talker. Because a constant latent can't reconstruct
  varied chunks, this is the always-on **anti-collapse anchor**, and it's what
  `val_loss` / `--score` measure.
- **Prediction = the HRM loop, run sequentially with memory.** `forward_self_
  supervised`: per chunk *t*, `h_t = loop(z_t, memory)` reads the *accumulating*
  gestalt memory (Thought Gestalt cross-thought reasoning), writes `h_t` back,
  and `pred_head(h_t)` predicts chunk *t+1*'s EMA-target latent (scaled cosine, k
  =4). Trains the loop + encoder + `pred_head` + the memory readers/writers to
  reason forward. Gradients via the inner-loop 2→5 truncation; memory credit
  bounded by the memory window.

Collapse defenses (a shared encoder under a predictive loss can collapse — the
first real run hit cosine → 0.996; notes §5): the always-on autoencoder anchor,
the **variance floor** (`losses.variance_regularization`) on the shared latent,
and the **slow EMA target** (momentum 0.996). The `latent_std` monitor is logged
every eval; `val_loss` is reconstruction-only so it stays comparable across
stages. The primary collapse signal is a **`val_loss` regression when SSL turns on
at Stage B** (see below); `latent_std` is width-dependent (§ TRAINING.md).

History (notes §24–§27): the SSL was once a separate linear `ssl_proj` head
(removed §26 — the on-loop loss is more collapse-robust and the projection head
isn't load-bearing), then briefly an on-loop-but-*parallel* predictor sharing the
loop with reconstruction (restructured §27 — the loop was fighting itself, and
the parallel/empty-memory SSL never trained the gestalt memory). Verified no
collapse at smoke scale; **re-validate at `small`+ before the big run.**

## Scaling up

The smoke path (`train_real.py` / `run_small.py`) chunks on-the-fly every epoch,
loads single-process, and checkpoints only at the end — fine for 1M tokens, not
for a real run. The scaled path fixes that:

```bash
# 1. Pre-chunk the corpus ONCE into a shard cache (needs `datasets` + local gpt2
#    tokenizer in ../gpt2_tok). Chunk dims come from the size preset.
#    Multi-config datasets NEED --name (else `datasets` picks the default config
#    -- for fineweb-edu that's the full multi-TB corpus); use --streaming so a
#    --max-tokens-capped prep doesn't download a whole snapshot first.
python data_prep.py --dataset HuggingFaceFW/fineweb-edu --name sample-10BT \
    --streaming --preset small --max-tokens 1200000000

# 2. Train from the cache (no tokenizer/SaT at train time; fast, worker-friendly).
python train_scaled.py --preset small --cache chunk_cache --device cuda --amp \
    --batch-size 16 --grad-accum 4 --num-workers 8

# resume:
python train_scaled.py --preset small --cache chunk_cache --resume runs/scaled/checkpoint.pt
```

- **Size presets** (`config.MODEL_PRESETS`): `smoke` (the 1M-token model), `small`
  (d_model 512, ~100M+ params), `base` (d_model 768). `data_prep.py` and
  `train_scaled.py` must use the **same** preset (chunk dims are baked into the
  cache; `CachedChunkDataset` asserts they match).
- **Trainer** adds AMP (`--amp`, bf16/fp16; enable on CUDA), gradient
  accumulation (`--grad-accum`), a warmup→cosine LR schedule, periodic
  checkpoint/resume, and **fixed per-stage step budgets** (`--stage-steps
  A,B,C,D,E,F`) instead of the smoke's plateau hack. It logs the `latent_std`
  collapse monitor every eval (dropout-free eval-mode reading since review 7,
  so true collapse reads ~0) so a regression is caught early. Checkpoints
  carry the resolved training schedule and a resume with a drifted command
  line warns loudly; `--archive-every N` keeps numbered rollback snapshots.
- The anti-collapse recipe (reconstruction anchor always-on, variance floor,
  momentum 0.996) is carried over; the separate SSL projection head was removed
  and SSL now runs on the HRM loop (notes §26).

Verified end-to-end offline (prepare → train → checkpoint → resume → multi-worker)
on a tiny synthetic cache; **no large training run has been done**.

> **Pre-scale review (notes §11):** a second full review before the first big
> run found and fixed a critical bug — the inner-loop gradient truncation was
> a silent no-op (full-document BPTT, always) — plus a ponder-contaminated val
> metric, doc-length-dependent ponder scaling, warmup ramps that ignored the
> fixed stage budgets, an unmasked input-lane read, and missing RNG state in
> checkpoints.

> **Ninth pre-scale review (notes §22):** independent full pass — **A→E training
> path clean** (offline A→E + schedule-exact resume re-verified; six-invariant
> autograd audit of truncation/isolation all pass). Two *inference-path* fixes in
> `generate.py`: free-running generation decoded a chunk **before** writing its
> thought to memory (training and `--score` write first, so the Talker always
> trains with its own thought as the newest slot — the decode-time memory was
> one slot short), and the §20.2 config-field rename (`parcae_*` → `decay_*`)
> crashed `generate.load()` on every pre-rename checkpoint (legacy names now
> mapped forward; unknown fields warn instead of crash). Neither touches
> training. Also documented (not changed): Stage-E ACT's ponder/halting sees
> padded rows (an aspect of the per-batch-ACT simplification), and `_cap_span`
> drops the separator punctuation when capping over-long sentences.
>
> **Eighth pre-scale review (notes §21):** verification pass — ran the A→E path
> end-to-end offline at both `smoke` and `small` presets (+ schedule-exact
> resume), byte-compiled, and ran an adversarial gradient-flow audit. Fixed one
> real (small) bug: in Stage E the **ACT ponder-cost gradient leaked one thought
> back through the raw h/l state chain** — the ponder term reads the halting head
> at the first cycle boundary (op 3), *before* the ACT rolling truncation's first
> cut (op 5), so cross-thought credit was flowing through the raw state instead
> of only the gestalt memory (§3.6). Fixed by forcing a truncation cut at each
> thought's entry (`hrm_loop._TruncationSchedule`); this also closes the §18.2
> footgun (a thought halting in ≤ `grad_window` steps). Verified by direct
> autograd test (entering-state grad `None` after, non-`None` before) and an
> old-vs-new A→E diff (`nll`/`ssl`/`lstd` bit-identical; only `ponder`/`gen`/
> `val_loss` move ~1e-4, the size of the severed leak). Confirmed *not* bugs
> (already §3.6-documented intended behavior): the memory autograd graph spans
> the whole document in Stages C+ via transitive credit — the `memory_grad_window`
> bounds direct per-hop reach/magnitude, not graph depth (budget GPU memory via
> the bench). Memory-headroom estimate: at `small` on 128 GB the dominant term is
> the Talker logits (`N·L·vocab` retained across chunks) at ~13–26 GB @ batch 64
> plus ~2.5 GB model/optimizer — ample headroom, so throughput (not OOM) is the
> real constraint; still confirm the *GPU-visible* memory ceiling from
> `rocm_smoke.py`'s `mem_get_info` line, since 128 GB unified ≠ 128 GB allocable.
>
> **Seventh pre-scale review (notes §19):** independent pass (different model
> family than the prior six) before the big run. No critical bug in the A→E
> scaled path. Fixed: `data_prep.py` couldn't select a HF dataset *config*
> (`--name`/`--streaming` added — the documented fineweb-edu `sample-10BT` plan
> was otherwise unexecutable and would begin a full-corpus download); the
> Talker got **end-of-chunk supervision** (first pad position of short chunks
> trained as EOS — without it generation could never terminate a chunk and
> every generated chunk carried an untrained 48/64-token tail that was
> re-encoded into the next latent); `generate.py --score` was chunk-weighted
> (short chunks over-weighted; incomparable with the token-weighted baseline);
> the `latent_std` monitor read dropout noise (~0.05–0.08) instead of ~0 at
> true collapse (now eval-mode); checkpoints now embed the training schedule
> and warn on resume drift; numbered checkpoint archives; a stale-mixed-cache
> guard in `CachedChunkDataset`. Documented, not changed: Stage F trains user
> turns through the grounded/self path (two-lane separation not exercised by
> the current Stage F loop); the variance floor still sees train-mode latents
> (at full collapse its effective hinge is ~4–5× weaker than nominal — the
> grounded anchor at frequency 1.0 remains the real defense; treat
> `latent_std ≲ 0.1` as collapse at scale).

> **Third pre-scale review (notes §15):** found and fixed a critical
> inference-path bug — generation fed the SSL-projection-space predictor's
> output to the HRM loop as an encoder-space latent; a new gradient-isolated
> `gen_predictor` head (trained from Stage D, logged as `gen`) restores a
> *trained* next-latent map for generation — plus a truncation footgun
> (`grad_window >= 8` silently reverted the §11 fix), a checkpoint/resume LR
> off-by-one, non-atomic checkpoint writes, whole-document tokenization waste
> in prep, and a corpus-ordered val split. Two things are documented, not
> changed: the ACT halting head gets gradient only from the ponder cost (so
> expect `halt_prob → 1` in Stage E — the depth dial doesn't learn yet), and
> memory truncation bounds direct-but-not-transitive credit (activation memory
> still spans the document in Stages C+; trust the bench's peak-GB).

## What's simplified relative to the design doc, and why

The design doc is a pure architecture spec with several explicitly-flagged
open questions (§6). This implementation makes concrete, commented choices
where the doc leaves things open, and keeps the two "honest limits" from
§4.3 honest in code too:

- **Chunk boundaries** use the real method: `SegmentAnyTextChunker` implements
  "SaT Capped" exactly as Thought Gestalt's paper describes it — SaT
  (Segment Any Text; Frohmann et al., 2024) predicts sentence boundaries,
  and a punctuation-aware fallback recursively splits any sentence longer
  than `max_chunk_len` tokens into shorter coherent spans. `sat_model` and
  `tokenizer` are dependency-injected (see `train.build_sat_chunker`). The
  real path (`LATENT_USE_HF=1`) injects the actual SaT model + a HuggingFace
  tokenizer; the offline default injects stub `RegexSentenceSegmenter` +
  `WhitespaceStubTokenizer` (data.py) so the *exact same SaT-Capped code path*
  runs over synthetic text with no downloads. A **learned** chunk-boundary
  policy that interacts with adaptive inner-loop depth (rather than a fixed
  segmentation model) is still an open question per §6.
- **ACT halting** uses an expected-value (soft) ponder cost rather than a
  hard, stochastic halting/REINFORCE mechanism, to keep the training graph
  simple and fully differentiable — a reasonable implementation choice
  where the doc specifies the *behavior* ("adaptive depth, ACT-style") but
  not the exact halting mechanics.
- **The anti-sycophancy auxiliary loss** flagged in §4.3 as a genuine open
  requirement (the lane separation is only an *affordance* without it) is
  *not* implemented — it would need real dialogue data with contrastive
  agree/disagree labels, which the synthetic dataset doesn't have. The
  two-lane architecture and role tagging it would act on are fully wired
  up in `input_lane.py` / `gestalt_memory.py` / `model.py`.
- **Cross-call memory persistence** (§4.2: "memory has to survive across
  calls, not just within one generation") is demonstrated within a single
  Python process (`GestaltMemoryBank` persists across turns within one
  dialogue in `train_stage_f`); serializing it across actual separate API
  calls is an infrastructure concern outside this reference implementation.
- **No large training run has been done.** Everything is verified at
  smoke/offline scale. Consequently: the model is not near gpt2 quality; the
  `--amp` path is implemented but untested (no CUDA available during
  development, so it defaults off — sanity-check it on the first CUDA run);
  and `CachedChunkDataset` loads shards into RAM (fine to a few million
  examples; switch to memory-mapping for much larger corpora — the on-disk
  format is unchanged).
- **ACT halting is per-batch**, not per-thought (a single halting decision per
  step, soft/expected-value ponder cost). Per-thought adaptive depth (§1.1)
  would need a real ACT accumulator.
- **Stage A skips the H-update entirely** (the chunk encoder's latent feeds the
  Talker directly), where spec §5.1 describes "one H-update, no L-iteration".
  Stage-A memory slots therefore hold encoder latents, a distribution the
  loop's readers re-learn from Stage B (whose warmup exists for exactly that).
- **The scaled prep path (`data_prep.py`) uses regex sentence boundaries + the
  gpt2 tokenizer, not the SaT model** — same SaT-Capped capping logic, stub
  boundary detector. Only `train.py LATENT_USE_HF=1` wires real SaT. Training
  and inference are consistent (generate.py uses the same regex chunker), but
  a big run from the cache is trained on regex boundaries.
- **The Stage F loop trains every dialogue turn — user turns included —
  through the grounded (self-lane) path**: user text is injected into the HRM
  recurrent state and reconstructed by the Talker, so the §4.2 two-lane
  separation is wired but not actually exercised by Stage F training as
  written. Fine for the A→E run (Stage F isn't part of it); must be fixed
  before Stage F fine-tuning is treated as real.
