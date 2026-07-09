# Latent-Thought Reasoning Architecture — Reference Implementation

A PyTorch implementation of the architecture described in
`latent-thought-architecture.md`, combining JEPA-Reasoner, HRM-Text,
Thought Gestalt, and Parcae into a single model that thinks in latent
"thoughts" (chunk-level vectors), each decoded into tokens by a separate
Talker module, using a Parcae-stabilized HRM loop as the thinking
mechanism.

This is a from-scratch reference implementation meant to be readable and
traceable back to the design doc, not a production training pipeline. It runs
end-to-end (Stages A→F) both **offline** (synthetic text, no downloads) and on
**real text** (a HuggingFace corpus), can **save a checkpoint**, plot training
curves, and **generate** from the model.

> **See [`notes.md`](notes.md) for the full engineering log** — every bug found
> and fixed, all training-run results, and the theory/observation notes behind
> the decisions here. This README is the map; `notes.md` is the story.

## File map

| File | Design-doc section | Contents |
|---|---|---|
| `config.py` | throughout | `ModelConfig`/`TrainConfig`/`DataConfig`, each field commented with its source section. `MODEL_PRESETS` + `model_config()` give size presets (`smoke`/`small`/`base`). `TrainConfig` also holds the scaling knobs (AMP, grad-accum, LR schedule, checkpointing, per-stage step budgets). |
| `utils.py` | §3.6, §5.7.2 | Seeding, the post-hoc detach helper used by the *memory* truncation (the inner loop cuts its graph mid-loop instead — see `hrm_loop._TruncationSchedule` and notes §11.1), plateau detector. |
| `parcae.py` | §0, §3.3 | Spectral-norm-constrained (discretized negative-diagonal) recurrent state transition, shared by the L- and H-modules. |
| `norm.py` | §0, §3.3 | MagicNorm: Pre-LN wrapper + hard normalization at recurrent-module exit. |
| `chunker.py` | §3.1, §5.1 | `SegmentAnyTextChunker` — "SaT Capped" (SaT sentence boundaries + punctuation-aware length capping), matching Thought Gestalt's preprocessing. `encode_recent` slices recent raw tokens for the input lane. `TokenIdBoundaryChunker` is a legacy token-id fallback. |
| `gestalt_memory.py` | §1.2, §3.6, §4.2 | FIFO memory bank with role tags (USER/SELF/SYSTEM) and truncated-gradient cross-attention reader. |
| `hrm_loop.py` | §1.1, §3.2, §3.5, §5.5 | The inner HRM loop: fast L-module, slow H-module, Parcae stability, ACT adaptive depth (ponder cost rewards halting), and in-loop truncated BPTT (`_TruncationSchedule` — the §3.5 grad window, cut mid-loop on the carried states). |
| `talker.py` | §1.3 | Lightweight causal decoder reconstructing a chunk's tokens from a finished thought. Teacher forcing is right-shifted internally (learned start vector) so it can't trivially copy the input; boolean causal mask (MPS/AMP-safe). |
| `input_lane.py` | §4.1, §4.2 | Bidirectional input-lane encoder; read-only via cross-attention, never writes recurrent self-state. All-masked-row guard against NaN. |
| `ema_target.py` | §2.1, §3.4 | `ChunkEncoder` (the shared latent producer) + `EMATargetEncoder`, a momentum copy of the encoder **and** the SSL projection head, with `state_dict`/`load_state_dict` for checkpointing. All-pad-row guard. |
| `losses.py` | §2, §5.5, §2.4 | Scaled cosine loss, grounded NLL, ACT ponder cost, and the VICReg-style `variance_regularization` anti-collapse floor. |
| `model.py` | §1-§4, §2.4 | `LatentThoughtModel`: shared `chunk_encoder`, HRM reasoner, Talker, two-lane input, `ssl_proj` (separate SSL head), `forward_grounded` (reconstruction), `forward_self_supervised` (SSL + variance), `latent_collapse_metric`. |
| `curriculum.py` | §5 | Stage A→F flags/loss-plan. Gates on validation-loss plateau *or* fixed per-stage step budgets; `state_dict`/`load` for resume. |
| `data.py` | §3.1, §5.6 | Real-text pipeline: HF mixture stream (`iter_hf_mixture`), single-dataset stream (`iter_hf_single`), offline synthetic-text + stub-SaT fallback, `ReservePadTokenizer` (id-0 = PAD), in-pipeline chunking + length bucketing, and `CachedChunkDataset` (map-style over a pre-chunked shard cache). |
| `train.py` | §5, §5.7 | Smoke training entry (offline by default; `LATENT_USE_HF=1` = real streaming mixture). Device-aware; grounded/SSL loss orchestration. |
| `run_small.py` | — | ~1M-token smoke run on real text (pile-10k) via the offline stub chunker (only `datasets` needed). |
| `train_real.py` | — | ~1.5M-token run with the **real gpt2 tokenizer** (decodable output); writes `runs/model.pt` + `runs/metrics.json`. |
| `plot_metrics.py` | — | Renders `runs/metrics.json` to `runs/loss_curves.png` (val loss, train NLL/SSL, and the latent-std collapse monitor, with stage bands). |
| `generate.py` | §1.3, §2.4 | Use a checkpoint: tokenize a prompt → read it through the HRM loop → predict/decode continuation with the Talker → detokenize. `--score` reports perplexity. |
| `data_prep.py` | scaling | Offline pre-chunking: run chunking + tokenization once, write sharded chunk tensors + manifest to a cache dir. |
| `trainer.py` | scaling | `Trainer`: AMP autocast, gradient accumulation, warmup→cosine LR schedule, checkpoint/resume, fixed-budget curriculum gating, collapse monitor. |
| `train_scaled.py` | scaling | Scale-oriented entry point: trains from the pre-chunked cache via `Trainer`, using a `MODEL_PRESETS` size preset. |

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

## The SSL-collapse fix (design-doc §2.4)

The first real run exposed a real pathology: with one **shared** chunk encoder
feeding both losses and the self-supervised (SSL) loss at equal weight, the SSL
loss collapsed the latent (cosine → 0.996) and *dragged reconstruction down with
it*. The fix, now the default:

- **Reconstruction (grounded) loss is the always-on anchor** — it's an
  autoencoder (encode chunk → HRM → decode same chunk), which cannot be
  satisfied by a constant latent.
- **Separate SSL projection head** (`model.ssl_proj`, with its own EMA copy in
  `ema_target.py`) so SSL can only collapse *its own* head, not the shared
  encoder.
- **SSL demoted**: cosine weight ≈0.1, EMA momentum 0.98→0.996.
- **Variance floor** (`losses.variance_regularization`): a dormant safety net
  that only activates as the latent nears collapse.
- **`latent_std` collapse monitor** logged every eval (see the third panel in
  `loss_curves.png`); validation is measured **reconstruction-only** so it's
  comparable across the Stage-D boundary.

Verified: with the fix, validation holds flat through Stages D/E (no regression)
and `latent_std` stays healthy. Full before/after numbers in `notes.md`.

## Scaling up

The smoke path (`train_real.py` / `run_small.py`) chunks on-the-fly every epoch,
loads single-process, and checkpoints only at the end — fine for 1M tokens, not
for a real run. The scaled path fixes that:

```bash
# 1. Pre-chunk the corpus ONCE into a shard cache (needs `datasets` + local gpt2
#    tokenizer in ../gpt2_tok). Chunk dims come from the size preset.
python data_prep.py --dataset NeelNanda/pile-10k --preset small --max-tokens 100000000

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
  collapse monitor every eval so a regression is caught early.
- The anti-collapse recipe (reconstruction anchor always-on, separate SSL head,
  variance floor, momentum 0.996) is carried over unchanged.

Verified end-to-end offline (prepare → train → checkpoint → resume → multi-worker)
on a tiny synthetic cache; **no large training run has been done**.

> **Pre-scale review (notes §11):** a second full review before the first big
> run found and fixed a critical bug — the inner-loop gradient truncation was
> a silent no-op (full-document BPTT, always) — plus a ponder-contaminated val
> metric, doc-length-dependent ponder scaling, warmup ramps that ignored the
> fixed stage budgets, an unmasked input-lane read, and missing RNG state in
> checkpoints. If Stage E halts too late on the next run, bump
> `act_ponder_cost` (see notes §11.3).

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
