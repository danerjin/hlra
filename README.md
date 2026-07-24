# Latent-Thought Reasoning Architecture — Reference Implementation

A PyTorch reference implementation of the architecture in
[`latent-thought-architecture.md`](latent-thought-architecture.md), combining **JEPA-Reasoner**,
**HRM-Text**, **Thought Gestalt**, and **Parcae** into a model that thinks in latent "thoughts"
(chunk-level vectors), each decoded into tokens by a separate **Talker**, using a bounded recurrent
**HRM loop** as the reasoning mechanism.

It runs end-to-end (Stages A→F) both **offline** (synthetic text, no downloads) and on **real text**
(a HuggingFace corpus), checkpoints, plots training curves, and generates.

> This README is the map; [`latent-thought-architecture.md`](latent-thought-architecture.md) is the
> design. The full engineering history (every revision and dead end) is in
> [`archive/`](archive/) — read it only if you want the story of how the design got here.

## The design in one screen

Two objectives share the chunk encoder but touch the rest of the model **disjointly**:

- **Reconstruction = a pure autoencoder codec.** `encode chunk t → Talker decodes chunk t`, no loop,
  no memory. Trains the **encoder + Talker**. It's the always-on anti-collapse anchor (a constant
  latent can't reconstruct varied chunks) and is what `val_loss` / `--score` measure.
- **Prediction = the HRM loop, run sequentially with memory.** Per chunk *t*: `h_t = loop(z_t,
  memory)` (the loop reasons while reading its accumulating gestalt memory) → write `h_t` → `pred_head
  (h_t)` predicts the next chunk's EMA-target latent. Trains the **loop + encoder + pred_head +
  memory**. This is where reasoning and cross-thought memory live.

The loop is in *prediction only* on purpose: making one thought both decode the current chunk and
predict the next pulls it two ways. Removing it from reconstruction leaves a clean codec and frees the
loop to reason forward. See the design doc §2.3.

## File map

| File | Contents |
|---|---|
| `config.py` | `ModelConfig` / `TrainConfig` / `DataConfig`, size presets (`smoke`/`small`/`base`/`large`/`xl`, plus the wide-thought `*-w3` variants). |
| `chunker.py` | `SegmentAnyTextChunker` ("SaT Capped": sentence boundaries + punctuation-aware length capping). |
| `ema_target.py` | `ChunkEncoder` (shared latent producer) + `EMATargetEncoder` (momentum copy; encoder-space target). |
| `decay_gate.py` | `DiagonalDecayGate` — the per-channel `exp(-softplus·dt)` carry path (Parcae's diagonal case). |
| `norm.py` | MagicNorm: `hard_normalize` (the ‖h‖=√d shell that bounds the loop at any depth) + an unused Pre-LN wrapper (the L/H cells rely on the loop's norm invariants instead). |
| `hrm_loop.py` | `HRMInnerLoop`: fast L / slow H modules, decay gate, hard-norm, memory + input cross-attention, ACT halting, in-loop truncated BPTT. |
| `gestalt_memory.py` | FIFO memory of thoughts with role tags; truncated-gradient cross-attention reader. |
| `talker.py` | The Talker: causal decoder reconstructing a chunk from a latent (internal right-shift so it can't copy). |
| `input_lane.py` | Read-only bidirectional input-lane encoder (Stage F). |
| `model.py` | `LatentThoughtModel`: `forward_grounded` (autoencoder codec, the anchor), `forward_self_supervised` (on-loop sequential predictor + memory), `predict_next_latent` (generation), `latent_collapse_metric`. |
| `losses.py` | Scaled cosine (prediction), grounded NLL (reconstruction), ACT ponder cost, variance floor. |
| `curriculum.py` | Stages A→F: A autoencoder-only, B loop+SSL (memory detached), C un-detach memory, D ACT, E consolidate, F dialogue. |
| `data.py` / `data_prep.py` | Real-text pipeline (HF streams) + offline synthetic fallback; offline pre-chunking into a shard cache. |
| `trainer.py` / `train_scaled.py` | Scale-ready trainer (AMP, grad-accum, LR schedule, checkpoint/resume, per-stage budgets) + its entry point. |
| `train.py` / `train_real.py` / `run_small.py` | Smoke entry points (offline; ~1M-token real run). |
| `generate.py` | Load a checkpoint → read prompt → the loop predicts the next latent (rescaled onto the encoder-latent shell; the cosine loss trains direction, not scale) → the codec Talker decodes it. `--score` = reconstruction perplexity. |
| `baseline_gpt.py` | Standard GPT baseline for a matched-scale comparison. |
| `rocm_smoke.py` / `bench_throughput.py` / `profile_transition.py` | GPU finiteness check, throughput/ETA sweep, L-gate profiler. |
| `plot_metrics.py` / `plot_comparison.py` | Render training curves / the baseline comparison. |
| `../poster_figs.py` | Poster-grade figures from a run's `metrics.json`, sized to the exact slots in `poster.py` (4.8×2.55 in, so font points are printed points). ARC-C data comes from `poster_data/arc_c.json`, which you fill in — no benchmark numbers are invented. |

## Running it

```bash
pip install torch                # offline synthetic path needs only torch
python train.py                  # offline: synthetic text + stub chunker, walks A→F

pip install torch datasets transformers matplotlib
python train_real.py             # ~1.5M-token real run (gpt2 tokenizer) -> runs/model.pt
python plot_metrics.py           # -> runs/loss_curves.png
python generate.py "The history of science shows that"
python generate.py --score "some text to score"
```

**This is smoke-scale — not gpt2 quality, and can't be at this scale.** Its purpose is to exercise the
full architecture end-to-end. For a real run, see [`TRAINING.md`](TRAINING.md) (the copy-paste guide
for the A→E scaled run) and [`STRIX_HALO.md`](STRIX_HALO.md) (the ROCm/GPU setup).

## Chatting with a trained checkpoint

Two interactive testers wrap the `generate.py` inference path (both prompt for a checkpoint path on
start, then load it once — point them at the final A→E run's `model.pt`). They share `chat_core.py`.

```bash
python chat.py                      # CLI: prompts for the checkpoint path, then a REPL
python chat.py runs/scaled/model.pt # or pass it directly

python web_chat.py                  # web UI: prompts for the path, serves http://127.0.0.1:8000
python web_chat.py runs/scaled/model.pt --port 8000
```

Both surface the model's **chunk-level "thoughts"**: text is shown split at chunk borders (`|` in the
CLI; numbered pills in the web UI), for both how the input is segmented and each generated chunk.
CLI commands: `<text>` generate · `:score <t>` perplexity · `:chunks <t>` segmentation · `:sep` toggle
borders · `:temp f` · `:n k` · `:q`. The web UI adds a debug sidebar: a Generate/Score mode switch,
chunk-visualization + input-segmentation + per-message-perplexity toggles, temperature / #chunks dials,
and a field to load a different checkpoint. (`web_chat.py` is stdlib-only — no Flask; inference runs on
CPU, single-user.) Output coherence tracks the run's scale — smoke-scale is not coherent by design.

## Scaling up

```bash
# 1. Pre-chunk the corpus ONCE into a shard cache (needs datasets + local gpt2 tokenizer in ../gpt2_tok).
python data_prep.py --dataset HuggingFaceFW/fineweb-edu --name sample-10BT --streaming \
    --preset small --max-tokens 1200000000
# 2. Train from the cache (no tokenizer/SaT at train time; worker-friendly).
#    The anti-collapse flags are NOT optional -- without them the predictor collapses and
#    generation is mush, while val_loss stays perfect. See TRAINING.md §0 and §3.
python train_scaled.py --preset small --cache chunk_cache --device cuda --amp \
    --batch-size 32 --stage-steps <~1 epoch, see TRAINING.md> \
    --var-weight 3.0 --lr-schedule per-stage \
    --sbert-distill-weight 10.0 --pred-head-hidden <2 x d_latent> \
    --pred-token-weight 1.0 --pred-contrastive-weight 0.3
```

Size presets (`config.MODEL_PRESETS`): `smoke` (~43M) → `small` (512-d, ~152M) → `base`/`large`/`xl`.
`data_prep.py` and `train_scaled.py` must use the **same** preset.

**Token width vs. thought width.** A token is one word; a **thought** is a whole chunk of many
tokens. So the thought/chunk-latent width can be a multiple of the token width — `d_latent =
latent_mult · d_model` — with the chunk encoder, gestalt memory, HRM loop, `pred_head`, and EMA target
at `d_latent`, and only the token-level pieces (token embeddings, the Talker's token stream, the input
lane's raw tokens) at `d_model`. `latent_mult = 1` (all five baseline presets) is `d_latent == d_model`
and an **exact no-op** — byte-identical to the pre-`latent_mult` code. The `*-w3` presets
(`small-w3`/`base-w3`/`large-w3`/`xl-w3`) are `latent_mult = 3`, each rebalanced to its baseline tier's
parameter budget by trading token width for thought width — a matched-param A/B for "a chunk latent
needs more capacity than a token." Widening the thought moves the anti-collapse machinery into the
wider space, so `cosine_loss_k` / the variance floor should be re-tuned at `d_latent`. See the design
doc §1.1.

## Evaluating a trained checkpoint

`files/lm_eval_adapter.py` exposes the model to EleutherAI's
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) as a `TemplateLM`,
and `files/run_lm_eval.py` is the one-command runner:

```bash
pip install "lm_eval==0.4.4"
python files/run_lm_eval.py --ckpt runs/scaled/model.pt                 # default: lambada_openai,hellaswag
python files/run_lm_eval.py --ckpt runs/scaled/model.pt --tasks reasoning --output results/lm_eval.json
python files/run_lm_eval.py --ckpt runs/scaled/model.pt --tasks arc_challenge --score-mode latent_cos
```

**Scoring is at chunk granularity, not token.** The model has no native token-level conditional
log-probability, so the adapter *cannot* use the reconstruction path — that leaks the answer into
its own conditioning (see the `lm_eval_adapter.py` module docstring). It scores a continuation the
way the model *generates*: read the context through the HRM loop, then for each continuation chunk
score the true tokens under the latent that `pred_head` forecasts off the running thought (the
tokens being scored never enter the latent they are scored under). The dependency-free core,
`_score_continuation`, is unit-testable with `lm_eval` absent: `python files/lm_eval_adapter.py`
runs a self-test (no checkpoint, no downloads).

**Two scoring modes** (`--score-mode`): `token_nll` (default) is the Talker token NLL above — a real
conditional log-likelihood, so it works for perplexity tasks (LAMBADA) *and* multiple choice.
`latent_cos` scores each continuation chunk by the cosine between the predicted latent and the true
chunk's own encoding — the SSL objective's native target, read *without* the Talker decode. It is a
**ranking** score for multiple-choice `acc` (the "which option is closest to what the loop predicted
next" reading), not a log-likelihood: use it for MC tasks, not for perplexity or `acc_norm`.

**Task choice is load-bearing.** Cloze / sentence-completion tasks map cleanly onto chunk scoring,
so the default is `lambada_openai,hellaswag`, and the `reasoning` suite
(`copa,piqa,hellaswag,arc_challenge`) is the multiple-choice set whose options are sentence- or
phrase-length. **ARC-Challenge** is wired in (`--tasks arc_challenge`) for when the model is scaled
up, but it is the hardest of these and sits near chance at `small` scale — a low number there is a
scale limit, not a scoring artifact, so it is not a small-scale headline. Tasks like `winogrande` or
MMLU, whose options differ by a *single token*, are the genuinely degenerate case for chunk scoring.
StoryCloze/ROCStories also fits well but needs a manual (gated) dataset download, so it is not in the
suite. Scoring runs on **CPU** (the shared inference path is CPU-only); datasets are fetched from the
HF Hub on first run.

**Opt-in ARC "statement" variant** (`--tasks arc_challenge_statement`) rewrites each answer option
into a full declarative sentence — a longer, more differentiated span than ARC-C's short phrases —
via `files/tasks/arc_templater.py`, with three backends: `--arc-templater deterministic` (default) is a
crude reproducible template (`"<q> The answer is <opt>"`); `regex` declarativizes the common
"What/Which is/are …?" stems into a real sentence with the verbatim option slotted in (no LLM, faithful
by construction, but only ~16% of ARC-C stems match a rule — the rest fall back to the crude template);
`ollama` (with `--arc-templater-model`, default `gemma4`) is a temp-0 LLM rewrite, cached to disk and
gated by three guards — content-drift, editorializing, and length — any of which falls back to the
verbatim-safe template. It is a **distinct, clearly-labelled**
task, never a substitute for standard `arc_challenge` — report it as "ARC-C (statement-rewritten by
&lt;model&gt;)". **Model choice dominates:** on a 24-option bake-off, `gemma4` produced faithful,
on-template rewrites 100% of the time while `phi3` passed only 46% (paraphrase drift, rambling, one
label flip) — run `python files/tasks/arc_templater.py --compare modelA,modelB --n N` to re-measure.
Rewrites can only be validated as *helpful* on the trained checkpoint.

## Status and honest limits

- **Verified at smoke and `small` (512-d) scale**, on offline synthetic and real gpt2 text: full A→E
  runs healthy, `val_loss` falls through the Stage-B predictor boundary, no latent collapse.
- **A large `small-w3` run reached a near-lossless codec (`val_loss` 0.0067) and generated mush.**
  The failure was the **predictor**, not the codec: `pred_head` collapsed to a near-constant forecast
  (`pred_collapse` 0.98) while the loop's own states stayed diverse (`hstate_collapse` 0.88).
- **Collapse monitoring — two independent failures.** `val_loss` (plus `latent_std`, width-dependent)
  watches the *encoder*. It is **blind to predictor collapse**: `forward_grounded` is encoder→Talker
  and never touches the loop, so a collapsed predictor and a good one score identically. Read
  `pred_collapse` / `hstate_collapse`, at **peak LR only** (the escape is learning-rate-gated) and over
  a ≥4-reading window (single-reading noise ≈ ±0.02). `tok_nll` is *not* a collapse signal — it is
  detached to the Talker. See `TRAINING.md` §0.
- **The upstream cause is latent anisotropy.** A reconstruction-only space is a narrow cone
  (random-pair cosine ~0.50), which puts the centroid close to every target and makes the constant
  forecast nearly optimal. Reconstruction only requires each latent to be *decodable*, and a tight
  huddle is decodable — so the encoder-side anti-collapse anchor is itself the cause of the
  predictor-side failure. Distilling a frozen sentence encoder through a *learned projection* opens the
  cone to **0.11** (below the teacher's own 0.21) and is, so far, the only thing whose predictor
  escaped. **It must run from step 0** — retrofitting recovered ~19% of the cone; prevention worked
  first try. See `latent-thought-architecture.md` §2.4 and `notes.md`.
- **The Talker is a rigid exact-latent decoder** (NLL 0.0042 under the true latent, 41.4 under a
  shuffled one) but receives a ~0.5-cos *predicted* latent at generation — the train/serve exposure
  gap, and the direct reason a good codec still produced mush.
- **ACT adaptive depth** trains the loop's executed depth but not the halting policy (the soft ponder
  cost gives no compute-vs-quality gradient; halting degenerates to minimum depth). A real ACT
  accumulator is future work.
- **`ssl_loss_weight`** (co-equal with reconstruction) is validated collapse-free but may want tuning
  at full scale.
- **Stage F** (two-lane dialogue, anti-sycophancy loss) is designed but not yet exercised.
- **Termination:** end-of-**chunk** is properly trained (PAD is a supervised stop, §19.2).
  A chunk that exactly fills `max_chunk_len` has no PAD slot and so goes unsupervised, but
  that is rare: **≳0.42%** (123 of 29,568 chunks over 1401 real documents at
  `max_chunk_len=64`; mean chunk length 19.9). Read that as a **lower bound**: the cache
  measured predates the 2026-07-11 chunker (17.2% of its chunks are ≤3 tokens), and v3
  glues those fragments into fewer, larger chunks — which both shrinks the denominator
  and pushes more chunks to the cap. End-of-**turn** is a Stage-F head
  (`--end-weight`, **off by default**; the plumbing runs but there is *no evidence yet* it
  learns — see `STAGE_F.md` §2.1). End-of-**document** does *not* exist: there is no EOS
  token and `generate.py` emits a caller-supplied chunk count. Benign for A→E (the
  objective is next-chunk prediction and docs truncate at `max_chunks_per_doc` anyway); it
  matters only for free-running generation. See `experiments.md` #5.
- The `--amp` path is implemented; sanity-check it on the first CUDA run (`rocm_smoke.py`,
  6 checks covering the training-mode and eval-mode/monitoring paths — it must end `PASS`).
- A 2026-07-10 pre-flight review (gradient-routing audit, truncation severance, A→E walk,
  resume equivalence) found the training path clean; see `notes.md` for the three
  inference/tooling fixes it landed and the items it flagged (notably the input-lane
  target leak on generic documents — **fixed 2026-07-13**, see `notes.md`).
- A second, comprehensive 2026-07-10 review (full spec+code pass, three independent adversarial
  audits, all invariants re-verified) landed a chunker rewrite (`_cap_span` no longer explodes
  long sentences into one-word chunks or deletes delimiters — **re-prep any cache built before
  it**), an ACT ponder/halt fix for ended documents, a resume dataset-fingerprint guard, and a
  bit-exact pad-row encoder skip (~30-45% off both encoder passes); see `notes.md` for the full
  list and the flagged-not-changed items.
- A third review (2026-07-11, four independent adversarial audits) again found the A→E training
  semantics clean and landed run-robustness + data-quality fixes: a **non-finite gradient guard**
  in the trainer (a single NaN grad no longer destroys all weights via the global clip norm — the
  step is skipped, with a hard-fail after 25 consecutive), a **splitter-fragment merge** in the
  chunker ('Dr.'/'2.'-style 1-3-token chunks: 17% of the old cache → 0.3%) and a
  character-boundary hard fallback (no more U+FFFD corruption or over-cap chunks on unicode) —
  **re-prep any cache built before it (again)** — an aligned input-lane raw window, a
  generation-path fix (no double loop pass on the last prompt chunk), gradient-finiteness gates
  in `rocm_smoke.py` [4]/[5], and a `bench_throughput.py` step that mirrors the real trainer.
  Details in `notes.md`.
- A fourth review (2026-07-11, three independent adversarial audits: 86 float64
  gradient/truncation checks, a real-tokenizer chunker fuzz, and a SIGKILL-and-resume trainer
  audit) again found the A→E training semantics clean and landed three small off-path fixes:
  a `--log-every 0` crash guard in the trainer, an ACT halt-vote host-sync removal in
  `hrm_loop.py` (verified bit-identical losses/gradients; saves up to one sync per chunk per
  step in Stages D/E), and a clear error for an empty chunk cache. See `notes.md` for the
  full audit results and the accepted-as-is flags.
- A fifth review (2026-07-11, three independent adversarial audits: probe-verified gradient
  routing/SSL alignment/truncation, a bit-identical kill-and-resume trainer check, and a
  real-tokenizer data-pipeline fuzz + prep→load round-trip) found the A→E training semantics
  clean for the fifth time and landed **docs/process fixes only**: the spec's Pre-LN claim
  corrected (the hard-norm carries the stability argument; `PreNormWrapper` is unused), the
  ACT stage label fixed (D, not E+), and the stale pre-fix shakedown cache renamed with a
  DO-NOT-TRAIN marker (its manifest is indistinguishable from a fresh cache's). The post-audit
  chat testers and `generate.py` separator kwarg were reviewed clean. Details in `notes.md`.
