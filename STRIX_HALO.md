# Running the A→E pretraining on AMD Strix Halo (Radeon 8060S, ROCm)

Practical guide for taking the `small` (~153M) A→E run to a single Strix Halo
box (Ryzen AI Max, RDNA 3.5 iGPU `gfx1151`, 128 GB unified memory). The code
needs **no changes** — ROCm exposes the GPU as `torch.cuda`, so `pick_device()`
returns `"cuda"` and the bf16 AMP path works as-is. The open questions are
(1) does ROCm run it, and (2) is it fast enough. Two scripts answer both
*before* you prep a single token.

## 0. The one-paragraph verdict
Memory is a non-issue and actually an advantage: the model+optimizer is ~2.5 GB
and the ~1.2B-token chunk cache (~10 GB) loads into the 128 GB unified memory
with room to spare, so the in-RAM `CachedChunkDataset` is fine (memmap optional).
The real risk is **throughput**: `forward_self_supervised` (the on-loop predictor) is a sequential per-chunk
loop of many small ops, which is launch-overhead-bound and underutilizes any
GPU. The 128 GB is the lever — it lets you run a large batch to amortize that
overhead. Measure it before committing.

## 1. Environment (fresh Linux venv — NOT the Mac torch-2.2.2 venv)
```bash
python3.10 -m venv .venv-rocm && source .venv-rocm/bin/activate
# Install a torch ROCm wheel matching your ROCm install (6.4+/7.0 for gfx1151):
pip install --index-url https://download.pytorch.org/whl/rocm6.x torch
# Train-time deps (not needed for the two smoke/bench scripts below):
pip install transformers datasets            # tokenizer + corpus for data_prep
```
gfx1151 is newer than ROCm's historically-blessed targets; if torch doesn't see
the GPU, the usual fix is an ISA override:
```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1       # or 11.0.0, per your ROCm build
export HIP_VISIBLE_DEVICES=0
```
Unified memory: on Linux the amdgpu **GTT** pool lets the GPU allocate most of
the 128 GB; confirm your GTT size is large (kernel `amdgpu.gttsize`, or recent
defaults). This is what makes large batches possible.

## 2. Step 1 — does it run? (`rocm_smoke.py`)
```bash
cd files && python rocm_smoke.py --preset small     # add HSA_OVERRIDE if needed
```
Runs on synthetic tensors (no data). Verifies torch sees the GPU, a bf16 matmul
is finite, and one real `forward_grounded` + `forward_self_supervised` + ACT
step — **losses AND gradients** — stays finite under bf16 autocast, i.e. the ops
most likely to NaN under mixed precision (hard_normalize division, decay-gate
exp/softplus, masked softmax, CE) in both the forward and the backward kernels.
**Use bf16, not fp16** (RDNA 3.5 has bf16; keeps fp32 range, needs no
GradScaler). `PASS` == the training path is numerically safe on this GPU. As a
second line of defense, the trainer itself skips any optimizer step whose global
grad norm is non-finite (and hard-fails after 25 consecutive), so a mid-run
kernel glitch can no longer NaN the weights.

## 3. Step 2 — is it fast enough? (`bench_throughput.py`)
```bash
python bench_throughput.py --preset small --batch-size 4,16,32,64 --amp
python bench_throughput.py --preset small --batch-size 64 --stage E --amp   # ACT path
```
Sweeps batch size and reports, per batch: step time, dense tok/s (hardware
ceiling), **real tok/s** (`--fill-frac`, ~0.6 for real docs — set it to your
cache's actual non-pad ratio), peak GB, and the **ETA in days** for
`--token-budget` (default 1.2B, the embedding-corrected Chinchilla figure).

Expect dense tok/s to *climb with batch* until the GPU saturates — that plateau,
and the largest batch that fits 128 GB with headroom, is what you run at.

### Throughput → wall-clock (1.2B tokens)
| real tok/s | days for 1.2B |
|---|---|
| 200 (MPS-like, batch 4) | ~69 |
| 500 | ~28 |
| 1,000 | ~14 |
| 2,000 | ~7 |
| 5,000 | ~3 |
| 10,000 | ~1.4 |

If large-batch real tok/s lands in the thousands, the run is days and this box is
viable. If it's stuck in the hundreds even at large batch, the sequential chunk
loop is the bottleneck (next section), not the GPU.

## 4. Why throughput is loop-bound, and what can actually be sped up
`forward_self_supervised` walks a document's chunks **sequentially** through the HRM loop. What that means
for optimization:

- **Already batched** (notes §11.7): the chunk *encoder* runs once over all
  chunks, not per-chunk. Done.
- **Inherently sequential — leave it:** the *thought recurrence* across chunks.
  Thought `t` seeds its state from thought `t-1` and cross-attends to the gestalt
  memory of thoughts `0..t-1`, which grows as you go. This is the architecture's
  sequential thought loop (§1) — it cannot be parallelized across chunks without
  changing semantics. The L/H steps *within* a chunk are likewise a recurrence.
- **The Talker is already off the sequential path.** Reconstruction is a pure
  autoencoder codec (`forward_grounded`) that decodes all `B*N` chunks in **one
  parallel Talker call** with an empty memory — it is not inside the loop. The
  sequential predictor (`forward_self_supervised`) uses only a cheap linear
  `pred_head`, not the Talker. So the per-step sequential cost is the **HRM loop's
  L/H recurrence + memory cross-attention**, per chunk — that's the thing large
  batch amortizes.

**Order of operations:** (1) large batch — free, no code change, measured by the
bench; (2) the HRM thought recurrence stays sequential by design (it cannot be
parallelized across chunks without changing semantics). If the ETA is still
unacceptable, the levers are budget/preset/hardware, below.

If none of that gets the ETA acceptable, the levers are: smaller token budget
(the 1.0–1.5B bracket has slack), a smaller preset, or different hardware.

## 5. Go / no-go checklist before prepping data or launching
- [ ] `rocm_smoke.py` prints **PASS** under `--amp-dtype bf16`.
- [ ] `bench_throughput.py` real tok/s at the max-fitting batch gives an ETA you
      can live with (rule of thumb: target ≲ 1–2 weeks for the full budget).
- [ ] peak GB at that batch leaves comfortable headroom under 128 GB.
- [ ] then: `data_prep.py` (parallelize if 1.2B tokens is slow single-process),
      `train_scaled.py --preset small --amp --amp-dtype bf16 --lr-schedule per-stage`
      with a batch size from the bench, and a short real-data shakedown watching
      `latent_std` across a stage boundary before the full launch.

## 6. Caveats carried from the rest of the review
- **AMP was never run in development** (no CUDA/ROCm) — `rocm_smoke.py` is the
  first execution of it. Treat a PASS as necessary, not sufficient; watch
  `latent_std` and loss finiteness over the first few hundred real steps too.
- **Smoke-tuned hyperparameters** — `cosine_loss_k` (width-dependent, tuned at
  d=192 not 512), `act_ponder_cost` (0.01, set before the M7 per-thought-mean
  fix), and the anti-collapse weights need eyeballing at scale.
- **~1.2B tokens is an embedding-corrected estimate, not a fitted optimum** — the
  20:1 constant is borrowed from next-token LM; a small token sweep would
  calibrate it for this reconstruction+SSL objective.
- **Data source (notes §15.6)** — pile-10k holds only ~20M usable tokens; the
  1.2B budget needs a big single corpus (e.g. fineweb-edu `sample-10BT`) or
  mixture support wired into `data_prep.py`. Time a 1k-doc prep dry run first.
- **Stage E expectation (notes §15.5)** — the halting head is trained only by
  the ponder cost, so `halt_prob → 1` (always halt at minimum depth) is the
  *expected* Stage-E behavior, not a regression; don't burn run time tuning
  `act_ponder_cost` against it. (notes §21.5 fixed a small ACT gradient leak —
  the ponder cost was reaching one thought back through the raw h/l chain; the
  fix is in `hrm_loop._TruncationSchedule` and does not change this `halt_prob → 1`
  expectation.)
- **Peak memory (notes §15.5, §21.2)** — activation graphs span whole documents
  in Stages C+ (transitive memory credit — intended per §3.6, *not* a leak the
  `memory_grad_window` should have stopped); the bench's peak-GB already includes
  this. At `small`, the single largest activation term is the **Talker logits**
  (`N·L·vocab` = 32·64·50258 ≈ 103M elements/batch-item, retained across all
  chunks until backward): ~13 GB bf16 @ batch 64 (up to ~26 GB if cross-entropy
  keeps an fp32 copy), on top of the ~2.5 GB model+optimizer. So OOM is unlikely
  at `small` on 128 GB — but `--grad-accum N` is the escape hatch if a batch
  doesn't fit (N smaller micro-batches, each freeing its graph, ~N× less
  activation memory at the same effective batch).
- **"128 GB" is not necessarily allocable to the GPU** — unified memory is shared
  with the CPU/OS and gated by the amdgpu GTT pool (§1). The real ceiling is what
  `rocm_smoke.py` prints on startup: `device memory: XXX GB total, YYY GB free`
  (`torch.cuda.mem_get_info()`). Check that number against the peak-GB the bench
  reports **before** committing a batch size — don't assume the full 128 GB.
