# STRIX_HALO.md — gfx1151 end-to-end run book (copy-paste)

From a **fresh AMD Strix Halo box** (Ryzen AI Max, Radeon 8060S iGPU, RDNA 3.5
`gfx1151`, 128 GB unified memory) to a **running A→E training job**. Every block is
copy-paste. This is the *verified* path mapped 2026-07 on `zhang-...-NucBox-EVO-X2`
— it corrects a lot of plausible-but-wrong first guesses (stock PyTorch wheels,
`HSA_OVERRIDE`, `HF_HUB_DISABLE_XET`, streaming fineweb-edu, CPU SaT). The design
lives in [`latent-thought-architecture.md`](latent-thought-architecture.md); the
generic (any-GPU) training reference is [`TRAINING.md`](TRAINING.md) and the long-form
A→E walkthrough is [`archive/TRAINING.md`](archive/TRAINING.md).

Paths below assume the repo is at `~/hlra`; adjust if yours differs.

---

## 0. TL;DR — the whole path

```
1. groups: render + video          (admin, one-time)
2. pip install torch  from AMD's gfx1151 index  (--no-cache-dir)   — NO HSA_OVERRIDE
3. export LATENT_MANUAL_LAYERNORM=1        (REQUIRED: LN-backward kernel writes NaN grads)
   optional: TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1  (flash attention; smoke-validated)
                (rocBLAS; skips slow hipBLASLt GEMM autotuning)
4. rocm_smoke.py  -> PASS   (if training later "stalls": §7.5 -- check checkpoint.pt + py-spy, NOT the log)
5. data: SaT on GPU + get the parquet on-box (HF Xet 403 escape) + data_prep --local-glob
6. queue training to auto-start after prep
```

Env vars to keep set (put in `~/.bashrc` so every shell + nohup'd job has them):
```bash
echo 'export LATENT_MANUAL_LAYERNORM=1' >> ~/.bashrc              # REQUIRED: broken LayerNorm-backward kernel (step 2)
echo 'export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1' >> ~/.bashrc  # optional: flash attention (rocm_smoke validates it)
echo 'export' >> ~/.bashrc         # rocBLAS: skips hipBLASLt's slow per-shape GEMM autotune
# (no HSA_OVERRIDE_GFX_VERSION with a native gfx1151 wheel — it BREAKS kernel launch)
```

---

## 1. Install PyTorch for gfx1151

**1a. GPU access (needs the admin if you lack sudo).** ROCm device nodes are gated
to the `render`/`video` groups; without them `torch.cuda` sees zero devices
(`hipErrorNoDevice`). Check, and if missing, have an admin add you:
```bash
groups                                   # want 'render' AND 'video'
# admin runs, then you FULLY log out and back in (new login, not just a new shell):
#   sudo usermod -aG render,video $USER
```

**1b. Install torch from AMD's gfx1151 index.** The **stock**
`download.pytorch.org/whl/rocm*` wheels do **NOT** contain gfx1151 kernels
(`get_arch_list()` has gfx1100/1101/1102 + gfx1200/1201 but not gfx1151), and
`HSA_OVERRIDE_GFX_VERSION` masquerading fails with `invalid device function` /
`no kernel image`. Use AMD's native gfx1151 index. **`--no-cache-dir` is REQUIRED**
— pip's HTTP cache uses msgpack, which crashes with `ValueError: Memoryview is too
large` on the >4 GB torch wheel.
```bash
python3.10 -m venv ~/hlra/.venv-rocm && source ~/hlra/.venv-rocm/bin/activate
pip install --pre --no-cache-dir torch --index-url https://rocm.nightlies.amd.com/v2/gfx1151/
pip install --no-cache-dir transformers datasets wtpsplit tqdm matplotlib pyarrow "numpy<2"
```
(Fallback if AMD's index is broken: scottt's self-contained gfx1151 wheels —
<https://github.com/scottt/rocm-TheRock/releases> — but those are cp311/cp312, so
make a matching venv with `uv`: `uv python install 3.11 && uv venv --python 3.11`.)

**1c. Verify — GPU visible, can compute, NO override.**
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0)); \
print(torch.cuda.get_arch_list()); x=torch.randn(2048,2048,device='cuda'); print(float((x@x).sum()))"
```
Want `True`, `Radeon 8060S Graphics`, **`gfx1151` in the arch list**, and a finite
number. **Do NOT set `HSA_OVERRIDE_GFX_VERSION`** with a native wheel — it forces an
ISA mismatch and breaks kernel launches.

**Harmless noise, ignore it:** `/opt/amdgpu/share/libdrm/amdgpu.ids: No such file`
(cosmetic PCI→name table) and `Mem/Flash Efficient attention … experimental` (SDPA
warnings; the run works).

---

## 2. The LayerNorm-backward workaround (required on the current wheel)

The 2026-04 `rocm7.13` alpha wheel has a **racy `native_layer_norm_backward`
kernel**: it writes nondeterministic NaN/Inf into LayerNorm weight/bias gradients (a
different LN param each run; both bf16 and fp32; kernel serialization doesn't help).
`rocm_smoke.py` check `[3]` fails `grads finite: False`. The model's own
`hard_normalize` (a manual norm) is fine on the same GPU, so the fix routes
`F.layer_norm` through manual primitive ops (`files/rocm_compat.py`), gated by an
env var — off by default, so it's a no-op on any working GPU:
```bash
export LATENT_MANUAL_LAYERNORM=1        # you added this to ~/.bashrc in step 0
```
Drop it once AMD ships a wheel with a fixed LN kernel. (SaT segmentation at prep is
forward-only, so it is unaffected — see step 5.)

---

## 3. Does it run? — `rocm_smoke.py` (the gate)

```bash
cd ~/hlra/files
LATENT_MANUAL_LAYERNORM=1 python rocm_smoke.py --preset small-w3     # must end PASS
```
It runs the real `forward_grounded` + `forward_self_supervised` + ACT step under
bf16 autocast and checks **losses AND gradients** stay finite (the sequential loop's
*backward* is the untested ROCm path). `PASS` = training-safe. If it `FAIL`s on
`[3]`, you forgot `LATENT_MANUAL_LAYERNORM=1`.

---

## 4. How fast / how big? — `bench_throughput.py` (optional)

```bash
LATENT_MANUAL_LAYERNORM=1 python bench_throughput.py --preset small-w3 --batch-size 16,32,64 --amp
```
**Measured on this box** (`small-w3`, manual-LN): **batch 32 fits; batch 64 OOMs.**
Two facts that matter:
- **The GPU allocates from the amdgpu GTT pool, ~68 GB — NOT the full 128 GB.** Size
  batches against what `rocm_smoke.py` prints (`device memory: XXX GB`), not 128.
- The **manual-LN workaround materializes fp32 intermediates**, costing extra
  activation memory (part of why 64 OOMs) *and* a little speed. Use `--grad-accum N`
  to reach a larger *effective* batch without the single-forward memory. A fixed LN
  kernel would buy both back.

The bottleneck is the **sequential per-chunk HRM loop** (launch-overhead-bound); a
large batch amortizes it. The thought recurrence across chunks is inherently
sequential by design and cannot be parallelized without changing semantics.

---

## 5. Data — prep the chunk cache

Training reads a **pre-chunked cache** (SaT-Capped: SaT sentence boundaries +
length capping). Three things bite on this box; all are handled below.

### 5a. Run SaT on the GPU (else prep takes WEEKS)
SaT segmentation on CPU is ~seconds/doc → **weeks** for 1.2 B tokens. `build_sat_chunker`
now moves SaT to the GPU automatically (forward-only, so the §2 LN-backward bug does
not apply); look for `[chunker] SaT on cuda (half)` in the log. Override with
`LATENT_SAT_DEVICE=cpu`. Even on GPU it's per-doc-overhead-bound (~10k tok/s here →
~a day for 1.2 B); the `--regex` fast fallback (5c) skips SaT entirely.

### 5b. Get the corpus on-box — the HF **Xet 403** wall
fineweb-edu (and many datasets) serve parquet through HF's **Xet** CDN. On this box
*every* Xet request (`cas-bridge.xethub.hf.co`) `403`'d — for **both** `datasets`
streaming **and** `hf download` — even authenticated. **`HF_HUB_DISABLE_XET=1` is
IGNORED.** The CLI is **`hf`** now (`huggingface-cli` is deprecated / no-ops). The
reliable fix: **download the parquet on a machine that CAN reach Xet (your laptop),
`rsync` it over, and prep from local files.**

```bash
# --- on your laptop / any machine with normal internet ---
pip install -U huggingface_hub && hf auth login          # paste an HF read token
# each sample/10BT shard is ~2.15 GB ≈ ~1-1.5 B tokens -> you only need 1-2 for 1.2 B:
hf download HuggingFaceFW/fineweb-edu --repo-type dataset \
  --include "sample/10BT/00[0-1]_*.parquet" --local-dir ~/fineweb_local
# ship the shards to the box (NOT the ~20 GB cache — prep on the box, disk lands there):
rsync -avP ~/fineweb_local/sample/ daniel@<box>:~/hlra/fineweb_local/sample/
```
Escape-hatch notes: (1) if `hf_xet` even on the laptop misbehaves, `pip uninstall -y
hf_xet` forces the classic LFS path; (2) a non-Xet corpus (`Skylion007/openwebtext`,
`allenai/c4 --name en`) streams fine from anywhere — the architecture only needs long
English prose; (3) **storage:** shards are 2.15 GB each and the cache is ~20 GB at the
observed 0.26 fill — do the download+prep where the disk is (the box, 1.9 TB), keeping
the laptop to ~4 GB of parquet.

### 5c. Prep from the local parquet (offline, on the box)
`--local-glob` reads parquet with **pyarrow directly** (no `datasets.load_dataset`,
which can do a hub check and hang) — zero network. **Prep is NOT resumable**: a crash
restarts from zero (delete the partial dir first), so run under `nohup`/`tmux`.

```bash
cd ~/hlra/files
# dry-run first (do NOT skip) -- confirms local read + GPU-SaT + write:
python data_prep.py --local-glob "$HOME/hlra/fineweb_local/**/*.parquet" \
  --preset small-w3 --docs 1000 --out chunk_cache_dryrun     # -> "wrote 1000 examples"
rm -rf ~/hlra/chunk_cache_dryrun

# the real cache (~a day with GPU-SaT; --max-tokens caps it; survives SSH disconnect):
nohup python data_prep.py --local-glob "$HOME/hlra/fineweb_local/**/*.parquet" \
  --preset small-w3 --max-tokens 1200000000 --out chunk_cache > prep.log 2>&1 &
tail -f prep.log
```
**Want it in minutes instead of a day?** Add **`--regex`** (fast regex sentence
chunker — an *approximation* of SaT boundaries, fine for a first run):
```bash
nohup python data_prep.py --regex --local-glob "$HOME/hlra/fineweb_local/**/*.parquet" \
  --preset small-w3 --max-tokens 1200000000 --out chunk_cache > prep.log 2>&1 &
```
Prep finishes with `wrote <N> examples (~1.2B tokens)`. **`--preset` must match
training** (`small-w3` here → pass `--var-weight 3.0` at launch, step 6).

---

## 6. Queue training to auto-start when prep finishes

Prep takes hours; this waits for it, refuses to train on a half-cache, computes the
stage budget, and launches the real run — all `nohup`'d, all logged to
`pipeline.log`. Paste while prep is still running:

A nohup'd script does NOT inherit an un-activated shell, so it must not rely on
bare `python` — it would fall back to whatever interpreter is on the default PATH
(often a torch-less base python). The robust fix is to reuse **the exact interpreter
prep is running**, which is proven to have the working gfx1151 torch — whether that's
a venv or a system-wide install. Note `readlink /proc/PID/exe` alone is NOT enough: a
venv's `python` symlinks to the base interpreter, so it resolves to e.g.
`/usr/bin/python3.10` even inside a venv. Check `VIRTUAL_ENV` too (below).

```bash
cat > ~/run_pipeline.sh <<'EOF'
#!/bin/bash
PY="__PY__"                              # the exact interpreter prep used (has the working gfx1151 torch)
export LATENT_MANUAL_LAYERNORM=1
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1   # flash attention (optional)
PREP_PID="$1"
cd ~/hlra/files
log(){ echo "[$(date '+%F %T')] $*"; }
log "STATUS: python=$PY torch=$("$PY" -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>&1)"
log "STATUS: waiting for prep (PID $PREP_PID)..."
while kill -0 "$PREP_PID" 2>/dev/null; do sleep 60; done
if [ ! -f ~/hlra/chunk_cache/manifest.json ]; then
  log "STATUS: ABORTED — prep died without manifest.json (half cache). NOT training. See prep.log."; exit 1
fi
EX=$("$PY" -c "import json,os;print(json.load(open(os.path.expanduser('~/hlra/chunk_cache/manifest.json')))['total'])")
log "STATUS: prep finished — $EX examples"
BATCH=32
STAGE_STEPS=$("$PY" -c "u=max(1,($EX//$BATCH)//6);print(f'{u},{u},{u},{u},{2*u},0')")
log "STATUS: launching training — BATCH=$BATCH STAGE_STEPS=$STAGE_STEPS"
"$PY" train_scaled.py --preset small-w3 --cache chunk_cache --device cuda --amp --amp-dtype bf16 \
  --batch-size "$BATCH" --stage-steps "$STAGE_STEPS" --var-weight 3.0 --lr-schedule per-stage \
  --num-workers 2 --log-every 50 --checkpoint-every 1000 --archive-every 5000 --out runs/scaled
log "STATUS: train_scaled.py exited (rc=$?)"
EOF
# resolve the SAME interpreter prep runs under -- venv-aware (a bare readlink of
# /proc/exe resolves a venv's python symlink to the base interpreter, so also read
# VIRTUAL_ENV); works for a venv OR a system-wide torch install:
PREP_PID=$(pgrep -f 'data_prep.py' | head -1)
PREP_VENV=$(tr '\0' '\n' < /proc/$PREP_PID/environ | sed -n 's/^VIRTUAL_ENV=//p')
if [ -n "$PREP_VENV" ]; then PY="$PREP_VENV/bin/python"; else PY=$(readlink -f /proc/$PREP_PID/exe); fi
sed -i "s|__PY__|$PY|" ~/run_pipeline.sh
chmod +x ~/run_pipeline.sh
nohup ~/run_pipeline.sh "$PREP_PID" > ~/hlra/files/pipeline.log 2>&1 &
echo "QUEUED (prep PID=$PREP_PID, python=$PY). Read anytime: tail -f ~/hlra/files/pipeline.log"
sleep 2; head -2 ~/hlra/files/pipeline.log   # verify: python=<path> torch=<ver> True
```

**Already queued a watcher without this?** Editing the file won't fix the running one
(it launched with its own environment). Kill and re-queue against the still-running
prep: `pkill -f run_pipeline.sh`, then re-paste the block above — prep is a separate
process and keeps going. Confirm what a live watcher holds with:
`tr '\0' '\n' < /proc/"$(pgrep -f run_pipeline.sh | head -1)"/environ | grep -E '^(VIRTUAL_ENV|PATH)='`.

**To launch training manually instead** (cache already prepped): compute the budget
and run — remember the box-specifics **`LATENT_MANUAL_LAYERNORM=1`,
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`, `--preset small-w3`, `--var-weight 3.0`,
batch 32**:
```bash
cd ~/hlra/files && export LATENT_MANUAL_LAYERNORM=1 TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 && export BATCH=32
export STAGE_STEPS=$(python3 -c "import json,os;m=json.load(open(os.path.expanduser('~/hlra/chunk_cache/manifest.json')));print(','.join([str(max(1,(m['total']//$BATCH)//6))]*4+[str(2*max(1,(m['total']//$BATCH)//6)),'0']))")
nohup python train_scaled.py --preset small-w3 --cache chunk_cache --device cuda --amp --amp-dtype bf16 \
  --batch-size "$BATCH" --stage-steps "$STAGE_STEPS" --var-weight 3.0 --lr-schedule per-stage \
  --num-workers 2 --log-every 50 --checkpoint-every 1000 --archive-every 5000 --out runs/scaled > train.log 2>&1 &
```

---

## 7. Monitor, disconnect, resume

**Watch** (the one number that matters is `val_loss` across the A→B boundary — no jump
= healthy; a jump when the predictor turns on at Stage B = collapse):
```bash
grep STATUS ~/hlra/files/pipeline.log            # milestones (prep done, training launched)
grep -E 'stage=(A|B)' ~/hlra/files/pipeline.log | tail -4
tail -f ~/hlra/files/pipeline.log                # or train.log for the manual launch
```
Healthy line: `[step 2000] stage=B ... logs={'nll':6.9,'ssl':0.7,...} val_loss=6.85 latent_std=0.42`.

**Disconnect safely.** `nohup` (and the queue) survive SSH drops / wifi changes /
laptop close — `Ctrl-C` the `tail` (that only stops the log-follow) and disconnect;
`tail -f` the log again when you're back. It does **not** survive a box reboot, and
prep is not resumable (training *is* — see below). For the training run, `tmux new -s
train` gives you a re-attachable live terminal (`Ctrl-B D` to detach, `tmux attach -t
train` to return).

**Stop / resume training** (training checkpoints; prep does not):
```bash
pgrep -af train_scaled.py ; kill <PID>           # graceful: writes checkpoint.pt, loses ≤1 step
# resume: re-run the SAME launch command -> auto-resumes from runs/scaled/checkpoint.pt
```
Guards on resume: hard-fails if the cache changed size (`LATENT_ALLOW_DATA_CHANGE=1`
overrides); loud `WARNING: resume schedule differs` if flags differ (fix them).

**When it finishes** (after Stage E):
```bash
python plot_metrics.py runs/scaled                                     # runs/scaled/loss_curves.png
python generate.py --ckpt ~/hlra/runs/scaled/model.pt --score "a sentence to score"
```
Off-the-server transfer of the checkpoint: **`python push_to_hf.py --ckpt
runs/scaled/model.pt --repo <you>/hlra-smallw3 --strip --bf16`** (uploads
inference-only weights, ~4–8× smaller, no git limit), or `rsync -avP
daniel@<box>:~/hlra/runs/scaled/model.pt <local>`.

### 7.5 "It's stalled!" — first suspect the LOG, not the GPU

**Root cause, found the hard way (2026-07-16): a logging bug made healthy runs look
dead for 30-50 minutes.** `Trainer._emit` routed every step line through
`tqdm.write()` whenever `self._bar is not None` — but `_make_progress_bar` passes
`disable=None` for `progress="auto"`, which makes tqdm *hide* itself off-TTY while
still returning a live object. So under `nohup`, `_bar` was not-None, every line went
through `tqdm.write()` (which does **not** flush), and with stdout block-buffered to a
file the step lines sat in the ~8 KB buffer — **~1300 steps of lag**. The run trained
fine the whole time; only the log was frozen. Fixed by checking `.disable` too.

**The real numbers (measured, `small-w3` @ batch 32):** the first optimizer step pays a
one-off **~3 min** kernel warmup; after that **~0.2 step/s**, so a 45k-step A→E run is a
**~2.5 day** job, checkpointing every 1000 steps (~83 min). That is the architecture's
cost here — the per-chunk HRM loop is sequential by design and launch-overhead-bound
(§4) — not a fault. Anything that looks like a 30-50 min "stall" is the log, not the run.

**If a run looks stalled, in this order:**
1. **Is the checkpoint advancing?** This is the only signal that never lied:
   ```bash
   ls -l --time-style=+%H:%M:%S ~/hlra/runs/scaled/checkpoint.pt   # should jump every ~1000 steps
   ```
   Advancing ⇒ it is training; **the log is behind, not the run.**
2. **`py-spy` — the only tool that actually answers it** (`pip install py-spy` in the
   VENV; needs ptrace: `sudo sysctl -w kernel.yama.ptrace_scope=0`):
   ```bash
   sudo "$(which py-spy)" dump --pid $(pgrep -f train_scaled.py | head -1)
   ```
   Take **two** dumps ~30 s apart:
   - Frames **move** (`_train_loop:281` ↔ `evaluate`) ⇒ **training normally**.
     `_train_loop` at the `torch.isfinite(total_norm)` line is the GPU **sync point** —
     that's where a step legitimately spends its wall time.
   - Frames **frozen** on one low-level call ⇒ genuinely stuck.
3. Only then look at the GPU (`dmesg | grep -i "ring.*timeout\|gpu reset"`).

**Signals that MISLED us for a full day — do not trust them:**
- **GPU% = 100** — reads 100 when the *driver* churns, or when a kernel spins. Never proof of progress.
- **Disk I/O** (`/proc/PID/io write_bytes`) — dominated by *checkpoint* writes (~1.6 GB
  each), never the compiler. Frozen in every state.
- **"It's a 20-40 min kernel compile, just wait"** — we asserted this repeatedly; `py-spy`
  showed **model frames, never a compiler frame**. There was no compile.
- **`svm_range_restore_work` in dmesg** — real, but its timestamps were **hours stale**
  (~78k-88k uptime vs ~164k current). It was a red herring; `HSA_XNACK=0` was not the fix.
- **The log itself** — buffered (above), *and* `>` truncates it on every relaunch, *and*
  `tail -f` follows the dead handle across a relaunch. Use **`tail -F`**, and confirm the
  live file with `ls -l /proc/$PID/fd/1`.

**Use `tail -F`, not `tail -f`** — a relaunch recreates `train.log` and plain `-f`
follows the dead handle and goes silent.

---

## 8. Troubleshooting (everything that bit us, with the fix)

| Symptom | Cause → fix |
|---|---|
| `Could not find a version that satisfies torch` on `whl/rocm6.x` | `rocm6.x` is a literal, not a wildcard → use a real index; for gfx1151, **AMD's `rocm.nightlies.amd.com/v2/gfx1151/`** (§1b). |
| `ValueError: Memoryview is too large` (download completes then errors) | pip's msgpack cache can't hold the >4 GB wheel → **`--no-cache-dir`** (§1b). |
| `cuda? False`, `hipErrorNoDevice` | not in `render`/`video` groups → admin `usermod -aG render,video`, re-login (§1a). |
| `invalid device function` / `no kernel image` (matmul) | wheel lacks gfx1151 kernels, or you set `HSA_OVERRIDE` → use the gfx1151 wheel, **no override** (§1). |
| `rocm_smoke.py [3] grads finite: False` (nondeterministic LN params) | broken `native_layer_norm_backward` → **`LATENT_MANUAL_LAYERNORM=1`** (§2). |
| **Run looks stalled**: log silent, GPU 100%, no `[step N]` for 30+ min | **Suspect the LOG first (§7.5).** A `tqdm.write()`/buffering bug hid ~1300 steps of output — the run was training fine. Check `checkpoint.pt`'s mtime (advancing ⇒ healthy) and take two `py-spy` dumps (frames moving ⇒ healthy). GPU%, disk I/O, and "it's compiling" are all worthless here. |
| The `scaled_dot_product_attention` **warning text** itself | cosmetic → **ignore** (the *warning* is harmless; the slow *math fallback* it implies is the row above). |
| `amdgpu.ids: No such file` | cosmetic → **ignore** (§1c). |
| `403 Forbidden … cas-bridge.xethub.hf.co`, "Reconstructing…" hangs | HF **Xet** CDN blocked; `HF_HUB_DISABLE_XET` is ignored → download elsewhere + `rsync` + `--local-glob`, or `pip uninstall hf_xet`, or a non-Xet corpus (§5b). |
| `huggingface-cli … deprecated and no longer works` | use **`hf`** (`hf download`, `hf auth login`). |
| `--local-glob` hangs after `Loading weights` | (old code) `load_dataset` hub check → `git pull` (pyarrow reader), or `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` (§5c). |
| prep at `2900% CPU`, 500 examples in ~15 min | SaT on CPU → **GPU-SaT** (auto after `git pull`), or **`--regex`** (§5a/5c). |
| ran out of disk mid-download | shards are 2.15 GB each; you need 1–2 → keep those, `rm -rf ~/fineweb_local/.cache` (partials), prep on the box (§5b). |
| OOM at batch 64 (`small-w3`) | ~68 GB GTT + manual-LN memory tax → batch 32, or `--grad-accum N` (§4). |
| `WARNING: non-finite grad norm … skipping` | one NaN grad; trainer skips the step (weights safe), hard-fails after 25 → check LR / re-run `rocm_smoke`. |
| `val_loss` rises at Stage B + `ssl`→0 + `latent_std` craters | **latent collapse** → lower `--ssl-weight` toward 0.5, relaunch from an `--archive-every` snapshot. |

---

## 9. Caveats carried from review (still true)

- **AMP + gfx1151 were never run in development** — `rocm_smoke.py` PASS is necessary,
  not sufficient; watch `val_loss`/`latent_std` over the first few hundred real steps.
- **Smoke-tuned hyperparameters** (`cosine_loss_k`, `act_ponder_cost`, anti-collapse
  weights) may want eyeballing at scale; `small-w3` already retunes `cosine_loss_k`
  and needs `--var-weight 3.0`.
- **~1.2 B tokens** is an embedding-corrected estimate, not a fitted optimum — the
  1.0–1.5 B bracket has slack, and `--max-tokens 600000000` is a fine smaller first run.
- **Fill ~0.26** on fineweb-edu here (lower than the old 0.4–0.5 guess) → more examples
  per token budget → a ~20 GB cache and more prep time; re-check the bench ETA with the
  cache's real fill (`python plot_metrics.py` / the manifest).
- **Stage E**: the halting head is trained only by the ponder cost, so `halt_prob → 1`
  (always halt at minimum depth) is *expected*, not a regression.
