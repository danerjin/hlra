# TRAINING.md — quickstart for the A→E run

> **Just want it to run?** [`training_easy.md`](training_easy.md) is the two-paste
> version — one command to verify the box, one command for the whole prep→train
> pipeline. Come back here for the *why*, the flags, and monitoring.

The **no-brainer path** to a trained model. Copy-paste, top to bottom.

- **On a Strix Halo / gfx1151 box** (the reference setup): follow the complete
  end-to-end run book in **[`STRIX_HALO.md`](STRIX_HALO.md)** — install → GPU →
  data → queued training, with every gotcha inlined. Do that instead of this file.
- This quickstart is the **generic (any CUDA/ROCm GPU)** version. The full current
  troubleshooting matrix is in [`STRIX_HALO.md`](STRIX_HALO.md) §8.
- The long-form step-by-step A→E walkthrough (older, but the deepest *why* behind
  each step) is preserved in [`archive/TRAINING.md`](archive/TRAINING.md).
- **Stage F** (chatbot fine-tuning) is a separate optional phase → §6 + [`STAGE_F.md`](STAGE_F.md).

---

## 0. The one number that decides success

`val_loss` = the autoencoder reconstruction (encode a chunk → the Talker decodes it;
no reasoning loop). It's the anti-collapse anchor. **It must NOT jump up when the
predictor turns on at Stage B** — a rise there is latent collapse. `lstd` (latent std)
is a secondary, width-dependent monitor; judge it against its own Stage-A band, never
an absolute threshold. Untrained `val_loss` ≈ 10.8; a healthy run drives it well
below and keeps falling through the A→B boundary.

---

## 1. Setup

```bash
export PROJECT=~/hlra                      # repo root; edit to yours
cd "$PROJECT" && source .venv-rocm/bin/activate   # or your torch env
pip install -r training.txt                # datasets transformers wtpsplit tqdm matplotlib pyarrow
cd "$PROJECT/files"
# GPU sanity (must PASS; ROCm/gfx1151: prepend LATENT_MANUAL_LAYERNORM=1 -- see STRIX_HALO.md §2):
python rocm_smoke.py --preset small-w3
```

## 2. Data → chunk cache (one-time)

```bash
# streaming (works where the dataset's CDN is reachable):
python data_prep.py --dataset HuggingFaceFW/fineweb-edu --name sample-10BT --streaming \
  --preset small-w3 --docs 1000 --out chunk_cache_dryrun && rm -rf "$PROJECT/chunk_cache_dryrun"
python data_prep.py --dataset HuggingFaceFW/fineweb-edu --name sample-10BT --streaming \
  --preset small-w3 --max-tokens 1200000000 --out chunk_cache
```
- **SaT segmentation runs on the GPU** automatically when one is free (`LATENT_SAT_DEVICE`
  to override) — on CPU it takes weeks. Add **`--regex`** for a ~1000× faster
  approximate chunker (fine for a first run).
- **If the HF dataset won't download** (Xet-CDN 403, streaming hangs): download the
  parquet elsewhere, `rsync` it in, and prep with **`--local-glob "DIR/**/*.parquet"`**
  (reads local files offline). Full recipe in [`STRIX_HALO.md`](STRIX_HALO.md) §5.
- `--preset` must match training. `small-w3` needs `--var-weight 3.0` at launch.
- Prep is **not resumable** → run under `nohup`/`tmux`; it ends `wrote <N> examples`.
- **Killed/stalled mid-prep?** The manifest is only written at the very end, but every
  finished `shard_*.pt` is on disk. Reconstruct a valid manifest and train on the
  **partial** cache (0.5–1 B tokens is a fine first run — the budget has slack):
  ```bash
  pkill -f data_prep                                   # stop the stalled prep first
  python make_manifest.py "$PROJECT/chunk_cache" small-w3   # scans shards -> writes manifest.json
  ```
  It skips a half-written trailing shard automatically and stamps the current
  `chunker_version` so the loader accepts the salvaged cache.

## 3. Stage budget + launch

```bash
export BATCH=32                            # size to your GPU memory
export STAGE_STEPS=$(python3 -c "import json,os;m=json.load(open(os.path.expanduser('$PROJECT/chunk_cache/manifest.json')));u=max(1,(m['total']//$BATCH)//6);print(f'{u},{u},{u},{u},{2*u},0')")
echo "$STAGE_STEPS"

# sanity (tiny; confirms stages advance, no nan):
python train_scaled.py --preset small-w3 --cache chunk_cache --device cuda --amp --amp-dtype bf16 \
  --batch-size 16 --stage-steps 5,5,5,5,5,0 --var-weight 3.0 --log-every 1 --out runs/scaled_sanity
rm -rf "$PROJECT/runs/scaled_sanity"

# real run (background, survives disconnect):
nohup python train_scaled.py --preset small-w3 --cache chunk_cache --device cuda --amp --amp-dtype bf16 \
  --batch-size "$BATCH" --stage-steps "$STAGE_STEPS" --var-weight 3.0 --lr-schedule per-stage \
  --num-workers 8 --log-every 50 --checkpoint-every 1000 --archive-every 5000 \
  --out runs/scaled > train.log 2>&1 &
tail -f train.log
```
On ROCm/gfx1151 prepend `LATENT_MANUAL_LAYERNORM=1` and use the auto-start queue in
[`STRIX_HALO.md`](STRIX_HALO.md) §6.

## 4. Monitor

Run all of these from `$PROJECT/files` (where `prep.log` / `train.log` are written).

### 4.1 Is it alive, and did the workaround engage?
```bash
pgrep -af 'data_prep|train_scaled' || echo "NOT RUNNING"
grep -m1 "manual LayerNorm active" train.log   # ROCm/gfx1151: MUST print once, else the run is corrupt -> kill & relaunch with LATENT_MANUAL_LAYERNORM=1
ls -l --time-style=+%H:%M runs/scaled/checkpoint.pt   # mtime should keep advancing (proof it's still writing)
```

### 4.2 How much time is left?
**During prep** — reads the live token count from the log (never hardcode it), assumes the 1.2 B target:
```bash
T=$(grep -oE '~[0-9]+ tokens' prep.log | tail -1 | grep -oE '[0-9]+')
S=$(ps -o etimes= -p "$(pgrep -f data_prep | head -1)" | tr -d ' ')
python3 -c "t=$T;s=$S;b=1.2e9;r=t/s;print(f'{t/1e6:.0f}M tok · {r/1000:.1f}k tok/s · elapsed {s/3600:.1f}h · ETA ~{(b-t)/r/3600:.1f}h to {b/1e9:.1f}B')"
```
**During training** — total steps ≈ one epoch (`examples // BATCH`), so % and ETA fall out of the last `[step N]`:
```bash
export BATCH=32   # the value you launched with
TOT=$(python3 -c "import json,os;m=json.load(open(os.path.expanduser('$PROJECT/chunk_cache/manifest.json')));print(m['total']//$BATCH)")
N=$(grep -oE '\[step [0-9]+\]' train.log | tail -1 | grep -oE '[0-9]+')
S=$(ps -o etimes= -p "$(pgrep -f train_scaled | head -1)" | tr -d ' ')
python3 -c "n=$N;tot=$TOT;s=$S;r=n/max(s,1);print(f'step {n}/{tot} ({100*n/tot:.0f}%) · {r:.2f} step/s · elapsed {s/3600:.1f}h · ETA ~{(tot-n)/max(r,1e-9)/3600:.1f}h')"
```
ETA is a running average — it's pessimistic early (model-load + warmup are in the elapsed time) and settles after a few hundred steps. Stage E (the `2*u` block) is the longest.

### 4.3 Which stage / what are the losses doing?
```bash
grep -E 'stage=(A|B|C|D|E)' train.log | tail -3   # val_loss must be flat/lower across A->B
```
Healthy: `[step N] stage=B ... logs={'nll':6.9,'ssl':0.7,...} val_loss=6.85 lstd=0.42`.
Collapse = `val_loss` rises at B **and** `ssl`→0 **and** `lstd` craters, together (§0).

### 4.4 Is the GPU actually working?
```bash
rocm-smi --showuse --showmeminfo vram      # ROCm/gfx1151: GPU% high, VRAM/GTT stable (not climbing to OOM)
# nvidia-smi                               # CUDA equivalent
watch -n5 'tail -n2 train.log'             # live step counter without holding a tail -f open
```
On Strix Halo the "VRAM" is the shared GTT — watch it stay flat, not creep (a slow climb is the allocator-pool issue the prep fix addresses; training doesn't hit it, but it's the thing to eyeball).

## 5. Stop / resume / finish

```bash
pgrep -af train_scaled.py ; kill <PID>        # graceful checkpoint, loses <=1 step
# resume: re-run the SAME launch command (auto-resumes from runs/scaled/checkpoint.pt)
python plot_metrics.py runs/scaled            # when done -> loss_curves.png
```
Resume hard-fails if the cache changed size, and warns loudly if flags differ. Never
re-prep/touch the cache dir mid-run.

## 6. Stage F — chatbot fine-tuning (optional, UNVALIDATED)

Fine-tune the finished A→E **`small-w3`** checkpoint into a chatbot with a separate
driver (`train_dialogue.py`, **not** `train_scaled.py`). Every feature is opt-in and
byte-identical to A→E when off; it is **smoke-only** (never trained on real dialogue;
the 2026-07-14 review found the anti-sycophancy loss doesn't yet reliably train the
trust gate). Design, flags, and caveats: **[`STAGE_F.md`](STAGE_F.md)**.

**Precondition:** the A→E run finished → `runs/scaled/model.pt` exists on the box. It
carries the `small-w3` config, so Stage F inherits it — **don't pass `--preset` with
`--ckpt`** (the checkpoint's config wins). On ROCm/gfx1151 keep
`LATENT_MANUAL_LAYERNORM=1` exported (Stage F trains → the LayerNorm workaround applies).

### 6.1 Offline smoke (plumbing check — no ckpt, no downloads, ~1 min)
```bash
cd ~/hlra/files && export LATENT_MANUAL_LAYERNORM=1
python train_dialogue.py --offline --preset small-w3 --multi-turn --persona \
  --steps 20 --batch-size 2 --out runs/dlg_sanity && rm -rf ~/hlra/runs/dlg_sanity
```
Confirms the path runs and the losses aren't `nan` (a **fresh** `small-w3` model — the
real fine-tune below loads the trained one via `--ckpt`).

### 6.2 Real fine-tune off the small-w3 checkpoint (background)
The foundation (`runs/scaled/model.pt`) is loaded **read-only**; Stage-F writes to a
**separate** `--out runs/dialogue` — it never overwrites the A→E checkpoint.
```bash
cd ~/hlra/files && export LATENT_MANUAL_LAYERNORM=1
nohup python train_dialogue.py --ckpt runs/scaled/model.pt \
  --hf-chat HuggingFaceH4/no_robots --split train \
  --multi-turn --soft-tags --content-tags --trust-gate --vector-gate --persona --gestalt-readout \
  --batch-size 8 --steps 3000 --out runs/dialogue > dialogue.log 2>&1 &
tail -f dialogue.log
```
- **Dataset:** `HuggingFaceH4/no_robots` (10k clean instruct dialogues, `messages`
  schema, downloads without Xet issues — verified). Scale up with
  `--hf-chat HuggingFaceH4/ultrachat_200k --split train_sft`.
- **Any `messages`-schema chat dataset works** (role `assistant`→SELF, `user`→USER,
  `system`→SYSTEM). **Don't pass `--preset` with `--ckpt`** — the checkpoint's config wins.
- **Transcript data** (you choose who is SELF — the reasoner vs. an advocate):
  swap in `--hf-transcript <ID> --text-field text --target-speaker "SOCRATES" --system-speakers "NARRATOR"`.
- Add `--rag` for latent RAG. Full flag table: [`STAGE_F.md`](STAGE_F.md) §4–6.
- **Watch:** `nll` (anchor, should hold) · `cos`/`gen` (should fall — `gen` = response
  quality) · `syco` · `trust=USER:../SELF:..`. Output: `runs/dialogue/model.pt`.

### 6.3 Get the checkpoint back + share it
Works for **either** checkpoint (`runs/scaled/model.pt` A→E, or `runs/dialogue/model.pt`
Stage-F).

```bash
# --- push to HuggingFace (handles multi-GB; no git 100 MB limit) ---
hf auth login                                            # once; paste a WRITE token
python push_to_hf.py --ckpt runs/dialogue/model.pt --repo <you>/hlra-chat --strip --bf16
#   --strip = inference-only weights (~4x smaller); repo is PRIVATE by default (--public to share).
#   If the push fails on the box (network), rsync the file to your laptop and push from there.

# --- OR rsync the checkpoint back to your laptop (run this ON your laptop) ---
rsync -avP daniel@<box>:~/hlra/runs/dialogue/model.pt ~/hlra/runs/dialogue/model.pt

# --- chat with it (proper Stage-F two-lane serving: input lane + response seed + cross-turn memory) ---
python dialogue_chat.py runs/dialogue/model.pt     # CLI REPL (:source for RAG · :reset · :temp · :n)
python web_chat.py runs/dialogue/model.pt          # web UI -> pick the "Chat" mode toggle
#   (chat.py also loads it, but only runs the plain A→E generation path — no dialogue memory.)
```

---

## 7. Command reference — the exact invocations that worked (reference gfx1151 box)

Every command below is a **known-good** one from a real A→E bring-up (paths are the
reference box `daniel@…:~/hlra`; edit to yours). Copy-paste in order. Box-specific
setup (install, LayerNorm workaround, Xet-escape) lives in
[`STRIX_HALO.md`](STRIX_HALO.md); this is the operational cheat-sheet.

### 7.1 Activate the env and prove torch sees the GPU
```bash
source ~/hlra/.venv-rocm/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # -> ...rocm... True
cd ~/hlra/files && LATENT_MANUAL_LAYERNORM=1 python rocm_smoke.py --preset small-w3   # must end PASS
```

### 7.2 Prep the chunk cache (the command that ran)
```bash
cd ~/hlra/files
# from LOCAL parquet (Xet-CDN escape hatch; downloaded on the Mac + rsync'd to the box):
nohup python data_prep.py --local-glob "/home/daniel/hlra/fineweb_local/**/*.parquet" \
  --preset small-w3 --max-tokens 1200000000 --out chunk_cache > prep.log 2>&1 &
# want it in minutes not hours? add --regex (fast approx chunker, fine for a first run).
```
Prep is **not resumable** and writes `manifest.json` only at the very end. Salvage a
killed/stalled prep (0.5–1 B tokens is a fine first run):
```bash
pkill -f data_prep
python make_manifest.py ~/hlra/chunk_cache small-w3     # rebuilds manifest.json from the shards on disk
```

### 7.3 Which interpreter/venv is a running process using?
`readlink /proc/PID/exe` alone is **misleading** — a venv's `python` symlinks to the
base interpreter, so it shows e.g. `/usr/bin/python3.10` even inside a venv. Read
`VIRTUAL_ENV` to be sure:
```bash
PID=$(pgrep -f data_prep.py | head -1)
echo "cmdline: $(tr '\0' ' ' < /proc/$PID/cmdline)"
tr '\0' '\n' < /proc/$PID/environ | grep -E '^(VIRTUAL_ENV|PATH)='   # VIRTUAL_ENV= line -> that's the venv
```

### 7.4 Queue training to auto-start when prep finishes (venv-aware)
The full paste-able block is [`STRIX_HALO.md`](STRIX_HALO.md) §6 — it bakes in **the exact
interpreter prep runs** (so it works for a venv *or* a system-wide torch), exports
`LATENT_MANUAL_LAYERNORM=1`, refuses to train on a half-cache, and logs to
`pipeline.log`. Verify it armed correctly:
```bash
head -2 ~/hlra/files/pipeline.log
#   -> STATUS: python=/home/daniel/hlra/.venv-rocm/bin/python torch=...rocm... True
#   -> STATUS: waiting for prep (PID ....)
```
Manage the watcher (prep is a separate process and keeps running):
```bash
pgrep -af 'run_pipeline|data_prep|train_scaled'     # what's running
pkill -f run_pipeline.sh                            # stop ONLY the watcher, then re-paste the §6 block to re-queue
```

### 7.5 Monitor (queued run logs to `pipeline.log`, manual run to `train.log`)
> The auto-start runs training *inside* the nohup'd watcher, so training output goes to
> **`pipeline.log`**, not `train.log`. Point these at whichever applies. Run from `~/hlra/files`.
```bash
LOG=pipeline.log                                    # or train.log for a manual launch

# alive + workaround engaged + still checkpointing:
pgrep -af 'data_prep|train_scaled' || echo "NOT RUNNING"
grep -m1 "manual LayerNorm active" "$LOG"           # MUST print once (else corrupt -> relaunch)

# prep ETA (live token count, never hardcoded):
T=$(grep -oE '~[0-9]+ tokens' prep.log | tail -1 | grep -oE '[0-9]+'); S=$(ps -o etimes= -p "$(pgrep -f data_prep|head -1)"|tr -d ' ')
python3 -c "t=$T;s=$S;b=1.2e9;r=t/s;print(f'{t/1e6:.0f}M tok · {r/1000:.1f}k tok/s · ETA ~{(b-t)/r/3600:.1f}h to {b/1e9:.1f}B')"

# training progress + ETA (total steps ~= examples//BATCH):
TOT=$(python3 -c "import json,os;m=json.load(open(os.path.expanduser('~/hlra/chunk_cache/manifest.json')));print(m['total']//32)")
N=$(grep -oE '\[step [0-9]+\]' "$LOG" | tail -1 | grep -oE '[0-9]+'); S=$(ps -o etimes= -p "$(pgrep -f train_scaled|head -1)"|tr -d ' ')
python3 -c "n=$N;tot=$TOT;s=$S;r=n/max(s,1);print(f'step {n}/{tot} ({100*n/tot:.0f}%) · {r:.2f} step/s · ETA ~{(tot-n)/max(r,1e-9)/3600:.1f}h')"

# health (THE check) + GPU:
grep -E 'stage=(A|B|C|D|E)' "$LOG" | tail -5       # val_loss must be flat/lower across A->B
rocm-smi --showuse --showmeminfo vram
```

### 7.6 Detach / retrieve
```bash
# disconnect SSH without killing anything (already nohup'd -> just close the session, or):
exit                                                # nohup jobs survive logout

# when done, get runs/scaled/model.pt off the box -> §6.3 (push_to_hf.py or rsync).
```

---

**In doubt:** the run is healthy as long as **`val_loss` trends down with no jump when
Stage B turns the predictor on**, and `lstd` holds its Stage-A band. Watch `val_loss`
above all; keep checkpoints. Full Strix-Halo path + troubleshooting:
[`STRIX_HALO.md`](STRIX_HALO.md); long-form A→E walkthrough: [`archive/TRAINING.md`](archive/TRAINING.md).
