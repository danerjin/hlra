# TRAINING.md — quickstart for the A→E run

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

```bash
grep -E 'stage=(A|B|C)' train.log | tail -3   # val_loss must be flat/lower across A->B
```
Healthy: `[step N] stage=B ... logs={'nll':6.9,'ssl':0.7,...} val_loss=6.85 lstd=0.42`.
Collapse = `val_loss` rises at B **and** `ssl`→0 **and** `lstd` craters, together.

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
```bash
cd ~/hlra/files && export LATENT_MANUAL_LAYERNORM=1
nohup python train_dialogue.py --ckpt runs/scaled/model.pt \
  --hf-chat <HF_CHAT_DATASET> --hf-name <subset> \
  --multi-turn --soft-tags --content-tags --trust-gate --persona --gestalt-readout \
  --batch-size 8 --steps <N> --out runs/dialogue > dialogue.log 2>&1 &
tail -f dialogue.log
```
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

# --- chat with it ---
python chat.py runs/dialogue/model.pt        # quick generation via the A→E path
#   (the FULL Stage-F two-lane serving — input lane, response seed, cross-turn memory —
#    is dialogue.DialogueSession; see STAGE_F.md §serving.)
```

---

**In doubt:** the run is healthy as long as **`val_loss` trends down with no jump when
Stage B turns the predictor on**, and `lstd` holds its Stage-A band. Watch `val_loss`
above all; keep checkpoints. Full Strix-Halo path + troubleshooting:
[`STRIX_HALO.md`](STRIX_HALO.md); long-form A→E walkthrough: [`archive/TRAINING.md`](archive/TRAINING.md).
