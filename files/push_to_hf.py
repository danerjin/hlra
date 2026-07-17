"""
push_to_hf.py
=============
Upload a trained checkpoint (an A→E `model.pt` or a Stage-F dialogue `model.pt`)
to a HuggingFace **model** repo. This is the clean way to get a multi-GB
checkpoint off the training box: HF handles large files, so there's no git 100 MB
limit to fight.

It can **strip** the checkpoint to inference-only weights first (~4× smaller — it
drops the AdamW optimizer state + EMA + RNG that only *resume* needs, keeping
`model_state` + `model_cfg` + the Stage-F adapter), and optionally **cast** the
weights to bf16 (halves it again). A stripped checkpoint still loads with
`generate.load` / `chat.py` / `train_dialogue.py --ckpt`.

Usage (once): `hf auth login`  (paste a WRITE token), then:
    # full A→E checkpoint, stripped, private:
    python push_to_hf.py --ckpt runs/scaled/model.pt   --repo <you>/hlra-smallw3 --strip
    # Stage-F chatbot, stripped + bf16:
    python push_to_hf.py --ckpt runs/dialogue/model.pt --repo <you>/hlra-chat    --strip --bf16

Load it back anywhere:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("<you>/hlra-smallw3", "model.pt")
    # then, inside this repo:  python generate.py --ckpt "$p" --score "some text"

Note: uploads go through HF's storage endpoint. If the push fails on the training
box with a network/Xet error, `rsync` the checkpoint to a machine with normal
internet and run this there instead — it only needs the file + `huggingface_hub`.
"""
from __future__ import annotations

import argparse
import os
import tempfile

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import torch

# What inference / a Stage-F fine-tune actually reads (generate.load reads
# model_cfg + model_state; train_dialogue --ckpt also reads adapter_state). The
# optimizer/ema/rng/metrics are RESUME-only and are the bulk of the file size.
_KEEP = ("model_state", "model_cfg", "vocab_size", "stage_reached",
         "tokenizer_name", "chunker", "adapter_state")


def strip_checkpoint(ckpt: dict, bf16: bool = False) -> dict:
    slim = {k: ckpt[k] for k in _KEEP if k in ckpt}
    if bf16:
        for key in ("model_state", "adapter_state"):
            if key in slim:
                slim[key] = {k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v)
                             for k, v in slim[key].items()}
    return slim


def model_card(repo: str, meta: dict) -> str:
    return f"""---
license: other
library_name: pytorch
tags:
- latent-thought
- hrm
- jepa
- thought-gestalt
---

# {repo}

A **latent-thought reasoning** checkpoint from
[danerjin/hlra](https://github.com/danerjin/hlra) — a model that thinks in
chunk-level latent "thoughts" (HRM-Text × JEPA-Reasoner × Thought Gestalt × Parcae).

- stage reached: `{meta.get('stage_reached', '?')}`  ·  vocab: `{meta.get('vocab_size', '?')}`
- research checkpoint — **not** GPT-2 quality at this scale (by design).

## Load
```python
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download("{repo}", "model.pt")
# in the hlra repo:
#   python generate.py --ckpt "$ckpt" --score "a sentence to score"
#   python chat.py "$ckpt"                     # if this is a Stage-F chatbot checkpoint
```
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to model.pt")
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. danerjin/hlra-smallw3")
    ap.add_argument("--public", action="store_true", help="make the repo public (default: PRIVATE)")
    ap.add_argument("--strip", action="store_true",
                    help="upload inference-only weights (~4x smaller: drop optimizer/ema/rng)")
    ap.add_argument("--bf16", action="store_true", help="cast weights to bf16 (halve size again)")
    ap.add_argument("--filename", default="model.pt", help="path in the repo")
    ap.add_argument("--token", default=None, help="HF write token (else uses `hf auth login`)")
    ap.add_argument("--no-card", action="store_true", help="skip generating README.md")
    args = ap.parse_args()

    from huggingface_hub import HfApi, create_repo

    # Relative paths resolve against the PROJECT, not the cwd: TRAINING.md 6.3 sits
    # under a `cd ~/hlra/files` block and documents `--ckpt runs/scaled/model.pt`,
    # which train_scaled writes to PROJECT/runs/scaled. Matches generate.py:223 and
    # chat_core.py:29. (This one at least failed loudly.)
    ckpt_path = os.path.expanduser(args.ckpt)
    if not os.path.isabs(ckpt_path) and not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(PROJECT, ckpt_path)
    ckpt_path = os.path.abspath(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"no checkpoint at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = {k: ckpt.get(k) for k in ("stage_reached", "vocab_size", "tokenizer_name")}

    upload_path, tmp = ckpt_path, None
    if args.strip or args.bf16:
        slim = strip_checkpoint(ckpt, bf16=args.bf16)
        tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        torch.save(slim, tmp.name)
        tmp.close()
        upload_path = tmp.name
        print(f"[push_to_hf] stripped {os.path.getsize(ckpt_path)/1e9:.2f} GB "
              f"-> {os.path.getsize(upload_path)/1e9:.2f} GB")
    del ckpt  # free RAM before the upload streams the file from disk

    api = HfApi(token=args.token)
    create_repo(args.repo, repo_type="model", private=not args.public,
                exist_ok=True, token=args.token)
    print(f"[push_to_hf] uploading {args.filename} to {args.repo} "
          f"({'public' if args.public else 'private'})...")
    api.upload_file(path_or_fileobj=upload_path, path_in_repo=args.filename,
                    repo_id=args.repo, repo_type="model")
    if not args.no_card:
        api.upload_file(path_or_fileobj=model_card(args.repo, meta).encode(),
                        path_in_repo="README.md", repo_id=args.repo, repo_type="model")
    if tmp is not None:
        os.unlink(tmp.name)
    print(f"[push_to_hf] done -> https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
