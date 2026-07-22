"""
probe_latent_use.py
===================
Guard against the Talker MEMORIZING / shortcutting. score_tokens (and the token-grounding
loss, and reconstruction) are TEACHER-FORCED: the Talker sees the chunk's real previous
tokens. So a Talker that is merely a good LM over those tokens can drive NLL down while
IGNORING the latent (decoder posterior-collapse). If that is happening, a falling tok_nll
is NOT evidence the latents are good.

Direct test: decode each chunk's tokens under (a) its REAL latent, (b) a SHUFFLED latent
(a different chunk's), (c) a ZERO latent -- the teacher-forced tokens are identical in all
three, so the only thing that changes is the latent.
  * nll(SHUFFLED) >> nll(REAL)  -> the Talker RELIES on the latent; a wrong latent hurts.
    Genuine latent use; reconstruction / tok_nll reflect latent quality.
  * nll(SHUFFLED) ~= nll(REAL)  -> the Talker decodes about as well with the WRONG latent;
    it is shortcutting through the teacher-forced tokens (memorizing / LM), NOT using the
    latent. A falling tok_nll then tells you nothing about the latents.

Run (CPU fine):
    python files/probe_latent_use.py --ckpt runs/scaled/anticollapse/model.pt --cache chunk_cache
"""
import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
PROJECT = os.path.dirname(_HERE)

from config import ModelConfig                      # noqa: E402
from data import CachedChunkDataset                 # noqa: E402
from model import LatentThoughtModel                # noqa: E402


def _resolve(p: str) -> str:
    p = os.path.expanduser(p)
    return p if os.path.isabs(p) or os.path.exists(p) else os.path.join(PROJECT, p)


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to model.pt / checkpoint.pt")
    ap.add_argument("--cache", default="chunk_cache", help="chunk cache dir (real data)")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-rows", type=int, default=512, help="cap chunks scored (memory)")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    ckpt = torch.load(_resolve(args.ckpt), map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_cfg"]) if isinstance(ckpt.get("model_cfg"), dict) else ckpt["model_cfg"]
    model = LatentThoughtModel(cfg, chunker=None).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[latent-use] {args.ckpt}: d_latent={cfg.d_latent}")

    ds = CachedChunkDataset(_resolve(args.cache))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    toks, lats = [], []
    for i, batch in enumerate(loader):
        if i >= args.batches:
            break
        ct, cm, _ri, _rm = (t.to(device) for t in batch)
        b, c, L = ct.shape
        flat = ct.reshape(b * c, L)
        vecs = model._encode_real_rows(flat, model.chunk_encoder).reshape(b * c, -1)
        valid = cm.reshape(b * c).bool()
        toks.append(flat[valid])
        lats.append(vecs[valid])
    tok = torch.cat(toks, 0)
    lat = torch.cat(lats, 0)
    n = tok.shape[0]
    if n > args.max_rows:
        idx = torch.randperm(n)[:args.max_rows]
        tok, lat = tok[idx], lat[idx]
        n = args.max_rows
    if n < 2:
        raise SystemExit("need >=2 chunks; try more --batches")
    print(f"[latent-use] scoring {n} chunks (teacher-forced token NLL)\n")

    def per_tok_nll(latent):
        s, cnt = model.score_tokens(tok, latent)
        return float(s / cnt.clamp_min(1.0))

    # shuffle so no row keeps its own latent
    perm = (torch.arange(n) + 1 + torch.randint(0, n - 1, (1,))) % n
    n_real = per_tok_nll(lat)
    n_shuf = per_tok_nll(lat[perm])
    n_zero = per_tok_nll(torch.zeros_like(lat))

    print(f"  nll(REAL latent)     = {n_real:.4f}   (reconstruction -- the chunk's own latent)")
    print(f"  nll(SHUFFLED latent) = {n_shuf:.4f}   (a different chunk's latent)")
    print(f"  nll(ZERO latent)     = {n_zero:.4f}   (no latent at all)")
    dep = n_shuf - n_real
    print(f"\n  latent dependence = nll(shuffled) - nll(real) = {dep:+.4f}")
    if dep < 0.5:
        print("  VERDICT: WEAK -- the Talker decodes about as well with the WRONG latent as the right")
        print("           one, so it is shortcutting through the teacher-forced tokens (memorizing /")
        print("           acting as an LM), NOT using the latent. A falling tok_nll is not evidence of")
        print("           good latents; and free-running generation (no teacher forcing) will be poor.")
    else:
        print("  VERDICT: the Talker RELIES on the latent -- a wrong latent hurts sharply. Genuine")
        print("           latent use, so reconstruction / tok_nll do reflect latent quality, not")
        print("           memorization of the token stream.")


if __name__ == "__main__":
    main()
