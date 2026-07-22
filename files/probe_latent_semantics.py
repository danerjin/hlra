"""
probe_latent_semantics.py
=========================
Tests the hypothesis: are the codec's latents SEMANTICALLY organized (similar meaning
-> similar latent, a smooth manifold that is predictable), or merely
RECONSTRUCTION-optimal (invertible but geometrically arbitrary, so "predict the next
latent" is predicting an arbitrary point)? If the space is arbitrary, adding a semantic
objective to the encoder (SimCSE / SBERT-distill) is justified; if it is already
semantic, prediction is limited by multimodality/decode, not by the encoder.

Two tests:
  (1) ADJACENCY (always; no external deps). Same-document consecutive chunks are
      topically related, so their latents should be MORE similar than random pairs IF
      the space captures any discourse/topical structure. adj >> rand = some structure;
      adj ~= rand = arbitrary.
  (2) SBERT CORRELATION (best-effort; needs sentence-transformers + a gpt2 tokenizer).
      Decode chunks to text, embed with a pretrained sentence encoder, and correlate
      SBERT cosine with OUR latent cosine over random pairs. High correlation = our
      latents track meaning (semantic); ~0 = they don't (arbitrary). This is the direct
      test; it is skipped with a clear message if the deps are missing.

Run (CPU fine):
    python files/probe_latent_semantics.py --ckpt runs/scaled/model.pt --cache chunk_cache
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="chunk_cache")
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sbert", default="all-MiniLM-L6-v2", help="sentence-transformers model")
    ap.add_argument("--no-sbert", action="store_true", help="skip the SBERT correlation test")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    ckpt = torch.load(_resolve(args.ckpt), map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_cfg"]) if isinstance(ckpt.get("model_cfg"), dict) else ckpt["model_cfg"]
    model = LatentThoughtModel(cfg, chunker=None).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[semantics] {args.ckpt}: d_latent={cfg.d_latent}")

    ds = CachedChunkDataset(_resolve(args.cache))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    adj_cos = []            # cosine of same-doc consecutive chunk latents
    all_lat, all_tok = [], []
    for i, batch in enumerate(loader):
        if i >= args.batches:
            break
        ct, cm, _ri, _rm = (t.to(device) for t in batch)
        B, C, L = ct.shape
        vecs = model._encode_real_rows(ct.reshape(B * C, L), model.chunk_encoder).reshape(B, C, -1)
        vn = F.normalize(vecs, dim=-1)
        for b in range(B):
            vt = [t for t in range(C) if bool(cm[b, t])]
            for j in range(len(vt) - 1):
                if vt[j + 1] == vt[j] + 1:      # truly consecutive (not across a gap)
                    adj_cos.append(float((vn[b, vt[j]] * vn[b, vt[j + 1]]).sum()))
            for t in vt:
                all_lat.append(vecs[b, t])
                all_tok.append(ct[b, t])
    if len(all_lat) < 4:
        raise SystemExit("too few chunks; try more --batches")

    lat = torch.stack(all_lat, 0)
    latn = F.normalize(lat, dim=-1)
    # random-pair baseline: pair each latent with a shuffled partner
    perm = (torch.arange(len(lat)) + 1 + torch.randint(0, len(lat) - 1, (1,))) % len(lat)
    rand_cos = (latn * latn[perm]).sum(-1)

    adj = sum(adj_cos) / max(1, len(adj_cos))
    rnd = float(rand_cos.mean())
    print(f"\n(1) ADJACENCY ({len(adj_cos)} adjacent pairs, {len(lat)} chunks):")
    print(f"      adjacent same-doc cos = {adj:.4f}")
    print(f"      random-pair      cos = {rnd:.4f}")
    print(f"      lift (adjacent - random) = {adj - rnd:+.4f}")
    if adj - rnd < 0.05:
        print("      -> latents barely distinguish adjacent from random: little topical/semantic")
        print("         structure. The space looks RECONSTRUCTION-ARBITRARY -- a semantic encoder")
        print("         objective (SimCSE / SBERT-distill) is justified.")
    else:
        print("      -> adjacent chunks are clearly more similar: the space carries topical")
        print("         structure. Some semantic organization already present.")

    if args.no_sbert:
        return
    # (2) SBERT correlation -- best effort.
    try:
        from sentence_transformers import SentenceTransformer          # noqa: E402
        from scipy.stats import spearmanr                              # noqa: E402
        try:
            from transformers import GPT2TokenizerFast
            tok = GPT2TokenizerFast.from_pretrained("gpt2")
        except Exception:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("gpt2")
    except Exception as e:
        print(f"\n(2) SBERT correlation SKIPPED ({type(e).__name__}: {e}). "
              f"Install sentence-transformers + scipy + transformers, or read (1) alone.")
        return

    print(f"\n(2) SBERT correlation (model={args.sbert}) ...")
    m = min(400, len(all_tok))
    texts = [tok.decode([int(t) for t in all_tok[k] if int(t) != 0]).strip() for k in range(m)]
    keep = [k for k in range(m) if texts[k]]
    texts = [texts[k] for k in keep]
    sub = latn[torch.tensor(keep)]
    sb = SentenceTransformer(args.sbert, device=args.device)
    emb = torch.tensor(sb.encode(texts, normalize_embeddings=True, show_progress_bar=False))
    # random pairs among the kept chunks; compare our-cos vs sbert-cos
    q = (torch.arange(len(sub)) + 1 + torch.randint(0, len(sub) - 1, (1,))) % len(sub)
    our_cos = (sub * sub[q]).sum(-1).numpy()
    sbert_cos = (emb * emb[q]).sum(-1).numpy()
    rho, _ = spearmanr(our_cos, sbert_cos)
    print(f"      Spearman(our latent cos, SBERT cos) over {len(sub)} pairs = {rho:.3f}")
    if rho < 0.2:
        print("      -> our latent geometry barely tracks meaning. ARBITRARY -> semantic objective")
        print("         is well justified.")
    elif rho < 0.5:
        print("      -> partial semantic structure; a semantic regularizer could still sharpen it.")
    else:
        print("      -> latents already track meaning well; prediction is limited elsewhere")
        print("         (multimodality / decode), not by a non-semantic encoder.")


if __name__ == "__main__":
    main()
