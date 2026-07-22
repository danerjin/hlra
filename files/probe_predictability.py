"""
probe_predictability.py
=======================
Gating test for the "make the encoder semantic" hypothesis, BEFORE building/distilling
anything. Question: are semantically-organized latents (a real sentence embedder, SBERT)
more SEQUENTIALLY PREDICTABLE than our reconstruction codec's latents? If a genuine
semantic space is ALSO flat sequentially, then the next chunk is intrinsically
unpredictable (multimodal / discourse), semantics won't help, and distillation is a
waste. If SBERT is much more predictable, the semantic-encoder direction is justified.

Measure: the "persistence" predictability proxy -- how much closer is the NEXT chunk than
a random chunk (adjacency lift = cos(t, t+1) - cos(t, random)). Computed in BOTH spaces
on the SAME chunks:
  * our codec latents (encode)
  * SBERT embeddings (decode chunk -> text -> sentence-transformer)
SBERT lift >> codec lift  -> semantic latents are more predictable; build the semantic
encoder (distill test next). SBERT lift ~= codec lift (both small) -> even a perfect
semantic space isn't sequentially predictable here; do NOT build it.

Needs sentence-transformers + transformers (gpt2 tokenizer).
Run:  python files/probe_predictability.py --ckpt runs/scaled/model.pt --cache chunk_cache
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


def _lift(emb, adj_idx):
    """adjacency lift = mean cos(adjacent) - mean cos(random), on normalized emb.
    adj_idx: list of (i, j) index pairs that are consecutive same-doc chunks."""
    z = F.normalize(emb, dim=-1)
    if not adj_idx:
        return None
    ai = torch.tensor([i for i, _ in adj_idx]); aj = torch.tensor([j for _, j in adj_idx])
    adj = (z[ai] * z[aj]).sum(-1).mean()
    perm = (torch.arange(len(z)) + 1 + torch.randint(0, len(z) - 1, (1,))) % len(z)
    rnd = (z * z[perm]).sum(-1).mean()
    return float(adj), float(rnd), float(adj - rnd)


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="chunk_cache")
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sbert", default="all-MiniLM-L6-v2")
    args = ap.parse_args(argv)

    try:
        from sentence_transformers import SentenceTransformer
        try:
            from transformers import GPT2TokenizerFast
            tok = GPT2TokenizerFast.from_pretrained("gpt2")
        except Exception:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("gpt2")
    except Exception as e:
        raise SystemExit(f"needs sentence-transformers + transformers ({type(e).__name__}: {e}). "
                         f"pip install sentence-transformers")

    device = torch.device(args.device)
    ckpt = torch.load(_resolve(args.ckpt), map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ckpt["model_cfg"]) if isinstance(ckpt.get("model_cfg"), dict) else ckpt["model_cfg"]
    model = LatentThoughtModel(cfg, chunker=None).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()

    ds = CachedChunkDataset(_resolve(args.cache))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    lat, texts, adj = [], [], []            # codec latents, chunk texts, adjacency pairs (indices into lat)
    for i, batch in enumerate(loader):
        if i >= args.batches:
            break
        ct, cm, _ri, _rm = (t.to(device) for t in batch)
        B, C, L = ct.shape
        vecs = model._encode_real_rows(ct.reshape(B * C, L), model.chunk_encoder).reshape(B, C, -1)
        for b in range(B):
            vt = [t for t in range(C) if bool(cm[b, t])]
            base = len(lat)
            for k, t in enumerate(vt):
                lat.append(vecs[b, t])
                texts.append(tok.decode([int(x) for x in ct[b, t] if int(x) != 0]).strip())
                if k > 0 and vt[k] == vt[k - 1] + 1:
                    adj.append((base + k - 1, base + k))
    lat = torch.stack(lat, 0)
    print(f"[predict] {len(lat)} chunks, {len(adj)} adjacent pairs")

    sb = SentenceTransformer(args.sbert, device=args.device)
    sbe = torch.tensor(sb.encode(texts, normalize_embeddings=False, show_progress_bar=False))

    c_adj, c_rnd, c_lift = _lift(lat, adj)
    s_adj, s_rnd, s_lift = _lift(sbe, adj)
    print(f"\n  {'space':<14}{'adjacent cos':>14}{'random cos':>12}{'lift':>10}")
    print(f"  {'our codec':<14}{c_adj:>14.4f}{c_rnd:>12.4f}{c_lift:>+10.4f}")
    print(f"  {'SBERT':<14}{s_adj:>14.4f}{s_rnd:>12.4f}{s_lift:>+10.4f}")
    print(f"\n  predictability ratio (SBERT lift / codec lift) = {s_lift / c_lift:.2f}x" if c_lift > 1e-6 else "")
    if s_lift < 0.05 or s_lift < 1.5 * c_lift:
        print("  VERDICT: even a real semantic space is NOT much more sequentially predictable here.")
        print("           The next chunk is intrinsically hard (multimodal/discourse). A semantic")
        print("           encoder (distill / pretrain) is unlikely to help -- do NOT build it.")
    else:
        print("  VERDICT: SBERT latents are markedly MORE predictable than our codec's. Semantic")
        print("           organization would make next-latent prediction easier -- the semantic")
        print("           encoder direction is justified. Distill test is the next step.")


if __name__ == "__main__":
    main()
