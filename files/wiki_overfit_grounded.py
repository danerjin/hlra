"""
wiki_overfit_grounded.py
========================
The latent side of the baseline comparison (notes §13.2/§13.3): can the
latent-thought architecture memorize a single Wikipedia page, and how fast?

This is the *decisive* grounded-only test: the reconstruction/autoencoder codec
path ONLY -- encode a chunk, decode that same chunk with the Talker (no HRM loop,
no memory, no SSL/ACT/lanes/curriculum), the post-§27 reconstruction anchor
(model.forward_grounded) -- so the measurement is the architecture's raw capacity
to fit the page through the 192-d thought bottleneck. The
optimizer/schedule/batch/step-budget are IDENTICAL to baseline_gpt.py (AdamW,
warmup 100 -> cosine to 0.1x over 1500 steps, batch 4), so the comparison plot's
"same optimizer, schedule, batch; only the architecture differs" claim holds
literally -- and the cosine tail is what stabilizes late memorization (a
held-constant LR oscillates upward past ~step 900). Reports teacher-forced page
perplexity (token-weighted, exactly as generate.score / baseline_gpt.eval_ppl) on
a curve over training steps, written to runs/wiki_overfit_grounded/metrics.json
for plot_comparison.py.

Corpus: the pre-chunked `wiki_cache` (53 paragraph docs of the "Solar System"
article, ~5.8k gpt2 tokens, smoke preset dims). Build it once with data_prep if
missing. Run:  python wiki_overfit_grounded.py
"""
from __future__ import annotations

import os
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import math

import torch
from torch.utils.data import DataLoader

from config import model_config
from model import LatentThoughtModel
from gestalt_memory import GestaltMemoryBank
from data import CachedChunkDataset, collate_chunked
from losses import grounded_nll_loss
from utils import set_seed
from baseline_gpt import lr_at   # shared warmup->cosine schedule, so "same schedule" is literal

CACHE = os.path.join(PROJECT, "wiki_cache")
OUT = os.path.join(PROJECT, "runs", "wiki_overfit_grounded")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def page_ppl(model, ds, cfg, device) -> tuple[float, float]:
    """Token-weighted teacher-forced reconstruction NLL over every doc in the
    page. Pure autoencoder codec -- encode a chunk, decode that SAME chunk with
    the Talker from an EMPTY memory, NO HRM loop -- exactly what the current
    model.forward_grounded / generate.score do (notes §27), so the number is
    directly comparable to the GPT baselines and to the run's val_loss."""
    model.eval()
    tot_nll, n_tok = 0.0, 0
    for i in range(len(ds)):
        ct, cm, _, _ = ds[i]
        ct = ct.unsqueeze(0).to(device)          # (1, C, L)
        cm = cm.unsqueeze(0).to(device)
        for t in range(ct.shape[1]):
            if not bool(cm[0, t]):
                continue
            chunk_ids = ct[:, t, :]
            latent = model.chunk_encoder(chunk_ids, chunk_ids != 0)
            empty_mem = GestaltMemoryBank(cfg.memory_capacity, cfg.d_model)
            logits = model.talker(chunk_ids, latent, empty_mem)
            k = int((chunk_ids != 0).sum())
            if k == 0:
                continue
            tot_nll += grounded_nll_loss(logits, chunk_ids, chunk_ids != 0).item() * k
            n_tok += k
    model.train()
    avg = tot_nll / max(n_tok, 1)
    return avg, math.exp(min(avg, 20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--full-bptt", action="store_true",
                    help="DIAGNOSTIC (now MOOT post-§27): patched the inner-loop truncation, but "
                         "reconstruction no longer runs the HRM loop -- so this flag is inert here. "
                         "Kept only so old invocations don't error. Use train_scaled for loop paths.")
    ap.add_argument("--talker-copy", action="store_true",
                    help="DIAGNOSTIC: reproduce the pre-C1 (§1.2) Talker copy bug (no right shift), "
                         "which drives NLL->0 / ppl->1.0 by copying the target instead of using the "
                         "thought. Tests whether the original §13.2 '->1.0' was this artifact.")
    args = ap.parse_args()

    if args.full_bptt:
        import hrm_loop
        hrm_loop._TruncationSchedule.maybe_detach = lambda self, h, l: (h, l)
        print("[wiki_overfit] DIAGNOSTIC full-BPTT: inner-loop truncation DISABLED "
              "(pre-C5/§11.1 behavior -- full-document raw-chain BPTT)")

    if args.talker_copy:
        # DIAGNOSTIC: reproduce the pre-C1 (§1.2) Talker teacher-forcing bug -- feed
        # the target tokens UNSHIFTED, so decoder position i sees token i and can
        # copy it (NLL -> 0, ppl -> 1.0) WITHOUT using the thought. Tests whether
        # the original §13.2 '->1.0' was this copy artifact rather than real
        # reconstruction. Everything else (mask, memory readout) is unchanged.
        import torch as _t
        from talker import Talker as _Talker

        def _copy_forward(self, target_tokens, thought, memory):
            batch, chunk_len = target_tokens.shape
            device = target_tokens.device
            positions = _t.arange(chunk_len, device=device).unsqueeze(0)
            x = self.token_embed(target_tokens) + self.pos_embed(positions)  # NO right shift
            memory_readout = self.memory_reader(thought, memory)
            thought_kv = _t.stack([thought, memory_readout], dim=1)
            causal_mask = _t.triu(_t.ones(chunk_len, chunk_len, dtype=_t.bool, device=device), diagonal=1)
            for layer in self.layers:
                x = layer(x, thought_kv, causal_mask)
            return self.lm_head(self.out_norm(x))

        _Talker.forward = _copy_forward
        print("[wiki_overfit] DIAGNOSTIC talker-copy: Talker right-shift DISABLED "
              "(pre-C1/§1.2 behavior -- decoder can copy the target, NLL->0)")

    device = pick_device()
    set_seed(0)
    ds = CachedChunkDataset(CACHE)
    cfg = model_config("smoke", vocab_size=ds.vocab_size)
    model = LatentThoughtModel(cfg, chunker=None).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_chunked, drop_last=True)
    it = iter(loader)

    STEPS, LOG, WARMUP, BASE_LR = args.steps, max(1, args.steps // 15), 100, args.lr
    print(f"[wiki_overfit] device={device} params={n_params/1e6:.1f}M docs={len(ds)} "
          f"tokens={ds.manifest['tokens']}  grounded-only, warmup{WARMUP}->cosine LR (== baseline)")
    metrics = []
    nll0, ppl0 = page_ppl(model, ds, cfg, device)
    print(f"[step 0] page_nll={nll0:.4f} page_ppl={ppl0:.1f}  (chance ~{math.exp(10.825):.0f})")
    metrics.append({"step": 0, "page_nll": round(nll0, 4), "page_ppl": round(ppl0, 1)})

    for step in range(1, STEPS + 1):
        lr = lr_at(step - 1, BASE_LR, WARMUP, STEPS)   # same warmup->cosine as baseline_gpt
        for g in opt.param_groups:
            g["lr"] = lr
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        ct, cm, ri, rm = (t.to(device) for t in batch)
        # Current reconstruction is a pure autoencoder codec (encoder -> Talker,
        # no loop, no memory), notes §27: forward_grounded now takes just
        # (chunk_tensor, chunk_mask) and returns the reconstruction NLL.
        nll = model.forward_grounded(ct, cm)
        opt.zero_grad(set_to_none=True)
        nll.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % LOG == 0:
            pnll, pppl = page_ppl(model, ds, cfg, device)
            print(f"[step {step}] train_nll={nll.item():.4f} page_nll={pnll:.4f} page_ppl={pppl:.1f}",
                  flush=True)
            metrics.append({"step": step, "train_nll": round(nll.item(), 4),
                            "page_nll": round(pnll, 4), "page_ppl": round(pppl, 1)})

    os.makedirs(args.out, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "model_cfg": _cfg_dict(cfg),
                "params": n_params, "final_ppl": metrics[-1]["page_ppl"]},
               os.path.join(args.out, "model.pt"))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n=== latent grounded-only (batch {args.batch_size}): final page ppl "
          f"{metrics[-1]['page_ppl']} (from {metrics[0]['page_ppl']}) ===\n"
          f"[wiki_overfit] saved -> {args.out}")


def _cfg_dict(cfg):
    from dataclasses import asdict
    return asdict(cfg)


if __name__ == "__main__":
    main()
