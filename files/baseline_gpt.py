"""
baseline_gpt.py
===============
A plain, standard decoder-only Transformer LM (GPT-style), as a *fair-scale
baseline* for the latent-thought architecture. Same gpt2 tokenizer, same text,
same optimizer/schedule/step budget as the latent run, so the only thing that
differs is the architecture.

Two size presets, because "same scale" is ambiguous for the latent model (90%
of its 43M params are four separate vocab-scale embedding tables; only ~4.5M is
transformer compute):

  * ``same-params``  (~43M): matches the latent model's *total parameter count*
    -- a standard GPT of the same on-disk size (naturally deeper/wider, since it
    doesn't duplicate embedding tables).
  * ``same-compute`` (~14M): matches the latent model's *width* (d_model=192)
    and its non-embedding transformer compute (~4.5M) -- fewer total params,
    but the same "reasoning" capacity and width.

This is a memorization comparison on one Wikipedia page (there is no held-out
set at this size): how well can each architecture, at matched scale, fit the
page? Report is teacher-forced per-token perplexity on the same text, plus a
greedy generation sample.

NOTE on task asymmetry (stated so the comparison is read honestly): a standard
LM does *pure causal next-token* prediction (no lookahead). The latent model's
reconstruction NLL is an *autoencoder* through a latent bottleneck -- it gets to
condition on an encoding of the very chunk it decodes (lookahead), but must pass
everything through a 192-d thought vector + FIFO memory. Different tasks; both
are "teacher-forced per-token NLL on the page" under each model's own objective.

Run:
    python baseline_gpt.py --preset same-params   --out runs/baseline_same_params
    python baseline_gpt.py --preset same-compute  --out runs/baseline_same_compute
"""
from __future__ import annotations

import os
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(PROJECT, ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
TOKENIZER_DIR = os.path.join(PROJECT, "gpt2_tok")

import math
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import set_seed
from config import ArchConfig
from modern import ModernAttention, RoPE, SwiGLU, make_norm, build_ffn

# Standard GPT shapes. d_ff = 4*d_model (the usual ratio).
PRESETS = {
    # ~43M total, matched to the latent model's TOTAL params (standard shape).
    "same-params":  dict(d_model=512, n_layers=6,  n_heads=8, d_ff=2048),
    # ~14M total, matched to the latent model's WIDTH (192) + compute (~4.5M).
    "same-compute": dict(d_model=192, n_layers=10, n_heads=6, d_ff=768),
}


class Block(nn.Module):
    """Standard pre-LN transformer decoder block.

    Legacy (arch.is_legacy, the default): the exact stock block -- nn.LayerNorm +
    nn.MultiheadAttention + GELU FFN, driven by an external boolean causal mask.
    Modern (any arch flag on): RMSNorm / QK-normed GQA over SDPA (is_causal) /
    SwiGLU, with RoPE passed in by the parent (no external mask). `rope` is the
    parent GPT's shared RoPE cache, or None."""

    def __init__(self, d_model, n_heads, d_ff, dropout, arch: ArchConfig):
        super().__init__()
        self.arch = arch
        if arch.is_legacy:
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.ln2 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
            )
        else:
            self.ln1 = make_norm(arch, d_model)
            self.attn = ModernAttention(d_model, n_heads, dropout,
                                        n_kv_heads=arch.n_kv_heads, qk_norm=arch.qk_norm)
            self.ln2 = make_norm(arch, d_model)
            self.ffn = build_ffn(arch, d_model, d_ff, dropout)

    def forward(self, x, causal_mask, rope=None):
        h = self.ln1(x)
        if self.arch.is_legacy:
            x = x + self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)[0]
        else:
            x = x + self.attn(h, is_causal=True, rope=rope)
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    """A minimal, standard GPT: token + learned positional embeddings, a stack
    of pre-LN blocks, a final LN, and a tied LM head (weight-shared with the
    token embedding, the usual memory-saving choice)."""

    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, block_size,
                 dropout=0.1, arch: ArchConfig | None = None):
        super().__init__()
        arch = arch if arch is not None else ArchConfig()
        self.arch = arch
        self.block_size = block_size
        self.token_embed = nn.Embedding(vocab_size, d_model)
        # RoPE replaces the learned absolute position table -- so it exists only
        # when arch.rope is off. (A modern-but-rope=False build, e.g. RMSNorm-only,
        # still uses the learned table.)
        self.pos_embed = None if arch.rope else nn.Embedding(block_size, d_model)
        self.rope = RoPE(d_model // n_heads, block_size) if arch.rope else None
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, arch) for _ in range(n_layers)]
        )
        self.ln_f = make_norm(arch, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # GPT-2 initialization. Without this, PyTorch's default N(0,1) embeddings
        # make the tied-head logits explode (init CE in the hundreds), so the
        # first few hundred steps are wasted recovering -- which would unfairly
        # handicap the baseline and, worse, hits the wider model harder.
        self.apply(self._init_weights)
        for blk in self.blocks:  # scale residual output projections by 1/sqrt(2*L) (GPT-2)
            with torch.no_grad():
                blk.attn.out_proj.weight.mul_(1.0 / math.sqrt(2 * n_layers))
                # Legacy/GELU FFN is an nn.Sequential (last Linear at index 3);
                # SwiGLU's residual projection is `down`.
                ffn_out = blk.ffn.down if isinstance(blk.ffn, SwiGLU) else blk.ffn[3]
                ffn_out.weight.mul_(1.0 / math.sqrt(2 * n_layers))
        self.lm_head.weight = self.token_embed.weight   # weight tying (after init)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.MultiheadAttention):
            if m.in_proj_weight is not None:
                nn.init.normal_(m.in_proj_weight, mean=0.0, std=0.02)
            if m.in_proj_bias is not None:
                nn.init.zeros_(m.in_proj_bias)
            nn.init.normal_(m.out_proj.weight, mean=0.0, std=0.02)
            if m.out_proj.bias is not None:
                nn.init.zeros_(m.out_proj.bias)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        x = self.token_embed(idx)
        if self.pos_embed is not None:                 # learned absolute positions (legacy / non-RoPE modern)
            pos = torch.arange(t, device=idx.device).unsqueeze(0)
            x = x + self.pos_embed(pos)
        x = self.drop(x)
        # Legacy blocks take an explicit boolean causal mask (True = disallow),
        # MPS/AMP-safe. Modern blocks use SDPA is_causal + RoPE, so the mask is
        # unused there (None) -- built only when needed.
        mask = (None if not self.arch.is_legacy
                else torch.triu(torch.ones(t, t, dtype=torch.bool, device=idx.device), diagonal=1))
        for blk in self.blocks:
            x = blk(x, mask, rope=self.rope)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                   ignore_index=-1)
        return logits, loss


# ----------------------------------------------------------------------
def load_stream(tokenizer):
    """Tokenize the same wiki paragraphs used by the latent run into one stream."""
    docs = json.load(open(os.path.join(PROJECT, "wiki_docs.json")))
    ids = []
    for d in docs:
        ids.extend(tokenizer.encode(d))
    return torch.tensor(ids, dtype=torch.long), docs


def get_batch(stream, block_size, batch_size, device, generator):
    ix = torch.randint(len(stream) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([stream[i:i + block_size] for i in ix]).to(device)
    y = torch.stack([stream[i + 1:i + 1 + block_size] for i in ix]).to(device)
    return x, y


def lr_at(step, base, warmup, total, floor=0.1):
    if step < warmup:
        return base * (step + 1) / warmup
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog)))


@torch.no_grad()
def eval_ppl(model, stream, block_size, device):
    """Teacher-forced per-token NLL over the whole stream (non-overlapping blocks)."""
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(stream) - 1, block_size):
        chunk = stream[i:i + block_size + 1]
        if len(chunk) < 2:
            break
        x = chunk[:-1].unsqueeze(0).to(device)
        y = chunk[1:].unsqueeze(0).to(device)
        _, loss = model(x, y)
        tot += loss.item() * (len(chunk) - 1)
        n += len(chunk) - 1
    model.train()
    avg = tot / max(n, 1)
    return avg, math.exp(min(avg, 20))


@torch.no_grad()
def generate(model, tokenizer, prompt, device, n_new=40):
    model.eval()
    ids = tokenizer.encode(prompt)
    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(n_new):
        logits, _ = model(idx[:, -model.block_size:])
        nxt = int(logits[0, -1].argmax())
        idx = torch.cat([idx, torch.tensor([[nxt]], device=device)], dim=1)
    model.train()
    return tokenizer.decode(idx[0].tolist())


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="same-params", choices=list(PRESETS))
    # Modern-architecture A/B: --modern turns on the full stack (RMSNorm + RoPE +
    # QK-norm + SwiGLU); the granular flags override individual pieces so you can
    # bisect which upgrade moves the metric. Legacy (no flags) is byte-identical
    # to the original baseline.
    ap.add_argument("--modern", action="store_true",
                    help="RMSNorm + RoPE + QK-norm + SwiGLU (the full modern stack)")
    ap.add_argument("--arch-norm", choices=["layer", "rms"], default=None)
    ap.add_argument("--arch-rope", dest="arch_rope", action="store_true", default=None)
    ap.add_argument("--arch-qk-norm", dest="arch_qk_norm", action="store_true", default=None)
    ap.add_argument("--arch-ffn", choices=["gelu", "swiglu"], default=None)
    ap.add_argument("--arch-kv-heads", type=int, default=0, help="GQA groups (0 = full MHA)")
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = pick_device()
    set_seed(0)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    stream, docs = load_stream(tok)
    cfg = PRESETS[args.preset]
    # Assemble the ArchConfig: --modern is the full stack; granular flags win over it.
    base = dict(norm="rms", rope=True, qk_norm=True, ffn="swiglu") if args.modern else {}
    if args.arch_norm is not None:     base["norm"] = args.arch_norm
    if args.arch_rope is not None:     base["rope"] = args.arch_rope
    if args.arch_qk_norm is not None:  base["qk_norm"] = args.arch_qk_norm
    if args.arch_ffn is not None:      base["ffn"] = args.arch_ffn
    if args.arch_kv_heads:             base["n_kv_heads"] = args.arch_kv_heads
    arch = ArchConfig(**base)
    model = GPT(vocab_size=tok.vocab_size, block_size=args.block_size, arch=arch, **cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_emb = model.token_embed.weight.numel()
    n_emb += model.pos_embed.weight.numel() if model.pos_embed is not None else 0
    n_nonemb = n_params - n_emb
    print(f"[baseline {args.preset}] device={device} params={n_params:,} "
          f"(non-embedding {n_nonemb:,})  tokens={len(stream)}  cfg={cfg}  "
          f"arch={'legacy' if arch.is_legacy else arch}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(0)
    metrics = []
    for step in range(args.steps):
        lr = lr_at(step, args.lr, args.warmup, args.steps)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(stream, args.block_size, args.batch_size, device, gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % args.log_every == 0 or step == 0:
            nll, ppl = eval_ppl(model, stream, args.block_size, device)
            print(f"[step {step+1}] lr={lr:.2e} train_nll={loss.item():.4f} "
                  f"page_nll={nll:.4f} page_ppl={ppl:.1f}", flush=True)
            metrics.append({"step": step + 1, "train_nll": round(loss.item(), 4),
                            "page_nll": round(nll, 4), "page_ppl": round(ppl, 1)})

    nll, ppl = eval_ppl(model, stream, args.block_size, device)
    print(f"\n=== {args.preset}: final teacher-forced page NLL {nll:.3f}  perplexity {ppl:.1f} "
          f"(chance {math.exp(10.825):.0f}) ===")
    seed = docs[3][:60]
    print(f"\ngreedy generation (seed = {seed!r}):")
    print("  ", generate(model, tok, seed, device).replace("\n", " "))

    if args.out:
        out = os.path.join(PROJECT, args.out)
        os.makedirs(out, exist_ok=True)
        import dataclasses
        torch.save({"model_state": model.state_dict(), "cfg": cfg,
                    "arch": dataclasses.asdict(arch),
                    "preset": args.preset, "block_size": args.block_size,
                    "params": n_params, "final_nll": nll, "final_ppl": ppl},
                   os.path.join(out, "model.pt"))
        with open(os.path.join(out, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[baseline] saved -> {out}")


if __name__ == "__main__":
    main()
