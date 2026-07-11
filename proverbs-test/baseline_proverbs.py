"""
baseline_proverbs.py
====================
Fair-scale GPT baseline for the Proverbs run: SAME data, tokenizer, optimizer,
schedule, step budget, batch size, and device as train_scaled.py -- only the
architecture differs (a plain causal decoder-only Transformer instead of the
latent-thought loop). Reuses the GPT model + helpers from files/baseline_gpt.py.

Crucially it reproduces the *identical* seed-0 train/val chapter split that
train_scaled.py derived from the 31-chapter cache, so the GPT is trained on the
same 23 chapters and evaluated on the same 8 held-out chapters as the latent model.

  preset same-params  : d512x6  (~44M, matches the latent model's TOTAL params)
  preset same-compute : d192x10 (~14M, matches the latent model's WIDTH+compute)

HONEST ASYMMETRY (same caveat as files/baseline_gpt.py): the GPT does pure causal
next-token prediction. The latent model's reconstruction NLL is an autoencoder
through a latent bottleneck (it conditions on an encoding of the chunk it decodes
-- lookahead -- but must pass everything through a 192-d thought + FIFO memory).
Different objectives; both are "teacher-forced per-token NLL under each model's own
task." Compare the *shape* (memorize train / generalize to held-out) more than the
absolute nats.

Run:
    python baseline_proverbs.py --preset same-params  --out runs/baseline_same_params
    python baseline_proverbs.py --preset same-compute --out runs/baseline_same_compute
"""
import os, sys, json, math, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.join(os.path.dirname(HERE), "files")
sys.path.insert(0, FILES)

import torch
from baseline_gpt import GPT, PRESETS, lr_at, eval_ppl, generate  # noqa: E402
from utils import set_seed                                          # noqa: E402

TOKENIZER_DIR = os.path.join(os.path.dirname(HERE), "gpt2_tok")

# The same probe sentences scored against the latent model, so the two models'
# numbers sit side by side.
PROBES = [
    ("Proverbs 1:7 (verbatim, in TRAIN)",
     "The fear of Yahweh is the beginning of knowledge; but the foolish despise wisdom and instruction."),
    ("reworded 15:1 (held-out/paraphrase)",
     "A soft answer turns away wrath, but a harsh word stirs up anger."),
    ("out-of-domain (modern finance)",
     "The quarterly earnings report exceeded analyst expectations by a wide margin."),
]


def load_chapters():
    with open(os.path.join(HERE, "proverbs.jsonl")) as f:
        return [json.loads(line)["text"] for line in f if line.strip()]


def same_split(n):
    """Reproduce train_scaled.py's split EXACTLY: seed-0 randperm, val = first
    max(8, n//20) indices. The cache preserved chapter order, so dataset index i
    == chapter i, and these indices name the same held-out chapters."""
    val_n = min(256, max(8, n // 20))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
    return sorted(perm[val_n:]), sorted(perm[:val_n])   # train_idx, val_idx


def build_stream(chapters, idxs, tok):
    ids = []
    for i in idxs:
        ids.extend(tok.encode(chapters[i]))
    return torch.tensor(ids, dtype=torch.long)


def get_batch(stream, block_size, batch_size, device, generator):
    hi = len(stream) - block_size - 1
    ix = torch.randint(max(1, hi), (batch_size,), generator=generator)
    x = torch.stack([stream[i:i + block_size] for i in ix]).to(device)
    y = torch.stack([stream[i + 1:i + 1 + block_size] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def score_text(model, tok, text, device, block_size):
    """Teacher-forced causal per-token NLL of `text` (the GPT analogue of the
    latent model's --score). One left-to-right pass, no lookahead."""
    model.eval()
    ids = tok.encode(text)[:block_size + 1]
    if len(ids) < 2:
        return float("nan"), float("nan")
    x = torch.tensor(ids[:-1], device=device).unsqueeze(0)
    y = torch.tensor(ids[1:], device=device).unsqueeze(0)
    _, loss = model(x, y)
    model.train()
    nll = float(loss)
    return nll, math.exp(min(nll, 20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="same-params", choices=list(PRESETS))
    ap.add_argument("--steps", type=int, default=1800)      # match the latent A->E budget
    ap.add_argument("--batch-size", type=int, default=8)    # match train_scaled --batch-size 8
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)       # match
    ap.add_argument("--warmup", type=int, default=100)      # match total_steps//50 -> 100
    ap.add_argument("--log-every", type=int, default=25)    # match
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = args.device
    set_seed(0)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    chapters = load_chapters()
    train_idx, val_idx = same_split(len(chapters))
    train_stream = build_stream(chapters, train_idx, tok)
    val_stream = build_stream(chapters, val_idx, tok)
    print(f"[baseline_proverbs] {len(chapters)} chapters -> "
          f"{len(train_idx)} train / {len(val_idx)} val "
          f"(val chapters = {[i+1 for i in val_idx]})")
    print(f"[baseline_proverbs] train tokens={len(train_stream)} val tokens={len(val_stream)}")

    cfg = PRESETS[args.preset]
    model = GPT(vocab_size=tok.vocab_size, block_size=args.block_size, **cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[baseline {args.preset}] device={device} params={n_params:,} cfg={cfg}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(0)
    metrics = []
    for step in range(args.steps):
        lr = lr_at(step, args.lr, args.warmup, args.steps)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(train_stream, args.block_size, args.batch_size, device, gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % args.log_every == 0 or step == 0:
            tr_nll, tr_ppl = eval_ppl(model, train_stream, args.block_size, device)
            va_nll, va_ppl = eval_ppl(model, val_stream, args.block_size, device)
            print(f"[step {step+1}] lr={lr:.2e} train_nll={loss.item():.4f} "
                  f"train_ppl={tr_ppl:.1f} val_ppl={va_ppl:.1f}", flush=True)
            metrics.append({"step": step + 1, "train_nll": round(tr_nll, 4),
                            "train_ppl": round(tr_ppl, 1), "val_nll": round(va_nll, 4),
                            "val_ppl": round(va_ppl, 1)})

    tr_nll, tr_ppl = eval_ppl(model, train_stream, args.block_size, device)
    va_nll, va_ppl = eval_ppl(model, val_stream, args.block_size, device)
    print(f"\n=== {args.preset}: TRAIN ppl {tr_ppl:.1f} (nll {tr_nll:.3f}) | "
          f"HELD-OUT VAL ppl {va_ppl:.1f} (nll {va_nll:.3f})  [chance ~{math.exp(10.825):.0f}] ===")

    print("\nprobe scores (teacher-forced causal NLL/token):")
    probe_out = []
    for label, text in PROBES:
        nll, ppl = score_text(model, tok, text, device, args.block_size)
        print(f"  {label:38s} nll={nll:.3f}  ppl={ppl:.1f}")
        probe_out.append({"label": label, "nll": round(nll, 3), "ppl": round(ppl, 1)})

    seed = chapters[train_idx[0]][:60]
    print(f"\ngreedy generation (seed = {seed!r}):")
    print("  ", generate(model, tok, seed, device).replace("\n", " "))

    if args.out:
        out = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
        os.makedirs(out, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "cfg": cfg, "preset": args.preset,
                    "block_size": args.block_size, "params": n_params,
                    "final_train_ppl": tr_ppl, "final_val_ppl": va_ppl}, os.path.join(out, "model.pt"))
        json.dump({"metrics": metrics, "probes": probe_out,
                   "final": {"train_ppl": tr_ppl, "val_ppl": va_ppl,
                             "train_nll": tr_nll, "val_nll": va_nll},
                   "params": n_params, "val_chapters": [i + 1 for i in val_idx]},
                  open(os.path.join(out, "metrics.json"), "w"), indent=2)
        print(f"\n[baseline_proverbs] saved -> {out}")


if __name__ == "__main__":
    main()
