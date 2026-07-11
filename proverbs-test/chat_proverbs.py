"""
chat_proverbs.py
================
Tiny interactive wrapper around files/generate.py for the Proverbs smoke model.
Loads the checkpoint ONCE, then loops: read a line with input(), generate a
continuation. This is the same inference path generate.py uses (its load/
generate/score functions) -- just kept warm so you don't reload 472 MB per turn.

Usage:
    python chat_proverbs.py                       # uses runs/proverbs/model.pt
    python chat_proverbs.py --ckpt <path>

Chunk borders (the model thinks in chunk-level "thoughts") are shown with a
'|' separator: both for how your input text is split into chunks, and between
the chunks the model generates. Toggle the markers with ':sep'.

At the prompt:
    <any text>       -> generate a continuation from that text
    :score <text>    -> report avg NLL/token + perplexity for <text>
    :chunks <text>   -> just show how <text> is split into chunks (no generation)
    :sep             -> toggle the '|' chunk-border markers on/off
    :q  /  <empty>   -> quit
"""
import os, sys, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.join(os.path.dirname(HERE), "files")
sys.path.insert(0, FILES)  # import the repo's generate.py + deps

from generate import load, generate, score, _decode  # noqa: E402

DEFAULT_CKPT = os.path.join(HERE, "runs", "proverbs", "model.pt")


def chunk_view(chunker, cfg, text, sep):
    """Show how the SaT-Capped chunker splits `text` into the chunk-level
    'thoughts' the model reads -- each chunk decoded back to text, joined by
    `sep`. This is exactly the segmentation read_prompt/score operate on."""
    ct, cm = chunker.chunk_batch([text])
    tok = chunker.tokenizer
    parts = [_decode(tok, ct[0, t]).strip()
             for t in range(ct.shape[1]) if bool(cm[0, t])]
    parts = [p for p in parts if p]
    return sep.join(parts), len(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--chunks", type=int, default=3, help="how many chunks to generate")
    ap.add_argument("--temperature", type=float, default=0.9)
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"no checkpoint at {args.ckpt} -- train first (train_scaled.py)")

    print(f"[chat] loading {args.ckpt} ...")
    model, chunker, cfg, ckpt = load(args.ckpt)
    print(f"[chat] ready. stage={ckpt.get('stage_reached')} vocab={ckpt.get('vocab_size')}")
    print("[chat] commands: <text> generate | :score <text> | :chunks <text> | :sep | :q")
    print("[chat] chunk borders shown as '|'  (smoke-scale model -- output not coherent)\n")

    show_borders = True   # toggled by :sep

    while True:
        try:
            text = input("proverbs> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text or text in (":q", ":quit", ":exit"):
            break

        if text == ":sep":
            show_borders = not show_borders
            print(f"  chunk-border markers: {'ON' if show_borders else 'OFF'}\n")
            continue

        if text.startswith(":chunks"):
            payload = text[len(":chunks"):].strip()
            if not payload:
                print("  (no text to chunk)\n")
                continue
            view, n = chunk_view(chunker, cfg, payload, " | " if show_borders else " ")
            print(f"  [{n} chunks] {view}\n")
            continue

        if text.startswith(":score"):
            payload = text[len(":score"):].strip()
            if not payload:
                print("  (nothing to score)\n")
                continue
            nll, ppl = score(model, chunker, cfg, payload)
            print(f"  avg NLL/token = {nll:.3f}   perplexity = {ppl:.1f}\n")
            continue

        # Show how the model reads (chunks) the input, then generate.
        sep = " | " if show_borders else " "
        view, n = chunk_view(chunker, cfg, text, sep)
        print(f"  read  [{n} chunks]: {view}")
        cont = generate(model, chunker, cfg, text,
                        n_chunks=args.chunks, temperature=args.temperature,
                        separator=sep)
        print(f"  gen:  {cont}\n")

    print("[chat] bye.")


if __name__ == "__main__":
    main()
