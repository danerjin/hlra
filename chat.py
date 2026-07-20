"""
chat.py
=======
Interactive tester for a trained latent-thought checkpoint (the real A->E run).
On start it PROMPTS for a checkpoint path (input()), loads it once, then loops:
type text to continue it, with chunk borders shown as '|'.

  python chat.py                       # asks for the checkpoint path
  python chat.py runs/scaled/model.pt  # or pass it directly

At the prompt:
    <any text>       -> generate a continuation from that text
    :score <text>    -> teacher-forced reconstruction perplexity of <text>
    :chunks <text>   -> just show how <text> splits into chunks (no generation)
    :auto <text>     -> autoencoder round-trip: encode each chunk to its latent,
                        then decode that same latent back (encoder->Talker, no loop)
    :latent <text>   -> :auto plus a dump of the RAW latent vector per chunk
                        (norm, first components, full vector)
    :sep             -> toggle the '|' chunk-border markers on/off
    :temp <float>    -> set sampling temperature (default 0.9)
    :n <int>         -> set how many chunks to generate (default 3)
    :q  /  <empty>   -> quit
"""
import sys
import chat_core


def ask_checkpoint(argv):
    if len(argv) > 1:
        return argv[1]
    while True:
        try:
            path = input("checkpoint path: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if path:
            return path
        print("  (please enter a path, e.g. runs/scaled/model.pt)")


def main():
    ckpt_path = ask_checkpoint(sys.argv)
    print(f"[chat] loading {ckpt_path} ...")
    try:
        model, chunker, cfg, ckpt = chat_core.load_checkpoint(ckpt_path)
    except FileNotFoundError as e:
        raise SystemExit(f"[chat] {e}")
    info = chat_core.ckpt_summary(cfg, ckpt)
    print(f"[chat] ready. stage={info['stage_reached']} step={info['global_step']} "
          f"d_model={info['d_model']} vocab={info['vocab_size']}")
    print("[chat] commands: <text> | :score <t> | :chunks <t> | :auto <t> | :latent <t> "
          "| :sep | :temp f | :n k | :q")
    print("[chat] chunk borders shown as '|'  (output coherence depends on the run's scale)\n")

    show_borders = True
    temperature = 0.9
    n_chunks = 3

    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text or text in (":q", ":quit", ":exit"):
            break

        if text == ":sep":
            show_borders = not show_borders
            print(f"  chunk-border markers: {'ON' if show_borders else 'OFF'}\n")
            continue
        if text.startswith(":temp"):
            try:
                temperature = float(text.split(None, 1)[1])
                print(f"  temperature = {temperature}\n")
            except (IndexError, ValueError):
                print("  usage: :temp 0.8\n")
            continue
        if text.startswith(":n"):
            try:
                n_chunks = max(1, int(text.split(None, 1)[1]))
                print(f"  generating {n_chunks} chunks\n")
            except (IndexError, ValueError):
                print("  usage: :n 3\n")
            continue
        if text.startswith(":chunks"):
            payload = text[len(":chunks"):].strip()
            if not payload:
                print("  (no text to chunk)\n"); continue
            parts = chat_core.input_chunks(chunker, payload)
            sep = " | " if show_borders else " "
            print(f"  [{len(parts)} chunks] {sep.join(parts)}\n")
            continue
        if text.startswith(":score"):
            payload = text[len(":score"):].strip()
            if not payload:
                print("  (nothing to score)\n"); continue
            nll, ppl = chat_core.score_text(model, chunker, cfg, payload)
            print(f"  avg NLL/token = {nll:.3f}   perplexity = {ppl:.1f}\n")
            continue
        if text.startswith(":auto") or text.startswith(":latent"):
            dump_latent = text.startswith(":latent")
            cue = ":latent" if dump_latent else ":auto"
            payload = text[len(cue):].strip()
            if not payload:
                print("  (no text to autoencode)\n"); continue
            # greedy: for a codec-fidelity check we want the argmax reconstruction
            rows = chat_core.autoencode(model, chunker, cfg, payload, greedy=True)
            if not rows:
                print("  (nothing encodable)\n"); continue
            for i, r in enumerate(rows):
                lat = r["latent"]
                match = "  <-- exact" if r["recon"] == r["original"] else ""
                print(f"  chunk {i}:")
                print(f"    in : {r['original']!r}")
                print(f"    out: {r['recon']!r}{match}")
                print(f"    latent: dim={lat.numel()} norm={lat.norm():.4f} "
                      f"std={lat.std():.4f}")
                if dump_latent:
                    head = ", ".join(f"{v:+.4f}" for v in lat[:8].tolist())
                    print(f"    latent[:8] = [{head}, ...]")
                    print(f"    latent full = {lat.tolist()}")
            print()
            continue

        sep = " | " if show_borders else " "
        read = chat_core.input_chunks(chunker, text)
        print(f"  read [{len(read)} chunks]: {sep.join(read)}")
        gen = chat_core.generate_chunks(model, chunker, cfg, text,
                                        n_chunks=n_chunks, temperature=temperature)
        print(f"  gen: {sep.join(gen)}\n")

    print("[chat] bye.")


if __name__ == "__main__":
    main()
