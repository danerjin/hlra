"""
dialogue_chat.py
================
REPL for a Stage-F **chatbot** checkpoint, using the FULL two-lane serving path
(`dialogue.DialogueSession`): the user turn enters the read-only input lane, the
reply opens from the learned response seed, and reply "thoughts" + aged user turns
persist in the gestalt memory ACROSS turns (cross-turn memory). `:source` injects a
retrieved document (latent RAG). This is the proper Stage-F chat — `chat.py` only
runs the A→E generation path and ignores the dialogue machinery.

  python dialogue_chat.py                        # prompts for the checkpoint path
  python dialogue_chat.py runs/dialogue/model.pt # or pass it directly

At the prompt:
    <text>          -> your turn; prints the model's reply
    :source <text>  -> inject a retrieved source into memory (RAG; needs a 4-role/--rag model)
    :reset          -> clear the conversation memory (start a fresh session)
    :temp <float>   -> sampling temperature (default 0.9)
    :n <int>        -> max reply chunks (default 6)
    :greedy         -> toggle greedy decoding
    :q  /  <empty>  -> quit

NOTE: coherence tracks the run's scale, and Stage F is UNVALIDATED — treat output
as a plumbing demo of the serving path, not a working assistant.
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
        print("  (please enter a path, e.g. runs/dialogue/model.pt)")


def main():
    ckpt_path = ask_checkpoint(sys.argv)
    print(f"[dialogue] loading {ckpt_path} ...")
    try:
        model, adapter, chunker, cfg, ckpt = chat_core.load_dialogue_checkpoint(ckpt_path)
    except FileNotFoundError as e:
        raise SystemExit(f"[dialogue] {e}")

    info = chat_core.ckpt_summary(cfg, ckpt)
    n_roles = len(getattr(cfg, "role_tags", ("USER", "SELF", "SYSTEM")))
    have_adapter = "adapter_state" in ckpt
    print(f"[dialogue] ready. stage={info['stage_reached']} d_latent={cfg.d_latent} "
          f"roles={n_roles} adapter={'loaded' if have_adapter else 'ZERO-INIT (untrained)'}")
    if info["stage_reached"] != "F":
        print("[dialogue] NOTE: stage_reached is not 'F' -- the dialogue path will run "
              "but is untrained; expect noise.")
    print("[dialogue] commands: <text> | :source <t> | :reset | :temp f | :n k | :greedy | :q\n")

    # Pass `ckpt`: it carries end_gate_trained / stage_f_use_act, so the session
    # serves the way the checkpoint was trained. Without it the turn-end gate can
    # never turn on and a --no-act checkpoint would serve with ACT on.
    session = chat_core.new_dialogue_session(model, adapter, chunker, cfg, ckpt=ckpt)
    temperature, n_chunks, greedy = 0.9, 6, False

    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text or text in (":q", ":quit", ":exit"):
            break

        if text == ":reset":
            session.memory.reset()
            session.source_memory = None
            print("  (conversation memory cleared)\n")
            continue
        if text == ":greedy":
            greedy = not greedy
            print(f"  greedy decoding: {'ON' if greedy else 'OFF'}\n")
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
                print(f"  reply chunks = {n_chunks}\n")
            except (IndexError, ValueError):
                print("  usage: :n 6\n")
            continue
        if text.startswith(":source"):
            payload = text[len(":source"):].strip()
            if not payload:
                print("  (no source text)\n")
                continue
            try:
                n = session.add_source(payload)
                print(f"  (injected {n} RETRIEVED source chunk(s) into memory)\n")
            except Exception as e:
                print(f"  (add_source failed: {e}\n   -- needs a model built with a "
                      f"4-entry role_tags, i.e. trained with --rag)\n")
            continue

        reply, _read = chat_core.dialogue_reply(session, text, n_chunks=n_chunks,
                                                temperature=temperature, greedy=greedy)
        print(f"bot> {' '.join(reply) if reply else '(empty)'}\n")

    print("[dialogue] bye.")


if __name__ == "__main__":
    main()
