"""
run_lm_eval.py -- one-command results-day benchmark runner.
==========================================================
Drives the trained latent-thought checkpoint through EleutherAI
lm-evaluation-harness via `lm_eval_adapter.LatentThoughtLM` and prints/saves a
results table. This is the push-button path meant to be run the moment the A->E
run's `model.pt` lands, so nothing about the harness is discovered on results day.

    .venv/bin/python files/run_lm_eval.py --ckpt runs/scaled/model.pt
    .venv/bin/python files/run_lm_eval.py --ckpt runs/scaled/model.pt \
        --tasks lambada_openai,hellaswag --limit 200 --output results/lm_eval.json

WHICH BENCHMARK IS HONEST HERE (read before quoting a headline number)
---------------------------------------------------------------------
This model scores continuations at CHUNK granularity off the predictive chain
(there is no token-level conditional logprob -- see lm_eval_adapter's module
docstring). That makes the task choice load-bearing:

  * DEFAULT = `lambada_openai`. A cloze/last-word task with sentence-length
    context; it maps cleanly onto chunk-level scoring and is reported as
    perplexity + accuracy. This is the honest headline benchmark for this model.
  * `hellaswag` also sits well -- multiple choice whose options are
    sentence-length endings (not single tokens).
  * `arc_challenge` is this adapter's DOCUMENTED WORST CASE: its options differ
    by a token or two, and a one-token continuation is a single pred_head->Talker
    decode with none of the multi-chunk context the model trains on. At `small`
    scale it will also sit near chance. It is available here for completeness,
    but do not lead with it. (This is why `notes.md`/`STAGE_F.md` call
    LAMBADA/HellaSwag the honest choice over ARC-C.)

Scoring runs on CPU (the shared inference path is CPU-only, enforced by the
adapter). Datasets are fetched from the HF Hub on first run; the adapter clears
the offline flag `import generate` sets so the download can happen.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# The adapter lives next to this file; make sibling imports work when launched
# from the project root (`.venv/bin/python files/run_lm_eval.py`).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Import the adapter FIRST: it clears the TRANSFORMERS_OFFLINE flag that
# `import generate` sets, which must happen before lm_eval/datasets import (they
# cache the offline flag at their own import time). Importing lm_eval before the
# adapter would re-introduce the offline-download failure this runner exists to
# avoid.
import lm_eval_adapter  # noqa: E402  (import-order is deliberate; see above)

# Honest default: the two tasks that map cleanly onto chunk-level scoring.
DEFAULT_TASKS = "lambada_openai,hellaswag"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=lm_eval_adapter.DEFAULT_CKPT,
                   help="path to the trained checkpoint (default: runs/model.pt)")
    p.add_argument("--tasks", default=DEFAULT_TASKS,
                   help="comma-separated lm-eval task names "
                        f"(default: {DEFAULT_TASKS})")
    p.add_argument("--limit", type=int, default=None,
                   help="cap examples per task (omit for the full task; "
                        "use e.g. 200 for a fast dry-run)")
    p.add_argument("--num-fewshot", type=int, default=None,
                   help="few-shot examples (default: the task's own default)")
    p.add_argument("--output", default=None,
                   help="write the full results JSON here (e.g. results/lm_eval.json)")
    p.add_argument("--device", default="cpu",
                   help="ignored beyond cpu -- the inference path is CPU-only")
    args = p.parse_args(argv)

    # Fail early and clearly if the harness isn't installed, rather than deep in
    # lm_eval's internals.
    if not lm_eval_adapter._HAVE_LM_EVAL:
        print("lm_eval is not installed. Install it first:\n"
              "    .venv/bin/python -m pip install 'lm_eval==0.4.4'", file=sys.stderr)
        return 2

    from lm_eval import simple_evaluate
    from lm_eval.utils import make_table

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"[run_lm_eval] ckpt={args.ckpt}  tasks={tasks}  "
          f"limit={args.limit}  device=cpu", flush=True)

    lm = lm_eval_adapter.LatentThoughtLM(ckpt=args.ckpt, device=args.device)
    print(f"[run_lm_eval] loaded: d_model={lm.cfg.d_model} "
          f"d_latent={lm.cfg.d_latent} vocab={lm.cfg.vocab_size}", flush=True)

    res = simple_evaluate(
        model=lm,
        tasks=tasks,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        bootstrap_iters=0,   # stderr via bootstrap is meaningless at chunk granularity
    )

    print("\n" + make_table(res))

    if args.output:
        out = args.output
        if not os.path.isabs(out):
            out = os.path.join(lm_eval_adapter.PROJECT, out)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        # `res` carries non-JSON-serialisable bits (functions); keep the parts a
        # results table / poster needs.
        payload = {
            "results": res.get("results"),
            "configs": res.get("configs"),
            "config": {k: v for k, v in (res.get("config") or {}).items()
                       if isinstance(v, (str, int, float, bool, type(None), list, dict))},
            "n-samples": res.get("n-samples"),
        }
        with open(out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\n[run_lm_eval] wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
