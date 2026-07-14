"""
lm_eval_adapter.py
==================
An EleutherAI lm-evaluation-harness adapter for this latent-thought language
model. It exposes the model as an `lm_eval` `TemplateLM` so the standard
harness tasks (HellaSwag, ARC, MMLU, LAMBADA, ...) can drive it, but the whole
scoring core lives in a single dependency-free method -- `_score_continuation`
-- so it can be unit-tested without `lm_eval` installed at all.

WHY THE OBVIOUS SCORING PATH IS WRONG
-------------------------------------
Every harness likelihood task boils down to `loglikelihood(context,
continuation)`: the conditional log P(continuation | context) under the model.
For a normal decoder LM you read that straight off the next-token logits. This
model has NO native token-level conditional logprob, and the reconstruction
path CANNOT be repurposed to fake one.

`forward_grounded` / `generate.py --score` compute a *reconstruction*
(autoencoder) NLL: they encode each chunk to a latent and decode THAT SAME
chunk's tokens from THAT SAME latent (empty memory, no HRM loop, notes §27).
The target tokens are baked into the very latent that then decodes them -- the
answer leaks into its own conditioning. That is exactly what you want for an
anti-collapse codec anchor, and exactly WRONG for scoring a prediction: it
measures "can the Talker copy a chunk it was handed", not "did the model
predict this continuation". Feeding a candidate answer through it would score
every candidate near-perfectly regardless of the context, so it cannot rank
continuations. It must not be used here.

HOW WE ACTUALLY SCORE: THE PREDICTIVE CHAIN
-------------------------------------------
We score the continuation the same way the model *generates* -- off the
forward-prediction map `pred_head`, the JEPA/SSL head the loop is trained on
(model.forward_self_supervised; generation in generate.py):

  1. READ the context: chunk it, run the HRM inner loop over its chunks to build
     the gestalt memory and carry the running thought `h` (mirrors
     generate.read_prompt). `h` is the finished thought after the last context
     chunk.
  2. For each continuation chunk t (in chunker order):
       a. pred = pred_head(prev_thought)            # prev_thought = h for t=0
          rescale pred onto the encoder-latent norm shell (model._rescale_to,
          ref_norm = sqrt(d_latent)) -- the cosine SSL objective trains
          pred_head's DIRECTION but not its scale, and the Talker consumes the
          latent unnormalized, so an un-rescaled prediction is off-distribution
          (see model.predict_next_latent / _rescale_to).
       b. s, n = score_tokens(chunk_t_tokens, pred) # teacher-forced token NLL
          of the TRUE continuation tokens decoded from the PREDICTED latent
          (empty memory = codec convention). Accumulate summed NLL + token count.
       c. Encode the TRUE continuation chunk and run the loop to advance the
          thought/memory (teacher forcing), then set prev_thought to it.
  3. logprob = -(sum of chunk NLLs); is_greedy = True iff every supervised token
     was the Talker's argmax from the predicted latent (best-effort).

This is the ONLY path that scores a continuation as a genuine prediction: the
tokens being scored never enter the latent they are scored under (step b), while
teacher forcing (step c) still conditions later chunks on the true earlier ones,
exactly as an autoregressive conditional likelihood requires.

CHUNK-BOUNDARY CAVEAT (read before trusting a task's numbers)
-------------------------------------------------------------
Scoring granularity is the CHUNK, and chunk boundaries are decided by the
chunker (SaT Capped), NOT by the harness. A continuation is whatever chunks the
chunker cuts it into; there is no per-token conditional. Two consequences:

  * The context/continuation split the harness intends may not fall on a chunk
    boundary. We chunk `context` and `continuation` INDEPENDENTLY so the
    continuation's tokens are never smuggled into the context's thought, which
    is the property that makes the score meaningful -- at the cost of the exact
    boundary being the chunker's choice.
  * Single-token (or few-token) continuations are the DEGENERATE WORST CASE.
    An MMLU-style " A" / " B" answer is one tiny chunk: the whole score is a
    single pred_head->Talker decode of ~one token, with none of the multi-chunk
    teacher-forced context the model was trained to use. Multiple-choice tasks
    whose options differ by a single token will be the least reliable use of
    this adapter; cloze/continuation tasks with sentence-length completions
    (LAMBADA, HellaSwag endings) sit much better on the chunking.

The load-bearing, unit-testable logic is `_score_continuation`; the `lm_eval`
wrapper around it is thin and only imported if the harness is installed.
"""
from __future__ import annotations

import os
import sys

# Repo modules import each other by BARE name (`from model import ...`), so the
# directory holding this file must be on sys.path. It already is when Python is
# launched from within files/, but the self-test is meant to run from the
# project root (`.venv/bin/python files/lm_eval_adapter.py`), where it is not --
# so insert it unconditionally, first, before importing any sibling module.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import math

import torch

import generate  # load(), read_prompt(), generate() -- the shared inference path
from config import model_config
from model import LatentThoughtModel, SELF
from gestalt_memory import GestaltMemoryBank
from data import build_offline_chunker, PAD

PROJECT = os.path.dirname(_HERE)
DEFAULT_CKPT = os.path.join(PROJECT, "runs", "model.pt")


# ----------------------------------------------------------------------
# lm_eval is an OPTIONAL dependency. Guard the import so this module (and its
# self-test / _score_continuation core) works with the harness absent. When
# present we subclass TemplateLM; when absent we subclass `object` and the
# harness-facing methods simply are not used.
# ----------------------------------------------------------------------
try:
    from lm_eval.api.model import TemplateLM
    from lm_eval.api.registry import register_model

    _HAVE_LM_EVAL = True
except Exception:  # pragma: no cover - exercised only when lm_eval is installed
    TemplateLM = object  # type: ignore[assignment,misc]
    _HAVE_LM_EVAL = False

    def register_model(*_names):  # no-op decorator stand-in
        def _wrap(cls):
            return cls

        return _wrap


@register_model("latent_thought", "latent-thought-lm")
class LatentThoughtLM(TemplateLM):
    """
    lm-evaluation-harness model wrapper for the latent-thought model.

    Construction loads a trained checkpoint through `generate.load` (the exact
    loader generation uses, so config/tokenizer/legacy-field handling is shared
    and can't drift). Pass `offline=True` to instead build a FRESH (untrained)
    model on a size preset with the offline stub chunker -- a dependency-free
    smoke that needs no checkpoint and no downloads (used by the self-test).

    The harness entry points (`loglikelihood`, `loglikelihood_rolling`,
    `generate_until`) are thin: they parse the harness request objects and hand
    the actual work to `_score_continuation` / `generate.generate`.
    """

    def __init__(self, ckpt: str = None, offline: bool = False,
                 preset: str = "smoke", vocab_size: int = 1024,
                 device: str = "cpu", **kwargs):
        # TemplateLM.__init__ is cheap/no-arg in the harness versions we target;
        # skip it entirely when running without lm_eval (TemplateLM is `object`).
        if _HAVE_LM_EVAL:
            super().__init__()
        self.device = torch.device(device)
        # generate.read_prompt (the shared context-reading path) builds its
        # tensors on CPU and never moves them, so a non-CPU model would hit a
        # CPU/GPU mismatch. The whole generate.py inference path is CPU-only by
        # design; keep the adapter consistent rather than silently crash on GPU.
        if self.device.type != "cpu":
            import warnings
            warnings.warn("LatentThoughtLM scoring runs on CPU (the shared inference "
                          "path is CPU-only); ignoring device=%s." % device)
            self.device = torch.device("cpu")

        if offline:
            # Dependency-free path: a fresh, UNTRAINED model on a preset. Good
            # enough to exercise the scoring plumbing end-to-end (finite,
            # continuation-dependent numbers) -- NOT to produce meaningful
            # likelihoods (an untrained pred_head/Talker predicts noise).
            cfg = model_config(preset, vocab_size=vocab_size)
            chunker = build_offline_chunker(cfg)
            model = LatentThoughtModel(cfg, chunker)
            model.eval()
        else:
            ckpt_path = ckpt or DEFAULT_CKPT
            if not os.path.isabs(ckpt_path):
                ckpt_path = os.path.join(PROJECT, ckpt_path)
            if not os.path.exists(ckpt_path):
                raise SystemExit(
                    f"no checkpoint at {ckpt_path} -- pass ckpt=... or offline=True")
            model, chunker, cfg, _ckpt = generate.load(ckpt_path)

        self.model = model.to(self.device)
        self.chunker = chunker
        self.cfg = cfg
        self._ref_norm = float(cfg.d_latent) ** 0.5  # encoder-latent norm shell

    # ==================================================================
    # THE CORE: dependency-free, unit-testable continuation scoring.
    # ==================================================================
    @torch.no_grad()
    def _score_continuation(self, context: str, continuation: str):
        """
        Log P(continuation | context) via the predictive chain (see the module
        docstring). Returns (logprob: float, is_greedy: bool).

        `logprob` is the negative summed teacher-forced token NLL of the
        continuation's tokens, each chunk decoded from the latent pred_head
        forecasts off the running thought (NOT from an encoding of the chunk
        itself -- that would leak the answer). `is_greedy` is True iff every
        supervised token was the Talker's argmax from the predicted latent.

        Does NOT touch lm_eval -- callable directly for tests.
        """
        model, chunker, cfg = self.model, self.chunker, self.cfg

        # ---- 1. READ the context: build gestalt memory + carry the thought.
        # Reuse generate.read_prompt verbatim so the read path is identical to
        # generation (chunk -> encode -> hrm_loop -> write SELF thought).
        memory, h_state, l_state, _last_latent = generate.read_prompt(
            model, chunker, cfg, context)
        prev_thought = h_state  # pred_head(h_t) -> chunk t+1 (may be None: empty ctx)

        # ---- 2. Chunk the continuation and score chunk-by-chunk.
        ct, cm = chunker.chunk_batch([continuation])            # (1, C, L), (1, C)
        total_nll = 0.0
        total_tok = 0
        all_greedy = True

        for t in range(ct.shape[1]):
            if not bool(cm[0, t]):
                continue
            chunk_ids = ct[:, t, :].to(self.device)             # (1, L)

            # a. Predict this chunk's encoder-space latent off the PREVIOUS
            #    thought, then rescale onto the encoder-latent norm shell.
            if prev_thought is None:
                # Empty/too-short context: no thought yet. Predict off a zero
                # thought (the loop's own initial H-state), the only defined
                # stance available -- degenerate but finite.
                prev_thought = torch.zeros(1, cfg.d_latent, device=self.device)
            pred = model.pred_head(prev_thought)                # (1, d_latent)
            ref_norm = pred.new_full((pred.shape[0], 1), self._ref_norm)
            pred = model._rescale_to(pred, ref_norm)

            # b. Teacher-forced token NLL of the TRUE tokens under the PREDICTED
            #    latent (empty-memory codec convention). The tokens being scored
            #    never entered `pred` -- this is the no-leak conditional score.
            s, n = model.score_tokens(chunk_ids, pred)
            total_nll += float(s)
            total_tok += int(n)
            if all_greedy:
                all_greedy = self._chunk_is_greedy(chunk_ids, pred)

            # c. Teacher forcing: ingest the TRUE continuation chunk to advance
            #    the thought/memory, exactly as read_prompt does per chunk.
            latent = model.chunk_encoder(chunk_ids, chunk_ids != 0)
            h_state, _ = model.hrm_loop(latent, memory, None, h_state=h_state,
                                        l_state=l_state, grad_window=5, use_act=False)
            l_state = h_state
            memory.write(h_state.detach(), SELF)
            prev_thought = h_state

        # Negative NLL is the log-likelihood the harness expects. A continuation
        # that produced NO scorable chunks -- an empty string, or one the chunker
        # dropped to zero chunks (whitespace-only, unusual unicode) -- must NOT
        # score 0.0: that is the MAXIMUM possible log-likelihood and would rank a
        # degenerate candidate above every real (negative-scoring) one. Return a
        # large-negative score so it can never win, and is not "greedy".
        if total_tok == 0:
            return -1e30, False
        return -total_nll, bool(all_greedy)

    @torch.no_grad()
    def _chunk_is_greedy(self, chunk_ids: torch.Tensor, latent: torch.Tensor) -> bool:
        """Best-effort argmax check for one chunk: True iff, decoding the chunk's
        tokens from `latent` under the codec Talker, every SUPERVISED position's
        argmax equals the true token. Uses the same supervised-position mask
        score_tokens uses (real tokens + the end-of-chunk PAD stop), so "greedy"
        means the model's own greedy decode would reproduce the chunk. This is
        an approximation of the harness's exact-match `is_greedy` at chunk (not
        whole-sequence) granularity."""
        empty_mem = GestaltMemoryBank(self.cfg.memory_capacity, self.cfg.d_latent)
        logits = self.model.talker(chunk_ids, latent, empty_mem)   # (1, L, vocab)
        mask = self.model._talker_target_mask(chunk_ids, chunk_ids.shape[1])
        argmax = logits.argmax(dim=-1)
        # A position passes if it's unsupervised OR its argmax matches the truth.
        return bool(((argmax == chunk_ids) | (~mask)).all())

    # ==================================================================
    # lm-evaluation-harness entry points (used only when lm_eval is present).
    # ==================================================================
    @staticmethod
    def _req_args(request):
        """Extract (context, continuation)-style args from a harness request.
        New lm_eval passes `Instance` objects with `.args`; be tolerant of a
        bare tuple too."""
        return getattr(request, "args", request)

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        """Score a batch of (context, continuation) requests. Returns a list of
        (logprob, is_greedy) tuples, one per request -- the harness contract for
        multiple-choice / cloze likelihood tasks."""
        out = []
        for req in requests:
            context, continuation = self._req_args(req)
            out.append(self._score_continuation(context, continuation))
        return out

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        """Rolling log-likelihood of a whole string (perplexity-style tasks like
        WikiText/LAMBADA-ppl): score the full text as a continuation from an
        EMPTY context. Returns a list of floats (logprob only)."""
        out = []
        for req in requests:
            (text,) = self._req_args(req)
            logprob, _greedy = self._score_continuation("", text)
            out.append(logprob)
        return out

    def generate_until(self, requests, disable_tqdm: bool = False):
        """Free-form generation tasks: run the shared generation path
        (generate.generate -- read the prompt, then pred_head->Talker decode new
        chunks) and truncate at the first requested stop string. Greedy by
        default (temperature is ignored under greedy)."""
        results = []
        for req in requests:
            context, gen_kwargs = self._req_args(req)
            gen_kwargs = dict(gen_kwargs or {})
            until = gen_kwargs.get("until") or []
            if isinstance(until, str):
                until = [until]
            n_chunks = int(gen_kwargs.get("max_gen_chunks", 3))
            text = generate.generate(self.model, self.chunker, self.cfg, context,
                                     n_chunks=n_chunks, greedy=True)
            for stop in until:
                if stop and stop in text:
                    text = text.split(stop)[0]
            results.append(text)
        return results

    # ---- TemplateLM abstract-method satisfiers ------------------------
    # We override `loglikelihood` wholesale (we work at chunk, not token,
    # granularity), so the token-level machinery below is not on our hot path --
    # but TemplateLM declares some of it abstract, so provide concrete
    # definitions so the class can be instantiated under lm_eval.
    @property
    def eot_token_id(self):
        return PAD

    def tok_encode(self, string: str, **kwargs):
        return self.chunker.tokenizer.encode(string, add_special_tokens=False)

    def _loglikelihood_tokens(self, requests, **kwargs):
        # This model has no token-level conditional logprob (that is the whole
        # reason for the chunk-level predictive chain). loglikelihood is
        # overridden to bypass this; it must exist only to satisfy the ABC.
        raise NotImplementedError(
            "LatentThoughtLM scores at chunk granularity via _score_continuation; "
            "there is no token-level _loglikelihood_tokens path.")


# ======================================================================
# Self-test: runs with NO checkpoint and NO lm_eval, from the project root.
#   .venv/bin/python files/lm_eval_adapter.py
# ======================================================================
def _self_test() -> int:
    torch.manual_seed(0)
    print("[self-test] building an OFFLINE untrained 'smoke' model "
          "(no checkpoint, no downloads, no lm_eval)...")
    lm = LatentThoughtLM(offline=True, preset="smoke", vocab_size=1024)
    print(f"[self-test] lm_eval importable: {_HAVE_LM_EVAL}  "
          f"d_model={lm.cfg.d_model} d_latent={lm.cfg.d_latent} "
          f"vocab={lm.cfg.vocab_size}")

    context = "some context text here."
    cont_a = "a candidate continuation."
    cont_b = "an entirely different ending that goes another way."

    lp_a, greedy_a = lm._score_continuation(context, cont_a)
    print(f"\ncontext      : {context!r}")
    print(f"continuation : {cont_a!r}")
    print(f"  logprob = {lp_a:.4f}   is_greedy = {greedy_a}")

    lp_b, greedy_b = lm._score_continuation(context, cont_b)
    print(f"continuation : {cont_b!r}")
    print(f"  logprob = {lp_b:.4f}   is_greedy = {greedy_b}")

    # Sanity assertions: finite, and the two continuations score differently
    # (proving the score actually depends on the continuation, not a constant).
    ok = True
    for name, lp in (("A", lp_a), ("B", lp_b)):
        if not math.isfinite(lp):
            print(f"[self-test] FAIL: logprob {name} is not finite ({lp})")
            ok = False
    if lp_a == lp_b:
        print("[self-test] FAIL: the two continuations scored IDENTICALLY "
              "-- the score is not continuation-dependent.")
        ok = False
    else:
        print(f"\n[self-test] the two continuations differ by "
              f"{abs(lp_a - lp_b):.4f} nats -- scoring is continuation-dependent.")

    # An empty continuation must score exactly 0.0 (log P over no tokens).
    lp_empty, _ = lm._score_continuation(context, "")
    if lp_empty != 0.0:
        print(f"[self-test] FAIL: empty continuation scored {lp_empty} (expected 0.0)")
        ok = False

    print("\n[self-test] " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
