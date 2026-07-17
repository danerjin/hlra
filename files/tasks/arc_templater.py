"""
arc_templater.py -- opt-in "statement" variant of ARC-Challenge.
================================================================
Turns each (question, answer-option) pair into a single declarative SENTENCE,
so the thing the model scores is a full clause rather than a bare noun phrase.
Motivation: ARC-C's hard distractors are short, topically-identical minimal pairs
("an infectious, cell-cycle disease" vs "a non-infectious, chronic disease";
"the atom" vs "the electron"). A one- or two-word option is a single
chunk with almost nothing for the chunk-level scorer to grip; a full sentence
gives it a longer, more differentiated span. See README "Evaluating a trained
checkpoint" and the module docstring of `lm_eval_adapter.py`.

    Q:  "What is the smallest unit of copper that maintains its characteristics?"
    opt "the atom"   ->  "The atom is the smallest unit of copper."

This is scored by the SAME adapter and the same two score modes (`token_nll` /
`latent_cos`); the templater only changes the CHOICE TEXT the harness ranks. It
is exposed as the custom lm-eval task `arc_challenge_statement` (see
`files/tasks/arc_challenge_statement.yaml`), NEVER as a replacement for standard
`arc_challenge`.

WHY THE GUARDRAILS (read before trusting a number from this)
------------------------------------------------------------
Rewriting the benchmark with an LLM is powerful and dangerous:

  * REPRODUCIBILITY. A fresh rewrite every run means the score drifts run to run.
    So every rewrite is CACHED to disk (keyed by backend|model|question|option)
    and reused forever. The cache file is the reproducible artifact -- commit it
    and you can defend the exact inputs. Temperature is 0.
  * CONTAMINATION. A stronger model preprocessing the benchmark can leak signal
    (e.g. by disambiguating the correct option more than the distractors). This
    is therefore a DISTINCT, clearly-labelled task, and the prompt asks for a
    mechanical restatement that does NOT judge correctness -- symmetric across
    all four options. A number from it is "ARC-C (statement-rewritten by <model>)",
    not "ARC-C". Report it as such.
  * EVAL-TIME DEPENDENCY. The default backend is `deterministic` (a rule-based
    template, no LLM) so the eval never hard-depends on ollama. `ollama` is the
    opt-in refinement; any ollama failure falls back to the deterministic
    template for that item rather than crashing the run.

MODEL CHOICE MATTERS A LOT (measured 2026-07-17, `--compare`)
-------------------------------------------------------------
The rewriter's capability is the whole ballgame. On 24 real ARC-C options, with
the acceptance gate below (faithful AND not editorializing AND not over-long):

  * phi3   -- 46% accepted. Too loose: it paraphrases (synonym drift), rambles
              (too-long), and once flipped the correct "an infectious, cell-cycle
              disease" to "a non-infectious, immune system disorder" (a LABEL
              CORRUPTION the guard caught). Not usable for this.
  * gemma4 -- 100% accepted. It follows "reuse the exact words, only reorder"
              and produces the intended form verbatim: "The most likely effect
              of this increase in rotation is that planetary days will become
              shorter." This is why gemma4 is the default ollama model.

So the ollama path DOES work -- with a capable instruction-follower. Use gemma4
(or better); phi3 mostly falls back to the deterministic template and buys
nothing. (Small sample -- spot-check the cache at scale before trusting a
headline number.) Re-run the bake-off any time with
`--compare modelA,modelB --n N`.

Config is read from the environment (set by `run_lm_eval.py --arc-templater ...`,
or exported by hand):
  ARC_TEMPLATER_BACKEND  deterministic (default) | regex | ollama
  ARC_TEMPLATER_MODEL    ollama model tag (default: gemma4; phi3 rejects ~half)
  ARC_TEMPLATER_CACHE    cache json path (default: poster_data/arc_statements_cache.json)
  ARC_TEMPLATER_LIMIT    if set, only rewrite+keep the first N docs (keeps
                         --limit dry-runs cheap; matches lm-eval's "first N").
  ARC_TEMPLATER_OLLAMA_URL  default http://localhost:11434/api/generate

Standalone self-test (no lm_eval, no ollama needed -- exercises the deterministic
backend, the disk cache, and a monkeypatched LLM backend):
    .venv/bin/python files/tasks/arc_templater.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))            # files/tasks
PROJECT = os.path.dirname(os.path.dirname(_HERE))             # repo root
_DEFAULT_CACHE = os.path.join(PROJECT, "poster_data", "arc_statements_cache.json")

# Deliberately CONSTRAINED: phi3 (and small models generally) paraphrase too
# freely -- they swap synonyms ("gravity" -> "gravitational", "stronger" ->
# "intensify") and even negate ("infectious" -> "non-infectious"), which drifts
# the answer's meaning and trips the faithfulness guard so the rewrite is thrown
# away. The fix is to forbid rephrasing: reuse the EXACT words of the question and
# option, only REORDER them and add minimal glue. That preserves every content
# token verbatim, so the guard passes and we get a real full sentence instead of
# a fallback. The one-shot example anchors the "reorder, don't reword" behavior.
_OLLAMA_PROMPT = (
    "Combine the question and the answer option into ONE declarative sentence.\n"
    "STRICT RULES:\n"
    "1. Reuse the EXACT words from the Question and the Option. Do NOT replace any "
    "word with a synonym. Do NOT add, remove, negate, or change any information.\n"
    "2. You may ONLY reorder the existing words and insert small connective words "
    "if needed (is, are, was, the, a, an, that, of, to, will).\n"
    "3. Every word of the Option must appear, unchanged, in your sentence.\n"
    "4. Do NOT state whether the option is correct.\n"
    "5. Output ONLY the sentence, nothing else.\n\n"
    "Example:\n"
    "Question: What is the smallest unit of copper that maintains its properties?\n"
    "Option: the atom\n"
    "Sentence: The atom is the smallest unit of copper that maintains its properties.\n\n"
    "Question: {q}\nOption: {o}\nSentence:"
)


# ----------------------------------------------------------------------
# Backends. Each maps (question, option) -> a declarative sentence string.
# ----------------------------------------------------------------------
def _lower_lead(s: str) -> str:
    """Lower-case the first letter of an ordinary word so it reads mid-sentence,
    but leave acronyms/proper nouns alone (heuristic: only lower it when the 2nd
    char is already lower-case, so 'DNA'/'Earth' -> unchanged, 'The' -> 'the')."""
    s = s.strip()
    if len(s) >= 2 and s[0].isupper() and s[1].islower():
        return s[0].lower() + s[1:]
    return s


def deterministic(question: str, option: str) -> str:
    """Rule-based baseline: no LLM, fully reproducible, deliberately crude. Not a
    grammatical rewrite -- it just presents the option as a stated answer in one
    span. The point of the deterministic backend is to be the honest, dependency-
    free floor the LLM rewrite (and the regex backend) fall back to."""
    q = question.strip()
    opt = _lower_lead(option).rstrip(".")
    return f"{q} The answer is {opt}."


def _cap(s: str) -> str:
    s = s.strip()
    return (s[0].upper() + s[1:]) if s else s


# No-LLM declarativization for the common ARC stem forms. Each rule turns the
# QUESTION into a statement frame and slots the VERBATIM option in, so it is
# faithful by construction (only the option's leading case is lowered for flow;
# is_faithful is case-insensitive). Anything unmatched falls back to
# deterministic(). This is the "just do it with regex" path -- nicer sentences
# than deterministic for the forms it covers, still no model, fully reproducible.
_REGEX_RULES = [
    # "[context.] What/Which is|are|was|were (the) X?"  ->
    #   "[context.] The X is|are {option}."   Any leading context sentence(s) are
    #   kept verbatim as a prefix; only the trailing interrogative clause is
    #   declarativized. Group 1 = optional context ending in .!?; 2 = verb; 3 = predicate.
    #   "An astronomer observes ... impact. Which is the most likely effect ...?"
    #     -> "An astronomer observes ... impact. The most likely effect ... is {option}."
    #   "What is the smallest unit of copper?" -> "The smallest unit of copper is {option}."
    (re.compile(r"^(.*?[.!?]\s+)?(?:what|which)\s+(is|are|was|were)\s+(.+?)\s*\?$", re.I),
     lambda m, opt: "%s%s %s %s." % (m.group(1) or "", _cap(m.group(3)),
                                     m.group(2).lower(), _lower_lead(opt).rstrip("."))),
    # "What/Which does|do|will|would|can X ...?"  ->  "X ...: {option}." is too
    #   varied to declarativize cleanly, so those fall through to deterministic.
]


def regex_statement(question: str, option: str) -> str:
    """Rule-based declarativizer: convert common ARC question forms into a
    statement with the verbatim option slotted in; fall back to deterministic()
    for stems no rule matches. No LLM, reproducible, faithful by construction.

    Coverage is intentionally narrow: only "[context.] What/Which is/are X?"
    forms, where the answer is the predicate ("The X is {option}."). Measured
    ~16% of ARC-C options match a rule (the rest fall back to the crude
    template); 100% stay faithful. It is NOT broadened to "Which of these is a
    Y?" forms, where the answer is the SUBJECT ("{option} is a Y.") -- a regex
    cannot tell the two apart, and mislabelling the roles yields backwards
    sentences. For full coverage use the ollama backend with a capable model."""
    q = question.strip()
    for rx, fn in _REGEX_RULES:
        m = rx.match(q)
        if m:
            return fn(m, option)
    return deterministic(question, option)


_STOP = {"a", "an", "the", "of", "to", "in", "on", "is", "are", "was", "were",
         "be", "will", "would", "that", "this", "these", "those", "and", "or",
         "as", "at", "it", "its", "by", "for", "with", "become", "becomes"}


def _content_tokens(text: str) -> set:
    """Lower-cased word tokens worth checking for content preservation: split on
    non-alphanumerics, drop stopwords and 1-2 char tokens. 'cell-cycle' -> {cell,
    cycle}; 'non-infectious' -> {non, infectious} (so 'infectious' alone does not
    spuriously match the negated form)."""
    toks = re.split(r"[^a-z0-9]+", text.lower())
    return {t for t in toks if len(t) >= 3 and t not in _STOP}


def is_faithful(option: str, rewrite: str, min_overlap: float = 0.5) -> bool:
    """Heuristic guard against an LLM DRIFTING the option's content (the live
    failure that motivated this: phi3 rewrote the correct 'an infectious,
    cell-cycle disease' as 'a non-infectious, immune system disorder' -- it
    flipped the answer). Require that at least `min_overlap` of the option's
    content tokens survive as word-tokens in the rewrite. Imperfect by nature
    (rewriting is rephrasing), so it is a floor, not a proof; a rejected rewrite
    falls back to the verbatim-safe deterministic template."""
    opt = _content_tokens(option)
    if not opt:                       # option was all stopwords/short -> nothing to check
        return True
    kept = opt & _content_tokens(rewrite)
    return (len(kept) / len(opt)) >= min_overlap


# Judgment/hedge cues: phi3 sometimes appends a correctness verdict ("...is not
# necessarily accurate") that the word-level faithfulness check lets through
# because the option's words are still present. That commentary is asymmetric
# signal (it editorializes about THIS option), so reject it.
_JUDGE_RE = re.compile(
    r"\b(correct|incorrect|accurate|inaccurate|not necessarily|is true|is false|"
    r"is wrong|the right answer|best answer|however|therefore|in conclusion)\b",
    re.IGNORECASE)


def editorializes(rewrite: str) -> bool:
    """True if the rewrite injects a correctness judgment or hedge -- a rule-4
    violation the faithfulness guard can miss."""
    return bool(_JUDGE_RE.search(rewrite))


def within_length(question: str, option: str, rewrite: str, slack: int = 6) -> bool:
    """A restatement should be about as long as the question plus the option, not
    an expansion. Reject rewrites that balloon (the other way phi3 drifts:
    padding the option with an explanatory clause)."""
    return len(rewrite.split()) <= len(question.split()) + len(option.split()) + slack


def reject_reason(question: str, option: str, rewrite: str):
    """None if the rewrite is acceptable, else a short reason string. One gate for
    all three failure modes so statement() and the comparison tool agree."""
    if not is_faithful(option, rewrite):
        return "content-drift"
    if editorializes(rewrite):
        return "editorializes"
    if not within_length(question, option, rewrite):
        return "too-long"
    return None


def ollama(question: str, option: str, model: str, url: str, timeout: float = 90.0) -> str:
    """One temp-0 ollama generation. Returns the rewritten sentence, or raises on
    any transport/parse failure (the caller falls back to `deterministic`)."""
    import urllib.request

    body = json.dumps({
        "model": model,
        "prompt": _OLLAMA_PROMPT.format(q=question.strip(), o=option.strip()),
        "stream": False,
        "options": {"temperature": 0},
    }).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    text = (resp.get("response") or "").strip()
    # Keep the first non-empty line, drop a leading "Sentence:" echo if present.
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("sentence:"):
            line = line[len("sentence:"):].strip()
        if line:
            return line
    raise ValueError("empty ollama response")


# ----------------------------------------------------------------------
# Disk cache + the single entry point.
# ----------------------------------------------------------------------
def _key(backend: str, model: str, question: str, option: str) -> str:
    h = hashlib.sha1(("%s|%s|%s|%s" % (backend, model, question, option)).encode())
    return h.hexdigest()


def _load_cache(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(path: str, cache: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=0, sort_keys=True)
    os.replace(tmp, path)


def statement(question: str, option: str, *, backend: str, model: str,
              url: str, cache: dict, _ollama=ollama) -> str:
    """Cache-aware (question, option) -> sentence. Mutates `cache` in place; the
    caller is responsible for persisting it. `_ollama` is injectable for tests.
    An ollama failure falls back to the deterministic template for THAT item so a
    single flaky generation never aborts a 4,688-item run."""
    k = _key(backend, model, question, option)
    if k in cache:
        return cache[k]
    if backend == "ollama":
        try:
            s = _ollama(question, option, model, url)
            reason = reject_reason(question, option, s)
            if reason:
                # Content drift, editorializing, or ballooning -> don't score a
                # corrupted/biased option; fall back to the verbatim template.
                print("[arc_templater] rewrite rejected (%s) for %r -> %r; "
                      "deterministic fallback" % (reason, option, s))
                s = deterministic(question, option)
        except Exception as e:  # noqa: BLE001 -- any failure -> deterministic floor
            s = deterministic(question, option)
            print("[arc_templater] ollama failed (%s); deterministic fallback for: %r"
                  % (type(e).__name__, option))
    elif backend == "regex":
        s = regex_statement(question, option)
    else:
        s = deterministic(question, option)
    cache[k] = s
    return s


# ----------------------------------------------------------------------
# lm-eval hook: referenced as `!function arc_templater.process_docs` from the
# task yaml. Runs once over the split at task-load time.
# ----------------------------------------------------------------------
def process_docs(dataset):
    """Add a `statement_choices` column (one rewritten sentence per answer
    option, aligned to `choices.text`). Config comes from the environment so the
    bare-function signature lm-eval requires can still be parameterised."""
    backend = os.environ.get("ARC_TEMPLATER_BACKEND", "deterministic")
    model = os.environ.get("ARC_TEMPLATER_MODEL", "gemma4")
    cache_path = os.environ.get("ARC_TEMPLATER_CACHE", _DEFAULT_CACHE)
    url = os.environ.get("ARC_TEMPLATER_OLLAMA_URL",
                         "http://localhost:11434/api/generate")
    limit = os.environ.get("ARC_TEMPLATER_LIMIT")

    if limit:
        n = min(int(limit), len(dataset))
        dataset = dataset.select(range(n))

    # Preflight: a missing/unreachable ollama model would silently fall back to
    # deterministic for EVERY option (a whole run quietly not-templated). Probe
    # once and warn loudly instead of burying it in per-item fallback logs.
    if backend == "ollama":
        try:
            ollama("Which is a test?", "a test", model, url, timeout=30.0)
        except Exception as e:  # noqa: BLE001
            print("[arc_templater] WARNING: ollama model %r unreachable (%s: %s). "
                  "EVERY option will fall back to the deterministic template -- "
                  "this run is NOT LLM-templated. Pull the model / start ollama, "
                  "or pass --arc-templater deterministic to silence this."
                  % (model, type(e).__name__, str(e)[:80]))

    cache = _load_cache(cache_path)
    n_before = len(cache)
    total = len(dataset)
    print("[arc_templater] backend=%s model=%s: rewriting choices for %d docs "
          "(cache: %s, %d entries)"
          % (backend, model, total, cache_path, n_before))

    def _add(doc, idx):
        q = doc["question"]
        doc["statement_choices"] = [
            statement(q, opt, backend=backend, model=model, url=url, cache=cache)
            for opt in doc["choices"]["text"]
        ]
        if backend == "ollama" and (idx + 1) % 25 == 0:
            _save_cache(cache_path, cache)  # periodic flush on the slow path
            print("[arc_templater]   %d/%d docs" % (idx + 1, total))
        return doc

    dataset = dataset.map(_add, with_indices=True, load_from_cache_file=False)
    if len(cache) != n_before:
        _save_cache(cache_path, cache)
        print("[arc_templater] cache now %d entries (+%d) -> %s"
              % (len(cache), len(cache) - n_before, cache_path))
    return dataset


# ----------------------------------------------------------------------
# Backend comparison: which ollama model produces the most ACCEPTED rewrites
# (faithful, non-editorializing, not over-long) on real ARC-C? A diagnostic to
# pick a model, not part of the eval path.
#   .venv/bin/python files/tasks/arc_templater.py --compare phi3,gemma4 --n 6
# ----------------------------------------------------------------------
def compare_backends(models, n: int = 6, url: str = None) -> int:
    url = url or os.environ.get("ARC_TEMPLATER_OLLAMA_URL",
                                "http://localhost:11434/api/generate")
    # ARC-C is already cached locally (lm-eval downloaded it); force offline so
    # this diagnostic never depends on the Hub.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", os.path.join(PROJECT, ".hf_cache"))
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    docs = [ds[i] for i in range(min(n, len(ds)))]
    n_opts = sum(len(d["choices"]["text"]) for d in docs)
    print("Comparing %r on %d ARC-C questions (%d options).\n"
          "Accepted = faithful AND not editorializing AND not over-long.\n"
          % (models, len(docs), n_opts))

    for model in models:
        accepted, total, reasons, samples = 0, 0, {}, []
        for d in docs:
            q = d["question"]
            for opt in d["choices"]["text"]:
                total += 1
                try:
                    rw = ollama(q, opt, model, url)
                except Exception as e:  # noqa: BLE001
                    reasons["error"] = reasons.get("error", 0) + 1
                    continue
                r = reject_reason(q, opt, rw)
                if r is None:
                    accepted += 1
                    if sum(t == "OK" for _, _, t in samples) < 3:
                        samples.append((opt, rw, "OK"))
                else:
                    reasons[r] = reasons.get(r, 0) + 1
                    if sum(t != "OK" for _, _, t in samples) < 3:
                        samples.append((opt, rw, r))
        pct = 100.0 * accepted / total if total else 0.0
        print("=== %s ===" % model)
        print("  accepted %d/%d (%.0f%%)   rejects: %s"
              % (accepted, total, pct, reasons or "none"))
        for opt, rw, tag in samples:
            print("   [%-13s] %r" % (tag, rw))
            print("   %15s option: %r" % ("", opt))
        print()
    return 0


# ----------------------------------------------------------------------
# Self-test: no lm_eval, no ollama required.
# ----------------------------------------------------------------------
def _self_test() -> int:
    import tempfile

    ok = True
    q = "What is the smallest unit of copper that maintains its characteristics?"

    # 1. deterministic: reproducible, option-dependent, one sentence.
    d1 = deterministic(q, "the atom")
    d2 = deterministic(q, "the electron")
    print("[self-test] deterministic:\n   %r\n   %r" % (d1, d2))
    if d1 == d2 or "the atom" not in d1 or not d1.endswith("."):
        print("[self-test] FAIL: deterministic template malformed / not option-dependent")
        ok = False
    if deterministic(q, "the atom") != d1:
        print("[self-test] FAIL: deterministic template is not reproducible")
        ok = False

    # 1b. regex backend: declarativizes a matched stem, keeps the option verbatim,
    #     and falls back to deterministic on an unmatched stem.
    r_hit = regex_statement("Which is the most likely effect of the impact?",
                            "Planetary days will become shorter.")
    print("[self-test] regex (matched):\n   %r" % r_hit)
    if r_hit != "The most likely effect of the impact is planetary days will become shorter.":
        print("[self-test] FAIL: regex did not declarativize the matched stem as expected")
        ok = False
    if not is_faithful("Planetary days will become shorter.", r_hit):
        print("[self-test] FAIL: regex output dropped the option's content")
        ok = False
    r_miss = regex_statement("How do plants make food?", "by photosynthesis")
    if r_miss != deterministic("How do plants make food?", "by photosynthesis"):
        print("[self-test] FAIL: regex did not fall back to deterministic on an unmatched stem")
        ok = False

    # 2. cache round-trip + a monkeypatched 'LLM' backend that we can assert is
    #    called EXACTLY once per unique (q, option) (second call hits the cache).
    calls = {"n": 0}

    def fake_llm(question, option, model, url):
        calls["n"] += 1
        return "STATEMENT[%s]" % option

    cache = {}
    s1 = statement(q, "the atom", backend="ollama", model="fake", url="",
                   cache=cache, _ollama=fake_llm)
    s1b = statement(q, "the atom", backend="ollama", model="fake", url="",
                    cache=cache, _ollama=fake_llm)
    if s1 != "STATEMENT[the atom]" or s1 != s1b:
        print("[self-test] FAIL: injected backend / cache returned wrong value")
        ok = False
    if calls["n"] != 1:
        print("[self-test] FAIL: cache miss -- backend called %d times, expected 1"
              % calls["n"])
        ok = False

    # 3. ollama-failure falls back to deterministic (never raises).
    def boom(*a, **k):
        raise RuntimeError("simulated ollama down")

    s_fb = statement(q, "the nucleus", backend="ollama", model="fake", url="",
                     cache={}, _ollama=boom)
    if s_fb != deterministic(q, "the nucleus"):
        print("[self-test] FAIL: ollama failure did not fall back to deterministic")
        ok = False

    # 3b. faithfulness guard: the exact live failure (content drift) is caught,
    #     and a faithful rephrase is allowed.
    drift = "DFTD is best described as a non-infectious, immune system disorder."
    faith = "An infectious, cell-cycle disease is the best description of DFTD."
    if is_faithful("an infectious, cell-cycle disease", drift):
        print("[self-test] FAIL: guard passed a content-drifting rewrite")
        ok = False
    if not is_faithful("an infectious, cell-cycle disease", faith):
        print("[self-test] FAIL: guard rejected a faithful rewrite")
        ok = False
    # and statement() routes an unfaithful LLM output to the deterministic floor
    s_drift = statement("What describes DFTD?", "an infectious, cell-cycle disease",
                        backend="ollama", model="fake", url="", cache={},
                        _ollama=lambda *a, **k: drift)
    if s_drift != deterministic("What describes DFTD?", "an infectious, cell-cycle disease"):
        print("[self-test] FAIL: unfaithful rewrite was not replaced by deterministic")
        ok = False

    # 3c. anti-editorializing + length filters (the live phi3 failures where the
    #     option's words survive but commentary/padding is bolted on).
    q_g = ("An astronomer observes that a planet rotates faster after a meteorite "
           "impact. Which is the most likely effect of this increase in rotation?")
    opt_g = "Planetary gravity will become stronger."
    editorial = ("However, the statement Planetary gravity will become stronger is "
                 "not necessarily accurate.")
    if reject_reason(q_g, opt_g, editorial) != "editorializes":
        print("[self-test] FAIL: editorializing rewrite not rejected (got %r)"
              % reject_reason(q_g, opt_g, editorial))
        ok = False
    clean = "Planetary gravity will become stronger after the impact."
    if reject_reason(q_g, opt_g, clean) is not None:
        print("[self-test] FAIL: a clean rewrite was rejected (%r)"
              % reject_reason(q_g, opt_g, clean))
        ok = False
    balloon = ("Planetary gravity will become stronger " + "and heavier " * 20).strip()
    if reject_reason(q_g, opt_g, balloon) != "too-long":
        print("[self-test] FAIL: over-long rewrite not rejected (got %r)"
              % reject_reason(q_g, opt_g, balloon))
        ok = False

    # 4. disk cache persists and reloads.
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "c.json")
        _save_cache(p, {"k": "v"})
        if _load_cache(p) != {"k": "v"}:
            print("[self-test] FAIL: disk cache did not round-trip")
            ok = False

    # 5. live ollama, ONLY if explicitly requested (env), so the test stays
    #    dependency-free by default.
    if os.environ.get("ARC_TEMPLATER_SELFTEST_OLLAMA"):
        model = os.environ.get("ARC_TEMPLATER_MODEL", "phi3")
        try:
            live = ollama(q, "the atom", model,
                          "http://localhost:11434/api/generate")
            print("[self-test] live ollama(%s): %r" % (model, live))
        except Exception as e:  # noqa: BLE001
            print("[self-test] live ollama unavailable: %s" % e)

    print("[self-test] " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    argv = sys.argv[1:]
    if "--compare" in argv:
        i = argv.index("--compare")
        models = (argv[i + 1].split(",") if i + 1 < len(argv)
                  and not argv[i + 1].startswith("--") else ["phi3", "gemma4"])
        n = int(argv[argv.index("--n") + 1]) if "--n" in argv else 6
        raise SystemExit(compare_backends(models, n))
    raise SystemExit(_self_test())
