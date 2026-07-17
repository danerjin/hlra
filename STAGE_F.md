# Stage F — Chatbot fine-tuning, tagging, RAG (design + implementation)

Stage F turns the A→E foundation model into a **chatbot**. Everything here is
**additive and opt-in**: with all Stage-F flags off, the model is byte-identical
to the validated A→E model (the `latent_mult=1`-style discipline), and no A→E
code path (`forward_grounded` / `forward_self_supervised` / `trainer.py`) is
touched. **Status: implemented and smoke-verified on offline synthetic data;
NOT trained, NOT validated.** Nothing here has seen a real run. (The old
"uncommitted-review-only" tag was stale — this file has been committed since
`703d6a9c`.)

> The design rationale lives in `latent-thought-architecture.md` §4. This file is
> the map of what got built and how to drive it.

---

## 1. The core idea: SFT is the prediction objective, with the input/self boundary

The A→E predictive objective (`pred_head(h_t) ≈ EMA(z_{t+1})`) already *is*
next-thought prediction. Stage F makes it **supervised** by switching the data to
dialogue and masking the loss to the **assistant (SELF)** turns — the latent-space
analog of SFT prompt-masking. Three separations, enforced at three levels:

| Layer | What | Where enforced |
|---|---|---|
| **1. Structural** | who may write the recurrent belief state | free: the input lane is only ever cross-attention K/V; it has no path to `h_state`/`l_state` (`input_lane.py`) |
| **2. Informational** | the target must not be visible while predicted | data contract: user turn → lane, assistant turn → target, disjoint strings (`dialogue_data.tensorize_*`). For the *document* predictor (`forward_self_supervised` with lanes on), the raw-token lane is **dropped** — the cached window is the doc's trailing tokens, i.e. future chunks, so it would leak the target; only causal prior-turn (USER/SYSTEM) gestalts may enter. Fixed 2026-07-13. |
| **3. Behavioral** | "the user asserted X" ≠ "I concluded X" | a **training signal**: `anti_sycophancy_loss` + the trust gate (role tags alone are only an affordance) |

Layer 3 is the load-bearing one — a model can have a perfect structural boundary
and still be a sycophant.

## 2. The Stage-F objective (`model.forward_dialogue`)

Per assistant chunk *t* (masked to valid response chunks; user turn is never a
target):

```
h_{-1} = loop(response_seed, memory, lane)          # open the reply (seed injection)
h_{t-1}= loop(z_{t-1}, memory, lane)                # teacher-forced: ingest TRUE prev chunk
cos :  pred_head(h_{t-1}) ~ EMA(z_t)                # predict the next thought (latent SFT)
gen :  Talker(rescale(pred_head(h_{t-1}))) -> z_t's TRUE tokens   # decode the prediction
```

- **`score_tokens(token_ids, latent)`** is the shared primitive: teacher-forced
  token NLL of given tokens decoded from an *externally supplied* latent. It is
  the one op neither A→E forward exposes (`forward_grounded` leaks the target into
  its own conditioning), and it closes the gap where A→E only ever trains the
  Talker from `encode(chunk)`, never from `pred_head`'s off-distribution latent.
  It is also the primitive the **lm-eval adapter** scores with.
- The **reconstruction anchor** (`forward_grounded`) keeps running so the codec
  doesn't drift during SFT.
- The **response seed** (a learned injection to open a reply, since §4.1 forbids
  compressing the user turn into a thought) lives in `dialogue.DialogueAdapter`,
  deliberately OUT of the base model so the A→E `state_dict` stays byte-identical.

## 2.1 The learned turn-end (`end_head`, `losses.turn_end_loss`)

**The gap.** PAD is a *trained* end-of-**chunk** stop — `model._talker_target_mask`
supervises every real token plus the first PAD of a shorter-than-max chunk (§19.2),
and `dialogue._decode_chunk` breaks on it. Nothing was the end of a **turn**. There
is no EOS token anywhere in the model, and `DialogueSession.reply` looped
`for _ in range(max_chunks)` with no reachable exit (its `if not ids: break` is dead
code — `_decode_chunk` bans PAD at position 0, so it always returns ≥ 1 id). A
chatbot that cannot stop talking emits exactly `max_chunks` chunks every turn,
whether the answer finished at chunk 2 or was still going at chunk 6.

**The objective.** After the loop ingests response chunk *t* and forms the thought
`h_t`, a head predicts *the turn ends here*:

```
end:  sigmoid(end_head(h_t))  ~  1[ chunk t is the last chunk of the response ]
```

`h_t` is read at the **same point in the loop** `reply()` reaches when it must
decide whether to emit another chunk, through **the same head** — the head is read
after the loop ingests chunk *t*, in both paths, and the alternative (predicting off
the thought that *generated* chunk *t*) would answer a different question the
generation loop cannot ask.

**But "the same tensor" is only true with ACT OFF, and that is a real caveat.**
`hrm_loop`'s ACT halt vote is a **batch mean**, so a row's loop depth is decided by
its batchmates. Training runs `B = batch_size`; `reply()` runs `B = 1`. With ACT on
(the Stage-F default) the gate is therefore supervised on an `h_t` the server does
not reproduce — measured on a discriminative halting head, row 0's end logit moved
1.36 between B=8 and B=1, and the two sides landed on **opposite sides of the 0.5
threshold** (P(end) 0.574 training vs 0.258 serving) at exactly the final chunk, the
only "end"-labelled thought. With ACT off the two agree to ~4e-06. This is not
introduced by the gate — per-row halting is `experiments.md` #2 — but the gate is the
first thing that depends on train/serve `h_t` agreement, so `train_dialogue.py` warns
when both are on and **`--no-act` exists to actually act on that advice** (ACT was
hardcoded on in `stage_f_flags` before, which made the recommendation unfollowable).

**The label is free, and was being discarded.** It is already in the SFT batch:
`resp_mask[:, t+1]` says whether a chunk *t+1* exists. So there is **no data-format
change** — `tensorize_sft` / `collate_sft` are untouched.

**The truncation confound (`model._turn_end_labels`).** The free label is not
unconditionally correct. `chunker.chunk_batch` caps a response at
`M = max_chunks_per_doc`, so a row that fills all *M* slots either ended exactly at
*M* or was cut off at *M* — indistinguishable. Its final "the turn ends here" label
is therefore **unknown, not True**; trained on naively it teaches *every turn ends
after exactly M chunks*. Only that one ambiguous label per filled row is masked out;
every earlier "a chunk follows" label is correct either way and is kept.

**The masking deletes only POSITIVES, so watch `end_pos`, not `end_n`.** A filled
row's single "end" label *is* the one dropped — every label it keeps is a "continue".
So a batch of long, M-filling responses (the realistic long-SFT-answer regime) yields
`end_n = 44` with **zero positives**: the BCE falls to 0.000, `end_acc` rises to
1.000, `end_n` looks generous, and the head has learned **"never end"** — the exact
failure this objective exists to prevent, wearing a perfect scorecard. `end_pos`
(surviving positives) is the only honest metric; at `end_pos = 0` nothing else means
anything, and `train_dialogue.py` warns after 50 consecutive dry batches. If your
data fills `max_chunks_per_doc`, raise it or use shorter responses.

Corollary: position `t = M-1` can never be supervised (a row reaching it is filled →
masked; a shorter row has no such position), so the gate is untrained at exactly the
cap. And `max_chunks_per_doc = 1` masks *everything* — no shipped preset does this
(12/32/48/64), but it fails silently to `end_pos = 0` rather than erroring.

**Where it lives.** `end_head` is in `dialogue.DialogueAdapter`, next to
`response_seed` and for the same reason: the **base model's** `state_dict` is
unchanged, so an A→E checkpoint still loads strict and the A→E run is untouched.
Verified: `forward_grounded` / `forward_self_supervised` /
`forward_self_supervised_halt` are bit-for-bit unchanged (losses + every
per-parameter grad norm, float64, across every stage flag combo and 5 arch configs).

Note the scope: this is the **base model's** state_dict. The **adapter's** grew by
`end_head.*`, so a Stage-F checkpoint written *before* the gate no longer loads
strict — `train_dialogue` and `chat_core` both load it non-strictly and say so.

⚠️ **Adding a new head is not free, even switched off.** `nn.Linear.__init__` runs
`reset_parameters()`, which **consumes global RNG draws**; zeroing the weights
afterward fixes the values but does **not** rewind the stream. With `dropout=0.1`
live, merely constructing `end_head` shifted every later dropout mask — 130/137 base
tensors differed after 3 Stage-F steps at `end_weight=0`, i.e. "off" was not off.
`end_head` is therefore built with `torch.nn.utils.skip_init` (no draw on any device,
unlike CPU-only `get`/`set_rng_state`). **Do the same for the next head you add**, and
note that a probe which reseeds before its training loop cannot see this class of bug.

**Gradient routing.** `end_grad=False` (default) reads a **detached** `h_t`, so the
BCE trains only the head and never reshapes the reasoning — the
`forward_self_supervised_halt` convention. Verified: with it off, *every* base-model
parameter receives **exactly zero** gradient from this term (the only tensors with a
gradient are `end_head.weight`/`.bias`). With `--end-grad` the gradient does reach the
loop and encoder — but **not at step 0**: `end_head.weight` is zero-init, so ∂/∂`h_t`
is identically zero until the weight moves off zero (which the head's own gradient
does immediately). Connectivity is real; step-0 magnitude is zero by construction.

**Serving.** `max_chunks` becomes a *cap*, not the reply length — but **only for a
checkpoint that actually trained the gate**. `train_dialogue.save` records
`end_gate_trained`; `chat_core.new_dialogue_session(..., ckpt)` reads it and switches
the gate on only then. `DialogueSession(use_end_head=False)` is the default, which is
exactly the pre-gate fixed-length behavior. This is deliberate: an untrained head is
*not* inert (P = 0.018 **per chunk** ≈ 10% of 6-chunk replies), so defaulting it on
would truncate replies from any A→E or `end_weight=0` checkpoint.

```bash
# --no-act is recommended WITH the gate: ACT's batch-mean halt vote makes the
# server's B=1 h_t differ from the one the gate was trained on (see above).
python train_dialogue.py --ckpt runs/scaled/model.pt --multi-turn \
    --end-weight 0.5 --no-act
python train_dialogue.py ... --end-weight 0.5 --no-act --end-grad   # the A/B (see limits)
```
`save()` records both `end_gate_trained` and `stage_f_use_act`, and
`chat_core.new_dialogue_session(..., ckpt)` reads them, so serving runs the loop the
way training did instead of guessing.

**Honest limits — read before trusting it.** Turning this on does *not* mean the gate
works:
- **Unvalidated on real dialogue.** Smoke-scale only.
- **There is NO evidence yet that the gate extracts signal — the plumbing is only
  verified to *run*.** An earlier draft of this section claimed that a smoke overfit
  batch reaching BCE 0.455 against a 0.598 base-rate entropy showed "real signal,"
  because a constant head cannot beat `H(p)`. That inference is **invalid** and the
  claim is withdrawn. With ~14 supervised points in a 192-d latent the head can
  separate *any* labeling: a null run of the identical head on **pure noise features
  with random labels** reaches BCE 0.008 and beats the base-rate entropy in **20/20
  seeds**. Beating `H(p)` here is what a signal-free head does. Worse, 0.455 is far
  *above* the null's 0.008 — the measurement shows an **underfit** head, and says
  nothing about signal in either direction. A real answer needs a **held-out split**,
  which has not been run.
- **The detached head may not be able to learn this at all.** `end_grad=False` lets it
  read only what `cos`/`gen` already put in `h_t`, and nothing forces them to encode "I
  am done" — the same shape as the halt gate degenerating to a constant P(halt) ≈ 0.95.
  If the gate will not move on real data, **try `--end-grad` first**.
- **`end_acc` is imbalance-blind** — one "end" per turn, so an always-continue head
  scores ≈ 1 − 1/M (0.714 on the smoke batch). Never read alone. And see the `end_pos`
  warning above: at `end_pos = 0`, `end`/`end_acc` look *perfect* and mean nothing.
- **An untrained gate is NOT inert at serve time.** sigmoid(−4.0) = 0.018 is a
  *per-chunk* rate, so ~10% of 6-chunk replies (≈20% of 12-chunk) would stop early at
  random. Serving therefore keeps the gate **off** unless the checkpoint's
  `end_gate_trained` flag says it was really trained (`DialogueSession(use_end_head=)`,
  set by `chat_core.new_dialogue_session`). Off = exactly the pre-gate behavior.

## 3. Anti-sycophancy (`forward_anti_sycophancy` + `losses.anti_sycophancy_loss`)

Two user turns that differ ONLY in an asserted premise (asserts X vs. not-X); the
correct answer is the same. Each premise is compressed into a **USER gestalt in
memory** (not the lane), and the model's opening stance must be invariant to which
was asserted. The loss = both variants match the role-invariant truth **and** each
other. The premise flows through the **trust-gated memory read**, so the loss
drives `trust(USER)` down. This also trains the USER memory path (A→E only ever
writes SELF).

## 4. Tagging (`config.*` flags, `GestaltCrossAttentionReader`)

All opt-in; off = the discrete `nn.Embedding` tag, byte-identical.

| Flag | Effect |
|---|---|
| `soft_role_tags` | tag = soft mixture over a shared learned **codebook** + learned temperature (graded provenance, roles share/warm structure) |
| `soft_role_content` | the mixture also bends with slot **content** (the *dynamic* shift; needs `soft_role_tags`) |
| `trust_gate` | learned scalar in (0,1) per slot, from the tag, scaling the slot's **value** (not key) — attend but discount. The anti-sycophancy hook. |
| `trust_gate_vector` | per-dimension gate: discount a polarity subspace, keep topic |
| `persona_tags` (`n_personas`) | per-**speaker** embedding, indexed by a conversation-local id (0..P-1, generalizes across dialogues), added on top of the role. Distinguishes >3 speakers without a global vocabulary. |
| `gestalt_readout` | project self-thoughts AND external content through one projection onto the thought shell, so the memory bank is homogeneous (§Q2) |

- **Roles vs personas**: role = coarse provenance (USER/SELF/SYSTEM/RETRIEVED);
  persona = *who* within this conversation. Many speakers → distinct personas
  (and/or a larger `role_tags`).
- **Dynamic tags**: `reader.tag_trajectory(memory, device)` returns the per-slot
  soft-mixture weights — read across a speaker's successive turns to *see* their
  provenance mixture shift during the conversation.
- Per-slot tags are per-batch-aware: `GestaltMemoryBank` role/persona ids may be a
  python int (A→E: everything SELF) or a `(batch,)` tensor (multi-turn: each
  example's own speaker sequence).

## 5. Latent RAG (`RETRIEVED` role, §Q3)

- Build the model with a 4-entry `role_tags=("USER","SELF","SYSTEM","RETRIEVED")`.
- **`model.inject_source(memory, source_chunks, source_mask)`**: encode a retrieved
  source into per-chunk gestalts tagged RETRIEVED — the loop cross-attends the
  source's *gist* at O(#chunks) instead of O(#tokens) in a context window.
- **`DialogueSession.add_source(text, ground_talker=)`**: serving-time injection;
  `ground_talker` also keeps the raw source as a decode-time Talker grounding
  memory for verbatim fidelity (§4.1 — latents are lossy for exact quotes/numbers).
- **MECHANISM ONLY**: the loop's read of RETRIEVED slots and the Talker grounding
  are untrained until a retrieval-augmented Stage-F dataset exists.

## 6. Data (`dialogue_data.py`)

Socratic / courtroom / debate transcripts are excellent sources: long cross-turn
dependencies stress the gestalt memory, and adversarial assertions are natural
anti-sycophancy material. You choose **who is SELF** (imitate the reasoner vs. an
advocate) via the speaker→role map.

- `parse_transcript` / `transcript_to_turns(text, target_speaker, system_speakers)`
  — `SPEAKER:`-style transcripts → `(role_id, persona_id, text)` turns (target →
  SELF/persona 0; other speakers → distinct personas).
- `messages_to_turns` / `iter_hf_chat_turns` — chat/instruct datasets (messages
  format).
- `iter_hf_transcript_turns` — HF datasets whose text field is a transcript.
- `tensorize_dialogue_sft` — one multi-turn SFT example: prior turns → role+persona
  gestalts in memory, the immediately-preceding turn → the input lane, the SELF
  turn → the target (8-tuple; `collate_dialogue_sft`).
- Offline: `DialogueSFTCorpus`, `ContrastiveCorpus`, `MultiTurnDialogueCorpus`
  (runnable with no downloads).

## 7. Running it (`train_dialogue.py`)

Standalone driver (does NOT touch the A→E `Trainer`). Loads an A→E checkpoint and
fine-tunes with grounded anchor + cosine + generative NLL + anti-sycophancy.

```bash
# offline smoke of the whole path (no ckpt, no downloads):
python train_dialogue.py --offline --preset smoke --steps 20 --multi-turn --persona

# a real fine-tune off the A→E run, with the full tag/RAG stack:
python train_dialogue.py --ckpt runs/scaled/model.pt --multi-turn \
    --soft-tags --content-tags --trust-gate --vector-gate --persona --gestalt-readout --rag
```

Flags: `--multi-turn` (role+persona-tagged aged context), `--soft-tags`,
`--content-tags` (implies soft), `--trust-gate`, `--vector-gate`, `--persona`,
`--gestalt-readout`, `--rag` (adds RETRIEVED; `_reconcile_role_tables` pads a
3-role checkpoint into the 4-role model), `--end-weight` / `--end-grad` (the
learned turn-end, §2.1 — **off by default; a served chatbot needs it**). Loss
weights live in `config.StageFConfig`.

## 8. Evaluation (`lm_eval_adapter.py`)

`LatentThoughtLM` plugs into EleutherAI lm-eval-harness. It CANNOT use the
reconstruction path (it leaks the answer); it scores via the **predictive chain**
(context → loop → `pred_head` → `score_tokens` on the continuation). Single-token
MMLU-style continuations are the degenerate worst case; LAMBADA/cloze map best.
`_score_continuation` is dependency-free and unit-testable without `lm_eval`.

## 9. Honest limits

- **Unvalidated** — smoke-only on synthetic data; no real dialogue run.
- **The turn-end gate is OFF by default** (`end_weight=0`) and, when on, is
  unvalidated on real dialogue — the detached head learns only weakly at smoke scale
  and may need `--end-grad`. See §2.1's honest limits. With it off the model **cannot
  end its own turn** at all; `train_dialogue.py` says so on every run.
- **Behavioral separation (Layer 3) is not yet trained.** The 2026-07-14 review found the
  anti-sycophancy loss routes correctly but does not actually move the trust gate — SGD
  reduces it via the response seed / encoder instead, and the scalar gate is self-defeating
  (discounts topic + polarity together). Treat "the loss drives `trust(USER)` down" (§3) as
  an affordance the current loss does *not* reliably train, not as achieved. Options and a
  recommendation in [`antisycophancy_trust_gate_note.md`](antisycophancy_trust_gate_note.md).
- **RAG is mechanism-only** — needs retrieval-augmented training data; the Talker
  grounding reader is untrained dead weight until then.
- **Real HF loaders are coded, not run** against an actual dataset.
- **Multi-party persona** assumes ≤ `n_personas` distinct speakers per conversation.

**2026-07-14 review (4 adversarial audits) — no target leak, no garbage-training, halt gate
clean.** Fixes landed off the frozen A→E path: Stage-F resume now restores the response
seed/EMA/optimizer (was silently dropped); lm-eval no longer scores a zero-chunk continuation
as max-likelihood; `--soft-tags` now warns that it discards the trained discrete `role_embed`.
Full findings + the deferred low-severity items in `notes.md`.
