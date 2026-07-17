# Stage F — Chatbot fine-tuning, tagging, RAG (design + implementation)

Stage F turns the A→E foundation model into a **chatbot**. Everything here is
**additive and opt-in**: with all Stage-F flags off, the model is byte-identical
to the validated A→E model (the `latent_mult=1`-style discipline), and no A→E
code path (`forward_grounded` / `forward_self_supervised` / `trainer.py`) is
touched. **Status: implemented and smoke-verified on offline synthetic data;
NOT trained, NOT validated, uncommitted-review-only.** Nothing here has seen a
real run.

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

`h_t` is **the same tensor** `reply()` holds when it must decide whether to emit
another chunk, read through **the same head** — so train and serve agree by
construction rather than by convention.

**The label is free, and was being discarded.** It is already in the SFT batch:
`resp_mask[:, t+1]` says whether a chunk *t+1* exists. So there is **no data-format
change** — `tensorize_sft` / `collate_sft` are untouched.

**The truncation confound (`model._turn_end_labels`).** The free label is not
unconditionally correct. `chunker.chunk_batch` caps a response at
`M = max_chunks_per_doc`, so a row that fills all *M* slots either ended exactly at
*M* or was cut off at *M* — indistinguishable. Its final "the turn ends here" label
is therefore **unknown, not True**; trained on naively it teaches *every turn ends
after exactly M chunks*. Only that one ambiguous label per filled row is masked out;
every earlier "a chunk follows" label is correct either way and is kept. `end_n` in
the log is how many labels survived, so an over-masked batch is visible.

**Where it lives.** `end_head` is in `dialogue.DialogueAdapter`, next to
`response_seed` and for the same reason: the A→E `state_dict` stays **byte-identical**,
so an A→E checkpoint still loads strict. Verified: `forward_grounded` /
`forward_self_supervised` are bit-for-bit unchanged (losses + every per-parameter
grad norm, float64), and `end_weight=0` reproduces pre-change Stage F bit-for-bit.

**Gradient routing.** `end_grad=False` (default) reads a **detached** `h_t`, so the
BCE trains only the head and never reshapes the reasoning — the
`forward_self_supervised_halt` convention. Verified: with it off the loop and encoder
receive **exactly zero** gradient from this term; with `--end-grad` it reaches both.

**Serving.** `max_chunks` becomes a *cap*, not the reply length. An untrained head is
a constant P(end) ≈ 0.018 (bias init −4.0), so it effectively never fires and an
untrained adapter degrades to the old fixed-length behavior;
`reply(use_end_head=False)` restores it exactly.

```bash
python train_dialogue.py --ckpt runs/scaled/model.pt --multi-turn --end-weight 0.5
python train_dialogue.py ... --end-weight 0.5 --end-grad      # the A/B (see limits)
```

**Honest limits — read before trusting it.** Turning this on does *not* mean the gate
works:
- **Unvalidated on real dialogue.** Smoke-scale only.
- **The detached head learns only weakly.** On a smoke overfit batch where the end is
  *deliberately inferable* from content, the BCE beats the base-rate entropy (0.455 vs
  0.598 — a constant head cannot), so the plumbing extracts real signal. But the margin
  is small and `end_acc` never left the base rate. **This is the same failure shape as
  the halt gate** (which degenerated to a constant P(halt) ≈ 0.95): a detached head can
  only read what `cos`/`gen` already put in `h_t`, and nothing forces them to encode
  "I am done". If the gate will not move on real data, **try `--end-grad` first**.
- **`end_acc` is imbalance-blind** — one "end" per turn, so an always-continue head
  scores ≈ 1 − 1/M (0.714 on the smoke batch). Read it *with* the BCE, never alone.
  This is the `antisycophancy_trust_gate_note.md` #1 lesson applied up front: the
  metric exists so "is it working?" is observable rather than hoped.

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
