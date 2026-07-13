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
| **2. Informational** | the target must not be visible while predicted | data contract: user turn → lane, assistant turn → target, disjoint strings (`dialogue_data.tensorize_*`) |
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
3-role checkpoint into the 4-role model). Loss weights live in
`config.StageFConfig`.

## 8. Evaluation (`lm_eval_adapter.py`)

`LatentThoughtLM` plugs into EleutherAI lm-eval-harness. It CANNOT use the
reconstruction path (it leaks the answer); it scores via the **predictive chain**
(context → loop → `pred_head` → `score_tokens` on the continuation). Single-token
MMLU-style continuations are the degenerate worst case; LAMBADA/cloze map best.
`_score_continuation` is dependency-free and unit-testable without `lm_eval`.

## 9. Honest limits

- **Unvalidated** — smoke-only on synthetic data; no real dialogue run.
- **RAG is mechanism-only** — needs retrieval-augmented training data; the Talker
  grounding reader is untrained dead weight until then.
- **Scalar trust gate discounts a whole slot** (topic + polarity) — the vector gate
  is the finer tool, still unproven.
- **Real HF loaders are coded, not run** against an actual dataset.
- **Multi-party persona** assumes ≤ `n_personas` distinct speakers per conversation.
