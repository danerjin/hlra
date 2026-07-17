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

**Caveat: ACT can make "the same tensor" untrue — but measurement says it currently
does not.** `hrm_loop`'s ACT halt vote is a **batch mean**, so a row's loop depth is
decided by its batchmates. Training runs `B = batch_size`; `reply()` runs `B = 1`. In
principle the gate is then supervised on an `h_t` the server never computes: forced
with a *synthetic discriminative* halting head, row 0's end logit moves 1.36 between
B=8 and B=1, landing on opposite sides of the 0.5 threshold.

**But the halting head measured here does not discriminate.** On `runs/model.pt` — a
*toy smoke* checkpoint, so treat this as indicative, not settled — P(halt) over 64
varied thoughts spans **[0.554, 0.674]** (mean 0.619, std 0.031), the whole range above
0.5. An independent probe hooking the halting head inside the live ACT loop on real
text agrees: P(halt) ∈ **[0.523, 0.994]**, 100% above 0.5. Because `hrm_loop` thresholds
the batch mean at 0.5, a range entirely above 0.5 means every row votes halt and the
batch mean and a per-row vote reach the *same decision* — so the skew is ~zero here.
That is the documented ACT degeneracy ("halting degenerates toward minimum depth")
doing the gate a favour. **At scale, unknown** — both measurements are smoke-scale.
Note also that at serve `B = 1` makes the "batch mean" the row's *own* vote —
**serving already halts per-row; it is TRAINING that lets batchmates decide a row's
depth.**

**So ACT stays ON in Stage F** — `curriculum.py`'s Stage F is `use_act=True`, Stages
D and E consolidated with it, and adaptive depth is one of the architecture's central
claims. Turning it off would run F in a regime the model was never consolidated in.
The default is ACT on. `--no-act` exists only as a **diagnostic** for isolating the
gate from ACT in an A/B, not as a recommendation. The real fix, if the halting head
ever *does* become discriminative at scale, is per-row halting (`experiments.md` #2) —
not deleting adaptive depth. `train_dialogue.py` notes the interaction so it stays
watched.

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
*not* inert (P = 0.018 **per chunk** → 8.7% of 6-chunk replies stop early), so defaulting it on
would truncate replies from any A→E or `end_weight=0` checkpoint.

> ⚠️ **`stage_reached` does NOT identify a Stage-F checkpoint.** `F` is also the A→E
> curriculum's *terminal* stage name, and `trainer.py` saves `curriculum.stage.name` —
> so every **completed** A→E run's `model.pt` is stamped `stage_reached="F"` too
> (`runs/model.pt` is the counterexample sitting in this repo). Only `save()` here
> writes `adapter_state`, so **its presence is the discriminator**. Keying on the stage
> name made the documented A→E→F handoff load the A→E optimizer (model params only)
> into this driver's optimizer (model + adapter) and die with a param-group size
> mismatch — on every finished run. Fixed 2026-07-16; `dialogue_chat.py` had the same
> confusion.

```bash
# ACT stays ON (the curriculum's Stage-F setting; D/E consolidated with it).
python train_dialogue.py --ckpt runs/scaled/model.pt --multi-turn --end-weight 0.5
python train_dialogue.py ... --end-weight 0.5 --end-grad    # the A/B (see limits)
python train_dialogue.py ... --end-weight 0.5 --no-act      # DIAGNOSTIC only: isolates
                                                            # the gate from ACT's depth
```
`save()` records both `end_gate_trained` and `stage_f_use_act`; `chat_core.new_dialogue_session(..., ckpt)`
reads them, and both dialogue front-ends (`dialogue_chat.py`, `web_chat.py`) pass `ckpt`
— so serving runs the loop the way training did instead of guessing. (`chat.py` has no
dialogue path at all; it is the A→E generate/score tester.) Resuming with different
`--end-weight`/`--no-act` flags warns, mirroring `trainer.py`'s schedule/halt guards.

```bash
.venv/bin/python files/dialogue.py    # self-test: RNG trap, labels, end_pos, NULL control
```

**Honest limits — read before trusting it.** Turning this on does *not* mean the gate
works:
- **Unvalidated on real dialogue.** Smoke-scale only.
- **There is NO evidence yet that the gate extracts signal — the plumbing is only
  verified to *run*.** An earlier draft of this section claimed that a smoke overfit
  batch reaching BCE 0.455 against a 0.598 base-rate entropy showed "real signal,"
  because a constant head cannot beat `H(p)`. That inference is **invalid** and the
  claim is withdrawn. With ~14 supervised points in a 192-d latent the head can
  separate *any* labeling: a null run of the identical head on **pure noise features
  with random labels** also beats the base-rate entropy, in **20/20 seeds**, at every
  optimization budget that trains at all. Beating `H(p)` here is simply what a
  signal-free head does. The null's exact BCE is a function of budget, not a fixed
  reference (≈1.14 at 20 steps, ≈0.47 at 700, ≈0.07 at 2000) — and **budget-matched to
  the real run it lands on ≈0.47, i.e. right on top of the 0.455 being reported**. So
  the measurement is indistinguishable from noise, and says nothing about signal in
  either direction. A real answer needs a **held-out split**, which has not been run.
- **The detached head may not be able to learn this at all.** `end_grad=False` lets it
  read only what `cos`/`gen` already put in `h_t`, and nothing forces them to encode "I
  am done" — the same shape as the halt gate degenerating to a constant P(halt) ≈ 0.95.
  If the gate will not move on real data, **try `--end-grad` first**.
- **`end_acc` is imbalance-blind** — one "end" per turn, so an always-continue head
  scores ≈ 1 − 1/M (0.714 on the smoke batch). Never read alone. And see the `end_pos`
  warning above: at `end_pos = 0`, `end`/`end_acc` look *perfect* and mean nothing.
- **An untrained gate is NOT inert at serve time.** sigmoid(−4.0) = 0.018 is a
  *per-chunk* rate, and a 6-chunk reply has 5 chances to stop early — 8.7% of them
  would (18.1% for 12 chunks), at random. Serving therefore keeps the gate **off** unless the checkpoint's
  `end_gate_trained` flag says it was really trained (`DialogueSession(use_end_head=)`,
  set by `chat_core.new_dialogue_session`). Off = exactly the pre-gate behavior.

## 2.2 Phantom memory slots (`GestaltMemoryBank.write(valid=)`)

Found by the 2026-07-16 cross-review, **pre-existing and independent of the turn-end
gate** — the gate merely exposed it.

**The bug.** A `memory.write` puts ONE slot into the bank for the WHOLE batch (the bank
is `(batch, n_slots, d)`). `_write_context` looped per context position guarded by a
**batch-level `.any()`**, so one row having context at position *j* forced a slot onto
*every* row. Rows without it got `_encode_real_rows`' **exact-zero latent**, tagged with
a real role — and `context_roles` defaults to `torch.zeros`, i.e. **`USER`**. That slot
is not inert: the reader computes `kv = stacked + tags`, so a zero vector plus the USER
tag is a fully attendable *"the user said ⟨nothing⟩"* memory.

Note what it violated: `_encode_real_rows`' own docstring promises *"pad-row latents
feed only dead paths"*. True for A→E — and false the moment they are written to memory.

**Measured on the real corpus** (ultrachat — `training_easy.md`'s recommended Stage-F
data, not a code default; `--hf-chat` defaults to the offline synthetic corpus — via
`iter_hf_chat_turns`, preset `small`, B=8): **~45%** of every row's context memory was fabricated, and **~28% of rows
had zero real context — 100% phantom**. (Two independent draws: 45.7/28.0 and
45.4/27.5 — a sampling distribution over batches, so read them as ~45/~28, not to
three significant figures.) It made a row's `h_t` a function of its
batchmates' context *length* — so it corrupted the `h_t` that `cos`/`gen` are computed
from, not just the gate's input (the *sign* of the effect on those losses at random
init is not measured, and inferring one would be the same mistake as §2.1's withdrawn
signal claim). It was **not** mitigated by `--no-act` — the ACT halt-vote skew documented in §2.1 is a
*different*, and measurably benign, coupling. Serving (B=1) never reproduced it.

**The fix.** `GestaltMemoryBank.write(..., valid=)` takes an optional `(batch,)` bool
marking which rows the slot is real for; `valid_mask()` returns `(batch, n_slots)`, or
**`None` when no slot carries one** — in which case the reader takes its original
unmasked attention, byte-identical. That is the whole A→E path and every B=1 serve, so
**A→E is untouched** (verified: losses + every per-parameter grad norm, float64).
Three Stage-F writers pass their real per-row mask, but **only one was a live bug**
— the other two are defensive applications of the same pattern, and saying otherwise
overstates the fix:
- `_write_context` — **real, measured** (the 45.7% above).
- `inject_source` — **unreachable**: its only caller is `DialogueSession.add_source`
  (B=1 serving), where the column is always all-True. Marked so a future *batched*
  caller cannot reintroduce it.
- `forward_dialogue`'s SELF write — **semantically inert** (its junk slots are per-row,
  feed no loss, and `resp_mask`'s left-packing means a row never reactivates; verified
  0.0e+00 on ragged-context batches). Removing it is *not universally* bit-exact: where
  it is the bank's only validity source (every `_write_context` column all-True, which
  now collapses to `None`), the reader falls back to its unmasked branch and outputs
  move ~5e-7 — float32 noise, the same floor check [5] measures. `resp_mask` is a left-packed
  contiguous prefix, so a row never reactivates, contributes to no loss after its last
  chunk, and its junk slots are per-row — unobservable. Kept so the invariant holds by
  construction rather than by luck of left-packing. (An earlier draft justified it as
  "an inactive row keeps a stale `h`". That is **false** — `active_mask` gates only the
  ponder cost and halt vote, so such rows *keep evolving on pad-chunk latents*
  (`hrm_loop.py:320`) and write fresh garbage, not a stale duplicate. The claim
  contradicted the file it named.)

One trap worth knowing: attention over a **fully-masked row is NaN, not zero**, and 28%
of real rows have no context at all. Those rows are left unmasked and their output
zeroed explicitly — matching the reader's existing "no memory yet" branch.

`python files/dialogue.py` check [5] guards this: row 0's own P(end) must not move when
a batchmate's context length changes. It is verified *sensitive* — stubbing `valid_mask`
to `None` makes it fail (drift 6.6e-2 vs 4.8e-7 of float32 batch-shape noise). **It
guards `_write_context` only**: reverting either of the other two `valid=` arguments
leaves the suite green, because both are inert (below). Check [6] guards the round-3
fixes (persona, the all-True collapse, the non-tensor guard), which shipped with no
coverage at all — each is verified to turn the suite red when reverted.

**Two latent hazards this does NOT close**, both measured, neither reachable today:
- **FIFO eviction is still batch-coupled.** `valid` marks a slot dead; it does not
  protect it from `write`'s `pop(0)`. If `memory_capacity` were below the slots written
  per example, a long batchmate's writes would evict a *short* row's real context — the
  same coupling at the same magnitude (1.43 vs the unfixed 1.39). The real headroom is
  **2×, not the 4–8× the capacity ratio suggests**: `forward_dialogue` writes context
  **plus** SELF, up to `2 × max_chunks_per_doc` slots per example — so `small` (the
  recommended preset) has exactly 2.0× margin. **Now enforced**: `train_dialogue` refuses
  to start when `memory_capacity < 2 × max_chunks_per_doc`, so this is a checked
  precondition rather than a config invariant nothing verified.
- **`filtered_stacked` cannot express validity.** Its return has no per-row mask, so a
  masked slot would be handed back intact for every row. Its only callers are A→E (whose
  banks never carry validity) — **now enforced**: it raises `NotImplementedError` on a
  bank with masked slots rather than silently leaking, so a future Stage-F caller hits
  an error instead of the bug class.

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
- **Cross-turn memory has a train/serve granularity mismatch** (pre-existing, not
  fixed). Training's `_write_context` writes **one slot per chunk** of each prior turn,
  each with its own role/persona; serving's `_age_user_turn` writes **one mean-pooled
  slot per user turn**. A checkpoint trained against per-chunk USER gestalts is served
  against pooled ones — real in **every** config. Under `--gestalt-readout`
  (TRAINING.md's recommended command) it is worse: serving pools *before* `_gestalt`,
  which is then a `Linear`+`hard_normalize` and does **not** commute with the mean
  (measured 1.13 vs 0.0e+00). With the readout **off — the default —** `_gestalt` is the
  identity and commutes exactly, so the mismatch there is purely one of granularity.
  Same class as the bugs §2.2 fixed; left as a design decision (per-chunk aging would
  grow the bank fast) rather than patched blind.
- **The turn-end gate is OFF by default** (`end_weight=0`) and, when on, is
  unvalidated: there is **no evidence it extracts signal at all** (§2.1 — the earlier
  "learns only weakly" reading was withdrawn as an invalid inference). `--end-grad` is
  the first thing to try. See §2.1's honest limits. With it off the model **cannot
  end its own turn** at all; `train_dialogue.py` says so on every run.
- **Behavioral separation (Layer 3) is not yet trained.** The 2026-07-14 review found the
  anti-sycophancy loss routes correctly but does not actually move the trust gate — SGD
  reduces it via the response seed / encoder instead, and the scalar gate is self-defeating
  (discounts topic + polarity together). Treat "the loss drives `trust(USER)` down" (§3) as
  an affordance the current loss does *not* reliably train, not as achieved. Options and a
  recommendation in [`antisycophancy_trust_gate_note.md`](antisycophancy_trust_gate_note.md).
- **RAG is mechanism-only** — needs retrieval-augmented training data; the Talker
  grounding reader is untrained dead weight until then.
- **Real HF loaders are coded but barely exercised.** The ultrachat stream has been
  read for *measurement* (§2.2's context-length statistics; `TRAINING.md` reports it
  100% multi-turn) — but never *trained* on.
- **Multi-party persona** assumes ≤ `n_personas` distinct speakers per conversation —
  and *serving cannot express a third speaker at all*: `DialogueSession` tags every
  user turn persona 1 (the two-party case), so a multi-party conversation trained
  with distinct speaker ids is served as if it had one user.

**2026-07-14 review (4 adversarial audits) — no target leak, no garbage-training, halt gate
clean.** Fixes landed off the frozen A→E path: Stage-F resume now restores the response
seed/EMA/optimizer (was silently dropped); lm-eval no longer scores a zero-chunk continuation
as max-likelihood; `--soft-tags` now warns that it discards the trained discrete `role_embed`.
Full findings + the deferred low-severity items in `notes.md`.
