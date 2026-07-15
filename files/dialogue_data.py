"""
dialogue_data.py
================
Stage F (chatbot fine-tuning, §4) data pipeline. Turns dialogue into the tensors
`model.forward_dialogue` / `model.forward_anti_sycophancy` consume, and ships
offline synthetic corpora so the whole Stage-F path is runnable with no
downloads (as the A-E offline path is).

The load-bearing property here is the Layer-2 (§4) SEPARATION CONTRACT, enforced
by construction in `tensorize_sft`:

    the user turn's tokens go ONLY into the input-lane window (user_ids);
    the assistant turn's tokens go ONLY into the prediction target (resp_chunks).

They are tokenized from disjoint strings, so the assistant response can never
appear in the lane the loop cross-attends -- the SSL-target leak that the notes
flag as "fix before training Stage F on generic documents" simply cannot arise
on this dialogue path. (It arose only in the A-E pretraining lane, whose raw
window was a slice of the SAME document it was predicting.)

Example tensor shapes (per example; collate stacks a leading batch dim):
  SFT:          resp_chunks (M, L) long, resp_mask (M,) bool,
                user_ids (W,) long,      user_mask (W,) bool
  contrastive:  user_ids_a/b (W,) long,  user_mask_a/b (W,) bool,
                answer_chunks (M, L) long, answer_mask (M,) bool
where M = max_chunks_per_doc, L = max_chunk_len, W = recent_token_window.
"""
from __future__ import annotations

import json
import random
import re
from typing import Callable, Iterable, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import IterableDataset

from data import PAD, USER, SELF, SYSTEM


# ======================================================================
# Tensorization (the separation contract lives in tensorize_sft)
# ======================================================================
def _user_window(chunker, cfg, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """The user turn as the input-lane raw window: trailing recent_token_window
    token ids, left-packed with a validity mask. Never contains assistant
    tokens (see module docstring)."""
    W = cfg.recent_token_window
    ids = chunker.tokenizer.encode(text)
    tail = ids[-W:] if len(ids) > W else ids
    raw = torch.full((W,), PAD, dtype=torch.long)
    mask = torch.zeros(W, dtype=torch.bool)
    if tail:
        t = torch.tensor(tail, dtype=torch.long)
        raw[: t.numel()] = t
        mask[: t.numel()] = True
    return raw, mask


def tensorize_sft(user_text: str, assistant_text: str, chunker, cfg):
    """(user turn, assistant turn) -> (resp_chunks, resp_mask, user_ids, user_mask).
    The assistant turn is chunked (the prediction target); the user turn becomes
    the input-lane window. Disjoint strings => leak-free by construction."""
    ct, cm = chunker.chunk_batch([assistant_text])       # (1, M, L), (1, M)
    resp_chunks, resp_mask = ct[0], cm[0]
    user_ids, user_mask = _user_window(chunker, cfg, user_text)
    return resp_chunks, resp_mask, user_ids, user_mask


def tensorize_contrastive(user_true: str, user_false: str, answer_text: str, chunker, cfg):
    """Two premises differing only in the asserted stance, plus the (shared,
    role-invariant) correct answer. The premise is CHUNKED (not a lane window):
    at train time model.forward_anti_sycophancy compresses it into a USER gestalt
    written to memory, so the trust gate on the memory reader -- not the input
    lane -- is what must learn to discount it. Returns
    (premise_a_chunks, premise_a_mask, premise_b_chunks, premise_b_mask,
     answer_chunks, answer_mask)."""
    ca, ma = chunker.chunk_batch([user_true])
    cb, mb = chunker.chunk_batch([user_false])
    cans, mans = chunker.chunk_batch([answer_text])
    return ca[0], ma[0], cb[0], mb[0], cans[0], mans[0]


def collate_sft(batch):
    return (torch.stack([b[0] for b in batch], 0),   # resp_chunks (B, M, L)
            torch.stack([b[1] for b in batch], 0),   # resp_mask   (B, M)
            torch.stack([b[2] for b in batch], 0),   # user_ids    (B, W)
            torch.stack([b[3] for b in batch], 0))   # user_mask   (B, W)


def collate_contrastive(batch):
    return tuple(torch.stack([b[i] for b in batch], 0) for i in range(6))


# ======================================================================
# Offline synthetic corpora (runnable with no downloads)
# ======================================================================
_TOPICS = ["gardening", "algebra", "sailing", "pottery", "astronomy", "cooking",
           "chess", "geology", "cycling", "origami", "birdsong", "weaving"]


def _fake_sentence(rng: random.Random, n_words: int) -> str:
    words = [f"word{rng.randint(0, 399)}" for _ in range(n_words)]
    return " ".join(words) + rng.choice([".", "?", "!"])


def _fake_paragraph(rng: random.Random, min_s: int, max_s: int) -> str:
    return " ".join(_fake_sentence(rng, rng.randint(5, 14))
                    for _ in range(rng.randint(min_s, max_s)))


class DialogueSFTCorpus:
    """Offline (user turn, assistant turn) pairs. Assistant turns span several
    sentences so they chunk into multiple thoughts and exercise the sequential
    loop + gestalt memory. Real SFT data (ideally with the §4.3 agree/disagree
    contrast) is pluggable via any iterator of (user, assistant) strings."""

    def __init__(self, n: int, seed: int = 0):
        self.n = n
        self.seed = seed

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        rng = random.Random(self.seed)
        for _ in range(self.n):
            user = _fake_paragraph(rng, 1, 2)
            assistant = _fake_paragraph(rng, 3, 7)
            yield user, assistant


class ContrastiveCorpus:
    """Offline anti-sycophancy pairs (§4.3): two user turns asserting opposite
    stances on a topic, with a fixed correct answer independent of the user's
    assertion. Structure over semantics (smoke): the point is that the ONLY
    difference between the two inputs is the user-asserted premise."""

    def __init__(self, n: int, seed: int = 0):
        self.n = n
        self.seed = seed

    def __iter__(self) -> Iterator[Tuple[str, str, str]]:
        rng = random.Random(self.seed + 1)
        for _ in range(self.n):
            topic = rng.choice(_TOPICS)
            ctx = _fake_sentence(rng, rng.randint(6, 12))
            user_true = f"{ctx} I am sure {topic} is clearly the best. Do you agree?"
            user_false = f"{ctx} I am sure {topic} is clearly the worst. Do you agree?"
            # Fixed, role-invariant answer: the model must not flip with the user.
            answer = (f"The evidence about {topic} is mixed and depends on the goal. "
                      f"{_fake_paragraph(rng, 2, 4)}")
            yield user_true, user_false, answer


# ======================================================================
# Datasets
# ======================================================================
class DialogueSFTDataset(IterableDataset):
    """Streams (user, assistant) strings from `pair_factory` (a zero-arg callable
    returning a fresh iterator) and tensorizes each. Drops examples whose
    assistant turn produced no chunks."""

    def __init__(self, pair_factory, chunker, cfg, max_examples: int = None):
        self.pair_factory = pair_factory
        self.chunker = chunker
        self.cfg = cfg
        self.max_examples = max_examples

    def __iter__(self):
        emitted = 0
        for user, assistant in self.pair_factory():
            ex = tensorize_sft(user, assistant, self.chunker, self.cfg)
            if int(ex[1].sum()) == 0:
                continue
            yield ex
            emitted += 1
            if self.max_examples is not None and emitted >= self.max_examples:
                return


class ContrastiveDataset(IterableDataset):
    """Streams (user_true, user_false, answer) triples and tensorizes each."""

    def __init__(self, triple_factory, chunker, cfg, max_examples: int = None):
        self.triple_factory = triple_factory
        self.chunker = chunker
        self.cfg = cfg
        self.max_examples = max_examples

    def __iter__(self):
        emitted = 0
        for ut, uf, ans in self.triple_factory():
            ex = tensorize_contrastive(ut, uf, ans, self.chunker, self.cfg)
            if int(ex[5].sum()) == 0:      # answer produced no chunks
                continue
            yield ex
            emitted += 1
            if self.max_examples is not None and emitted >= self.max_examples:
                return


# ======================================================================
# REAL / multi-turn dialogue (transcripts: chat, debate, courtroom, socratic)
# ----------------------------------------------------------------------
# A dialogue is a list of (role_id, text) turns. SFT imitates ONE speaker
# (mapped to SELF); to predict a SELF turn k we feed:
#   * turns[k-1]        -> the input LANE (raw tokens, current input)
#   * turns[:k-1]       -> role-tagged aged gestalts in MEMORY (cross-turn context)
#   * turns[k]          -> the SELF response target
# This is what exercises cross-turn dependency -- exactly why socratic/debate/
# courtroom transcripts (long-range references, adversarial assertions) are good
# sources. You choose WHO is SELF (the reasoner? an advocate?) via the role map.
# ======================================================================
def speaker_role(speaker: str, target_speaker: str, system_speakers: Tuple[str, ...] = ()) -> int:
    """Map a transcript speaker name to a role id: the target -> SELF (the voice
    we imitate), designated narrators/moderators/judges -> SYSTEM, everyone else
    -> USER. Case-insensitive substring match so 'THE COURT'/'JUDGE' etc. group."""
    s = speaker.strip().lower()
    if s == target_speaker.strip().lower():
        return SELF
    if any(k.lower() in s for k in system_speakers):
        return SYSTEM
    return USER


def tensorize_dialogue_sft(turns: List[Tuple[int, int, str]], target_idx: int, chunker, cfg,
                           context_cap: Optional[int] = None):
    """One multi-turn SFT example predicting turns[target_idx] (a SELF turn).
    A turn is (role_id, persona_id, text). Returns the 8-tuple (context_chunks,
    context_mask, context_roles, context_personas, user_ids, user_mask,
    resp_chunks, resp_mask). Context keeps the trailing `context_cap` (default
    max_chunks_per_doc) chunks of prior turns, each tagged by its speaker role AND
    conversation-local persona (bounded recall = the gestalt FIFO)."""
    A = cfg.max_chunks_per_doc if context_cap is None else context_cap
    L = cfg.max_chunk_len
    rc, rm = chunker.chunk_batch([turns[target_idx][2]])
    resp_chunks, resp_mask = rc[0], rm[0]
    user_text = turns[target_idx - 1][2] if target_idx - 1 >= 0 else ""
    user_ids, user_mask = _user_window(chunker, cfg, user_text)

    ctx_rows, ctx_roles, ctx_personas = [], [], []
    for role, persona, text in turns[: max(0, target_idx - 1)]:
        cc, cm = chunker.chunk_batch([text])
        for row in cc[0][cm[0]]:                 # each kept chunk of this prior turn
            ctx_rows.append(row)
            ctx_roles.append(role)
            ctx_personas.append(persona)
    ctx_rows = ctx_rows[-A:]; ctx_roles = ctx_roles[-A:]; ctx_personas = ctx_personas[-A:]

    context_chunks = torch.zeros(A, L, dtype=torch.long)
    context_mask = torch.zeros(A, dtype=torch.bool)
    context_roles = torch.zeros(A, dtype=torch.long)
    context_personas = torch.zeros(A, dtype=torch.long)
    for i, (row, r, p) in enumerate(zip(ctx_rows, ctx_roles, ctx_personas)):
        context_chunks[i] = row
        context_mask[i] = True
        context_roles[i] = r
        # Clamp the conversation-local speaker id into the persona table
        # (transcripts with more distinct speakers than n_personas collapse the
        # overflow onto the last slot, rather than index-erroring the embedding).
        context_personas[i] = min(int(p), cfg.n_personas - 1)
    return (context_chunks, context_mask, context_roles, context_personas,
            user_ids, user_mask, resp_chunks, resp_mask)


def collate_dialogue_sft(batch):
    return tuple(torch.stack([b[i] for b in batch], 0) for i in range(len(batch[0])))


_SPEAKER_LINE = re.compile(r"^\s*([A-Z][A-Za-z0-9 .'\-]{0,40}?):\s+", re.MULTILINE)


def parse_transcript(text: str) -> List[Tuple[str, str]]:
    """Split a 'SPEAKER: utterance' transcript into (speaker, utterance) turns.
    Handles debate/courtroom/socratic transcripts and play/screenplay dialogue.
    Consecutive lines without a new speaker label attach to the current turn.
    Returns [] if no speaker labels are found (caller can fall back).

    LIMITATION: the label heuristic (a capitalized token at line start followed
    by ': ') can false-positive on prose lines like 'Note: ...' or 'Q: ...',
    fragmenting an utterance and inventing a speaker. This affects turn-boundary
    quality on messy sources, never crashes; pre-clean or use a dataset with
    clean speaker labels for production."""
    marks = list(_SPEAKER_LINE.finditer(text))
    if not marks:
        return []
    turns = []
    for i, m in enumerate(marks):
        speaker = m.group(1).strip()
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        utt = text[start:end].strip()
        if utt:
            turns.append((speaker, utt))
    return turns


def transcript_to_turns(text: str, target_speaker: str,
                        system_speakers: Tuple[str, ...] = ()) -> List[Tuple[int, int, str]]:
    """Parse a transcript into (role_id, persona_id, text) turns. Roles map the
    target speaker -> SELF, system speakers -> SYSTEM, else USER. Personas are
    conversation-local speaker ids: the SELF speaker is persona 0 (matching
    forward_dialogue's self_persona), every other distinct speaker gets 1, 2, ...
    by first appearance -- so >3 speakers are individually distinguishable."""
    persona_map = {}

    def persona_of(sp: str) -> int:
        if speaker_role(sp, target_speaker, system_speakers) == SELF:
            return 0
        key = sp.strip().lower()
        if key not in persona_map:
            persona_map[key] = len(persona_map) + 1
        return persona_map[key]

    return [(speaker_role(sp, target_speaker, system_speakers), persona_of(sp), utt)
            for sp, utt in parse_transcript(text)]


# Role-name -> (role_id, persona_id). Covers the two common chat schemas: OpenAI/HF
# `role` (assistant/user/system) and ShareGPT `from` (gpt/human/system, plus bot/ai).
_ROLE_MAP = {
    "assistant": (SELF, 0), "gpt": (SELF, 0), "bot": (SELF, 0), "ai": (SELF, 0), "model": (SELF, 0),
    "system": (SYSTEM, 2), "instruction": (SYSTEM, 2),
    "user": (USER, 1), "human": (USER, 1),
}


def _coerce_messages(msgs) -> Optional[List[dict]]:
    """Normalize a dataset's `messages` cell to a list[dict], tolerating the JSON
    formatting real datasets ship. `no_robots`/`ultrachat` give a native list, but
    depending on the `datasets`/pyarrow version (or a plain-string column) the same
    field can arrive as a JSON STRING -- in which case the old `isinstance(list)`
    gate silently dropped every example. Handles: native list; JSON-string of a list;
    JSON-string of a {"messages":[...]} / {"conversations":[...]} wrapper."""
    if isinstance(msgs, list):
        return msgs
    if isinstance(msgs, str):
        try:
            parsed = json.loads(msgs)
        except (ValueError, TypeError):
            return None
        if isinstance(parsed, dict):
            parsed = parsed.get("messages") or parsed.get("conversations")
        return parsed if isinstance(parsed, list) else None
    return None


def messages_to_turns(messages) -> List[Tuple[int, int, str]]:
    """Map a chat 'messages' list ([{role, content}, ...]) to (role_id, persona_id,
    text) turns. Accepts either the native list OR a JSON-string of one (see
    `_coerce_messages`), and both the `role`/`content` and ShareGPT `from`/`value`
    key schemes. Unknown roles fall back to USER/persona 1."""
    msgs = _coerce_messages(messages)
    if not msgs:
        return []
    out = []
    for msg in msgs:
        if not isinstance(msg, dict):
            continue
        r = str(msg.get("role", msg.get("from", ""))).lower().strip()
        role, persona = _ROLE_MAP.get(r, (USER, 1))
        text = msg.get("content", msg.get("value", "")) or ""
        if not isinstance(text, str):
            text = str(text)
        if text.strip():
            out.append((role, persona, text))
    return out


def iter_hf_chat_turns(hf_id: str, split: str = "train", name: Optional[str] = None,
                       messages_field: str = "messages", streaming: bool = True,
                       max_docs: Optional[int] = None) -> Iterator[List[Tuple[int, int, str]]]:
    """Stream a chat/instruct dataset that stores a per-example list of messages
    (role/content, or ShareGPT from/value; native list or JSON string) and yield
    (role_id, persona_id, text) turn lists. Lazy-imports `datasets`."""
    from datasets import load_dataset
    ds = load_dataset(hf_id, name, split=split, streaming=streaming)
    for i, ex in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            return
        # try the named field, then fall back to the two common alternates so a
        # ShareGPT-style dataset works without a flag.
        msgs = ex.get(messages_field)
        if msgs is None:
            msgs = ex.get("messages") or ex.get("conversations")
        turns = messages_to_turns(msgs)
        if turns and any(r == SELF for r, _, _ in turns):
            yield turns


def iter_hf_transcript_turns(hf_id: str, text_field: str, target_speaker: str,
                             system_speakers: Tuple[str, ...] = (), split: str = "train",
                             name: Optional[str] = None, streaming: bool = True,
                             max_docs: Optional[int] = None) -> Iterator[List[Tuple[int, int, str]]]:
    """Stream a dataset whose `text_field` holds a 'SPEAKER: ...' transcript
    (debate/courtroom/socratic) and yield role-mapped (role, persona, text) turn
    lists. Skips docs with no parseable turns for the target speaker."""
    from datasets import load_dataset
    ds = load_dataset(hf_id, name, split=split, streaming=streaming)
    for i, ex in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            return
        text = ex.get(text_field) or ""
        turns = transcript_to_turns(text, target_speaker, system_speakers)
        if turns and any(r == SELF for r, _, _ in turns):
            yield turns


class DialogueTurnsDataset(IterableDataset):
    """Streams (role_id, text) turn-lists from `turns_factory` (any of the
    iter_hf_* above, or a list) and emits one multi-turn SFT example per SELF turn
    that has at least one preceding turn. This is the real-data loader: point it
    at a chat dataset (iter_hf_chat_turns) or transcripts (iter_hf_transcript_turns)."""

    def __init__(self, turns_factory: Callable[[], Iterable], chunker, cfg,
                 min_context_turns: int = 1, max_examples: Optional[int] = None):
        self.turns_factory = turns_factory
        self.chunker = chunker
        self.cfg = cfg
        self.min_context_turns = min_context_turns
        self.max_examples = max_examples

    def __iter__(self):
        emitted = 0
        for turns in self.turns_factory():
            for k in range(len(turns)):
                if turns[k][0] != SELF or k < self.min_context_turns:
                    continue
                ex = tensorize_dialogue_sft(turns, k, self.chunker, self.cfg)
                if int(ex[7].sum()) == 0:        # empty response (resp_mask is index 7)
                    continue
                yield ex
                emitted += 1
                if self.max_examples is not None and emitted >= self.max_examples:
                    return


class MultiTurnDialogueCorpus:
    """Offline multi-turn dialogues (alternating USER/SELF) for smoking the
    multi-turn / cross-turn-context path with no downloads."""

    def __init__(self, n: int, turns: int = 6, seed: int = 0):
        self.n, self.turns, self.seed = n, turns, seed

    def __iter__(self) -> Iterator[List[Tuple[int, int, str]]]:
        rng = random.Random(self.seed + 7)
        for _ in range(self.n):
            # (role, persona, text): USER=persona 1, SELF=persona 0 (self).
            yield [((USER, 1, _fake_paragraph(rng, 1, 3)) if t % 2 == 0
                    else (SELF, 0, _fake_paragraph(rng, 1, 3)))
                   for t in range(self.turns)]
