"""
chunker.py
==========
Chunk-boundary policy (§3.1, §5.1). A "thought" is meant to be a
semantically complete unit (sentence/clause), not a fixed-length token
window -- bisecting a clause would force one thought to encode half a
proposition, reintroducing the compounding-fragility problem chunking was
supposed to remove.

Two chunkers are provided:

1. `SegmentAnyTextChunker` -- the *real* method. Thought Gestalt (the
   source paper behind the gestalt-memory mechanism this architecture
   reuses) segments its training corpora with **SaT Capped**: sentence
   boundaries predicted by SaT (Segment Any Text; Frohmann et al., 2024,
   "Segment any text: A universal approach for robust, efficient and
   adaptable sentence segmentation"), with a punctuation-aware fallback
   that caps any sentence longer than a max token length into shorter,
   semantically coherent spans (the paper uses L=64 tokens; this class
   takes that as `max_chunk_len`). This is the chunker to use whenever the
   input is real text and a tokenizer is available.

2. `TokenIdBoundaryChunker` -- a trivial legacy fallback that splits
   already-tokenized synthetic sequences on a fixed boundary-token id. It
   exists only so the synthetic demo corpus in `data.py` (which has no
   real text, just integers) remains runnable end-to-end without a real
   tokenizer or the SaT model download. Real training runs should use
   `SegmentAnyTextChunker` instead.

Both expose the same `chunk_batch(...) -> (chunk_tensor, chunk_mask)`
interface so `model.py` doesn't care which one it's holding.
"""
from __future__ import annotations

from typing import List, Protocol, Sequence

import torch


# Chunking-algorithm version. BUMP THIS whenever the chunk-boundary policy
# changes in a way that makes an existing cache stale (different chunk contents
# for the same input). It is stamped into every cache manifest by data_prep.py
# and hard-checked by data.CachedChunkDataset on load, so a stale cache can no
# longer masquerade as fresh (its manifest dims are identical to a new cache's --
# see notes.md, the fifth pre-flight review). History:
#   1  original _cap_span (split-and-recurse; emitted 1-word fragments, dropped
#      delimiters) -- pre-2026-07-10.
#   2  2026-07-10 _cap_span rewrite (delimiter-preserving greedy re-pack).
#   3  2026-07-11 splitter-fragment merge (min_chunk_tokens) + character-boundary
#      hard fallback (verbatim, cap-exact, no U+FFFD). This is the current method;
#      every big-run cache must be prepped at version >= 3.
CHUNKER_VERSION = 3


# ----------------------------------------------------------------------
# Real method: SaT Capped (Thought Gestalt's actual preprocessing)
# ----------------------------------------------------------------------
class _Tokenizer(Protocol):
    """Minimal duck-typed interface this module needs from a tokenizer
    (e.g. a HuggingFace `PreTrainedTokenizer`), so `SegmentAnyTextChunker`
    can be unit-tested with a stub instead of a real network-downloaded
    tokenizer."""

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]: ...
    def decode(self, ids: Sequence[int]) -> str: ...


class _Segmenter(Protocol):
    """Minimal duck-typed interface for a SaT-like sentence segmenter, so
    tests don't need to download the real SaT model over the network."""

    def split(self, text_or_texts, **kwargs): ...


class SegmentAnyTextChunker:
    """
    Implements "SaT Capped" exactly as described in the Thought Gestalt
    paper's preprocessing section: split text into sentences with SaT, then
    apply a punctuation-aware fallback that recursively splits any sentence
    whose token length exceeds `max_chunk_len` into shorter coherent spans,
    falling back to a hard token-count split only if no punctuation is
    available to split on.

    `sat_model` and `tokenizer` are injected rather than constructed here,
    so this class has no hidden network dependency and is directly
    testable. In real use:

        from wtpsplit import SaT
        from transformers import AutoTokenizer
        chunker = SegmentAnyTextChunker(
            sat_model=SaT("sat-3l-sm"),          # downloads from HF hub
            tokenizer=AutoTokenizer.from_pretrained("your-model-tokenizer"),
            max_chunk_len=64,                     # L=64, matching the paper
            max_chunks_per_doc=32,
        )
    """

    # Fallback split points, tried in priority order -- coarser punctuation
    # (sentence-internal clause breaks) before finer (word breaks), so a
    # capped span stays as semantically coherent as possible.
    DEFAULT_FALLBACK_PUNCTUATION: Sequence[str] = (";", ":", ",", " — ", " - ", " ")

    def __init__(
        self,
        sat_model: _Segmenter,
        tokenizer: _Tokenizer,
        max_chunk_len: int,
        max_chunks_per_doc: int,
        pad_token_id: int = 0,
        fallback_punctuation: Sequence[str] = DEFAULT_FALLBACK_PUNCTUATION,
        min_chunk_tokens: int = 4,
    ):
        self.sat_model = sat_model
        self.tokenizer = tokenizer
        self.max_chunk_len = max_chunk_len
        self.max_chunks_per_doc = max_chunks_per_doc
        self.pad_token_id = pad_token_id
        self.fallback_punctuation = fallback_punctuation
        # Splitter-artifact repair (see chunk_document): "sentences" shorter
        # than this many tokens are glued to their neighbor before capping.
        self.min_chunk_tokens = min_chunk_tokens

    def _cap_span(self, span: str, punct_idx: int = 0) -> List[str]:
        """
        Split `span` at the coarsest available punctuation-aware fallback
        boundary and greedily RE-PACK consecutive pieces into chunks of at
        most `max_chunk_len` tokens ("SaT Capped"'s length-capping half).

        Two properties the naive split-and-recurse version lacked (it made
        9% of real-cache chunks single-token and deleted every delimiter):
          * delimiters stay attached to their preceding piece, so the emitted
            chunks concatenate back to the original text verbatim (no lost
            commas/dashes, no GPT-2 leading-space retokenization inflation);
          * pieces are greedily merged up to the token cap, so a long comma
            sentence becomes a few near-cap chunks -- one word/clause per
            chunk can no longer happen unless a single piece is itself huge
            (then it recurses to the next-finer delimiter).
        """
        ids = self.tokenizer.encode(span, add_special_tokens=False)
        if len(ids) <= self.max_chunk_len:
            return [span] if span.strip() else []

        for i in range(punct_idx, len(self.fallback_punctuation)):
            punct = self.fallback_punctuation[i]
            if punct not in span:
                continue
            parts = span.split(punct)
            # Re-attach the delimiter to the piece it terminates.
            pieces = [p + punct for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])
            if len(pieces) <= 1:
                continue
            chunks: List[str] = []
            cur = ""
            for piece in pieces:
                if cur and not cur.strip():
                    # Never DROP a whitespace-only accumulator (it would break
                    # the verbatim-concatenation property): fold it into the
                    # piece that follows it instead.
                    piece = cur + piece
                    cur = ""
                if len(self.tokenizer.encode(piece, add_special_tokens=False)) > self.max_chunk_len:
                    # This piece alone exceeds the cap: flush and recurse on it
                    # with the next-finer delimiters only (this one is used up).
                    if cur.strip():
                        chunks.append(cur)
                    cur = ""
                    chunks.extend(self._cap_span(piece, i + 1))
                    continue
                tentative = cur + piece if cur else piece
                # Token counts are not additive under BPE; re-encode the merge
                # so the cap is exact.
                if cur and len(self.tokenizer.encode(tentative, add_special_tokens=False)) > self.max_chunk_len:
                    chunks.append(cur)
                    cur = piece
                else:
                    cur = tentative
            if cur.strip():
                chunks.append(cur)
            elif cur and chunks and len(self.tokenizer.encode(chunks[-1] + cur,
                                                              add_special_tokens=False)) <= self.max_chunk_len:
                # Trailing whitespace stays attached -- but only if the merged
                # chunk still re-encodes within the cap (BPE can grow by one).
                chunks[-1] = chunks[-1] + cur
            return [c for c in chunks if c.strip()]

        # No punctuation left to split on: hard fallback so we never emit a
        # span longer than max_chunk_len, no matter what. Split on a CHARACTER
        # boundary and recurse (binary split until every piece fits). The old
        # form -- decode(ids[i:i+L]) windows -- sliced the id stream, which is
        # NOT length-stable or lossless under byte-level BPE: a window edge
        # can split a multi-byte character, yielding U+FFFD corruption and
        # decoded windows that re-encode past the cap (verified on CJK/emoji
        # runs). A character split can neither corrupt text nor change total
        # bytes, and the cap check at the top of this method guarantees
        # termination (each half re-enters with strictly fewer characters).
        if len(span) < 2:
            return [span] if span.strip() else []
        mid = len(span) // 2
        return self._cap_span(span[:mid], punct_idx) + self._cap_span(span[mid:], punct_idx)

    def _merge_splitter_fragments(self, sentences: List[str]) -> List[str]:
        """
        Glue degenerate splitter fragments to their neighbor before capping.

        A sentence-boundary *stub* (RegexSentenceSegmenter splits after any
        [.!?]+whitespace) emits abbreviations and list markers as standalone
        "sentences" -- 'Dr.', 'on Jan.', '2.' -- which then become 1-3-token
        chunks: degenerate thoughts that burn chunk slots and pollute the
        SSL prediction targets (measured 11% of chunks <=3 tokens on
        list/abbreviation-heavy text; ~53% on numbered lists). A real SaT
        model would not produce these boundaries, so merging them is a repair
        of the stub's approximation, not a change of chunk granularity:
        anything already >= min_chunk_tokens is left exactly as split.
        """
        if self.min_chunk_tokens <= 1:
            return sentences
        merged: List[str] = []
        counts: List[int] = []
        for s in sentences:
            if merged and counts[-1] < self.min_chunk_tokens:
                merged[-1] = merged[-1] + " " + s
                counts[-1] = len(self.tokenizer.encode(merged[-1], add_special_tokens=False))
            else:
                merged.append(s)
                counts.append(len(self.tokenizer.encode(s, add_special_tokens=False)))
        # A trailing fragment has no successor: glue it backward instead.
        if len(merged) >= 2 and counts[-1] < self.min_chunk_tokens:
            tail = merged.pop()
            merged[-1] = merged[-1] + " " + tail   # may exceed the cap; _cap_span handles it
        return merged

    def chunk_document(self, text: str) -> List[str]:
        """SaT sentence split, then cap every sentence to `max_chunk_len` tokens."""
        sentences = self.sat_model.split(text)
        sentences = self._merge_splitter_fragments(sentences)
        capped_spans: List[str] = []
        for sentence in sentences:
            capped_spans.extend(self._cap_span(sentence))
        return capped_spans[: self.max_chunks_per_doc]

    def encode_recent(self, texts: List[str], window: int):
        """
        Tokenize each document and keep its most recent `window` tokens for the
        input lane (§4.1) -- raw, full-fidelity, never chunked. Left-aligned
        with pad on the right; the boolean mask marks the real tokens. This is
        the text-side analogue of the recent-token slice the input lane wants,
        owned here because the chunker holds the tokenizer.

        Returns:
          raw_ids:  (batch, window) long
          raw_mask: (batch, window) bool
        """
        batch = len(texts)
        raw = torch.full((batch, window), self.pad_token_id, dtype=torch.long)
        raw_mask = torch.zeros(batch, window, dtype=torch.bool)
        for b, text in enumerate(texts):
            ids = self.tokenizer.encode(text, add_special_tokens=False)[-window:]
            take = len(ids)
            if take:
                raw[b, :take] = torch.tensor(ids, dtype=torch.long)
                raw_mask[b, :take] = True
        return raw, raw_mask

    def chunk_batch(self, texts: List[str]):
        """
        texts: list of `batch` raw document strings.

        Returns:
          chunk_tensor: (batch, max_chunks_per_doc, max_chunk_len) long
          chunk_mask:   (batch, max_chunks_per_doc) bool, True where a chunk is real
        """
        batch = len(texts)
        chunk_tensor = torch.full(
            (batch, self.max_chunks_per_doc, self.max_chunk_len),
            self.pad_token_id, dtype=torch.long,
        )
        chunk_mask = torch.zeros(batch, self.max_chunks_per_doc, dtype=torch.bool)

        for b, text in enumerate(texts):
            spans = self.chunk_document(text)
            for c_idx, span in enumerate(spans):
                ids = self.tokenizer.encode(span, add_special_tokens=False)[: self.max_chunk_len]
                chunk_tensor[b, c_idx, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                chunk_mask[b, c_idx] = True

        return chunk_tensor, chunk_mask


# ----------------------------------------------------------------------
# Legacy fallback: fixed boundary-token-id split, for the synthetic
# token-id-only demo corpus in data.py (no real text, no tokenizer).
# ----------------------------------------------------------------------
class TokenIdBoundaryChunker:
    """
    Splits a batch of already-tokenized synthetic sequences into chunks by
    cutting at a fixed set of boundary token ids. This has none of SaT
    Capped's robustness (no real sentence-boundary model, no
    punctuation-aware length capping) -- it exists purely so this
    repository's synthetic demo (`data.py`) is runnable without a network
    call to fetch the SaT model or a real tokenizer. Prefer
    `SegmentAnyTextChunker` for anything trained on real text.
    """

    def __init__(self, boundary_token_ids: List[int], max_chunk_len: int,
                 max_chunks_per_doc: int, pad_token_id: int = 0):
        self.boundary_token_ids = set(boundary_token_ids)
        self.max_chunk_len = max_chunk_len
        self.max_chunks_per_doc = max_chunks_per_doc
        self.pad_token_id = pad_token_id

    def _split_one(self, ids: List[int]) -> List[List[int]]:
        chunks, current = [], []
        for tok in ids:
            current.append(tok)
            if tok in self.boundary_token_ids:
                chunks.append(current)
                current = []
        if current:
            chunks.append(current)
        return chunks

    def chunk_batch(self, token_ids: torch.Tensor, lengths: List[int]):
        batch = token_ids.shape[0]
        chunk_tensor = torch.full(
            (batch, self.max_chunks_per_doc, self.max_chunk_len),
            self.pad_token_id, dtype=torch.long,
        )
        chunk_mask = torch.zeros(batch, self.max_chunks_per_doc, dtype=torch.bool)

        for b in range(batch):
            ids = token_ids[b, : lengths[b]].tolist()
            chunks = self._split_one(ids)[: self.max_chunks_per_doc]
            for c_idx, chunk in enumerate(chunks):
                chunk = chunk[: self.max_chunk_len]
                chunk_tensor[b, c_idx, : len(chunk)] = torch.tensor(chunk, dtype=torch.long)
                chunk_mask[b, c_idx] = True

        return chunk_tensor, chunk_mask
