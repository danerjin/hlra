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
    ):
        self.sat_model = sat_model
        self.tokenizer = tokenizer
        self.max_chunk_len = max_chunk_len
        self.max_chunks_per_doc = max_chunks_per_doc
        self.pad_token_id = pad_token_id
        self.fallback_punctuation = fallback_punctuation

    def _cap_span(self, span: str) -> List[str]:
        """
        Recursively split `span` at the finest available punctuation-aware
        fallback boundary until every resulting piece tokenizes to at most
        `max_chunk_len` tokens ("SaT Capped"'s length-capping half).
        """
        ids = self.tokenizer.encode(span, add_special_tokens=False)
        if len(ids) <= self.max_chunk_len:
            return [span] if span.strip() else []

        for punct in self.fallback_punctuation:
            if punct in span:
                pieces = [p.strip() for p in span.split(punct) if p.strip()]
                if len(pieces) > 1:
                    capped: List[str] = []
                    for piece in pieces:
                        capped.extend(self._cap_span(piece))
                    return capped

        # No punctuation left to split on: hard token-count fallback so we
        # never emit a span longer than max_chunk_len, no matter what.
        return [
            self.tokenizer.decode(ids[i : i + self.max_chunk_len])
            for i in range(0, len(ids), self.max_chunk_len)
        ]

    def chunk_document(self, text: str) -> List[str]:
        """SaT sentence split, then cap every sentence to `max_chunk_len` tokens."""
        sentences = self.sat_model.split(text)
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
