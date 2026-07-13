"""
data.py
=======
Real-text data pipeline for Stages A-E (§5.6: generic long-document text, no
speaker roles), plus an offline synthetic-text fallback so the whole curriculum
still runs with no network / no model downloads.

Design
------
Chunking is done *in the data pipeline*, not the training loop: each document is
segmented into chunks (SaT Capped, chunker.py) and tokenized up front, so
  (a) length bucketing by chunk count is possible (drop docs too short to
      exercise the gestalt memory), and
  (b) the SaT cost is paid in DataLoader workers, off the training hot path.

Each example is a 4-tuple of tensors, ready for model.forward_grounded /
forward_self_supervised:
    chunk_tensor (max_chunks, max_chunk_len) long
    chunk_mask   (max_chunks,)               bool
    raw_ids      (recent_token_window,)       long   -- input lane (§4.1)
    raw_mask     (recent_token_window,)       bool

Two source tiers, chosen by train.py depending on what's available:
  * REAL   -- `iter_hf_mixture` streams the configured HuggingFace mixture
              (config.DataConfig.sources) and a real SaT chunker segments it.
  * OFFLINE -- `SyntheticTextCorpus` emits real *strings* (fake words +
              punctuation) and `build_offline_chunker` runs the exact SaT-Capped
              code path with stub tokenizer/segmenter (no downloads).

PAD convention: id 0 is PAD everywhere (the model masks on `id != 0`). Real
tokenizers whose id 0 is a real token are wrapped by `ReservePadTokenizer`,
which offsets every id by +1 so 0 stays reserved; the stub tokenizer simply
never emits 0.
"""
from __future__ import annotations

import os
import re
import json
import random
import zlib
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import IterableDataset, Dataset

MANIFEST = "manifest.json"

# Role ids match model.py (USER=0, SELF=1, SYSTEM=2).
PAD = 0
USER, SELF, SYSTEM = 0, 1, 2
SPECIAL_TOKENS = 1  # only id 0 (PAD) is reserved in the stub vocab


# ======================================================================
# Offline stubs implementing chunker.py's _Tokenizer / _Segmenter protocols.
# These let the *real* SaT-Capped chunker run with zero downloads, for the
# dry run and for CI. Replace with a real tokenizer + SaT model for real runs.
# ======================================================================
class WhitespaceStubTokenizer:
    """Deterministic whitespace tokenizer hashing words into [SPECIAL, vocab).
    id 0 (PAD) is never emitted. `decode` is only used by the chunker's hard
    length-fallback and need not round-trip, so it returns placeholder words."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self._span = max(1, vocab_size - SPECIAL_TOKENS)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        toks = text.split()
        return [SPECIAL_TOKENS + (zlib.crc32(t.encode("utf-8")) % self._span) for t in toks]

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(f"w{int(i)}" for i in ids)


class RegexSentenceSegmenter:
    """Stub SaT: split text into sentences on terminal punctuation."""

    _SPLIT = re.compile(r"(?<=[.!?])\s+")

    def split(self, text_or_texts, **kwargs):
        if isinstance(text_or_texts, str):
            return [s for s in self._SPLIT.split(text_or_texts) if s.strip()]
        return [[s for s in self._SPLIT.split(t) if s.strip()] for t in text_or_texts]


class ReservePadTokenizer:
    """
    Wraps a real HuggingFace tokenizer so that id 0 stays reserved for PAD:
    every real id is offset by +1 on encode and -1 on decode. Size the model's
    vocab_size as (base_vocab + 1) when using this. Keeps the model's `id != 0`
    pad convention valid for tokenizers (e.g. gpt2) whose native id 0 is a real
    token.
    """

    def __init__(self, base_tokenizer):
        self.base = base_tokenizer

    @property
    def vocab_size(self) -> int:
        return self.base.vocab_size + 1

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return [i + 1 for i in self.base.encode(text, add_special_tokens=add_special_tokens)]

    def decode(self, ids: Sequence[int]) -> str:
        return self.base.decode([max(0, int(i) - 1) for i in ids])


def build_offline_chunker(model_cfg):
    """A real SegmentAnyTextChunker wired with offline stubs (no downloads)."""
    from chunker import SegmentAnyTextChunker
    return SegmentAnyTextChunker(
        sat_model=RegexSentenceSegmenter(),
        tokenizer=WhitespaceStubTokenizer(model_cfg.vocab_size),
        max_chunk_len=model_cfg.max_chunk_len,
        max_chunks_per_doc=model_cfg.max_chunks_per_doc,
        pad_token_id=PAD,
    )


def build_regex_gpt2_chunker(model_cfg, tokenizer_name: str = "gpt2"):
    """
    Real subword tokenizer (default gpt2) + regex sentence boundaries. Gives
    *decodable* output (so generated ids -> real text) without the SaT model
    download -- only `transformers` is needed. Returns (chunker, vocab_size),
    where vocab_size = base_vocab + 1 (id 0 reserved for PAD). Use the same
    tokenizer_name for training and inference so a saved checkpoint matches.
    """
    from transformers import AutoTokenizer
    from chunker import SegmentAnyTextChunker
    base = AutoTokenizer.from_pretrained(tokenizer_name)
    # We tokenize whole documents (no model context involved); silence the
    # per-document ">1024 tokens" warning, which floods (and slows) data prep.
    base.model_max_length = int(1e12)
    tok = ReservePadTokenizer(base)
    chunker = SegmentAnyTextChunker(
        sat_model=RegexSentenceSegmenter(), tokenizer=tok,
        max_chunk_len=model_cfg.max_chunk_len,
        max_chunks_per_doc=model_cfg.max_chunks_per_doc, pad_token_id=PAD,
    )
    return chunker, tok.vocab_size


# ======================================================================
# Text sources
# ======================================================================
def iter_hf_mixture(data_cfg) -> Iterator[str]:
    """
    Stream the configured HuggingFace mixture (config.DataConfig.sources),
    interleaved by weight, yielding raw document strings. Lazy-imports
    `datasets` so importing this module never requires it.
    """
    from datasets import load_dataset, interleave_datasets

    streams, probs = [], []
    for s in data_cfg.sources:
        ds = load_dataset(s.hf_id, s.name, split=s.split, streaming=data_cfg.streaming)
        ds = ds.map(lambda ex, f=s.text_field: {"text": ex.get(f, "")})
        streams.append(ds)
        probs.append(s.weight)
    total = sum(probs) or 1.0
    probs = [p / total for p in probs]

    mixed = interleave_datasets(streams, probabilities=probs, seed=data_cfg.seed)
    mixed = mixed.shuffle(buffer_size=data_cfg.shuffle_buffer, seed=data_cfg.seed)
    for ex in mixed:
        text = ex.get("text")
        if text:
            yield text


def iter_hf_single(hf_id: str, text_field: str = "text", name: Optional[str] = None,
                   split: str = "train", streaming: bool = True,
                   max_docs: Optional[int] = None) -> Iterator[str]:
    """
    Stream a single small HuggingFace dataset (for quick 1M-token smoke tests),
    yielding raw document strings. Lazy-imports `datasets`. Example:
        iter_hf_single("NeelNanda/pile-10k", "text", max_docs=400)
    """
    from datasets import load_dataset

    ds = load_dataset(hf_id, name, split=split, streaming=streaming)
    for i, ex in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            return
        text = ex.get(text_field)
        if text:
            yield text


def iter_local_parquet(files, text_field: str = "text",
                       max_docs: Optional[int] = None) -> Iterator[str]:
    """
    Yield document strings from LOCAL parquet file(s) -- the no-network escape
    hatch when HF's Xet streaming CDN 403s on a dataset (e.g. fineweb-edu).
    `files` is a path, a recursive glob, or a list of paths. Download the shards
    first with the CLASSIC (non-Xet) path, which does honor HF_HUB_DISABLE_XET:
        HF_HUB_DISABLE_XET=1 huggingface-cli download HuggingFaceFW/fineweb-edu \\
            --repo-type dataset --include "sample/10BT/*.parquet" --local-dir DIR
    then prep with `data_prep.py --local-glob "DIR/**/*.parquet"`. Reads with the
    'parquet' builder in streaming mode so it never loads a whole shard into RAM
    and hits no network.
    """
    import glob as _glob
    from datasets import load_dataset

    if isinstance(files, str):
        files = sorted(_glob.glob(os.path.expanduser(files), recursive=True))
    if not files:
        raise SystemExit("iter_local_parquet: no parquet files matched -- download shards first "
                         "(see the docstring / STRIX_HALO.md).")
    ds = load_dataset("parquet", data_files=files, split="train", streaming=True)
    for i, ex in enumerate(ds):
        if max_docs is not None and i >= max_docs:
            return
        text = ex.get(text_field)
        if text:
            yield text


class SyntheticTextCorpus:
    """
    Offline stand-in for real prose: emits documents of several sentences of
    fake words + punctuation, so the SaT-Capped chunker produces many chunks
    per doc. Real *strings*, not token ids -- exercises the full text path.
    """

    def __init__(self, n_docs: int, vocab_words: int = 400, seed: int = 0,
                 min_sents: int = 8, max_sents: int = 20,
                 min_words: int = 5, max_words: int = 14):
        self.n_docs = n_docs
        self.vocab_words = vocab_words
        self.seed = seed
        self.min_sents, self.max_sents = min_sents, max_sents
        self.min_words, self.max_words = min_words, max_words

    def _sentence(self, rng: random.Random) -> str:
        n = rng.randint(self.min_words, self.max_words)
        words = [f"word{rng.randint(0, self.vocab_words - 1)}" for _ in range(n)]
        return " ".join(words) + rng.choice([".", "?", "!"])

    def _doc(self, rng: random.Random) -> str:
        n = rng.randint(self.min_sents, self.max_sents)
        return " ".join(self._sentence(rng) for _ in range(n))

    def __iter__(self) -> Iterator[str]:
        rng = random.Random(self.seed)
        for _ in range(self.n_docs):
            yield self._doc(rng)


class DialogueTextCorpus:
    """
    Offline multi-turn USER/SELF dialogues for Stage F (§5.6). Each item is a
    list of (role_id, text) turns; role_id in {USER, SELF}. Real dialogue data
    (ideally with agree/disagree contrast for the §4.3 anti-sycophancy loss)
    is deferred -- this keeps the Stage F path runnable meanwhile.
    """

    def __init__(self, n_dialogues: int, turns: int = 4, seed: int = 0):
        self.n_dialogues = n_dialogues
        self.turns = turns
        self._syn = SyntheticTextCorpus(n_dialogues * turns, seed=seed, min_sents=2, max_sents=5)

    def __iter__(self):
        rng = random.Random(self._syn.seed)
        for _ in range(self.n_dialogues):
            turns = []
            for t in range(self.turns):
                role = USER if t % 2 == 0 else SELF
                turns.append((role, self._syn._doc(rng)))
            yield turns


# ======================================================================
# Chunking + Dataset
# ======================================================================
def chunk_text_example(text: str, chunker, window: int):
    """Chunk + tokenize one document into the 4-tensor example tuple.

    The document is pre-truncated to a character budget generously covering
    what can actually be kept (max_chunks_per_doc * max_chunk_len tokens, at a
    conservative 8 chars/token). Without this, a long document (e.g. a PG-19
    book) is SaT-segmented and tokenized in FULL -- typically 10-100x more work
    than the ~2k tokens that survive.

    The input-lane raw window is the trailing `window` token ids of the KEPT
    chunks, so it covers text the chunks actually contain *by construction*.
    (The old form -- `chunker.encode_recent` on the truncated text's tail --
    was disjoint from the kept chunks for any document longer than the chunk
    capacity: the 8-chars/token budget deliberately overshoots what
    max_chunks_per_doc can hold, so the text tail lay beyond the last kept
    chunk. Measured 0% overlap on 8k+-char docs. raw_ids/raw_mask are unused
    in Stages A-E -- input lanes are Stage F -- but the cache should not bake
    in a window that violates its own contract.)
    """
    max_chars = getattr(chunker, "max_chunks_per_doc", 0) * getattr(chunker, "max_chunk_len", 0) * 8
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
    chunk_tensor, chunk_mask = chunker.chunk_batch([text])       # (1, C, L), (1, C)
    kept = chunk_tensor[0][chunk_mask[0]]                         # (n_kept, L), doc order
    flat = kept[kept != PAD]                                      # 1-D real ids, doc order
    tail = flat[-window:]
    raw_ids = torch.full((window,), PAD, dtype=torch.long)
    raw_mask = torch.zeros(window, dtype=torch.bool)
    if tail.numel():
        raw_ids[: tail.numel()] = tail
        raw_mask[: tail.numel()] = True
    return chunk_tensor[0], chunk_mask[0], raw_ids, raw_mask


class DocumentChunkDataset(IterableDataset):
    """
    Streams documents from `text_factory` (a zero-arg callable returning a fresh
    text iterator), chunks each, and yields the 4-tensor example -- dropping any
    document with fewer than `min_chunks` chunks (length bucketing: too-short
    docs never exercise the gestalt memory or cross-thought credit assignment).
    """

    def __init__(self, text_factory: Callable[[], Iterable[str]], chunker,
                 window: int, min_chunks: int, max_examples: Optional[int] = None,
                 max_tokens: Optional[int] = None):
        self.text_factory = text_factory
        self.chunker = chunker
        self.window = window
        self.min_chunks = min_chunks
        self.max_examples = max_examples
        self.max_tokens = max_tokens          # stop after ~this many real (non-pad) tokens

    def __iter__(self):
        emitted, tokens = 0, 0
        for text in self.text_factory():
            ex = chunk_text_example(text, self.chunker, self.window)
            if int(ex[1].sum()) < self.min_chunks:
                continue
            yield ex
            emitted += 1
            tokens += int((ex[0] != 0).sum())
            if self.max_examples is not None and emitted >= self.max_examples:
                return
            if self.max_tokens is not None and tokens >= self.max_tokens:
                return


def collate_chunked(batch):
    """Stack a list of 4-tensor examples into batched tensors."""
    chunk_tensor = torch.stack([b[0] for b in batch], dim=0)
    chunk_mask = torch.stack([b[1] for b in batch], dim=0)
    raw_ids = torch.stack([b[2] for b in batch], dim=0)
    raw_mask = torch.stack([b[3] for b in batch], dim=0)
    return chunk_tensor, chunk_mask, raw_ids, raw_mask


# ======================================================================
# Offline chunk cache (built by data_prep.py) -> map-style Dataset
# ======================================================================
class CachedChunkDataset(Dataset):
    """
    Map-style dataset over a directory of pre-chunked shards (see data_prep.py).
    Chunking + tokenization are done ONCE offline, so training does no SaT work
    and can use DataLoader workers + shuffling. Token id tensors are stored as
    int32 on disk (ids < 2^31) and cast to long on access.

    Shards are concatenated into RAM at init (simple and fast for up to a few
    million examples). For much larger corpora, switch the shard loads to
    memory-mapping -- the on-disk format is unchanged.
    """

    def __init__(self, cache_dir: str, expect: Optional[dict] = None):
        with open(os.path.join(cache_dir, MANIFEST)) as f:
            self.manifest = json.load(f)
        cfg = self.manifest["config"]
        if expect is not None:
            for k in ("max_chunk_len", "max_chunks_per_doc", "recent_token_window"):
                if expect.get(k) != cfg.get(k):
                    raise ValueError(
                        f"cache/model mismatch on {k}: cache={cfg.get(k)} model={expect.get(k)}. "
                        f"Rebuild the cache with the matching preset (data_prep.py).")
        cts, cms, ris, rms = [], [], [], []
        for shard in self.manifest["shards"]:
            d = torch.load(os.path.join(cache_dir, shard))
            cts.append(d["chunk_tensor"]); cms.append(d["chunk_mask"])
            ris.append(d["raw_ids"]); rms.append(d["raw_mask"])
        if not cts:
            raise ValueError(
                f"cache at {cache_dir} contains no shards -- the prep run kept zero "
                f"documents (all filtered by min_chunks, or --max-tokens too small). "
                f"Re-run data_prep.py with looser limits.")
        self.chunk_tensor = torch.cat(cts, 0)
        self.chunk_mask = torch.cat(cms, 0)
        self.raw_ids = torch.cat(ris, 0)
        self.raw_mask = torch.cat(rms, 0)
        # Guard against a stale/mixed cache (e.g. a re-prep into an existing
        # dir that crashed mid-way: the old manifest survives next to a mix of
        # old and new shards, and training would silently use the blend).
        expected = self.manifest.get("total")
        if expected is not None and self.chunk_tensor.shape[0] != expected:
            raise ValueError(
                f"cache inconsistent: manifest says {expected} examples but shards "
                f"hold {self.chunk_tensor.shape[0]}. The cache dir likely mixes "
                f"shards from different prep runs -- re-run data_prep.py into a "
                f"FRESH directory.")
        self.vocab_size = cfg.get("vocab_size")
        self.config = cfg

    def __len__(self) -> int:
        return self.chunk_tensor.shape[0]

    def __getitem__(self, i):
        return (self.chunk_tensor[i].long(), self.chunk_mask[i],
                self.raw_ids[i].long(), self.raw_mask[i])
