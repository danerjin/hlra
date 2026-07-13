"""
config.py
=========
Single source of truth for every hyperparameter used across the architecture.

Grouping the numbers in one place makes the "why every non-obvious choice is
justified" section of the design doc (§3) easy to audit: each field below is
commented with the section of `latent-thought-architecture.md` it comes from.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # ---- basic sizes ---------------------------------------------------
    # vocab_size must match the tokenizer. Real runs set it to
    # (tokenizer_vocab + 1): id 0 is reserved for PAD, and real token ids are
    # offset by +1 so the model's `id != 0` pad convention stays valid even for
    # tokenizers (e.g. gpt2) whose id 0 is a real token -- see data.ReservePadTokenizer.
    vocab_size: int = 32001        # gpt2 (50257) or your tokenizer + 1; small default for the dry run
    # d_model is the TOKEN width -- the word-level embedding dimension. Tokens
    # are looked up at this width, and the token-level modules that read/emit
    # individual tokens (the Talker's token stream, the input lane's raw-token
    # stream) work here. It is deliberately narrow: a token carries one word's
    # worth of meaning.
    d_model: int = 128             # token (word-level) embedding width
    # A THOUGHT is a whole chunk (a clause/sentence of many tokens), so it needs
    # more capacity than a single token -- otherwise the Talker, decoding a
    # d_model latent back into many tokens, loses information. So the thought /
    # chunk-latent width is a multiple of the token width:
    #     d_latent = latent_mult * d_model
    # and EVERYTHING that carries a thought lives at d_latent: the chunk encoder's
    # pooled output, the gestalt memory, pred_head, the EMA target, and the whole
    # HRM loop. The encoder runs its transformer body at d_latent (not just a
    # projection after pooling) so the pooled thought genuinely uses the width.
    # This departs from JEPA-Reasoner's token==thought "Latent Dim" on purpose:
    # their analyzed latents were ~token-sized, ours are multi-token chunks.
    # latent_mult=1 (d_latent == d_model) is an exact no-op -- every widening
    # projection is Identity and every cross-width attention is built plain -- so
    # validated d_latent==d_model runs are byte-identical.
    latent_mult: int = 1           # d_latent = latent_mult * d_model (thought/chunk-latent width)
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1

    # ---- chunking (§1, §3.1) -------------------------------------------
    max_chunk_len: int = 64        # L=64, matching Thought Gestalt's "SaT Capped" (§3.1, §5.1)
    max_chunks_per_doc: int = 32   # enough thoughts/doc to actually exercise the gestalt memory (§1.2)

    # ---- inner HRM loop (§1.1, §3.2) -----------------------------------
    l_steps_per_h_update: int = 3  # "3-fast : 1-slow" ratio, empirical (§3.2)
    h_updates_per_thought: int = 2 # "two high-level cycles" per HRM-Text
    # Warmup deep-credit-assignment schedule (§3.5): backprop window grows
    # from 2 -> 5 steps over training. These are the start/end of that ramp.
    inner_loop_grad_window_start: int = 2
    inner_loop_grad_window_end: int = 5

    # ---- diagonal decay gate (§0, §3.3) --------------------------------
    # Per-channel decay of the state-transition cell; the exp(-softplus·dt)
    # construction keeps each channel's decay strictly inside (0, 1) for any
    # number of steps (see decay_gate.py). Note: boundedness at arbitrary
    # depth comes from MagicNorm's hard-norm (norm.py), not from this gate.
    decay_min: float = 0.01
    decay_max: float = 0.99

    # ---- gestalt memory (§1.2, §3.6) -----------------------------------
    memory_capacity: int = 64      # FIFO capacity (slots), per example
    # Backward truncation window for the *outer* thought-memory recurrence.
    memory_grad_window_start: int = 1
    memory_grad_window_end: int = 5

    # ---- self-supervised JEPA loss (§2.1, §3.4) -------------------------
    cosine_loss_k: float = 4.0     # scale factor sweeping showed k=4 works at this width
    # EMA momentum for the target encoder. Raised from the source paper's 0.98:
    # a faster-moving target lets a small model chase it into a collapsed
    # (constant) latent; a slower target (higher momentum) is a stronger
    # anti-collapse defense (§3.4). Retune per scale.
    ema_momentum: float = 0.996

    # ---- adaptive computation time / test-time compute dial (§1.1, §5.5) -
    act_ponder_cost: float = 0.01
    act_max_ponder_steps: int = 6

    # ---- role tags for the two-lane input/self separation (§4.2) -------
    role_tags: tuple = ("USER", "SELF", "SYSTEM")

    # ---- chunk encoder (§2.1, shared latent producer) -------------------
    chunk_encoder_layers: int = 2

    # ---- talker (§1.3) ---------------------------------------------------
    talker_layers: int = 2

    # ---- input lane encoder (§4.1, §4.2) --------------------------------
    input_lane_layers: int = 2
    recent_token_window: int = 128  # raw tokens kept at full fidelity before aging into gestalts

    # The next-latent prediction (§2.1's self-supervised signal AND generation) is
    # produced by the HRM loop itself: loop(encode(chunk_t)) -> pred_head predicts
    # chunk t+1's encoder-space latent (model.forward_self_supervised). There is no
    # separate linear SSL head or detached gen MLP -- those were the §2.4
    # collapse-era shortcut, removed after the notes §25.1 A/B showed the on-loop
    # loss is more collapse-robust and needs no isolation head.

    @property
    def d_latent(self) -> int:
        """The thought / chunk-latent width: a multiple of the token width
        d_model. Everything that carries a thought lives here -- the chunk
        encoder's pooled output and transformer body, the gestalt memory,
        pred_head, the EMA target, and the whole HRM loop. Token-level modules
        (Talker token stream, input-lane raw tokens) stay at d_model and
        cross-attend into this space. latent_mult=1 -> d_latent == d_model."""
        return self.latent_mult * self.d_model

    @property
    def latent_d_ff(self) -> int:
        """FFN width for the modules that operate at d_latent -- the chunk
        encoder body and the HRM loop's decay-gate sublayers. Scaled by the same
        factor as the width: latent_d_ff = latent_mult * d_ff. Since every preset
        sets d_ff = 4 * d_model, this is 4 * d_latent (the FFN:width ratio HRM-
        Text/JEPA-Reasoner use), and it is exactly d_ff -- an unconditional no-op
        -- at latent_mult=1 whatever d_ff:d_model ratio a config chose. d_ff
        itself stays the token-level FFN width (Talker, input lane)."""
        return self.latent_mult * self.d_ff

    def __post_init__(self) -> None:
        if not isinstance(self.latent_mult, int) or self.latent_mult < 1:
            raise ValueError(f"latent_mult must be an int >= 1, got {self.latent_mult!r}")
        # The encoder body, the loop, and their memory cross-attention all run at
        # embed_dim=d_latent, so n_heads must divide d_latent. Since d_latent =
        # latent_mult * d_model and n_heads already divides d_model, this holds --
        # but a hand-set preset could violate it, so check rather than fail
        # cryptically inside MultiheadAttention.
        if self.d_latent % self.n_heads != 0:
            raise ValueError(
                f"d_latent ({self.d_latent}) must be divisible by n_heads ({self.n_heads})")


@dataclass
class TrainConfig:
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_steps_per_stage: int = 200
    grad_clip: float = 1.0
    # Stage transitions are gated on a validation-loss-plateau signal (§5.7.2)
    # rather than fixed iteration counts.
    plateau_patience: int = 10
    plateau_min_delta: float = 1e-3
    # MOOT since the §27 restructure (kept only so old checkpoints/CLIs don't
    # break): reconstruction is now a cheap parallel autoencoder codec (encoder ->
    # Talker, no loop) and always runs every step as the anti-collapse anchor.
    # The expensive path is now the SEQUENTIAL on-loop SSL, not reconstruction, so
    # there is nothing to thin here. Leave at 1.0.
    grounded_loss_min_frequency: float = 1.0
    # The grounded (reconstruction) loss is the always-on anti-collapse anchor on
    # the shared chunk encoder (runs every step at frequency 1.0). The on-loop SSL
    # (model.forward_self_supervised) is the forward-prediction signal that trains
    # the loop; it is now the MAIN predictive objective, so it runs co-equal with
    # reconstruction (weight 1.0), not demoted to a whisper. The variance floor
    # (VICReg-style) hard-floors the shared latent's per-dim variance as the
    # anti-collapse backstop. (Notes §25.1: on-loop SSL held latent_std healthy at
    # this weight where the old linear SSL flirted with collapse. RE-TUNE at scale.)
    ssl_loss_weight: float = 1.0              # weight on the on-loop SSL cosine prediction term
    ssl_var_weight: float = 2.0              # weight on the anti-collapse variance regularizer
    log_every: int = 10
    device: str = "cpu"
    seed: int = 0

    # ---- scaling knobs (used by trainer.Trainer / train_scaled.py) -------
    # These default to no-ops so the smoke path (train_real/run_small) is
    # unchanged; train_scaled.py overrides them for real runs.
    grad_accum_steps: int = 1                 # micro-batches per optimizer step
    amp: bool = False                         # mixed-precision autocast (enable on CUDA)
    amp_dtype: str = "bf16"                   # "bf16" | "fp16" (fp16 uses a GradScaler on CUDA)
    num_workers: int = 0                      # DataLoader workers (>0 needs the cached dataset)
    warmup_steps: int = 0                     # linear LR warmup (optimizer steps)
    total_steps: int = 0                      # horizon for cosine decay; 0 disables the schedule
    min_lr_ratio: float = 0.1                 # cosine decays to this fraction of `lr`
    # Per-stage LR schedule (curriculum fix): when stage_steps is set, give each
    # stage its own warmup->cosine over its own budget instead of one global
    # cosine across A..E (which starves the late stages -> D/E "regression").
    per_stage_lr: bool = False
    checkpoint_every: int = 0                 # steps between checkpoints; 0 = only at end
    # Additionally keep a NUMBERED checkpoint_{step}.pt every this many steps
    # (0 = off). The rolling checkpoint.pt is crash-safe but gives no rollback
    # depth if a slow pathology is noticed late; ~2-4 snapshots/stage is cheap
    # insurance on a multi-day run (each snapshot is model+optimizer+EMA sized).
    checkpoint_archive_every: int = 0
    # Fixed per-stage step budgets (A..F) for the curriculum. If set, stage
    # transitions happen on these budgets instead of the noisy plateau gate --
    # far more predictable for long runs. None -> keep plateau gating.
    stage_steps: tuple = None


@dataclass
class SourceSpec:
    """One corpus in the Stages A-E mixture. `weight` is a sampling
    probability (the specs' weights are normalized at interleave time)."""
    hf_id: str                    # HuggingFace dataset id
    text_field: str               # which column holds the document text
    weight: float                 # relative sampling weight
    name: str | None = None       # dataset config/subset name, if any
    split: str = "train"


def _default_mixture() -> list:
    """
    The recommended Stages A-E mixture (§ dataset discussion): mostly general
    prose, a reasoning slice, and a deliberate long-document emphasis so the
    gestalt memory + cross-thought credit assignment actually receive gradient.
    Weights are the target token proportions, not doc counts.
    """
    return [
        SourceSpec("HuggingFaceFW/fineweb-edu", "text", 0.45, name="sample-10BT"),  # general prose
        SourceSpec("pg19",                      "text", 0.15),                        # long books
        SourceSpec("wikimedia/wikipedia",       "text", 0.10, name="20231101.en"),   # encyclopedic
        SourceSpec("togethercomputer/RedPajama-Data-1T", "text", 0.10, name="arxiv"),# long technical
        SourceSpec("open-web-math/open-web-math","text", 0.15),                       # reasoning
        SourceSpec("bigcode/the-stack-smol",    "content", 0.05),                     # light code
    ]


@dataclass
class DataConfig:
    """
    Real-text data pipeline (Stages A-E). The pipeline is offline-safe: if the
    `datasets`/`wtpsplit` deps or network are unavailable, train.py falls back
    to a synthetic *text* corpus + stub chunker so the whole thing still runs
    (see data.py). None of these fields trigger a download until the real
    pipeline is actually selected.
    """
    tokenizer_name: str = "gpt2"          # swap for your model's tokenizer
    sat_model_name: str = "sat-3l-sm"     # Segment Any Text model (Thought Gestalt's "SaT Capped")
    streaming: bool = True                # stream from the Hub rather than downloading whole shards
    min_chunks: int = 4                   # length bucketing: drop docs too short to exercise memory
    shuffle_buffer: int = 10_000          # streaming shuffle buffer size
    seed: int = 0
    sources: list = field(default_factory=_default_mixture)
    # ---- offline chunk cache (data_prep.py / data.CachedChunkDataset) ----
    cache_dir: str = "chunk_cache"        # where pre-chunked shards are written/read
    shard_size: int = 4096                # examples per shard file


# ----------------------------------------------------------------------
# Model-size presets. Only the size-related fields differ; everything else
# stays at ModelConfig defaults. vocab_size is set from the tokenizer at
# build time (gpt2 -> 50258). Use config.model_config("small").
#
# A scaling ladder from the smoke model up to ~1B params. Total-parameter
# counts below are measured at the gpt2 vocab (50258); the four vocab-scale
# embedding tables (token embed for the chunk encoder, Talker, and input lane,
# plus the Talker's output head) dominate at small widths and shrink to a
# minority of the budget as d_model grows (see the emb% column). Head dim is
# held at 64 (n_heads = d_model / 64) for every rung above smoke.
#
# d_latent is the thought/chunk-latent width (= latent_mult * d_model); the five
# baseline rungs are latent_mult=1 (d_latent == d_model, token==thought). The
# `*-w3` rungs are the WIDE-THOUGHT ladder: latent_mult=3, each sized to its
# baseline tier's budget by trading token width for thought width -- the matched-
# param A/B for "a chunk latent needs more capacity than a token".
#
#   preset     d_model  d_latent  n_heads  layers (ce/talk/in)  ~total params  ~emb share
#   smoke        192      192        6          2 / 2 / 2            43M           90%
#   small        512      512        8          4 / 4 / 3           152M          68%
#   small-w3     320      960        5          4 / 4 / 3           156M          41%
#   base         768      768       12          6 / 6 / 4           305M          51%
#   base-w3      448     1344        7          6 / 6 / 4           324M          28%
#   large       1024     1024       16          8 / 8 / 6           560M          37%
#   large-w3     576     1728        9          8 / 8 / 6           595M          19%
#   xl          1280     1280       20         12 / 12 / 8          1.03B         25%
#   xl-w3        640     1920       10         12 / 12 / 8          940M          14%
#
# (Note the emb share collapses at a fixed budget -- small 68% -> small-w3 41% --
# because widening the thought spends the freed embedding params on the encoder/
# loop/pred_head, which is the whole point.)
#
# On powers of 2: with an ODD multiple (x3) you cannot make BOTH d_model and
# d_latent powers of two, and you don't need to -- the real GPU constraint is
# divisibility by the head dim (held at 64) and, ideally, 128-alignment. The
# small/base/large -w3 d_models (320/448/576) are budget-matched multiples of 64
# with odd head counts (5/7/9), NOT multiples of 128 -- the price of matching the
# tier budget with a x3 multiple. xl-w3 (640) is the exception: the xl budget
# happens to land on a clean 128-aligned, even-head config. If you prefer
# 128-alignment + even heads over an exact budget match on the smaller rungs, snap
# d_model up/down to a multiple of 128 (e.g. base-w3 384 -> ~249M or 512 -> ~408M
# straddling base's 305M) and accept the size drift. In all cases d_latent = 3*
# d_model stays a multiple of 64 (960/1344/1728/1920), so attention is always valid.
#
# All presets ship latent_mult=1 (d_latent == d_model) so the validated runs are
# unchanged. To make thoughts wider than tokens -- so a chunk latent decodes back
# into many tokens without an information bottleneck -- pass e.g.
# model_config("small", latent_mult=3): the chunk encoder, gestalt memory,
# pred_head, EMA target, and the HRM loop all run at 3*d_model, while token
# embeddings and the Talker's token stream stay word-level at d_model. The
# encoder/loop FFN auto-scales with the thought width (cfg.latent_d_ff =
# latent_mult * d_ff = 4*d_latent for the shipped 4x presets); d_ff itself remains
# the d_model-proportioned FFN of the token-level modules (Talker, input lane).
# NOTE: widening the thought moves the anti-collapse machinery (EMA target,
# cosine SSL, per-dim variance floor) into the wider space -- re-tune
# cosine_loss_k and the variance weight at the new width.
#
# The non-size hyperparameters (decay-gate min/max, ema_momentum, cosine_loss_k,
# act_ponder_cost, the grad-truncation windows, loss weights) were tuned on the
# smoke run and should be RE-TUNED per scale -- notably cosine_loss_k is width-
# dependent (§3.4) and the anti-collapse weights/momentum were set on the 1.5M-
# token toy (notes §9). Treat large/xl as *starting points*, not validated.
# ----------------------------------------------------------------------
MODEL_PRESETS = {
    # the current ~1M-token smoke model (kept so nothing regresses) -- ~43M
    "smoke": dict(d_model=192, n_heads=6, d_ff=768, chunk_encoder_layers=2,
                  talker_layers=2, input_lane_layers=2, max_chunk_len=48,
                  max_chunks_per_doc=12, recent_token_window=96, memory_capacity=64),
    # a genuinely-trained small model target -- ~153M params at gpt2 vocab
    "small": dict(d_model=512, n_heads=8, d_ff=2048, chunk_encoder_layers=4,
                  talker_layers=4, input_lane_layers=3, max_chunk_len=64,
                  max_chunks_per_doc=32, recent_token_window=256, memory_capacity=128),
    # wide-thought variant of `small` (latent_mult=3): tokens stay word-level at
    # d_model=320 but a THOUGHT is 3x wider (d_latent=960), so a chunk latent can
    # decode into many tokens without an information bottleneck. Rebalanced from
    # `small` by trading token width (512->320) for thought width to hold the same
    # ~150M budget -- the matched-param A/B against `small`. head_dim stays 64
    # (5 heads; d_latent 960 / 5 = 192 per head in the loop). d_ff=1280 is the
    # token-level FFN; the encoder/loop FFN is latent_d_ff = 3*1280 = 3840.
    #
    # Anti-collapse re-tuned for the 960-d thought space (the terms were set at
    # <=512-d). cosine_loss_k 4.0 -> 5.5: the scaled-cosine gradient falls as
    # 1/sqrt(d) (measured: 0.73x weaker at 960 vs 512 == sqrt(512/960)), so k must
    # rise ~sqrt(960/512)=1.37x to keep the predictive learning signal at the same
    # magnitude. The variance-floor WEIGHT (train_cfg.ssl_var_weight) is a
    # per-dim-mean hinge and so width-invariant in effect, but the natural per-dim
    # latent std drops 0.461 -> 0.304 at init (still 3x over the 0.1 floor, but
    # ~34% less margin), so the recommended run uses ssl_var_weight 2.0 -> 3.0 as a
    # firmer (still-dormant-when-healthy) backstop -- pass `--var-weight 3.0` to
    # train_scaled.py. Both remain STARTING points: watch latent_std through the
    # Stage-B predictor boundary (the real collapse signal, §2.4) and adjust.
    "small-w3": dict(d_model=320, n_heads=5, d_ff=1280, latent_mult=3, cosine_loss_k=5.5,
                     chunk_encoder_layers=4, talker_layers=4, input_lane_layers=3,
                     max_chunk_len=64, max_chunks_per_doc=32, recent_token_window=256,
                     memory_capacity=128),
    # mid-range -- ~307M params
    "base": dict(d_model=768, n_heads=12, d_ff=3072, chunk_encoder_layers=6,
                 talker_layers=6, input_lane_layers=4, max_chunk_len=64,
                 max_chunks_per_doc=32, recent_token_window=512, memory_capacity=256),
    # wide-thought `base` (latent_mult=3): tokens 448, thoughts d_latent=1344,
    # rebalanced to hold base's ~305M budget (~324M). d_model=448 = 7*64 keeps
    # head_dim=64 (loop attends at 1344/7 = 192); it is a multiple of 64 but not
    # 128, and n_heads=7 is odd -- the price of matching the budget with a x3
    # (odd) multiple (see the powers-of-2 note in the ladder comment above).
    # cosine_loss_k 4.0 -> 6.5 = round(4*sqrt(1344/512)); same 1/sqrt(d) cosine-
    # gradient law as small-w3 (see there for the derivation). --var-weight ~3.0
    # advised (natural per-dim std ~0.27 at init, ~2.7x over the 0.1 floor).
    "base-w3": dict(d_model=448, n_heads=7, d_ff=1792, latent_mult=3, cosine_loss_k=6.5,
                    chunk_encoder_layers=6, talker_layers=6, input_lane_layers=4,
                    max_chunk_len=64, max_chunks_per_doc=32, recent_token_window=512,
                    memory_capacity=256),
    # ~0.5B -- first rung where transformer compute outweighs the embeddings
    "large": dict(d_model=1024, n_heads=16, d_ff=4096, chunk_encoder_layers=8,
                  talker_layers=8, input_lane_layers=6, max_chunk_len=64,
                  max_chunks_per_doc=48, recent_token_window=768, memory_capacity=384),
    # wide-thought `large` (latent_mult=3): tokens 576, thoughts d_latent=1728,
    # rebalanced to hold large's ~560M budget (~595M). d_model=576 = 9*64 keeps
    # head_dim=64 (loop attends at 1728/9 = 192); multiple of 64 (not 128),
    # n_heads=9 odd -- same budget-vs-alignment tradeoff as base-w3.
    # cosine_loss_k 4.0 -> 7.5 = round(4*sqrt(1728/512)); same law as small-w3.
    # --var-weight ~3.0 advised (natural per-dim std ~0.25, ~2.5x over the floor).
    "large-w3": dict(d_model=576, n_heads=9, d_ff=2304, latent_mult=3, cosine_loss_k=7.5,
                     chunk_encoder_layers=8, talker_layers=8, input_lane_layers=6,
                     max_chunk_len=64, max_chunks_per_doc=48, recent_token_window=768,
                     memory_capacity=384),
    # ~1B -- the large-scale target (untrained; needs the scaled data + GPU path)
    "xl": dict(d_model=1280, n_heads=20, d_ff=5120, chunk_encoder_layers=12,
               talker_layers=12, input_lane_layers=8, max_chunk_len=64,
               max_chunks_per_doc=64, recent_token_window=1024, memory_capacity=512),
    # wide-thought `xl` (latent_mult=3): tokens 640, thoughts d_latent=1920, ~940M
    # (~9% under xl's ~1B; the tier is loose). Unlike the smaller w3 rungs, the
    # budget here lands on a CLEAN config: d_model=640 = 5*128 is 128-aligned with
    # an EVEN head count (10), and d_latent=1920 is 128-aligned too -- so xl-w3
    # keeps head_dim=64 (loop attends at 1920/10 = 192) without the odd-head
    # compromise base-w3/large-w3 make to hit their budgets.
    # cosine_loss_k 4.0 -> 8.0 = round(4*sqrt(1920/512)); same law as small-w3.
    # --var-weight ~3.0 advised (natural per-dim std ~0.26, ~2.6x over the floor).
    "xl-w3": dict(d_model=640, n_heads=10, d_ff=2560, latent_mult=3, cosine_loss_k=8.0,
                  chunk_encoder_layers=12, talker_layers=12, input_lane_layers=8,
                  max_chunk_len=64, max_chunks_per_doc=64, recent_token_window=1024,
                  memory_capacity=512),
}


def model_config(preset: str = "small", vocab_size: int = 50258, **overrides) -> ModelConfig:
    """Build a ModelConfig from a size preset (see MODEL_PRESETS)."""
    if preset not in MODEL_PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {list(MODEL_PRESETS)}")
    return ModelConfig(vocab_size=vocab_size, **{**MODEL_PRESETS[preset], **overrides})
