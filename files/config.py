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
    d_model: int = 128             # shared width for tokens, thoughts, and memory slots
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
    # Fraction of Stage-D+ steps that run the grounded (reconstruction) loss.
    # §2.4 CORRECTION to §5.7.1: reconstruction is the always-on anti-collapse
    # anchor and must run EVERY step (=1.0); the empirical collapse (notes §5)
    # happened precisely when reconstruction was thinned to a low floor while
    # SSL ran every step. Compute is managed by keeping SSL (cheap/parallel) as
    # the frequent one, NOT by thinning the anchor. Default 1.0 so every entry
    # point is safe-by-default; lower it only with eyes open.
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
#   preset   d_model   layers (ce/talk/in)   ~total params   ~emb share
#   smoke      192           2 / 2 / 2            43M            90%
#   small      512           4 / 4 / 3           153M            67%
#   base       768           6 / 6 / 4           307M            50%
#   large     1024           8 / 8 / 6           560M            37%
#   xl        1280          12 / 12 / 8          1.03B           25%
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
    # mid-range -- ~307M params
    "base": dict(d_model=768, n_heads=12, d_ff=3072, chunk_encoder_layers=6,
                 talker_layers=6, input_lane_layers=4, max_chunk_len=64,
                 max_chunks_per_doc=32, recent_token_window=512, memory_capacity=256),
    # ~0.5B -- first rung where transformer compute outweighs the embeddings
    "large": dict(d_model=1024, n_heads=16, d_ff=4096, chunk_encoder_layers=8,
                  talker_layers=8, input_lane_layers=6, max_chunk_len=64,
                  max_chunks_per_doc=48, recent_token_window=768, memory_capacity=384),
    # ~1B -- the large-scale target (untrained; needs the scaled data + GPU path)
    "xl": dict(d_model=1280, n_heads=20, d_ff=5120, chunk_encoder_layers=12,
               talker_layers=12, input_lane_layers=8, max_chunk_len=64,
               max_chunks_per_doc=64, recent_token_window=1024, memory_capacity=512),
}


def model_config(preset: str = "small", vocab_size: int = 50258, **overrides) -> ModelConfig:
    """Build a ModelConfig from a size preset (see MODEL_PRESETS)."""
    if preset not in MODEL_PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {list(MODEL_PRESETS)}")
    return ModelConfig(vocab_size=vocab_size, **{**MODEL_PRESETS[preset], **overrides})
