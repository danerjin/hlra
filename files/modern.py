"""
modern.py
=========
The "modern transformer" primitives (RMSNorm, RoPE, grouped-query attention with
QK-norm, SwiGLU) and the two drop-in blocks (`ModernEncoderLayer`,
`ModernAttention`) that the TOKEN-LEVEL modules build when their ArchConfig
(config.py) turns the corresponding flag on. When the ArchConfig is `is_legacy`
(the default), the wired modules never touch this file at all -- they build the
exact stock nn.MultiheadAttention / nn.TransformerEncoderLayer / nn.LayerNorm /
GELU path, keeping the state_dict byte-identical for old checkpoints. So nothing
here runs on the validated A-E path unless a flag is explicitly set.

Scope: these are the standard decoder-only upgrades applied where they mean
something -- the token-sequence transformers (chunk encoder, Talker, input lane,
baseline GPT). They are deliberately NOT used by the HRM loop or the gestalt
memory readers; see ArchConfig for why (MagicNorm hard-norm bounds the loop's
state, and RoPE is meaningless for a gated recurrence over pooled thoughts).

Design choices, and why:
  * SDPA backend. All attention routes through F.scaled_dot_product_attention, so
    it picks the flash / mem-efficient kernel where available (CUDA) and a correct
    math fallback elsewhere (ROCm gfx1151, MPS, CPU). No hand-rolled softmax.
  * GQA by KV expansion. torch 2.2 has no `enable_gqa` on SDPA, so K/V are
    repeat_interleaved up to the query head count -- correct and cheap; swap for
    the native arg on torch >= 2.5.
  * No bias in projections. The modern path is a fresh arch (its own checkpoints),
    so it adopts the current no-bias-in-attn/FFN convention rather than matching
    the stock MultiheadAttention parameterization.
  * RoPE buffers are non-persistent -- derivable from (head_dim, max_len, base), so
    they stay out of the state_dict and never trip a strict load.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ArchConfig


# ----------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root-mean-square LayerNorm (Zhang & Sennrich 2019): normalize by the RMS
    of the last dim, one learned per-channel scale, no mean-subtraction and no
    bias. Cheaper than LayerNorm and -- the reason it matters on this project's
    Strix Halo box -- it never calls the broken gfx1151 native LayerNorm-backward
    kernel that LATENT_MANUAL_LAYERNORM=1 exists to route around."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the norm in fp32 for stability under AMP, then cast back.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def make_norm(arch: ArchConfig, dim: int) -> nn.Module:
    """LayerNorm (legacy) or RMSNorm, per the arch flag."""
    return RMSNorm(dim) if arch.norm == "rms" else nn.LayerNorm(dim)


# ----------------------------------------------------------------------
class RoPE(nn.Module):
    """Rotary position embedding (Su et al. 2021), LLaMA/HF "rotate-half"
    convention. Applied to Q and K inside attention -- position enters as a
    per-position rotation of channel pairs, so only RELATIVE position reaches the
    dot product. Replaces the learned absolute position table in the module that
    turns it on. Bidirectional-safe (the rotation is symmetric), so the same
    module serves the causal Talker and the bidirectional encoders.

    cos/sin are cached up to `max_len` as non-persistent buffers (derivable, so
    excluded from the state_dict)."""

    def __init__(self, head_dim: int, max_len: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE needs an even head_dim, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (hd/2,)
        pos = torch.arange(max_len).float()                                            # (max_len,)
        freqs = torch.outer(pos, inv_freq)                                             # (max_len, hd/2)
        emb = torch.cat([freqs, freqs], dim=-1)                                        # (max_len, hd)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)
        self.max_len = max_len

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_heads, seq, head_dim). Broadcasts cos/sin (seq, head_dim)
        # over the batch and head axes; GQA-safe (works for any head count).
        seq = x.shape[-2]
        if seq > self.max_len:
            raise ValueError(f"RoPE sequence length {seq} exceeds cached max_len {self.max_len}")
        cos = self.cos[:seq].to(x.dtype)
        sin = self.sin[:seq].to(x.dtype)
        return x * cos + self._rotate_half(x) * sin


# ----------------------------------------------------------------------
class ModernAttention(nn.Module):
    """
    Multi/grouped-query attention over SDPA, with optional QK-norm and RoPE.

    Serves self- and cross-attention (pass `kv` for cross), and the two-width
    (kdim != d_model) cross-attention the Talker needs to read a d_latent thought
    from a d_model token stream -- the same capability the stock path got from
    MultiheadAttention(kdim=..., vdim=...).

    Masking: exactly one of `is_causal` / `key_padding_mask` is used per call
    (the wired modules never need both). key_padding_mask follows the codebase's
    nn convention: (batch, seq_kv) bool, True == PAD == ignore.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float,
                 kdim: Optional[int] = None, n_kv_heads: int = 0, qk_norm: bool = False,
                 bias: bool = False):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if not n_kv_heads else n_kv_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        self.head_dim = d_model // n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.dropout_p = dropout
        kdim = d_model if kdim is None else kdim

        # bias defaults off (the modern-LLM convention used by the token-level
        # path). The CORE readers pass bias=True so their projections structurally
        # mirror nn.MultiheadAttention (in_proj_bias + out_proj.bias), which makes
        # a future remap from an A-E checkpoint exact rather than approximate.
        self.q_proj = nn.Linear(d_model, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(kdim, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(kdim, self.n_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, d_model, bias=bias)

        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None

    def forward(self, x: torch.Tensor, kv: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                is_causal: bool = False, key_padding_mask: Optional[torch.Tensor] = None,
                rope: Optional[RoPE] = None) -> torch.Tensor:
        # `value` lets keys and values come from DIFFERENT tensors -- needed by the
        # gestalt memory reader, whose trust gate scales the value (value = kv * g)
        # while leaving the key untouched. Defaults to kv (standard self/cross-attn).
        kv = x if kv is None else kv
        val_src = kv if value is None else value
        B, Tq, _ = x.shape
        Tk = kv.shape[1]
        q = self.q_proj(x).view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)       # (B, nh, Tq, hd)
        k = self.k_proj(kv).view(B, Tk, self.n_kv_heads, self.head_dim).transpose(1, 2)    # (B, nkv, Tk, hd)
        v = self.v_proj(val_src).view(B, Tk, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if rope is not None:                          # only for self-attn over token positions
            q = rope(q)
            k = rope(k)
        if self.n_rep > 1:                            # GQA: expand K/V groups to the query head count
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        attn_mask = None
        if key_padding_mask is not None:
            # (B, Tk) True=pad -> additive bias (B, 1, 1, Tk) with -inf at pads.
            bias = torch.zeros(B, 1, 1, Tk, dtype=q.dtype, device=q.device)
            bias = bias.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
            attn_mask = bias
            is_causal = False

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal,
            dropout_p=self.dropout_p if self.training else 0.0,
        )                                             # (B, nh, Tq, hd)
        out = out.transpose(1, 2).reshape(B, Tq, self.n_heads * self.head_dim)
        return self.out_proj(out)


# ----------------------------------------------------------------------
def _swiglu_hidden(d_ff: int, multiple_of: int = 8) -> int:
    """SwiGLU uses THREE weight matrices (gate, up, down) vs a GELU FFN's two, so
    to keep the parameter/compute budget matched to a GELU FFN of width `d_ff`
    the inner width is scaled by 2/3 (the LLaMA convention), then rounded up to a
    multiple of `multiple_of` for alignment."""
    h = int(2 * d_ff / 3)
    return ((h + multiple_of - 1) // multiple_of) * multiple_of


class SwiGLU(nn.Module):
    """SwiGLU FFN (Shazeer 2020): down(SiLU(gate(x)) * up(x)). Inner width is the
    2/3-scaled `d_ff` so the block matches a GELU FFN of width d_ff in params."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        hidden = _swiglu_hidden(d_ff)
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.drop(F.silu(self.gate(x)) * self.up(x)))


def build_ffn(arch: ArchConfig, d_model: int, d_ff: int, dropout: float) -> nn.Module:
    """SwiGLU (arch.ffn == 'swiglu') or the stock Linear-GELU-Dropout-Linear FFN.
    The GELU branch is structurally the same nn.Sequential the legacy blocks use,
    so a modern-but-ffn='gelu' block keeps the familiar FFN."""
    if arch.ffn == "swiglu":
        return SwiGLU(d_model, d_ff, dropout)
    return nn.Sequential(
        nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
    )


# ----------------------------------------------------------------------
class ModernEncoderLayer(nn.Module):
    """
    Pre-LN transformer ENCODER layer (bidirectional self-attention), a drop-in for
    nn.TransformerEncoderLayer(norm_first=True) used by the input lane and chunk
    encoder when their arch is non-legacy. Same residual structure; the norm,
    attention (QK-norm / GQA), FFN (SwiGLU), and optional RoPE follow `arch`.

    `rope` is a shared RoPE module owned by the parent encoder (one cache for the
    whole stack), passed in per call; None when arch.rope is off.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, arch: ArchConfig):
        super().__init__()
        self.norm1 = make_norm(arch, d_model)
        self.norm2 = make_norm(arch, d_model)
        self.attn = ModernAttention(d_model, n_heads, dropout,
                                    n_kv_heads=arch.n_kv_heads, qk_norm=arch.qk_norm)
        self.ffn = build_ffn(arch, d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None,
                rope: Optional[RoPE] = None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), key_padding_mask=key_padding_mask, rope=rope))
        x = x + self.ffn(self.norm2(x))
        return x


class ModernEncoder(nn.Module):
    """A stack of ModernEncoderLayer with one shared RoPE cache -- the modern
    counterpart to nn.TransformerEncoder over the input lane / chunk encoder.
    Takes and returns (batch, seq, d_model); key_padding_mask is (batch, seq)
    bool, True == pad (the caller guards all-pad rows, as the stock path does)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float,
                 n_layers: int, arch: ArchConfig, max_len: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [ModernEncoderLayer(d_model, n_heads, d_ff, dropout, arch) for _ in range(n_layers)]
        )
        self.rope = RoPE(d_model // n_heads, max_len) if arch.rope else None

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # `src_key_padding_mask` name matches nn.TransformerEncoder so this is a
        # true drop-in at the call sites (input lane, chunk encoder).
        for layer in self.layers:
            x = layer(x, key_padding_mask=src_key_padding_mask, rope=self.rope)
        return x


# ----------------------------------------------------------------------
def remap_legacy_core_readers(state_dict: dict, model: nn.Module):
    """
    Import an OLD checkpoint (core readers = stock nn.MultiheadAttention) into a
    model built with `core_qk_norm=True` (core readers = ModernAttention).

    This is the EXACT remap: nn.MultiheadAttention packs Q/K/V into one
    `in_proj_weight` (3E x E) + `in_proj_bias` (3E), while ModernAttention keeps
    them split as `q_proj/k_proj/v_proj`. We slice the packed projection into the
    three, copy `out_proj` verbatim (same key name in both), and because the core
    ModernAttention is built with bias=True, every learned weight AND bias
    transfers with zero loss. The only params without an old counterpart are the
    new per-head QK RMSNorm scales (`q_norm/k_norm.weight`); they initialize to 1,
    so the readers resume with their trained projections intact and QK-norm added
    as a unit-scale normalization on top. (QK-norm is therefore not a byte-for-byte
    continuation of the old attention -- it rescales Q,K to unit RMS by design --
    but the transfer of the learned projections is exact.)

    Handles BOTH stock-MHA forms: the packed `in_proj_weight` (same-width readers:
    the memory readers, and the input reader at latent_mult=1) and the separate
    `q_proj_weight/k_proj_weight/v_proj_weight` (the two-width input reader at
    latent_mult>1, where kdim=d_input != d_latent). Biases are packed (3E) in both.

    Scoped to the three CORE readers only -- identified via the `core_qk_norm`
    flag on their parent GestaltCrossAttentionReader / HRMInnerLoop -- so a
    token-level ModernAttention (e.g. a modern Talker self-attn) is never touched
    even though a legacy checkpoint also has `...self_attn.in_proj_weight`.

    Returns a NEW state_dict (input is not mutated). A no-op for any reader that is
    already in ModernAttention form, so it is safe to call unconditionally whenever
    the target model has `core_qk_norm=True`.
    """
    from gestalt_memory import GestaltCrossAttentionReader
    from hrm_loop import HRMInnerLoop

    # The core-reader ModernAttention modules, by their state_dict path.
    targets = []
    for name, mod in model.named_modules():
        if isinstance(mod, GestaltCrossAttentionReader) and getattr(mod, "core_qk_norm", False):
            targets.append((f"{name}.attn", mod.attn))
        elif isinstance(mod, HRMInnerLoop) and getattr(mod, "core_qk_norm", False):
            targets.append((f"{name}.input_reader", mod.input_reader))

    new_sd = dict(state_dict)
    remapped = []
    for path, attn in targets:
        if not isinstance(attn, ModernAttention):
            continue
        packed = state_dict.get(f"{path}.in_proj_weight")
        sep_q = state_dict.get(f"{path}.q_proj_weight")
        if packed is None and sep_q is None:
            continue  # this reader is already modern in the checkpoint -> leave as-is
        E = attn.n_heads * attn.head_dim
        # --- projection WEIGHTS ---
        if packed is not None:
            if packed.shape[0] != 3 * E:
                raise ValueError(f"{path}.in_proj_weight has rows {packed.shape[0]}, expected 3*{E}")
            new_sd[f"{path}.q_proj.weight"] = packed[0:E].clone()
            new_sd[f"{path}.k_proj.weight"] = packed[E:2 * E].clone()
            new_sd[f"{path}.v_proj.weight"] = packed[2 * E:3 * E].clone()
            new_sd.pop(f"{path}.in_proj_weight", None)
        else:
            new_sd[f"{path}.q_proj.weight"] = state_dict[f"{path}.q_proj_weight"].clone()
            new_sd[f"{path}.k_proj.weight"] = state_dict[f"{path}.k_proj_weight"].clone()
            new_sd[f"{path}.v_proj.weight"] = state_dict[f"{path}.v_proj_weight"].clone()
            for k in ("q_proj_weight", "k_proj_weight", "v_proj_weight"):
                new_sd.pop(f"{path}.{k}", None)
        # --- projection BIASES (always packed 3E in stock MHA) ---
        ib = state_dict.get(f"{path}.in_proj_bias")
        if ib is not None:
            new_sd[f"{path}.q_proj.bias"] = ib[0:E].clone()
            new_sd[f"{path}.k_proj.bias"] = ib[E:2 * E].clone()
            new_sd[f"{path}.v_proj.bias"] = ib[2 * E:3 * E].clone()
            new_sd.pop(f"{path}.in_proj_bias", None)
        # out_proj.weight / out_proj.bias have identical key names in both modules
        # (NonDynamicallyQuantizableLinear vs nn.Linear) -> carried over untouched.
        # New QK RMSNorm scales have no old counterpart: seed from the freshly
        # built module (ones) so the returned state_dict is complete for strict load.
        if attn.q_norm is not None:
            new_sd[f"{path}.q_norm.weight"] = attn.q_norm.weight.detach().clone()
            new_sd[f"{path}.k_norm.weight"] = attn.k_norm.weight.detach().clone()
        remapped.append(path)
    return new_sd, remapped
