"""
talker.py
=========
The Talker (§1.3): a separate, lightweight decoder-only stack that takes a
finished latent thought (post-loop, post-H-module) plus cross-attention
into the persistent gestalt memory, and autoregressively emits the tokens
for that chunk.

JEPA-Reasoner's "error containment" property is preserved by construction:
the *next* thought conditions only on this thought's latent H-state and the
memory it wrote -- never on the Talker's sampled tokens. Wiring that
guarantee is a property of how model.py calls these modules (the Talker's
output never feeds back into HRMInnerLoop's inputs), not something the
Talker module itself needs to enforce.

Ablations in the source paper show the Talker is a pure readout head: it
cannot produce meaningful content without a good latent. Architecturally
that just means the Talker should be *small* relative to the Reasoner and
have no independent recurrent state of its own -- exactly what this class
implements (a couple of standard causal decoder layers, no recurrence).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import ArchConfig
from modern import ModernAttention, RoPE, make_norm, build_ffn
from gestalt_memory import GestaltCrossAttentionReader, GestaltMemoryBank


class TalkerDecoderLayer(nn.Module):
    """One causal self-attention + cross-attention (to the thought) + FFN block.

    The token stream (self-attention, FFN) runs at the token width d_model; the
    thought it cross-attends is a chunk-level object at d_latent, so the cross-
    attention bridges the two widths (kdim/vdim=d_latent) when they differ. This
    is what lets a small word-level Talker read a wide chunk-level thought,
    querying different d_model-projections of it at each token.

    Legacy (arch.is_legacy, the default): the exact stock block above -- stock
    MultiheadAttention (self + two-width cross) + LayerNorm + GELU FFN, byte-
    identical for old checkpoints. Modern: RMSNorm / QK-normed GQA / SwiGLU, with
    RoPE (from the parent Talker) on the SELF-attention only -- the cross-attention
    reads a length-2 [thought, memory] set with no token order, so it gets no
    RoPE (matching the loop/memory readers, which are never rotary)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float,
                 d_latent: int | None = None, arch: ArchConfig | None = None):
        super().__init__()
        d_latent = d_model if d_latent is None else d_latent
        self.arch = arch = arch if arch is not None else ArchConfig()
        if arch.is_legacy:
            self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            if d_latent == d_model:
                self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            else:
                self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True,
                                                        kdim=d_latent, vdim=d_latent)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
            )
        else:
            self.self_attn = ModernAttention(d_model, n_heads, dropout,
                                             n_kv_heads=arch.n_kv_heads, qk_norm=arch.qk_norm)
            self.cross_attn = ModernAttention(d_model, n_heads, dropout, kdim=d_latent,
                                              n_kv_heads=arch.n_kv_heads, qk_norm=arch.qk_norm)
            self.ffn = build_ffn(arch, d_model, d_ff, dropout)
        self.norm1 = make_norm(arch, d_model)
        self.norm2 = make_norm(arch, d_model)
        self.norm3 = make_norm(arch, d_model)

    def forward(self, x: torch.Tensor, thought_kv: torch.Tensor, causal_mask: torch.Tensor,
                rope: "RoPE | None" = None):
        h = self.norm1(x)
        if self.arch.is_legacy:
            x = x + self.self_attn(h, h, h, attn_mask=causal_mask, need_weights=False)[0]
            h = self.norm2(x)
            x = x + self.cross_attn(h, thought_kv, thought_kv, need_weights=False)[0]
        else:
            x = x + self.self_attn(h, is_causal=True, rope=rope)
            h = self.norm2(x)
            x = x + self.cross_attn(h, kv=thought_kv)      # length-2 KV, no RoPE, no mask
        x = x + self.ffn(self.norm3(x))
        return x


class Talker(nn.Module):
    """
    Autoregressively reconstructs a chunk's tokens from a latent, cross-
    attending to the latent plus a memory readout slot. NOTE (§27): every
    live caller -- forward_grounded, generate.talker_decode -- passes an
    EMPTY memory bank (the codec conditions purely on the chunk's own
    latent), so `memory_reader` receives no gradient and its readout is a
    constant zero. It is kept so existing checkpoints load and as the hook
    for a future memory-conditioned decoding experiment; it is NOT trained.
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int,
                 dropout: float, n_layers: int, n_roles: int, max_chunk_len: int,
                 d_latent: int | None = None,
                 soft_role_tags: bool = False, soft_role_codebook: int = 16,
                 trust_gate: bool = False, soft_role_content: bool = False,
                 trust_gate_vector: bool = False, persona_tags: bool = False,
                 n_personas: int = 16, arch: ArchConfig | None = None,
                 core_qk_norm: bool = False):
        super().__init__()
        # The Talker is a word-level readout: its token embeddings, positions,
        # start vector, self-attention, FFN, and LM head are all at the token
        # width d_model. Only the thought it decodes from (and the memory readout
        # keyed by it) live at the wider thought width d_latent. d_latent defaults
        # to d_model (no widening), giving the original single-width Talker.
        d_latent = d_model if d_latent is None else d_latent
        self.arch = arch = arch if arch is not None else ArchConfig()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        # RoPE (on the self-attention) replaces the learned position table, so the
        # table exists only when arch.rope is off. The shared RoPE cache is built
        # once here and passed to every decoder layer's self-attention.
        self.pos_embed = None if arch.rope else nn.Embedding(max_chunk_len, d_model)
        self.rope = RoPE(d_model // n_heads, max_chunk_len) if arch.rope else None
        # Learned start-of-chunk vector. The decoder input is shifted right by
        # one (this vector at position 0, then tokens[:-1]) so position i is
        # never fed token i -- otherwise the NLL is trivially satisfiable by
        # copying the input and the thought is bypassed entirely. The very
        # first token must therefore be produced from the thought + memory
        # alone, which is what forces the latent to actually carry content.
        # (Unaffected by RoPE: the shift is a teacher-forcing correctness device,
        # not a positional encoding.)
        self.start_embed = nn.Parameter(torch.zeros(d_model))
        self.layers = nn.ModuleList(
            [TalkerDecoderLayer(d_model, n_heads, d_ff, dropout, d_latent=d_latent, arch=arch)
             for _ in range(n_layers)]
        )
        # Keyed by the (d_latent) thought; reads d_latent thoughts out of memory.
        self.memory_reader = GestaltCrossAttentionReader(
            d_latent, n_heads, n_roles, dropout,
            soft_role_tags=soft_role_tags, soft_role_codebook=soft_role_codebook,
            trust_gate=trust_gate, soft_role_content=soft_role_content,
            trust_gate_vector=trust_gate_vector, persona_tags=persona_tags,
            n_personas=n_personas, core_qk_norm=core_qk_norm)
        self.out_norm = make_norm(arch, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        target_tokens: torch.Tensor,   # (batch, chunk_len) ground-truth chunk tokens; shifted right *internally*
        thought: torch.Tensor,         # (batch, d_latent) finished latent thought (chunk-level width)
        memory: GestaltMemoryBank,
    ) -> torch.Tensor:
        """Returns logits: (batch, chunk_len, vocab_size)."""
        batch, chunk_len = target_tokens.shape
        device = target_tokens.device
        positions = torch.arange(chunk_len, device=device).unsqueeze(0)

        # Teacher forcing with a right shift: decoder input at position i is
        # token i-1 (the learned start vector at position 0), so the causal
        # stack predicts token i from strictly-earlier tokens + the thought,
        # never from token i itself. The caller passes the ground-truth chunk
        # tokens as both this argument and the NLL target.
        start = self.start_embed.expand(batch, 1, -1)                # (batch, 1, d_model)
        shifted = self.token_embed(target_tokens[:, :-1])            # (batch, chunk_len-1, d_model)
        x = torch.cat([start, shifted], dim=1)
        if self.pos_embed is not None:                               # learned positions (non-RoPE)
            x = x + self.pos_embed(positions)

        # The thought vector plus a memory readout form the cross-attention
        # key/value set (a length-2 "sequence": [thought, memory_summary]).
        memory_readout = self.memory_reader(thought, memory)     # (batch, d_model)
        thought_kv = torch.stack([thought, memory_readout], dim=1)  # (batch, 2, d_model)

        # Boolean causal mask (True = disallow attending to future positions).
        # Preferred over a float `-inf` mask: the latter NaNs on MPS/AMP, and a
        # bool mask is the modern, backend-safe form MultiheadAttention expects.
        causal_mask = torch.triu(
            torch.ones(chunk_len, chunk_len, dtype=torch.bool, device=device), diagonal=1
        )

        for layer in self.layers:
            x = layer(x, thought_kv, causal_mask, rope=self.rope)

        x = self.out_norm(x)
        return self.lm_head(x)
