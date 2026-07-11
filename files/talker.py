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

from gestalt_memory import GestaltCrossAttentionReader, GestaltMemoryBank


class TalkerDecoderLayer(nn.Module):
    """One causal self-attention + cross-attention (to the thought) + FFN block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, thought_kv: torch.Tensor, causal_mask: torch.Tensor):
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, attn_mask=causal_mask, need_weights=False)[0]
        h = self.norm2(x)
        x = x + self.cross_attn(h, thought_kv, thought_kv, need_weights=False)[0]
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
                 dropout: float, n_layers: int, n_roles: int, max_chunk_len: int):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_chunk_len, d_model)
        # Learned start-of-chunk vector. The decoder input is shifted right by
        # one (this vector at position 0, then tokens[:-1]) so position i is
        # never fed token i -- otherwise the NLL is trivially satisfiable by
        # copying the input and the thought is bypassed entirely. The very
        # first token must therefore be produced from the thought + memory
        # alone, which is what forces the latent to actually carry content.
        self.start_embed = nn.Parameter(torch.zeros(d_model))
        self.layers = nn.ModuleList(
            [TalkerDecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.memory_reader = GestaltCrossAttentionReader(d_model, n_heads, n_roles, dropout)
        self.out_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        target_tokens: torch.Tensor,   # (batch, chunk_len) ground-truth chunk tokens; shifted right *internally*
        thought: torch.Tensor,         # (batch, d_model) finished latent thought
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
        x = torch.cat([start, shifted], dim=1) + self.pos_embed(positions)

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
            x = layer(x, thought_kv, causal_mask)

        x = self.out_norm(x)
        return self.lm_head(x)
