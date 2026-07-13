"""
gestalt_memory.py
=================
The persistent gestalt memory (§1.2, §3.6, §4.2): a fixed-capacity FIFO bank
of finished thought vectors, read by cross-attention from the inner HRM loop
(the predictive branch's reader -- the one that trains). The Talker also
holds a `GestaltCrossAttentionReader`, but since the §27 restructure the
codec path always hands it an EMPTY bank (reconstruction conditions on the
chunk's own latent only), so that reader is untrained dead weight kept for
checkpoint compatibility -- do not wire a populated bank into the Talker
without training it first. Writes are *not* detached by default -- gradient
from a later thought's loss is allowed to reach back into the state that
produced an earlier thought, subject to truncation (handled by the caller
via `utils.truncate_gradient_window`, applied to the list of stored vectors
before each cross-attention read).

Role tags (§4.2): every slot also carries a role id (USER / SELF / SYSTEM)
so that cross-attention can learn source-dependent weighting instead of
being forced to blend everything into an undifferentiated context. This is
the mechanism that makes it *possible* to represent "the user asserted X"
separately from "I concluded X" (though §4.3 is explicit that this is only
an affordance, not a guarantee, without a training signal that exploits it).
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class GestaltMemoryBank:
    """
    A per-batch-element FIFO buffer of (vector, role_id) pairs. This is a
    plain container (not an nn.Module) because its "parameters" are just the
    stored activations from the forward pass, not learned weights -- the
    learned piece is the cross-attention reader below.
    """

    def __init__(self, capacity: int, d_model: int):
        self.capacity = capacity
        self.d_model = d_model
        self.vectors: List[torch.Tensor] = []   # each: (batch, d_model)
        self.role_ids: List[int] = []            # parallel list of role ids

    def write(self, vector: torch.Tensor, role_id: int) -> None:
        """Push a new thought vector (and its role tag) into the FIFO."""
        self.vectors.append(vector)
        self.role_ids.append(role_id)
        if len(self.vectors) > self.capacity:
            self.vectors.pop(0)
            self.role_ids.pop(0)

    def apply_grad_truncation(self, window: int) -> None:
        """
        Detach all but the trailing `window` entries in-place, implementing
        the outer-memory warmup schedule (§3.6, §5.3): unbounded forward
        reads, bounded backward credit assignment.
        """
        from utils import truncate_gradient_window
        self.vectors = truncate_gradient_window(self.vectors, window)

    def stacked(self) -> Optional[torch.Tensor]:
        """Return (batch, n_slots, d_model), or None if memory is empty."""
        if not self.vectors:
            return None
        return torch.stack(self.vectors, dim=1)

    def role_id_tensor(self, device) -> Optional[torch.Tensor]:
        if not self.role_ids:
            return None
        return torch.tensor(self.role_ids, device=device, dtype=torch.long)

    def filtered_stacked(self, role_ids_wanted: List[int]):
        """
        Return the subset of stored slots whose role id is in
        `role_ids_wanted`, stacked as (batch, n_matching, d_model). Used by
        the input lane (§4.1) to retrieve *aged input* gestalt summaries
        (typically USER-tagged) as its second context tier, distinct from
        the Reasoner's own unrestricted cross-attention into the full bank.
        Returns None if nothing matches.
        """
        matches = [v for v, r in zip(self.vectors, self.role_ids) if r in role_ids_wanted]
        if not matches:
            return None
        return torch.stack(matches, dim=1)

    def reset(self) -> None:
        self.vectors = []
        self.role_ids = []

    def __len__(self) -> int:
        return len(self.vectors)


class GestaltCrossAttentionReader(nn.Module):
    """
    Learned cross-attention reader used by both the inner HRM loop and the
    Talker to read from a GestaltMemoryBank. Adds a role-tag embedding to
    each memory slot's key/value so attention can be source-dependent.
    """

    def __init__(self, d_model: int, n_heads: int, n_roles: int, dropout: float = 0.1):
        super().__init__()
        # `d_model` here is the width of a stored thought -- d_latent in the
        # widened design (the memory holds chunk-level thoughts, not tokens). The
        # loop and the Talker both query at that same width, so this is a plain
        # single-width cross-attention.
        self.role_embed = nn.Embedding(n_roles, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, memory: GestaltMemoryBank) -> torch.Tensor:
        """
        query: (batch, d_model) or (batch, seq, d_model) -- the state doing the reading.
        Returns a tensor of the same shape as `query`, the attended-memory
        readout (added by the caller as a residual, or used directly).
        """
        single_vector = query.dim() == 2
        q = query.unsqueeze(1) if single_vector else query   # (batch, 1 or seq, d_model)

        stacked = memory.stacked()
        if stacked is None:
            # No memory yet (first thought of a document): nothing to attend to.
            return torch.zeros_like(query)

        role_ids = memory.role_id_tensor(query.device)
        kv = stacked + self.role_embed(role_ids).unsqueeze(0)  # broadcast role tag per slot
        out, _ = self.attn(self.norm_q(q), kv, kv, need_weights=False)
        return out.squeeze(1) if single_vector else out
