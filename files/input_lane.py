"""
input_lane.py
=============
Implements §4.1 and §4.2's "two lanes feeding the memory bank, not one".

Input lane: raw tokens (this turn, within `recent_token_window`) plus
aged-out gestalt summaries (prior turns/context), encoded by a stack that
*never writes into the Reasoner's H/L recurrent state directly* -- it only
ever gets to be cross-attended to. This is a stronger separation than
HRM-Text's PrefixLM mask alone provides (§4.2): masking changes attention
patterns but both input and output still flow through the same weights and
can collapse into the same recurrent state. Here the input lane's output is
architecturally restricted to being keys/values for cross-attention -- it
has no path to being written as `h_state` or `l_state` in hrm_loop.py.

Full bidirectional attention is used across the raw-token portion (this is
what HRM-Text's PrefixLM mask does for instruction tokens, §0), since input
is a fixed, externally-given artifact with no compounding-error process to
protect against (§4.1) -- unlike the Reasoner's own self-generation.

Self lane: the H/L thought loop's own recurrent state (hrm_loop.py) is the
*only* thing allowed to write into "what I currently believe" -- it is
represented directly in model.py by simply never routing input-lane output
into HRMInnerLoop's h_state/l_state initial conditions across turns; only
the gestalt memory (self-authored thoughts) persists that way.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class InputLaneEncoder(nn.Module):
    """
    A small bidirectional transformer encoder over the concatenation of
    (a) raw recent tokens and (b) aged gestalt summary vectors, projected to
    a common width. Its output is intended purely as cross-attention
    key/value context -- callers must not use it to initialize or overwrite
    any recurrent state.
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int,
                 dropout: float, n_layers: int, max_len: int):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        # A "type" embedding distinguishing raw-token slots from aged-gestalt
        # slots, since they arrive from different modalities (token id vs.
        # already-pooled vector) even though both end up at width d_model.
        self.type_embed = nn.Embedding(2, d_model)  # 0 = raw token, 1 = aged gestalt

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,  # Pre-LN, MagicNorm-style
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(
        self,
        raw_token_ids: torch.Tensor,          # (batch, n_raw) recent raw tokens
        raw_mask: torch.Tensor,               # (batch, n_raw) bool, True = real token
        aged_gestalts: torch.Tensor | None,    # (batch, n_aged, d_model) or None
        aged_mask: torch.Tensor | None,        # (batch, n_aged) bool or None
    ):
        """
        Returns a pair:
          kv:   (batch, n_raw + n_aged, d_model) read-only key/value context
                for cross-attention by the Reasoner (hrm_loop.py) and the
                Talker (for exact quoting / fidelity, §4.3.2);
          mask: (batch, n_raw + n_aged) bool, True where the slot is a real
                token/gestalt -- readers must pass it as their cross-attention
                key_padding_mask (inverted) so pad slots are never attended.
        """
        batch, n_raw = raw_token_ids.shape
        device = raw_token_ids.device
        positions = torch.arange(n_raw, device=device).unsqueeze(0)
        raw_embeds = (
            self.token_embed(raw_token_ids)
            + self.pos_embed(positions)
            + self.type_embed(torch.zeros_like(raw_token_ids))
        )

        if aged_gestalts is not None and aged_gestalts.shape[1] > 0:
            n_aged = aged_gestalts.shape[1]
            aged_type = self.type_embed(
                torch.ones(batch, n_aged, dtype=torch.long, device=device)
            )
            aged_embeds = aged_gestalts + aged_type
            combined = torch.cat([raw_embeds, aged_embeds], dim=1)
            combined_mask = torch.cat([raw_mask, aged_mask], dim=1)
        else:
            combined = raw_embeds
            combined_mask = raw_mask

        # TransformerEncoder expects True = *ignore* for its padding mask.
        key_padding_mask = ~combined_mask
        # Guard fully-masked rows (e.g. a turn whose recent-token window happens
        # to hold no real tokens): an all-True mask row NaNs the attention
        # softmax. Let such rows attend freely, then zero their output so they
        # contribute nothing as cross-attention context downstream.
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked] = False
        out = self.encoder(combined, src_key_padding_mask=key_padding_mask)
        if all_masked.any():
            out = torch.where(all_masked.view(-1, 1, 1), torch.zeros_like(out), out)
        return out, combined_mask
