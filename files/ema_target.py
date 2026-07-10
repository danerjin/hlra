"""
ema_target.py
=============
The EMA target encoder (§2.1, §3.4) used by the self-supervised latent
loss. A momentum copy of a chunk encoder whose weights are updated as
`theta' <- m * theta' + (1 - m) * theta` (no gradient through it directly),
following JEPA/BYOL/DINO-style self-distillation.

The momentum (0.98) is a self-distillation-collapse defense (§3.4): too low
and predictor/target can collapse to a trivial constant, too high and the
target is stale. It's a value the source paper's own sweep found for its
setup, not a universal constant, so it lives in config.py as a tunable.

We reuse the same chunk-pooling architecture as the online chunk encoder
(a small transformer + mean pool) so the two are directly comparable via
cosine distance in the same representation space.
"""
from __future__ import annotations

import copy
from contextlib import nullcontext

import torch
import torch.nn as nn


class ChunkEncoder(nn.Module):
    """Encodes one chunk of token ids into a single vector via a small
    bidirectional transformer + masked mean pool. Used both as the "online"
    predictor target-space encoder and (as a frozen EMA copy) as the target
    encoder for the self-supervised loss."""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int,
                 dropout: float, max_len: int, n_layers: int = 2):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, chunk_ids: torch.Tensor, chunk_mask: torch.Tensor) -> torch.Tensor:
        batch, chunk_len = chunk_ids.shape
        positions = torch.arange(chunk_len, device=chunk_ids.device).unsqueeze(0)
        x = self.token_embed(chunk_ids) + self.pos_embed(positions)

        # Guard all-pad rows: an entirely-masked key_padding_mask row makes the
        # attention softmax divide by zero -> NaN. Such rows occur for padded
        # (absent) chunks within a batch. Let them attend freely; their pooled
        # output is forced to 0 below anyway (every token is masked out of the
        # mean), so the encoder stays finite for padded chunks.
        key_padding_mask = ~chunk_mask
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked] = False
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        mask_f = chunk_mask.unsqueeze(-1).float()
        pooled = (x * mask_f).sum(1) / mask_f.sum(1).clamp_min(1.0)  # all-pad -> 0 vector
        return self.out_norm(pooled)


class EMATargetEncoder:
    """
    Wraps a frozen momentum copy of a ChunkEncoder. Not an nn.Module itself
    (so it's never accidentally included in the optimizer's parameter list);
    callers must exclude its parameters from gradient updates and instead
    call `update()` after each optimizer step.

    The target may include an EMA copy of the SSL projection head as well
    (BYOL-style): the online path is predictor(proj(encoder(x))) and the target
    path is proj_ema(encoder_ema(x')). Projecting into a separate SSL space is
    what lets the SSL objective collapse *its own head* harmlessly instead of
    flattening the shared chunk encoder that reconstruction depends on.
    """

    def __init__(self, online_encoder: ChunkEncoder, momentum: float, online_proj=None):
        self.momentum = momentum
        self.target_encoder = copy.deepcopy(online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        # Deterministic target: keep the EMA encoder in eval mode so dropout is
        # OFF, the standard BYOL/DINO choice. A stochastic target would make the
        # SSL cosine target jitter per call; nothing flips this back to train
        # (EMATargetEncoder is not an nn.Module, so model.train() never reaches it).
        self.target_encoder.eval()
        self.target_proj = None
        if online_proj is not None:
            self.target_proj = copy.deepcopy(online_proj)
            for p in self.target_proj.parameters():
                p.requires_grad_(False)
            self.target_proj.eval()

    @torch.no_grad()
    def update(self, online_encoder: ChunkEncoder, online_proj=None) -> None:
        """EMA update: theta' <- m * theta' + (1 - m) * theta (§3.4)."""
        for p_target, p_online in zip(self.target_encoder.parameters(), online_encoder.parameters()):
            p_target.mul_(self.momentum).add_(p_online.detach(), alpha=1.0 - self.momentum)
        if self.target_proj is not None and online_proj is not None:
            for p_target, p_online in zip(self.target_proj.parameters(), online_proj.parameters()):
                p_target.mul_(self.momentum).add_(p_online.detach(), alpha=1.0 - self.momentum)

    @torch.no_grad()
    def encode(self, chunk_ids: torch.Tensor, chunk_mask: torch.Tensor) -> torch.Tensor:
        # The target encoder is kept in eval() (dropout off, deterministic
        # target). Under AMP autocast, an eval-mode nn.TransformerEncoder takes
        # the fused BetterTransformer path (torch._transformer_encoder_layer_fwd),
        # which mixes autocast-cast bf16 activations with the encoder's fp32
        # weights and raises "mat1 and mat2 must have the same dtype" -- this
        # crashes forward_self_supervised the moment the SSL loss turns on
        # (Stage D) on the --amp run. Since this is a detached momentum target,
        # compute it in full precision with autocast disabled: correct, more
        # accurate, and backend-agnostic. No-op on the CPU/non-AMP path.
        dev = chunk_ids.device.type
        ctx = (torch.autocast(device_type=dev, enabled=False)
               if dev in ("cpu", "cuda") else nullcontext())
        with ctx:
            z = self.target_encoder(chunk_ids, chunk_mask)
            if self.target_proj is not None:
                z = self.target_proj(z)
        return z

    def to(self, device):
        self.target_encoder = self.target_encoder.to(device)
        if self.target_proj is not None:
            self.target_proj = self.target_proj.to(device)
        return self

    def state_dict(self):
        sd = {"encoder": self.target_encoder.state_dict()}
        if self.target_proj is not None:
            sd["proj"] = self.target_proj.state_dict()
        return sd

    def load_state_dict(self, sd):
        self.target_encoder.load_state_dict(sd["encoder"])
        if self.target_proj is not None and "proj" in sd:
            self.target_proj.load_state_dict(sd["proj"])
