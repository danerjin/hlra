"""
ema_target.py
=============
The EMA target encoder (§2.1, §3.4) used by the self-supervised latent
loss. A momentum copy of a chunk encoder whose weights are updated as
`theta' <- m * theta' + (1 - m) * theta` (no gradient through it directly),
following JEPA/BYOL/DINO-style self-distillation.

The momentum is a self-distillation-collapse defense (§3.4): too low and
predictor/target can collapse to a trivial constant, too high and the target
is stale. The source paper's sweep found 0.98 for its setup; this project
raised it to 0.996 (config.ema_momentum) because a faster-moving target lets a
small model chase it into a collapsed latent. It's not a universal constant, so
it lives in config.py as a tunable (re-tune per scale).

We reuse the same chunk-pooling architecture as the online chunk encoder
(a small transformer + mean pool) so the two are directly comparable via
cosine distance in the same representation space.
"""
from __future__ import annotations

import copy
from contextlib import nullcontext

import torch
import torch.nn as nn

from config import ArchConfig
from modern import ModernEncoder, make_norm


class ChunkEncoder(nn.Module):
    """Encodes one chunk of token ids into a single vector via a small
    bidirectional transformer + masked mean pool. Used both as the "online"
    predictor target-space encoder and (as a frozen EMA copy) as the target
    encoder for the self-supervised loss."""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, d_ff: int,
                 dropout: float, max_len: int, n_layers: int = 2,
                 d_latent: int | None = None, arch: ArchConfig | None = None):
        super().__init__()
        # Tokens are looked up at the word-level width d_model, then lifted into
        # the (possibly wider) thought width d_latent BEFORE the transformer body,
        # so the pooled chunk latent genuinely uses the full d_latent capacity --
        # projecting only after a d_model mean-pool would confine the latent to a
        # d_model subspace and defeat the point (the chunk-decode bottleneck).
        # d_latent == d_model makes in_proj an Identity and every width below
        # equal to d_model, i.e. the original single-width encoder exactly.
        d_latent = d_model if d_latent is None else d_latent
        self.arch = arch = arch if arch is not None else ArchConfig()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.in_proj = nn.Identity() if d_latent == d_model else nn.Linear(d_model, d_latent)
        # The encoder body runs at d_latent, so RoPE here rotates d_latent//n_heads
        # per head (the __post_init__ evenness check covers this width too). Learned
        # position table exists only when RoPE is off.
        self.pos_embed = None if arch.rope else nn.Embedding(max_len, d_latent)
        if arch.is_legacy:
            layer = nn.TransformerEncoderLayer(
                d_model=d_latent, nhead=n_heads, dim_feedforward=d_ff,
                dropout=dropout, batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        else:
            self.encoder = ModernEncoder(d_latent, n_heads, d_ff, dropout, n_layers,
                                         arch, max_len=max_len)
        self.out_norm = make_norm(arch, d_latent)

    def forward(self, chunk_ids: torch.Tensor, chunk_mask: torch.Tensor) -> torch.Tensor:
        batch, chunk_len = chunk_ids.shape
        positions = torch.arange(chunk_len, device=chunk_ids.device).unsqueeze(0)
        x = self.in_proj(self.token_embed(chunk_ids))
        if self.pos_embed is not None:                 # learned positions (non-RoPE)
            x = x + self.pos_embed(positions)

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
    call `update()` after each optimizer step. `encode(x')` is the stop-grad
    target for the self-supervised loss: the EMA target encoder's latent of the
    next chunk (encoder space -- the space the HRM loop is injected with, and
    the space model.pred_head predicts into).
    """

    def __init__(self, online_encoder: ChunkEncoder, momentum: float):
        self.momentum = momentum
        self.target_encoder = copy.deepcopy(online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        # Deterministic target: keep the EMA encoder in eval mode so dropout is
        # OFF, the standard BYOL/DINO choice. A stochastic target would make the
        # SSL cosine target jitter per call; nothing flips this back to train
        # (EMATargetEncoder is not an nn.Module, so model.train() never reaches it).
        self.target_encoder.eval()

    @torch.no_grad()
    def update(self, online_encoder: ChunkEncoder) -> None:
        """EMA update: theta' <- m * theta' + (1 - m) * theta (§3.4)."""
        for p_target, p_online in zip(self.target_encoder.parameters(), online_encoder.parameters()):
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
            return self.target_encoder(chunk_ids, chunk_mask)

    def to(self, device):
        self.target_encoder = self.target_encoder.to(device)
        return self

    def state_dict(self):
        return {"encoder": self.target_encoder.state_dict()}

    def load_state_dict(self, sd):
        self.target_encoder.load_state_dict(sd["encoder"])
