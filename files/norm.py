"""
norm.py
=======
MagicNorm (§0, §3.3): two pieces of normalization discipline used together
inside every recurrent module (the L-module and the H-module):

1. `PreNormWrapper` -- ordinary Pre-LN: normalize *before* a sublayer, add
   the sublayer output as a residual. This is the "PreNorm internally" half.

2. `hard_normalize` -- a *hard* normalization applied once, at the exit of
   each recurrent module's forward pass (i.e. after the L-module or
   H-module produces its updated state), projecting the state back onto a
   fixed-norm shell (norm = sqrt(d_model)). This bounds forward variance
   across the *forward* horizon N even though gradients are only ever
   truncated to the last K steps -- MagicNorm's stability argument is about
   training dynamics under truncated BPTT (§3.3). This hard-norm is *also*
   what bounds the state at arbitrary loop depth; the diagonal decay gate
   (decay_gate.py) does not -- it shapes the on-shell dynamics, it does not
   bound them. Keeping both is deliberate, not redundant.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PreNormWrapper(nn.Module):
    """Wraps `sublayer` with Pre-LN + residual: x + sublayer(norm(x))."""

    def __init__(self, d_model: int, sublayer: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.sublayer = sublayer

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x + self.sublayer(self.norm(x), *args, **kwargs)


def hard_normalize(x: torch.Tensor, target_norm: float | None = None) -> torch.Tensor:
    """
    Rescale the last dimension of `x` to have exactly `target_norm` (default
    sqrt(d_model)) L2 norm. Applied at the *exit* of each recurrent module,
    i.e. once per L-step and once per H-step, not inside every sublayer.
    """
    d_model = x.shape[-1]
    if target_norm is None:
        target_norm = d_model ** 0.5
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return x / norm * target_norm
