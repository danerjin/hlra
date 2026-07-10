"""
decay_gate.py
=============
A per-channel **diagonal decay gate** -- the recurrent state-transition cell
used by both the fast L-module and the slow H-module (they share this
primitive but hold separate weights, see hrm_loop.py). The update run is:

    h_{n+1} = a ⊙ h_n + B·e + R(h_n, e)

where:
  - `a` is a per-channel decay in (0, 1), the diagonal of the state-transition
    matrix. It is a learned **leaky-integrator carry path**: each channel picks
    its own memory timescale. This is the S4/Mamba diagonal discretization
    (below), not the spectral-norm machinery of the Parcae paper -- see the
    honesty note.
  - `B·e` is the (RMS-normalized) injection of the current external input `e`
    (a chunk embedding, or the H-module's context, depending on the caller).
  - `R(h_n, e)` is an ordinary nonlinear residual (a small feed-forward
    sublayer) over the concatenation of state and input.

Where the stability actually comes from (honesty note)
------------------------------------------------------
`R` is an unconstrained nonlinear term, so a contractive `a` does **not** bound
this map. Boundedness at arbitrary loop depth comes entirely from MagicNorm's
hard-normalization (norm.py), which re-projects h onto the fixed-norm shell
‖h‖=√d at the exit of every L- and H-step. This gate does two real jobs:
  1. it is the linear carry path that propagates state across steps (without
     it, all state carry-over would have to route through `R`); and
  2. keeping the linear part contractive *shapes* the on-shell dynamics toward
     convergence rather than orbiting -- which is the mechanism we *hope* buys
     predictable, saturating test-time-depth scaling. That is an empirical
     claim to be measured (see profile_transition.py / the §5.5 sweep), not a
     guarantee this primitive provides. The Parcae paper's spectral-norm
     constraint on a full looped-transformer transition is the inspiration; we
     implement only the trivial diagonal case.

Negative-diagonal discretization
---------------------------------
Following the S4/Mamba family: a continuous-time negative-diagonal generator
`-softplus(theta)` is discretized with a per-channel step size `dt` via
`a = exp(-softplus(theta) * dt)`. Because `softplus(theta) * dt > 0`, each
`a` lies in `(0, 1)` for every channel by construction, for any number of
steps -- there is no "eigenvalue drifted outside the unit circle" failure mode
no matter how deep the loop runs at test time.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagonalDecayGate(nn.Module):
    """
    A single diagonal-decay recurrent update, reused for both the fast
    L-module and the slow H-module inner-loop updates (they share this
    primitive but hold separate parameters -- see hrm_loop.py).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float,
                 min_decay: float = 0.01, max_decay: float = 0.99):
        super().__init__()
        self.d_model = d_model
        self.min_decay = min_decay
        self.max_decay = max_decay

        # Raw, unconstrained parameters for the negative-diagonal generator.
        # At init (theta = log_dt = 0) the decay is exp(-softplus(0)^2) ~= 0.62
        # per channel -- comfortably inside the unit circle with moderate
        # forgetting, a stable place to begin training a fresh recurrence.
        self.theta = nn.Parameter(torch.zeros(d_model))
        # Per-channel discretization step size, also learned but kept positive.
        self.log_dt = nn.Parameter(torch.zeros(d_model))

        # Injection projection B (applied to the external input e). The
        # injection is *normalized* -- we RMS-normalize e before the linear
        # projection, which bounds the injection's contribution to the update
        # independent of the raw scale of e.
        self.B = nn.Linear(d_model, d_model, bias=False)

        # Nonlinear residual R(h, e): a small feed-forward "transformer
        # sublayer" operating on the concatenation of state and input. Its
        # inputs are bounded not by an explicit Pre-LN (norm.PreNormWrapper is
        # not used here) but by the loop's invariants: h enters hard-normalized
        # to ||h||=sqrt(d) from the previous step's exit (hrm_loop), and the
        # injected e is LayerNormed/normalized upstream. MagicNorm's hard-norm
        # half is what actually carries the stability argument in this loop.
        self.residual = nn.Sequential(
            nn.Linear(d_model * 2, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def diagonal_decay(self) -> torch.Tensor:
        """
        Returns the diagonal entries of the state transition, each guaranteed
        to lie strictly inside (min_decay, max_decay) subset of (0, 1). This is
        the elementwise spectral norm of the (diagonal) state-transition matrix.
        """
        dt = F.softplus(self.log_dt) + 1e-4
        decay = torch.exp(-F.softplus(self.theta) * dt)
        # Clamp softly into [min_decay, max_decay] for numerical headroom;
        # the exp(...) construction already guarantees (0, 1), this just
        # keeps values away from the extremes where gradients vanish.
        return decay.clamp(self.min_decay, self.max_decay)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        h: (batch, d_model) current recurrent state.
        e: (batch, d_model) external input/injection for this step.
        Returns the next state h_{n+1}, same shape as h.
        """
        a = self.diagonal_decay()                       # (d_model,)
        e_norm = e / (e.norm(dim=-1, keepdim=True) + 1e-6) * (self.d_model ** 0.5)
        injection = self.B(e_norm)                       # (batch, d_model)
        nonlinear = self.residual(torch.cat([h, e], dim=-1))  # (batch, d_model)
        h_next = a.unsqueeze(0) * h + injection + nonlinear
        return h_next
