"""
losses.py
=========
The two training objectives (§2) plus the ACT ponder penalty (§1.1, §5.5).

1. `scaled_cosine_loss` -- the self-supervised latent loss (§2.1, §3.4):
       L(theta, theta') = k * (1 - cos(h_pred, h_target))
   The scale factor k (default 4, config.cosine_loss_k) exists because raw
   cosine distance gives vanishing gradients when the loss is already small
   -- the same saturating-gradient rationale behind label smoothing / focal
   loss elsewhere (§3.4), not something specific to latent reasoning.

2. `grounded_nll_loss` -- ordinary next-token cross-entropy on the Talker's
   output for the realized tokens of a chunk (§2.2). This is what keeps
   latents *decodable* rather than merely self-consistent (§3.7).

3. `ponder_cost_loss` -- ACT's ponder penalty (§5.5): a small cost per
   ponder step, pushing the model toward the cheapest depth that doesn't
   hurt the primary losses.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def scaled_cosine_loss(pred: torch.Tensor, target: torch.Tensor, k: float = 4.0) -> torch.Tensor:
    """
    pred, target: (batch, d_model). `target` should already be detached
    (e.g. produced by EMATargetEncoder.encode under torch.no_grad()).
    """
    cos_sim = F.cosine_similarity(pred, target, dim=-1)
    return (k * (1.0 - cos_sim)).mean()


def grounded_nll_loss(logits: torch.Tensor, targets: torch.Tensor,
                       target_mask: torch.Tensor) -> torch.Tensor:
    """
    logits:  (batch, chunk_len, vocab_size)
    targets: (batch, chunk_len) ground-truth token ids for this chunk
    target_mask: (batch, chunk_len) bool, True where the position is supervised
        (the caller decides: real tokens, plus the end-of-chunk PAD position in
        model.forward_grounded).
    """
    vocab_size = logits.shape[-1]
    loss_per_token = F.cross_entropy(
        logits.reshape(-1, vocab_size), targets.reshape(-1), reduction="none"
    ).reshape(targets.shape)
    mask_f = target_mask.float()
    return (loss_per_token * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def ponder_cost_loss(ponder_cost: torch.Tensor, weight: float) -> torch.Tensor:
    """Simple linear penalty on accumulated ACT ponder cost (§5.5)."""
    return weight * ponder_cost


def variance_regularization(z: torch.Tensor, target_std: float = 0.1, eps: float = 1e-4) -> torch.Tensor:
    """
    VICReg-style anti-collapse term. Collapse = the encoder outputs (nearly)
    the same vector for every input, i.e. per-dimension variance -> 0. This
    penalizes the batch's per-dimension standard deviation falling below
    `target_std` via a hinge, so the loss *directly* pushes back on collapse
    regardless of what the cosine-prediction term is doing. `target_std` is a
    low *safety floor* (below the encoder's natural per-dim std), so this term
    is dormant in normal operation and only activates as the latent approaches
    collapse -- it must not force a scale, only prevent the crash to zero.

    z: (N, d) a batch of latents (N valid chunks). Returns a scalar.
    """
    if z.shape[0] < 2:
        return torch.zeros((), device=z.device)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)      # (d,)
    return torch.clamp(target_std - std, min=0.0).mean()
