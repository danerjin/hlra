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


def supervised_halt_loss(halt_logits: torch.Tensor, cos_dist: torch.Tensor,
                          supervise_mask: torch.Tensor, epsilon: float = 0.01) -> torch.Tensor:
    """
    TRM-style supervised halt gate (experiments.md #2). Replaces the soft ponder
    cost with a BCE that gives the halting head a *quality-grounded* signal: at
    each candidate depth c, the target is "halt now" iff running one more cycle
    would improve the SSL prediction by less than `epsilon`.

      halt_logits   (C, N):  the halting head's pre-sigmoid output at each of the
                             C candidate cycles, for N supervised rows. Computed
                             on a DETACHED H-state by the caller, so this loss
                             trains ONLY the halting head -- it never reshapes the
                             reasoning (the primary SSL/gen losses do that).
      cos_dist      (C, N):  per-cycle cosine distance (1 - cos) of pred_head(h_c)
                             to the EMA target, DETACHED (a label, not a gradient
                             path). Built under no_grad by the caller.
      supervise_mask (C, N): bool, True where (cycle c, row n) is a legal halt
                             point (c >= min-depth floor, row has a real t+1
                             target). Rows/cycles outside get no gradient.

    Target: halt_c = 1 when (cos_dist_c - cos_dist_{c+1}) < epsilon (the next
    cycle barely helps), else 0. The last candidate cycle has no successor, so its
    target is 1 (must stop at the cap). Self-calibrating: epsilon is in cosine-
    distance units, and the target adapts to whatever improvement curve the model
    currently produces. Returns a scalar (mean BCE over supervised entries; 0 if
    none).
    """
    C = halt_logits.shape[0]
    # marginal improvement from doing ONE more cycle; last cycle -> +inf (always halt)
    nxt = torch.cat([cos_dist[1:], torch.full_like(cos_dist[:1], float("inf"))], dim=0)
    improvement = cos_dist - nxt                       # >0 means the next cycle helps
    target = (improvement < epsilon).float()           # halt when it stops helping
    bce = F.binary_cross_entropy_with_logits(halt_logits, target, reduction="none")
    m = supervise_mask.float()
    return (bce * m).sum() / m.sum().clamp_min(1.0)


def anti_sycophancy_loss(pred_a: torch.Tensor, pred_b: torch.Tensor, target: torch.Tensor,
                         k: float = 4.0, agree_weight: float = 1.0) -> torch.Tensor:
    """
    The Layer-3 (§4.3) contrastive signal that makes the USER/SELF role tags
    *behaviorally* load-bearing instead of a mere representational affordance.

    `pred_a`, `pred_b`: the model's predicted opening-response latent under two
    contexts that differ ONLY in a USER-asserted premise (e.g. the user asserts
    X vs. asserts not-X). `target`: the role-invariant correct-answer latent
    (already detached, e.g. an EMA-target encoding of the true answer's stance).

    A sycophant lets the user's assertion move its answer, so pred_a and pred_b
    diverge and chase whichever premise the user stated. This loss penalizes
    exactly that: each variant must match the fixed truth (the two scaled-cosine
    terms) AND the two variants must agree with each other (the `agree` term) --
    i.e. the model must *discount* the user's assertion, which it can only do by
    reading the role tag. Uses the same scaled-cosine form (§3.4) as the
    predictive objective so the magnitudes are comparable.

    NOTE this is the loss the design (§4.3) calls for but had never implemented;
    the contrastive pairs it consumes are constructed in dialogue_data.py. It is
    a starting point, not a validated recipe -- the pair construction is where
    the real signal lives.
    """
    la = scaled_cosine_loss(pred_a, target, k)
    lb = scaled_cosine_loss(pred_b, target, k)
    agree = (k * (1.0 - F.cosine_similarity(pred_a, pred_b, dim=-1))).mean()
    return la + lb + agree_weight * agree


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
