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
                          supervise_mask: torch.Tensor, epsilon: float = 0.01,
                          target_mode: str = "marginal") -> torch.Tensor:
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

    Target (selected by `target_mode`):
      "marginal"      (default) halt_c = 1 when (cos_dist_c - cos_dist_{c+1}) <
                      epsilon (the next cycle barely helps), else 0. The last
                      candidate cycle has no successor -> target 1 (stop at cap).
                      Simple but halts early on a gently-but-steadily improving
                      curve (the per-step slope is always small), which is the
                      opposite of "think harder on hard chunks".
      "best_relative" halt_c = 1 when cos_dist_c is within epsilon of the BEST
                      cos_dist this chunk reaches across its LEGAL (supervised)
                      cycles -- "keep going until you're about as good as you'll
                      get". Robust to gentle slopes; epsilon is a gap-to-best
                      tolerance. On a flat curve best ~ every cycle -> target ~ all
                      1 -> halt at the min-depth floor (no wasted compute), so it
                      does not regress the inert-depth case.
    Both are self-calibrating (epsilon in cosine-distance units, target adapts to
    the model's current curve). Returns a scalar (mean BCE over supervised
    entries; 0 if none).
    """
    C = halt_logits.shape[0]
    if target_mode == "best_relative":
        # Best (min) cos_dist over LEGAL cycles only, per row: a sub-floor cycle
        # can't be selected, so it must not define "best" (else legal cycles could
        # all read as "far from best" -> never halt -> forced to the cap).
        masked_cd = cos_dist.masked_fill(~supervise_mask, float("inf"))
        best = masked_cd.min(dim=0, keepdim=True).values          # (1, N)
        target = (cos_dist <= best + epsilon).float()             # within epsilon of best -> halt
    else:  # "marginal" -- byte-identical to the original
        # marginal improvement from doing ONE more cycle; last cycle -> +inf (always halt)
        nxt = torch.cat([cos_dist[1:], torch.full_like(cos_dist[:1], float("inf"))], dim=0)
        improvement = cos_dist - nxt                   # >0 means the next cycle helps
        target = (improvement < epsilon).float()       # halt when it stops helping
    bce = F.binary_cross_entropy_with_logits(halt_logits, target, reduction="none")
    m = supervise_mask.float()
    return (bce * m).sum() / m.sum().clamp_min(1.0)


def turn_end_loss(end_logits: torch.Tensor, end_target: torch.Tensor,
                  supervise_mask: torch.Tensor):
    """
    Stage-F learned turn-end (STAGE_F.md §2.1). The token-level end-of-chunk stop
    (`model._talker_target_mask`'s supervised PAD, §19.2) lets the Talker end a
    CHUNK; nothing lets the model end a TURN. Without this the reply length is a
    caller-supplied constant (`DialogueSession.reply(max_chunks=...)`), so replies
    run on or get cut mid-thought regardless of whether the answer finished.

    This is a per-thought BCE: after the loop has ingested response chunk t and
    formed the thought h_t, predict "the turn ends here" -- i.e. there is no
    chunk t+1.

      end_logits    (N,): the end head's pre-sigmoid output for N supervised
                          thoughts. Read off a DETACHED h_t by default (the
                          `supervised_halt_loss` convention), so the BCE trains
                          only the head and never reshapes the reasoning; pass
                          `end_grad=True` in forward_dialogue to let it through.
      end_target    (N,): 1.0 where the turn ends at this thought, else 0.0.
      supervise_mask (N,): bool, True where the label is TRUSTWORTHY.

    WHY supervise_mask IS NOT JUST "is this chunk real":
    the label is free -- it is `resp_mask` -- but it is NOT unconditionally
    correct. `chunker.chunk_batch` caps a response at `max_chunks_per_doc`, so a
    response that FILLS all M slots is indistinguishable from one that was
    truncated at M. Its final chunk's "the turn ends here" label is therefore
    unknown, not True. Training on it naively teaches "every turn ends after
    exactly M chunks". The caller masks exactly those final positions out; every
    non-final position ("a chunk follows" = 0) is correct either way and is kept.

    Returns (loss, n_supervised). Loss is the mask-mean BCE, 0.0 when nothing is
    supervised (an all-truncated batch).

    `n_supervised` is NOT a health metric -- it counts negatives too. The masking
    above drops a filled row's single POSITIVE while keeping all of its negatives,
    so a batch of long responses can report a large n_supervised with zero
    positives, and BCE/accuracy both look perfect while the head learns "never
    end". `forward_dialogue` therefore also returns `end_pos` (surviving
    positives), which is the number to watch.
    """
    m = supervise_mask.float()
    n = m.sum()
    bce = F.binary_cross_entropy_with_logits(end_logits, end_target, reduction="none")
    return (bce * m).sum() / n.clamp_min(1.0), n


@torch.no_grad()
def turn_end_accuracy(end_logits: torch.Tensor, end_target: torch.Tensor,
                      supervise_mask: torch.Tensor) -> torch.Tensor:
    """Fraction of SUPERVISED thoughts whose P(end) > 0.5 matches the label.

    Exists because of the anti-sycophancy lesson (`antisycophancy_trust_gate_note.md`
    #1): an auxiliary head can be wired correctly, receive gradient, and still not
    learn the intended behavior. This turns "is the end head working?" from a hope
    into a logged number. Note the label is heavily imbalanced (one 'end' per
    turn), so a head that always predicts "continue" scores ~1-1/M -- read this
    next to the BCE, not alone.
    """
    m = supervise_mask.float()
    correct = ((end_logits > 0).float() == end_target).float()
    return (correct * m).sum() / m.sum().clamp_min(1.0)


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


def trust_prior_loss(trust_by_role: torch.Tensor, user_idx: int, self_idx: int,
                     margin: float = 0.1, floor: float = 0.2) -> torch.Tensor:
    """
    Explicit provenance prior on the trust gate (review #2, option 3). The
    anti-sycophancy loss is *supposed* to drive trust(USER) down, but as an
    emergent signal it barely reaches the gate (SGD prefers the response
    seed/encoder; see antisycophancy_trust_gate_note.md). This makes "distrust a
    user assertion more than your own conclusion" a FIRST-CLASS objective: a
    one-sided hinge that pushes trust(USER) at least `margin` below trust(SELF).

    `trust_by_role`: the per-role trust scalars, (n_roles,), from
    `reader.trust_by_role(...)` (the mean over dims for the vector gate), kept
    IN-GRAPH so this trains `trust_proj` directly. Relative, not absolute: it
    anchors to the model's own SELF trust (which the SFT read keeps high), so it
    encodes a prior ("trust yourself more than the user") without dictating an
    arbitrary absolute level. `margin` is how much less USER is trusted -- a floor
    on the GAP, not a target, so it does not keep pushing once the gap is met.

    `floor` is the lower-floor SAFETY: the gap hinge is one-sided (no restoring
    force), so a too-strong prior can crush trust(USER) toward ~0 -- a fully
    zeroed slot means the loop cannot read the user's assertion AT ALL, past the
    intended "attend but discount." A second hinge keeps trust(USER) >= `floor`
    so the slot stays noticeable. Set floor=0.0 for the pure note version.
    Returns a scalar. (For the vector gate this regularizes the per-role MEAN;
    combined with the anti-sycophancy + topic-preservation pressure the optimizer
    tends to lower a polarity SUBSPACE rather than uniformly -- a subspace-targeted
    prior is the finer, unimplemented refinement.)
    """
    trust_user = trust_by_role[user_idx]
    gap = trust_by_role[self_idx] - trust_user               # want gap >= margin
    below = torch.clamp(margin - gap, min=0.0)               # push USER margin below SELF
    collapse = torch.clamp(floor - trust_user, min=0.0)      # but keep USER >= floor (noticeable)
    return below + collapse
