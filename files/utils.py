"""
utils.py
========
Small, reusable helpers shared across modules:

- `set_seed`: reproducibility.
- `truncate_gradient_window`: implements the "long forward horizon, short
  backward horizon" trick used at *two* levels in the design (§3.5 inner
  loop, §3.6 outer memory) -- detach everything except the most recent
  `window` steps so full BPTT never has to unroll the whole history.
- `PlateauDetector`: gates curriculum-stage transitions on validation-loss
  plateaus instead of fixed step counts (§5.7.2).
"""
from __future__ import annotations

import random
from collections import deque
from typing import List, Sequence

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def truncate_gradient_window(history: Sequence[torch.Tensor], window: int) -> List[torch.Tensor]:
    """
    Given a chronological list of tensors, return a new list where every entry
    *outside* the trailing `window` is detached from the autograd graph.

    Used for the outer-memory warmup credit assignment (§3.6): history =
    the gestalt bank's stored thought vectors. Detaching stored entries works
    there because all *future* reads consume the returned (detached) list.
    Only the trailing `window` entries keep gradient; older entries are
    treated as fixed context -- the asymmetric "unbounded forward reads,
    bounded backward credit assignment" pattern of §3.6.

    NOTE the window bounds the DIRECT read, not total backward depth: an
    in-window slot was written by a thought whose own graph contains ITS
    in-window memory reads, so credit chains transitively (attenuated per
    hop) back through the whole document, exactly as the design doc states
    ("distant credit still reaches back transitively through memory ...
    the activation graph spans the document"). Budget backward compute and
    activation memory for full-document depth from Stage C onward.

    NOTE: this post-hoc detach is NOT valid for truncating an already-executed
    recurrence (e.g. the inner HRM loop): detaching recorded step states does
    not cut the final state's graph, which still reaches back through every
    step. The inner loop instead cuts its carried states mid-loop -- see
    hrm_loop._TruncationSchedule.
    """
    if window <= 0:
        return [h.detach() for h in history]
    n = len(history)
    cutoff = max(0, n - window)
    return [h.detach() if i < cutoff else h for i, h in enumerate(history)]


def linear_warmup_window(step: int, total_warmup_steps: int, start: int, end: int) -> int:
    """
    Linearly grow an integer gradient window from `start` to `end` over
    `total_warmup_steps` optimizer steps. Used for both the 2->5 inner-loop
    schedule and the 1->5 outer-memory schedule (§3.5, §5.3).
    """
    if total_warmup_steps <= 0:
        return end
    frac = min(1.0, step / total_warmup_steps)
    return int(round(start + frac * (end - start)))


class PlateauDetector:
    """
    Tracks a validation loss stream and reports whether it has plateaued,
    i.e. hasn't improved by at least `min_delta` for `patience` consecutive
    checks. Used to gate curriculum stage transitions (§5.7.2) rather than
    relying on fixed iteration counts copied from the source papers.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self._best = float("inf")
        self._stale_checks = 0
        self._history: deque = deque(maxlen=patience * 4)

    def update(self, value: float) -> bool:
        """Push a new validation loss value; return True iff plateaued."""
        self._history.append(value)
        if value < self._best - self.min_delta:
            self._best = value
            self._stale_checks = 0
        else:
            self._stale_checks += 1
        return self._stale_checks >= self.patience

    def reset(self) -> None:
        self._best = float("inf")
        self._stale_checks = 0
        self._history.clear()
