"""
curriculum.py
=============
Implements §5's staged training curriculum. Nothing in this architecture
can be trained end-to-end from a random init (§5.0): the Talker needs good
latents, latents need Talker feedback to stay decodable, the inner loop
can't be safely deepened until it's near a stable fixed point, the outer
memory gradient is the same problem one level up, and ACT needs a signal
that already reflects "more compute helped" before it can learn anything.

Stages (§5.1-§5.6), in order:

  A. Ground the Talker on a shallow, fixed Reasoner (no loop, no memory grad).
  B. Turn on the inner HRM loop, fixed depth, memory writes detached.
  C. Un-detach the gestalt memory, with its own truncation warmup.
  D. Bring in the self-supervised JEPA loss, alongside the grounded loss.
  E. Turn on adaptive depth (ACT).
  F. Chatbot fine-tuning: two-lane input/self separation, cross-turn memory.

Stage transitions are gated on a validation-loss plateau (§5.7.2), not
fixed step counts -- `Curriculum.step()` is called once per training step
with the latest validation loss and returns whether a stage transition
just happened.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from model import StageFlags
from utils import PlateauDetector, linear_warmup_window


class Stage(Enum):
    A = auto()
    B = auto()
    C = auto()
    D = auto()
    E = auto()
    F = auto()


# Stages are visited strictly in this order (§5.0: "each stage's stability
# as the precondition for turning on the next mechanism").
_STAGE_ORDER = [Stage.A, Stage.B, Stage.C, Stage.D, Stage.E, Stage.F]


@dataclass
class StageLossPlan:
    """What losses are active/weighted at a given stage."""
    use_grounded_loss: bool = True
    use_self_supervised_loss: bool = False
    grounded_loss_weight: float = 1.0
    self_supervised_loss_weight: float = 0.0


class Curriculum:
    """
    Owns the current stage, the plateau detector gating transitions, and
    the per-stage `StageFlags` / `StageLossPlan` the training loop and
    model.forward_grounded consume.
    """

    def __init__(self, model_cfg, train_cfg):
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.stage_idx = 0
        self.step_in_stage = 0
        self.detector = PlateauDetector(train_cfg.plateau_patience, train_cfg.plateau_min_delta)

    @property
    def stage(self) -> Stage:
        return _STAGE_ORDER[self.stage_idx]

    def is_last_stage(self) -> bool:
        return self.stage_idx == len(_STAGE_ORDER) - 1

    def advance_step(self, val_loss: Optional[float]) -> bool:
        """
        Call once per training step (or per validation check). Returns True
        iff a stage transition just occurred. `val_loss` may be None on
        steps where validation wasn't run; the plateau detector is only fed
        real values.

        Two gating modes:
          * fixed budget -- if train_cfg.stage_steps is set (a per-stage tuple
            of optimizer-step counts), advance when the current stage's budget
            is spent. Predictable, the right choice for long runs.
          * plateau -- otherwise advance on a validation-loss plateau (§5.7.2).
        """
        self.step_in_stage += 1
        if self.is_last_stage():
            return False

        stage_steps = getattr(self.train_cfg, "stage_steps", None)
        if stage_steps is not None:
            if self.step_in_stage >= stage_steps[self.stage_idx]:
                self.stage_idx += 1
                self.step_in_stage = 0
                self.detector.reset()
                return True
            return False

        if val_loss is None:
            return False
        plateaued = self.detector.update(val_loss)
        # Require a minimum dwell time in each stage so a lucky early
        # plateau reading doesn't skip a stage before it's had a chance to
        # do anything (a practical safeguard on top of the plateau gate).
        min_dwell = self.train_cfg.plateau_patience
        if plateaued and self.step_in_stage >= min_dwell:
            self.stage_idx += 1
            self.step_in_stage = 0
            self.detector.reset()
            return True
        return False

    def state_dict(self):
        return {"stage_idx": self.stage_idx, "step_in_stage": self.step_in_stage}

    def load_state_dict(self, sd):
        self.stage_idx = sd["stage_idx"]
        self.step_in_stage = sd["step_in_stage"]

    def _stage_budget(self) -> int:
        """
        How many optimizer steps the *current* stage lasts -- the horizon the
        §3.5/§5.3 warmup windows ramp over ("over this stage's own duration").
        Under fixed-budget gating that's the stage's own step budget; under
        plateau gating there is no known duration, so max_steps_per_stage is
        the stand-in.
        """
        stage_steps = getattr(self.train_cfg, "stage_steps", None)
        if stage_steps is not None:
            return stage_steps[self.stage_idx]
        return self.train_cfg.max_steps_per_stage

    # ------------------------------------------------------------------
    def stage_flags(self) -> StageFlags:
        cfg = self.model_cfg
        stage = self.stage

        if stage == Stage.A:
            return StageFlags(use_hrm_loop=False, detach_memory=True,
                               inner_loop_grad_window=0, memory_grad_window=0,
                               use_act=False, use_input_lanes=False)

        # Stage B onward: inner loop is on. Its grad window warms up 2->5
        # (§3.5) over this stage's own duration.
        inner_window = linear_warmup_window(
            self.step_in_stage, self._stage_budget(),
            cfg.inner_loop_grad_window_start, cfg.inner_loop_grad_window_end,
        )

        if stage == Stage.B:
            return StageFlags(use_hrm_loop=True, detach_memory=True,
                               inner_loop_grad_window=inner_window, memory_grad_window=0,
                               use_act=False, use_input_lanes=False)

        # Stage C onward: memory is un-detached, with its own warmup (§3.6, §5.3).
        # The inner-loop schedule is deepened first (per §5.3's "staggering, not
        # simultaneity"), so by Stage C the inner window is already at its max.
        memory_window = linear_warmup_window(
            self.step_in_stage, self._stage_budget(),
            cfg.memory_grad_window_start, cfg.memory_grad_window_end,
        )
        inner_window = cfg.inner_loop_grad_window_end  # already deepened, held fixed from here on

        if stage == Stage.C:
            return StageFlags(use_hrm_loop=True, detach_memory=False,
                               inner_loop_grad_window=inner_window, memory_grad_window=memory_window,
                               use_act=False, use_input_lanes=False)

        if stage == Stage.D:
            # Self-supervised loss turns on (loss_plan below), no new model behavior flags.
            return StageFlags(use_hrm_loop=True, detach_memory=False,
                               inner_loop_grad_window=inner_window,
                               memory_grad_window=cfg.memory_grad_window_end,
                               use_act=False, use_input_lanes=False)

        if stage == Stage.E:
            return StageFlags(use_hrm_loop=True, detach_memory=False,
                               inner_loop_grad_window=inner_window,
                               memory_grad_window=cfg.memory_grad_window_end,
                               use_act=True, use_input_lanes=False)

        # Stage F: chatbot fine-tuning, two-lane separation turned on.
        return StageFlags(use_hrm_loop=True, detach_memory=False,
                           inner_loop_grad_window=inner_window,
                           memory_grad_window=cfg.memory_grad_window_end,
                           use_act=True, use_input_lanes=True)

    def loss_plan(self) -> StageLossPlan:
        stage = self.stage
        if stage in (Stage.A, Stage.B, Stage.C):
            # Grounded loss only -- self-supervised loss deliberately withheld
            # until Stage D (§5.4: reversing JEPA-Reasoner's own phase order,
            # so the EMA target isn't pretrained against meaningless latents).
            return StageLossPlan(use_grounded_loss=True, use_self_supervised_loss=False,
                                  grounded_loss_weight=1.0, self_supervised_loss_weight=0.0)

        # From Stage D onward both losses run together. The grounded loss is
        # inherently sequential and expensive (§5.7.1), so its *frequency*
        # (handled by train.py, not here) is thinned relative to the cheap,
        # parallel self-supervised loss -- but never below a floor, to avoid
        # the ungrounded-drift failure mode §3.7 warns about.
        return StageLossPlan(use_grounded_loss=True, use_self_supervised_loss=True,
                              grounded_loss_weight=1.0, self_supervised_loss_weight=1.0)
