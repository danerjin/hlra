"""
curriculum.py
=============
Implements §5's staged training curriculum. Nothing in this architecture
can be trained end-to-end from a random init (§5.0): the Talker needs good
latents, latents need Talker feedback to stay decodable, the inner loop
can't be safely deepened until it's near a stable fixed point, the outer
memory gradient is the same problem one level up, and ACT needs a signal
that already reflects "more compute helped" before it can learn anything.

Stages (§5, restructured -- notes §27), in order:

  A. Autoencoder codec only (encoder -> Talker, no loop). Grounds the encoder +
     Talker and makes the EMA target meaningful.
  B. Turn on the HRM loop + the on-loop SSL predictor (fixed depth, inner-loop
     grad warmup 2->5). Memory writes detached (no cross-thought grad yet).
  C. Un-detach the gestalt memory, with its own 1->5 truncation warmup
     (cross-thought reasoning -- the loop reads its accumulating memory).
  D. Turn on adaptive depth (ACT).
  E. Consolidation (full config, extra budget).
  F. Chatbot fine-tuning: two-lane input/self separation, cross-turn memory.

The autoencoder (reconstruction) anchor runs EVERY stage; the on-loop SSL
predictor runs from B onward. `val_loss` is the autoencoder reconstruction.

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
    # Token-grounded consolidation (train/serve exposure fix): per-stage multiplier
    # on the token-grounding loss. The Talker is trained (Stage A/B) to decode REAL
    # encoder latents, but at generation it decodes the loop's PREDICTED latents,
    # which are only ~0.5-cos aligned (the probe's train/serve gap). This term
    # decodes the PREDICTED latent through the Talker vs the real next tokens, so the
    # Talker learns to decode imperfect predictions. Effective weight = this *
    # train_cfg.ssl_token_weight (0 by default => byte-identical; --pred-token-weight
    # enables it). Gated to Stage D+ in loss_plan() so it grounds on mature, not noisy,
    # predictions.
    token_ground_weight: float = 0.0


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
        self._skip_zero_budget_stages()

    def _skip_zero_budget_stages(self) -> None:
        """
        Under fixed-budget gating, a stage whose budget is 0 should be SKIPPED,
        not trained for one stray step: advance_step increments step_in_stage
        before checking the budget, and the trainer reads the stage's flags at
        the top of the loop -- so without this, `--stage-steps 0,...` (e.g.
        "relaunch skipping A") silently trained one step under the skipped
        stage's flags. The last stage (F, budget 0 by convention) is exempt;
        the trainer breaks on reaching it.
        """
        stage_steps = getattr(self.train_cfg, "stage_steps", None)
        if stage_steps is None:
            return
        while not self.is_last_stage() and stage_steps[self.stage_idx] == 0:
            self.stage_idx += 1
            self.step_in_stage = 0

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
                self._skip_zero_budget_stages()
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
        # The plateau detector is part of the gating state: without it a
        # resumed plateau-gated run resets _best/_stale_checks and delays the
        # next transition. (Fixed-budget runs never consult it.)
        return {"stage_idx": self.stage_idx, "step_in_stage": self.step_in_stage,
                "plateau_best": self.detector._best,
                "plateau_stale": self.detector._stale_checks}

    def load_state_dict(self, sd):
        self.stage_idx = sd["stage_idx"]
        self.step_in_stage = sd["step_in_stage"]
        # .get: checkpoints from before these keys existed resume fine.
        self.detector._best = sd.get("plateau_best", float("inf"))
        self.detector._stale_checks = sd.get("plateau_stale", 0)

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
            # Autoencoder codec only (encoder + Talker). No loop, no SSL -- the
            # loop flags are irrelevant here because SSL (the only loop user) is
            # off. Grounds the encoder + Talker and makes the EMA target
            # meaningful before prediction turns on (notes §27).
            return StageFlags(use_hrm_loop=False, detach_memory=True,
                               inner_loop_grad_window=0, memory_grad_window=0,
                               use_act=False, use_input_lanes=False)

        # Stage B onward: the loop + on-loop SSL are on. The SSL loop's inner grad
        # window warms 2->5 (§3.5) over this stage's own duration.
        inner_window = linear_warmup_window(
            self.step_in_stage, self._stage_budget(),
            cfg.inner_loop_grad_window_start, cfg.inner_loop_grad_window_end,
        )

        if stage == Stage.B:
            # Loop + SSL, fixed depth, memory writes DETACHED (loop reads the
            # accumulating memory but no cross-thought gradient yet -- isolate
            # "is the single-thought prediction loop stable").
            return StageFlags(use_hrm_loop=True, detach_memory=True,
                               inner_loop_grad_window=inner_window, memory_grad_window=0,
                               use_act=False, use_input_lanes=False)

        # Stage C onward: memory is un-detached, with its own 1->5 warmup (§3.6,
        # §5.3) -- cross-thought credit through the gestalt memory. Inner window
        # already at max (deepened first, §5.3's "staggering, not simultaneity").
        memory_window = linear_warmup_window(
            self.step_in_stage, self._stage_budget(),
            cfg.memory_grad_window_start, cfg.memory_grad_window_end,
        )
        inner_window = cfg.inner_loop_grad_window_end

        if stage == Stage.C:
            return StageFlags(use_hrm_loop=True, detach_memory=False,
                               inner_loop_grad_window=inner_window, memory_grad_window=memory_window,
                               use_act=False, use_input_lanes=False)

        if stage == Stage.D:
            # ACT: adaptive inner-loop depth (§5.5), once fixed-depth prediction is stable.
            return StageFlags(use_hrm_loop=True, detach_memory=False,
                               inner_loop_grad_window=inner_window,
                               memory_grad_window=cfg.memory_grad_window_end,
                               use_act=True, use_input_lanes=False)

        if stage == Stage.E:
            # Full config, extra consolidation budget (same flags as D).
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
        if stage == Stage.A:
            # Autoencoder codec ONLY -- ground the encoder + Talker (and make the
            # EMA target meaningful) before the predictive SSL turns on. This is
            # the §5.4 "don't pretrain the target against meaningless latents"
            # logic, now satisfied by construction (notes §27).
            return StageLossPlan(use_grounded_loss=True, use_self_supervised_loss=False,
                                  grounded_loss_weight=1.0, self_supervised_loss_weight=0.0)

        # Stage B onward: the always-on autoencoder anchor + the on-loop SSL
        # predictor (which trains the loop + memory to reason forward). The
        # autoencoder is cheap (codec, parallel) so it runs every step; the SSL is
        # the sequential/expensive one now.
        #
        # Token-grounded prediction from Stage B (notes 2026-07-21): decode the loop's
        # PREDICTED latent through the Talker vs the real next tokens
        # (StageLossPlan.token_ground_weight). Originally gated to D+ ("ground on mature
        # predictions"), but the foundation collapsed as a LATE-TRAINING attractor, and
        # this term's gradient into pred_head is centroid-PROOF (a constant prediction
        # decodes to generic tokens => high token NLL). So it must run DURING the whole
        # collapse-prone window as a PREVENTATIVE, not arrive in D as a fix -- if the
        # attractor forms in B/C, a D+ gate is too late. On from B here. The cost
        # (grounding the Talker on noisier early predictions) is accepted; the codec's
        # reconstruction anchor still trains it on real latents in parallel. Effective
        # only when train_cfg.ssl_token_weight > 0 (default 0 => byte-identical A-E).
        token_ground = 1.0 if stage.value >= Stage.B.value else 0.0
        return StageLossPlan(use_grounded_loss=True, use_self_supervised_loss=True,
                              grounded_loss_weight=1.0, self_supervised_loss_weight=1.0,
                              token_ground_weight=token_ground)
