"""
hrm_loop.py
===========
The "inner HRM loop" (§1.1): produces a single thought vector via bounded
recurrent deliberation, structurally identical to HRM-Text's L/H split:

  - L-module (fast): several inner steps of local refinement per H-update.
  - H-module (slow): updates once per group of L-steps, carrying the
    "strategic" state forward -- this is the role Thought Gestalt's
    sentence-vector plays, generalized to arbitrary thought-chunks.

Both modules are diagonal-decay-gated recurrences (decay_gate.py) with
MagicNorm's hard normalization applied at the exit of every L-step and every
H-step (norm.py) -- that hard-norm is what bounds the state at any depth, not
the decay gate. The L:H ratio (3:1) is an empirical HRM-Text hyperparameter,
*not* derived (§3.2) -- once ACT is turned on (Stage D, §5.5) the model can
learn to spend a different number of fast/slow steps per thought; before
that it's a fixed schedule from the config.

Credit-assignment truncation (§3.5): only the trailing `grad_window` L/H
steps of a thought carry gradient; earlier steps (and the raw H/L-state
chain carried in from previous thoughts) are treated as fixed context. The
cut MUST be applied to the carried states *during* the loop
(`_TruncationSchedule`): detaching a recorded history after the fact does
nothing, because the final state's autograd graph still reaches back
through every step. Cross-thought credit assignment flows through the
gestalt memory (whose own truncation window the caller controls), not
through the raw state chain. The warmup window grows from
`inner_loop_grad_window_start` to `..._end` over training (curriculum.py).

Adaptive depth / ACT (§1.1, §5.5): a small halting network looks at the
current H-state and decides, per thought, whether to run another
L-group-then-H-update cycle or stop. A filler word gets a shallow pass; a
load-bearing inference step gets many inner iterations. This is only turned
on in Stage D, once fixed-depth dynamics are already stable.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from decay_gate import DiagonalDecayGate
from norm import hard_normalize
from gestalt_memory import GestaltCrossAttentionReader, GestaltMemoryBank


class _TruncationSchedule:
    """
    Truncated BPTT for the inner loop (§3.5): cut the autograd graph so at
    most the trailing `window` L/H steps carry gradient back from the
    returned thought. The cut is applied to the carried (h, l) states at a
    step boundary, which is the only thing that actually limits backward
    depth -- detaching a recorded step history after the loop has run is a
    no-op, since the final state's graph still reaches through every step.

      * fixed depth (`total_steps` known): a single cut exactly `window`
        steps before the end -- HRM-Text's "backprop through only the final
        K steps", exactly.
      * ACT (`total_steps` unknown until halting): a rolling cut every
        `window` steps, bounding the backward horizon to at most `window`.

    Because the entering h/l states (the previous thought's) sit before the
    cut, this also severs the raw cross-thought state chain, leaving the
    gestalt memory -- with its own truncation window -- as the one
    cross-thought gradient path, per §3.6.
    """

    def __init__(self, window: int, total_steps: Optional[int]):
        self.window = window
        self.total = total_steps
        self.step = 0        # steps executed so far
        self.in_graph = 0    # steps since the last cut

    def maybe_detach(self, h: torch.Tensor, l: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Call once before each L- or H-step, on the carried states."""
        if self.window <= 0:
            return h.detach(), l.detach()
        if self.total is not None:
            # One exact cut `window` steps before the end. If the window covers
            # the whole thought (window >= total), cut at step 0: every step of
            # this thought keeps gradient, but the *entering* (previous-thought)
            # states are still severed -- the raw cross-thought chain must never
            # survive, or truncation silently degrades to full-document BPTT.
            cut = self.step == max(0, self.total - self.window)
        else:
            # Rolling cut every `window` steps bounds the backward horizon. Also
            # force a cut at the thought's first step (`self.step == 0`) to sever
            # the entering (previous-thought) h/l states, mirroring the
            # fixed-depth max(0, total-window) guarantee. This closes TWO paths of
            # raw cross-thought BPTT that the plain rolling cut leaves open:
            #   (1) the footgun (notes §18.2): a thought that halts in <= window
            #       steps never triggers the rolling cut, so the *returned* state
            #       stays wired to the previous thought. Not reachable at the
            #       shipped config (min ACT depth n_cycles*(l_steps+1)=8 > window
            #       <=5), but silent if the cycle/window counts change.
            #   (2) LIVE at the shipped config: the ponder cost reads halt_prob on
            #       the H-state at each cycle boundary (op index l_steps=3), which
            #       is BEFORE the first rolling cut (op index `window`=5) -- so the
            #       ponder gradient leaked one thought back through the raw h/l
            #       chain even though the returned state was severed. §18.2 missed
            #       this because it only checked the returned state. Cross-thought
            #       credit must flow through the gestalt memory (its own window),
            #       never the raw chain (§3.6); the entry cut enforces that for the
            #       ponder term too. (Verified: entering-state grad is None after;
            #       non-None before. Stage-E metrics shift ~1e-4, the size of the
            #       severed ponder-weight=0.01 leak; nll/ssl unaffected.)
            cut = self.step == 0 or self.in_graph >= self.window
        if cut:
            self.in_graph = 0
            return h.detach(), l.detach()
        return h, l

    def tick(self) -> None:
        self.step += 1
        self.in_graph += 1


class HaltingHead(nn.Module):
    """ACT-style halting probability from the current H-state (§1.1, §5.5)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Returns a halting probability in (0, 1) per batch element.
        return torch.sigmoid(self.proj(h)).squeeze(-1)


class HRMInnerLoop(nn.Module):
    """
    Produces one thought vector from a chunk embedding, given the current
    persistent gestalt memory. Encapsulates the L/H recurrence, the diagonal
    decay gate, MagicNorm hard-normalization, cross-attention into memory,
    and (optionally) ACT adaptive depth.
    """

    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float,
                 l_steps_per_h_update: int, h_updates_per_thought: int,
                 n_roles: int, min_decay: float, max_decay: float,
                 act_max_ponder_steps: int):
        super().__init__()
        self.d_model = d_model
        self.l_steps_per_h_update = l_steps_per_h_update
        self.h_updates_per_thought = h_updates_per_thought
        self.act_max_ponder_steps = act_max_ponder_steps

        # Separate decay-gated recurrences for the fast and slow modules --
        # they share the primitive (decay_gate.py) but hold independent weights.
        self.l_transition = DiagonalDecayGate(d_model, d_ff, dropout, min_decay, max_decay)
        self.h_transition = DiagonalDecayGate(d_model, d_ff, dropout, min_decay, max_decay)

        # Cross-attention into the persistent gestalt memory (read only; the
        # write happens once per finished thought, outside this module).
        self.memory_reader = GestaltCrossAttentionReader(d_model, n_heads, n_roles, dropout)

        # Cross-attention into the input lane (raw tokens + aged gestalts),
        # per §4.2: the Reasoner's H/L state may only *read* the input lane,
        # never have it written directly into its recurrent state.
        self.input_reader = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.input_norm = nn.LayerNorm(d_model)

        self.halting_head = HaltingHead(d_model)

        # Normalizes the incoming (already order-aware) chunk latent before it
        # is injected into the L/H recurrences, bounding the injection scale.
        self.chunk_pool_norm = nn.LayerNorm(d_model)

    def _one_l_group_then_h_update(
        self, h_state: torch.Tensor, l_state: torch.Tensor, chunk_embed: torch.Tensor,
        input_lane_kv: Optional[torch.Tensor], input_lane_pad: Optional[torch.Tensor],
        memory: GestaltMemoryBank, trunc: _TruncationSchedule,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run l_steps_per_h_update fast updates, then one slow H update."""
        for _ in range(self.l_steps_per_h_update):
            h_state, l_state = trunc.maybe_detach(h_state, l_state)
            # Fast local refinement conditioned on the chunk and the current H context.
            e = chunk_embed + h_state  # inject chunk latent + current strategic (H) context
            l_state = self.l_transition(l_state, e)
            l_state = hard_normalize(l_state)          # MagicNorm hard-norm at L-step exit
            trunc.tick()

        h_state, l_state = trunc.maybe_detach(h_state, l_state)
        # Read from input lane (raw tokens / aged gestalts) and persistent memory,
        # both purely as cross-attention context feeding the H-module's injection.
        query = l_state.unsqueeze(1)
        if input_lane_kv is not None:
            input_ctx, _ = self.input_reader(self.input_norm(query), input_lane_kv, input_lane_kv,
                                               key_padding_mask=input_lane_pad,
                                               need_weights=False)
            input_ctx = input_ctx.squeeze(1)
        else:
            input_ctx = torch.zeros_like(l_state)
        memory_ctx = self.memory_reader(l_state, memory)

        h_injection = l_state + input_ctx + memory_ctx
        h_state = self.h_transition(h_state, h_injection)
        h_state = hard_normalize(h_state)               # MagicNorm hard-norm at H-step exit
        trunc.tick()
        return h_state, l_state

    def forward(
        self,
        chunk_embed: torch.Tensor,                # (batch, d_model) order-aware chunk latent
        memory: GestaltMemoryBank,
        input_lane_kv: Optional[torch.Tensor],
        h_state: Optional[torch.Tensor] = None,
        l_state: Optional[torch.Tensor] = None,
        grad_window: int = 5,
        use_act: bool = False,
        input_lane_mask: Optional[torch.Tensor] = None,  # (batch, n_kv) bool, True = real slot
        active_mask: Optional[torch.Tensor] = None,       # (batch,) bool, True = row has a real chunk
    ):
        """
        Runs the bounded recurrent deliberation for one thought.

        Returns:
          h_state: (batch, d_model) final H-state == the thought vector.
          ponder_cost: scalar tensor, ACT ponder penalty (0 if use_act=False).
        """
        batch, d_model = chunk_embed.shape
        device = chunk_embed.device

        if h_state is None:
            h_state = torch.zeros(batch, d_model, device=device)
        if l_state is None:
            l_state = torch.zeros(batch, d_model, device=device)

        # The chunk latent is produced order-awarely by the shared ChunkEncoder
        # (model.py) -- the same representation the JEPA self-supervised loss
        # predicts, so that loss regularizes exactly this injection. Re-norm it
        # here to bound the injection scale into the recurrence.
        chunk_embed = self.chunk_pool_norm(chunk_embed)

        # Convert the input-lane validity mask into MultiheadAttention's
        # key_padding_mask convention (True = ignore), guarding fully-masked
        # rows the same way input_lane.py does (an all-True row NaNs softmax;
        # such rows attend freely to the lane's zeroed outputs instead).
        input_lane_pad = None
        if input_lane_kv is not None and input_lane_mask is not None:
            input_lane_pad = ~input_lane_mask
            all_masked = input_lane_pad.all(dim=1)
            if all_masked.any():
                input_lane_pad = input_lane_pad.clone()
                input_lane_pad[all_masked] = False

        ponder_cost = torch.zeros((), device=device)

        n_cycles = self.h_updates_per_thought
        steps_per_cycle = self.l_steps_per_h_update + 1
        if not use_act:
            # Fixed-depth schedule (Stages A-D): run exactly n_cycles H-updates.
            # Total step count is known, so the truncation cut is exact.
            trunc = _TruncationSchedule(grad_window, total_steps=n_cycles * steps_per_cycle)
            for _ in range(n_cycles):
                h_state, l_state = self._one_l_group_then_h_update(
                    h_state, l_state, chunk_embed, input_lane_kv, input_lane_pad, memory, trunc)
        else:
            # ACT-style adaptive depth (Stage D+, §1.1, §5.5): keep going until
            # the halting head says stop, or a max-steps safety cap is hit.
            # Depth is unknown up front, so truncation falls back to a rolling
            # cut every `grad_window` steps (backward horizon <= grad_window).
            trunc = _TruncationSchedule(grad_window, total_steps=None)
            # Only rows whose current chunk is real may contribute to the
            # ponder cost and the halt vote. The sequential caller (model.
            # forward_self_supervised) runs the loop whenever ANY row is
            # active, so rows whose document already ended keep evolving on
            # pad-chunk latents; without the mask their garbage halt
            # probabilities (a) polluted the ponder gradient into the halting
            # head + h_transition and (b) voted on the whole batch's depth.
            # (Their SSL predictions were always excluded -- this closes the
            # two remaining leaks. All-rows-active batches are unchanged.)
            w = active_mask.float() if active_mask is not None else torch.ones(batch, device=device)
            denom = w.sum().clamp_min(1.0)
            for step in range(self.act_max_ponder_steps):
                h_state, l_state = self._one_l_group_then_h_update(
                    h_state, l_state, chunk_embed, input_lane_kv, input_lane_pad, memory, trunc)
                halt_prob = self.halting_head(h_state)
                # Penalize *continuing*, i.e. reward halting sooner: a higher
                # halt probability lowers the ponder cost, pushing the model
                # toward the shallowest depth that doesn't hurt the primary loss.
                # Each executed step also adds a term, so deeper loops cost
                # strictly more. (Accumulating halt_prob itself would invert
                # this and penalize the model for wanting to stop.)
                ponder_cost = ponder_cost + ((1.0 - halt_prob) * w).sum() / denom
                # Stochastic halting during training would need a full ACT
                # accumulator; for clarity we use expected-value ponder cost
                # and a soft, differentiable "continue probability" schedule
                # rather than hard branching, so the graph stays simple.
                # The float() is a host-device sync; only pay it on steps where
                # the vote can actually trigger a break (step+1 >= n_cycles) --
                # the decision sequence is identical, but the sync on earlier
                # steps (one per chunk, per optimizer step, in Stages D/E) is
                # skipped on a launch-overhead-bound workload.
                if step + 1 >= n_cycles and float((halt_prob * w).sum() / denom) > 0.5:
                    break

        # grad_window <= 0 means "no gradient through the loop at all" (the
        # per-step cuts above leave the final step's ops in-graph, so finish
        # the job on the returned value -- and on the ponder cost, whose
        # halt_prob terms otherwise keep a 1-op graph into the halting head
        # and h_transition).
        if grad_window <= 0:
            h_state = h_state.detach()
            ponder_cost = ponder_cost.detach()

        return h_state, ponder_cost
