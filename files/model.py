"""
model.py
========
Wires every component into the full architecture described in §1-§4:

    tokens --chunker--> chunks --HRM inner loop (Reasoner)--> thoughts
                                        |                         |
                                        v                         v
                                 gestalt memory <---------- Talker (tokens out)

Two forward passes are exposed, matching §2's "two losses, two granularities".
**Both run the HRM loop and train it** -- that is the point (notes §26): the loop
is the reasoner, so both objectives shape it, just used two different ways:

  - `forward_grounded`: the reconstruction/autoencoder branch (§2.2), SEQUENTIAL.
    Per chunk t: encode -> HRM loop (reading + writing the persistent gestalt
    memory, carrying h/l state across chunks) -> Talker decodes the SAME chunk t.
    Masked-NLL against chunk t's own tokens. Trains the loop to produce a
    *decodable* thought. (Stage A is the one exception: the loop is off, so it's
    encoder-latent -> Talker directly -- the shallow fixed Reasoner that grounds
    the Talker first, §5.1.) This is the always-on anti-collapse anchor.

  - `forward_self_supervised`: the JEPA-style predictive branch (§2.1), run ON the
    HRM loop but PARALLEL across chunks (each chunk's loop is independent -- fresh
    state, EMPTY memory, no Talker -- so no sequential dependency). `pred_head(
    loop(encode(chunk_t)))` predicts chunk t+1's EMA-target latent, for every
    chunk pair at once. Trains the loop to *predict forward* (notes §25/§26).

So: reconstruction = loop + memory + Talker (decodable); prediction = loop alone
(forward). Same loop weights, two objectives. `pred_head` is also what generation
uses (`predict_next_latent`). The former linear SSL / separate projection head /
detached gen MLP were removed (§26).

`role_ids` (USER=0 / SELF=1 / SYSTEM=2, see config.role_tags) are threaded
through explicitly, per §4.2 -- the caller decides whether a given
document's chunks are "self" (the model's own prior turns) or "user"
content, and that's the tag written into the memory bank.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from config import ModelConfig
from input_lane import InputLaneEncoder
from hrm_loop import HRMInnerLoop
from gestalt_memory import GestaltMemoryBank
from talker import Talker
from ema_target import ChunkEncoder, EMATargetEncoder
from losses import scaled_cosine_loss, grounded_nll_loss, ponder_cost_loss, variance_regularization

USER, SELF, SYSTEM = 0, 1, 2  # role-tag ids, matching config.role_tags order


@dataclass
class StageFlags:
    """
    The subset of curriculum state that changes model *behavior* (as
    opposed to just loss weighting, which train.py handles). See
    curriculum.py for the full per-stage settings.
    """
    use_hrm_loop: bool = True          # False only in Stage A (shallow, fixed Reasoner)
    detach_memory: bool = True         # True in Stage B: memory writes exist but no grad back
    inner_loop_grad_window: int = 5
    memory_grad_window: int = 5
    use_act: bool = False              # Stage E+
    use_input_lanes: bool = False       # Stage F: role-tagged two-lane separation


class LatentThoughtModel(nn.Module):
    """
    `chunker` is injected rather than built internally, since the right
    chunker depends on the data pipeline: a `SegmentAnyTextChunker`
    (chunker.py) wired with the real SaT model + a HuggingFace tokenizer for
    real text, or the same class wired with offline stubs (data.py) for the
    no-download synthetic-text path -- both are SaT Capped, matching Thought
    Gestalt's actual preprocessing. The chunker exposes
    `chunk_batch(...) -> (chunk_tensor, chunk_mask)`, so nothing else in this
    class needs to know which backend it has.
    """

    def __init__(self, cfg: ModelConfig, chunker):
        super().__init__()
        self.cfg = cfg
        self.chunker = chunker

        # --- Shared chunk encoder -------------------------------------
        # One order-aware chunk encoder produces the chunk latent used *both* as
        # the injection into the Reasoner's HRM loop and as the online latent for
        # the JEPA self-supervised loss (§2.1). Sharing it is what makes the
        # self-supervised loss actually regularize the representation the loop and
        # Talker consume, instead of training a disconnected encoder; it also
        # gives the Reasoner an order-aware chunk vector rather than a
        # bag-of-words mean-pool.
        self.chunk_encoder = ChunkEncoder(
            vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_heads=cfg.n_heads,
            d_ff=cfg.d_ff, dropout=cfg.dropout, max_len=cfg.max_chunk_len,
            n_layers=cfg.chunk_encoder_layers,
        )
        self.hrm_loop = HRMInnerLoop(
            d_model=cfg.d_model, d_ff=cfg.d_ff, n_heads=cfg.n_heads, dropout=cfg.dropout,
            l_steps_per_h_update=cfg.l_steps_per_h_update,
            h_updates_per_thought=cfg.h_updates_per_thought,
            n_roles=len(cfg.role_tags), min_decay=cfg.decay_min,
            max_decay=cfg.decay_max, act_max_ponder_steps=cfg.act_max_ponder_steps,
        )

        # --- Input lane (§4.1, §4.2) -----------------------------------
        self.input_lane = InputLaneEncoder(
            vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_heads=cfg.n_heads,
            d_ff=cfg.d_ff, dropout=cfg.dropout, n_layers=cfg.input_lane_layers,
            max_len=cfg.recent_token_window,
        )

        # --- Talker (§1.3) -----------------------------------------------
        self.talker = Talker(
            vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_heads=cfg.n_heads,
            d_ff=cfg.d_ff, dropout=cfg.dropout, n_layers=cfg.talker_layers,
            n_roles=len(cfg.role_tags), max_chunk_len=cfg.max_chunk_len,
        )

        # --- Self-supervised JEPA branch (§2.1) — the HRM loop IS the predictor.
        # The self-supervised loss runs ON the inner loop (the reasoner), exactly
        # as JEPA-Reasoner runs SSL on its reasoner transformer: for each chunk t
        # the loop produces a finished thought, and `pred_head` maps that thought
        # to chunk t+1's encoder-space latent (forward_self_supervised). The
        # gradient reaches the loop AND the shared encoder, so deliberation --
        # and, under ACT, its depth -- is trained to reason forward, not only to
        # reconstruct. The former linear SSL (a separate ssl_proj/latent_predictor
        # projection head + a detached gen MLP) was removed: it was the §2.4
        # collapse-era shortcut that severed the loop from its own predictive
        # objective, and the A/B (notes §25.1) showed the on-loop loss is MORE
        # collapse-robust and the projection head is not load-bearing. Collapse is
        # held by the always-on reconstruction anchor + the variance floor + the
        # slow EMA target, not by an isolation head.
        self.pred_head = nn.Linear(cfg.d_model, cfg.d_model)

    # ------------------------------------------------------------------
    def encode_chunks(self, chunk_tensor: torch.Tensor) -> torch.Tensor:
        """
        Shared order-aware chunk latents, (batch, n_chunks, d_model), WITH grad.
        The grounded, self-supervised, and generation paths all consume the same
        shared-encoder representation, so a caller that runs more than one of them
        in a single step should encode ONCE here and pass the result in via each
        method's `chunk_vecs=` argument -- avoiding 2-3 redundant encoder passes
        per step (the online encoder is otherwise re-run per branch).
        """
        batch, n_chunks, chunk_len = chunk_tensor.shape
        flat_ids = chunk_tensor.reshape(batch * n_chunks, chunk_len)
        return self.chunk_encoder(flat_ids, flat_ids != 0).reshape(batch, n_chunks, -1)

    # ------------------------------------------------------------------
    # Self-supervised branch: the HRM loop IS the predictor (§2.1 / notes §25).
    # Parallel across chunks, no memory unroll -- but NOT loop-free.
    # ------------------------------------------------------------------
    def forward_self_supervised(self, chunk_tensor: torch.Tensor, chunk_mask: torch.Tensor,
                                 stage: StageFlags, ema: EMATargetEncoder, cos_weight: float = 1.0,
                                 var_weight: float = 0.0, chunk_vecs=None) -> torch.Tensor:
        """
        The self-supervised loss, run ON the HRM loop -- the loop is the reasoner
        that predicts the next chunk's latent, exactly as JEPA-Reasoner runs SSL
        on its reasoner transformer (§2.1). Returns

            cos_weight * cosine_prediction_loss + var_weight * variance_reg

        PARALLEL across chunks: every chunk's loop is independent -- fresh state,
        EMPTY memory -- so there is no sequential cross-thought dependency (the
        "fully parallel across chunks / without unrolling through memory" of §2.1;
        the Talker and the memory chain are skipped, the loop is NOT). The whole
        batch*n_chunks set runs through the loop in ONE call. Gradients use the
        inner-loop 2->5 truncation (stage.inner_loop_grad_window), the §3.5 method
        the design specifies for exactly this -- so the loop, and under ACT its
        depth, is trained to reason forward.

        Online = pred_head(loop(encode(chunk_t))); target = the EMA target
        encoder's next-chunk latent (encoder space, stop-grad). Collapse is held
        by the always-on reconstruction anchor + the variance floor (on the shared
        latent) + the slow EMA target -- the A/B (notes §25.1) showed the former
        projection-head isolation is not needed once SSL is on the loop.
        """
        batch, n_chunks, chunk_len = chunk_tensor.shape
        device = chunk_tensor.device
        flat_ids = chunk_tensor.reshape(batch * n_chunks, chunk_len)
        if chunk_vecs is None:
            chunk_vecs = self.chunk_encoder(flat_ids, flat_ids != 0).reshape(batch, n_chunks, -1)
        flat_z = chunk_vecs.reshape(batch * n_chunks, -1)

        # One loop call over every chunk, each independent (empty memory, fresh
        # state) -- the parallel-across-chunks reasoner pass.
        empty_mem = GestaltMemoryBank(self.cfg.memory_capacity, self.cfg.d_model)
        h_flat, _ = self.hrm_loop(flat_z, empty_mem, None, h_state=None, l_state=None,
                                  grad_window=stage.inner_loop_grad_window, use_act=stage.use_act)
        online = self.pred_head(h_flat).reshape(batch, n_chunks, -1)

        with torch.no_grad():
            tgt = ema.encode(flat_ids, flat_ids != 0).reshape(batch, n_chunks, -1)

        pair = chunk_mask[:, :-1] & chunk_mask[:, 1:]
        pred, target = online[:, :-1][pair], tgt[:, 1:][pair]
        cos = (scaled_cosine_loss(pred, target, k=self.cfg.cosine_loss_k)
               if pred.numel() > 0 else torch.zeros((), device=device))
        flat_valid = chunk_mask.reshape(batch * n_chunks)
        var = (variance_regularization(flat_z[flat_valid]) if var_weight > 0
               else torch.zeros((), device=device))
        return cos_weight * cos + var_weight * var

    @torch.no_grad()
    def predict_next_latent(self, latent, memory, h_state=None, l_state=None,
                             grad_window: int = 5, use_act: bool = False):
        """
        Inference-time next-latent prediction (generate.py): run the inner loop on
        `latent` and read the next latent off the finished thought via pred_head
        (the same map forward_self_supervised trains). Returns (pred_latent,
        thought); the thought is the carried loop state.
        """
        h, _ = self.hrm_loop(latent, memory, None, h_state=h_state, l_state=l_state,
                              grad_window=grad_window, use_act=use_act)
        return self.pred_head(h), h

    @torch.no_grad()
    def latent_collapse_metric(self, chunk_tensor: torch.Tensor, chunk_mask: torch.Tensor) -> float:
        """
        Mean per-dimension standard deviation of the shared chunk latents over a
        batch. ~0 means the encoder has collapsed to a (near-)constant vector;
        healthy representations sit well above 0. A cheap collapse monitor.

        Measured in eval mode: with dropout active, dropout noise alone reads
        as per-dim std ~0.05-0.08 (same order as the 0.1 collapse floor), so a
        fully collapsed encoder would log a healthy-looking nonzero value --
        exactly the silent failure this monitor exists to catch. Eval mode
        makes true collapse read as ~0. (Monitor only; training-time behavior
        is untouched.)
        """
        batch, n_chunks, chunk_len = chunk_tensor.shape
        flat_ids = chunk_tensor.reshape(batch * n_chunks, chunk_len)
        flat_valid = chunk_mask.reshape(batch * n_chunks)
        was_training = self.chunk_encoder.training
        if was_training:
            self.chunk_encoder.eval()
        try:
            z = self.chunk_encoder(flat_ids, flat_ids != 0)[flat_valid]
        finally:
            if was_training:
                self.chunk_encoder.train()
        if z.shape[0] < 2:
            return 0.0
        return z.std(dim=0).mean().item()

    # ------------------------------------------------------------------
    # Grounded branch: sequential over chunks, uses the HRM loop + memory + Talker.
    # ------------------------------------------------------------------
    def forward_grounded(
        self,
        chunk_tensor: torch.Tensor,          # (batch, n_chunks, chunk_len)
        chunk_mask: torch.Tensor,            # (batch, n_chunks) bool
        raw_token_ids: torch.Tensor,          # (batch, n_raw) recent raw tokens for input lane
        raw_mask: torch.Tensor,               # (batch, n_raw) bool
        memory: GestaltMemoryBank,
        role_id: int,
        stage: StageFlags,
        ponder_weight: float,
        chunk_vecs=None,
    ):
        """
        Returns (nll_loss, ponder_loss, thoughts) where `thoughts` is the
        list of per-chunk H-state vectors produced (useful for logging/eval).

        `chunk_vecs` (batch, n_chunks, d_model), if supplied by the caller via
        encode_chunks, is the shared order-aware chunk latent -- reused here
        instead of re-encoding, so a step that also runs the SSL/gen branches
        pays for exactly one online encoder pass.
        """
        batch, n_chunks, chunk_len = chunk_tensor.shape
        device = chunk_tensor.device

        input_lane_kv, input_lane_mask = None, None
        if stage.use_input_lanes:
            aged = memory.filtered_stacked([USER, SYSTEM])
            aged_mask = None
            if aged is not None:
                aged_mask = torch.ones(aged.shape[0], aged.shape[1], dtype=torch.bool, device=device)
            input_lane_kv, input_lane_mask = self.input_lane(raw_token_ids, raw_mask, aged, aged_mask)

        # Order-aware chunk latents from the shared encoder -- the same
        # representation the self-supervised loss predicts. All chunks are
        # independent of the loop state, so encode the whole document in one
        # batched call instead of once per chunk inside the sequential loop.
        # Reuse the caller's shared encode when provided (encode_chunks).
        if chunk_vecs is None:
            flat_ids = chunk_tensor.reshape(batch * n_chunks, chunk_len)
            chunk_vecs = self.chunk_encoder(flat_ids, flat_ids != 0).reshape(batch, n_chunks, -1)

        h_state, l_state = None, None
        total_nll = torch.zeros((), device=device)
        total_ponder = torch.zeros((), device=device)
        n_valid_chunks = 0
        thoughts = []

        for t in range(n_chunks):
            active = chunk_mask[:, t]
            if not active.any():
                continue

            chunk_ids = chunk_tensor[:, t, :]                       # (batch, chunk_len)
            chunk_token_mask = chunk_ids != 0
            chunk_vec = chunk_vecs[:, t]                             # (batch, d_model)

            if stage.use_hrm_loop:
                h_state, ponder = self.hrm_loop(
                    chunk_vec, memory, input_lane_kv,
                    h_state=h_state, l_state=l_state,
                    grad_window=stage.inner_loop_grad_window, use_act=stage.use_act,
                    input_lane_mask=input_lane_mask,
                )
                l_state = h_state  # re-seed next chunk's L-state from this thought (§1.1)
            else:
                # Stage A: shallow, fixed Reasoner -- the chunk encoder's latent
                # directly, no recurrence (§5.1's "shallow, fixed-depth Reasoner").
                h_state = chunk_vec
                ponder = torch.zeros((), device=device)

            total_ponder = total_ponder + ponder
            thoughts.append(h_state)

            # Write this thought into the persistent gestalt memory (§1.2).
            write_vec = h_state.detach() if stage.detach_memory else h_state
            memory.write(write_vec, role_id)
            memory.apply_grad_truncation(stage.memory_grad_window)

            # Grounded NLL: teacher-force the Talker on this chunk's tokens.
            # The target mask also covers the FIRST pad position of each active
            # shorter-than-max chunk: one supervised end-of-chunk step, with PAD
            # (reserved id 0, never a real token) acting as EOS. Without it, no
            # position past a chunk's true length ever receives gradient, so
            # inference-time decoding has no trained way to terminate a chunk --
            # it would emit max_chunk_len tokens with an untrained tail every
            # time, and the garbage tail would be re-encoded into the next
            # latent. Full-length chunks get no end mark: a capped span
            # legitimately "continues".
            logits = self.talker(chunk_ids, h_state, memory)
            target_mask = chunk_token_mask
            lengths = chunk_token_mask.sum(-1)                       # (batch,)
            has_end = active & (lengths > 0) & (lengths < chunk_len)
            if has_end.any():
                target_mask = chunk_token_mask.clone()
                target_mask[has_end, lengths[has_end]] = True        # target there is id 0 = PAD/EOS
            chunk_nll = grounded_nll_loss(logits, chunk_ids, target_mask)
            total_nll = total_nll + chunk_nll
            n_valid_chunks += 1

        if n_valid_chunks > 0:
            # Normalize both losses per thought, so their magnitudes (and the
            # effective ponder weight) don't scale with how many chunks a
            # document happens to have -- doc lengths vary from min_chunks to
            # max_chunks_per_doc within a run.
            total_nll = total_nll / n_valid_chunks
            total_ponder = total_ponder / n_valid_chunks
        ponder_loss = ponder_cost_loss(total_ponder, ponder_weight)
        return total_nll, ponder_loss, thoughts
