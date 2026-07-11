"""
model.py
========
Wires every component into the full architecture described in §1-§4:

    tokens --chunker--> chunks --HRM inner loop (Reasoner)--> thoughts
                                        |                         |
                                        v                         v
                                 gestalt memory <---------- Talker (tokens out)

Two forward passes, cleanly split by role (notes §27) -- the HRM loop lives in
ONE of them, the predictive one:

  - `forward_grounded`: the reconstruction/autoencoder anchor (§2.2). A pure codec
    -- encode chunk t -> Talker decodes the SAME chunk t -- with **no HRM loop and
    no memory**, run in parallel over all chunks. Anchors the shared encoder
    against collapse (a constant latent can't reconstruct varied chunks) and
    trains the Talker to decode encoder-space latents. Always on.

  - `forward_self_supervised`: the JEPA-style predictive branch (§2.1), run ON the
    HRM loop and SEQUENTIALLY so the loop reads its accumulating gestalt memory
    (Thought Gestalt's cross-thought reasoning). Per chunk t: h_t = loop(encode(
    chunk_t), memory) -> write h_t to memory -> pred_head(h_t) predicts chunk
    t+1's EMA-target latent. Trains the loop + encoder + pred_head + the memory
    readers/writers to *reason forward*. Carries the ACT ponder.

So: reconstruction = encoder + Talker (a codec, no loop); prediction = the loop +
memory (forward reasoning). The loop is trained ONLY by prediction -- freed from
the "preserve the current chunk vs. shift to the next" conflict that putting it in
reconstruction created (notes §27). `pred_head` is also what generation uses
(`predict_next_latent`); the Talker decodes the loop's predicted (encoder-space)
latent, the same space it was trained on.

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
    # Reconstruction anchor: a PURE autoencoder (encoder -> Talker). No HRM loop.
    # ------------------------------------------------------------------
    def forward_grounded(self, chunk_tensor: torch.Tensor, chunk_mask: torch.Tensor,
                          chunk_vecs=None) -> torch.Tensor:
        """
        The reconstruction/autoencoder anchor (§2.2, notes §27): encode each chunk
        to a latent, decode THAT SAME chunk's tokens with the Talker. A pure codec
        -- **no HRM loop, no memory** -- run in parallel over all chunks. This
        anchors the shared encoder against collapse (a constant latent cannot
        reconstruct varied chunks) and trains the Talker to decode encoder-space
        latents: the space the predictor (forward_self_supervised) forecasts into
        and generation decodes from. The HRM loop is deliberately NOT here --
        reconstructing the *current* chunk needs a faithful codec, not reasoning,
        and putting the loop here fought its predictive objective (notes §27).
        """
        batch, n_chunks, chunk_len = chunk_tensor.shape
        device = chunk_tensor.device
        flat_ids = chunk_tensor.reshape(batch * n_chunks, chunk_len)
        if chunk_vecs is None:
            chunk_vecs = self.chunk_encoder(flat_ids, flat_ids != 0).reshape(batch, n_chunks, -1)
        flat_valid = chunk_mask.reshape(batch * n_chunks)
        if not bool(flat_valid.any()):
            return torch.zeros((), device=device)
        ids = flat_ids[flat_valid]                                   # (n_real, L)
        z = chunk_vecs.reshape(batch * n_chunks, -1)[flat_valid]      # (n_real, d)
        # Codec Talker: conditions purely on the chunk's own latent (empty memory).
        empty_mem = GestaltMemoryBank(self.cfg.memory_capacity, self.cfg.d_model)
        logits = self.talker(ids, z, empty_mem)
        tok_mask = ids != 0
        # End-of-chunk (EOS/PAD) supervision on the first pad position of each
        # shorter-than-max chunk, so decoding can terminate (notes §19.2).
        lengths = tok_mask.sum(-1)
        target_mask = tok_mask
        has_end = (lengths > 0) & (lengths < chunk_len)
        if has_end.any():
            target_mask = tok_mask.clone()
            target_mask[has_end, lengths[has_end]] = True
        return grounded_nll_loss(logits, ids, target_mask)

    # ------------------------------------------------------------------
    # Self-supervised branch: the HRM loop IS the predictor, SEQUENTIALLY, reading
    # the accumulating gestalt memory (§2.1, notes §25/§27).
    # ------------------------------------------------------------------
    def forward_self_supervised(self, chunk_tensor: torch.Tensor, chunk_mask: torch.Tensor,
                                 raw_token_ids: torch.Tensor, raw_mask: torch.Tensor,
                                 memory: GestaltMemoryBank, role_id: int, stage: StageFlags,
                                 ema: EMATargetEncoder, cos_weight: float = 1.0,
                                 var_weight: float = 0.0, ponder_weight: float = 0.0,
                                 chunk_vecs=None):
        """
        The predictive branch (§2.1), run ON the HRM loop and SEQUENTIALLY so the
        loop reads its *accumulating* gestalt memory as it goes -- Thought
        Gestalt's cross-thought reasoning (§1.2, §3.6). Per chunk t:
          h_t = loop(encode(chunk_t), memory)      # attends over all prior thoughts
          memory.write(h_t)                        # detached in Stage B, un-detached C+
          pred_head(h_t) -> predict chunk t+1's EMA-target latent (encoder space)
        Trains the loop, the shared encoder, pred_head, AND the memory
        readers/writers (which are dead code unless memory is populated -- notes
        §27). Gradients use the inner-loop 2->5 truncation (§3.5); memory credit is
        bounded by stage.memory_grad_window (§3.6). Carries the ACT ponder cost.

        Returns (ssl_loss = cos_weight*cos + var_weight*var, ponder_loss). Collapse
        is held by the always-on autoencoder anchor + the variance floor + the slow
        EMA target (notes §25.1/§26.1).
        """
        batch, n_chunks, chunk_len = chunk_tensor.shape
        device = chunk_tensor.device

        input_lane_kv, input_lane_mask = None, None
        if stage.use_input_lanes:
            aged = memory.filtered_stacked([USER, SYSTEM])
            aged_mask = (torch.ones(aged.shape[0], aged.shape[1], dtype=torch.bool, device=device)
                         if aged is not None else None)
            input_lane_kv, input_lane_mask = self.input_lane(raw_token_ids, raw_mask, aged, aged_mask)

        flat_ids = chunk_tensor.reshape(batch * n_chunks, chunk_len)
        if chunk_vecs is None:
            chunk_vecs = self.chunk_encoder(flat_ids, flat_ids != 0).reshape(batch, n_chunks, -1)
        with torch.no_grad():
            tgt = ema.encode(flat_ids, flat_ids != 0).reshape(batch, n_chunks, -1)

        h_state = l_state = None
        total_ponder = torch.zeros((), device=device)
        n_valid = 0
        preds, targets = [], []
        for t in range(n_chunks):
            active = chunk_mask[:, t]
            if not active.any():
                continue
            h_state, ponder = self.hrm_loop(
                chunk_vecs[:, t], memory, input_lane_kv,
                h_state=h_state, l_state=l_state,
                grad_window=stage.inner_loop_grad_window, use_act=stage.use_act,
                input_lane_mask=input_lane_mask,
            )
            l_state = h_state
            total_ponder = total_ponder + ponder
            n_valid += 1
            write_vec = h_state.detach() if stage.detach_memory else h_state
            memory.write(write_vec, role_id)
            memory.apply_grad_truncation(stage.memory_grad_window)
            if t + 1 < n_chunks:
                pair = active & chunk_mask[:, t + 1]
                if pair.any():
                    preds.append(self.pred_head(h_state)[pair])
                    targets.append(tgt[:, t + 1][pair])

        cos = (scaled_cosine_loss(torch.cat(preds, 0), torch.cat(targets, 0), k=self.cfg.cosine_loss_k)
               if preds else torch.zeros((), device=device))
        flat_valid = chunk_mask.reshape(batch * n_chunks)
        var = (variance_regularization(chunk_vecs.reshape(batch * n_chunks, -1)[flat_valid])
               if var_weight > 0 else torch.zeros((), device=device))
        if n_valid > 0:
            total_ponder = total_ponder / n_valid
        return cos_weight * cos + var_weight * var, ponder_cost_loss(total_ponder, ponder_weight)

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

