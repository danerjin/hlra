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
import torch.nn.functional as F

from config import ModelConfig
from input_lane import InputLaneEncoder
from hrm_loop import HRMInnerLoop
from gestalt_memory import GestaltMemoryBank, GestaltReadout
from talker import Talker
from ema_target import ChunkEncoder, EMATargetEncoder
from losses import (scaled_cosine_loss, grounded_nll_loss, ponder_cost_loss,
                    variance_regularization, prediction_variance_loss,
                    prediction_contrastive_loss,
                    prediction_collapse_metric, anti_sycophancy_loss,
                    supervised_halt_loss, turn_end_loss, turn_end_accuracy)

USER, SELF, SYSTEM = 0, 1, 2  # role-tag ids, matching config.role_tags order
# RETRIEVED (3) is the latent-RAG provenance tag (§Q3). It is only usable when
# the model is built with a 4-entry role_tags, e.g. role_tags=("USER","SELF",
# "SYSTEM","RETRIEVED"); A-E ships 3 roles, so RETRIEVED is opt-in and never
# referenced on the validated path.
RETRIEVED = 3
# Conversation-local persona for a RETRIEVED slot. A source is NOT a speaker, but None
# maps to 0 == SELF's persona, so it needs some non-SELF id. There is no free slot:
# personas are speaker ids numbered 0=SELF then 1,2,... by first appearance
# (dialogue_data.transcript_to_turns), so ANY choice collides with a speaker once the
# table fills. Take the TOP slot -- it collides only with the LAST speaker a saturated
# table can hold -- and with every OVERFLOW speaker, since dialogue_data clamps all
# of them into that same top bucket -- rather than with the 3rd (a
# fixed 3 collided with speaker #3 on a 5-party transcript). The RETRIEVED *role* tag
# is what actually separates a source; this id only has to avoid asserting it was SELF.
# No training convention exists (RAG is untrained), so it is arbitrary by design.
def _retrieved_persona(cfg):
    return cfg.n_personas - 1


@dataclass
class StageFlags:
    """
    The subset of curriculum state that changes model *behavior* (as
    opposed to just loss weighting, which train.py handles). See
    curriculum.py for the full per-stage settings.
    """
    # INFORMATIONAL: nothing reads this flag. The loop lives only in
    # forward_self_supervised, which Stage A (the only no-loop stage) never
    # calls -- so "loop off" is enforced by the loss plan, not this flag.
    use_hrm_loop: bool = True          # False only in Stage A (shallow, fixed Reasoner)
    detach_memory: bool = True         # True in Stage B: memory writes exist but no grad back
    inner_loop_grad_window: int = 5
    memory_grad_window: int = 5
    use_act: bool = False              # Stage D+
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
        # A thought is a chunk-level object, so the shared encoder runs its
        # transformer body at the wider thought width cfg.d_latent (embedding
        # lookup stays at the token width d_model) and its FFN scales with that
        # width (cfg.latent_d_ff). Its pooled output -- the chunk latent that
        # feeds the loop, the JEPA SSL loss, and (via the Talker) reconstruction
        # -- is d_latent. At latent_mult=1 (d_latent == d_model) this is the
        # original single-width encoder.
        self.chunk_encoder = ChunkEncoder(
            vocab_size=cfg.vocab_size, d_model=cfg.d_model, d_latent=cfg.d_latent,
            n_heads=cfg.n_heads, d_ff=cfg.latent_d_ff, dropout=cfg.dropout,
            max_len=cfg.max_chunk_len, n_layers=cfg.chunk_encoder_layers,
            arch=cfg.arch,
        )
        # The loop works natively at the thought width cfg.d_latent -- it reads a
        # d_latent chunk latent and emits a d_latent thought (what memory,
        # pred_head, and the Talker consume). Its FFN scales with that width
        # (cfg.latent_d_ff). The only other width it touches is the token-width
        # input lane (d_input=cfg.d_model), which it cross-attends.
        self.hrm_loop = HRMInnerLoop(
            d_latent=cfg.d_latent, d_input=cfg.d_model, d_ff=cfg.latent_d_ff,
            n_heads=cfg.n_heads, dropout=cfg.dropout,
            l_steps_per_h_update=cfg.l_steps_per_h_update,
            h_updates_per_thought=cfg.h_updates_per_thought,
            n_roles=len(cfg.role_tags), min_decay=cfg.decay_min,
            max_decay=cfg.decay_max, act_max_ponder_steps=cfg.act_max_ponder_steps,
            soft_role_tags=cfg.soft_role_tags, soft_role_codebook=cfg.soft_role_codebook,
            trust_gate=cfg.trust_gate, soft_role_content=cfg.soft_role_content,
            trust_gate_vector=cfg.trust_gate_vector, persona_tags=cfg.persona_tags,
            n_personas=cfg.n_personas, core_qk_norm=cfg.core_qk_norm,
        )

        # --- Input lane (§4.1, §4.2) -----------------------------------
        # Token-level encoder at d_model; aged gestalts (d_latent thoughts) are
        # projected down to d_model inside it.
        self.input_lane = InputLaneEncoder(
            vocab_size=cfg.vocab_size, d_model=cfg.d_model, d_latent=cfg.d_latent,
            n_heads=cfg.n_heads, d_ff=cfg.d_ff, dropout=cfg.dropout,
            n_layers=cfg.input_lane_layers, max_len=cfg.recent_token_window,
            arch=cfg.arch,
        )

        # --- Talker (§1.3) -----------------------------------------------
        # Word-level readout at d_model; cross-attends the d_latent thought.
        self.talker = Talker(
            vocab_size=cfg.vocab_size, d_model=cfg.d_model, d_latent=cfg.d_latent,
            n_heads=cfg.n_heads, d_ff=cfg.d_ff, dropout=cfg.dropout,
            n_layers=cfg.talker_layers, n_roles=len(cfg.role_tags),
            max_chunk_len=cfg.max_chunk_len,
            soft_role_tags=cfg.soft_role_tags, soft_role_codebook=cfg.soft_role_codebook,
            trust_gate=cfg.trust_gate, soft_role_content=cfg.soft_role_content,
            trust_gate_vector=cfg.trust_gate_vector, persona_tags=cfg.persona_tags,
            n_personas=cfg.n_personas, arch=cfg.arch, core_qk_norm=cfg.core_qk_norm,
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
        # Maps a finished thought to the next chunk's EMA-target latent -- both
        # are chunk-level, so this is d_latent -> d_latent.
        _phh = getattr(cfg, "pred_head_hidden", 0)
        self.pred_head = (
            nn.Sequential(nn.Linear(cfg.d_latent, _phh), nn.GELU(), nn.Linear(_phh, cfg.d_latent))
            if _phh and _phh > 0 else nn.Linear(cfg.d_latent, cfg.d_latent)
        )

        # --- Gestalt-readout projection (§4 / Q2, Stage-F/RAG only) ---------
        # Homogenizes what gets written to memory: self-thoughts AND external
        # content pass through one projection onto the thought shell. None (the
        # default) writes raw, exactly as A-E does. Never used by
        # forward_self_supervised, so the A-E path is untouched.
        self.gestalt_readout = GestaltReadout(cfg.d_latent) if cfg.gestalt_readout else None

    # ------------------------------------------------------------------
    def _encode_real_rows(self, flat_ids: torch.Tensor, encode_fn) -> torch.Tensor:
        """
        Run `encode_fn(ids, mask)` only on rows that contain at least one real
        token; padded (absent) chunk rows get an exact zero latent. Documents
        rarely fill max_chunks_per_doc, so this skips ~35-50% of encoder rows
        on a real cache. Loss-invariant: pad-row latents feed only dead paths
        (forward_grounded/variance/SSL pairs all select valid chunks; the
        sequential loop's inactive rows contribute to no loss -- their ponder/
        halt-vote exclusion is enforced by active_mask in hrm_loop).
        """
        has_tok = (flat_ids != 0).any(dim=1)
        if bool(has_tok.all()):
            return encode_fn(flat_ids, flat_ids != 0)
        if not bool(has_tok.any()):
            return encode_fn(flat_ids, flat_ids != 0)  # degenerate; guards handle it
        real = encode_fn(flat_ids[has_tok], flat_ids[has_tok] != 0)
        out = real.new_zeros(flat_ids.shape[0], real.shape[-1])
        out[has_tok] = real
        return out

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
        return self._encode_real_rows(flat_ids, self.chunk_encoder).reshape(batch, n_chunks, -1)

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
            chunk_vecs = self._encode_real_rows(flat_ids, self.chunk_encoder).reshape(batch, n_chunks, -1)
        flat_valid = chunk_mask.reshape(batch * n_chunks)
        if not bool(flat_valid.any()):
            return torch.zeros((), device=device)
        ids = flat_ids[flat_valid]                                   # (n_real, L)
        z = chunk_vecs.reshape(batch * n_chunks, -1)[flat_valid]      # (n_real, d)
        # Codec Talker: conditions purely on the chunk's own latent (empty memory).
        empty_mem = GestaltMemoryBank(self.cfg.memory_capacity, self.cfg.d_latent)
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
                                 var_weight: float = 0.0, pred_var_weight: float = 0.0,
                                 contrastive_weight: float = 0.0, contrastive_temp: float = 0.07,
                                 contrastive_hard: bool = False, token_weight: float = 0.0,
                                 ponder_weight: float = 0.0,
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

        Halt mode (experiments.md #2): with cfg.halt_mode == "supervised" AND ACT
        on (Stage D+), the depth is trained by a supervised BCE halt gate instead
        of the ponder cost -- dispatched here to a separate method so THIS one (the
        validated A-E/ponder path) is byte-identical at the default halt_mode.
        """
        if self.cfg.halt_mode == "supervised" and stage.use_act:
            return self.forward_self_supervised_halt(
                chunk_tensor, chunk_mask, raw_token_ids, raw_mask, memory, role_id,
                stage, ema, cos_weight=cos_weight, var_weight=var_weight,
                chunk_vecs=chunk_vecs)
        batch, n_chunks, chunk_len = chunk_tensor.shape
        device = chunk_tensor.device

        input_lane_kv, input_lane_mask = None, None
        if stage.use_input_lanes:
            # LEAK GUARD (§4 SSL-target leak). raw_token_ids is the document's own
            # trailing window (data.chunk_text_example: the ids of the LAST kept
            # chunks), so feeding it here would let the loop cross-attend to chunk
            # t+1's own tokens while predicting t+1 -- the SSL target leaking
            # through the lane. There is no causal single-window fix: any static
            # document window contains future chunks. A self-supervised *document*
            # also has no external "input turn" to legitimately place in the lane
            # (that is forward_dialogue's user turn, disjoint from the SELF target
            # by construction). So the raw-token document lane is dropped; the only
            # causally-safe lane content is prior-turn aged gestalts already in
            # memory (USER/SYSTEM), snapshotted here before the loop writes any
            # current-document thought. When there are none (the A->E-shaped
            # document case: memory holds only SELF), the lane stays None -- exactly
            # equivalent to use_input_lanes=False, and leak-free.
            aged = memory.filtered_stacked([USER, SYSTEM])
            if aged is not None:
                aged_mask = torch.ones(aged.shape[0], aged.shape[1],
                                       dtype=torch.bool, device=device)
                no_raw = raw_token_ids[:, :0]        # (batch, 0): no document tokens in the lane
                no_raw_mask = raw_mask[:, :0]
                input_lane_kv, input_lane_mask = self.input_lane(no_raw, no_raw_mask, aged, aged_mask)

        flat_ids = chunk_tensor.reshape(batch * n_chunks, chunk_len)
        if chunk_vecs is None:
            chunk_vecs = self._encode_real_rows(flat_ids, self.chunk_encoder).reshape(batch, n_chunks, -1)
        with torch.no_grad():
            tgt = self._encode_real_rows(flat_ids, ema.encode).reshape(batch, n_chunks, -1)

        h_state = l_state = None
        total_ponder = torch.zeros((), device=device)
        n_valid = 0
        preds, targets, hs_all, groups, next_tok = [], [], [], [], []
        batch_idx = torch.arange(batch, device=device)   # doc id per row, for hard-negative InfoNCE
        for t in range(n_chunks):
            active = chunk_mask[:, t]
            if not active.any():
                continue
            h_state, ponder = self.hrm_loop(
                chunk_vecs[:, t], memory, input_lane_kv,
                h_state=h_state, l_state=l_state,
                grad_window=stage.inner_loop_grad_window, use_act=stage.use_act,
                input_lane_mask=input_lane_mask,
                active_mask=active,   # ended docs neither pay ponder nor vote on depth
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
                    hs_all.append(h_state[pair])
                    groups.append(batch_idx[pair])
                    if token_weight > 0:
                        next_tok.append(chunk_tensor[:, t + 1][pair])   # (n_active, L) tokens of t+1

        pred_cat = torch.cat(preds, 0) if preds else None
        cos = (scaled_cosine_loss(pred_cat, torch.cat(targets, 0), k=self.cfg.cosine_loss_k)
               if preds else torch.zeros((), device=device))
        # Anti-collapse on the PREDICTIONS themselves (the encoder-side twin is
        # `var` below). Dormant at the default weight 0.0 -> byte-identical.
        pvw = pred_var_weight
        pred_var = (prediction_variance_loss(pred_cat)
                    if (pvw > 0 and pred_cat is not None) else torch.zeros((), device=device))
        group_cat = torch.cat(groups, 0) if (contrastive_hard and groups) else None
        contrast = (prediction_contrastive_loss(pred_cat, torch.cat(targets, 0),
                                                contrastive_temp, group_ids=group_cat)
                    if (contrastive_weight > 0 and pred_cat is not None) else torch.zeros((), device=device))
        # Diagnose WHERE a collapse lives: pred_collapse is pred_head's OUTPUT, but if
        # the loop's h_state is itself constant the information is already gone upstream
        # and no predictor-side term can recover it (that distinction decides whether a
        # collapsed checkpoint can be RESUMED or must be retrained).
        self.last_pred_collapse = (prediction_collapse_metric(pred_cat)
                                   if pred_cat is not None else 0.0)
        self.last_hstate_collapse = (prediction_collapse_metric(torch.cat(hs_all, 0))
                                     if hs_all else 0.0)
        # TOKEN-GROUNDED prediction (JEPA-Reasoner's next-token phase on our reasoner):
        # decode the PREDICTED next latent through the codec Talker, teacher-forced
        # against chunk t+1's own tokens. Distributional over the vocabulary, so a
        # centroid latent (which beats the cosine/InfoNCE terms) pays high NLL here --
        # it decodes to generic tokens, not THIS chunk's. The prediction is put on the
        # target's norm shell first (cosine trains direction only; the Talker consumes
        # unnormalized -- same rescale generation uses).
        if token_weight > 0 and pred_cat is not None:
            tgt_cat = torch.cat(targets, 0)
            # ref_norm must be a per-row column (N, 1) so it broadcasts over d_latent
            # (keepdim=True) -- same shape predict_next_latent passes to _rescale_to.
            pred_scaled = self._rescale_to(pred_cat, tgt_cat.norm(dim=-1, keepdim=True))
            nll_sum, n_tok = self.score_tokens(torch.cat(next_tok, 0), pred_scaled)
            token_nll = nll_sum / n_tok.clamp_min(1.0)
        else:
            token_nll = torch.zeros((), device=device)
        self.last_token_nll = float(token_nll) if token_weight > 0 else 0.0
        flat_valid = chunk_mask.reshape(batch * n_chunks)
        var = (variance_regularization(chunk_vecs.reshape(batch * n_chunks, -1)[flat_valid])
               if var_weight > 0 else torch.zeros((), device=device))
        if n_valid > 0:
            total_ponder = total_ponder / n_valid
        return (cos_weight * cos + var_weight * var + pvw * pred_var
                + contrastive_weight * contrast + token_weight * token_nll,
                ponder_cost_loss(total_ponder, ponder_weight))

    # ------------------------------------------------------------------
    # Supervised halt gate (experiments.md #2): the ACT-alternative predictor.
    # Structurally a copy of forward_self_supervised, differing only in the depth
    # mechanism -- per-row halt SELECTION + a BCE halt loss instead of the ponder
    # cost. Kept separate so the validated method above stays byte-identical; it
    # is only ever reached via that method's guarded dispatch (halt_mode ==
    # "supervised" and use_act).
    # ------------------------------------------------------------------
    def forward_self_supervised_halt(self, chunk_tensor: torch.Tensor, chunk_mask: torch.Tensor,
                                      raw_token_ids: torch.Tensor, raw_mask: torch.Tensor,
                                      memory: GestaltMemoryBank, role_id: int, stage: StageFlags,
                                      ema: EMATargetEncoder, cos_weight: float = 1.0,
                                      var_weight: float = 0.0, chunk_vecs=None):
        """
        TRM-style supervised halt gate. Per chunk t, the loop runs to the ACT cap
        recording the H-state after every cycle (`hrm_loop.forward_halt_trace`).
        Then, per row:
          * SELECT a halt depth -- the first cycle at/after the min-depth floor
            (h_updates_per_thought) whose halt prob > 0.5, else the cap. The
            SELECTED thought drives the primary SSL prediction + the memory write,
            so training sees the SAME depth inference will use (no train/test
            depth mismatch), and depth is PER ROW (the gain over the ponder path's
            batch-mean vote).
          * SUPERVISE the halting head with BCE toward a self-calibrating target:
            halt when one more cycle would cut the SSL cosine distance by <
            cfg.halt_epsilon. The head reads a DETACHED H-state, so the BCE trains
            only the head -- the primary losses (unchanged) shape the reasoning.

        Returns (cos_weight*cos + var_weight*var, supervised_halt_weight*halt_bce),
        matching forward_self_supervised's 2-tuple contract so the trainer needs no
        change (the second term flows into `total` exactly where the ponder did).
        """
        batch, n_chunks, chunk_len = chunk_tensor.shape
        device = chunk_tensor.device
        cap = self.hrm_loop.act_max_ponder_steps
        floor = max(0, self.hrm_loop.h_updates_per_thought - 1)   # 0-indexed min-depth cycle

        # --- input lane (identical leak guard to forward_self_supervised) ---
        input_lane_kv, input_lane_mask = None, None
        if stage.use_input_lanes:
            aged = memory.filtered_stacked([USER, SYSTEM])
            if aged is not None:
                aged_mask = torch.ones(aged.shape[0], aged.shape[1], dtype=torch.bool, device=device)
                no_raw = raw_token_ids[:, :0]
                no_raw_mask = raw_mask[:, :0]
                input_lane_kv, input_lane_mask = self.input_lane(no_raw, no_raw_mask, aged, aged_mask)

        flat_ids = chunk_tensor.reshape(batch * n_chunks, chunk_len)
        if chunk_vecs is None:
            chunk_vecs = self._encode_real_rows(flat_ids, self.chunk_encoder).reshape(batch, n_chunks, -1)
        with torch.no_grad():
            tgt = self._encode_real_rows(flat_ids, ema.encode).reshape(batch, n_chunks, -1)

        h_state = l_state = None
        preds, targets = [], []
        halt_logits_all, cos_dist_all, sup_mask_all = [], [], []
        for t in range(n_chunks):
            active = chunk_mask[:, t]
            if not active.any():
                continue
            trace = self.hrm_loop.forward_halt_trace(          # (cap, batch, d_latent)
                chunk_vecs[:, t], memory, input_lane_kv,
                h_state=h_state, l_state=l_state,
                grad_window=stage.inner_loop_grad_window,
                input_lane_mask=input_lane_mask)

            # Halt logits/probs on a DETACHED trace (BCE trains only the head).
            flat = trace.detach().reshape(cap * batch, -1)
            logits = self.hrm_loop.halting_head.logit(flat).reshape(cap, batch)   # (cap, batch)
            probs = torch.sigmoid(logits)

            # Per-row selected depth: first cycle >= floor with prob>0.5, else cap-1.
            halt_ok = probs.detach() > 0.5
            if floor > 0:
                halt_ok[:floor] = False
            has = halt_ok.any(0)
            first = torch.argmax(halt_ok.float(), dim=0)                          # 0 if none
            sel = torch.where(has, first, torch.full_like(first, cap - 1)).clamp_min(floor)
            h_sel = trace[sel, torch.arange(batch, device=device)]                # (batch, d), keeps grad

            l_state = h_sel
            write_vec = h_sel.detach() if stage.detach_memory else h_sel
            memory.write(write_vec, role_id)
            memory.apply_grad_truncation(stage.memory_grad_window)
            h_state = h_sel

            if t + 1 < n_chunks:
                pair = active & chunk_mask[:, t + 1]
                if pair.any():
                    preds.append(self.pred_head(h_sel)[pair])
                    targets.append(tgt[:, t + 1][pair])
                    # Per-cycle cosine distance of pred_head(h_c) to the target
                    # (the BCE label; detached, built under no_grad).
                    with torch.no_grad():
                        pred_c = self.pred_head(trace.detach().reshape(cap * batch, -1)).reshape(cap, batch, -1)
                        tgt_next = tgt[:, t + 1].unsqueeze(0).expand(cap, -1, -1)
                        cos_dist = 1.0 - F.cosine_similarity(pred_c, tgt_next, dim=-1)   # (cap, batch)
                    sup = pair.unsqueeze(0).expand(cap, -1).clone()
                    if floor > 0:
                        sup[:floor] = False                    # no halting below the min-depth floor
                    halt_logits_all.append(logits)
                    cos_dist_all.append(cos_dist)
                    sup_mask_all.append(sup)

        cos = (scaled_cosine_loss(torch.cat(preds, 0), torch.cat(targets, 0), k=self.cfg.cosine_loss_k)
               if preds else torch.zeros((), device=device))
        flat_valid = chunk_mask.reshape(batch * n_chunks)
        var = (variance_regularization(chunk_vecs.reshape(batch * n_chunks, -1)[flat_valid])
               if var_weight > 0 else torch.zeros((), device=device))
        if halt_logits_all:
            halt = supervised_halt_loss(
                torch.cat(halt_logits_all, dim=1), torch.cat(cos_dist_all, dim=1),
                torch.cat(sup_mask_all, dim=1), epsilon=self.cfg.halt_epsilon,
                target_mode=self.cfg.halt_target)
        else:
            halt = torch.zeros((), device=device)
        return cos_weight * cos + var_weight * var, self.cfg.supervised_halt_weight * halt

    @torch.no_grad()
    def predict_next_latent(self, latent, memory, h_state=None, l_state=None,
                             grad_window: int = 5, use_act: bool = False,
                             reuse_thought: bool = False):
        """
        Inference-time next-latent prediction (generate.py): run the inner loop on
        `latent` and read the next latent off the finished thought via pred_head
        (the same map forward_self_supervised trains). Returns (pred_latent,
        thought); the thought is the carried loop state.

        `reuse_thought=True` skips the loop pass and reads pred_head directly off
        the caller's `h_state` -- for the case where the loop has ALREADY run on
        `latent` (generate.read_prompt runs it on every prompt chunk, including
        the last). Re-running it would ingest the same chunk twice from a state
        that already contains it, reading a memory holding its own thought -- a
        configuration training never produces. This matches the training
        convention pred_head(h_t) -> chunk t+1 exactly.

        The SSL loss is a scaled COSINE (scale-invariant), so pred_head learns the
        target's direction but its output norm is unconstrained (measured ~0.6x the
        encoder-latent norm on a trained checkpoint). The Talker consumes the latent
        as unnormalized cross-attention K/V, so a mis-scaled latent shifts it off the
        distribution it was trained on. Rescale the prediction onto the encoder-latent
        shell: encoder latents end in a LayerNorm, so their norms concentrate tightly
        and the incoming `latent`'s own norm is the right target (fall back to
        sqrt(d) if the caller passed a zero vector, e.g. an empty prompt).
        """
        if reuse_thought and h_state is not None:
            h = h_state
        else:
            h, _ = self.hrm_loop(latent, memory, None, h_state=h_state, l_state=l_state,
                                  grad_window=grad_window, use_act=use_act)
        pred = self.pred_head(h)
        tgt_norm = latent.norm(dim=-1, keepdim=True)
        tgt_norm = torch.where(tgt_norm > 1e-3, tgt_norm,
                               torch.full_like(tgt_norm, self.cfg.d_latent ** 0.5))
        pred = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-6) * tgt_norm
        return pred, h

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

    # ==================================================================
    # Stage F (chatbot fine-tuning, §4). Everything below is ADDITIVE: these
    # are new methods (no new nn.Parameters, so the A-E state_dict is
    # byte-identical and A-E resume is unaffected) and no A-E code path calls
    # them. The one Stage-F-only learned parameter -- the response seed -- lives
    # in dialogue.DialogueAdapter and is passed in, keeping it out of the base
    # model. The A-E forwards (forward_grounded / forward_self_supervised) are
    # untouched.
    # ==================================================================
    @staticmethod
    def _talker_target_mask(ids: torch.Tensor, chunk_len: int) -> torch.Tensor:
        """The supervised-position mask forward_grounded uses: every real token
        plus the first PAD (end-of-chunk stop) of a shorter-than-max chunk, so
        decoding can learn to terminate (§19.2). Duplicated from forward_grounded
        deliberately -- that method is audited and must stay byte-identical, so
        this shares the *logic* without editing it."""
        tok_mask = ids != 0
        lengths = tok_mask.sum(-1)
        target_mask = tok_mask
        has_end = (lengths > 0) & (lengths < chunk_len)
        if has_end.any():
            target_mask = tok_mask.clone()
            target_mask[has_end, lengths[has_end]] = True
        return target_mask

    def score_tokens(self, token_ids: torch.Tensor, latent: torch.Tensor,
                     memory: GestaltMemoryBank = None):
        """
        Teacher-forced token NLL of `token_ids` decoded from an EXTERNALLY
        supplied `latent`, under the codec Talker. Returns (nll_sum, n_tokens):
        the summed cross-entropy over supervised positions and their count, so a
        caller aggregating over many chunks can token-weight correctly.

        This is the one operation neither A-E forward exposes, and the shared
        primitive behind BOTH the Stage-F generative loss and a future lm-eval
        loglikelihood adapter: forward_grounded scores tokens under
        `encode(tokens)` (the target leaks into its own conditioning -- fine for
        an autoencoder anchor, useless for scoring a prediction), while this
        scores given tokens under a GIVEN latent (e.g. pred_head's forecast).
        Empty memory = the codec convention (§27) unless a bank is passed.
        """
        n_rows = token_ids.shape[0]
        if n_rows == 0:
            zero = latent.sum() * 0.0            # keep the graph/device, value 0
            return zero, zero
        if memory is None:
            memory = GestaltMemoryBank(self.cfg.memory_capacity, self.cfg.d_latent)
        logits = self.talker(token_ids, latent, memory)             # (n, L, vocab)
        target_mask = self._talker_target_mask(token_ids, token_ids.shape[1])
        vocab = logits.shape[-1]
        per_tok = F.cross_entropy(
            logits.reshape(-1, vocab), token_ids.reshape(-1), reduction="none"
        ).reshape(token_ids.shape)
        m = target_mask.float()
        return (per_tok * m).sum(), m.sum()

    @staticmethod
    def _rescale_to(pred: torch.Tensor, ref_norm: torch.Tensor) -> torch.Tensor:
        """Put `pred` on the norm shell of `ref_norm` (per row). The cosine
        objective trains pred_head's direction, not its scale, but the Talker
        consumes latents unnormalized -- so the generative loss (and generation,
        predict_next_latent) must rescale the prediction onto the encoder-latent
        norm the Talker was trained on before decoding."""
        ref_norm = ref_norm.clamp_min(1e-3)
        return pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-6) * ref_norm

    def _open_response(self, response_seed: torch.Tensor, memory: GestaltMemoryBank,
                       input_lane_kv, input_lane_mask, stage, n: int, active=None):
        """The response-initiation thought (§4): the loop reads the user turn
        (only via the input lane -- never compressed into a thought, per §4.1)
        with a learned seed injection, producing the thought whose pred_head is
        the model's opening stance for the reply's first chunk. Returns
        (h, ponder)."""
        seed = response_seed.unsqueeze(0).expand(n, -1)
        return self.hrm_loop(
            seed, memory, input_lane_kv, h_state=None, l_state=None,
            grad_window=stage.inner_loop_grad_window, use_act=stage.use_act,
            input_lane_mask=input_lane_mask, active_mask=active)

    def _gestalt(self, x: torch.Tensor) -> torch.Tensor:
        """Map content into the shared memory space before writing (§4/Q2).
        Identity when the gestalt-readout is disabled (raw write, A-E behavior)."""
        return self.gestalt_readout(x) if self.gestalt_readout is not None else x

    def _write_context(self, memory: GestaltMemoryBank, context_chunks, context_mask,
                       context_roles, context_personas=None):
        """Write prior-turn chunks into memory as role-tagged aged gestalts (§4.1,
        §4.2) BEFORE the response loop, giving it cross-turn memory. Each context
        chunk -> encoder latent -> gestalt-readout -> one slot, tagged with that
        chunk's SPEAKER role (and, if given, conversation-local persona id). Roles/
        personas differ per example, so they are written as per-batch tensors.
        Detached (fixed context). No-op if context_chunks is None/empty."""
        if context_chunks is None or context_chunks.shape[1] == 0:
            return
        B, A, L = context_chunks.shape
        flat = context_chunks.reshape(B * A, L)
        z = self._encode_real_rows(flat, self.chunk_encoder).reshape(B, A, -1)
        for j in range(A):
            col = context_mask[:, j]
            if not bool(col.any()):
                continue
            role = context_roles[:, j] if context_roles.dim() == 2 else int(context_roles[j])
            persona = None
            if context_personas is not None:
                persona = (context_personas[:, j] if context_personas.dim() == 2
                           else int(context_personas[j]))
            # `valid=col`: this guard is a batch-level ANY, so one row having context
            # at j forces a slot on EVERY row. Rows without it get _encode_real_rows'
            # exact-zero latent, and a zero vector plus a real role tag is a fully
            # attendable "the user said <nothing>" slot -- ~45% of context memory on
            # the real corpus, with ~28% of rows entirely fabricated, making a row's
            # h_t depend on its batchmates' context LENGTH. Mark the real rows.
            memory.write(self._gestalt(z[:, j]).detach(), role, persona, valid=col)

    @torch.no_grad()
    def inject_source(self, memory: GestaltMemoryBank, source_chunks: torch.Tensor,
                      source_mask: torch.Tensor, role: int = RETRIEVED) -> int:
        """Latent RAG (§Q3): encode a retrieved source into per-chunk gestalts and
        write them to `memory` tagged RETRIEVED -- provenance-distinct from
        USER/SELF/SYSTEM -- so the loop cross-attends the source's GIST at
        O(#chunks) instead of O(#tokens) in a context window. Fidelity caveat
        (§4.1): a chunk latent is lossy, so for verbatim quotes/numbers ALSO
        ground the Talker on the raw source at decode time (DialogueSession).
        Requires a model built with a 4+-entry role_tags. Returns slots written.

        MECHANISM ONLY: the loop's read of RETRIEVED slots is untrained until a
        retrieval-augmented Stage-F dataset exists -- this wires the path, it does
        not make the model good at RAG."""
        if role >= len(self.cfg.role_tags):
            raise ValueError(
                f"inject_source got role id {role}, but the model has only "
                f"{len(self.cfg.role_tags)} role tags {self.cfg.role_tags}. Build the model "
                f"with a 4+-entry role_tags (e.g. append 'RETRIEVED') to use latent RAG; "
                f"otherwise the memory reader would index its role table out of range.")
        B, A, L = source_chunks.shape
        flat = source_chunks.reshape(B * A, L)
        z = self._encode_real_rows(flat, self.chunk_encoder).reshape(B, A, -1)
        n = 0
        for j in range(A):
            col = source_mask[:, j]
            if not bool(col.any()):
                continue
            # `valid=col` is DEFENSIVE: same batch-level-ANY hazard as _write_context,
            # but currently unreachable -- the only caller is DialogueSession.add_source
            # (B=1), where `col` is always all-True (and write() then collapses it back
            # to None). Marked so a future batched caller cannot reintroduce the phantom.
            #
            # `persona`: a retrieved source has NO speaker, but omitting it is not
            # neutral -- persona_id_tensor maps None -> 0, and 0 is SELF's persona, so
            # the bank would assert the source was the model's own thought (the same bug
            # just fixed in _age_user_turn, which survived here because this line was
            # edited without noticing it). No training convention exists for RETRIEVED
            # (RAG is mechanism-only, never trained), so this id is arbitrary -- it only
            # has to not be SELF's. Clamped like dialogue_data.py:259 does.
            # n_personas >= 2 is checked in ModelConfig.__post_init__ AND re-checked in
            # train_dialogue._apply_feature_flags -- the latter is the one that matters,
            # since --persona mutates persona_tags after the constructor has run, so the
            # constructor's guard never sees the config that would trip it.
            persona = _retrieved_persona(self.cfg) if self.cfg.persona_tags else None
            memory.write(self._gestalt(z[:, j]), role, persona, valid=col)
            n += 1
        return n

    @staticmethod
    def _turn_end_labels(resp_mask: torch.Tensor):
        """Turn-end labels + a trustworthiness mask, both derived from `resp_mask`
        alone (STAGE_F.md §2.1). No data-format change: the label is already in the
        SFT batch, it was just being discarded.

        For the thought h_t formed after ingesting response chunk t:
          target_t = 1  iff chunk t is real AND there is no chunk t+1
          valid_t  = chunk t is real, MINUS the truncation confound below.

        THE TRUNCATION CONFOUND. `chunker.chunk_batch` caps a response at M =
        max_chunks_per_doc, so a row whose mask is all-True either ended exactly
        at M or was cut off at M -- indistinguishable here. Its FINAL position's
        "the turn ends here" label is therefore unknown, so it is masked out.
        Every earlier position's label ("a chunk follows" = 0) is correct whether
        or not the row was truncated, so those are kept: only the one ambiguous
        label per filled row is dropped. Without this the head would learn "turns
        end after exactly M chunks" from the truncation artifact.

        Returns (target (B, M) float, valid (B, M) bool).
        """
        B, M = resp_mask.shape
        nxt = torch.cat(
            [resp_mask[:, 1:], torch.zeros(B, 1, dtype=resp_mask.dtype, device=resp_mask.device)],
            dim=1)                                          # "is there a chunk t+1"
        target = (resp_mask & ~nxt).float()                 # last real chunk -> end
        filled = resp_mask.all(dim=1, keepdim=True)         # (B,1) used every slot -> maybe truncated
        valid = resp_mask & ~(target.bool() & filled)       # drop only the ambiguous final label
        return target, valid

    def forward_dialogue(self, resp_chunks: torch.Tensor, resp_mask: torch.Tensor,
                         user_ids: torch.Tensor, user_mask: torch.Tensor,
                         ema: EMATargetEncoder, response_seed: torch.Tensor, stage,
                         context_chunks=None, context_mask=None, context_roles=None,
                         context_personas=None, var_weight: float = 0.0, chunk_vecs=None,
                         end_head=None, end_grad: bool = False):
        """
        Stage-F supervised fine-tuning in latent space (§4, §5 stage F). One
        example is (user turn -> assistant response); the model learns to
        produce the assistant turn conditioned on the user turn.

        The input/self SEPARATION is enforced here by construction:
          * The user turn enters ONLY as raw tokens in the input lane
            (`user_ids`), cross-attended by the loop -- never written into the
            recurrent state (Layer 1, structural: input_lane output has no path
            to h_state/l_state, see input_lane.py).
          * The assistant response (`resp_chunks`) is the prediction target and
            is NEVER routed into the lane, so the SSL-target leak (§4, the
            flagged pre-Stage-F bug) cannot occur in this path (Layer 2).
          * Response thoughts are written to memory tagged SELF; aged prior
            turns carry their own USER/SELF/SYSTEM tags -- the substrate the
            anti-sycophancy loss (Layer 3) exploits.

        Objective, per assistant chunk t (masked to valid response chunks only --
        the latent-space analog of SFT prompt-masking; the user turn is given,
        never a prediction target):
          h_{t-1} = loop(prev, memory, lane)         # prev = seed for t=0, else z_{t-1}
          cos:  pred_head(h_{t-1})  ~  EMA(z_t)        # predict the next thought
          gen:  Talker(pred_head(h_{t-1})) -> z_t's TRUE tokens   # decode the prediction
        Teacher-forced: the loop always ingests the TRUE previous chunk z_{t-1}.

        TURN-END (§2.1), active only when `end_head` is passed: after ingesting
        chunk t and forming h_t, the head predicts "the turn ends here" against a
        label read straight off `resp_mask` (see _turn_end_labels). This is the
        one thing the A-E objective cannot supply -- PAD ends a CHUNK (§19.2),
        nothing ends a TURN -- and it is what lets DialogueSession.reply stop on
        its own instead of emitting a caller-supplied constant number of chunks.
        `end_grad=False` (default) reads a DETACHED h_t, so the BCE trains only
        the head and never reshapes the reasoning -- the `forward_self_supervised_halt`
        convention. `end_grad=True` lets it shape the thought (the A/B).

        Returns a dict of UNWEIGHTED scalar tensors {cos, gen, var, ponder}, plus
        {end, end_acc, end_n} when `end_head` is given; the driver
        (train_dialogue.py) applies the StageFConfig weights and adds the
        always-on reconstruction anchor separately.
        """
        B, M, L = resp_chunks.shape
        device = resp_chunks.device

        # Current user turn -> input lane (raw tokens). Prior turns -> role-tagged
        # aged gestalts in memory (below). The response (resp_chunks) is never in
        # either -- the Layer-2 leak-free contract.
        input_lane_kv, input_lane_mask = self.input_lane(user_ids, user_mask, None, None)

        flat = resp_chunks.reshape(B * M, L)
        # Reuse a shared online-encoder pass when the caller already ran one for
        # the reconstruction anchor (matches the A-E trainer's single-encode
        # convention -- avoids a second encoder pass over the same chunks).
        z = (chunk_vecs if chunk_vecs is not None
             else self._encode_real_rows(flat, self.chunk_encoder).reshape(B, M, -1))
        with torch.no_grad():
            tgt = self._encode_real_rows(flat, ema.encode).reshape(B, M, -1)

        memory = GestaltMemoryBank(self.cfg.memory_capacity, self.cfg.d_latent)
        self._write_context(memory, context_chunks, context_mask, context_roles, context_personas)
        # Self-thoughts are the model's own turn -> persona 0 (reserved for SELF)
        # when personas are in use, else None.
        self_persona = 0 if context_personas is not None else None

        any_resp = resp_mask.any(dim=1)
        h, ponder = self._open_response(response_seed, memory, input_lane_kv,
                                        input_lane_mask, stage, B, active=any_resp)
        l_state = h
        prev_thought = h
        total_ponder = ponder
        n_loops = 1

        preds, targets = [], []
        gen_sum = torch.zeros((), device=device)
        gen_tok = torch.zeros((), device=device)
        # Turn-end: labels come from resp_mask alone; collected per t and reduced
        # once after the loop (see _turn_end_labels for the truncation masking).
        end_target_all, end_valid_all = (self._turn_end_labels(resp_mask)
                                         if end_head is not None else (None, None))
        end_logits, end_targets, end_valids = [], [], []
        for t in range(M):
            active = resp_mask[:, t]
            if not active.any():          # trailing pad columns (chunks are left-packed)
                continue
            # Predict assistant chunk t from the PREVIOUS thought (seed for t=0).
            pred_t = self.pred_head(prev_thought)[active]
            preds.append(pred_t)
            targets.append(tgt[:, t][active])
            # Generative token loss: decode the TRUE tokens of chunk t from the
            # PREDICTED latent, rescaled onto the encoder-latent shell (below).
            ref_norm = tgt[:, t][active].norm(dim=-1, keepdim=True)
            gen_latent = self._rescale_to(pred_t, ref_norm)
            s, n = self.score_tokens(resp_chunks[:, t][active], gen_latent)
            gen_sum = gen_sum + s
            gen_tok = gen_tok + n
            # Teacher forcing: ingest the TRUE chunk t to form the next thought.
            h, ponder = self.hrm_loop(
                z[:, t], memory, input_lane_kv, h_state=h, l_state=l_state,
                grad_window=stage.inner_loop_grad_window, use_act=stage.use_act,
                input_lane_mask=input_lane_mask, active_mask=active)
            l_state = h
            total_ponder = total_ponder + ponder
            n_loops += 1
            # Self-thought written through the same gestalt-readout as context, so
            # the bank stays homogeneous (§4/Q2). Identity when readout is off.
            write_vec = self._gestalt(h)
            # `valid=active`: DEFENSIVE, not a live fix -- measured SEMANTICALLY inert.
            # An inactive row does NOT keep a stale h (an earlier version of this
            # comment said so and was wrong; hrm_loop.py:320 says the opposite --
            # active_mask gates only the ponder cost and halt vote, so such rows "keep
            # evolving on pad-chunk latents" and write FRESH garbage per column). It is
            # unobservable anyway: resp_mask is a left-packed contiguous prefix, so a
            # row never reactivates and contributes to no loss after its last chunk,
            # and slots are per-row so its junk never reaches a batchmate. Kept because
            # it costs nothing and the invariant ("a slot is real only for rows that
            # had content") should hold by construction, not by luck of left-packing.
            memory.write(write_vec.detach() if stage.detach_memory else write_vec,
                         SELF, self_persona, valid=active)
            memory.apply_grad_truncation(stage.memory_grad_window)
            prev_thought = h
            # Turn-end: `h` here is the thought formed AFTER ingesting chunk t --
            # precisely the state DialogueSession.reply holds when it must decide
            # whether to emit another chunk, so train and serve read the SAME
            # tensor off the SAME head. Detached by default: the BCE trains the
            # head only, never the reasoning (the halt-gate convention).
            if end_head is not None:
                h_end = h if end_grad else h.detach()
                end_logits.append(end_head(h_end[active]).squeeze(-1))
                end_targets.append(end_target_all[:, t][active])
                end_valids.append(end_valid_all[:, t][active])

        cos = (scaled_cosine_loss(torch.cat(preds, 0), torch.cat(targets, 0),
                                  k=self.cfg.cosine_loss_k)
               if preds else torch.zeros((), device=device))
        gen = gen_sum / gen_tok.clamp_min(1.0)
        flat_valid = resp_mask.reshape(B * M)
        var = (variance_regularization(z.reshape(B * M, -1)[flat_valid])
               if var_weight > 0 else torch.zeros((), device=device))
        # `var_weight` only GATES whether the (cheap) variance floor is computed;
        # all four terms are returned RAW and the driver applies StageFConfig
        # weights uniformly. `ponder` is the mean per-loop ponder cost.
        ponder = total_ponder / max(n_loops, 1)
        out = {"cos": cos, "gen": gen, "var": var, "ponder": ponder}
        if end_head is not None:
            if end_logits:
                el = torch.cat(end_logits, 0)
                et = torch.cat(end_targets, 0)
                ev = torch.cat(end_valids, 0)
            else:                          # no real response chunks in the batch
                el = torch.zeros(0, device=device)
                et = torch.zeros(0, device=device)
                ev = torch.zeros(0, dtype=torch.bool, device=device)
            out["end"], out["end_n"] = turn_end_loss(el, et, ev)
            out["end_acc"] = turn_end_accuracy(el, et, ev)
            # end_pos: surviving POSITIVE labels -- the only honest health metric.
            # end_n counts negatives too, and for a FILLED row the truncation mask
            # drops that row's single positive while keeping all its negatives. So a
            # batch of long (M-filling) responses yields end_n=44 / positives=0:
            # BCE -> 0.000, end_acc -> 1.000, end_n -> 44, every logged number
            # "healthy" while the head learns "never end" -- the exact failure this
            # objective exists to prevent, wearing a perfect scorecard. If end_pos is
            # 0 the gate is not being trained, whatever the other numbers say.
            out["end_pos"] = (et * ev.float()).sum()
        return out

    def _premise_gestalt(self, premise_chunks: torch.Tensor, premise_mask: torch.Tensor):
        """Compress a chunked premise into one gestalt vector (masked mean of the
        chunk encoder's latents). This is the aged-input-gestalt representation
        (§4.1) -- written to memory with a role tag so the trust gate can act on
        it. Carries grad (trains the encoder to summarize premises)."""
        z = self.encode_chunks(premise_chunks)                 # (B, C, d)
        valid = premise_mask.float().unsqueeze(-1)             # (B, C, 1)
        pooled = (z * valid).sum(1) / valid.sum(1).clamp_min(1.0)  # (B, d)
        return self._gestalt(pooled)                           # homogeneous with other memory writes

    def forward_anti_sycophancy(self, premise_a_chunks: torch.Tensor, premise_a_mask: torch.Tensor,
                                premise_b_chunks: torch.Tensor, premise_b_mask: torch.Tensor,
                                answer_chunks: torch.Tensor, answer_mask: torch.Tensor,
                                ema: EMATargetEncoder, response_seed: torch.Tensor, stage,
                                agree_weight: float = 1.0, premise_role: int = USER,
                                freeze_escape: bool = False):
        """
        The Layer-3 (§4.3) contrastive step, routed through the TRUST-GATED
        memory reader. `premise_*_a/b` are two user assertions that differ ONLY
        in stance (asserts X vs. asserts not-X); the answer is the same either
        way. Each premise is compressed into a role-tagged (USER) gestalt and
        written to memory; the model opens its response reading ONLY that gestalt
        (no input lane), so the premise's influence flows entirely through the
        gated memory read. The opening stance must be invariant to which was
        asserted -- achievable by driving trust(USER) down (the gate discounts
        the slot's value) while the key path still lets the loop NOTICE it.

        This is also what trains the USER memory path (the Q2 train/serve gap):
        A-E only ever writes SELF, so without this step the USER-tagged read is
        untrained. Returns the scalar loss.

        `freeze_escape` (review #2, option 2): detach the two cheap escape routes
        this loss otherwise flows into -- the response seed (the dominant grad
        sink, ~57 vs the trust gate's ~0.03) and the premise-encoder path -- so
        the syco gradient concentrates on the memory read + trust gate, the
        mechanism it is supposed to train. It cuts only the syco term's grad to
        those params (via detach here), not the SFT terms' in the same backward;
        default off = byte-identical. NB: the loop's L/H transitions still carry
        grad (fully isolating the gate needs a loop-internal change) -- see
        antisycophancy_trust_gate_note.md.
        """
        B = premise_a_chunks.shape[0]
        active = answer_mask.any(dim=1)
        seed = response_seed.detach() if freeze_escape else response_seed
        mem_a = GestaltMemoryBank(self.cfg.memory_capacity, self.cfg.d_latent)
        mem_b = GestaltMemoryBank(self.cfg.memory_capacity, self.cfg.d_latent)
        pg_a = self._premise_gestalt(premise_a_chunks, premise_a_mask)
        pg_b = self._premise_gestalt(premise_b_chunks, premise_b_mask)
        if freeze_escape:
            # Detach the premise content -> the trust gate discounts a FIXED
            # (well-trained-elsewhere) premise representation instead of the loss
            # hiding polarity by re-encoding it. The gate still gets grad: it
            # depends on the role tag, not the content.
            pg_a, pg_b = pg_a.detach(), pg_b.detach()
        mem_a.write(pg_a, premise_role)
        mem_b.write(pg_b, premise_role)
        h_a, _ = self._open_response(seed, mem_a, None, None, stage, B, active=active)
        h_b, _ = self._open_response(seed, mem_b, None, None, stage, B, active=active)
        # Restrict the loss to rows with a real answer: a padded/empty-answer row
        # has tgt0=0 (pad-row zero latent) and would inject a spurious gradient
        # pulling pred toward the zero vector. (The shipped ContrastiveDataset
        # drops empty answers, so this is a robustness guard, not a live bug.)
        if not bool(active.any()):
            return torch.zeros((), device=answer_chunks.device)
        pred_a, pred_b = self.pred_head(h_a)[active], self.pred_head(h_b)[active]
        with torch.no_grad():
            tgt0 = self._encode_real_rows(answer_chunks[:, 0], ema.encode)[active]  # role-invariant truth
        return anti_sycophancy_loss(pred_a, pred_b, tgt0,
                                    k=self.cfg.cosine_loss_k, agree_weight=agree_weight)

