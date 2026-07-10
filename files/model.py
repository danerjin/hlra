"""
model.py
========
Wires every component into the full architecture described in §1-§4:

    tokens --chunker--> chunks --HRM inner loop (Reasoner)--> thoughts
                                        |                         |
                                        v                         v
                                 gestalt memory <---------- Talker (tokens out)

Two independent forward passes are exposed, matching §2's "two losses, two
granularities":

  - `forward_self_supervised`: the cheap, fully-parallel JEPA-style branch
    (§2.1). Does *not* touch the HRM loop, the memory, or the Talker --
    it predicts one chunk's EMA-target latent from the previous chunk's
    online-encoded latent, for every chunk pair in the batch at once.

  - `forward_grounded`: the expensive, sequential branch (§2.2). Walks the
    document's chunks in order, running the HRM inner loop, writing to (and
    reading from) the persistent gestalt memory, and running the Talker to
    get an NLL loss against the realized tokens.

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
            n_roles=len(cfg.role_tags), min_decay=cfg.parcae_min_decay,
            max_decay=cfg.parcae_max_decay, act_max_ponder_steps=cfg.act_max_ponder_steps,
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

        # --- Self-supervised JEPA branch (§2.1) --------------------------
        # `ssl_proj` maps the shared chunk latent into a *separate* SSL space
        # before the prediction loss (BYOL-style projection head). This is the
        # anti-collapse isolation: if the cheap SSL objective wants to collapse,
        # it collapses this head, not the shared self.chunk_encoder that the
        # reconstruction (grounded) path decodes from (§6 open question, resolved
        # toward separate heads by the observed collapse). `latent_predictor` is
        # the online-only predictor on top of the projection. The EMA target
        # (ema_target.py) holds momentum copies of BOTH chunk_encoder and
        # ssl_proj; the caller wires it once initial weights exist.
        self.ssl_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.d_model)
        )
        self.latent_predictor = nn.Linear(cfg.d_model, cfg.d_model)

        # --- Generation predictor (encoder-space next-latent head) --------
        # The SSL branch above predicts in the *projected* SSL space, which the
        # §2.4 collapse fix deliberately isolates from the shared encoder space
        # the HRM loop consumes -- so `latent_predictor` cannot seed generation
        # (its output lives in the wrong space). `gen_predictor` is the map
        # generation actually needs: shared latent of chunk t -> shared latent
        # of chunk t+1. It is trained with BOTH input and target detached
        # (forward_gen_predictor), so its loss reaches only this head: it can
        # neither collapse nor otherwise perturb the shared encoder. Purely a
        # readout for inference (generate.py).
        self.gen_predictor = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.d_model)
        )

    # ------------------------------------------------------------------
    # Self-supervised branch: fully parallel, no sequential dependency.
    # ------------------------------------------------------------------
    def forward_self_supervised(self, chunk_tensor: torch.Tensor, chunk_mask: torch.Tensor,
                                 ema: EMATargetEncoder, cos_weight: float = 1.0,
                                 var_weight: float = 0.0) -> torch.Tensor:
        """
        chunk_tensor: (batch, n_chunks, chunk_len)
        chunk_mask:   (batch, n_chunks) bool

        Predicts chunk_{t+1}'s EMA-target latent from chunk_t's online latent,
        for every valid consecutive pair, batched together (§2.1). Returns

            cos_weight * cosine_prediction_loss + var_weight * variance_reg

        where the variance regularizer (losses.variance_regularization) hard-
        floors the shared latent's per-dim variance so it cannot collapse. The
        cosine term operates in the *projected* SSL space (self.ssl_proj), the
        variance term on the *shared* latent it must protect.
        """
        batch, n_chunks, chunk_len = chunk_tensor.shape
        flat_ids = chunk_tensor.reshape(batch * n_chunks, chunk_len)
        flat_token_mask = (flat_ids != 0)  # pad id is 0 by convention (chunker.py)
        flat_valid = chunk_mask.reshape(batch * n_chunks)

        shared_latents = self.chunk_encoder(flat_ids, flat_token_mask)      # (B*N, d) shared
        online_proj = self.ssl_proj(shared_latents).reshape(batch, n_chunks, -1)

        with torch.no_grad():
            target_latents = ema.encode(flat_ids, flat_token_mask).reshape(batch, n_chunks, -1)

        # Valid (t, t+1) pairs: both chunks must be real, per chunk_mask.
        pair_valid = chunk_mask[:, :-1] & chunk_mask[:, 1:]         # (batch, n_chunks-1)
        pred = self.latent_predictor(online_proj[:, :-1])           # predict "next" from "current"
        target = target_latents[:, 1:]
        pred = pred[pair_valid]
        target = target[pair_valid]

        cos = (scaled_cosine_loss(pred, target, k=self.cfg.cosine_loss_k)
               if pred.numel() > 0 else torch.zeros((), device=chunk_tensor.device))
        # Variance regularizer on the shared latent (only the real chunks).
        var = variance_regularization(shared_latents[flat_valid]) if var_weight > 0 else \
            torch.zeros((), device=chunk_tensor.device)
        return cos_weight * cos + var_weight * var

    def forward_gen_predictor(self, chunk_tensor: torch.Tensor, chunk_mask: torch.Tensor) -> torch.Tensor:
        """
        Train the encoder-space next-latent head used by generation: predict
        chunk_{t+1}'s *shared* latent from chunk_t's, both computed under
        no_grad, so the loss trains ONLY self.gen_predictor -- gradient-isolated
        from the shared encoder and the SSL branch by construction. The scaled
        cosine loss suffices because the HRM injection LayerNorms the latent
        (chunk_pool_norm), making the injection invariant to positive rescaling.
        """
        batch, n_chunks, chunk_len = chunk_tensor.shape
        flat_ids = chunk_tensor.reshape(batch * n_chunks, chunk_len)
        with torch.no_grad():
            z = self.chunk_encoder(flat_ids, flat_ids != 0).reshape(batch, n_chunks, -1)
        pair_valid = chunk_mask[:, :-1] & chunk_mask[:, 1:]
        pred = self.gen_predictor(z[:, :-1][pair_valid])
        target = z[:, 1:][pair_valid]
        if pred.numel() == 0:
            return torch.zeros((), device=chunk_tensor.device)
        return scaled_cosine_loss(pred, target, k=self.cfg.cosine_loss_k)

    @torch.no_grad()
    def latent_collapse_metric(self, chunk_tensor: torch.Tensor, chunk_mask: torch.Tensor) -> float:
        """
        Mean per-dimension standard deviation of the shared chunk latents over a
        batch. ~0 means the encoder has collapsed to a (near-)constant vector;
        healthy representations sit well above 0. A cheap collapse monitor.
        """
        batch, n_chunks, chunk_len = chunk_tensor.shape
        flat_ids = chunk_tensor.reshape(batch * n_chunks, chunk_len)
        flat_valid = chunk_mask.reshape(batch * n_chunks)
        z = self.chunk_encoder(flat_ids, flat_ids != 0)[flat_valid]
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
    ):
        """
        Returns (nll_loss, ponder_loss, thoughts) where `thoughts` is the
        list of per-chunk H-state vectors produced (useful for logging/eval).
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
            logits = self.talker(chunk_ids, h_state, memory)
            chunk_nll = grounded_nll_loss(logits, chunk_ids, chunk_token_mask)
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
