"""
gestalt_memory.py
=================
The persistent gestalt memory (§1.2, §3.6, §4.2): a fixed-capacity FIFO bank
of finished thought vectors, read by cross-attention from the inner HRM loop
(the predictive branch's reader -- the one that trains). The Talker also
holds a `GestaltCrossAttentionReader`, but since the §27 restructure the
codec path always hands it an EMPTY bank (reconstruction conditions on the
chunk's own latent only), so that reader is untrained dead weight kept for
checkpoint compatibility -- do not wire a populated bank into the Talker
without training it first. Writes are *not* detached by default -- gradient
from a later thought's loss is allowed to reach back into the state that
produced an earlier thought, subject to truncation (handled by the caller
via `utils.truncate_gradient_window`, applied to the list of stored vectors
before each cross-attention read).

Role tags (§4.2): every slot also carries a role id (USER / SELF / SYSTEM)
so that cross-attention can learn source-dependent weighting instead of
being forced to blend everything into an undifferentiated context. This is
the mechanism that makes it *possible* to represent "the user asserted X"
separately from "I concluded X" (though §4.3 is explicit that this is only
an affordance, not a guarantee, without a training signal that exploits it).
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class GestaltMemoryBank:
    """
    A per-batch-element FIFO buffer of (vector, role_id) pairs. This is a
    plain container (not an nn.Module) because its "parameters" are just the
    stored activations from the forward pass, not learned weights -- the
    learned piece is the cross-attention reader below.
    """

    def __init__(self, capacity: int, d_model: int):
        self.capacity = capacity
        self.d_model = d_model
        self.vectors: List[torch.Tensor] = []   # each: (batch, d_model)
        self.role_ids: List[int] = []            # parallel list of role ids
        self.persona_ids: List = []              # parallel per-speaker ids (or None)
        self.valids: List = []                   # parallel per-row validity (or None = all rows)

    def write(self, vector: torch.Tensor, role_id, persona_id=None, valid=None) -> None:
        """Push a new thought vector (and its role tag) into the FIFO.

        `role_id` is either a python int (one role for the whole batch -- the
        A-E convention, everything is SELF) OR a (batch,) long tensor giving a
        DIFFERENT role per batch element. Per-element roles are needed once a
        batch mixes provenance -- e.g. multi-turn dialogue where each example's
        aged context has its own speaker sequence (dialogue_data), or RAG where
        some slots are RETRIEVED. The int path is unchanged and byte-identical.

        `persona_id` (optional, int or (batch,) tensor) is the conversation-local
        SPEAKER id for personalized tags; None means "no persona" (the A-E and
        single-turn paths). It is stored in parallel and only consumed when the
        reader has persona_tags enabled.

        `valid` (optional, (batch,) bool tensor) marks WHICH ROWS this slot is real
        for. None (the default, and the whole A-E path) means "real for every row"
        -- and while EVERY slot is None the reader takes its original unmasked
        attention, byte-identical.

        Why this exists: a write puts ONE slot into the bank for the WHOLE batch,
        because the bank is (batch, n_slots, d). A caller that writes per-position
        in a loop guarded by a batch-level `.any()` therefore forces a slot onto
        rows that had no content at that position -- those rows get whatever
        `vector[b]` happens to be, which for `_encode_real_rows` is an exact ZERO
        latent, tagged with a real role. That is not inert: the reader computes
        `kv = stacked + tags`, so a zero vector plus the USER tag is a fully
        attendable "the user said <nothing>" slot. Measured on the real dialogue
        corpus, 45.7% of every row's context memory was fabricated this way and 28%
        of rows had NO real context at all -- and it made a row's h_t depend on its
        batchmates' context length. Pass `valid` so the reader can ignore them."""
        self.vectors.append(vector)
        self.role_ids.append(role_id)
        self.persona_ids.append(persona_id)
        self.valids.append(valid)
        if len(self.vectors) > self.capacity:
            self.vectors.pop(0)
            self.role_ids.pop(0)
            self.persona_ids.pop(0)
            self.valids.pop(0)

    def apply_grad_truncation(self, window: int) -> None:
        """
        Detach all but the trailing `window` entries in-place, implementing
        the outer-memory warmup schedule (§3.6, §5.3): unbounded forward
        reads, bounded backward credit assignment.
        """
        from utils import truncate_gradient_window
        self.vectors = truncate_gradient_window(self.vectors, window)

    def stacked(self) -> Optional[torch.Tensor]:
        """Return (batch, n_slots, d_model), or None if memory is empty."""
        if not self.vectors:
            return None
        return torch.stack(self.vectors, dim=1)

    def role_id_tensor(self, device) -> Optional[torch.Tensor]:
        """Role ids for the stored slots. Returns (n_slots,) when every slot has
        a scalar (int) role -- the A-E path, unchanged -- or (batch, n_slots)
        when any slot carries a per-element role tensor (scalar roles are then
        broadcast to the batch)."""
        if not self.role_ids:
            return None
        if all(not torch.is_tensor(r) for r in self.role_ids):
            return torch.tensor(self.role_ids, device=device, dtype=torch.long)  # (n_slots,)
        batch = next(r.shape[0] for r in self.role_ids if torch.is_tensor(r))
        cols = [r.to(device).long() if torch.is_tensor(r)
                else torch.full((batch,), int(r), device=device, dtype=torch.long)
                for r in self.role_ids]
        return torch.stack(cols, dim=1)                                          # (batch, n_slots)

    def persona_id_tensor(self, device) -> Optional[torch.Tensor]:
        """Conversation-local speaker ids for the slots, same shape convention as
        role_id_tensor. None if no slot carries a persona (unspecified -> 0)."""
        if not self.persona_ids or all(p is None for p in self.persona_ids):
            return None
        ps = [0 if p is None else p for p in self.persona_ids]
        if all(not torch.is_tensor(p) for p in ps):
            return torch.tensor(ps, device=device, dtype=torch.long)             # (n_slots,)
        batch = next(p.shape[0] for p in ps if torch.is_tensor(p))
        cols = [p.to(device).long() if torch.is_tensor(p)
                else torch.full((batch,), int(p), device=device, dtype=torch.long)
                for p in ps]
        return torch.stack(cols, dim=1)                                          # (batch, n_slots)

    def valid_mask(self, device) -> Optional[torch.Tensor]:
        """(batch, n_slots) bool, True where the slot is REAL for that row.

        Returns **None** when no slot carries a validity -- the A-E convention and
        every B=1 serving path -- so the reader keeps its original unmasked
        attention and stays byte-identical. A slot written with `valid=None`
        alongside masked slots broadcasts to all-True (it IS real for every row)."""
        if not self.valids or all(v is None for v in self.valids):
            return None
        batch = next(v.shape[0] for v in self.valids if torch.is_tensor(v))
        cols = [v.to(device).bool() if torch.is_tensor(v)
                else torch.ones(batch, device=device, dtype=torch.bool)
                for v in self.valids]
        return torch.stack(cols, dim=1)                                           # (batch, n_slots)

    def filtered_stacked(self, role_ids_wanted: List[int]):
        """
        Return the subset of stored slots whose role id is in
        `role_ids_wanted`, stacked as (batch, n_matching, d_model). Used by
        the input lane (§4.1) to retrieve *aged input* gestalt summaries
        (typically USER-tagged) as its second context tier, distinct from
        the Reasoner's own unrestricted cross-attention into the full bank.
        Returns None if nothing matches. Only scalar-role slots are filterable
        (this feeds the A-E input lane, which is int-role); per-element-role
        slots are skipped.
        """
        matches = [v for v, r in zip(self.vectors, self.role_ids)
                   if not torch.is_tensor(r) and r in role_ids_wanted]
        if not matches:
            return None
        return torch.stack(matches, dim=1)

    def reset(self) -> None:
        self.vectors = []
        self.role_ids = []
        self.persona_ids = []
        self.valids = []

    def __len__(self) -> int:
        return len(self.vectors)


class GestaltReadout(nn.Module):
    """
    §4 / Q2 fix: project ANY content -- a self-thought H-state OR an encoded
    external chunk (aged user turn, RAG source) -- into a common "gestalt space"
    before it is written to memory, so the bank is homogeneous and the reader
    sees ONE distribution instead of two (H-states on the sqrt(d) shell vs.
    LayerNorm'd encoder latents). A Linear + MagicNorm hard-norm onto the thought
    shell. Initialized to identity, so at the start readout(x) == hard_norm(x):
    a self-thought (already hard-normed) passes through ~unchanged, while
    external content is merely projected onto the same shell -- making enabling
    it a gentle change to fine-tune from. Opt-in via config.gestalt_readout.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from norm import hard_normalize
        return hard_normalize(self.proj(x))


class GestaltCrossAttentionReader(nn.Module):
    """
    Learned cross-attention reader used by both the inner HRM loop and the
    Talker to read from a GestaltMemoryBank. Adds a role-tag to each memory
    slot's key/value so attention can be source-dependent.

    Role tags may be discrete (an nn.Embedding lookup) or SOFT (a learned soft
    mixture over a shared codebook, optionally content-conditioned). A per-slot
    TRUST GATE (scalar or vector) can scale each slot's value by its provenance
    trust. Role ids may be per-slot (n_slots,) OR per-batch-element
    (batch, n_slots); all tag/gate paths broadcast over whichever they get.
    """

    def __init__(self, d_model: int, n_heads: int, n_roles: int, dropout: float = 0.1,
                 soft_role_tags: bool = False, soft_role_codebook: int = 16,
                 trust_gate: bool = False, soft_role_content: bool = False,
                 trust_gate_vector: bool = False, persona_tags: bool = False,
                 n_personas: int = 16, core_qk_norm: bool = False):
        super().__init__()
        # `d_model` here is the width of a stored thought -- d_latent in the
        # widened design (the memory holds chunk-level thoughts, not tokens).
        self.soft_role_tags = soft_role_tags
        self.soft_role_content = soft_role_content and soft_role_tags
        if soft_role_tags:
            # Soft learned provenance: a shared codebook of K prototypes, a
            # per-role soft mixture over it, and a learned temperature -- a slot's
            # tag is a graded blend, not a hard bin, and roles share (hence warm)
            # each other's structure. The discrete role id still selects WHICH
            # mixture; it is not discarded.
            K = soft_role_codebook
            self.role_codebook = nn.Parameter(torch.randn(K, d_model) / (d_model ** 0.5))
            self.role_logits = nn.Parameter(torch.zeros(n_roles, K))   # uniform mix at init
            self.role_log_temp = nn.Parameter(torch.zeros(()))         # learned softness (temp=1 init)
            if self.soft_role_content:
                # Content-conditioned tags: the mixture also depends on the slot's
                # content, so provenance can bend per-slot (e.g. a source that
                # "reads as" contradictory). Zero-init => no effect at start, so
                # this stays a pure role prior until it learns to use content.
                self.content_head = nn.Linear(d_model, K)
                nn.init.zeros_(self.content_head.weight)
                nn.init.zeros_(self.content_head.bias)
        else:
            self.role_embed = nn.Embedding(n_roles, d_model)
        self.trust_gate = trust_gate
        if trust_gate:
            # A gate per slot in (0,1), projected from the role tag, scaling the
            # slot's VALUE (not key). Scalar => uniform discount; VECTOR (Linear
            # d->d) => a per-dimension gate that can discount an "assertion/
            # polarity" subspace while preserving a "topic" subspace -- the finer
            # anti-sycophancy control. Init open (~0.98): zero weight + bias 4.
            out_dim = d_model if trust_gate_vector else 1
            self.trust_proj = nn.Linear(d_model, out_dim)
            nn.init.zeros_(self.trust_proj.weight)
            nn.init.constant_(self.trust_proj.bias, 4.0)
        self.persona_tags = persona_tags
        if persona_tags:
            # Conversation-local per-speaker embedding, added on top of the coarse
            # role tag. Zero-init => inert at start, so enabling it is gentle.
            self.persona_embed = nn.Embedding(n_personas, d_model)
            nn.init.zeros_(self.persona_embed.weight)
        # core_qk_norm: QK-normed attention over SDPA (bias=True so the projections
        # mirror MultiheadAttention). The trust gate makes value != key, so the
        # reader passes a separate `value` in forward -- ModernAttention supports
        # it. Off = the exact stock reader (byte-identical). The query pre-norm
        # stays LayerNorm either way: this flag is QK-norm only, nothing else.
        self.core_qk_norm = core_qk_norm
        if core_qk_norm:
            from modern import ModernAttention
            self.attn = ModernAttention(d_model, n_heads, dropout, qk_norm=True, bias=True)
        else:
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(d_model)

    def _role_tags(self, role_ids: torch.Tensor, content: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Role tag vectors for the given ids, shape (*role_ids.shape, d).
        Discrete lookup when soft is off (byte-identical to the original reader);
        a temperature-softened mixture over the shared codebook when on,
        optionally shifted by a content term. `content` is the slots' stored
        vectors (…, d) and is only consulted when soft_role_content is set."""
        if not self.soft_role_tags:
            return self.role_embed(role_ids)
        logits = self.role_logits[role_ids]                    # (*role_ids.shape, K)
        if content is not None and self.soft_role_content:
            logits = logits + self.content_head(content)       # broadcast per-slot content shift
        temp = self.role_log_temp.exp().clamp_min(1e-2)
        w = torch.softmax(logits / temp, dim=-1)
        return w @ self.role_codebook                          # (…, d)

    def forward(self, query: torch.Tensor, memory: GestaltMemoryBank) -> torch.Tensor:
        """
        query: (batch, d_model) or (batch, seq, d_model) -- the state doing the reading.
        Returns a tensor of the same shape as `query`, the attended-memory readout.
        """
        single_vector = query.dim() == 2
        q = query.unsqueeze(1) if single_vector else query   # (batch, 1 or seq, d_model)

        stacked = memory.stacked()
        if stacked is None:
            # No memory yet (first thought of a document): nothing to attend to.
            return torch.zeros_like(query)

        role_ids = memory.role_id_tensor(query.device)         # (n_slots,) or (batch, n_slots)
        content = stacked if self.soft_role_content else None
        tags = self._role_tags(role_ids, content)              # (…, n_slots, d) or (n_slots, d)
        if tags.dim() == 2:
            tags = tags.unsqueeze(0)                           # (1, n_slots, d) -> broadcast over batch
        # Personalized tag: add the per-speaker embedding (who), on top of the
        # role tag (provenance). Skipped when persona_tags is off or no persona set.
        if self.persona_tags:
            persona_ids = memory.persona_id_tensor(query.device)
            if persona_ids is not None:
                ptags = self.persona_embed(persona_ids)
                if ptags.dim() == 2:
                    ptags = ptags.unsqueeze(0)
                tags = tags + ptags
        kv = stacked + tags
        # Trust gate: scale each slot's VALUE by its provenance trust, leaving the
        # KEY (kv) untouched -- the read can still ATTEND to an untrusted slot but
        # incorporates less of its content. Ungated when off (value == kv).
        value = kv
        if self.trust_gate:
            g = torch.sigmoid(self.trust_proj(tags))           # (…, n_slots, 1 or d)
            value = kv * g
        # Per-row slot validity: a slot is written for the WHOLE batch, so a caller
        # writing per-position can force a slot onto rows that had nothing there
        # (see GestaltMemoryBank.write). None => no slot is masked => the original
        # unmasked call, byte-identical (the A-E path and every B=1 serve).
        kpm, dead = None, None
        vmask = memory.valid_mask(query.device)                # (batch, n_slots) True=real
        if vmask is not None:
            # A row with NO valid slot would be fully masked, and attention over a
            # fully-masked row is NaN, not zero. Leave such rows unmasked (their
            # attention is meaningless either way) and zero their output afterwards
            # -- matching the `stacked is None` branch above, which is the same
            # situation one level up: nothing to attend to.
            dead = ~vmask.any(dim=1)                           # (batch,)
            kpm = (~vmask) & (~dead).unsqueeze(1)              # True = IGNORE this slot
        if self.core_qk_norm:
            out = self.attn(self.norm_q(q), kv=kv, value=value, key_padding_mask=kpm)
        else:
            out, _ = self.attn(self.norm_q(q), kv, value, need_weights=False,
                               key_padding_mask=kpm)
        if dead is not None and bool(dead.any()):
            out = out.masked_fill(dead.view(-1, 1, 1), 0.0)
        return out.squeeze(1) if single_vector else out

    def trust_by_role(self, n_roles: int, device) -> Optional[torch.Tensor]:
        """The learned per-role trust (mean over dims for the vector gate), for
        logging -- e.g. watch trust(USER) fall as anti-sycophancy trains. Uses
        the role prior only (no content term). None if the gate is off."""
        if not self.trust_gate:
            return None
        ids = torch.arange(n_roles, device=device)
        g = torch.sigmoid(self.trust_proj(self._role_tags(ids)))   # (n_roles, 1 or d)
        return g.mean(dim=-1)

    def trust_dims_by_role(self, n_roles: int, device) -> Optional[torch.Tensor]:
        """The full PER-DIMENSION trust gate per role, (n_roles, d) -- vector gate
        only (None if the gate is off or scalar). trust_by_role()'s mean hides a
        discounted polarity subspace: the mean can stay ~0.98 while a few dims fall
        to ~0, which is exactly the behavior the vector gate is meant to learn.
        Logging min/std across these dims makes that subspace observable. Role
        prior only (no content term)."""
        if not self.trust_gate or self.trust_proj.out_features == 1:
            return None
        ids = torch.arange(n_roles, device=device)
        return torch.sigmoid(self.trust_proj(self._role_tags(ids)))   # (n_roles, d)

    def tag_trajectory(self, memory: GestaltMemoryBank, device) -> Optional[torch.Tensor]:
        """The DYNAMIC-tag observation: the soft role-mixture weights per stored
        slot, (…, n_slots, K). Reading it across a speaker's successive turns shows
        how their provenance mixture SHIFTS during the conversation (content-
        conditioned). None unless soft_role_tags is on and memory is non-empty."""
        if not self.soft_role_tags:
            return None
        stacked = memory.stacked()
        if stacked is None:
            return None
        role_ids = memory.role_id_tensor(device)
        content = stacked if self.soft_role_content else None
        logits = self.role_logits[role_ids]
        if content is not None and self.soft_role_content:
            logits = logits + self.content_head(content)
        return torch.softmax(logits / self.role_log_temp.exp().clamp_min(1e-2), dim=-1)
