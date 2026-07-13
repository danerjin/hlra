"""
dialogue.py
===========
Stage F (chatbot fine-tuning, §4) runtime pieces that are NOT part of the base
model:

  * `DialogueAdapter` -- the Stage-F-only learned parameter(s), deliberately
    kept out of `LatentThoughtModel` so the validated A-E `state_dict` stays
    byte-identical (A-E checkpoints load with zero missing/unexpected keys and
    A-E resume is unaffected). Today that is just the learned "response seed":
    the injection the loop reasons from to produce the first reply thought,
    since §4.1 forbids compressing the user turn into a thought -- the loop must
    open the reply by *reading the user turn through the input lane* off a
    learned seed, the latent-space analog of the Talker's start vector.

  * `DialogueSession` -- the serving/inference path, multi-turn. It makes the
    two-lane input/self separation concrete at run time:
      - the CURRENT user turn enters only as raw tokens in the input lane;
      - the loop generates the reply autoregressively in latent space
        (pred_head -> Talker), writing each reply thought to memory tagged SELF;
      - the finished user turn is aged into a USER-tagged gestalt so later turns
        can recall it -- cross-turn memory (§4.2) without ever letting user
        content write the recurrent state.

The training objective that makes a checkpoint usable here lives in
`model.forward_dialogue` / `model.forward_anti_sycophancy`, driven by
`train_dialogue.py`. This module only builds/optimizes the adapter and serves.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from gestalt_memory import GestaltMemoryBank
from model import USER, SELF, SYSTEM, RETRIEVED
from data import PAD


class DialogueAdapter(nn.Module):
    """
    Stage-F-only learned parameters, held apart from the base model.

    `response_seed` (d_latent): the loop's injection for the response-initiation
    thought -- see model._open_response. Initialized to zeros (like the Talker's
    start vector); at zero the loop still conditions on the user turn via the
    input lane, so an untrained adapter degrades gracefully rather than
    catastrophically. Trained jointly with the base model in Stage F.
    """

    def __init__(self, d_latent: int):
        super().__init__()
        self.response_seed = nn.Parameter(torch.zeros(d_latent))


def _encode_user_turn(chunker, cfg, text: str, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize a user turn into the input-lane raw window: the trailing
    `recent_token_window` token ids, left-packed with a validity mask. Mirrors
    dialogue_data.tensorize_sft so training and serving see the same lane."""
    W = cfg.recent_token_window
    ids = chunker.tokenizer.encode(text)
    tail = ids[-W:] if len(ids) > W else ids
    raw = torch.full((1, W), PAD, dtype=torch.long, device=device)
    mask = torch.zeros(1, W, dtype=torch.bool, device=device)
    if tail:
        t = torch.tensor(tail, dtype=torch.long, device=device)
        raw[0, : t.numel()] = t
        mask[0, : t.numel()] = True
    return raw, mask


@torch.no_grad()
def _decode_chunk(model, latent, cfg, temperature: float, greedy: bool,
                  ground_memory=None) -> List[int]:
    """Autoregressively decode one chunk's token ids from an encoder-space
    `latent` via the codec Talker. Empty memory = the codec convention (§27);
    `ground_memory` (a bank of RETRIEVED source gestalts) instead grounds the
    Talker on the raw source for verbatim fidelity (§4.1) -- but the Talker's
    memory_reader is UNTRAINED (§27 dead weight), so this is inert until a
    RAG-augmented Stage-F trains it. Id 0 (PAD) is the trained end-of-chunk stop,
    banned at position 0 to rule out empty chunks."""
    max_len = cfg.max_chunk_len
    mem = ground_memory if ground_memory is not None else GestaltMemoryBank(cfg.memory_capacity, cfg.d_latent)
    ids: List[int] = []
    for _ in range(max_len):
        inp = torch.zeros(1, max_len, dtype=torch.long, device=latent.device)
        for j, g in enumerate(ids):
            inp[0, j] = g
        logits = model.talker(inp, latent, mem)[0, len(ids)]
        if not ids:
            logits[PAD] = -1e9
        if greedy:
            nxt = int(logits.argmax())
        else:
            probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
            nxt = int(torch.multinomial(probs, 1))
        if nxt == PAD:
            break
        ids.append(nxt)
    return ids


class DialogueSession:
    """
    A single multi-turn chat session over a trained Stage-F checkpoint. Holds
    the persistent gestalt memory across turns (never reset between turns -- that
    IS the cross-turn memory, §4.2). One instance per conversation.

    Usage:
        sess = DialogueSession(model, adapter, chunker, cfg)
        print(sess.reply("Hello, who are you?"))
        print(sess.reply("What did I just ask?"))   # can recall via aged USER gestalt
    """

    def __init__(self, model, adapter: DialogueAdapter, chunker, cfg,
                 use_act: bool = True):
        self.model = model
        self.adapter = adapter
        self.chunker = chunker
        self.cfg = cfg
        self.use_act = use_act
        self.device = next(model.parameters()).device
        self.memory = GestaltMemoryBank(cfg.memory_capacity, cfg.d_latent)
        self.source_memory = None   # set by add_source(): RETRIEVED gestalts for decode-time grounding
        # A minimal StageFlags-like view for the loop calls. Inference uses the
        # fully-warmed windows; grad windows are irrelevant under no_grad.
        from model import StageFlags
        self.flags = StageFlags(
            use_hrm_loop=True, detach_memory=True,
            inner_loop_grad_window=cfg.inner_loop_grad_window_end,
            memory_grad_window=cfg.memory_grad_window_end,
            use_act=use_act, use_input_lanes=True)

    @torch.no_grad()
    def reply(self, user_text: str, max_chunks: int = 6,
              temperature: float = 0.9, greedy: bool = False,
              separator: str = " ") -> str:
        model, cfg = self.model, self.cfg
        model.eval()
        # 1. Current user turn -> input lane ONLY (raw tokens, read-only).
        user_ids, user_mask = _encode_user_turn(self.chunker, cfg, user_text, self.device)
        lane_kv, lane_mask = model.input_lane(user_ids, user_mask, None, None)

        # 2. Open the reply: the loop reads the user turn (via the lane) off the
        #    learned seed -> the first reply thought. (Seed injection; the user
        #    turn is never written into the recurrent state -- §4.1.)
        h, _ = model._open_response(self.adapter.response_seed, self.memory,
                                    lane_kv, lane_mask, self.flags, n=1, active=None)
        l_state = h
        d = cfg.d_latent

        # 3. Autoregressive latent generation, conditioned on the lane at every
        #    step. pred_head forecasts the next reply chunk's latent; the Talker
        #    decodes it; the chunk is re-encoded and run through the loop to form
        #    the next thought, which is written to memory tagged SELF.
        out_chunks: List[str] = []
        for _ in range(max_chunks):
            pred = model.pred_head(h)
            pred = model._rescale_to(pred, torch.full((1, 1), d ** 0.5, device=self.device))
            ids = _decode_chunk(model, pred, cfg, temperature, greedy,
                                ground_memory=self.source_memory)
            if not ids:
                break
            out_chunks.append(_decode_ids(self.chunker.tokenizer, ids))
            gen = torch.zeros(1, cfg.max_chunk_len, dtype=torch.long, device=self.device)
            for j, g in enumerate(ids[: cfg.max_chunk_len]):
                gen[0, j] = g
            z = model.chunk_encoder(gen, gen != 0)
            h, _ = model.hrm_loop(
                z, self.memory, lane_kv, h_state=h, l_state=l_state,
                grad_window=cfg.inner_loop_grad_window_end, use_act=self.use_act,
                input_lane_mask=lane_mask, active_mask=None)
            l_state = h
            # Write through the SAME gestalt-readout (and SELF persona) as
            # training's forward_dialogue, so a checkpoint trained with
            # gestalt_readout / persona_tags sees a homogeneous bank at serve time.
            self.memory.write(self.model._gestalt(h.detach()), SELF,
                              0 if self.cfg.persona_tags else None)

        # 4. Age the finished user turn into a USER-tagged gestalt (a compressed
        #    summary) so later turns can recall it -- cross-turn memory (§4.2).
        #    The pooled encoder latent of the user turn is a reasonable summary.
        self._age_user_turn(user_text)
        return separator.join(c for c in out_chunks if c).strip()

    @torch.no_grad()
    def _age_user_turn(self, user_text: str) -> None:
        ct, cm = self.chunker.chunk_batch([user_text])
        ct = ct.to(self.device)
        cm = cm.to(self.device)
        if not bool(cm.any()):
            return
        z = self.model.encode_chunks(ct)                    # (1, C, d)
        valid = cm[0].float().unsqueeze(-1)                  # (C, 1)
        if float(valid.sum()) == 0:
            return
        summary = (z[0] * valid).sum(0) / valid.sum()        # masked mean -> (d,)
        self.memory.write(self.model._gestalt(summary.unsqueeze(0)), USER)

    @torch.no_grad()
    def add_source(self, source_text: str, ground_talker: bool = False) -> int:
        """Latent RAG (§Q3): retrieve/inject a source. Its chunks become RETRIEVED
        gestalts in the persistent memory, so the loop cross-attends the source's
        gist on subsequent replies at O(#chunks). With `ground_talker`, the same
        source is also kept as a decode-time Talker grounding memory for verbatim
        fidelity (inert until a RAG Stage-F trains the Talker's reader). Requires a
        model built with a 4+-entry role_tags. Returns slots injected."""
        ct, cm = self.chunker.chunk_batch([source_text])
        ct, cm = ct.to(self.device), cm.to(self.device)
        n = self.model.inject_source(self.memory, ct, cm, role=RETRIEVED)
        if ground_talker:
            self.source_memory = GestaltMemoryBank(self.cfg.memory_capacity, self.cfg.d_latent)
            self.model.inject_source(self.source_memory, ct, cm, role=RETRIEVED)
        return n


def _decode_ids(tok, ids) -> str:
    ids = [int(i) for i in ids if int(i) != PAD]
    return tok.decode(ids) if ids else ""
