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

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from gestalt_memory import GestaltMemoryBank
from model import USER, SELF, SYSTEM, RETRIEVED
from data import PAD


# Opening bias for the turn-end head: sigmoid(-4) ~= 0.018, so a zero-init weight
# makes an untrained head a constant P(end) ~= 1.8% PER CHUNK.
#
# That is NOT inert, and an earlier version of this comment claimed it was ("it
# degrades to today's fixed-length behavior rather than truncating replies at
# random") -- self-refuting, since a 6-chunk reply has 5 chances to stop early and
# 1 - 0.982^5 = 8.7% of them would (18.1% for 12 chunks). Stopping 9% of replies at
# random IS truncating at random. The bias only makes an untrained gate quiet, never
# safe; what actually makes it safe is DialogueSession(use_end_head=False) by
# default, driven by the checkpoint's `end_gate_trained` flag.
_END_BIAS_INIT = -4.0


class DialogueAdapter(nn.Module):
    """
    Stage-F-only learned parameters, held apart from the base model.

    `response_seed` (d_latent): the loop's injection for the response-initiation
    thought -- see model._open_response. Initialized to zeros (like the Talker's
    start vector); at zero the loop still conditions on the user turn via the
    input lane, so an untrained adapter degrades gracefully rather than
    catastrophically. Trained jointly with the base model in Stage F.

    `end_head` (d_latent -> 1): the learned TURN-end gate (STAGE_F.md §2.1).
    P(the turn ends at this thought), read off the thought h_t formed after
    ingesting response chunk t. This is what lets a reply stop on its own; PAD
    (§19.2) only ends a CHUNK. Lives HERE, not in the base model, for the same
    reason response_seed does: the A-E `state_dict` stays byte-identical, so an
    A-E checkpoint loads strict into Stage F and an A-E run is untouched.
    """

    def __init__(self, d_latent: int):
        super().__init__()
        self.response_seed = nn.Parameter(torch.zeros(d_latent))
        # `skip_init`, NOT a plain nn.Linear(...). nn.Linear.__init__ runs
        # reset_parameters(), which CONSUMES global RNG draws -- and the zeros_/
        # constant_ below fix the head's VALUES without rewinding the STREAM. So
        # merely constructing this head shifted every subsequent random draw:
        # with dropout live (cfg.dropout=0.1) that changes every Stage-F dropout
        # mask, and a run with the gate OFF no longer reproduced a pre-gate run
        # (measured: 130/137 base tensors differed after 3 steps at end_weight=0).
        # This head is fully determined and needs no randomness at all, so skip
        # the draw entirely -- unlike get/set_rng_state this holds on any device.
        self.end_head = torch.nn.utils.skip_init(nn.Linear, d_latent, 1)
        nn.init.zeros_(self.end_head.weight)
        nn.init.constant_(self.end_head.bias, _END_BIAS_INIT)


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

    `use_end_head` defaults to **False**: the turn-end gate is used only when the
    caller knows it was trained. An UNTRAINED gate is not inert -- sigmoid(-4.0)
    = 0.018 is a PER-CHUNK fire rate: a 6-chunk reply has 5 chances to stop early,
    so 8.7% of them would (18.1% for 12 chunks) for no reason. Defaulting it on would silently
    truncate replies from any A-E or end_weight=0 checkpoint, a regression the
    gate is supposed to prevent, not cause. `chat_core.load_dialogue_checkpoint`
    reads `end_gate_trained` off the checkpoint and turns it on when it is real.
    """

    def __init__(self, model, adapter: DialogueAdapter, chunker, cfg,
                 use_act: bool = True, use_end_head: bool = False):
        self.model = model
        self.adapter = adapter
        self.chunker = chunker
        self.cfg = cfg
        self.use_act = use_act
        self.use_end_head = use_end_head
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
              separator: str = " ", end_threshold: float = 0.5,
              use_end_head: bool = None) -> str:
        """Generate one assistant turn.

        `max_chunks` is a HARD CAP, not the reply length: with a TRAINED end head
        (StageFConfig.end_weight > 0) the reply stops when P(end) > `end_threshold`.

        `use_end_head=None` (default) inherits the session's setting, which is off
        unless the checkpoint says the gate was trained. Do NOT turn it on for an
        untrained gate: sigmoid(-4.0)=0.018 is a per-CHUNK rate, so ~10% of
        6-chunk replies would stop early at random. Off => exactly the pre-gate
        fixed-length behavior.
        """
        if use_end_head is None:
            use_end_head = self.use_end_head
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
                # Unreachable while _decode_chunk bans PAD at position 0 (it
                # always returns >=1 id); kept as a guard if that ever changes.
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

            # Turn-end gate: `h` is the thought after ingesting the chunk just
            # emitted -- read at the SAME point in the loop, through the SAME head,
            # that forward_dialogue supervises. NOT "identical by construction":
            # with ACT on, the halt vote is a batch mean, so training (B>1) can take
            # a different loop depth than this B=1 serve. Measured inert (the halting
            # head does not discriminate) -- see STAGE_F.md 2.1. This is what makes
            # `max_chunks` a cap rather than the reply's length.
            if use_end_head and getattr(self.adapter, "end_head", None) is not None:
                p_end = torch.sigmoid(self.adapter.end_head(h)).item()
                if p_end > end_threshold:
                    break

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


# ======================================================================
# Self-test for the turn-end gate. Runs offline, no checkpoint, no downloads:
#   .venv/bin/python files/dialogue.py
#
# These four checks exist because each one is a mistake that was actually MADE
# and shipped during this feature's review (see STAGE_F.md 2.1, notes.md):
#   [1] the RNG trap      -- constructing the head shifted every dropout mask
#   [2] the label mask    -- correct, incl. the truncation confound
#   [3] the LYING metric  -- end_n stays healthy while the gate learns "never end"
#   [4] the NULL control  -- beating base-rate entropy is NOT evidence of signal
# ======================================================================
def _self_test() -> int:
    import torch.nn.functional as F
    ok = True

    def chk(cond, msg):
        nonlocal ok
        print(("  PASS  " if cond else "  FAIL  ") + msg)
        ok = ok and bool(cond)

    print("[1] constructing DialogueAdapter must NOT consume global RNG")
    torch.manual_seed(0); ref = torch.randn(4)
    torch.manual_seed(0); a = DialogueAdapter(16); got = torch.randn(4)
    chk(torch.allclose(ref, got),
        "RNG stream unshifted (a plain nn.Linear WOULD shift it, silently changing "
        "every later dropout mask -- 130/137 base tensors once differed this way)")
    chk(bool((a.end_head.weight == 0).all()) and abs(float(a.end_head.bias) - _END_BIAS_INIT) < 1e-9,
        f"end_head is deterministic: weight all-zero, bias {_END_BIAS_INIT}")

    print("[2] _turn_end_labels: labels + the truncation confound")
    from model import LatentThoughtModel
    rm = torch.tensor([[1, 1, 1, 0, 0, 0],    # ends at t=2 -> trustworthy
                       [1, 0, 0, 0, 0, 0],    # single-chunk turn
                       [1, 1, 1, 1, 1, 1]],   # FILLED -> may be truncated
                      dtype=torch.bool)
    tgt, val = LatentThoughtModel._turn_end_labels(rm)
    chk(tgt[0].tolist() == [0, 0, 1, 0, 0, 0], "row0 ends at its last real chunk")
    chk(tgt[1].tolist() == [1, 0, 0, 0, 0, 0], "row1 single-chunk turn ends at t=0")
    chk(val[2].tolist() == [1, 1, 1, 1, 1, 0],
        "row2 (filled) has its AMBIGUOUS final label masked, earlier ones kept")

    print("[3] end_pos, not end_n: an all-filled batch must show ZERO positives")
    rm2 = torch.ones(4, 12, dtype=torch.bool)
    t2, v2 = LatentThoughtModel._turn_end_labels(rm2)
    n_sup, n_pos = int(v2.sum()), int((t2.bool() & v2).sum())
    chk(n_sup > 0 and n_pos == 0,
        f"end_n={n_sup} looks healthy while end_pos={n_pos} -- the masking deletes only "
        f"POSITIVES, so BCE/end_acc go perfect as the head learns 'never end'")

    print("[4] NULL control: beating base-rate entropy proves NOTHING here")
    torch.manual_seed(0)
    D, N, POS = 192, 14, 4
    H = -(POS / N) * math.log(POS / N) - (1 - POS / N) * math.log(1 - POS / N)
    noise = torch.randn(N, D)                       # pure noise features
    lab = torch.zeros(N); lab[torch.randperm(N)[:POS]] = 1.0   # random labels
    head = DialogueAdapter(D).end_head
    opt = torch.optim.Adam(head.parameters(), lr=5e-3)
    for _ in range(400):
        loss = F.binary_cross_entropy_with_logits(head(noise).squeeze(-1), lab)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    chk(float(loss) < H,
        f"a SIGNAL-FREE head on noise+random labels reaches BCE {float(loss):.4f} < the "
        f"{H:.3f} base-rate entropy -- so 'beats H(p)' is NOT evidence the gate learns "
        f"({N} points in {D}-d separate any labeling). Signal needs a HELD-OUT split.")

    print("[5] a row's thought must NOT depend on its batchmates' context length")
    from config import model_config
    from model import StageFlags
    from ema_target import EMATargetEncoder

    class _Stub:
        def __init__(self, c):
            self.tokenizer = None
            self.max_chunk_len = c.max_chunk_len
            self.max_chunks_per_doc = c.max_chunks_per_doc

    torch.manual_seed(0)
    cfg = model_config("smoke"); cfg.vocab_size = 64
    mdl = LatentThoughtModel(cfg, _Stub(cfg)); mdl.eval()
    em = EMATargetEncoder(mdl.chunk_encoder, momentum=0.996)
    adp = DialogueAdapter(cfg.d_latent)
    # end_head.weight is ZERO-init, so end_head(h) == bias for ANY h -- a probe using
    # it as shipped cannot detect a change in h at all (this check was vacuous once,
    # passing even with the mask disabled). Perturb the weight so the head reads h.
    nn.init.normal_(adp.end_head.weight, std=0.05)
    fl = StageFlags(use_hrm_loop=True, detach_memory=False, inner_loop_grad_window=5,
                    memory_grad_window=5, use_act=False, use_input_lanes=True)
    # Record ROW 0's OWN P(end): a batch-mean loss also moves when the batchmate's
    # content changes, which would confound this with a real coupling.
    seen = []

    class _Rec(nn.Module):
        def __init__(self, inner):
            super().__init__(); self.inner = inner

        def forward(self, x):
            seen.append(float(self.inner(x).squeeze(-1)[0])); return self.inner(x)

    rec = _Rec(adp.end_head)
    gg = torch.Generator().manual_seed(5)
    Mx, Lx, Ax = cfg.max_chunks_per_doc, cfg.max_chunk_len, 8

    def _ctx(n):
        cc = torch.zeros(Ax, Lx, dtype=torch.long); cm = torch.zeros(Ax, dtype=torch.bool)
        if n:
            cc[:n] = torch.randint(1, cfg.vocab_size, (n, Lx), generator=gg); cm[:n] = True
        return cc, cm

    rsp = torch.randint(1, cfg.vocab_size, (1, Mx, Lx), generator=gg); rsp[0, 2:] = 0
    rmk = torch.zeros(1, Mx, dtype=torch.bool); rmk[0, :2] = True
    Wx = cfg.recent_token_window
    ui = torch.randint(1, cfg.vocab_size, (1, Wx), generator=gg)
    umk = torch.ones(1, Wx, dtype=torch.bool)
    c0, m0 = _ctx(2)

    def _row0(mate, head=None):
        seen.clear()
        if mate is None:
            cc, cm = c0.unsqueeze(0), m0.unsqueeze(0); R, U, UM, RM = rsp, ui, umk, rmk
        else:
            c1, m1 = _ctx(mate)
            cc = torch.stack([c0, c1]); cm = torch.stack([m0, m1])
            R, U, UM, RM = (torch.cat([rsp] * 2), torch.cat([ui] * 2),
                            torch.cat([umk] * 2), torch.cat([rmk] * 2))
        with torch.no_grad():
            o = mdl.forward_dialogue(R, RM, U, UM, em, adp.response_seed, fl,
                                     context_chunks=cc, context_mask=cm,
                                     context_roles=torch.zeros(cc.shape[0], Ax, dtype=torch.long),
                                     end_head=head if head is not None else rec)
        return list(seen), o

    solo, _ = _row0(None)
    same, _ = _row0(2)
    longer, _ = _row0(6)
    # Tolerance, not equality: B=1 vs B=2 take different attention reductions, so
    # float32 noise is ~5e-7. The bug this guards moves the logit by ~6e-2 -- five
    # orders of magnitude larger -- so 1e-4 separates them with room to spare.
    # (Verified sensitive: stubbing valid_mask -> None makes this FAIL.)
    drift = max(abs(a - b) for a, b in zip(solo, longer)) if solo and longer else 1.0
    drift = max(drift, max(abs(a - b) for a, b in zip(solo, same)) if same else 0.0)
    chk(len(solo) > 0 and drift < 1e-4,
        f"row 0's own P(end) is unchanged by a batchmate with LONGER context "
        f"(max drift {drift:.2e}) -- a batch-level `.any()` once forced phantom "
        f"all-zero USER slots onto shorter rows (45.7% of context memory on the "
        f"real corpus), which moved this by ~6e-2")
    cz, mz = _ctx(0)
    cc = torch.stack([cz, _ctx(4)[0]]); cm = torch.stack([mz, _ctx(4)[1]])
    with torch.no_grad():
        o = mdl.forward_dialogue(torch.cat([rsp] * 2), torch.cat([rmk] * 2),
                                 torch.cat([ui] * 2), torch.cat([umk] * 2), em,
                                 adp.response_seed, fl, context_chunks=cc, context_mask=cm,
                                 context_roles=torch.zeros(2, Ax, dtype=torch.long),
                                 end_head=adp.end_head)
    chk(bool(torch.isfinite(o["cos"])),
        "a row with ZERO real context stays finite -- attention over a fully-masked "
        "row is NaN, not zero, so those rows are zeroed explicitly")

    print("\n[self-test] " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
