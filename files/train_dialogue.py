"""
train_dialogue.py
=================
Stage F (chatbot fine-tuning, §4) driver. STANDALONE on purpose: the A-E
trainer.Trainer is validated and A-E-shaped (4-tuple document batches, breaks at
Stage F), and this is a distinct, not-yet-validated fine-tune. It reuses the
model and the offline data path but does not touch trainer.py, forward_grounded,
or forward_self_supervised.

What it does, per step (see model.forward_dialogue for the objective and the
three-layer separation it enforces):

  reconstruction anchor  (forward_grounded on the assistant chunks)   -- keep the
                          codec from drifting during SFT (always-on anchor, §2.4)
  + cosine SSL           (forward_dialogue['cos'])  -- predict the assistant's
                          next thought latent, masked to SELF chunks (latent SFT)
  + generative NLL        (forward_dialogue['gen'])  -- decode the TRUE assistant
                          tokens from the PREDICTED latent (end-to-end SFT)
  + variance floor        (forward_dialogue['var'])
  + ACT ponder            (forward_dialogue['ponder'])
  + anti-sycophancy       (forward_anti_sycophancy, every syco_every steps) --
                          make the USER/SELF role tags behaviorally load-bearing

Everything runs offline (synthetic dialogues + contrastive pairs) so the path is
exercisable with no downloads and no A-E checkpoint. Point --ckpt at the final
A-E run's model.pt for a real fine-tune.

    python train_dialogue.py --ckpt runs/scaled/model.pt --preset small
    python train_dialogue.py --preset smoke --offline --steps 20   # tiny smoke

THIS SCRIPT IS NOT VALIDATED and has never been run on real dialogue data. It is
the implementation of the Stage-F design, ready for a run -- not a run itself.
"""
from __future__ import annotations

import os
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Keep the HF cache in-project. NOTE: we deliberately do NOT force
# TRANSFORMERS_OFFLINE/HF_HUB_OFFLINE here -- this driver's whole point is to train on
# a real dialogue corpus (--hf-chat), which must be fetched from the hub. huggingface_hub
# honours TRANSFORMERS_OFFLINE as a legacy alias for HF_HUB_OFFLINE, and datasets
# inherits it, so setting it made every --hf-chat run die with "Offline mode is enabled"
# -- the real data was unreachable by construction. (Copied in from generate.py /
# train_real.py, which only ever load the LOCAL gpt2_tok and so were unaffected.)
# data_prep.py, the project's other download-needing script, omits it for this reason.
# --offline and the gpt2 tokenizer stay offline regardless: the stub chunker touches no
# hub, and gpt2_tok is a local dir.
os.environ.setdefault("HF_HOME", os.path.join(PROJECT, ".hf_cache"))

import argparse
import dataclasses

import torch
from torch.utils.data import DataLoader

from config import ModelConfig, StageFConfig, model_config, MODEL_PRESETS
from model import LatentThoughtModel, StageFlags
from ema_target import EMATargetEncoder
from losses import trust_prior_loss
from dialogue import DialogueAdapter
from dialogue_data import (DialogueSFTCorpus, ContrastiveCorpus, DialogueSFTDataset,
                           ContrastiveDataset, collate_sft, collate_contrastive,
                           DialogueTurnsDataset, MultiTurnDialogueCorpus, collate_dialogue_sft)
from utils import set_seed
from rocm_compat import maybe_apply_rocm_workarounds

_LEGACY_CFG_FIELDS = {"parcae_min_decay": "decay_min", "parcae_max_decay": "decay_max"}


def build_chunker(cfg: ModelConfig, offline: bool):
    """gpt2 chunker for a real fine-tune; offline stub chunker for a smoke with
    no downloads. Returns a chunker whose .tokenizer/.chunk_batch the data
    pipeline uses."""
    if offline:
        from data import build_offline_chunker
        return build_offline_chunker(cfg)
    from data import build_regex_gpt2_chunker
    tok_dir = os.path.join(PROJECT, "gpt2_tok")
    chunker, _ = build_regex_gpt2_chunker(cfg, tok_dir if os.path.isdir(tok_dir) else "gpt2")
    return chunker


def _reconcile_role_tables(model, state):
    """Loading a checkpoint with FEWER roles than the model (e.g. enabling RAG:
    3 -> 4 roles adds RETRIEVED) makes role_embed/role_logits mismatch on dim 0,
    which load_state_dict rejects even with strict=False. Pad those tensors:
    copy the trained rows, leave the new role at fresh init."""
    msd = model.state_dict()
    for k, v in list(state.items()):
        if k in msd and v.shape != msd[k].shape:
            mv = msd[k]
            if v.dim() == mv.dim() and v.shape[1:] == mv.shape[1:] and v.shape[0] < mv.shape[0]:
                new = mv.clone()
                new[: v.shape[0]] = v
                state[k] = new
            elif v.dim() == mv.dim() and v.shape[1:] == mv.shape[1:] and v.shape[0] > mv.shape[0]:
                raise ValueError(
                    f"checkpoint '{k}' has {v.shape[0]} roles but the model has {mv.shape[0]} "
                    f"-- loading a higher-role checkpoint into a lower-role model is not "
                    f"supported (drop the extra role or rebuild the model with matching roles).")
    return state


def _apply_feature_flags(cfg, soft_tags, trust_gate, gestalt_readout, vector_gate,
                         content_tags, rag, persona):
    """Turn on the opt-in §4.2/§4.3/§Q2/§Q3 mechanisms for the fine-tune (additive
    to whatever the checkpoint had). content-tags implies soft-tags; rag appends
    the RETRIEVED role; persona adds the per-speaker embedding."""
    cfg.soft_role_tags = cfg.soft_role_tags or soft_tags or content_tags
    cfg.trust_gate = cfg.trust_gate or trust_gate
    cfg.trust_gate_vector = cfg.trust_gate_vector or vector_gate
    cfg.soft_role_content = cfg.soft_role_content or content_tags
    cfg.gestalt_readout = cfg.gestalt_readout or gestalt_readout
    cfg.persona_tags = cfg.persona_tags or persona
    if rag and len(cfg.role_tags) < 4:
        cfg.role_tags = tuple(cfg.role_tags) + ("RETRIEVED",)
    return cfg


def load_base_model(ckpt_path, preset, device, soft_tags=False, trust_gate=False,
                    gestalt_readout=False, vector_gate=False, content_tags=False,
                    rag=False, persona=False):
    """Load an A-E checkpoint's config + weights, or (no ckpt) a fresh model for
    a smoke. Returns (model, cfg). Uses strict=False so legacy/removed modules
    (ssl_proj etc.) are tolerated, exactly like generate.load. `soft_tags` /
    `trust_gate` turn on the §4.2/§4.3 memory mechanisms for the fine-tune: since
    A-E checkpoints have them off, the new tag/gate params are simply absent from
    the state_dict and initialize fresh (reported as 'missing')."""
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw = dict(ckpt["model_cfg"])
        for old, new in _LEGACY_CFG_FIELDS.items():
            if old in raw and new not in raw:
                raw[new] = raw.pop(old)
        known = {f.name for f in dataclasses.fields(ModelConfig)}
        cfg = ModelConfig(**{k: v for k, v in raw.items() if k in known})
        cfg = _apply_feature_flags(cfg, soft_tags, trust_gate, gestalt_readout,
                                   vector_gate, content_tags, rag, persona)
        model = LatentThoughtModel(cfg, chunker=None)
        state = _reconcile_role_tables(model, dict(ckpt["model_state"]))
        missing, unexpected = model.load_state_dict(state, strict=False)
        # Distinguish a top-level module that is ENTIRELY absent from the
        # checkpoint (a real problem -- the whole module is random) from one that
        # is only PARTIALLY fresh because an opt-in Stage-F feature (soft tags,
        # trust gate, gestalt readout) added a few tensors to a module whose bulk
        # DID load. Collapsing both to "missing module X" (the old behavior) made
        # `--soft-tags` on a valid A-E checkpoint print a scary, false
        # "hrm_loop/talker randomly initialized" -- indistinguishable from a
        # genuinely broken load.
        if missing:
            msd = model.state_dict()
            missing_set = set(missing)
            fully, partial = [], []
            for top in sorted({k.split(".")[0] for k in missing}):
                mod_keys = {k for k in msd if k.split(".")[0] == top}
                (fully if mod_keys <= missing_set else partial).append(top)
            if fully:
                print(f"[train_dialogue] WARNING: checkpoint has NO weights for modules "
                      f"{fully} (randomly initialized).")
            if partial:
                print(f"[train_dialogue] note: {len(missing_set)} new parameter(s) "
                      f"initialized fresh (opt-in Stage-F params added to existing "
                      f"modules {partial}); the rest of those modules loaded from the "
                      f"checkpoint.")
        if unexpected:
            # Tensors in the checkpoint the current model has no home for -- they
            # are silently DROPPED. The common case is a reparameterization:
            # --soft-tags removes the discrete `role_embed` (replaced by
            # `role_logits`), so the A-E-trained SELF role vector is discarded
            # here. Warn rather than swallow it (the old behavior).
            print(f"[train_dialogue] note: {len(unexpected)} checkpoint tensor(s) "
                  f"unused by this model and DROPPED (reparameterized/removed modules "
                  f"{sorted({k.split('.')[0] for k in unexpected})}; e.g. --soft-tags "
                  f"discards the trained discrete role_embed).")
        # Resume payload: only a Stage-F checkpoint (written by save() below) carries
        # an adapter/EMA/optimizer state COMPATIBLE with this driver's optimizer (base
        # + adapter param groups). An A-E checkpoint's optimizer is over model params
        # only, so we must NOT load it -- a fine-tune re-seeds EMA/optimizer fresh.
        #
        # Key on `adapter_state`, NOT on stage_reached=='F'. "F" is ALSO the A-E
        # curriculum's TERMINAL stage name (curriculum.Stage.F), and trainer.py saves
        # `curriculum.stage.name` -- so every COMPLETED A-E run's model.pt is stamped
        # "F" as well. Keying on it made the documented A-E -> F handoff misread the
        # foundation checkpoint as a resume and die loading the A-E optimizer into this
        # one ("param group that doesn't match the size of optimizer's group") -- the
        # exact failure the paragraph above says must not happen. It also silently
        # loaded the A-E EMA first. Only save() writes adapter_state, so its presence
        # is the real discriminator; `runs/model.pt` in this repo is the counterexample
        # (stage_reached='F', adapter_state absent).
        resume = None
        if ckpt.get("adapter_state") is not None:
            resume = {"adapter_state": ckpt.get("adapter_state"),
                      "ema": ckpt.get("ema"),
                      "optimizer": ckpt.get("optimizer"),
                      "step": int(ckpt.get("step") or 0),
                      # Carried so main() can guard CLI drift on resume: these two
                      # are CLI-sourced, so resuming without re-passing the original
                      # flags silently changes the run (trainer.py's schedule/halt
                      # guards exist for exactly this failure).
                      "end_gate_trained": ckpt.get("end_gate_trained"),
                      "stage_f_use_act": ckpt.get("stage_f_use_act")}
        print(f"[train_dialogue] loaded base checkpoint {ckpt_path} "
              f"(stage_reached={ckpt.get('stage_reached')}).")
        return model, cfg, resume
    print("[train_dialogue] NO --ckpt given: initializing a FRESH model (smoke only; "
          "a real Stage-F fine-tune must start from a trained A-E checkpoint).")
    cfg = model_config(preset)
    cfg = _apply_feature_flags(cfg, soft_tags, trust_gate, gestalt_readout,
                               vector_gate, content_tags, rag, persona)
    return LatentThoughtModel(cfg, chunker=None), cfg, None


def stage_f_flags(cfg: ModelConfig, use_act: bool = True) -> StageFlags:
    """Stage F: input lanes on, memory un-detached, windows fully warmed, ACT on
    (curriculum.py's Stage-F flags, reconstructed here since this driver does not
    use the A-E Curriculum).

    `use_act=False` (--no-act) is a DIAGNOSTIC, not a recommendation: ACT on is what
    curriculum.py's Stage F specifies and what D/E consolidated with, so a real run
    keeps it. It exists to isolate the turn-end gate from ACT's variable depth in an
    A/B -- ACT's halt vote is a batch mean, so training (B>1) lets batchmates decide a
    row's depth while a B=1 serve halts per-row. That skew is inert while the halting
    head stays undiscriminating (measured: P(halt) in [0.554, 0.674], all above 0.5);
    if it ever bites, the fix is per-row halting (experiments.md #2), not this."""
    return StageFlags(
        use_hrm_loop=True, detach_memory=False,
        inner_loop_grad_window=cfg.inner_loop_grad_window_end,
        memory_grad_window=cfg.memory_grad_window_end,
        use_act=use_act, use_input_lanes=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="A-E checkpoint (model.pt) to fine-tune from")
    ap.add_argument("--preset", default="small", choices=list(MODEL_PRESETS),
                    help="only used to size a FRESH model when --ckpt is omitted")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--offline", action="store_true", help="stub chunker, no downloads (smoke)")
    ap.add_argument("--soft-tags", action="store_true",
                    help="enable soft learned role tags (§4.2); off = discrete tags")
    ap.add_argument("--trust-gate", action="store_true",
                    help="enable the anti-sycophancy trust gate (§4.3) on the memory reader")
    ap.add_argument("--vector-gate", action="store_true",
                    help="make the trust gate per-dimension (discount a polarity subspace)")
    ap.add_argument("--syco-freeze", action="store_true",
                    help="anti-sycophancy: detach the response seed + premise encoder so the "
                         "syco gradient concentrates on the trust gate (review #2, option 2)")
    ap.add_argument("--trust-prior", action="store_true",
                    help="explicit provenance prior: a hinge driving trust(USER) below "
                         "trust(SELF), trained every step (review #2, option 3; needs --trust-gate)")
    ap.add_argument("--end-weight", type=float, default=None,
                    help="learned turn-end BCE weight (STAGE_F.md §2.1). 0 = off (default): the "
                         "model CANNOT end its own turn and reply() emits a fixed chunk count. "
                         "Use ~0.5 for a chatbot you intend to serve.")
    ap.add_argument("--no-act", action="store_true",
                    help="DIAGNOSTIC: run Stage F at fixed depth. ACT ON is the curriculum's "
                         "Stage-F setting (D/E consolidated with it) and is the default -- do "
                         "not use this for a real run. It exists to isolate the turn-end gate "
                         "from ACT's variable depth in an A/B.")
    ap.add_argument("--end-grad", action="store_true",
                    help="let the turn-end BCE shape the thought instead of training only the "
                         "head (default: detached, the supervised-halt-gate convention)")
    ap.add_argument("--content-tags", action="store_true",
                    help="content-condition the soft tags (implies --soft-tags)")
    ap.add_argument("--gestalt-readout", action="store_true",
                    help="homogenize memory writes through a shared readout projection (§Q2)")
    ap.add_argument("--rag", action="store_true",
                    help="add the RETRIEVED role so sources can be injected (§Q3)")
    ap.add_argument("--multi-turn", action="store_true",
                    help="use multi-turn dialogues with role-tagged aged context in memory")
    ap.add_argument("--persona", action="store_true",
                    help="personalized tags: per-speaker embedding (needs --multi-turn for data)")
    # --- real dialogue data (else offline synthetic corpora) ---
    ap.add_argument("--hf-chat", default=None,
                    help="HF chat dataset id (messages format) for real multi-turn SFT")
    ap.add_argument("--hf-transcript", default=None,
                    help="HF dataset id whose --text-field holds a 'SPEAKER: ...' transcript")
    ap.add_argument("--hf-name", default=None, help="HF dataset config/subset name")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-field", default="text", help="transcript text column (--hf-transcript)")
    ap.add_argument("--target-speaker", default=None,
                    help="which speaker is cast as SELF/persona-0 (--hf-transcript, required)")
    ap.add_argument("--system-speakers", default=None,
                    help="comma list of speakers mapped to SYSTEM (--hf-transcript)")
    ap.add_argument("--max-docs", type=int, default=None, help="cap streamed HF documents")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--n-dialogues", type=int, default=4096)
    ap.add_argument("--out", default="runs/dialogue")
    ap.add_argument("--progress", default="auto", choices=["auto", "on", "off"])
    args = ap.parse_args()
    maybe_apply_rocm_workarounds()   # opt-in gfx1151 kernel workarounds (LATENT_MANUAL_LAYERNORM=1)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    sf = StageFConfig()
    if args.steps is not None: sf.steps = args.steps
    if args.batch_size is not None: sf.batch_size = args.batch_size
    if args.lr is not None: sf.lr = args.lr
    if args.end_weight is not None: sf.end_weight = args.end_weight
    if args.end_grad: sf.end_grad = True
    set_seed(sf.seed)

    model, cfg, resume = load_base_model(args.ckpt, args.preset, device,
                                 soft_tags=args.soft_tags, trust_gate=args.trust_gate,
                                 gestalt_readout=args.gestalt_readout, vector_gate=args.vector_gate,
                                 content_tags=args.content_tags, rag=args.rag, persona=args.persona)
    model = model.to(device)
    adapter = DialogueAdapter(cfg.d_latent).to(device)
    ema = EMATargetEncoder(model.chunk_encoder, momentum=cfg.ema_momentum).to(device)
    flags = stage_f_flags(cfg, use_act=not args.no_act)

    # --- Anti-sycophancy measurement guard (Layer-3, 2026-07-14 review #2) ---
    # The contrastive loss is meant to drive trust(USER) down, but the review
    # found it does not train the trust gate as wired (SGD reduces it via the
    # response seed/encoder), and a SCALAR gate is self-defeating for this loss
    # (it discounts topic + polarity together, so it can't fall without hurting
    # the topic signal the loss needs). So when the term is active, insist the run
    # is correctly equipped and observable: require a gate, recommend the vector
    # gate. A warning, not an error, so the scalar-vs-vector A/B stays runnable.
    # See antisycophancy_trust_gate_note.md.
    # Learned turn-end (§2.1). OFF (end_weight=0) reproduces Stage F exactly as it
    # was; ON is what a served chatbot needs -- without it reply() emits a constant
    # number of chunks. Warn loudly when it is off, because the failure is silent:
    # the run trains fine and the checkpoint simply can never end a turn.
    end_on = bool(sf.end_weight > 0)
    if end_on:
        print(f"[train_dialogue] turn-end gate ON (end_weight={sf.end_weight}, "
              f"end_grad={sf.end_grad}): the reply learns to stop. WATCH `end_pos` "
              f"in the log -- it is the only honest health metric. end_n counts "
              f"negatives too, and a batch of long (M-filling) responses has its "
              f"positives masked away: end_pos=0 means the gate is NOT training, "
              f"however good end/end_acc look.", flush=True)
        if flags.use_act:
            # ACT ON is CORRECT here -- it is curriculum.py's Stage-F setting and D/E
            # consolidated with it. Note (not warn) the interaction: hrm_loop's halt
            # vote is a BATCH MEAN, so in training a row's depth is decided by its
            # batchmates while a B=1 serve halts per-row. That could skew the h_t the
            # gate is supervised on -- but only if the halting head DISCRIMINATES
            # between rows, and measured it does not (P(halt) in [0.554, 0.674] on a
            # trained checkpoint, all above 0.5 => batch-mean == per-row). The real
            # fix if that ever changes is per-row halting (experiments.md #2), not
            # turning adaptive depth off.
            print("[train_dialogue] note: gate ON with ACT ON (the curriculum's "
                  "Stage-F setting). ACT's halt vote is a batch mean, so if the "
                  "halting head ever discriminates between rows, training depth (B="
                  f"{sf.batch_size}) and serve depth (B=1) can diverge and the gate "
                  "would be trained on an h_t the server never computes. Measured, "
                  "the head does not discriminate, so this is inert -- watch it, "
                  "don't pre-emptively disable ACT. --no-act is a diagnostic.",
                  flush=True)
    else:
        print("[train_dialogue] NOTE: turn-end gate OFF (end_weight=0). The model "
              "cannot end its own turn -- DialogueSession.reply will emit exactly "
              "max_chunks chunks every time. Pass --end-weight 0.5 for a chatbot "
              "you intend to serve. See STAGE_F.md §2.1.", flush=True)

    # FIFO headroom. forward_dialogue writes up to max_chunks_per_doc context slots PLUS
    # max_chunks_per_doc SELF slots per example, and the bank's pop(0) is driven by the
    # BATCH's write count -- so if capacity is short, a long batchmate's writes evict a
    # SHORT row's real context. `valid` marks a slot dead; it does NOT protect it from
    # eviction, so masking cannot save this. Every shipped preset clears it, but the
    # margin is 2.0x -- NOT "4x+" -- at BOTH 512-d-class presets, INCLUDING `small-w3`,
    # which is what the A-E run actually uses (TRAINING.md 3); `smoke` is 2.67x and the
    # rest 4x. The thinnest margin sits on the preset that matters, which is why this is
    # a checked precondition now rather than a config invariant nothing verified. Enforce it. NB the margin is 2.0x -- NOT "4x+" -- at
    # BOTH 512-d-class presets, including `small-w3`, which is the one the A-E run
    # actually uses (TRAINING.md 3); `smoke` is 2.67x and the rest 4x. The thinnest
    # margin is on the preset that matters.
    _need = 2 * cfg.max_chunks_per_doc
    if cfg.memory_capacity < _need:
        raise SystemExit(
            f"[train_dialogue] memory_capacity={cfg.memory_capacity} < "
            f"2*max_chunks_per_doc={_need}. Stage F writes context + SELF slots per "
            f"example, so the FIFO would evict a short row's real context because a LONG "
            f"batchmate drove the write count -- a batch-composition dependence that the "
            f"per-row validity mask cannot fix (it marks slots dead, not un-evictable). "
            f"Raise memory_capacity to >= {_need}.")

    syco_on = bool(sf.syco_weight > 0 and sf.syco_every)
    if syco_on and not cfg.trust_gate:
        print(f"[train_dialogue] WARNING: anti-sycophancy loss is ON (syco_weight="
              f"{sf.syco_weight}) but no trust gate is enabled. The loss has no "
              f"provenance gate to train, so it can only satisfy itself via the "
              f"response seed/encoder -- it will NOT learn to distrust user "
              f"assertions. Add --trust-gate --vector-gate.", flush=True)
    elif syco_on and not cfg.trust_gate_vector:
        print(f"[train_dialogue] WARNING: anti-sycophancy loss is ON with a SCALAR "
              f"trust gate. A scalar gate discounts topic and polarity together, so "
              f"it cannot be driven down without destroying the topic signal the loss "
              f"needs (review #2) -- expect trust(USER) to barely move. Use "
              f"--vector-gate for the per-dimension polarity-subspace gate. See "
              f"antisycophancy_trust_gate_note.md.", flush=True)
    elif syco_on:
        print(f"[train_dialogue] anti-sycophancy ON with the vector trust gate; "
              f"logging trust(USER) mean + across-dim min/std every {sf.log_every} "
              f"steps (watch a polarity subspace fall while the mean holds).",
              flush=True)
    if syco_on and args.syco_freeze:
        print("[train_dialogue] --syco-freeze ON: response seed + premise encoder "
              "detached for the contrastive term (loop transitions still carry "
              "grad; full gate isolation needs a loop change).", flush=True)
    if args.trust_prior and not cfg.trust_gate:
        raise SystemExit("--trust-prior needs a trust gate to regularize; add --trust-gate "
                         "(and --vector-gate for the polarity-subspace form).")
    if args.trust_prior:
        print(f"[train_dialogue] --trust-prior ON: hinge driving trust(USER) at least "
              f"{sf.trust_prior_margin} below trust(SELF) but not below floor "
              f"{sf.trust_prior_floor}, weight {sf.trust_prior_weight}, trained every step "
              f"(review #2, option 3 -- a first-class provenance signal, not the "
              f"emergent one).", flush=True)

    chunker = build_chunker(cfg, args.offline)
    if args.hf_chat or args.hf_transcript:
        # Real dialogue data: stream turn-lists from the HF loader (dialogue_data),
        # one multi-turn SFT example per SELF turn. Always the multi-turn 8-tuple.
        from dialogue_data import iter_hf_chat_turns, iter_hf_transcript_turns
        if args.hf_chat:
            turns_factory = (lambda: iter_hf_chat_turns(
                args.hf_chat, split=args.split, name=args.hf_name, max_docs=args.max_docs))
            print(f"[train_dialogue] real chat data: {args.hf_chat}")
        else:
            if not args.target_speaker:
                raise SystemExit("--hf-transcript requires --target-speaker (who is SELF)")
            sys_spk = tuple(s for s in (args.system_speakers or "").split(",") if s)
            turns_factory = (lambda: iter_hf_transcript_turns(
                args.hf_transcript, args.text_field, args.target_speaker,
                system_speakers=sys_spk, split=args.split, name=args.hf_name, max_docs=args.max_docs))
            print(f"[train_dialogue] real transcript data: {args.hf_transcript} (SELF={args.target_speaker})")
        sft_ds = DialogueTurnsDataset(turns_factory, chunker, cfg)
        sft_collate = collate_dialogue_sft
    elif args.multi_turn:
        sft_ds = DialogueTurnsDataset(lambda: iter(MultiTurnDialogueCorpus(args.n_dialogues, seed=sf.seed)),
                                      chunker, cfg)
        sft_collate = collate_dialogue_sft
    else:
        sft_ds = DialogueSFTDataset(lambda: iter(DialogueSFTCorpus(args.n_dialogues, seed=sf.seed)),
                                    chunker, cfg)
        sft_collate = collate_sft
    con_ds = ContrastiveDataset(lambda: iter(ContrastiveCorpus(args.n_dialogues, seed=sf.seed)),
                                chunker, cfg)
    sft_loader = DataLoader(sft_ds, batch_size=sf.batch_size, collate_fn=sft_collate)
    con_loader = DataLoader(con_ds, batch_size=sf.batch_size, collate_fn=collate_contrastive)
    con_iter = iter(con_loader)

    # One optimizer over the base model AND the adapter's response seed.
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(adapter.parameters()),
                                  lr=sf.lr, weight_decay=sf.weight_decay)

    # Resume a prior Stage-F run: restore the trained response seed, the EMA
    # target, and the optimizer moments (all written by save() but previously
    # never read back -- a resumed run silently re-zeroed the seed and restarted
    # Adam/EMA cold). Absent for an A-E-foundation start (resume is None).
    start_step = 0
    if resume is not None:
        # A Stage-F checkpoint written BEFORE the turn-end gate has no
        # `end_head.*` in its adapter state, and its optimizer state has two
        # fewer params. Neither is an error -- it is just an older checkpoint --
        # so load non-strictly and say exactly what happened. (A strict load
        # raised "Missing key(s): end_head.weight, end_head.bias" and the
        # optimizer raised a param-group size mismatch, 138 vs 140.)
        # Stage-F drift guard, mirroring trainer.py's schedule/halt guards. Both of
        # these are CLI-sourced (--end-weight / --no-act), so a resume that omits
        # them silently changes the run: ACT flips back on, and `end_gate_trained`
        # regresses True->False while the TRAINED end_head is still sitting in
        # adapter_state -- which would permanently mismark a working gate as
        # untrained, and serving reads that flag to decide whether to use it.
        _drift = [("end_gate_trained", resume.get("end_gate_trained"), end_on),
                  ("stage_f_use_act", resume.get("stage_f_use_act"), flags.use_act)]
        _drift = [(k, o, n) for k, o, n in _drift if o is not None and bool(o) != bool(n)]
        if _drift:
            print("[train_dialogue] " + "!" * 60, flush=True)
            print("[train_dialogue] WARNING: resume Stage-F config differs from the "
                  "checkpoint (weights still load):", flush=True)
            for k, o, n in _drift:
                print(f"[train_dialogue]   {k}: checkpoint={o!r}  now={n!r}", flush=True)
            print("[train_dialogue] Re-pass the original --end-weight / --no-act to "
                  "continue the same run, or proceed if the switch is intentional.",
                  flush=True)
            print("[train_dialogue] " + "!" * 60, flush=True)

        pre_gate = False
        if resume["adapter_state"] is not None:
            missing, unexpected = adapter.load_state_dict(resume["adapter_state"],
                                                          strict=False)
            # ALL of end_head.* missing == a genuine pre-gate checkpoint. A PARTIAL
            # miss (e.g. only end_head.bias) is corruption, not an old file, and must
            # not be waved through as "predates the gate". Exact names, not a prefix:
            # startswith("end_head") would also swallow a future `end_head_foo`.
            _END_KEYS = {"end_head.weight", "end_head.bias"}
            _miss_end = _END_KEYS & set(missing)
            pre_gate = _miss_end == _END_KEYS
            if pre_gate:
                print("[train_dialogue] NOTE: this Stage-F checkpoint predates the "
                      "turn-end gate (no end_head in its adapter state). The seed "
                      "resumed; end_head starts from its untrained init.", flush=True)
            other = [k for k in missing if k not in _END_KEYS] + (
                sorted(_miss_end) if _miss_end and not pre_gate else [])
            if other or unexpected:
                raise SystemExit(f"[train_dialogue] adapter state mismatch beyond the "
                                 f"turn-end gate -- missing={other} unexpected={list(unexpected)}")
        if resume["ema"] is not None:
            ema.load_state_dict(resume["ema"])
        if resume["optimizer"] is not None:
            if pre_gate:
                # The saved moments were built over a 2-params-smaller group, so
                # load_state_dict would ValueError. Adam restarts cold for every
                # param, not just the head -- say so rather than fail or hide it.
                print("[train_dialogue] WARNING: optimizer state is from before the "
                      "turn-end gate (param group grew by 2); starting Adam cold. "
                      "Expect a brief loss bump.", flush=True)
            else:
                optimizer.load_state_dict(resume["optimizer"])
        start_step = resume["step"]
        print(f"[train_dialogue] resumed Stage-F state (response seed + EMA + "
              f"optimizer) from step {start_step}.")

    model.train()
    step, nonfinite_streak = start_step, 0
    end_dry = 0                      # consecutive batches with zero turn-end positives
    out_dir = os.path.join(PROJECT, args.out)
    print(f"[train_dialogue] device={device} d_latent={cfg.d_latent} steps={sf.steps} "
          f"batch={sf.batch_size} lr={sf.lr}  (offline={args.offline})")

    try:
        from tqdm.auto import tqdm
        bar = tqdm(total=sf.steps, initial=start_step,
                   disable=(None if args.progress == "auto" else args.progress == "off"),
                   desc="stage-F", unit="step")
    except Exception:
        bar = None

    while step < sf.steps:
        for batch in sft_loader:
            if step >= sf.steps:
                break
            batch = [t.to(device) for t in batch]
            if len(batch) == 8:     # multi-turn: role+persona-tagged aged context in memory
                (context_chunks, context_mask, context_roles, context_personas,
                 user_ids, user_mask, resp_chunks, resp_mask) = batch
            else:                   # single-turn
                resp_chunks, resp_mask, user_ids, user_mask = batch
                context_chunks = context_mask = context_roles = context_personas = None
            optimizer.zero_grad(set_to_none=True)

            # One shared online-encoder pass, reused by the anchor and the
            # dialogue branch (the A-E trainer's single-encode convention).
            chunk_vecs = model.encode_chunks(resp_chunks)
            # Reconstruction anchor (encoder + Talker codec) on the assistant chunks.
            nll = model.forward_grounded(resp_chunks, resp_mask, chunk_vecs=chunk_vecs)
            dlg = model.forward_dialogue(resp_chunks, resp_mask, user_ids, user_mask,
                                         ema, adapter.response_seed, flags,
                                         context_chunks=context_chunks, context_mask=context_mask,
                                         context_roles=context_roles, context_personas=context_personas,
                                         var_weight=sf.var_weight, chunk_vecs=chunk_vecs,
                                         end_head=(adapter.end_head if end_on else None),
                                         end_grad=sf.end_grad)
            loss = (sf.grounded_weight * nll
                    + sf.cos_weight * dlg["cos"]
                    + sf.gen_weight * dlg["gen"]
                    + sf.var_weight * dlg["var"]
                    + sf.ponder_weight * dlg["ponder"])
            if end_on:
                loss = loss + sf.end_weight * dlg["end"]
                # A batch whose responses all fill max_chunks_per_doc has every
                # positive masked away (only its final label says "end", and that
                # is the ambiguous one). The BCE then trains "never end" on pure
                # negatives while end/end_acc/end_n all look healthy. Count the
                # dry batches and say so -- this failure is otherwise invisible.
                end_dry = end_dry + 1 if int(dlg["end_pos"]) == 0 else 0
                if end_dry == 50:
                    print(f"[train_dialogue] WARNING: 50 consecutive batches with "
                          f"end_pos=0 -- NO turn-end positives are surviving the "
                          f"truncation mask, so the gate is learning 'never end' "
                          f"regardless of what end/end_acc say. Your responses are "
                          f"filling max_chunks_per_doc={cfg.max_chunks_per_doc}; "
                          f"raise it, or use shorter-response data. (If end_n is also "
                          f"0 the batch simply had no response chunks -- a data bug, "
                          f"not a masking one.)", flush=True)

            syco_val = None
            if sf.syco_every and (step % sf.syco_every == 0):
                try:
                    cb = next(con_iter)
                except StopIteration:
                    con_iter = iter(con_loader); cb = next(con_iter)
                pa, pam, pb, pbm, ac, am = (t.to(device) for t in cb)
                syco = model.forward_anti_sycophancy(pa, pam, pb, pbm, ac, am, ema,
                                                     adapter.response_seed, flags,
                                                     agree_weight=sf.syco_agree_weight,
                                                     freeze_escape=args.syco_freeze)
                loss = loss + sf.syco_weight * syco
                syco_val = float(syco)

            # Explicit provenance prior (review #2, option 3): a first-class hinge
            # driving trust(USER) below trust(SELF), trained EVERY step (cheap --
            # role-prior only) so the gate gets a direct signal instead of the
            # emergent-but-tiny one from the contrastive loss.
            tp_val = None
            if args.trust_prior:
                trust = model.hrm_loop.memory_reader.trust_by_role(len(cfg.role_tags), device)
                tp = trust_prior_loss(trust, cfg.role_tags.index("USER"),
                                      cfg.role_tags.index("SELF"), margin=sf.trust_prior_margin,
                                      floor=sf.trust_prior_floor)
                loss = loss + sf.trust_prior_weight * tp
                tp_val = float(tp)

            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(adapter.parameters()), sf.grad_clip)
            # Same non-finite guard as trainer.py: a single NaN grad makes the
            # global clip coefficient NaN and one step would destroy all weights.
            if bool(torch.isfinite(total_norm)):
                optimizer.step()
                nonfinite_streak = 0
            else:
                nonfinite_streak += 1
                print(f"[train_dialogue] WARNING: non-finite grad norm at step {step+1}; "
                      f"skipping ({nonfinite_streak} consecutive).", flush=True)
                if nonfinite_streak >= 25:
                    raise RuntimeError("25 consecutive non-finite gradient steps -- run is dead.")
            ema.update(model.chunk_encoder)
            step += 1
            if bar is not None:
                bar.update(1)

            if sf.log_every and step % sf.log_every == 0:
                msg = (f"[step {step}] nll={float(nll):.4f} cos={float(dlg['cos']):.4f} "
                       f"gen={float(dlg['gen']):.4f} var={float(dlg['var']):.4f} "
                       f"ponder={float(dlg['ponder']):.4f}"
                       + (f" syco={syco_val:.4f}" if syco_val is not None else "")
                       + (f" tprior={tp_val:.4f}" if tp_val is not None else "")
                       # end_acc is imbalanced (~1 'end' per turn): a head that
                       # always says "continue" scores ~1-1/M, so it is never read
                       # alone. end_pos (surviving POSITIVES) is the honest one --
                       # at end_pos=0 the other two look perfect and mean nothing.
                       + (f" end={float(dlg['end']):.4f} end_acc={float(dlg['end_acc']):.3f}"
                          f" end_pos={int(dlg['end_pos'])}/{int(dlg['end_n'])}"
                          if end_on else ""))
                # Watch trust(USER) fall relative to trust(SELF) as anti-sycophancy trains.
                reader = model.hrm_loop.memory_reader
                trust = reader.trust_by_role(len(cfg.role_tags), device)
                if trust is not None:
                    msg += " trust=" + "/".join(f"{r}:{float(t):.2f}"
                                                for r, t in zip(cfg.role_tags, trust))
                    # Vector gate: the per-role mean hides a discounted polarity
                    # subspace (mean holds ~0.98 while a few dims -> 0). Log USER's
                    # across-dim min/std so the subspace is observable (review #2).
                    dims = reader.trust_dims_by_role(len(cfg.role_tags), device)
                    if dims is not None and "USER" in cfg.role_tags:
                        u = dims[cfg.role_tags.index("USER")]
                        msg += (f" trustUSER[min={float(u.min()):.2f} "
                                f"std={float(u.std()):.3f}]")
                (bar.write(msg) if bar is not None else print(msg, flush=True))
            if sf.checkpoint_every and step % sf.checkpoint_every == 0:
                save(out_dir, "checkpoint.pt", model, adapter, ema, optimizer, cfg, step,
                     end_gate_trained=end_on, use_act=flags.use_act)

    if bar is not None:
        bar.close()
    save(out_dir, "model.pt", model, adapter, ema, optimizer, cfg, step,
         end_gate_trained=end_on, use_act=flags.use_act)
    print(f"[train_dialogue] done. {step} steps -> {os.path.join(out_dir, 'model.pt')}")


def save(out_dir, name, model, adapter, ema, optimizer, cfg, step, end_gate_trained=False,
         use_act=True):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save({
            "model_state": model.state_dict(),
            "adapter_state": adapter.state_dict(),   # Stage-F params (seed + end_head)
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_cfg": dataclasses.asdict(cfg),
            "vocab_size": cfg.vocab_size,
            "stage_reached": "F",
            "step": step,
            # Whether the turn-end gate was actually TRAINED (end_weight > 0). The
            # adapter always carries an end_head, so its presence proves nothing --
            # and an untrained gate is NOT inert (P=0.018 per chunk => 8.7% of
            # 6-chunk replies stop early). Serving reads this to decide whether the
            # gate may be used at all, instead of guessing from the weights.
            "end_gate_trained": bool(end_gate_trained),
            # Serving must run the loop the way training did: DialogueSession
            # defaults use_act=True, which would silently mismatch a --no-act run.
            "stage_f_use_act": bool(use_act),
            "tokenizer_name": "gpt2", "chunker": "regex_gpt2",
        }, f)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    print(f"[train_dialogue] checkpoint -> {path} (step {step})", flush=True)


if __name__ == "__main__":
    main()
