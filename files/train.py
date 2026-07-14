"""
train.py
========
Entry point tying everything together: builds the model, runs the §5 staged
curriculum (A -> F), gating stage transitions on validation-loss plateaus
(§5.7.2), and applying the two-loss interleave with a frequency floor on the
expensive grounded loss once the self-supervised loss joins in at Stage D
(§5.7.1).

Data (data.py) is chunked *in the pipeline*, so each batch already arrives as
(chunk_tensor, chunk_mask, raw_ids, raw_mask). Two tiers:
  * OFFLINE (default): synthetic text corpus + stub SaT-Capped chunker, no
    downloads -- runs anywhere.
  * REAL (opt-in via env LATENT_USE_HF=1): streams the config.DataConfig
    mixture from the HuggingFace Hub, segmented by the real SaT model.

Run with:  python train.py                 # offline synthetic text
           LATENT_USE_HF=1 python train.py  # real streaming mixture (needs deps+network)
"""
from __future__ import annotations

import os
import random

import torch
from torch.utils.data import DataLoader

from config import ModelConfig, TrainConfig, DataConfig
from model import LatentThoughtModel, USER, SELF
from gestalt_memory import GestaltMemoryBank
from ema_target import EMATargetEncoder
from curriculum import Curriculum, Stage
from data import (
    DocumentChunkDataset, DialogueTextCorpus, SyntheticTextCorpus,
    collate_chunked, chunk_text_example, iter_hf_mixture, build_offline_chunker,
)
from utils import set_seed


# ----------------------------------------------------------------------
# Chunker / data-source construction
# ----------------------------------------------------------------------
def build_sat_chunker(model_cfg: ModelConfig, data_cfg: DataConfig):
    """
    Build the *real* chunker matching Thought Gestalt's "SaT Capped"
    preprocessing (§5.1): SaT-predicted sentence boundaries plus a
    punctuation-aware fallback capping sentences to `max_chunk_len` tokens.
    Returns (chunker, vocab_size). The tokenizer is wrapped so id 0 stays
    reserved for PAD, so vocab_size == base_vocab + 1.
    """
    from wtpsplit import SaT
    from transformers import AutoTokenizer
    from chunker import SegmentAnyTextChunker
    from data import ReservePadTokenizer, PAD

    sat_model = SaT(data_cfg.sat_model_name)                       # downloads from HF hub
    # SaT sentence segmentation is the prep bottleneck; on CPU it's ~seconds per
    # doc, making a 1.2B-token prep take WEEKS. During prep the GPU is idle (prep
    # runs before training), so move SaT there -- forward-only inference, so the
    # LayerNorm-BACKWARD kernel bug (STRIX_HALO) does not apply. Env override:
    # LATENT_SAT_DEVICE=cpu to force CPU, or =cuda/mps to pin.
    import os as _os
    import torch as _torch
    _sat_dev = _os.environ.get(
        "LATENT_SAT_DEVICE", "cuda" if _torch.cuda.is_available() else "cpu")
    if _sat_dev != "cpu":
        try:
            sat_model.half().to(_sat_dev)
            print(f"[chunker] SaT on {_sat_dev} (half) -- GPU-accelerated segmentation", flush=True)
        except Exception as e:  # any ROCm/device hiccup -> stay on CPU, still correct
            print(f"[chunker] SaT GPU move failed ({e}); staying on CPU", flush=True)
    base = AutoTokenizer.from_pretrained(data_cfg.tokenizer_name)
    base.model_max_length = int(1e12)  # whole-doc tokenization; silence >1024 warnings
    tokenizer = ReservePadTokenizer(base)
    chunker = SegmentAnyTextChunker(
        sat_model=sat_model, tokenizer=tokenizer,
        max_chunk_len=model_cfg.max_chunk_len, max_chunks_per_doc=model_cfg.max_chunks_per_doc,
        pad_token_id=PAD,
    )
    return chunker, tokenizer.vocab_size


def build_pipeline(model_cfg: ModelConfig, data_cfg: DataConfig, use_real: bool):
    """
    Returns (chunker, train_text_factory, val_text_factory). REAL streams the
    configured mixture; OFFLINE uses a synthetic text corpus + stub chunker.
    Mutates model_cfg.vocab_size to match the tokenizer on the real path.
    """
    if use_real:
        chunker, vocab_size = build_sat_chunker(model_cfg, data_cfg)
        model_cfg.vocab_size = vocab_size
        train_factory = lambda: iter_hf_mixture(data_cfg)
        val_factory = lambda: iter_hf_mixture(data_cfg)  # streaming; may overlap (reference impl)
        print(f"[data] REAL mixture: {[s.hf_id for s in data_cfg.sources]}  vocab={vocab_size}")
        return chunker, train_factory, val_factory

    chunker = build_offline_chunker(model_cfg)
    train_factory = lambda: iter(SyntheticTextCorpus(n_docs=512, seed=0))
    val_factory = lambda: iter(SyntheticTextCorpus(n_docs=64, seed=1))
    print("[data] OFFLINE synthetic text + stub SaT-Capped chunker (set LATENT_USE_HF=1 for real data)")
    return chunker, train_factory, val_factory


def make_loader(text_factory, chunker, model_cfg, train_cfg, data_cfg,
                max_examples=None, max_tokens=None):
    ds = DocumentChunkDataset(text_factory, chunker, model_cfg.recent_token_window,
                              min_chunks=data_cfg.min_chunks, max_examples=max_examples,
                              max_tokens=max_tokens)
    return DataLoader(ds, batch_size=train_cfg.batch_size, collate_fn=collate_chunked)


def _to_device(batch, device):
    return tuple(t.to(device) for t in batch)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------


def train_stages_a_to_e(model, ema, curriculum: Curriculum, model_cfg, train_cfg,
                         optimizer, train_loader, val_loader, max_global_steps=None,
                         metrics=None):
    """Stages A-E train on generic long-document text, no speaker roles (§5.6)."""
    device = next(model.parameters()).device
    train_iter = iter(train_loader)

    global_step = 0
    while curriculum.stage.value <= Stage.E.value:
        if max_global_steps is not None and global_step >= max_global_steps:
            print(f"[reached max_global_steps={max_global_steps} in stage {curriculum.stage.name}]")
            break
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        chunk_tensor, chunk_mask, raw_ids, raw_mask = _to_device(batch, device)

        stage_flags = curriculum.stage_flags()
        loss_plan = curriculum.loss_plan()

        optimizer.zero_grad()
        total_loss = None
        logs = {"stage": curriculum.stage.name}

        # One shared online encoder pass, reused by both branches this step.
        chunk_vecs = model.encode_chunks(chunk_tensor)

        if loss_plan.use_grounded_loss:
            # Autoencoder anchor (encoder -> Talker, no loop), always on.
            nll = model.forward_grounded(chunk_tensor, chunk_mask, chunk_vecs=chunk_vecs)
            total_loss = nll
            logs["nll"] = round(nll.item(), 4)

        if loss_plan.use_self_supervised_loss:
            # On-loop SSL (§2.1/§27): the HRM loop predicts the next latent,
            # SEQUENTIALLY, reading its accumulating gestalt memory. Trains the
            # loop + encoder + memory to reason forward; carries the ACT ponder.
            memory = GestaltMemoryBank(model_cfg.memory_capacity, model_cfg.d_latent)
            ssl, ponder = model.forward_self_supervised(
                chunk_tensor, chunk_mask, raw_ids, raw_mask, memory, SELF, stage_flags, ema,
                cos_weight=train_cfg.ssl_loss_weight, var_weight=train_cfg.ssl_var_weight,
                ponder_weight=model_cfg.act_ponder_cost, chunk_vecs=chunk_vecs)
            total_loss = (ssl if total_loss is None else total_loss + ssl) + ponder
            logs["ssl"] = round(ssl.item(), 4)
            logs["ponder"] = round(ponder.item(), 4)

        if total_loss is not None:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            ema.update(model.chunk_encoder)
            logs["loss"] = round(total_loss.item(), 4)

        global_step += 1

        # Periodic validation to feed the plateau-gated curriculum (§5.7.2).
        val_loss = None
        if global_step % train_cfg.log_every == 0:
            val_loss = evaluate(model, ema, val_loader, model_cfg, curriculum)
            logs["lstd"] = round(model.latent_collapse_metric(chunk_tensor, chunk_mask), 4)
            print(f"[step {global_step}] stage={curriculum.stage.name} "
                  f"train_logs={logs} val_loss={val_loss:.4f}")
            if metrics is not None:
                metrics.append({"step": global_step, "stage": curriculum.stage.name,
                                "val_loss": val_loss, "latent_std": logs["lstd"],
                                **{k: v for k, v in logs.items() if k != "stage"}})

        transitioned = curriculum.advance_step(val_loss)
        if transitioned:
            print(f">>> curriculum advanced to stage {curriculum.stage.name}")
        if curriculum.stage == Stage.F:
            break  # hand off to the dialogue fine-tuning loop


@torch.no_grad()
def evaluate(model, ema, val_loader, model_cfg, curriculum: Curriculum) -> float:
    model.eval()
    device = next(model.parameters()).device
    losses = []
    for i, batch in enumerate(val_loader):
        if i >= 4:  # keep validation cheap
            break
        chunk_tensor, chunk_mask, raw_ids, raw_mask = _to_device(batch, device)
        # Reconstruction (autoencoder) NLL only -- the decodability signal, and
        # comparable across stage boundaries (independent of loop/SSL/ponder, so
        # no contaminated-eval jump; notes §5.6). Also what the plateau gate keys on.
        losses.append(model.forward_grounded(chunk_tensor, chunk_mask).item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def train_stage_f(model, ema, curriculum: Curriculum, model_cfg, train_cfg, optimizer, data_cfg):
    """
    Stage F (§5.6): chatbot fine-tuning. Turns on the two-lane input/self
    separation and role tagging; the persistent gestalt memory now spans an
    entire dialogue rather than resetting per document.
    """
    device = next(model.parameters()).device
    dialogues = DialogueTextCorpus(n_dialogues=64, turns=4, seed=0)

    for step, dialogue in enumerate(dialogues):
        # One memory bank persists for the whole dialogue (§4.2: "the gestalt
        # memory doesn't reset per turn").
        memory = GestaltMemoryBank(model_cfg.memory_capacity, model_cfg.d_latent)
        stage_flags = curriculum.stage_flags()
        loss_plan = curriculum.loss_plan()

        optimizer.zero_grad()
        total_loss = torch.zeros((), device=device)  # must match the per-turn losses' device
        n_turns = 0

        for role_id, text in dialogue:
            ct, cm, ri, rm = chunk_text_example(text, model.chunker, model_cfg.recent_token_window)
            ct, cm, ri, rm = _to_device((ct.unsqueeze(0), cm.unsqueeze(0),
                                          ri.unsqueeze(0), rm.unsqueeze(0)), device)
            # Autoencoder anchor (codec) per turn.
            total_loss = total_loss + model.forward_grounded(ct, cm)
            # On-loop SSL per turn: the loop reasons over the dialogue-spanning,
            # role-tagged memory (persists across turns, §4.2). Writes carry this
            # turn's role_id.
            if loss_plan.use_self_supervised_loss:
                ssl, ponder = model.forward_self_supervised(
                    ct, cm, ri, rm, memory, role_id, stage_flags, ema,
                    cos_weight=train_cfg.ssl_loss_weight, var_weight=train_cfg.ssl_var_weight,
                    ponder_weight=model_cfg.act_ponder_cost)
                total_loss = total_loss + ssl + ponder
            n_turns += 1

        total_loss = total_loss / max(n_turns, 1)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        ema.update(model.chunk_encoder)

        if step % train_cfg.log_every == 0:
            print(f"[stage F step {step}] loss={total_loss.item():.4f}")


def main():
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    data_cfg = DataConfig()
    set_seed(train_cfg.seed)

    use_real = os.environ.get("LATENT_USE_HF", "0") == "1"
    chunker, train_factory, val_factory = build_pipeline(model_cfg, data_cfg, use_real)

    model = LatentThoughtModel(model_cfg, chunker).to(train_cfg.device)
    ema = EMATargetEncoder(model.chunk_encoder, momentum=model_cfg.ema_momentum).to(train_cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    curriculum = Curriculum(model_cfg, train_cfg)

    train_loader = make_loader(train_factory, chunker, model_cfg, train_cfg, data_cfg)
    val_loader = make_loader(val_factory, chunker, model_cfg, train_cfg, data_cfg, max_examples=64)

    train_stages_a_to_e(model, ema, curriculum, model_cfg, train_cfg, optimizer, train_loader, val_loader)
    train_stage_f(model, ema, curriculum, model_cfg, train_cfg, optimizer, data_cfg)

    print("Training complete.")


if __name__ == "__main__":
    main()
