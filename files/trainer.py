"""
trainer.py
==========
A scale-ready training loop for Stages A-E, replacing the flat function in
train.py. Adds the infrastructure a real run needs while keeping the exact
loss semantics (reconstruction anchor + secondary SSL with anti-collapse
variance floor) verified on the smoke run:

  * gradient accumulation      (large effective batch on limited memory)
  * mixed precision autocast   (bf16/fp16; enable on CUDA)
  * LR schedule                (linear warmup -> cosine decay)
  * checkpoint + resume        (model, optimizer, EMA, curriculum, step, RNG)
  * curriculum gating          (fixed per-stage budgets, or plateau)
  * collapse monitoring        (latent std logged every eval)

The model/loss objects are unchanged; this only orchestrates them.
"""
from __future__ import annotations

import math
import os
import json
from contextlib import nullcontext
from dataclasses import asdict

import torch

from model import SELF
from gestalt_memory import GestaltMemoryBank
from curriculum import Curriculum, Stage


class Trainer:
    def __init__(self, model, ema, optimizer, curriculum: Curriculum,
                 model_cfg, train_cfg, train_loader, val_loader, ckpt_dir):
        self.model = model
        self.ema = ema
        self.optimizer = optimizer
        self.curriculum = curriculum
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.ckpt_dir = ckpt_dir
        self.device = next(model.parameters()).device

        self.global_step = 0
        self.metrics = []
        self._train_iter = None

        # Mixed precision. Off on CPU; a GradScaler is only needed for CUDA fp16.
        self.use_amp = bool(train_cfg.amp) and self.device.type != "cpu"
        self.amp_dtype = torch.bfloat16 if train_cfg.amp_dtype == "bf16" else torch.float16
        self.scaler = (torch.cuda.amp.GradScaler()
                       if self.use_amp and self.device.type == "cuda" and self.amp_dtype == torch.float16
                       else None)

    # ------------------------------------------------------------------
    def _autocast(self):
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def _lr(self, step: int) -> float:
        base, floor = self.train_cfg.lr, self.train_cfg.min_lr_ratio

        # Per-stage schedule (the curriculum fix): a staged curriculum is five
        # distinct optimization phases, so one global cosine over the whole A->E
        # horizon starves the late stages -- D and E end up training at the
        # cosine's near-zero tail and can't keep learning, which looks like a
        # D/E "regression". Instead, give each stage its own short warm-up +
        # cosine over that stage's own budget, so every stage starts with a
        # usable LR and still anneals to consolidate. (Verified: this removes
        # the D/E reconstruction regression; see notes SS12.)
        ss = getattr(self.train_cfg, "stage_steps", None)
        if getattr(self.train_cfg, "per_stage_lr", False) and ss is not None:
            budget = max(1, ss[self.curriculum.stage_idx])
            s = self.curriculum.step_in_stage
            warm = min(self.train_cfg.warmup_steps, max(1, budget // 10))
            if s < warm:
                return base * (s + 1) / warm
            prog = min(1.0, (s - warm) / max(1, budget - warm))
            return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog)))

        # Global warmup -> cosine (the original single-horizon schedule).
        warm, tot = self.train_cfg.warmup_steps, self.train_cfg.total_steps
        if warm > 0 and step < warm:
            return base * (step + 1) / warm
        if tot > 0:
            prog = min(1.0, (step - warm) / max(1, tot - warm))
            return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog)))
        return base

    def _next_batch(self):
        if self._train_iter is None:
            self._train_iter = iter(self.train_loader)
        try:
            batch = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            batch = next(self._train_iter)
        return tuple(t.to(self.device) for t in batch)

    def _loss_on(self, batch, flags, plan, run_grounded):
        ct, cm, ri, rm = batch
        total, logs = None, {}
        if plan.use_grounded_loss and run_grounded:
            memory = GestaltMemoryBank(self.model_cfg.memory_capacity, self.model_cfg.d_model)
            nll, ponder, _ = self.model.forward_grounded(
                ct, cm, ri, rm, memory, SELF, flags, self.model_cfg.act_ponder_cost)
            total = nll + ponder
            logs["nll"] = round(nll.item(), 4)
            logs["ponder"] = round(ponder.item(), 4)
        if plan.use_self_supervised_loss:
            ssl = self.model.forward_self_supervised(
                ct, cm, self.ema, cos_weight=self.train_cfg.ssl_loss_weight,
                var_weight=self.train_cfg.ssl_var_weight)
            total = ssl if total is None else total + ssl
            logs["ssl"] = round(ssl.item(), 4)
        return total, logs, (ct, cm)

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        flags = self.curriculum.stage_flags()
        losses = []
        for i, batch in enumerate(self.val_loader):
            if i >= 4:
                break
            ct, cm, ri, rm = tuple(t.to(self.device) for t in batch)
            memory = GestaltMemoryBank(self.model_cfg.memory_capacity, self.model_cfg.d_model)
            nll, _, _ = self.model.forward_grounded(
                ct, cm, ri, rm, memory, SELF, flags, self.model_cfg.act_ponder_cost)
            # Reconstruction NLL only: the ponder cost is a training-time
            # compute penalty, not a quality signal, and including it would
            # bump val at the Stage-E boundary exactly the way the
            # contaminated-eval lesson (notes §5.6) warns about.
            losses.append(nll.item())
        self.model.train()
        return sum(losses) / max(len(losses), 1)

    # ------------------------------------------------------------------
    def train(self, max_steps: int):
        import random
        accum = max(1, self.train_cfg.grad_accum_steps)
        while self.curriculum.stage.value <= Stage.E.value and self.global_step < max_steps:
            flags = self.curriculum.stage_flags()
            plan = self.curriculum.loss_plan()
            lr = self._lr(self.global_step)
            for g in self.optimizer.param_groups:
                g["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            step_logs, last_cc = {}, None
            for _ in range(accum):
                batch = self._next_batch()
                run_grounded = (not plan.use_self_supervised_loss
                                or random.random() < self.train_cfg.grounded_loss_min_frequency)
                with self._autocast():
                    loss, logs, cc = self._loss_on(batch, flags, plan, run_grounded)
                last_cc = cc
                if loss is None:
                    continue
                loss = loss / accum
                (self.scaler.scale(loss).backward() if self.scaler is not None else loss.backward())
                step_logs.update(logs)

            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip)
            if self.scaler is not None:
                self.scaler.step(self.optimizer); self.scaler.update()
            else:
                self.optimizer.step()
            self.ema.update(self.model.chunk_encoder, self.model.ssl_proj)
            self.global_step += 1

            val_loss = None
            if self.global_step % self.train_cfg.log_every == 0:
                val_loss = self.evaluate()
                lstd = self.model.latent_collapse_metric(*last_cc) if last_cc else 0.0
                print(f"[step {self.global_step}] stage={self.curriculum.stage.name} lr={lr:.2e} "
                      f"logs={step_logs} val_loss={val_loss:.4f} lstd={lstd:.4f}", flush=True)
                self.metrics.append({"step": self.global_step, "stage": self.curriculum.stage.name,
                                     "val_loss": val_loss, "latent_std": round(lstd, 4), **step_logs})

            if self.train_cfg.checkpoint_every and self.global_step % self.train_cfg.checkpoint_every == 0:
                self.save("checkpoint.pt")

            if self.curriculum.advance_step(val_loss):
                print(f">>> curriculum advanced to stage {self.curriculum.stage.name}", flush=True)
            if self.curriculum.stage == Stage.F:
                break

    # ------------------------------------------------------------------
    def save(self, name: str):
        import random
        import numpy as np
        os.makedirs(self.ckpt_dir, exist_ok=True)
        path = os.path.join(self.ckpt_dir, name)
        rng = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save({
            "rng": rng,
            "model_state": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "curriculum": self.curriculum.state_dict(),
            "global_step": self.global_step,
            "metrics": self.metrics,
            "model_cfg": asdict(self.model_cfg),
            "vocab_size": self.model_cfg.vocab_size,
            "tokenizer_name": "gpt2", "chunker": "regex_gpt2",
        }, path)
        with open(os.path.join(self.ckpt_dir, "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"[trainer] checkpoint -> {path} (step {self.global_step})", flush=True)

    def load(self, path: str):
        import random
        import numpy as np
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.ema.load_state_dict(ckpt["ema"])
        if self.scaler is not None and ckpt.get("scaler") is not None:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.curriculum.load_state_dict(ckpt["curriculum"])
        self.global_step = ckpt["global_step"]
        self.metrics = ckpt.get("metrics", [])
        rng = ckpt.get("rng")
        if rng is not None:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu().to(torch.uint8))
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in rng["cuda"]])
        print(f"[trainer] resumed from {path} at step {self.global_step} "
              f"(stage {self.curriculum.stage.name})", flush=True)
