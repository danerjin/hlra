"""
trainer.py
==========
A scale-ready training loop for Stages A-E, replacing the flat function in
train.py. Adds the infrastructure a real run needs while keeping the loss
semantics (reconstruction anchor + on-loop SSL predictor with an anti-collapse
variance floor, notes §26) verified on the smoke run:

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
                 model_cfg, train_cfg, train_loader, val_loader, ckpt_dir,
                 data_fingerprint: dict = None):
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
        # Identity of the dataset this run was launched against (train_scaled
        # passes the cache's example/token counts). Stored in every checkpoint
        # and compared on resume: the val/train split is a seeded randperm over
        # len(dataset), so a cache that changed size between launch and resume
        # silently reshuffles the split and leaks val docs into train --
        # poisoning val_loss, the run's collapse signal.
        self.data_fingerprint = data_fingerprint

        self.global_step = 0
        self.metrics = []
        self._train_iter = None
        self._nonfinite_streak = 0   # consecutive optimizer steps skipped on non-finite grads
        self._stop_requested = False # set by SIGINT/SIGTERM: checkpoint + exit at next step boundary
        self._bar = None             # tqdm progress bar (interactive only); None otherwise
        self._last_val_loss = None   # cached for the progress-bar postfix between eval steps
        self._last_lstd = None

        # Mixed precision. CUDA (incl. ROCm, which presents as cuda) only: CPU
        # gains nothing, and MPS autocast either raises outright (torch<=2.2)
        # or would re-expose the §18.1 eval-mode-encoder dtype mix. A GradScaler
        # is only needed for CUDA fp16.
        self.use_amp = bool(train_cfg.amp) and self.device.type == "cuda"
        if bool(train_cfg.amp) and not self.use_amp:
            print(f"[trainer] --amp requested but device is {self.device.type}; "
                  f"running full precision.", flush=True)
        self.amp_dtype = torch.bfloat16 if train_cfg.amp_dtype == "bf16" else torch.float16
        self.scaler = (torch.cuda.amp.GradScaler()
                       if self.use_amp and self.device.type == "cuda" and self.amp_dtype == torch.float16
                       else None)

    # Fields that define the training schedule/objective. Stored in every
    # checkpoint and compared on resume: the restored curriculum position and
    # LR are only meaningful against the SAME budgets/flags, so a resume with
    # a drifted command line must not proceed silently.
    _SCHEDULE_FIELDS = ("stage_steps", "per_stage_lr", "lr", "min_lr_ratio",
                        "warmup_steps", "total_steps", "batch_size",
                        "grad_accum_steps", "grounded_loss_min_frequency",
                        "ssl_loss_weight", "ssl_var_weight",
                        "amp", "amp_dtype")

    def _schedule_snapshot(self) -> dict:
        return {k: getattr(self.train_cfg, k, None) for k in self._SCHEDULE_FIELDS}

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

    def _loss_on(self, batch, flags, plan, want_logs=True):
        ct, cm, ri, rm = batch
        total, logs = None, {}
        # One shared online encoder pass per step, reused by both branches.
        chunk_vecs = self.model.encode_chunks(ct)
        if plan.use_grounded_loss:
            # Autoencoder anchor (encoder -> Talker, no loop): cheap, parallel,
            # always on -- the anchor never thins.
            nll = self.model.forward_grounded(ct, cm, chunk_vecs=chunk_vecs)
            total = nll
            if want_logs:
                logs["nll"] = round(nll.item(), 4)
        if plan.use_self_supervised_loss:
            # On-loop SSL (§2.1/§27): the HRM loop predicts the next latent,
            # SEQUENTIALLY, reading its accumulating gestalt memory. Trains the
            # loop + encoder + memory to reason forward; carries the ACT ponder.
            memory = GestaltMemoryBank(self.model_cfg.memory_capacity, self.model_cfg.d_latent)
            ssl, ponder = self.model.forward_self_supervised(
                ct, cm, ri, rm, memory, SELF, flags, self.ema,
                cos_weight=self.train_cfg.ssl_loss_weight, var_weight=self.train_cfg.ssl_var_weight,
                ponder_weight=self.model_cfg.act_ponder_cost, chunk_vecs=chunk_vecs)
            total = (ssl if total is None else total + ssl) + ponder
            if want_logs:
                logs["ssl"] = round(ssl.item(), 4)
                logs["ponder"] = round(ponder.item(), 4)
        # want_logs=False skips the .item() calls: each is a host-device sync,
        # and three syncs per micro-batch on a launch-overhead-bound workload
        # is measurable -- the values are only ever printed on log steps.
        return total, logs, (ct, cm)

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        losses = []
        for i, batch in enumerate(self.val_loader):
            if i >= 4:
                break
            ct, cm, ri, rm = tuple(t.to(self.device) for t in batch)
            # Reconstruction (autoencoder) NLL only -- the decodability signal,
            # comparable across stage boundaries (independent of loop/SSL/ponder,
            # so no contaminated-eval jump; notes §5.6).
            losses.append(self.model.forward_grounded(ct, cm).item())
        self.model.train()
        return sum(losses) / max(len(losses), 1)

    # ------------------------------------------------------------------
    # UX helpers (progress bar + graceful stop). None of these touch the loss
    # computation, gradient routing, or curriculum -- they only affect how the
    # run reports progress and how it shuts down.
    def _emit(self, msg: str):
        # Route log lines through tqdm.write when a bar is live so they don't
        # shred the progress bar; falls back to a plain flushed print otherwise
        # (including the non-TTY nohup run, where the bar is auto-disabled).
        if self._bar is not None:
            self._bar.write(msg)
        else:
            print(msg, flush=True)

    def _make_progress_bar(self, max_steps: int, progress: str):
        # progress: "auto" (bar on a TTY, silent line-logs when redirected),
        # "on" (force the bar), or "off" (never). tqdm is optional: if it isn't
        # installed (e.g. a minimal training venv) we degrade to plain prints.
        if progress == "off":
            return None
        try:
            from tqdm.auto import tqdm
        except Exception:
            return None
        disable = None if progress == "auto" else False  # None => tqdm hides itself off-TTY
        return tqdm(total=max_steps, initial=self.global_step, disable=disable,
                    dynamic_ncols=True, desc="train", unit="step")

    def _install_signal_handlers(self):
        # First SIGINT/SIGTERM: request a graceful stop -- the loop finishes the
        # current step, writes checkpoint.pt, and exits, so `kill <pid>` / Ctrl-C
        # loses at most one step and `--resume` lands exactly where it stopped.
        # A second signal falls through to the default handler for a hard quit.
        import signal

        def _handler(signum, _frame):
            if self._stop_requested:
                raise KeyboardInterrupt  # second signal -> hard exit
            self._stop_requested = True
            self._emit(f"[trainer] stop requested (signal {signum}); will checkpoint and "
                       f"exit after the current step. Send again to force-quit.")

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                pass  # not on the main thread (some launchers) -- skip, don't crash

    def train(self, max_steps: int, progress: str = "auto"):
        accum = max(1, self.train_cfg.grad_accum_steps)
        self._install_signal_handlers()
        self._bar = self._make_progress_bar(max_steps, progress)
        try:
            self._train_loop(max_steps, accum)
        finally:
            if self._bar is not None:
                self._bar.close()
                self._bar = None

    def _train_loop(self, max_steps: int, accum: int):
        import time
        _loop_t0 = time.time()
        _first_step_logged = False
        # With log_every=50 the first visible line is step 50; the FIRST optimizer
        # step also JIT-compiles GPU kernels (minutes on ROCm/gfx1151), so without
        # these markers a healthy startup looks hung. Emit loop-start + first-step.
        self._emit(f"[trainer] training loop starting @ step {self.global_step}, "
                   f"stage={self.curriculum.stage.name}, {max_steps} steps total. "
                   f"First step JIT-compiles GPU kernels -- minutes on ROCm/gfx1151 is normal.")
        while self.curriculum.stage.value <= Stage.E.value and self.global_step < max_steps:
            flags = self.curriculum.stage_flags()
            plan = self.curriculum.loss_plan()
            lr = self._lr(self.global_step)
            for g in self.optimizer.param_groups:
                g["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            step_logs, last_cc = {}, None
            want_logs = (self.train_cfg.log_every > 0
                         and (self.global_step + 1) % self.train_cfg.log_every == 0)
            for _ in range(accum):
                batch = self._next_batch()
                with self._autocast():
                    loss, logs, cc = self._loss_on(batch, flags, plan, want_logs)
                last_cc = cc
                # A batch with zero valid chunks returns a grad-free zero
                # (model.forward_grounded's empty guard); backward() on it
                # would raise. min_chunks filtering makes this near-impossible,
                # but a multi-day run shouldn't die on one degenerate batch.
                if loss is None or not loss.requires_grad:
                    continue
                loss = loss / accum
                (self.scaler.scale(loss).backward() if self.scaler is not None else loss.backward())
                step_logs.update(logs)

            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip)
            if self.scaler is not None:
                # The fp16 scaler already skips the step on inf/nan grads.
                self.scaler.step(self.optimizer); self.scaler.update()
            elif bool(torch.isfinite(total_norm)):
                self.optimizer.step()
                self._nonfinite_streak = 0
            else:
                # Non-finite guard (bf16/fp32 have no GradScaler to filter this):
                # clip_grad_norm_ computes ONE global norm, so a single NaN/Inf
                # grad element makes clip_coef NaN and scales EVERY parameter's
                # grad to NaN -- one unguarded optimizer.step() then destroys
                # all weights, and the run would keep training and overwriting
                # checkpoints with the corpse. Skip the step instead (grads are
                # zeroed at the top of the next iteration) and hard-fail if it
                # persists, so an unattended run can't burn days spinning.
                self._nonfinite_streak += 1
                print(f"[trainer] WARNING: non-finite grad norm ({float(total_norm)}) at step "
                      f"{self.global_step + 1}; skipping optimizer step "
                      f"({self._nonfinite_streak} consecutive).", flush=True)
                if self._nonfinite_streak >= 25:
                    raise RuntimeError(
                        f"25 consecutive non-finite gradient steps (last norm {float(total_norm)}). "
                        f"The run is numerically dead -- weights are still finite (steps were "
                        f"skipped), so inspect the last checkpoint and the data/LR before resuming.")
            self.ema.update(self.model.chunk_encoder)
            self.global_step += 1
            if not _first_step_logged:
                self._emit(f"[trainer] first optimizer step done in {time.time() - _loop_t0:.0f}s "
                           f"-- training is LIVE (next log at step {self.train_cfg.log_every}).")
                _first_step_logged = True
            if self._bar is not None:
                self._bar.update(1)
                self._bar.set_postfix(stage=self.curriculum.stage.name, lr=f"{lr:.1e}",
                                      val=("-" if self._last_val_loss is None
                                           else f"{self._last_val_loss:.3f}"),
                                      lstd=("-" if self._last_lstd is None
                                            else f"{self._last_lstd:.3f}"),
                                      refresh=False)

            val_loss = None
            if self.train_cfg.log_every and self.global_step % self.train_cfg.log_every == 0:
                val_loss = self.evaluate()
                lstd = self.model.latent_collapse_metric(*last_cc) if last_cc else 0.0
                self._last_val_loss, self._last_lstd = val_loss, lstd
                self._emit(f"[step {self.global_step}] stage={self.curriculum.stage.name} lr={lr:.2e} "
                           f"logs={step_logs} val_loss={val_loss:.4f} lstd={lstd:.4f}")
                self.metrics.append({"step": self.global_step, "stage": self.curriculum.stage.name,
                                     "val_loss": val_loss, "latent_std": round(lstd, 4), **step_logs})

            # Advance the curriculum BEFORE checkpointing: the checkpoint must
            # capture the post-step curriculum state (stage_idx/step_in_stage),
            # or a resume replays one extra step-in-stage and the per-stage LR
            # schedule drifts off the uninterrupted run by one step.
            if self.curriculum.advance_step(val_loss):
                self._emit(f">>> curriculum advanced to stage {self.curriculum.stage.name}")

            if self.train_cfg.checkpoint_every and self.global_step % self.train_cfg.checkpoint_every == 0:
                self.save("checkpoint.pt")
            # Numbered snapshots (rollback depth): the rolling checkpoint
            # alone can't rewind past a slow pathology (drift/late collapse)
            # noticed hours after onset. Checked independently of
            # checkpoint_every -- nesting it above silently degraded any
            # --archive-every that wasn't a multiple of --checkpoint-every
            # to their LCM (e.g. 800/500 -> archives only every 4000).
            arch = getattr(self.train_cfg, "checkpoint_archive_every", 0)
            if arch and self.global_step % arch == 0:
                self.save(f"checkpoint_{self.global_step:07d}.pt")

            # Graceful stop: a requested SIGINT/SIGTERM checkpoints here, at a
            # clean step boundary (after the curriculum has advanced, exactly
            # like a periodic checkpoint), then exits. `--resume checkpoint.pt`
            # continues from this step. Re-save only if the periodic write above
            # didn't already land on this step.
            if self._stop_requested:
                if not (self.train_cfg.checkpoint_every
                        and self.global_step % self.train_cfg.checkpoint_every == 0):
                    self.save("checkpoint.pt")
                self._emit(f"[trainer] stopped at step {self.global_step}; checkpoint saved. "
                           f"Re-run the same command to resume.")
                return

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
        # Atomic write: a crash/power-loss mid-save must never destroy the only
        # checkpoint of a multi-day run. Write to a temp file, fsync so the
        # bytes are on disk (rename-over alone can leave a truncated file after
        # power loss on some filesystems), then rename.
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            torch.save({
                "rng": rng,
                "model_state": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "ema": self.ema.state_dict(),
                "scaler": self.scaler.state_dict() if self.scaler is not None else None,
                "curriculum": self.curriculum.state_dict(),
                "train_schedule": self._schedule_snapshot(),
                "data_fingerprint": self.data_fingerprint,
                "global_step": self.global_step,
                "metrics": self.metrics,
                "model_cfg": asdict(self.model_cfg),
                "vocab_size": self.model_cfg.vocab_size,
                "stage_reached": self.curriculum.stage.name,
                "tokenizer_name": "gpt2", "chunker": "regex_gpt2",
            }, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        mtmp = os.path.join(self.ckpt_dir, "metrics.json.tmp")
        with open(mtmp, "w") as f:
            json.dump(self.metrics, f, indent=2)
        os.replace(mtmp, os.path.join(self.ckpt_dir, "metrics.json"))
        self._emit(f"[trainer] checkpoint -> {path} (step {self.global_step})")

    def load(self, path: str):
        import random
        import numpy as np
        # weights_only=False: the checkpoint carries Python/NumPy/torch RNG state
        # (not plain tensors), which torch>=2.6's weights_only=True default refuses
        # to unpickle -- an explicit False keeps resume working across torch versions.
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.ema.load_state_dict(ckpt["ema"])
        if self.scaler is not None and ckpt.get("scaler") is not None:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.curriculum.load_state_dict(ckpt["curriculum"])
        # Guard against silent schedule drift on resume: the restored
        # stage_idx/step_in_stage and LR curve are computed against the CLI's
        # budgets/flags, so any difference from the checkpoint's schedule means
        # the resumed run is NOT a continuation of the original one.
        saved = ckpt.get("train_schedule")
        if saved:
            diffs = {k: (v, getattr(self.train_cfg, k, None)) for k, v in saved.items()
                     if getattr(self.train_cfg, k, None) != (tuple(v) if isinstance(v, list) else v)}
            if diffs:
                print("[trainer] " + "!" * 60, flush=True)
                print("[trainer] WARNING: resume schedule differs from checkpoint:", flush=True)
                for k, (old, new) in diffs.items():
                    print(f"[trainer]   {k}: checkpoint={old!r}  now={new!r}", flush=True)
                print("[trainer] LR curve / stage boundaries will NOT match the "
                      "original run. Continue only if this is intentional.", flush=True)
                print("[trainer] " + "!" * 60, flush=True)
        # Hard-stop on a changed dataset: the val/train split is a seeded
        # randperm over len(dataset), so if the cache grew/shrank since launch,
        # most of the old val docs land in the new train set -- val_loss (the
        # collapse signal) becomes silently optimistic. Never re-prep a cache
        # mid-run; resume against the original cache, or start a fresh run.
        saved_fp = ckpt.get("data_fingerprint")
        if saved_fp and self.data_fingerprint and saved_fp != self.data_fingerprint:
            msg = (f"resume dataset differs from checkpoint: "
                   f"checkpoint={saved_fp} now={self.data_fingerprint}. "
                   f"The seeded val/train split depends on dataset size, so val docs "
                   f"leak into train and val_loss becomes meaningless. Resume with the "
                   f"ORIGINAL cache (set LATENT_ALLOW_DATA_CHANGE=1 to override).")
            if os.environ.get("LATENT_ALLOW_DATA_CHANGE") != "1":
                raise RuntimeError(msg)
            print(f"[trainer] WARNING (override): {msg}", flush=True)
        # Honest-resume note: the DataLoader iterator position is not part of
        # the checkpoint, so resuming rebuilds the iterator and re-shuffles --
        # the continuation is statistically equivalent but not sample-exact
        # (the interrupted epoch's unseen tail is partly skipped).
        print("[trainer] note: resume re-shuffles the data order (iterator position "
              "is not checkpointed); continuation is statistically equivalent, not "
              "sample-exact.", flush=True)
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
