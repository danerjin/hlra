"""
rocm_smoke.py
=============
Validate the training path on an AMD ROCm GPU (e.g. Strix Halo / Radeon 8060S,
gfx1151) BEFORE committing to a long run. Runs entirely on synthetic random
tensors -- no data, tokenizer, or download needed -- so it's the very first
thing to run on a fresh box.

Checks, in order (each must stay finite):
  1. torch sees the GPU. ROCm exposes the AMD GPU *as* CUDA, so
     torch.cuda.is_available() should be True; prints the HIP/ROCm build + name.
  2. a bf16 matmul on-device is finite (basic compute + the autocast dtype work).
  3. ONE real forward_grounded + backward of the model under bf16 autocast is
     finite. This is the load-bearing check: it exercises exactly the ops that
     can NaN under mixed precision -- the HRM loop's hard_normalize division,
     the decay gate's exp/softplus, the masked-softmax guards, and the CE loss.
  4. ONE forward_self_supervised (the SSL cosine/variance path) under autocast.

Exit 0 => the training path runs and stays finite on this GPU. Any NaN/Inf ->
prints where and exits 1.

Run (on the ROCm box, from files/):
    python rocm_smoke.py --preset small
    # gfx1151 sometimes needs an ISA override for the ROCm build in use:
    HSA_OVERRIDE_GFX_VERSION=11.5.1 python rocm_smoke.py --preset small

Notes:
  * Use bf16 (default). RDNA 3.5 has bf16; it keeps fp32 range, so no GradScaler
    is needed. fp16 would require the GradScaler path (see trainer.Trainer).
  * chunker is None here: forward_grounded/self_supervised never call it (only
    chunk_batch does, which this test doesn't use).
"""
from __future__ import annotations

import argparse
import sys

import torch

from config import model_config
from model import LatentThoughtModel, StageFlags, SELF
from gestalt_memory import GestaltMemoryBank
from ema_target import EMATargetEncoder


def finite(t) -> bool:
    return bool(torch.isfinite(t).all())


def pick_device() -> str:
    if torch.cuda.is_available():          # ROCm reports as "cuda"
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synth_batch(cfg, batch, device):
    """Random chunk tensors with the last chunk of each row left as PAD, so the
    all-pad-row guards (chunk encoder / input lane) get exercised too."""
    N, L, W = cfg.max_chunks_per_doc, cfg.max_chunk_len, cfg.recent_token_window
    ct = torch.randint(1, cfg.vocab_size, (batch, N, L), device=device)
    cm = torch.ones(batch, N, dtype=torch.bool, device=device)
    ct[:, -1, :] = 0            # last chunk is padding...
    cm[:, -1] = False           # ...and marked invalid
    ri = torch.randint(1, cfg.vocab_size, (batch, W), device=device)
    rm = torch.ones(batch, W, dtype=torch.bool, device=device)
    return ct, cm, ri, rm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=50258, help="gpt2(+1); only the size matters here")
    ap.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16"])
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    device = pick_device()
    ok = True

    # 1. device visibility ------------------------------------------------
    print("=" * 64)
    print(f"torch            : {torch.__version__}")
    print(f"torch.version.hip: {getattr(torch.version, 'hip', None)}  (None on non-ROCm builds)")
    print(f"cuda available   : {torch.cuda.is_available()}  (True == ROCm sees the AMD GPU)")
    if device != "cuda":
        print(f"!! device resolved to {device!r}, not 'cuda' -- ROCm/torch is not seeing the GPU.")
        print("   Check: ROCm install, a torch-ROCm wheel (not CPU/MPS), and HSA_OVERRIDE_GFX_VERSION.")
        sys.exit(1)
    print(f"device name      : {torch.cuda.get_device_name(0)}")
    try:
        free, total = torch.cuda.mem_get_info()
        print(f"device memory    : {total/1e9:.1f} GB total, {free/1e9:.1f} GB free")
    except Exception as e:
        print(f"device memory    : (mem_get_info unavailable: {e})")
    print("=" * 64)

    # 2. bf16 matmul ------------------------------------------------------
    a = torch.randn(1024, 1024, device=device, dtype=dtype)
    b = torch.randn(1024, 1024, device=device, dtype=dtype)
    c = a @ b
    torch.cuda.synchronize()
    print(f"[2] {args.amp_dtype} matmul finite: {finite(c)}")
    ok &= finite(c)

    # 3. forward_grounded + backward under autocast -----------------------
    cfg = model_config(args.preset, vocab_size=args.vocab)
    model = LatentThoughtModel(cfg, chunker=None).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[3] built {args.preset} model: {n_params/1e6:.1f}M params on {device}")

    ct, cm, ri, rm = synth_batch(cfg, args.batch_size, device)
    flags = StageFlags(use_hrm_loop=True, detach_memory=False, inner_loop_grad_window=5,
                       memory_grad_window=5, use_act=False, use_input_lanes=False)
    # Autoencoder anchor (encoder -> Talker, no loop): cheap/parallel, exercises
    # the Talker + encoder + CE under autocast.
    with torch.autocast(device_type="cuda", dtype=dtype):
        loss = model.forward_grounded(ct, cm)
    loss.backward()
    torch.cuda.synchronize()
    grad_finite = all(finite(p.grad) for p in model.parameters() if p.grad is not None)
    print(f"    autoencoder loss finite: {finite(loss)} (nll={float(loss):.3f})   grads finite: {grad_finite}")
    ok &= finite(loss) and grad_finite

    # 4. forward_self_supervised (on-loop SSL, Stage C: loop + un-detached memory) --
    # the load-bearing check: it runs the sequential HRM loop reading/writing the
    # gestalt memory, exercising hard_normalize / the decay gate / masked softmax /
    # cross-attention under autocast -- exactly the ops that can NaN in mixed precision.
    model.zero_grad(set_to_none=True)
    ema = EMATargetEncoder(model.chunk_encoder, momentum=cfg.ema_momentum).to(device)
    memory = GestaltMemoryBank(cfg.memory_capacity, cfg.d_model)
    with torch.autocast(device_type="cuda", dtype=dtype):
        ssl, ponder = model.forward_self_supervised(ct, cm, ri, rm, memory, SELF, flags, ema,
                                                    cos_weight=1.0, var_weight=2.0,
                                                    ponder_weight=cfg.act_ponder_cost)
        loss4 = ssl + ponder
    loss4.backward()
    torch.cuda.synchronize()
    print(f"[4] on-loop SSL (loop+memory) finite: {finite(loss4)} (ssl={float(ssl):.4f})")
    ok &= finite(loss4)

    # 5. ACT path (Stage D) -- variable inner-loop depth + halting head under autocast.
    model.zero_grad(set_to_none=True)
    flags_e = StageFlags(use_hrm_loop=True, detach_memory=False, inner_loop_grad_window=5,
                         memory_grad_window=5, use_act=True, use_input_lanes=False)
    memory = GestaltMemoryBank(cfg.memory_capacity, cfg.d_model)
    with torch.autocast(device_type="cuda", dtype=dtype):
        ssl_e, ponder_e = model.forward_self_supervised(ct, cm, ri, rm, memory, SELF, flags_e, ema,
                                                        cos_weight=1.0, var_weight=2.0,
                                                        ponder_weight=cfg.act_ponder_cost)
        loss_e = ssl_e + ponder_e
    loss_e.backward()
    torch.cuda.synchronize()
    print(f"[5] ACT (loop) loss finite: {finite(loss_e)} (ponder={float(ponder_e):.4f})")
    ok &= finite(loss_e)

    # 6. the monitoring path: eval-mode forward (Trainer.evaluate + the lstd
    # collapse metric run this every log_every steps, WITHOUT autocast). An
    # eval-mode nn.TransformerEncoder takes the fused BetterTransformer kernel
    # -- a different ROCm code path than the training-mode forwards above, and
    # the one that historically misbehaved when mixed with other dtypes (§18.1).
    model.eval()
    with torch.no_grad():
        val = model.forward_grounded(ct, cm)
    lstd = model.latent_collapse_metric(ct, cm)
    torch.cuda.synchronize()
    lstd_ok = lstd == lstd and abs(lstd) != float("inf")   # finite python float
    print(f"[6] eval-mode (fused) val path finite: {finite(val)} (val={float(val):.3f}, lstd={lstd:.4f})")
    ok &= finite(val) and lstd_ok
    model.train()

    print("=" * 64)
    if ok:
        print("PASS: training path runs and stays finite on this GPU under "
              f"{args.amp_dtype} autocast. Next: bench_throughput.py")
        sys.exit(0)
    print("FAIL: a NaN/Inf appeared above. If it's only under fp16, try --amp-dtype bf16 "
          "(recommended). If bf16 also NaNs, the ROCm kernel for one op is suspect -- "
          "note which check failed.")
    sys.exit(1)


if __name__ == "__main__":
    main()
