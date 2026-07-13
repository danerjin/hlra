"""
rocm_compat.py
==============
Opt-in workarounds for broken kernels on bleeding-edge ROCm / gfx1151 wheels.
OFF by default -- zero effect on any working GPU, the Mac dev box, or the
validated A-E path -- so the entry points can call it unconditionally.

Currently one workaround:

  LATENT_MANUAL_LAYERNORM=1
      Replace torch.nn.functional.layer_norm with a manual implementation built
      from primitive ops (mean / var / rsqrt / mul / add, computed in fp32).

      WHY: the AMD gfx1151 nightly wheel torch 2.12.0a0+rocm7.13 (April 2026
      alpha) has a racy `native_layer_norm_backward` kernel that writes
      NONDETERMINISTIC NaN/Inf into the LayerNorm weight/bias gradients (a
      different LayerNorm param each run at a fixed seed; both bf16 and fp32;
      kernel serialization does not fix it). It corrupts the first optimizer
      step. The model's own `hard_normalize` (a manual norm) is unaffected on the
      same GPU, which is the clue: only the fused LayerNorm *kernel* is broken, so
      a manual layer_norm sidesteps it with NO module or state_dict change (every
      nn.LayerNorm and nn.TransformerEncoderLayer resolves F.layer_norm at call
      time). Drop the env var once AMD ships a wheel with a fixed kernel.

Call `maybe_apply_rocm_workarounds()` once at the top of an entry point
(rocm_smoke.py / train_scaled.py / train_dialogue.py) before building the model.
"""
from __future__ import annotations

import os

_PATCHED = False


def _manual_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    """A drop-in for F.layer_norm from primitive ops (fp32 compute, matching the
    autocast-fp32 semantics of the real op), whose backward uses ordinary
    reduction kernels instead of the broken native_layer_norm_backward."""
    import torch
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    dims = tuple(range(-len(normalized_shape), 0))
    x = input.float()
    mean = x.mean(dims, keepdim=True)
    var = x.var(dims, unbiased=False, keepdim=True)
    out = (x - mean) * torch.rsqrt(var + eps)
    if weight is not None:
        out = out * weight.float()
    if bias is not None:
        out = out + bias.float()
    return out.to(input.dtype)


def maybe_apply_rocm_workarounds() -> None:
    """Install opt-in ROCm kernel workarounds based on env vars. Idempotent; a
    no-op unless a LATENT_* flag is set."""
    global _PATCHED
    if _PATCHED:
        return
    if os.environ.get("LATENT_MANUAL_LAYERNORM") == "1":
        import torch
        torch.nn.functional.layer_norm = _manual_layer_norm
        _PATCHED = True
        print("[rocm_compat] LATENT_MANUAL_LAYERNORM=1: manual LayerNorm active "
              "(routing around the broken native_layer_norm_backward kernel).",
              flush=True)
