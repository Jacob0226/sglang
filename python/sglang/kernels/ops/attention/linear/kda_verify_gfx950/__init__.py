"""gfx950 KDA target verify with the ReplaySSM ring (conv1d + safe-gate delta rule in one
launch). A build/load failure disables it and callers keep the Triton kernel."""

import ctypes
import logging
import os
import subprocess
from typing import Optional

import torch

from sglang.kernels.jit.utils.hipcc import HipKernel, build_gfx950_module

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kda_verify.hip")
_BVC = (32, 64)
_kernels: dict = {}
_disabled = False
logger = logging.getLogger(__name__)

_PTRS = (
    "x", "w", "cs", "cs_idx", "iw", "iw_idx", "a", "b", "A_log", "dt_bias", "state",
    "h0_idx", "o", "rawv", "rawk", "ring_g", "ring_beta", "ring_idx",
)
_I64 = (
    "x_stride", "cs_line", "cs_tok", "iw_line", "iw_step", "iw_win",
    "a_stride", "b_stride", "rawv_slot", "rawk_slot", "g_slot", "beta_slot",
)


class Args(ctypes.Structure):
    _fields_ = (
        [(n, ctypes.c_void_p) for n in _PTRS]
        + [(n, ctypes.c_int64) for n in _I64]
        + [(n, ctypes.c_int32) for n in ("L", "H", "HV")]
        + [("lower_bound", ctypes.c_float), ("scale", ctypes.c_float)]
    )


def _load() -> bool:
    global _disabled
    if _kernels:
        return True
    if _disabled:
        return False
    try:
        mod = build_gfx950_module(_SRC, "sglang_kda_verify_")
        for t in (6, 8):
            for bvc in _BVC:
                _kernels[(t, bvc)] = HipKernel(mod, f"kda_verify_t{t}_v{bvc}")
    except (OSError, subprocess.CalledProcessError, RuntimeError) as e:
        logger.warning("kda_verify_gfx950 disabled: %s", getattr(e, "stderr", None) or e)
        _kernels.clear()
        _disabled = True
        return False
    return True


def kda_verify_gfx950_available() -> bool:
    return _load()


def default_block_v(batch_size: int) -> int:
    # MI355X, GLM-5.3-Flash TP4 (HV 16, T 6): BV32 wins up to B4, BV64 from B6 to B64.
    return 32 if batch_size <= 4 else 64


def kda_verify_gfx950_supported(
    *,
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: Optional[torch.Tensor],
    conv_state: torch.Tensor,
    intermediate_conv_window: Optional[torch.Tensor],
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_states: torch.Tensor,
    T: int,
    num_q_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    lower_bound: Optional[float],
    use_qk_l2norm_in_kernel: bool,
) -> bool:
    return (
        not _disabled
        and T in (6, 8)
        and head_k_dim == head_v_dim == 128
        and num_v_heads % num_q_heads == 0
        and lower_bound is not None
        and conv_bias is None
        and use_qk_l2norm_in_kernel
        and mixed_qkv.dtype == a.dtype == b.dtype == torch.bfloat16
        and conv_weight.dtype == ssm_states.dtype == torch.float32
        and A_log.dtype == dt_bias.dtype == torch.float32
        and conv_state.dtype == torch.bfloat16
        and mixed_qkv.stride(-1) == 1
        and conv_state.stride(1) == 1
        and conv_weight.stride() == (4, 1)
        and ssm_states.is_contiguous()
        and A_log.is_contiguous()
        and dt_bias.is_contiguous()
        and (
            intermediate_conv_window is None
            or (
                intermediate_conv_window.dtype == torch.bfloat16
                and intermediate_conv_window.stride(2) == 1
            )
        )
    )


def kda_verify_gfx950(
    *,
    mixed_qkv,
    conv_weight,
    conv_state,
    conv_state_indices,
    intermediate_conv_window,
    intermediate_state_indices,
    a2,
    b2,
    A_log,
    dt_bias,
    ssm_states,
    cache_indices,
    scale,
    T,
    num_q_heads,
    num_v_heads,
    lower_bound,
    replayssm_rawv,
    replayssm_rawk,
    replayssm_g,
    replayssm_beta,
    ring_indices,
    block_v: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Returns o [seq_len, HV, V], or None if the kernel is unavailable."""
    if not _load():
        return None
    seq_len = mixed_qkv.shape[0]
    B = seq_len // T
    HV = num_v_heads
    bvc = block_v or default_block_v(B)
    o = mixed_qkv.new_empty(seq_len, HV, 128)
    iw = intermediate_conv_window
    iw_s = iw.stride() if iw is not None else (0, 0, 0, 0)
    ptr = lambda t: None if t is None else t.data_ptr()
    args = Args(
        ptr(mixed_qkv), ptr(conv_weight), ptr(conv_state), ptr(conv_state_indices),
        ptr(iw), ptr(intermediate_state_indices if iw is not None else conv_state_indices),
        ptr(a2), ptr(b2), ptr(A_log), ptr(dt_bias), ptr(ssm_states), ptr(cache_indices),
        ptr(o), ptr(replayssm_rawv), ptr(replayssm_rawk), ptr(replayssm_g),
        ptr(replayssm_beta), ptr(ring_indices if ring_indices is not None else cache_indices),
        mixed_qkv.stride(0), conv_state.stride(0), conv_state.stride(2),
        iw_s[0], iw_s[1], iw_s[3], a2.stride(0), b2.stride(0),
        replayssm_rawv.stride(0), replayssm_rawk.stride(0), replayssm_g.stride(0),
        replayssm_beta.stride(0),
        replayssm_rawv.shape[-2], num_q_heads, HV,
        float(lower_bound), float(scale),
    )
    stream = torch.cuda.current_stream().cuda_stream
    _kernels[(T, bvc)].launch((128 // bvc, B * HV), args, stream, block=256)
    return o
