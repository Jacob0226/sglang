"""
FlyDSL sparse-MLA decode -- Approach C (FlyDSL hgemm x 2 + torch softmax).

Why this exists
---------------
The fused single-pass FP8 fmha kernel target (``flydsl_sparse_mla.py``) is a
~600-line MLIR-level kernel that bumps a list of FlyDSL/MLIR API gaps before
end-to-end correctness can be checked (see that file's E.bug.1..6).  Each
fix is small but the iteration loop is slow without HW-level instrumentation.

This file gets us an **end-to-end FlyDSL implementation today** by composing
verified building blocks instead of fusing them:

    1. Gather K_nope / K_rope rows from the BF16 paged buffer    [torch index_select]
    2. QK   = Q_concat @ K_concat^T                             [flydsl_hgemm]
    3. P    = softmax(QK * sm_scale)                            [torch.softmax]
    4. O    = P @ K_nope_gathered                               [flydsl_hgemm]

Trade-off
---------
- Pros: validated FlyDSL hgemm (already verified bit-exact in our smoke test);
  no in-kernel softmax / FP8 cvt / SV layout headaches; runs end-to-end TODAY.
- Cons: stores intermediate ``S = (HG, TopK) bf16`` to global memory between
  the two hgemms (HG=8 * TopK=2048 * 2B = 32 KB extra HBM traffic per call --
  small).  PyTorch ``softmax`` adds one kernel launch.  This is ABOVE what
  a fully fused fmha would pay; expect TileLang main_kernel + combine to be
  competitive or better here.

Once the fused kernel lands (``flydsl_sparse_mla.py``) it should beat this
v2 by skipping the intermediate S writeback + ``softmax`` launch.

API
---
``flydsl_sparse_mla_decode_v2(q_all, kv_paged_uint8, indices, sm_scale, d_v=512)``
mirrors the signature of ``tilelang_sparse_fwd`` so it can be plugged into
``nsa_backend.py``'s tilelang dispatch directly.
"""

from __future__ import annotations

import torch

from sglang.srt.layers.attention.nsa.dequant_k_cache import dequantize_k_cache
from sglang.srt.layers.attention.nsa.tilelang_kernel_fp8 import (
    _split_paged_kv_fp8,
)


_FLYDSL_AVAILABLE = None
_FLYDSL_IMPORT_ERROR = None


def _try_import_flydsl():
    global _FLYDSL_AVAILABLE, _FLYDSL_IMPORT_ERROR
    if _FLYDSL_AVAILABLE is True:
        return True
    if _FLYDSL_AVAILABLE is False:
        return False
    try:
        from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm  # noqa: F401
        _FLYDSL_AVAILABLE = True
        return True
    except Exception as e:
        _FLYDSL_AVAILABLE = False
        _FLYDSL_IMPORT_ERROR = repr(e)
        return False


def _dequant_paged_kv_to_bf16(kv_paged_uint8: torch.Tensor):
    """Dequant the 656-byte/token FP8 paged buffer into a flat bf16 (T, 576).

    Uses the existing triton ``dequantize_k_cache`` kernel.  We can avoid this
    entirely once a full FP8 fmha lands; for v2 we go through bf16 because
    flydsl_hgemm only supports bf16/fp16.
    """
    if kv_paged_uint8.dim() == 3:
        kv_paged_uint8 = kv_paged_uint8.unsqueeze(2)
    nb, bs, hk, bytes_per_token = kv_paged_uint8.shape
    assert hk == 1
    assert bytes_per_token == 656, f"expected 656 B/token, got {bytes_per_token}"
    kv_bf16 = dequantize_k_cache(kv_paged_uint8.view(nb * bs, 1, bytes_per_token))
    # ``dequantize_k_cache`` returns (-1, 1, 576) bf16
    return kv_bf16.reshape(nb * bs, 576)


def flydsl_sparse_mla_decode_v2(
    q: torch.Tensor,
    kv_paged_uint8: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> torch.Tensor:
    """Drop-in alternative to ``tilelang_sparse_fwd`` using FlyDSL hgemms.

    Parameters
    ----------
    q : (seq_len, heads, dim+tail_dim) bf16
    kv_paged_uint8 : (num_blocks, block_size, 1, 656) uint8
        FP8 paged KV buffer in NSA layout.  We dequant it on-the-fly to bf16.
    indices : (seq_len, kv_group, topk) int32 -- gather indices into the
        flattened KV buffer.  ``< 0`` masks the slot.
    sm_scale : float
    d_v : int (== 512)

    Returns
    -------
    (seq_len, heads, d_v) bf16
    """
    if not _try_import_flydsl():
        raise RuntimeError(
            f"FlyDSL is not available: {_FLYDSL_IMPORT_ERROR!r}"
        )

    from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm

    assert q.dim() == 3 and indices.dim() == 3
    seq_len, heads, dim_total = q.shape
    d_rope = dim_total - d_v
    assert d_v == 512 and d_rope == 64
    assert indices.shape[1] == 1, "kv_group=1 expected"
    topk = indices.shape[-1]

    # --- Step 1: dequant + gather KV ---------------------------------
    kv_full_bf16 = _dequant_paged_kv_to_bf16(kv_paged_uint8)   # (T_total, 576)

    # Flatten indices to (batch, topk) and clamp negatives to 0 so the gather
    # is safe; we'll mask their contribution to softmax = 0 below.
    indices_flat = indices.squeeze(1).long()                   # (seq_len, topk)
    valid = indices_flat >= 0
    safe_idx = indices_flat.clamp(min=0)
    # gather K for ALL queries: shape (seq_len, topk, 576)
    k_concat = kv_full_bf16[safe_idx]                           # (seq_len, topk, 576)

    # --- Step 2: QK GEMM -- one per-query gemm via flydsl_hgemm ------
    # flydsl_hgemm: out = a @ b.T, a=(M,K), b=(N,K).  For each query in the
    # batch we have a=(heads, 576), b=(topk, 576) -> (heads, topk).
    #
    # We use a flatten-batch trick: concat queries across batch into a single
    # (seq_len*heads, 576) and the K stays per-query.  Since hgemm doesn't
    # do batched gemm, we just loop over seq_len.  For decode batches of 4
    # this is fine (4 hgemm calls, each tiny).
    out = torch.empty(
        seq_len, heads, d_v, dtype=q.dtype, device=q.device
    )
    s_buf = torch.empty(heads, topk, dtype=q.dtype, device=q.device)

    for b in range(seq_len):
        q_b = q[b].contiguous()                                # (heads, 576)
        k_b = k_concat[b].contiguous()                         # (topk, 576)

        # QK
        flydsl_hgemm(q_b, k_b, s_buf, auto_shuffle_b=True)

        # Mask invalid topk slots
        if not valid[b].all():
            mask_b = valid[b].unsqueeze(0).expand(heads, topk)
            s_buf = s_buf.masked_fill(~mask_b, float("-inf"))

        # Softmax (fp32 internally, output bf16)
        p = torch.softmax(s_buf.float() * sm_scale, dim=-1).to(q.dtype)

        # SV: O = P @ K_nope where K_nope = k_b[:, :d_v]
        # In hgemm form: a=P (heads, topk), b=K_nope^T (d_v, topk)
        k_nope_T = k_b[:, :d_v].T.contiguous()                 # (d_v, topk)
        flydsl_hgemm(p, k_nope_T, out[b], auto_shuffle_b=True)

    return out


__all__ = ["flydsl_sparse_mla_decode_v2"]
