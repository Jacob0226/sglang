"""
FP8-KV variant of the TileLang sparse-MLA decode kernel.

Drops in beside ``tilelang_kernel.tilelang_sparse_fwd`` (BF16 KV) and is
selected via ``--nsa-decode-backend tilelang_fp8``.  The existing BF16 kernel
(``main_kernel`` from ``sparse_mla_fwd_decode_partial``) is **left untouched**
so other models that depend on it are not affected.

Why this exists
---------------
The BF16 sparse-MLA decode kernel reads ~1.15 KB of KV per topk-token from
HBM (576 dims x 2 bytes).  On MI355X this is the dominant cost of decode
attention (~20 us per ``main_kernel`` call x 2 calls per layer = 40 us per
layer in GLM-5).  The KV cache is already stored in FP8 by SGLang when
``--kv-cache-dtype fp8_e4m3`` is on, so reading it as FP8 in the kernel
saves ~43% HBM traffic (656 vs 1152 bytes per token).

Layout this kernel consumes
---------------------------
The 656-byte-per-token paged buffer produced by
``nsa.quant_k_cache.quantize_k_cache``::

    [   0..511]  nope_fp8       (e4m3, 512 dims = 4 tiles x 128)
    [512..527]  scale_fp32     (4 tiles, dequant: bf16 = fp8 * scale)
    [528..655]  rope_bf16      (64 dims)

The wrapper ``tilelang_sparse_fwd_fp8`` slices the raw uint8 buffer into the
three views before calling the JIT-compiled kernel.

Implementation choices
----------------------
- KV nope is loaded into shared memory in **per-tile chunks** (BI x 128 fp8
  = 8 KB) and dequantized to bf16 into a single ``KV_shared_bf16`` (BI x D
  bf16 = 64 KB).  This bounds peak LDS (~120 KB) so the kernel still fits
  the gfx950 LDS budget; afterwards both the QK and SV ``T.gemm`` calls run
  bf16 x bf16 just like the existing kernel.
- ``H_per_block`` is set to 32 (vs 64 in the BF16 path) to keep
  ``Q_shared`` small enough for the LDS budget once we add the FP8 staging
  buffers.  Each CTA therefore covers half the heads of one query, so the
  grid grows by 2x along the head axis.
- Combine pass is identical to the BF16 path -- we reuse
  ``sparse_mla_fwd_decode_combine`` from ``tilelang_kernel`` to keep the
  binary surface minimal.
"""

from typing import Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from sglang.srt.layers.attention.nsa.tilelang_kernel import (
    pass_configs,
    sparse_mla_fwd_decode_combine,
)
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.utils import is_gfx95_supported, is_hip

_is_hip = is_hip()
_is_gfx95_supported = is_gfx95_supported()
_is_fp8_fnuz = is_fp8_fnuz()

BF16 = "bfloat16"
FP8 = "float8_e4m3fnuz" if _is_fp8_fnuz else "float8_e4m3"
FP32 = "float32"

# Match quant_k_cache.py
_DV = 512
_DROPE = 64
_TILE = 128
_NUM_TILES = _DV // _TILE  # 4


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_mla_fwd_decode_partial_fp8(
    heads,
    dim,
    tail_dim,
    topk,
    *,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_I=64,
    threads=256,
    h_per_block=32,
):
    """FP8-KV partial-attention kernel.

    Mirrors ``tilelang_kernel.sparse_mla_fwd_decode_partial`` but reads the
    KV nope dim in FP8 with per-tile fp32 scales, dequantizing in shared
    memory before the QK gemm.  Output (Partial_O / Partial_Lse) is bf16/fp32
    so the existing combine kernel can be reused unchanged.
    """
    assert is_causal, "non-causal is not supported"
    assert kv_group == 1
    assert topk % block_I == 0
    assert dim == _DV, f"FP8 KV layout fixes dim={_DV}, got {dim}"
    assert tail_dim == _DROPE, f"FP8 KV layout fixes tail_dim={_DROPE}, got {tail_dim}"

    # log2(e) = 1.44269504
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504
    else:
        sm_scale = sm_scale * 1.44269504

    batch = 1
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    head_kv = heads // kv_group
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    # h_per_block defaults to 32 (vs 64 in bf16 path) to leave LDS room for
    # the per-tile fp8 staging buffer.
    H_per_block = min(padded_H, h_per_block)
    REPLICATE_H = (head_kv + H_per_block - 1) // H_per_block if head_kv > H_per_block else 1
    BI = block_I
    NI = topk // block_I
    D = dim
    D_tail = tail_dim
    NT = _NUM_TILES  # 4 tiles of 128 dims each

    q_shape = [batch, seq_len, heads, dim + tail_dim]
    k_nope_shape = [batch, seq_len_kv, kv_group, dim]
    k_scale_shape = [batch, seq_len_kv, kv_group, NT]
    k_rope_shape = [batch, seq_len_kv, kv_group, tail_dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    partial_o_shape = [batch, seq_len, NI, heads, dim]
    partial_lse_shape = [batch, seq_len, NI, heads]

    indices_dtype = T.int32
    accum_dtype = T.float32

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, BF16),
        K_nope_fp8: T.Tensor(k_nope_shape, FP8),
        K_scale: T.Tensor(k_scale_shape, FP32),
        K_rope: T.Tensor(k_rope_shape, BF16),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Partial_O: T.Tensor(partial_o_shape, BF16),
        Partial_Lse: T.Tensor(partial_lse_shape, accum_dtype),
    ):
        with T.Kernel(seq_len * REPLICATE_H, NI, threads=threads) as (bx, by):
            # ---- Q (bf16) -------------------------------------------------
            Q_shared = T.alloc_shared([H_per_block, D], BF16)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], BF16)

            # ---- KV staging ----------------------------------------------
            # Persistent bf16 KV used by both the (tiled) QK gemm and the
            # final SV gemm.  Built up tile-by-tile from FP8 + scale below.
            KV_shared = T.alloc_shared([BI, D], BF16)
            # Per-tile FP8 scratch + dequant scratch (8 KB + 16 KB).
            KV_fp8_tile = T.alloc_shared([BI, _TILE], FP8)
            # Rope path stays bf16
            K_tail_shared = T.alloc_shared([BI, D_tail], BF16)
            # Mask + per-tile scales (registers / fragments)
            mask = T.alloc_fragment([BI], T.bool)
            scale_local = T.alloc_fragment([BI], accum_dtype)

            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            S_shared = T.alloc_shared([H_per_block, BI], BF16)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(acc_s, 0)

            # ---- Indices --------------------------------------------------
            b_i, g_i = 0, 0
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            topk_block_i = by

            H0 = 0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * H_per_block
            H1 = H0 + H_per_block

            T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

            for bi_i in T.Parallel(BI):
                mask[bi_i] = (
                    Indices[b_i, s_i, g_i, topk_block_i * BI + bi_i] >= 0
                )

            # ---- Rope (bf16, gather once) --------------------------------
            for bi_i, d_i in T.Parallel(BI, D_tail):
                K_tail_shared[bi_i, d_i] = K_rope[
                    b_i,
                    Indices[b_i, s_i, g_i, topk_block_i * BI + bi_i],
                    g_i,
                    d_i,
                ]

            # ---- Tiled FP8 -> bf16 dequant -------------------------------
            # For each of the NT=4 tiles of 128 dims:
            #   1. Gather BI fp8 values into KV_fp8_tile.
            #   2. Gather the BI per-tile scales into a fragment.
            #   3. Dequant -> bf16 directly into KV_shared[:, tile_off:].
            for tile_i in T.Pipelined(NT, num_stages=2):
                tile_off = tile_i * _TILE
                for bi_i, d_i in T.Parallel(BI, _TILE):
                    KV_fp8_tile[bi_i, d_i] = K_nope_fp8[
                        b_i,
                        Indices[b_i, s_i, g_i, topk_block_i * BI + bi_i],
                        g_i,
                        tile_off + d_i,
                    ]
                for bi_i in T.Parallel(BI):
                    scale_local[bi_i] = K_scale[
                        b_i,
                        Indices[b_i, s_i, g_i, topk_block_i * BI + bi_i],
                        g_i,
                        tile_i,
                    ]
                for bi_i, d_i in T.Parallel(BI, _TILE):
                    # cast fp8 -> fp32 -> bf16 with per-tile scale
                    KV_shared[bi_i, tile_off + d_i] = T.Cast(
                        BF16,
                        T.Cast(accum_dtype, KV_fp8_tile[bi_i, d_i])
                        * scale_local[bi_i],
                    )

            # ---- QK gemm (bf16 x bf16) -----------------------------------
            for h_i, bi_i in T.Parallel(H_per_block, BI):
                acc_s[h_i, bi_i] = T.if_then_else(
                    mask[bi_i], 0, -T.infinity(acc_s.dtype)
                )
            T.gemm(
                Q_shared,
                KV_shared,
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullCol,
            )
            T.gemm(
                Q_tail_shared,
                K_tail_shared,
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullCol,
            )

            # ---- Online softmax ------------------------------------------
            T.reduce_max(acc_s, m_i, dim=1, clear=True)
            for h_i in T.Parallel(H_per_block):
                m_i[h_i] = T.max(m_i[h_i], -(2**30))
            for h_i, bi_i in T.Parallel(H_per_block, BI):
                acc_s[h_i, bi_i] = T.exp2(
                    acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                )
            T.reduce_sum(acc_s, sumexp_i, dim=1)
            T.copy(acc_s, S_shared)

            # ---- SV gemm (bf16 x bf16) -----------------------------------
            T.gemm(
                S_shared,
                KV_shared,
                acc_o,
                policy=T.GemmWarpPolicy.FullCol,
            )

            # ---- Normalize + write partials ------------------------------
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] = acc_o[h_i, d_i] / T.if_then_else(
                    sumexp_i[h_i] == 0.0, 1.0, sumexp_i[h_i]
                )
            for h_i in T.Parallel(H_per_block):
                sumexp_i[h_i] = T.if_then_else(
                    sumexp_i[h_i] == 0.0,
                    -(2**30),
                    T.log2(sumexp_i[h_i]) + m_i[h_i] * sm_scale,
                )
            T.copy(acc_o, Partial_O[b_i, s_i, topk_block_i, H0:H1, :])
            T.copy(sumexp_i, Partial_Lse[b_i, s_i, topk_block_i, H0:H1])

    return main


def _split_paged_kv_fp8(
    kv_paged_uint8: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slice the 656-byte/token paged buffer into (nope_fp8, scale_fp32, rope_bf16).

    Layout matches ``nsa.quant_k_cache``::

        nope:   bytes [  0:512]  -> view as fp8_e4m3* (gfx950 -> e4m3fnuz)
        scale:  bytes [512:528]  -> view as float32 (4 per-tile scales)
        rope:   bytes [528:656]  -> view as bfloat16 (64 dims)

    The kernel takes them as separate tensors so the dequant index math
    inside the kernel can stay simple (no byte-offset arithmetic).
    """
    if kv_paged_uint8.dim() == 3:
        kv_paged_uint8 = kv_paged_uint8.unsqueeze(2)
    assert kv_paged_uint8.dim() == 4, kv_paged_uint8.shape
    nb, bs, hk, bytes_per_token = kv_paged_uint8.shape
    assert hk == 1
    assert bytes_per_token == _DV + _NUM_TILES * 4 + _DROPE * 2, (
        f"Expected {_DV + _NUM_TILES * 4 + _DROPE * 2} bytes/token, "
        f"got {bytes_per_token}"
    )

    raw = kv_paged_uint8.view(nb * bs, hk, bytes_per_token)

    nope_dtype = (
        torch.float8_e4m3fnuz if _is_fp8_fnuz else torch.float8_e4m3fn
    )
    nope_part = raw[..., :_DV].view(nope_dtype).reshape(nb, bs, hk, _DV)
    scale_part = (
        raw[..., _DV : _DV + _NUM_TILES * 4]
        .view(torch.float32)
        .reshape(nb, bs, hk, _NUM_TILES)
    )
    rope_part = (
        raw[..., _DV + _NUM_TILES * 4 :]
        .view(torch.bfloat16)
        .reshape(nb, bs, hk, _DROPE)
    )
    return nope_part, scale_part, rope_part


def tilelang_sparse_fwd_fp8(
    q: torch.Tensor,
    kv_paged_uint8: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> torch.Tensor:
    """Drop-in replacement for ``tilelang_sparse_fwd`` that consumes the
    FP8-quantized paged KV buffer (656 B/token) directly.

    Parameters
    ----------
    q : torch.Tensor
        ``(seq_len, heads, dim + tail_dim)`` bf16 -- same as bf16 path.
    kv_paged_uint8 : torch.Tensor
        Paged FP8 KV buffer ``(num_blocks, block_size, 1, 656)`` (uint8 view).
        This is what ``token_to_kv_pool.get_key_buffer`` returns when
        ``--kv-cache-dtype fp8_e4m3`` and ``nsa_kv_cache_store_fp8`` are on.
    indices : torch.Tensor
        ``(seq_len, kv_group, topk)`` int32 gather indices into the paged
        buffer (flattened num_blocks * block_size axis).  Same convention as
        bf16 path.
    sm_scale : float
        Softmax scale.
    d_v : int
        V (== nope) dim.  Must be 512 for the GLM-5 / DSv3 FP8 KV layout.

    Returns
    -------
    torch.Tensor
        ``(seq_len, heads, d_v)`` bf16 attention output.
    """
    assert q.dim() == 3 and indices.dim() == 3
    assert d_v == _DV
    assert _is_hip and _is_gfx95_supported, (
        "tilelang_sparse_fwd_fp8 is gfx950-only at the moment."
    )

    num_heads = q.shape[1]
    dim = q.shape[2]
    tail_dim = dim - d_v
    assert tail_dim == _DROPE
    topk = indices.shape[-1]
    assert topk == 2048

    # Reshape paged KV into (num_blocks*block_size, 1, X) so kernel sees a
    # flat seq_len_kv axis indexable by ``Indices``.
    nope_part, scale_part, rope_part = _split_paged_kv_fp8(kv_paged_uint8)
    flat_kv_len = nope_part.shape[0] * nope_part.shape[1]
    nope_part = nope_part.reshape(1, flat_kv_len, 1, _DV).contiguous()
    scale_part = scale_part.reshape(1, flat_kv_len, 1, _NUM_TILES).contiguous()
    rope_part = rope_part.reshape(1, flat_kv_len, 1, _DROPE).contiguous()

    if num_heads <= 64:
        kernel_partial = sparse_mla_fwd_decode_partial_fp8(
            num_heads, _DV, _DROPE, topk,
            sm_scale=sm_scale, block_I=64, threads=256, h_per_block=32,
        )
        kernel_combine = sparse_mla_fwd_decode_combine(
            num_heads, _DV, topk, head_per_block=4, block_I=64, threads=256,
        )
        partial_o, partial_lse = kernel_partial(
            q.unsqueeze(0),
            nope_part,
            scale_part,
            rope_part,
            indices.unsqueeze(0),
        )
        out = kernel_combine(partial_o, partial_lse)
        return out

    raise NotImplementedError(
        f"FP8 sparse-MLA decode kernel only configured for num_heads<=64, "
        f"got {num_heads}."
    )
