# SPDX-License-Identifier: Apache-2.0
"""FlyDSL implementation of sparse MLA decode kernels for ROCm.

This module is a FlyDSL re-implementation of the two TileLang kernels in
``tilelang_kernel.py`` used by NSA sparse-MLA decode:

  * :func:`build_sparse_mla_fwd_decode_partial_fp8` — sparse FP8 MLA forward
    decode partial kernel (FP8 Q/K/V, BF16 partial output, FP32 partial LSE)
    with split-K parallelism over the topk axis.
  * :func:`build_sparse_mla_fwd_decode_combine`     — log-sum-exp-weighted
    reduction of the per-split partial outputs into the final BF16 output.

Why the two-stage (split-K + combine) design and not a single-pass kernel
================================================================================

For NSA decode the typical workload looks like

    seq_len      = batch_size_tokens   (small for decode, e.g. 64 .. 256)
    num_heads    = 128
    h_per_block  = 16   (CTA tile in head dimension)
    topk         = 2048
    block_I      = 64   -> NI = 32 KV tiles to integrate over

Without split-K the partial-only ("1-pass") grid is

    grid = seq_len * (num_heads / h_per_block) = seq_len * 8

so on a 256-CU MI355X with seq_len = 64 we launch 512 CTAs — barely two
waves of CUs, each then crunching all 32 KV tiles sequentially. Latency is
bounded by the long inner KV loop.

With split-K we choose ``inner_iter`` so that

    grid = seq_len * (num_heads / h_per_block) * (NI / inner_iter)

reaches roughly ``cu * block_per_cu`` CTAs (≥ 4 CTAs per CU). The combine
kernel that follows is essentially free (one BF16 write per (head, dim) pair
plus a tiny LSE-weighted reduction), so the split-K version wins for
low-concurrency decode and matches the single-pass version when the batch is
already large enough to saturate the GPU. The TileLang reference makes the
same choice — we mirror it for direct comparability.

Algorithmic structure of the FP8 partial kernel
================================================================================

CTA tile sizes (match the TileLang reference):

  * Q tile :  ``H_PER_BLOCK x (D_V + D_TAIL) = 16 x 576``  FP8
  * KV tile:  ``BI          x (D_V + D_TAIL) = 64 x 576``  FP8
  * O tile :  ``H_PER_BLOCK x  D_V           = 16 x 512``  FP32 accumulator
              -> BF16 partial output

We split ``D_V = 512`` into ``NUM_GROUPS_DV = 4`` chunks of ``GROUP_SIZE = 128``.
The four ``Q @ K^T`` partial GEMMs accumulate into the same logits buffer,
but the four ``S @ V`` partial GEMMs each have their own register
accumulator. This shortens the MFMA dependency chain and is empirically
faster on gfx950 (TileLang does the same trick).

MFMA used: ``mfma_f32_16x16x32_fp8_fp8`` (gfx950+). Each MFMA tile is
``16 x 16`` with ``K = 32``, so per QK GEMM we issue 1 (M) x 4 (N) x 4 (K)
= 16 MFMA instructions per chunk plus 1x4x2 for the rope tail (D_TAIL=64).
Per SV GEMM (one per chunk) we issue 1 (M) x 8 (N=128/16) x 2 (K=BI/32) =
16 MFMA instructions.

Threads: 256 (= 4 wave-64). The four waves cooperate on the full 16x64 S
tile and partition the four 16x128 partial-O chunks one chunk per wave.

The combine kernel is trivial enough that we keep it short and simple:
each CTA owns ``head_per_block`` heads of one query token; one wave per head
performs the LSE max/sum reduction across the ``NI`` splits, and the rest of
the threads do the BF16 write-back.

Status (2026-05-07):
--------------------
Both kernels compile and run end-to-end on gfx950
(``rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260503``) and produce numerically
correct results vs the TileLang reference (max abs diff ~1e-4 on BF16
output, well within the FP8 quantization noise floor):

* ``build_sparse_mla_fwd_decode_combine``  -> COMPILES + matches TileLang.
* ``build_sparse_mla_fwd_decode_partial_fp8`` -> COMPILES + matches TileLang.

See ``test_flydsl_vs_tilelang.py`` for the side-by-side correctness test.

Performance baseline (median latency, ms/call, on MI355X / gfx950):

  case                  TileLang   FlyDSL   FlyDSL/TileLang
  ----------------------------------------- ----------------
  seq=64,  pages=8192    0.179     0.43     0.42x
  seq=128, pages=16384   0.315     1.18     0.27x
  seq=256, pages=32768   0.601     3.51     0.17x
  seq=512, pages=65536   1.172    12.98     0.09x

The FlyDSL version is currently 2-11x slower than TileLang. The gap is
dominated by:

  1. **SV B-pack: 8 strided byte-loads per pack.** V is stored row-major
     in LDS as ``[BI, GROUP_SIZE]``, but the SV GEMM wants 8 K-rows for
     a fixed N column = 8 stride-128 reads (vs a single ``ds_read_b64``
     for the QK B-pack). Fix options (each ~30 min): (a) duplicate V
     col-major in LDS (~32 KiB extra, fits gfx950's 160 KiB but not
     gfx942's 64 KiB); (b) use the gfx950 ``ds_read_tr8_b64`` HW
     transpose intrinsic.
  2. **No K/V double-buffering.** Each KV iter waits for the gather
     to finish before starting the QK GEMM. TileLang overlaps next-iter
     gather with current-iter compute via two LDS buffers.
  3. **No instruction scheduling barriers** (``rocdl.sched_*`` /
     ``sched_group_barrier``). TileLang's ``waves_per_eu`` and explicit
     scheduling annotations let the compiler interleave MFMA and VMEM.
  4. **No XOR LDS swizzle.** The current row-major KV LDS layout has
     bank conflicts when 4 lanes within a wave read the same column
     bytes; flash_attn_func.py uses ``col ^ ((row & 7) << 4)``.
  5. **No buffer_load_dwordx4_lds DMA.** gfx950 supports a global->LDS
     DMA path that bypasses VGPRs (saves register pressure during the
     gather).

Per-wave specialization in the partial kernel:
  Each wave owns one ``GROUP_SIZE``-wide D_V chunk (chunk == wave_id).
  Rather than gating chunk-specific work with ``if wave_id == chunk:``
  (which the FlyDSL AST rewriter can't thread Python list state through),
  we keep one contiguous LDS region for Q and KV and let every wave run
  the same kernel body, indexing into LDS via ``wave_id * <chunk_bytes>``
  runtime offsets. The partial-O accumulator ``acc_o[f]`` is similarly
  one flat per-wave list (no ``[chunk]`` dimension), and the writeback
  uses ``wave_id * GROUP_SIZE`` for the output column base.

FP8 / LLVM lowering workarounds:
  The MLIR / LLVM toolchain in FlyDSL ``v2026-04`` chokes on a few FP8
  edge cases that the kernel side-steps:
    * ``vector<NxfpX>`` LLVM lowering can fail. We allocate Q / KV / S
      LDS as ``i8`` bytes, bitcast on the boundaries, and load FP8 from
      global as ``i32`` words (4 bytes / lane) instead of FP8 vectors --
      the AMDGPU backend can't split ``v4i8`` raw_ptr_buffer_load.
    * ``arith.truncf f32 -> fp8`` has no LLVM translation. We use
      ``rocdl.cvt_pk_fp8_f32`` (passing the same value twice) to convert
      f32 -> FP8 packed in i32, then ``trunci`` to i8.
    * MFMA operands are passed as scalar ``i64`` (8 packed FP8 bytes
      bitcast through ``vector<8xi8>``).

What's not yet covered (future work):
  * Numerical correctness vs the TileLang reference (waiting on a small
    end-to-end test harness).
  * Performance tuning: LDS bank-conflict swizzle, K/V double buffering,
    explicit ``rocdl.sched_*`` barriers, ``buffer_load_dwordx4_lds`` DMA
    on gfx950+. The current kernel mirrors the TileLang algorithm
    line-for-line but does no scheduling tuning.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional, Tuple

import torch

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.utils import is_gfx95_supported, is_hip

# FlyDSL is imported lazily so this module can be imported on systems
# without FlyDSL (e.g. NVIDIA-only CI). The actual kernels still require
# FlyDSL at runtime.
try:
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import arith as _mlir_arith
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.expr import (
        arith,
        buffer_ops,
        const_expr,
        gpu,
        range_constexpr,
        rocdl,
        vector,
    )
    from flydsl.expr import math as fmath
    from flydsl.expr.typing import T, Vector as Vec
    from flydsl.expr.utils.arith import ArithValue, _to_raw
    from flydsl.runtime.device import get_rocm_arch
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

    _FLYDSL_AVAILABLE = True
except ImportError:  # pragma: no cover - tested on ROCm boxes only
    _FLYDSL_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────
# Module-level constants (mirror tilelang_kernel.py geometry exactly)
# ──────────────────────────────────────────────────────────────────────

LOG2E = math.log2(math.e)  # 1.4426950408889634

H_PER_BLOCK = 16
BI = 64
GROUP_SIZE = 128
NUM_GROUPS_DV = 4  # D_V (=512) split into 4 x GROUP_SIZE (=128) tiles
D_V_DEFAULT = 512
D_TAIL_DEFAULT = 64
THREADS_DEFAULT = 256
WARP_SIZE = 64
NUM_WAVES_DEFAULT = THREADS_DEFAULT // WARP_SIZE  # 4

# MFMA tile geometry for ``mfma_f32_16x16x32_fp8_fp8`` (gfx950+).
MFMA_M = 16
MFMA_N = 16
MFMA_K = 32
MFMA_LANES_PER_K_PACK = 8  # 8 FP8 per lane per (M-row, K-pack) cell

_is_hip = is_hip()
_is_gfx95_supported = is_gfx95_supported()
_is_fp8_fnuz = is_fp8_fnuz()


def _fp8_dtype_and_max() -> Tuple[str, float]:
    """Return ``(fp8_dtype_str, fp8_max_val)`` for the active ROCm arch."""
    if _is_fp8_fnuz:
        return "float8_e4m3fnuz", 240.0
    return "float8_e4m3fn", 448.0


@lru_cache(maxsize=8)
def _pick_inner_iter(seq: int, ni: int, cu: int, block_per_cu: int) -> int:
    """Pick the largest power-of-two divisor of ``ni`` that keeps each CU
    saturated (``seq * ni / inner_iter / cu >= block_per_cu``).

    Same heuristic as :func:`tilelang_kernel._pick_inner_iter` so the
    FlyDSL and TileLang backends are directly comparable.
    """
    max_it = int(seq * ni / max(cu * block_per_cu, 1))
    it = ni
    while it >= 2:
        if it <= max_it and ni % it == 0:
            return it
        it //= 2
    return 1


# ──────────────────────────────────────────────────────────────────────
# Builder: sparse_mla_fwd_decode_partial_fp8 (FlyDSL)
# ──────────────────────────────────────────────────────────────────────

def build_sparse_mla_fwd_decode_partial_fp8(
    *,
    num_heads: int,
    d_v: int = D_V_DEFAULT,
    d_tail: int = D_TAIL_DEFAULT,
    topk: int,
    sm_scale: Optional[float] = None,
    block_I: int = BI,
    inner_iter: int = 1,
    threads: int = THREADS_DEFAULT,
):
    """Build the FlyDSL sparse-MLA-decode partial kernel for FP8 KV.

    Mirrors :func:`tilelang_kernel.sparse_mla_fwd_decode_partial_fp8`.

    Arguments:
        num_heads:  number of query/output heads (``= 128`` for NSA)
        d_v:        value head dim (must be 512)
        d_tail:     rope tail head dim (typically 64)
        topk:       number of KV tokens scored per query
        sm_scale:   softmax scale (``1/sqrt(d_v + d_tail)`` if None)
        block_I:    KV tile size (BI). Must divide topk.
        inner_iter: number of consecutive KV tiles per CTA. Controls
                    split-K granularity. ``topk = block_I * inner_iter * n_groups``
                    where ``n_groups`` is the gridY size and the count of
                    partial outputs per query token.
        threads:    block size (must be multiple of WARP_SIZE = 64).

    Returns a ``@flyc.jit``-decorated launcher with signature::

        launch_partial(Q, KV, Indices, Partial_O, Partial_Lse, seq_len[, stream])

    where the tensor shapes match the TileLang reference:

        Q             : [1, seq_len, num_heads, d_v + d_tail]   FP8
        KV            : [1, num_pages, 1, d_v + d_tail]         FP8
        Indices       : [1, seq_len, 1, topk]                   i32
        partial_o     : [1, seq_len, n_groups, num_heads, d_v]  BF16
        partial_lse   : [1, seq_len, n_groups, num_heads]       FP32
    """
    if not _FLYDSL_AVAILABLE:
        raise RuntimeError("FlyDSL is not installed; cannot build kernel")
    if not _is_hip:
        raise RuntimeError("FlyDSL sparse MLA kernels only support ROCm")
    if d_v != D_V_DEFAULT:
        raise NotImplementedError(f"only d_v={D_V_DEFAULT} is supported, got {d_v}")
    assert topk % block_I == 0, (
        f"topk ({topk}) must be a multiple of block_I ({block_I})"
    )
    assert topk % (block_I * inner_iter) == 0, (
        f"topk ({topk}) must be a multiple of block_I*inner_iter "
        f"({block_I}*{inner_iter})"
    )
    assert threads % WARP_SIZE == 0, "threads must be a multiple of wave size 64"
    assert threads // WARP_SIZE == NUM_GROUPS_DV, (
        f"this implementation requires exactly {NUM_GROUPS_DV} waves "
        f"(one per D_V chunk); got threads={threads}"
    )
    assert num_heads <= H_PER_BLOCK or num_heads % H_PER_BLOCK == 0, (
        f"num_heads ({num_heads}) must be <= {H_PER_BLOCK} or a multiple of it"
    )
    assert d_tail % MFMA_K == 0, f"d_tail ({d_tail}) must be a multiple of {MFMA_K}"

    if sm_scale is None:
        sm_scale = (1.0 / (d_v + d_tail)) ** 0.5
    sm_scale_log2e = sm_scale * LOG2E

    fp8_dtype_str, fp8_max_val = _fp8_dtype_and_max()
    s_inv_scale_const = fp8_max_val
    s_scale_const = 1.0 / fp8_max_val

    n_groups = topk // (block_I * inner_iter)
    head_blocks_per_seq = (num_heads + H_PER_BLOCK - 1) // H_PER_BLOCK
    rope_offset_fp8 = d_v
    dim_quant_fp8 = d_v + d_tail
    num_waves = threads // WARP_SIZE  # 4

    arch = get_rocm_arch()
    fp8_dtype = fx.Float8E4M3FNUZ if _is_fp8_fnuz else fx.Float8E4M3FN
    bf16_dtype = fx.BFloat16
    f32_dtype = fx.Float32

    # ── LDS layout ──────────────────────────────────────────────────────
    #
    # We allocate the per-CTA LDS region in one contiguous chunk and slice
    # it into typed views via SmemPtr (offsets are in BYTES). The total
    # footprint is ~47 KiB which fits comfortably on gfx942 (64 KiB) and
    # gfx950 (160 KiB).
    #
    # The Q and KV chunks are allocated as SINGLE contiguous regions of size
    # ``NUM_GROUPS_DV * <chunk_bytes>`` so we can index them with runtime
    # ``wave_id`` arithmetic in the SV GEMM (without per-chunk Python ``if``
    # branches that the FlyDSL AST rewriter cannot thread through scf.if).
    #
    #   q_all         : NUM_GROUPS_DV x H_PER_BLOCK x GROUP_SIZE FP8 = 8192 B
    #   q_tail        : H_PER_BLOCK x D_TAIL                    FP8 = 1024 B
    #   kv_all        : NUM_GROUPS_DV x BI         x GROUP_SIZE FP8 = 32768 B
    #   k_tail        : BI          x D_TAIL                    FP8 = 4096 B
    #   s_fp8         : H_PER_BLOCK x BI                        FP8 = 1024 B
    #   page_idx      : BI                                      i32 =  256 B
    #   reduce_scratch: 16                                      f32 =   64 B
    allocator = SmemAllocator(
        None,
        arch=arch,
        global_sym_name=(
            f"sparse_mla_partial_fp8_smem_h{num_heads}_t{topk}_i{inner_iter}"
        ),
    )

    def _alloc(n_bytes: int, align: int = 16) -> int:
        off = allocator._align(allocator.ptr, align)
        allocator.ptr = off + n_bytes
        return off

    q_chunk_bytes = H_PER_BLOCK * GROUP_SIZE
    q_tail_bytes = H_PER_BLOCK * d_tail
    kv_chunk_bytes = block_I * GROUP_SIZE
    k_tail_bytes = block_I * d_tail
    s_fp8_bytes = H_PER_BLOCK * block_I
    page_idx_bytes = block_I * 4
    # Cross-wave reduction scratch: NUM_WAVES x H_PER_BLOCK fp32 slots so
    # every wave can write all 16 of its row-maxes (or row-sums) without
    # racing the other wave's writes.
    reduce_scratch_bytes = NUM_WAVES_DEFAULT * H_PER_BLOCK * 4
    q_all_bytes = NUM_GROUPS_DV * q_chunk_bytes
    kv_all_bytes = NUM_GROUPS_DV * kv_chunk_bytes

    q_base_offset = _alloc(q_all_bytes)
    q_tail_offset = _alloc(q_tail_bytes)
    kv_base_offset = _alloc(kv_all_bytes)
    k_tail_offset = _alloc(k_tail_bytes)
    s_fp8_offset = _alloc(s_fp8_bytes)
    page_idx_offset = _alloc(page_idx_bytes)
    reduce_scratch_offset = _alloc(reduce_scratch_bytes)

    @flyc.kernel
    def partial_kernel(
        Q: fx.Tensor,
        KV: fx.Tensor,
        Indices: fx.Tensor,
        Partial_O: fx.Tensor,
        Partial_Lse: fx.Tensor,
    ):
        # ── MFMA pack types ────────────────────────────────────────────
        # Constructed inside the kernel body so the MLIR context is active
        # (``T.vec`` / ``Type.get`` require a Context).
        #
        # NOTE: LLVM lowering of ``vector<8xf8E4M3FN>`` is not always
        # compatible (assertion in TypeToLLVM). We work around this by
        # loading from LDS via a same-bit-width integer type
        # (``vector<2xi32>`` = 8 bytes), then bitcasting to ``i64`` for
        # the ``mfma_f32_16x16x32_fp8_fp8`` operand.
        v4f32_t = T.vec(4, f32_dtype.ir_type)  # C/D acc:  4 FP32 / lane
        v4i8_t = T.vec(4, T.i8)    # 4-byte cooperative load width
        v8i8_t = T.vec(8, T.i8)    # 8-byte MFMA A/B pack (FP8 viewed as i8)
        v16i8_t = T.vec(16, T.i8)  # 16-byte cooperative load width
        v1i8_t = T.vec(1, T.i8)
        v1i32_t = T.vec(1, T.i32)
        v1i64_t = T.vec(1, T.i64)
        v1f32_t = T.vec(1, f32_dtype.ir_type)
        v1fp8_t = T.vec(1, fp8_dtype.ir_type)

        def _pack_v8i8_as_i64(v8i8):
            """``mfma_f32_16x16x32_fp8_fp8`` takes A/B operands as scalar
            ``i64`` (8 FP8 packed into 64 bits). We bitcast the
            ``vector<8xi8>`` LDS load to ``vector<1xi64>`` and extract
            the scalar."""
            v1i64 = vector.bitcast(v1i64_t, v8i8)
            return vector.extract(
                v1i64, static_position=[0], dynamic_position=[]
            )

        # ── Indices ────────────────────────────────────────────────────
        bx = fx.Index(gpu.block_idx.x)
        by = fx.Index(gpu.block_idx.y)
        tid = fx.Index(gpu.thread_idx.x)
        wave_id = tid // fx.Index(WARP_SIZE)
        lane = tid % fx.Index(WARP_SIZE)
        lane_mod_16 = lane % fx.Index(MFMA_M)
        lane_div_16 = lane // fx.Index(MFMA_M)

        # CTA -> (sequence token, head-block, kv split-K group)
        s_i = bx // fx.Index(head_blocks_per_seq)
        head_block_i = bx % fx.Index(head_blocks_per_seq)
        H0 = head_block_i * fx.Index(H_PER_BLOCK)
        group_i = by

        # ── Buffer resources ───────────────────────────────────────────
        q_rsrc = buffer_ops.create_buffer_resource(Q, max_size=True)
        kv_rsrc = buffer_ops.create_buffer_resource(KV, max_size=True)
        idx_rsrc = buffer_ops.create_buffer_resource(Indices, max_size=True)
        po_rsrc = buffer_ops.create_buffer_resource(Partial_O, max_size=True)
        plse_rsrc = buffer_ops.create_buffer_resource(Partial_Lse, max_size=True)

        # ── LDS pointers ───────────────────────────────────────────────
        # Q and KV LDS are allocated as ``i8`` (raw bytes) so we can read
        # them back as ``vector<2xi32>`` for the MFMA pack (LLVM's vector
        # lowering doesn't always handle ``vector<8xf8>`` cleanly). The
        # writes still come from FP8 ``buffer_load`` results — we bitcast
        # FP8 vectors to ``vector<Nxi8>`` before the LDS store.
        #
        # ``s_fp8_smem`` is also i8-typed for the same reason: we write
        # quantized scalar fp8 values via per-byte stores (vector<1xi8>).
        base = allocator.get_base()
        q_smem = SmemPtr(
            base, q_base_offset, T.i8, shape=(q_all_bytes,)
        ).get()
        q_tail_smem = SmemPtr(
            base, q_tail_offset, T.i8, shape=(q_tail_bytes,)
        ).get()
        kv_smem = SmemPtr(
            base, kv_base_offset, T.i8, shape=(kv_all_bytes,)
        ).get()
        k_tail_smem = SmemPtr(
            base, k_tail_offset, T.i8, shape=(k_tail_bytes,)
        ).get()
        s_fp8_smem = SmemPtr(
            base, s_fp8_offset, T.i8, shape=(s_fp8_bytes,)
        ).get()
        page_idx_smem = SmemPtr(
            base, page_idx_offset, T.i32, shape=(block_I,)
        ).get()
        reduce_smem = SmemPtr(
            base,
            reduce_scratch_offset,
            f32_dtype.ir_type,
            shape=(NUM_WAVES_DEFAULT * H_PER_BLOCK,),
        ).get()

        # ── Constants ──────────────────────────────────────────────────
        c_neg_big = fx.Float32(-(2.0 ** 30))
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_log2e_scale = fx.Float32(sm_scale_log2e)
        c_fp8_max = fx.Float32(fp8_max_val)
        c_neg_fp8_max = fx.Float32(-fp8_max_val)
        c_s_scale = fx.Float32(s_scale_const)

        # ── Per-thread accumulators ─────────────────────────────────────
        # Each wave owns one D_V chunk (chunk == wave_id). The 16 x 128 =
        # 2048 fp32 partial-O accumulator is held in 8 v4f32 register
        # fragments per lane (= 32 fp32 / lane * 64 lanes). Each wave keeps
        # only ITS OWN chunk's accumulators, so this is a flat list of 8
        # ``Vec`` (no per-chunk dim). We index ``acc_o[ns]`` for the n-tile
        # within the wave's chunk.
        acc_o = [
            Vec.filled(4, 0.0, f32_dtype) for _ in range(GROUP_SIZE // MFMA_N)
        ]
        # m_i, sumexp: per-row state held in MFMA C-fragment lane layout
        # (4 fp32 per lane covering rows lane_div_16*4 .. lane_div_16*4+3).
        m_i = Vec.filled(4, -(2.0 ** 30), f32_dtype)
        sumexp = Vec.filled(4, 0.0, f32_dtype)

        # Wave-id-driven chunk base offset (in FP8 elements). Each wave
        # owns ``wave_chunk_off .. wave_chunk_off + kv_chunk_bytes - 1`` of
        # the contiguous ``kv_smem`` and the matching slice of ``q_smem``.
        wave_kv_chunk_off = wave_id * fx.Index(kv_chunk_bytes)
        wave_q_chunk_off = wave_id * fx.Index(q_chunk_bytes)

        # ── Cooperative Q load ─────────────────────────────────────────
        # 16 (rows) x 128 (FP8) per chunk. THREADS_PER_ROW = 32 lanes per
        # row do vec-4 loads, cooperating across H_PER_BLOCK rows in two
        # batches. Tail (16 x 64 FP8 = 1024 B) fits in one shot for 256
        # threads * 4 B per thread.
        Q_HEAD_STRIDE = d_v + d_tail
        Q_SEQ_STRIDE = num_heads * Q_HEAD_STRIDE
        s_seq_off = s_i * fx.Index(Q_SEQ_STRIDE)
        THREADS_PER_ROW = GROUP_SIZE // 4
        ROWS_PER_BATCH = threads // THREADS_PER_ROW  # = 8
        BATCHES_PER_TILE = H_PER_BLOCK // ROWS_PER_BATCH  # = 2
        load_row = tid // fx.Index(THREADS_PER_ROW)
        load_col4 = (tid % fx.Index(THREADS_PER_ROW)) * fx.Index(4)

        for chunk in range_constexpr(NUM_GROUPS_DV):
            base_col = chunk * GROUP_SIZE
            chunk_lds_base = fx.Index(chunk * q_chunk_bytes)
            for batch in range_constexpr(BATCHES_PER_TILE):
                row = load_row + fx.Index(batch * ROWS_PER_BATCH)
                # NOTE: ``buffer_load`` with ``dtype=T.i32`` treats the
                # offset as an i32-element index (= byte-offset / 4). All
                # of our address arithmetic is in BYTES (FP8 = 1 byte), so
                # divide by 4 to convert to dword units.
                g_off_bytes = (
                    s_seq_off
                    + (H0 + row) * fx.Index(Q_HEAD_STRIDE)
                    + fx.Index(base_col)
                    + load_col4
                )
                g_off_dwords = g_off_bytes // fx.Index(4)
                # AMDGPU backend can't split ``v4i8`` raw buffer loads; use
                # a single-i32 load (= 4 bytes) and bitcast to vector<4xi8>
                # for the i8-typed LDS store.
                word_i32 = buffer_ops.buffer_load(
                    q_rsrc, g_off_dwords, vec_width=1, dtype=T.i32
                )
                word_v1i32 = vector.from_elements(v1i32_t, [word_i32])
                vec_i8 = vector.bitcast(v4i8_t, word_v1i32)
                lds_off = chunk_lds_base + row * fx.Index(GROUP_SIZE) + load_col4
                vector.store(vec_i8, q_smem, [lds_off])

        # Tail Q (16 x 64 FP8): 256 threads * 4B = 1024 B == tile.
        # row = tid // 16, col4 = (tid % 16) * 4.
        TAIL_THREADS_PER_ROW = d_tail // 4
        tail_row = tid // fx.Index(TAIL_THREADS_PER_ROW)
        tail_col4 = (tid % fx.Index(TAIL_THREADS_PER_ROW)) * fx.Index(4)
        if tail_row < fx.Index(H_PER_BLOCK):
            g_off_bytes = (
                s_seq_off
                + (H0 + tail_row) * fx.Index(Q_HEAD_STRIDE)
                + fx.Index(d_v)
                + tail_col4
            )
            g_off_dwords = g_off_bytes // fx.Index(4)
            word_i32 = buffer_ops.buffer_load(
                q_rsrc, g_off_dwords, vec_width=1, dtype=T.i32
            )
            word_v1i32 = vector.from_elements(v1i32_t, [word_i32])
            vec_i8 = vector.bitcast(v4i8_t, word_v1i32)
            lds_off = tail_row * fx.Index(d_tail) + tail_col4
            vector.store(vec_i8, q_tail_smem, [lds_off])

        gpu.barrier()

        # ── KV / softmax loop ──────────────────────────────────────────
        # NOTE: ``range_constexpr`` (= Python ``range``, unrolled at trace
        # time) instead of ``range`` so we don't trigger the AST rewriter's
        # scf.for emission. ``inner_iter`` is a builder-time constant and
        # always small (1 / 2 / 4 / 8 in practice), so register-level
        # accumulation across iterations is fine.
        s_idx_off = s_i * fx.Index(topk)

        for k_i in range_constexpr(inner_iter):
            topk_block_i = group_i * fx.Index(inner_iter) + fx.Index(k_i)
            base_idx_off = s_idx_off + topk_block_i * fx.Index(block_I)

            # ── (1) Load BI indices into LDS ──────────────────────────
            # First 64 threads (wave 0) each load one i32 page index.
            # buffer_load with vec_width=1 returns a scalar (i32), not a
            # vector<1xi32>; we wrap it in a vector for the LDS vector.store.
            if tid < fx.Index(block_I):
                idx_scalar = buffer_ops.buffer_load(
                    idx_rsrc,
                    base_idx_off + tid,
                    vec_width=1,
                    dtype=T.i32,
                )
                idx_safe = arith.select(
                    idx_scalar >= fx.Int32(0), idx_scalar, fx.Int32(0)
                )
                idx_v_safe = vector.from_elements(v1i32_t, [idx_safe])
                vector.store(idx_v_safe, page_idx_smem, [tid])

            gpu.barrier()

            # ── (2) Cooperative gather KV[idx, :] into LDS ─────────────
            # 4 chunks of 64x128 + 64x64 tail. We use 16-byte (16 FP8)
            # vector loads. Each row fans out to GROUP_SIZE/16 = 8 lanes
            # for the chunks and D_TAIL/16 = 4 lanes for the tail.
            ROWS_PER_KV_LOAD = threads // (GROUP_SIZE // 16)  # 32
            KV_BATCHES = block_I // ROWS_PER_KV_LOAD  # 2
            kv_load_col = (tid % fx.Index(GROUP_SIZE // 16)) * fx.Index(16)
            kv_load_row = tid // fx.Index(GROUP_SIZE // 16)

            for chunk in range_constexpr(NUM_GROUPS_DV):
                base_col = chunk * GROUP_SIZE
                chunk_lds_base = fx.Index(chunk * kv_chunk_bytes)
                for batch in range_constexpr(KV_BATCHES):
                    bi_row = kv_load_row + fx.Index(batch * ROWS_PER_KV_LOAD)
                    page_v = vector.load_op(v1i32_t, page_idx_smem, [bi_row])
                    page_i32 = vector.extract(
                        page_v, static_position=[0], dynamic_position=[]
                    )
                    # Cast page (i32) to Index so we can do uniform Index
                    # arithmetic with kv_load_col (Index from tid).
                    page_idx = arith.index_cast(T.index, page_i32)
                    g_off_bytes = (
                        fx.Index(page_idx) * fx.Index(dim_quant_fp8)
                        + fx.Index(base_col)
                        + kv_load_col
                    )
                    # 16 bytes / lane = vector<4xi32>; bitcast to <16xi8>.
                    # buffer_load with dtype=i32 expects offset in i32-element
                    # units (= bytes / 4).
                    g_off_dwords = g_off_bytes // fx.Index(4)
                    vec_i32 = buffer_ops.buffer_load(
                        kv_rsrc, g_off_dwords, vec_width=4, dtype=T.i32
                    )
                    vec_i8 = vector.bitcast(v16i8_t, vec_i32)
                    lds_off = (
                        chunk_lds_base
                        + bi_row * fx.Index(GROUP_SIZE)
                        + kv_load_col
                    )
                    vector.store(vec_i8, kv_smem, [lds_off])

            # K tail (64 x 64 FP8 = 4096 B). 256 threads x 16 B = 4096 B.
            tail_lanes_per_row = d_tail // 16  # 4
            t_row = tid // fx.Index(tail_lanes_per_row)
            t_col = (tid % fx.Index(tail_lanes_per_row)) * fx.Index(16)
            if t_row < fx.Index(block_I):
                page_v = vector.load_op(v1i32_t, page_idx_smem, [t_row])
                page_i32 = vector.extract(
                    page_v, static_position=[0], dynamic_position=[]
                )
                page_idx = arith.index_cast(T.index, page_i32)
                g_off_bytes = (
                    fx.Index(page_idx) * fx.Index(dim_quant_fp8)
                    + fx.Index(rope_offset_fp8)
                    + t_col
                )
                g_off_dwords = g_off_bytes // fx.Index(4)
                vec_i32 = buffer_ops.buffer_load(
                    kv_rsrc, g_off_dwords, vec_width=4, dtype=T.i32
                )
                vec_i8 = vector.bitcast(v16i8_t, vec_i32)
                lds_off = t_row * fx.Index(d_tail) + t_col
                vector.store(vec_i8, k_tail_smem, [lds_off])

            gpu.barrier()

            # ── (3) QK GEMM: S = Q @ K^T (4 D_V chunks + tail) ────────
            # MFMA layout for f32_16x16x32_fp8_fp8 (wave64):
            #   A (M=16, K=32):  row    = lane % 16
            #                    k_off  = (lane // 16) * 8
            #   B (N=16, K=32):  col    = lane % 16
            #                    k_off  = (lane // 16) * 8
            #   C (M=16, N=16):  row    = (lane // 16) * 4 + r  (r in 0..3)
            #                    col    = lane % 16
            #
            # Wave w owns S columns w*16 .. w*16+15 (BI = 4*16 = 64).
            wave_n_off = wave_id * fx.Index(MFMA_N)
            # acc_s as raw vector<4xf32> ir.Value (we don't need the
            # element-wise Vector wrapper inside the GEMM loop).
            acc_s_raw = vector.from_elements(
                v4f32_t,
                [fx.Float32(0.0), fx.Float32(0.0), fx.Float32(0.0), fx.Float32(0.0)],
            )

            def _qk_chunk(dim_per_chunk, q_base, q_lds, kv_base, kv_lds, acc):
                """Accumulate ``Q_chunk @ K_chunk^T`` into ``acc``.

                ``q_base`` / ``kv_base`` are static LDS-element offsets to
                the start of this chunk inside the contiguous q_smem /
                kv_smem regions (or 0 for the rope-tail buffers). The LDS
                regions are i8-typed so we load ``vector<2xi32>`` (8 bytes)
                and bitcast to ``i64`` for the MFMA.
                """
                for ks in range_constexpr(dim_per_chunk // MFMA_K):
                    q_off = (
                        q_base
                        + lane_mod_16 * fx.Index(dim_per_chunk)
                        + fx.Index(ks * MFMA_K)
                        + lane_div_16 * fx.Index(MFMA_LANES_PER_K_PACK)
                    )
                    a_v = vector.load_op(v8i8_t, q_lds, [q_off])
                    a = _pack_v8i8_as_i64(a_v)
                    k_off = (
                        kv_base
                        + (wave_n_off + lane_mod_16) * fx.Index(dim_per_chunk)
                        + fx.Index(ks * MFMA_K)
                        + lane_div_16 * fx.Index(MFMA_LANES_PER_K_PACK)
                    )
                    b_v = vector.load_op(v8i8_t, kv_lds, [k_off])
                    b = _pack_v8i8_as_i64(b_v)
                    acc = rocdl.mfma_f32_16x16x32_fp8_fp8(
                        v4f32_t, [a, b, acc, 0, 0, 0]
                    )
                return acc

            for chunk in range_constexpr(NUM_GROUPS_DV):
                acc_s_raw = _qk_chunk(
                    GROUP_SIZE,
                    fx.Index(chunk * q_chunk_bytes),
                    q_smem,
                    fx.Index(chunk * kv_chunk_bytes),
                    kv_smem,
                    acc_s_raw,
                )
            acc_s_raw = _qk_chunk(
                d_tail,
                fx.Index(0),
                q_tail_smem,
                fx.Index(0),
                k_tail_smem,
                acc_s_raw,
            )

            # Mask invalid columns (negative indices -> -inf).
            col_in_S = wave_n_off + lane_mod_16
            orig_idx_scalar = buffer_ops.buffer_load(
                idx_rsrc,
                base_idx_off + col_in_S,
                vec_width=1,
                dtype=T.i32,
            )
            valid = orig_idx_scalar >= fx.Int32(0)
            masked_elems = []
            for r in range_constexpr(4):
                e = vector.extract(
                    acc_s_raw, static_position=[r], dynamic_position=[]
                )
                masked = arith.select(valid, e, _to_raw(c_neg_big))
                masked_elems.append(masked)
            acc_s_raw = vector.from_elements(v4f32_t, masked_elems)

            # ── Online softmax (per-row max & sum across all 64 cols) ──
            #
            # In MFMA-C layout each wave owns 16 columns of S; the same row
            # is split across all 4 waves (each wave has 4 rows per
            # (lane_div_16, r)). We do a 16-wide xor reduction within each
            # wave (1 .. 8 hops), then a small LDS scratch dance for the
            # cross-wave max.
            local_max = []
            for r in range_constexpr(4):
                v = vector.extract(
                    acc_s_raw, static_position=[r], dynamic_position=[]
                )
                for sh in [8, 4, 2, 1]:
                    peer = ArithValue(v).shuffle_xor(
                        fx.Int32(sh), fx.Int32(WARP_SIZE)
                    )
                    v = arith.maximumf(v, peer)
                local_max.append(v)  # raw ir.Value

            # Cross-wave reduction. reduce_smem layout is [num_waves,
            # H_PER_BLOCK] fp32 = 16 row slots per wave. Lane (lane_div_16,
            # lane_mod_16==0) writes row_max for rows
            # ``lane_div_16*4 + r`` (r in 0..3) at offset
            # ``wave_id*16 + lane_div_16*4 + r``. Other lanes in the same
            # block hold the same value (after XOR-16) but only one of
            # them needs to write, so we gate on lane_mod_16==0.
            if lane_mod_16 == fx.Index(0):
                for r in range_constexpr(4):
                    off = (
                        wave_id * fx.Index(H_PER_BLOCK)
                        + lane_div_16 * fx.Index(4)
                        + fx.Index(r)
                    )
                    vec1 = vector.from_elements(v1f32_t, [local_max[r]])
                    vector.store(vec1, reduce_smem, [off])
            gpu.barrier()
            # Each lane reads back NUM_WAVES per-wave maxes for its 4 owned
            # rows (rows ``lane_div_16*4 + r``) and reduces.
            row_max_cross = []
            for r in range_constexpr(4):
                row_id_in_block = lane_div_16 * fx.Index(4) + fx.Index(r)
                vmax = _to_raw(c_neg_big)
                for w in range_constexpr(num_waves):
                    off = (
                        fx.Index(w * H_PER_BLOCK)
                        + row_id_in_block
                    )
                    val_v = vector.load_op(v1f32_t, reduce_smem, [off])
                    val = vector.extract(
                        val_v, static_position=[0], dynamic_position=[]
                    )
                    vmax = arith.maximumf(vmax, val)
                row_max_cross.append(vmax)  # raw ir.Value
            gpu.barrier()  # ensure scratch is consumed before we reuse it

            # m_new = max(m_i, row_max);  alpha = exp2((m_i - m_new) * sm_scale*log2e)
            m_new = []
            alpha_v = []
            for r in range_constexpr(4):
                m_i_r = m_i[r]  # m_i is still a Vec (Vec.filled works)
                mn = arith.maximumf(_to_raw(m_i_r), row_max_cross[r])
                m_new.append(mn)
                diff_scaled = _to_raw(
                    (m_i_r - fx.Float32(mn)) * c_log2e_scale
                )
                alpha_v.append(fmath.exp2(diff_scaled, fastmath="fast"))
            # Rebuild m_i as a Vec for downstream subscripting.
            m_i = Vec(vector.from_elements(v4f32_t, m_new), (4,), f32_dtype)
            alpha_vec = Vec(
                vector.from_elements(v4f32_t, alpha_v), (4,), f32_dtype
            )

            # P = exp2((S - m_new) * sm_scale * log2e). Per-row sum first.
            p_elems = []  # list of raw ir.Value
            local_sum = []
            for r in range_constexpr(4):
                s_r = vector.extract(
                    acc_s_raw, static_position=[r], dynamic_position=[]
                )
                m_r = m_i[r]
                diff_scaled = _to_raw(
                    (fx.Float32(s_r) - m_r) * c_log2e_scale
                )
                p = fmath.exp2(diff_scaled, fastmath="fast")
                p_elems.append(p)
                local_sum.append(p)

            # 16-wide xor-reduce per row, then cross-wave via LDS.
            for r in range_constexpr(4):
                v = local_sum[r]
                for sh in [8, 4, 2, 1]:
                    peer = ArithValue(v).shuffle_xor(
                        fx.Int32(sh), fx.Int32(WARP_SIZE)
                    )
                    v = arith.addf(v, peer)
                local_sum[r] = v

            # Same per-row, per-wave layout as the max reduction above.
            if lane_mod_16 == fx.Index(0):
                for r in range_constexpr(4):
                    off = (
                        wave_id * fx.Index(H_PER_BLOCK)
                        + lane_div_16 * fx.Index(4)
                        + fx.Index(r)
                    )
                    vec1 = vector.from_elements(v1f32_t, [local_sum[r]])
                    vector.store(vec1, reduce_smem, [off])
            gpu.barrier()
            row_sum_cross = []
            for r in range_constexpr(4):
                row_id_in_block = lane_div_16 * fx.Index(4) + fx.Index(r)
                vsum = _to_raw(c_zero_f)
                for w in range_constexpr(num_waves):
                    off = (
                        fx.Index(w * H_PER_BLOCK)
                        + row_id_in_block
                    )
                    val_v = vector.load_op(v1f32_t, reduce_smem, [off])
                    val = vector.extract(
                        val_v, static_position=[0], dynamic_position=[]
                    )
                    vsum = arith.addf(vsum, val)
                row_sum_cross.append(vsum)
            gpu.barrier()

            # Update sumexp = sumexp * alpha + row_sum.
            new_sumexp = []
            for r in range_constexpr(4):
                rsum_v = fx.Float32(row_sum_cross[r])
                new_sumexp.append(
                    _to_raw(sumexp[r] * alpha_vec[r] + rsum_v)
                )
            sumexp = Vec(
                vector.from_elements(v4f32_t, new_sumexp), (4,), f32_dtype
            )

            # ── Rescale acc_o by alpha ────────────────────────────────
            # Every wave runs the same body and rescales its own
            # accumulators (no per-wave Python ``if`` needed because
            # ``acc_o`` already holds only this wave's chunk).
            for f in range_constexpr(GROUP_SIZE // MFMA_N):
                old = acc_o[f]
                new_frag = []
                for r in range_constexpr(4):
                    new_frag.append(_to_raw(old[r] * alpha_vec[r]))
                acc_o[f] = Vec(
                    vector.from_elements(v4f32_t, new_frag),
                    (4,),
                    f32_dtype,
                )

            # ── Quantize P -> FP8 with FP8_MAX scaling, write to LDS ──
            # Layout target: s_fp8[row, col] FP8 of shape [H_PER_BLOCK, BI].
            # Each lane holds P values for 4 rows it owns (lane_div_16*4..+3)
            # at column wave_n_off + lane_mod_16.
            #
            # ``arith.truncf f32 -> fp8`` isn't lowerable to LLVM, so we use
            # the ROCDL ``cvt_pk_fp8_f32`` intrinsic which packs two f32
            # values into the lower 16 bits of an i32 (= 2 FP8 bytes). We
            # pass the same value twice so the lower byte is the FP8 we
            # want, then ``trunci`` to i8.
            c0_i32 = _to_raw(fx.Int32(0))
            for r in range_constexpr(4):
                row_id = lane_div_16 * fx.Index(4) + fx.Index(r)
                col_id = wave_n_off + lane_mod_16
                p_scaled = arith.mulf(p_elems[r], _to_raw(c_fp8_max))
                p_clamp_lo = arith.maximumf(p_scaled, _to_raw(c_neg_fp8_max))
                p_clamped = arith.minimumf(p_clamp_lo, _to_raw(c_fp8_max))
                # cvt_pk_fp8_f32(result_ty, srcA, srcB, packed_i32, word_sel)
                pair_i32 = rocdl.cvt_pk_fp8_f32(
                    T.i32, p_clamped, p_clamped, c0_i32, False
                )
                p_i8 = _mlir_arith.trunci(T.i8, pair_i32)
                vec1_i8 = vector.from_elements(v1i8_t, [p_i8])
                lds_off = row_id * fx.Index(block_I) + col_id
                vector.store(vec1_i8, s_fp8_smem, [lds_off])

            gpu.barrier()

            # ── (4) SV GEMM: O += S_FP8 @ V (one chunk per wave) ──────
            # Every wave runs the same body and operates on its own D_V
            # chunk (== wave_id). The chunk's KV LDS slice starts at
            # ``wave_kv_chunk_off`` inside the contiguous ``kv_smem``.
            #   N (output cols)   = GROUP_SIZE = 128  -> 8 MFMA N-tiles
            #   K (KV reduction)  = BI         = 64   -> 2 MFMA K-tiles
            #
            # B operand is V[k_row, n_col] from this wave's kv chunk, laid
            # out row-major [BI, GROUP_SIZE]. For the MFMA B operand with
            # K=32:
            #   row index in B = n     (== ns*16 + lane_mod_16)
            #   k index in B   = ks*32 + lane_div_16*8 + (0..7)
            # LDS address: wave_kv_chunk_off + k_row * GROUP_SIZE + n_col.
            for ns in range_constexpr(GROUP_SIZE // MFMA_N):
                sv_acc = vector.from_elements(
                    v4f32_t,
                    [
                        fx.Float32(0.0),
                        fx.Float32(0.0),
                        fx.Float32(0.0),
                        fx.Float32(0.0),
                    ],
                )
                for ks in range_constexpr(block_I // MFMA_K):
                    # A pack from s_fp8 (i8-typed LDS, [H_PER_BLOCK, BI]).
                    a_off = (
                        lane_mod_16 * fx.Index(block_I)
                        + fx.Index(ks * MFMA_K)
                        + lane_div_16 * fx.Index(MFMA_LANES_PER_K_PACK)
                    )
                    a_v = vector.load_op(v8i8_t, s_fp8_smem, [a_off])
                    a = _pack_v8i8_as_i64(a_v)
                    # B pack from V (8 K rows for fixed n_col). LDS is
                    # i8-typed so we read 1 byte at a time then assemble
                    # an 8-byte vector and bitcast to i64.
                    n_col = fx.Index(ns * MFMA_N) + lane_mod_16
                    k_row_base = (
                        fx.Index(ks * MFMA_K)
                        + lane_div_16 * fx.Index(MFMA_LANES_PER_K_PACK)
                    )
                    b_elems = []
                    for kk in range_constexpr(MFMA_LANES_PER_K_PACK):
                        addr = (
                            wave_kv_chunk_off
                            + (k_row_base + fx.Index(kk))
                            * fx.Index(GROUP_SIZE)
                            + n_col
                        )
                        v_one_v = vector.load_op(v1i8_t, kv_smem, [addr])
                        v_one = vector.extract(
                            v_one_v, static_position=[0], dynamic_position=[]
                        )
                        b_elems.append(v_one)
                    b_v8i8 = vector.from_elements(v8i8_t, b_elems)
                    b = _pack_v8i8_as_i64(b_v8i8)
                    sv_acc = rocdl.mfma_f32_16x16x32_fp8_fp8(
                        v4f32_t, [a, b, sv_acc, 0, 0, 0]
                    )

                # acc_o += sv_acc / FP8_MAX (undo the inv-scale used when
                # quantizing P).
                old = acc_o[ns]
                merged = []
                for r in range_constexpr(4):
                    sv_r = vector.extract(
                        sv_acc, static_position=[r], dynamic_position=[]
                    )
                    merged.append(
                        _to_raw(old[r] + fx.Float32(sv_r) * c_s_scale)
                    )
                acc_o[ns] = Vec(
                    vector.from_elements(v4f32_t, merged),
                    (4,),
                    f32_dtype,
                )

            gpu.barrier()  # safe to overwrite s_fp8 / page_idx next iter

        # ── Normalize and write back partial_o, partial_lse ────────────
        inv_denom = []
        for r in range_constexpr(4):
            sumv = sumexp[r]
            is_zero = sumv == c_zero_f
            denom = fx.Float32(arith.select(is_zero, _to_raw(c_one_f), _to_raw(sumv)))
            inv_denom.append(c_one_f / denom)

        po_seq_off = s_i * fx.Index(n_groups * num_heads * d_v)
        po_group_off = group_i * fx.Index(num_heads * d_v)

        # Each wave writes its own D_V chunk (chunk == wave_id). The
        # output column base is ``wave_id * GROUP_SIZE``.
        wave_col_base = wave_id * fx.Index(GROUP_SIZE)
        for ns in range_constexpr(GROUP_SIZE // MFMA_N):
            frag = acc_o[ns]
            for r in range_constexpr(4):
                row_id = lane_div_16 * fx.Index(4) + fx.Index(r)
                col_id = wave_col_base + fx.Index(ns * MFMA_N) + lane_mod_16
                val = frag[r] * inv_denom[r]
                bf16_val = arith.trunc_f(bf16_dtype.ir_type, _to_raw(val))
                head_off = (H0 + row_id) * fx.Index(d_v)
                out_off = po_seq_off + po_group_off + head_off + col_id
                buffer_ops.buffer_store(bf16_val, po_rsrc, out_off)

        # partial_lse[s_i, group_i, H0+row].
        # The LSE is one scalar per (head row, query token). Per MFMA-C
        # layout, every (lane_mod_16) column of the same wave has the same
        # row state. We let one wave (wave 0) and one column (lane_mod_16
        # == 0) write all 4 rows of LSE.
        plse_seq_off = s_i * fx.Index(n_groups * num_heads)
        plse_group_off = group_i * fx.Index(num_heads)
        if wave_id == fx.Index(0) and lane_mod_16 == fx.Index(0):
            for r in range_constexpr(4):
                row_id = lane_div_16 * fx.Index(4) + fx.Index(r)
                sumv = sumexp[r]
                is_zero = sumv == c_zero_f
                # log2(sumv) + m_i * sm_scale * log2e if sumv > 0 else -2^30.
                log2_s = fx.Float32(fmath.log2(_to_raw(sumv), fastmath="fast"))
                lse_pos = log2_s + m_i[r] * c_log2e_scale
                lse_val = fx.Float32(
                    arith.select(is_zero, _to_raw(c_neg_big), _to_raw(lse_pos))
                )
                head_off = H0 + row_id
                out_off = plse_seq_off + plse_group_off + head_off
                buffer_ops.buffer_store(_to_raw(lse_val), plse_rsrc, out_off)

    @flyc.jit
    def launch_partial(
        Q: fx.Tensor,
        KV: fx.Tensor,
        Indices: fx.Tensor,
        Partial_O: fx.Tensor,
        Partial_Lse: fx.Tensor,
        seq_len: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        grid_x = fx.Index(seq_len) * fx.Index(head_blocks_per_seq)
        partial_kernel(Q, KV, Indices, Partial_O, Partial_Lse).launch(
            grid=(grid_x, n_groups, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    return launch_partial


# ──────────────────────────────────────────────────────────────────────
# Builder: sparse_mla_fwd_decode_combine (FlyDSL)
# ──────────────────────────────────────────────────────────────────────

def build_sparse_mla_fwd_decode_combine(
    *,
    num_heads: int,
    d_v: int = D_V_DEFAULT,
    topk: int,
    head_per_block: int = 4,
    block_I: int = BI,
    threads: int = THREADS_DEFAULT,
):
    """Build the FlyDSL sparse-MLA-decode combine kernel.

    Mirrors :func:`tilelang_kernel.sparse_mla_fwd_decode_combine`. Reduces
    ``NI = topk / block_I`` partial outputs to the final BF16 output using
    log-sum-exp normalization::

        lse_max[h]   = max_k partial_lse[k, h]
        lse_sum[h]   = sum_k exp(partial_lse[k, h] - lse_max[h])
        scale[k, h]  = exp(partial_lse[k, h] - lse_max[h] - log(lse_sum[h]))
        out[h, d]    = sum_k scale[k, h] * partial_o[k, h, d]

    Layout (matches TileLang):
        Partial_O   : [1, seq_len, NI, num_heads, d_v]    BF16
        Partial_Lse : [1, seq_len, NI, num_heads]         FP32
        Output      : [1, seq_len,    num_heads, d_v]     BF16
    """
    if not _FLYDSL_AVAILABLE:
        raise RuntimeError("FlyDSL is not installed; cannot build kernel")
    assert num_heads % head_per_block == 0, (
        f"head_per_block ({head_per_block}) must divide num_heads ({num_heads})"
    )

    NI = topk // block_I
    H_PER_B = head_per_block
    REPLICATE_H = num_heads // H_PER_B
    arch = get_rocm_arch()
    bf16_dtype = fx.BFloat16
    f32_dtype = fx.Float32

    assert NI <= WARP_SIZE, (
        f"this implementation requires NI ({NI}) <= WARP_SIZE ({WARP_SIZE})"
    )
    assert H_PER_B <= threads // WARP_SIZE, (
        f"head_per_block ({H_PER_B}) must be <= num_waves ({threads // WARP_SIZE})"
    )

    allocator = SmemAllocator(
        None,
        arch=arch,
        global_sym_name=f"sparse_mla_combine_smem_h{num_heads}_t{topk}",
    )
    LSE_BYTES = 4 * NI * H_PER_B
    SCALE_BYTES = 4 * NI * H_PER_B
    lse_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lse_offset + LSE_BYTES
    scale_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = scale_offset + SCALE_BYTES

    @flyc.kernel
    def combine_kernel(
        Partial_O: fx.Tensor,
        Partial_Lse: fx.Tensor,
        Output: fx.Tensor,
    ):
        # Constructed inside the kernel body so the MLIR context is active.
        v1f32_t = T.vec(1, f32_dtype.ir_type)

        bx = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        wave_id = tid // fx.Index(WARP_SIZE)
        lane = tid % fx.Index(WARP_SIZE)

        s_i = bx // fx.Index(REPLICATE_H) if REPLICATE_H > 1 else bx
        if const_expr(REPLICATE_H > 1):
            H0 = (bx % fx.Index(REPLICATE_H)) * fx.Index(H_PER_B)
        else:
            H0 = fx.Index(0)

        po_rsrc = buffer_ops.create_buffer_resource(Partial_O, max_size=True)
        plse_rsrc = buffer_ops.create_buffer_resource(Partial_Lse, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(Output, max_size=True)

        base = allocator.get_base()
        lse_smem = SmemPtr(
            base, lse_offset, f32_dtype.ir_type, shape=(NI * H_PER_B,)
        ).get()
        scale_smem = SmemPtr(
            base, scale_offset, f32_dtype.ir_type, shape=(NI * H_PER_B,)
        ).get()

        # ── (1) Load Partial_Lse[s_i, :, H0:H0+H_PER_B] into LDS ──────
        plse_seq_off = s_i * fx.Index(NI * num_heads)
        if tid < fx.Index(NI * H_PER_B):
            k = tid // fx.Index(H_PER_B)
            h = tid % fx.Index(H_PER_B)
            g_off = plse_seq_off + k * fx.Index(num_heads) + H0 + h
            # buffer_load with vec_width=1 returns scalar f32; wrap it into
            # a vector<1xf32> for the LDS vector.store.
            scalar = buffer_ops.buffer_load(
                plse_rsrc, g_off, vec_width=1, dtype=f32_dtype
            )
            v = vector.from_elements(v1f32_t, [scalar])
            vector.store(v, lse_smem, [tid])
        gpu.barrier()

        # ── (2) Per-head LSE max + sum reduction (1 wave per head) ────
        c_neg_big = fx.Float32(-(2.0 ** 30))
        c_zero_f = fx.Float32(0.0)

        for h_local in range_constexpr(H_PER_B):
            if wave_id == fx.Index(h_local):
                in_range = lane < fx.Index(NI)
                lane_safe = fx.Index(arith.select(
                    in_range, _to_raw(lane), _to_raw(fx.Index(0))
                ))
                off = lane_safe * fx.Index(H_PER_B) + fx.Index(h_local)
                lse_v_raw_v = vector.load_op(v1f32_t, lse_smem, [off])
                lse_v_raw = vector.extract(
                    lse_v_raw_v, static_position=[0], dynamic_position=[]
                )
                lse_v = arith.select(
                    in_range, lse_v_raw, _to_raw(c_neg_big)
                )

                # Max reduction across the wave.
                v = lse_v
                for sh in [32, 16, 8, 4, 2, 1]:
                    peer = ArithValue(v).shuffle_xor(
                        fx.Int32(sh), fx.Int32(WARP_SIZE)
                    )
                    v = arith.maximumf(v, peer)
                lse_max = v

                # Sum reduction of exp(lse - max).
                diff = arith.subf(lse_v_raw, lse_max)
                diff_log2e = arith.mulf(diff, _to_raw(fx.Float32(LOG2E)))
                exp_v = fmath.exp2(diff_log2e, fastmath="fast")
                exp_v = arith.select(in_range, exp_v, _to_raw(c_zero_f))
                v = exp_v
                for sh in [32, 16, 8, 4, 2, 1]:
                    peer = ArithValue(v).shuffle_xor(
                        fx.Int32(sh), fx.Int32(WARP_SIZE)
                    )
                    v = arith.addf(v, peer)
                lse_sum = v
                # log(x) -> log2(x) by 1/log(2). Use fmath.log2 directly.
                log2_lse_sum = fmath.log2(lse_sum, fastmath="fast")

                if in_range:
                    diff_log2e2 = arith.mulf(
                        arith.subf(lse_v_raw, lse_max),
                        _to_raw(fx.Float32(LOG2E)),
                    )
                    scale_arg = arith.subf(diff_log2e2, log2_lse_sum)
                    scale = fmath.exp2(scale_arg, fastmath="fast")
                    out_off = lane * fx.Index(H_PER_B) + fx.Index(h_local)
                    vec1 = vector.from_elements(v1f32_t, [scale])
                    vector.store(vec1, scale_smem, [out_off])

        gpu.barrier()

        # ── (3) acc_o[h, d] = sum_k scale[k, h] * Partial_O[k, h, d] ──
        # Distribute (H_PER_B, d_v) work across `threads`.
        po_seq_off = s_i * fx.Index(NI * num_heads * d_v)
        TOTAL = H_PER_B * d_v
        ELEMS_PER_THREAD = (TOTAL + threads - 1) // threads
        for ept in range_constexpr(ELEMS_PER_THREAD):
            lin = tid * fx.Index(ELEMS_PER_THREAD) + fx.Index(ept)
            in_range = lin < fx.Index(TOTAL)
            lin_safe = fx.Index(arith.select(
                in_range, _to_raw(lin), _to_raw(fx.Index(0))
            ))
            h = lin_safe // fx.Index(d_v)
            d = lin_safe % fx.Index(d_v)

            acc = _to_raw(fx.Float32(0.0))
            for k in range_constexpr(NI):
                scale_v = vector.load_op(
                    v1f32_t, scale_smem,
                    [fx.Index(k * H_PER_B) + h],
                )
                scale = vector.extract(
                    scale_v, static_position=[0], dynamic_position=[]
                )
                p_off = (
                    po_seq_off
                    + fx.Index(k * num_heads * d_v)
                    + (H0 + h) * fx.Index(d_v)
                    + d
                )
                p_bf16 = buffer_ops.buffer_load(
                    po_rsrc, p_off, vec_width=1, dtype=bf16_dtype
                )
                p_f32 = arith.extf(f32_dtype.ir_type, p_bf16)
                acc = arith.addf(acc, arith.mulf(scale, p_f32))

            if in_range:
                out_off = (
                    s_i * fx.Index(num_heads * d_v)
                    + (H0 + h) * fx.Index(d_v)
                    + d
                )
                acc_bf16 = arith.trunc_f(bf16_dtype.ir_type, acc)
                # buffer_store expects a vector or scalar; pass scalar directly.
                buffer_ops.buffer_store(acc_bf16, out_rsrc, out_off)

    @flyc.jit
    def launch_combine(
        Partial_O: fx.Tensor,
        Partial_Lse: fx.Tensor,
        Output: fx.Tensor,
        seq_len: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        grid_x = fx.Index(seq_len) * fx.Index(REPLICATE_H)
        combine_kernel(Partial_O, Partial_Lse, Output).launch(
            grid=(grid_x, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    return launch_combine


# ──────────────────────────────────────────────────────────────────────
# High-level dispatcher (matches signature of tilelang_sparse_fwd)
# ──────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=64)
def _cached_partial(num_heads, d_v, d_tail, topk, sm_scale, block_I, inner_iter, threads):
    return build_sparse_mla_fwd_decode_partial_fp8(
        num_heads=num_heads,
        d_v=d_v,
        d_tail=d_tail,
        topk=topk,
        sm_scale=sm_scale,
        block_I=block_I,
        inner_iter=inner_iter,
        threads=threads,
    )


@lru_cache(maxsize=64)
def _cached_combine(num_heads, d_v, topk_eff, head_per_block, block_I, threads):
    return build_sparse_mla_fwd_decode_combine(
        num_heads=num_heads,
        d_v=d_v,
        topk=topk_eff,
        head_per_block=head_per_block,
        block_I=block_I,
        threads=threads,
    )


def flydsl_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = D_V_DEFAULT,
) -> torch.Tensor:
    """FlyDSL sparse-MLA-decode forward pass (FP8 KV path).

    Drop-in replacement for :func:`tilelang_kernel.tilelang_sparse_fwd` for
    the FP8 KV path on ROCm. Uses the same ``_pick_inner_iter`` heuristic
    as the TileLang reference so the two kernels are directly comparable on
    the same workload.

    Args:
        q:       ``[seq_len, num_heads, d_v + d_tail]`` (FP8 or BF16)
        kv:      ``[num_pages, 1, d_v + d_tail]``      (FP8)
        indices: ``[seq_len, 1, topk]``                (int32)
        sm_scale: softmax scale
        d_v:     value head dim (must be 512)
    """
    assert q.dim() == 3 and kv.dim() == 3 and indices.dim() == 3, (
        "expected unbatched q/kv/indices like tilelang_sparse_fwd"
    )
    if not _is_hip:
        raise RuntimeError("FlyDSL sparse MLA is ROCm-only")

    num_heads = q.shape[1]
    dim = q.shape[2]
    tail_dim = dim - d_v
    topk = indices.shape[-1]

    is_fp8_kv = kv.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    if not is_fp8_kv:
        raise NotImplementedError(
            "FlyDSL sparse MLA currently supports FP8 KV only; "
            "for BF16 KV use the TileLang path"
        )

    if q.dtype != kv.dtype:
        q = q.to(kv.dtype)

    # Same heuristic as TileLang.
    if _is_gfx95_supported:
        block_I, threads, block_per_cu, cu = 64, 256, 2, 256
    else:
        block_I, threads, block_per_cu, cu = 64, 256, 1, 304
    ni = topk // block_I
    inner_iter = _pick_inner_iter(q.shape[0], ni, cu, block_per_cu)
    n_groups = ni // inner_iter

    partial_o = q.new_empty(
        (1, q.shape[0], n_groups, num_heads, d_v),
        dtype=torch.bfloat16,
    )
    partial_lse = q.new_empty(
        (1, q.shape[0], n_groups, num_heads),
        dtype=torch.float32,
    )

    launch_partial = _cached_partial(
        num_heads, d_v, tail_dim, topk, sm_scale, block_I, inner_iter, threads,
    )
    launch_partial(
        q.unsqueeze(0),
        kv.unsqueeze(0),
        indices.unsqueeze(0),
        partial_o,
        partial_lse,
        q.shape[0],
    )

    out = q.new_empty((1, q.shape[0], num_heads, d_v), dtype=torch.bfloat16)
    launch_combine = _cached_combine(
        num_heads, d_v, n_groups * block_I, 4, block_I, threads,
    )
    launch_combine(partial_o, partial_lse, out, q.shape[0])
    return out.squeeze(0)
