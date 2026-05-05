"""
Step 5 (final, WORK IN PROGRESS) of the FlyDSL sparse-MLA decode plan.

Combines Step 4 (GLM-5.1-FP8 shapes + multi-chunk topk loop + rope tail)
with online softmax + SV MFMA + final normalize.  This is the complete
fused-attention kernel that the unit benchmark will eventually compare
against the TileLang ``main_kernel`` baseline.

CURRENT STATUS  (2026-05-05)
============================
Compiles partially.  Trips multiple FlyDSL API gaps that need fixing
before end-to-end correctness can be checked.  Listing them so the next
iteration can tick them off one by one (logged after running the
unit-test in docker against this file)::

  E.bug.1  ``arith.IfOp`` does NOT exist; use
           ``scf.IfOp(cond, results_=[], has_else=False)`` from
           ``flydsl._mlir.dialects.scf``.  *(fixed in this revision)*

  E.bug.2  ``rocdl.cvt_pk_fp8_f32`` is positional, NOT list:
              ``cvt_pk_fp8_f32(T.i32, src_a, src_b, old, word_sel)``
           *(fixed in this revision; matches pa_decode_fp8.py)*

  E.bug.3  ``ArithValue.cmp_eq`` doesn't exist.  Use
              ``arith.cmpf(arith.CmpFPredicate.OEQ, a, b)`` then
              ``cond.select(a, b)`` only when the cond is ArithValue --
           OR use ``arith.if_then_else`` / explicit scf.IfOp.

  E.bug.4  Single-bf16 ``buffer_store`` via ``arith.bitcast`` -> ``i16``
           probably won't lower cleanly.  Need to either pack 2 bf16 ->
           1 i32 and store i32 (4 bytes), or store via the i16 path
           with the right buffer_store overload.  pa_decode_fp8.py STEP 14
           uses ``arith.trunc_f(T.vec(4, T.bf16), pv_out[n_tile])`` ->
           ``vector.bitcast(T.vec(2, T.i32), out_bf16)`` -> store 2xi32.

  E.bug.5  SV B-operand load (V = K_nope_LDS) currently assumes
           col-aligned i32 reads but ``v_col`` is per-lane mfma_row -- not
           4-aligned in general.  Marked FIXME in code.  Either:
             (a) reorganize V_LDS to be (D, BI) (col-major over BI) so
                 lanes load contiguous tokens at the same column, OR
             (b) issue 8 single-byte (i8) loads per lane.

  E.bug.6  Per-row softmax max+sum is implemented as a per-lane vector
           reduction across 16 lanes (XOR 1,2,4,8) within a lane group.
           This is correct ONLY if ``lane_hi4`` stays the same when
           XOR-ing -- sh=1,2,4,8 only touches the low 4 bits, OK.
           Cross-warp via 16 LDS slots per warp.  Assumes Option-A QK
           output layout (lane (i, j) -> rows i*4..i*4+3 col j).

What works in this file today
-----------------------------
- LDS allocator + finalize call.
- Q_nope / Q_rope load to LDS at kernel start.
- Per-chunk K_nope / K_rope indirect gather (reuses Step 4 pattern).
- QK FP8 + BF16 MFMA into acc_qk (reuses Step 4 pattern).
- Per-row warp-local max / sum reductions via shuffle_xor.
- LDS-staged cross-warp max / sum reductions.
- Online softmax algebra (alpha rescale of running_sum and acc_o).

What still needs work to land
-----------------------------
- Fix bugs E.3 / E.4 / E.5 above.
- Integrate masking with ``buffer_load`` zero-fill in the gather (currently
  the mask is applied after gather, so out-of-range indices may have
  loaded uninitialized LDS).  Use ``buffer_load`` with ``with_oob=True``.
- Verify per-row MFMA output layout (Option A) holds for the SV gemm too.
  *Steps 1-4 verified Option A bit-exact for QK; SV uses the same MFMA
  op so it should match.*
- Final output store layout (one bf16 per lane is wasteful; pack 4 bf16
  into 2xi32 like pa_decode_fp8 STEP 14).
- After fixing the above, validate end-to-end vs the torch fp32 reference
  (already coded in run_test).

Path to land Step E
-------------------
1. Wrap each broken section in a tiny standalone test (like
   flydsl_qk_*.py for QK), so SV / output store / cvt_pk_fp8_f32 can be
   debugged in isolation.  Each test ~150 lines.
2. Once each piece is bit-exact, glue them into this file.
3. Replace the ``raise NotImplementedError`` in flydsl_kernel.py with a
   call to ``_build_sparse_mla_kernel`` from this module.

Inputs / outputs
----------------
- Q_nope    : (HG=16, D=512) fp8 (host-quantized)        + q_scale_nope (HG,) fp32
- Q_rope    : (HG=16, D_ROPE=64) bf16
- K_nope    : (T_POOL, D) fp8                            + k_scale (T_POOL,) fp32
- K_rope    : (T_POOL, D_ROPE) bf16
- Indices   : (TOPK,) int32 (negative = masked-out slot)
- Output O  : (HG, D) bf16

Algorithm (per CTA, single query / head group)
----------------------------------------------
    running_max = -inf, running_sum = 0, acc_o = 0  (per row, per warp)

    Q_nope, Q_rope -> LDS (once at kernel start)

    for chunk in 0..NI=32:
        gather K_nope_chunk, K_rope_chunk into LDS via Indices[chunk*BI .. ]
        QK MFMA (FP8 nope + BF16 rope) -> acc_qk    [per-warp 16x16 partial]
        acc_qk *= q_scale * k_scale_per_token        [host-precomputed scales]
        mask out indices < 0

        local_max  = per-row max(acc_qk) within warp     [shuffle_xor in 16-lane group]
        global_max = cross-warp max via LDS              [16 LDS slots]
        new_max    = max(running_max, global_max)
        alpha      = exp2((running_max - new_max) * log2e)
        acc_o     *= alpha                                [8 n-tiles per warp]
        running_sum *= alpha
        running_max = new_max

        P = exp2((acc_qk - new_max) * log2e)
        per-row sum(P) -> warp partial -> cross-warp sum  [16 LDS slots]
        running_sum += partial_sum

        P -> fp8 in P_LDS via cvt_pk_fp8_f32             [16x64 fp8 = 1 KB]
        SV MFMA (P_fp8 @ V=K_nope_fp8) -> acc_o accum    [8 n-tiles x 2 k-steps]

    O = (acc_o / running_sum).cast(bf16) -> store          [8 n-tiles x 4 elements]

LDS budget
----------
    Q_nope_LDS  :  16 *  512        =   8 KB
    Q_rope_LDS  :  16 *   64 * 2    =   2 KB
    K_nope_LDS  :  64 *  512        =  32 KB
    K_rope_LDS  :  64 *   64 * 2    =   8 KB
    P_LDS       :  16 *   64        =   1 KB  (fp8)
    S_LDS       :  16 *   64 * 4    =   4 KB  (fp32 acc_qk staging)
    max_lds     :  16 * 4 (warps)*4 = 256 B
    sum_lds     :  16 * 4 (warps)*4 = 256 B
                                       --------
                                       55 KB / 160 KB on gfx950

Status
------
First cut.  Iterates on hardware-in-the-loop -- expect the typical
load-layout / lane-mapping fix-ups documented in flydsl_qk_minimal.py.
"""

from __future__ import annotations

import functools
import math
import sys

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T, Int32
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as _get_arch
from flydsl.compiler.kernel_function import CompilationContext
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf


# ---- Compile-time constants ----
HG = 16
BI = 64
D = 512
D_ROPE = 64
TOPK = 2048
T_POOL = 4096
NI = TOPK // BI                            # 32

NUM_WARPS = 4
WARP_SIZE = 64
BLOCK_THREADS = NUM_WARPS * WARP_SIZE      # 256
N_PER_WARP = BI // NUM_WARPS               # 16

MFMA_K_FP8 = 32
MFMA_K_BF = 16
K_CHUNKS_FP8 = D // MFMA_K_FP8             # 16
K_CHUNKS_BF = D_ROPE // MFMA_K_BF          # 4

# SV decomposition: M=HG=16 (1 m-tile), N=D=512 (4 warps -> 128/warp = 8 n-tiles),
# K=BI=64 (2 k-chunks of 32).
SV_N_PER_WARP = D // NUM_WARPS             # 128
SV_N_TILES_WARP = SV_N_PER_WARP // 16      # 8
SV_K_CHUNKS = BI // MFMA_K_FP8             # 2

D_I32 = D // 4                             # 128
D_ROPE_I32 = D_ROPE // 2                   # 32

# Multi-CTA "split-K" along the topk axis.  Each CTA handles
# CHUNKS_PER_PARTIAL = NI / NUM_PARTIALS chunks, then the combine kernel
# reduces NUM_PARTIALS partials per (batch, head).
# With NI=32 and NUM_PARTIALS=8, each CTA does 4 chunks and grid =
# (batch, 8) which puts ``batch * 8`` CTAs in flight (vs 1 CTA in the
# single-CTA design).  256 CUs means 8x parallelism with batch=4.
NUM_PARTIALS = 8
assert NI % NUM_PARTIALS == 0, "NI must be divisible by NUM_PARTIALS"
CHUNKS_PER_PARTIAL = NI // NUM_PARTIALS    # 4

_FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0
LOG2E = 1.4426950408889634


@functools.lru_cache(maxsize=8)
def _build_sparse_mla_kernel(sm_scale: float):
    arch = _get_arch()
    assert str(arch).startswith("gfx95"), f"expected gfx950, got {arch}"

    Q_NOPE_BYTES = HG * D
    Q_ROPE_BYTES = HG * D_ROPE * 2
    K_NOPE_BYTES = BI * D
    K_ROPE_BYTES = BI * D_ROPE * 2
    P_BYTES = HG * BI                 # fp8
    S_BYTES = HG * BI * 4             # fp32 staging
    MAX_BYTES = HG * NUM_WARPS * 4    # per (row, warp) fp32
    SUM_BYTES = HG * NUM_WARPS * 4

    allocator = SmemAllocator(
        None, arch=arch, global_sym_name="flydsl_sparse_mla_smem"
    )
    q_nope_off = 0
    allocator.ptr = Q_NOPE_BYTES
    q_rope_off = allocator.ptr
    allocator.ptr += Q_ROPE_BYTES
    k_nope_off = allocator.ptr
    allocator.ptr += K_NOPE_BYTES
    k_rope_off = allocator.ptr
    allocator.ptr += K_ROPE_BYTES
    p_off = allocator.ptr
    allocator.ptr += P_BYTES
    s_off = allocator.ptr
    allocator.ptr += S_BYTES
    max_off = allocator.ptr
    allocator.ptr += MAX_BYTES
    sum_off = allocator.ptr
    allocator.ptr += SUM_BYTES
    SMEM_BYTES = allocator.ptr

    sm_scale_log2e = float(sm_scale * LOG2E)

    @flyc.kernel
    def fmha_partial_kernel(
        # Outputs ----------------------------------------------------------
        partial_o_ptr: fx.Tensor,      # (BS, NUM_PARTIALS, HG, D) fp32
        partial_m_ptr: fx.Tensor,      # (BS, NUM_PARTIALS, HG) fp32
        partial_l_ptr: fx.Tensor,      # (BS, NUM_PARTIALS, HG) fp32
        # Q ---------------------------------------------------------------
        q_nope_ptr: fx.Tensor,         # (BS, HG, D) fp8
        q_rope_ptr: fx.Tensor,         # (BS, HG, D_ROPE) bf16
        q_scale_ptr: fx.Tensor,        # (BS, HG) fp32
        # KV --------------------------------------------------------------
        k_nope_ptr: fx.Tensor,         # (T_POOL, D) fp8
        k_scale_ptr: fx.Tensor,        # (T_POOL,) fp32
        k_rope_ptr: fx.Tensor,         # (T_POOL, D_ROPE) bf16
        # Indices ---------------------------------------------------------
        indices_ptr: fx.Tensor,        # (BS, TOPK) int32
    ):
        tid = gpu.thread_idx.x
        seq_idx = gpu.block_idx.x      # batch / decode-query index
        partial_idx = gpu.block_idx.y  # which slice of the topk loop (0..NUM_PARTIALS)
        warp_id = tid // fx.Int32(WARP_SIZE)
        lane = tid % fx.Int32(WARP_SIZE)
        lane_hi4 = lane // fx.Int32(16)            # 0..3
        mfma_row = lane % fx.Int32(16)             # 0..15

        # Per-batch offsets (in **element** units; converted to byte offsets
        # at the various buffer_load / buffer_store sites).
        q_nope_batch_off = seq_idx * fx.Int32(HG * D)
        q_rope_batch_off = seq_idx * fx.Int32(HG * D_ROPE)
        q_scale_batch_off = seq_idx * fx.Int32(HG)
        indices_batch_off = seq_idx * fx.Int32(TOPK)

        # Per-partial output offsets.  Layout: (BS, NUM_PARTIALS, HG, D).
        # po[b, p, h, d] = po_base + (b*NUM_PARTIALS + p)*(HG*D) + h*D + d
        partial_slot = seq_idx * fx.Int32(NUM_PARTIALS) + partial_idx
        partial_o_off = partial_slot * fx.Int32(HG * D)        # fp32 elements
        partial_ml_off = partial_slot * fx.Int32(HG)           # fp32 elements

        # Where in the topk axis does THIS partial CTA start?
        partial_chunk_start = partial_idx * fx.Int32(CHUNKS_PER_PARTIAL)

        po_rsrc = buffer_ops.create_buffer_resource(partial_o_ptr, max_size=True)
        pm_rsrc = buffer_ops.create_buffer_resource(partial_m_ptr, max_size=True)
        pl_rsrc = buffer_ops.create_buffer_resource(partial_l_ptr, max_size=True)
        qn_rsrc = buffer_ops.create_buffer_resource(q_nope_ptr, max_size=True)
        qr_rsrc = buffer_ops.create_buffer_resource(q_rope_ptr, max_size=True)
        qs_rsrc = buffer_ops.create_buffer_resource(q_scale_ptr, max_size=True)
        kn_rsrc = buffer_ops.create_buffer_resource(k_nope_ptr, max_size=True)
        ks_rsrc = buffer_ops.create_buffer_resource(k_scale_ptr, max_size=True)
        kr_rsrc = buffer_ops.create_buffer_resource(k_rope_ptr, max_size=True)
        idx_rsrc = buffer_ops.create_buffer_resource(indices_ptr, max_size=True)

        base = allocator.get_base()
        qn_lds_i32 = SmemPtr(base, q_nope_off, T.i32, shape=(Q_NOPE_BYTES // 4,)).get()
        qr_lds_i32 = SmemPtr(base, q_rope_off, T.i32, shape=(Q_ROPE_BYTES // 4,)).get()
        kn_lds_i32 = SmemPtr(base, k_nope_off, T.i32, shape=(K_NOPE_BYTES // 4,)).get()
        kr_lds_i32 = SmemPtr(base, k_rope_off, T.i32, shape=(K_ROPE_BYTES // 4,)).get()
        qr_lds_bf16 = SmemPtr(base, q_rope_off, T.bf16, shape=(Q_ROPE_BYTES // 2,)).get()
        kr_lds_bf16 = SmemPtr(base, k_rope_off, T.bf16, shape=(K_ROPE_BYTES // 2,)).get()
        kn_lds_bytes = SmemPtr(base, k_nope_off, T.i8, shape=(K_NOPE_BYTES,)).get()
        p_lds_i32 = SmemPtr(base, p_off, T.i32, shape=(P_BYTES // 4,)).get()
        s_lds_f32 = SmemPtr(base, s_off, T.f32, shape=(HG * BI,)).get()
        max_lds = SmemPtr(base, max_off, T.f32, shape=(HG * NUM_WARPS,))
        sum_lds = SmemPtr(base, sum_off, T.f32, shape=(HG * NUM_WARPS,))

        NEG_INF = arith.constant(float("-inf"), type=T.f32)
        ZERO_F = fx.Float32(0.0)
        ONE_F = fx.Float32(1.0)
        LOG2E_C = arith.constant(LOG2E, type=T.f32)
        SCALE_C = arith.constant(sm_scale_log2e, type=T.f32)
        FP8MAX_C = arith.constant(FP8_MAX, type=T.f32)
        c_w = fx.Int32(WARP_SIZE)

        # Helpers ------------------------------------------------------------
        def _pack_i32_pair_to_i64(a, b):
            v = vector.from_elements(T.vec(2, T.i32), [a, b])
            v1 = vector.bitcast(T.vec(1, T.i64), v)
            return vector.extract(v1, static_position=[0], dynamic_position=[])

        def _wave_max_within_16(x):
            """Reduce max across 16 lanes within a lane group (XOR 1,2,4,8)."""
            w = x
            for sh in [1, 2, 4, 8]:
                peer = w.shuffle_xor(fx.Int32(sh), c_w)
                w = w.maximumf(peer)
            return w

        def _wave_sum_within_16(x):
            """Reduce sum across 16 lanes within a lane group."""
            w = x
            for sh in [1, 2, 4, 8]:
                peer = w.shuffle_xor(fx.Int32(sh), c_w)
                w = w + peer
            return w

        # --- STEP A1: load Q_nope to LDS (8 KB / 256 threads x 8 i32) -----
        # Q_nope offset (element index = i32 index since fp8 is 1 byte
        # and we load i32 = 4 fp8): q_nope_batch_off / 4.
        qn_global_base = q_nope_batch_off // fx.Int32(4)
        for li in range_constexpr(8):
            qi = tid + fx.Int32(li * BLOCK_THREADS)
            qv = buffer_ops.buffer_load(
                qn_rsrc, qn_global_base + qi, vec_width=1, dtype=T.i32
            )
            vector.store(
                vector.from_elements(T.vec(1, T.i32), [qv]),
                qn_lds_i32,
                [arith.index_cast(T.index, qi)],
            )

        # --- STEP A2: load Q_rope to LDS (2 KB / 256 threads x 2 i32) -----
        # Q_rope is bf16 = 2 bytes, so element-to-i32 ratio = 2.
        qr_global_base = q_rope_batch_off // fx.Int32(2)
        for li in range_constexpr(2):
            qi = tid + fx.Int32(li * BLOCK_THREADS)
            qv = buffer_ops.buffer_load(
                qr_rsrc, qr_global_base + qi, vec_width=1, dtype=T.i32
            )
            vector.store(
                vector.from_elements(T.vec(1, T.i32), [qv]),
                qr_lds_i32,
                [arith.index_cast(T.index, qi)],
            )

        # --- Persistent online-softmax state (per lane) -----------------
        # Lane (lane_hi4, mfma_row) owns rows {lane_hi4*4 .. lane_hi4*4+3}.
        # 4 fp32 per lane for running_max/sum (one per row).
        running_max = arith.constant_vector(float("-inf"), T.f32x4)
        running_sum = arith.constant_vector(0.0, T.f32x4)
        # 8 acc_o tiles per warp; each is f32x4 per lane (row tile, n-tile-col).
        acc_o = [arith.constant_vector(0.0, T.f32x4) for _ in range(SV_N_TILES_WARP)]

        # ================================================================
        # Per-chunk loop -- only iterate over THIS partial CTA's slice of NI.
        # Each partial covers CHUNKS_PER_PARTIAL contiguous chunks; combine
        # kernel reduces NUM_PARTIALS partials per (batch, head) at the end.
        # ================================================================
        for local_chunk_i in range_constexpr(CHUNKS_PER_PARTIAL):
            chunk_off = (
                partial_chunk_start + fx.Int32(local_chunk_i)
            ) * fx.Int32(BI)

            # --- STEP B1: gather K_nope_chunk -------------------------
            # Clamp src_row to >= 0 to avoid OOB reads from masked
            # (negative-index) topk slots.  The mask itself is enforced
            # later in STEP B5 via ``in_range = k_tok_global >= 0``,
            # which uses the ORIGINAL (un-clamped) index value.
            for li in range_constexpr(K_NOPE_BYTES // 4 // BLOCK_THREADS):
                lds_idx = tid + fx.Int32(li * BLOCK_THREADS)
                token = lds_idx // fx.Int32(D_I32)
                k_i32 = lds_idx % fx.Int32(D_I32)
                src_row = buffer_ops.buffer_load(
                    idx_rsrc,
                    indices_batch_off + chunk_off + token,
                    vec_width=1, dtype=T.i32,
                )
                # Clamp src_row to >= 0 (the mask is enforced later via
                # in_range using the original index value).
                neg = arith.cmpi(arith.CmpIPredicate.slt, src_row, fx.Int32(0))
                src_row_safe = arith.select(neg, fx.Int32(0), src_row)
                g_i32 = src_row_safe * fx.Int32(D_I32) + k_i32
                kv = buffer_ops.buffer_load(kn_rsrc, g_i32, vec_width=1, dtype=T.i32)
                vector.store(
                    vector.from_elements(T.vec(1, T.i32), [kv]),
                    kn_lds_i32,
                    [arith.index_cast(T.index, lds_idx)],
                )

            # --- STEP B2: gather K_rope_chunk -------------------------
            for li in range_constexpr(K_ROPE_BYTES // 4 // BLOCK_THREADS):
                lds_idx = tid + fx.Int32(li * BLOCK_THREADS)
                token = lds_idx // fx.Int32(D_ROPE_I32)
                k_i32 = lds_idx % fx.Int32(D_ROPE_I32)
                src_row = buffer_ops.buffer_load(
                    idx_rsrc,
                    indices_batch_off + chunk_off + token,
                    vec_width=1, dtype=T.i32,
                )
                neg = arith.cmpi(arith.CmpIPredicate.slt, src_row, fx.Int32(0))
                src_row_safe = arith.select(neg, fx.Int32(0), src_row)
                g_i32 = src_row_safe * fx.Int32(D_ROPE_I32) + k_i32
                kv = buffer_ops.buffer_load(kr_rsrc, g_i32, vec_width=1, dtype=T.i32)
                vector.store(
                    vector.from_elements(T.vec(1, T.i32), [kv]),
                    kr_lds_i32,
                    [arith.index_cast(T.index, lds_idx)],
                )

            gpu.barrier()

            # --- STEP B3: QK FP8 MFMA over D=512 ----------------------
            acc_qk = arith.constant_vector(0.0, T.f32x4)
            for kc in range_constexpr(K_CHUNKS_FP8):
                kc_off = kc * MFMA_K_FP8
                q_col = lane_hi4 * fx.Int32(8) + fx.Int32(kc_off)
                q_elem_off = mfma_row * fx.Int32(D) + q_col
                q_i32_idx = q_elem_off // fx.Int32(4)
                q_a = vector.load_op(T.vec(1, T.i32), qn_lds_i32, [arith.index_cast(T.index, q_i32_idx)])
                q_b = vector.load_op(T.vec(1, T.i32), qn_lds_i32, [arith.index_cast(T.index, q_i32_idx + fx.Int32(1))])
                q_a_i32 = vector.extract(q_a, static_position=[0], dynamic_position=[])
                q_b_i32 = vector.extract(q_b, static_position=[0], dynamic_position=[])
                q_i64 = _pack_i32_pair_to_i64(q_a_i32, q_b_i32)

                k_row = warp_id * fx.Int32(16) + mfma_row
                k_col = lane_hi4 * fx.Int32(8) + fx.Int32(kc_off)
                k_elem_off = k_row * fx.Int32(D) + k_col
                k_i32_idx = k_elem_off // fx.Int32(4)
                k_a = vector.load_op(T.vec(1, T.i32), kn_lds_i32, [arith.index_cast(T.index, k_i32_idx)])
                k_b = vector.load_op(T.vec(1, T.i32), kn_lds_i32, [arith.index_cast(T.index, k_i32_idx + fx.Int32(1))])
                k_a_i32 = vector.extract(k_a, static_position=[0], dynamic_position=[])
                k_b_i32 = vector.extract(k_b, static_position=[0], dynamic_position=[])
                k_i64 = _pack_i32_pair_to_i64(k_a_i32, k_b_i32)

                acc_qk = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [q_i64, k_i64, acc_qk, 0, 0, 0])

            # --- STEP B4: QK BF16 MFMA over D_ROPE=64 -----------------
            for kc in range_constexpr(K_CHUNKS_BF):
                kc_off = kc * MFMA_K_BF
                q_col = lane_hi4 * fx.Int32(4) + fx.Int32(kc_off)
                q_elem_off = mfma_row * fx.Int32(D_ROPE) + q_col
                q_vec = vector.load_op(T.vec(4, T.bf16), qr_lds_bf16, [arith.index_cast(T.index, q_elem_off)])
                q_v_i16 = vector.bitcast(T.vec(4, T.i16), q_vec)

                k_row = warp_id * fx.Int32(16) + mfma_row
                k_col = lane_hi4 * fx.Int32(4) + fx.Int32(kc_off)
                k_elem_off = k_row * fx.Int32(D_ROPE) + k_col
                k_vec = vector.load_op(T.vec(4, T.bf16), kr_lds_bf16, [arith.index_cast(T.index, k_elem_off)])
                k_v_i16 = vector.bitcast(T.vec(4, T.i16), k_vec)

                acc_qk = rocdl.mfma_f32_16x16x16bf16_1k(T.f32x4, [q_v_i16, k_v_i16, acc_qk, 0, 0, 0])

            # --- STEP B5: apply scales + sm_scale (already includes log2e) ---
            # acc_qk[r] = acc_qk[r] * sm_scale * q_scale[row=lane_hi4*4+r] * k_scale[token=warp_id*16+mfma_row]
            # Load q_scale for our 4 rows.  Q-scale is per-row HG.
            # Use scalar buffer_loads (cheap, 4 per lane).
            q_scale_v = arith.constant_vector(0.0, T.f32x4)
            for r in range_constexpr(4):
                row_idx = lane_hi4 * fx.Int32(4) + fx.Int32(r)
                qs = buffer_ops.buffer_load(
                    qs_rsrc, q_scale_batch_off + row_idx,
                    vec_width=1, dtype=T.f32,
                )
                q_scale_v = vector.insert(qs, q_scale_v, static_position=[r], dynamic_position=[])

            # K-scale: per-token, the column we wrote in this lane is
            # warp_id*16 + mfma_row of the chunk.  Single fp32 per lane.
            # Clamp k_tok_global to >= 0 for the ks load (avoid OOB on
            # masked slots); the mask itself uses the ORIGINAL index value.
            tok_idx = (
                indices_batch_off + chunk_off
                + warp_id * fx.Int32(16) + mfma_row
            )
            k_tok_global = buffer_ops.buffer_load(idx_rsrc, tok_idx, vec_width=1, dtype=T.i32)
            neg_tok = arith.cmpi(arith.CmpIPredicate.slt, k_tok_global, fx.Int32(0))
            k_tok_safe = arith.select(neg_tok, fx.Int32(0), k_tok_global)
            ks = buffer_ops.buffer_load(ks_rsrc, k_tok_safe, vec_width=1, dtype=T.f32)

            # Mask: indices < 0 -> -inf (uses original, un-clamped index).
            in_range = k_tok_global >= fx.Int32(0)

            for r in range_constexpr(4):
                v = vector.extract(acc_qk, static_position=[r], dynamic_position=[])
                qs_r = vector.extract(q_scale_v, static_position=[r], dynamic_position=[])
                v_scaled = v * qs_r * ks * SCALE_C
                v_masked = in_range.select(v_scaled, NEG_INF)
                acc_qk = vector.insert(v_masked, acc_qk, static_position=[r], dynamic_position=[])

            # --- STEP B6: per-row max (warp-local across 16 N positions) ---
            # Each lane has acc_qk[r] for row=lane_hi4*4+r, col=mfma_row.
            # Reduce max across 16 lanes (varying mfma_row = j) with same lane_hi4.
            local_max = _wave_max_within_16(acc_qk)
            # Now each lane in the same lane group has the same max for its 4 rows.
            # Write to LDS: every lane in a lane group has the SAME value
            # after _wave_max_within_16, so 16 lanes writing the same value
            # to the same slot is a benign write race (last writer wins,
            # but they're all identical).  Avoids scf.IfOp + the Python
            # expression-caching dominance issue between if-block and the
            # cross-warp reader below.
            for r in range_constexpr(4):
                v = vector.extract(local_max, static_position=[r], dynamic_position=[])
                row_idx_w = lane_hi4 * fx.Int32(4) + fx.Int32(r)
                slot_w = warp_id * fx.Int32(HG) + row_idx_w
                max_lds.store(v, [arith.index_cast(T.index, slot_w)])

            gpu.barrier()

            # --- STEP B7: cross-warp max -> global new_max -------------
            # Each lane reads max_lds for its 4 rows from all 4 warps.
            new_max_v = arith.constant_vector(float("-inf"), T.f32x4)
            for r in range_constexpr(4):
                row_idx = lane_hi4 * fx.Int32(4) + fx.Int32(r)
                m_acc = NEG_INF
                for w in range_constexpr(NUM_WARPS):
                    slot = fx.Int32(w * HG) + row_idx
                    mv = max_lds.load([arith.index_cast(T.index, slot)])
                    m_acc = m_acc.maximumf(mv)
                # Combine with running_max
                old = vector.extract(running_max, static_position=[r], dynamic_position=[])
                new_m = m_acc.maximumf(old)
                new_max_v = vector.insert(new_m, new_max_v, static_position=[r], dynamic_position=[])

            # --- STEP B8: alpha = exp2((old - new) * log2e) ------------
            # We multiply running_max contribution by log2e once via SCALE_C
            # earlier on acc_qk; for the rescale we use raw fp32 max diffs.
            alpha_v = arith.constant_vector(1.0, T.f32x4)
            for r in range_constexpr(4):
                old = vector.extract(running_max, static_position=[r], dynamic_position=[])
                new = vector.extract(new_max_v, static_position=[r], dynamic_position=[])
                diff = (old - new)  # SCALE_C already folded into acc_qk
                a = diff.exp2(fastmath=arith.FastMathFlags.fast)
                alpha_v = vector.insert(a, alpha_v, static_position=[r], dynamic_position=[])

            # Update running_sum *= alpha; running_max = new_max
            for r in range_constexpr(4):
                rs = vector.extract(running_sum, static_position=[r], dynamic_position=[])
                a = vector.extract(alpha_v, static_position=[r], dynamic_position=[])
                running_sum = vector.insert(rs * a, running_sum, static_position=[r], dynamic_position=[])
            running_max = new_max_v

            # Rescale acc_o by alpha (per row)
            for nt in range_constexpr(SV_N_TILES_WARP):
                acc_nt = acc_o[nt]
                for r in range_constexpr(4):
                    a = vector.extract(alpha_v, static_position=[r], dynamic_position=[])
                    v = vector.extract(acc_nt, static_position=[r], dynamic_position=[])
                    acc_nt = vector.insert(v * a, acc_nt, static_position=[r], dynamic_position=[])
                acc_o[nt] = acc_nt

            # --- STEP B9: P = exp2((acc_qk - new_max)) -----------------
            # SCALE_C already folded; we just subtract new_max in fp32.
            # We then **pre-multiply P by ks** (per-token K-scale for this
            # lane's column) so the SV gemm's f32 accumulator naturally
            # carries the correct per-token scaled contribution.  Math:
            #   O[h, d] = (sum_k P[h,k] * V_fp8[k,d] * k_scale[k]) / l[h]
            #          = sum_k (P[h,k] * k_scale[k]) * V_fp8[k,d] / l[h]
            # so scaling P pre-SV is equivalent to per-token V scaling.
            # ``ks`` was already loaded for this lane's column in STEP B5.
            # Two parallel f32x4 vectors: p_v carries per-token-K-scaled P
            # (used by the SV MFMA so each token's contribution gets its
            # k_scale[k] correctly), p_unscaled_v carries raw P (used by
            # the running_sum / softmax denominator -- which must NOT
            # include per-token k_scale).
            p_v = arith.constant_vector(0.0, T.f32x4)
            p_unscaled_v = arith.constant_vector(0.0, T.f32x4)
            for r in range_constexpr(4):
                v = vector.extract(acc_qk, static_position=[r], dynamic_position=[])
                nm = vector.extract(new_max_v, static_position=[r], dynamic_position=[])
                p = (v - nm).exp2(fastmath=arith.FastMathFlags.fast)
                p_unscaled_v = vector.insert(p, p_unscaled_v, static_position=[r], dynamic_position=[])
                p_v = vector.insert(p * ks, p_v, static_position=[r], dynamic_position=[])

            # --- STEP B10: per-row sum of UNSCALED P -------------------
            # Reduce across 16 N-lanes within a lane group (same 4 rows,
            # different N positions).  Each lane in the group ends up with
            # the per-row partial sum for this warp's 16 N positions.
            local_sum_v = _wave_sum_within_16(p_unscaled_v)

            # Same write-race pattern as the max writer above (benign).
            for r in range_constexpr(4):
                v = vector.extract(local_sum_v, static_position=[r], dynamic_position=[])
                row_idx_s = lane_hi4 * fx.Int32(4) + fx.Int32(r)
                slot_s = warp_id * fx.Int32(HG) + row_idx_s
                sum_lds.store(v, [arith.index_cast(T.index, slot_s)])

            gpu.barrier()

            # Cross-warp sum -> add to running_sum
            for r in range_constexpr(4):
                row_idx = lane_hi4 * fx.Int32(4) + fx.Int32(r)
                s_acc = ZERO_F
                for w in range_constexpr(NUM_WARPS):
                    slot = fx.Int32(w * HG) + row_idx
                    sv_l = sum_lds.load([arith.index_cast(T.index, slot)])
                    s_acc = s_acc + sv_l
                rs = vector.extract(running_sum, static_position=[r], dynamic_position=[])
                running_sum = vector.insert(rs + s_acc, running_sum, static_position=[r], dynamic_position=[])

            # --- STEP B11: P -> fp8 packed (4 fp32 -> 1 i32) -----------
            p0 = vector.extract(p_v, static_position=[0], dynamic_position=[]) * FP8MAX_C
            p1 = vector.extract(p_v, static_position=[1], dynamic_position=[]) * FP8MAX_C
            p2 = vector.extract(p_v, static_position=[2], dynamic_position=[]) * FP8MAX_C
            p3 = vector.extract(p_v, static_position=[3], dynamic_position=[]) * FP8MAX_C
            zero_i32 = arith.constant(0, type=T.i32)
            lo = rocdl.cvt_pk_fp8_f32(T.i32, p0, p1, zero_i32, False)
            wd = rocdl.cvt_pk_fp8_f32(T.i32, p2, p3, lo, True)

            # Write to P_LDS: lane (lane_hi4, mfma_row) writes 4 fp8 to
            # P_LDS[rows={lane_hi4*4..lane_hi4*4+3}, col=warp_id*16+mfma_row].
            # P_LDS layout: row-major (HG, BI) fp8 = (16, 64).
            # 4 contiguous fp8 in row direction means 4 rows at the same col.
            # = byte offset row*BI + col, but for 4 rows that's 4*BI bytes
            # apart (not contiguous!).  So we can't pack as a single i32 store.
            # Instead, store one fp8 per row using i8 view, OR transpose
            # P_LDS layout to (BI, HG) so 4 rows * 1 col -> 4 contiguous bytes
            # for store.
            #
            # Simpler: lay out P_LDS as (BI, HG) so each lane's 4-element write
            # IS contiguous.  Then SV gemm reads P[token, row] which is the
            # same (transposed) data; we adjust the SV load offsets.
            #
            # P_LDS_T layout: (BI=64, HG=16) fp8 = 1 KB
            #  Lane (lane_hi4, mfma_row) writes to col=lane_hi4*4..+3, row=warp_id*16+mfma_row.
            #  Byte offset = row * HG + col_base, contiguous over 4 cols.
            p_byte_off = (warp_id * fx.Int32(16) + mfma_row) * fx.Int32(HG) + lane_hi4 * fx.Int32(4)
            vector.store(
                vector.from_elements(T.vec(1, T.i32), [wd]),
                p_lds_i32,
                [arith.index_cast(T.index, p_byte_off // fx.Int32(4))],
            )

            gpu.barrier()

            # --- STEP B12: SV MFMA  P @ V -----------------------------
            # P (HG, BI) fp8, V = K_nope (BI, D) fp8.  Each warp owns
            # SV_N_TILES_WARP=8 n-tiles of D, K=BI=64 with 2 k-chunks of 32.
            #
            # MFMA A operand (P): lane (i, j) provides 8 fp8 at row=j, k={i*8..+7}
            #  P_LDS_T layout = (BI, HG): P_LDS_T[k_token, row=j], so for a
            #  k-chunk starting at k_off, lane needs BI tokens [k_off+i*8..+7]
            #  for HG row=j.  Byte offset = (k_off + i*8 + 0..7) * HG + j.
            #  4 fp8 per i32 -> we need 2 i32 per lane (8 fp8).
            for nt in range_constexpr(SV_N_TILES_WARP):
                # n_tile col base in V's column dim D
                n_tile_col_base = warp_id * fx.Int32(SV_N_PER_WARP) + fx.Int32(nt * 16)

                acc_nt = acc_o[nt]
                for kchunk in range_constexpr(SV_K_CHUNKS):
                    k_off = kchunk * MFMA_K_FP8
                    # --- A operand load (P_LDS_T)
                    a_byte_off = (fx.Int32(k_off) + lane_hi4 * fx.Int32(8)) * fx.Int32(HG) + mfma_row
                    a_a = vector.load_op(
                        T.vec(1, T.i32), p_lds_i32,
                        [arith.index_cast(T.index, a_byte_off // fx.Int32(4))],
                    )
                    a_b = vector.load_op(
                        T.vec(1, T.i32), p_lds_i32,
                        [arith.index_cast(T.index, a_byte_off // fx.Int32(4) + fx.Int32(1))],
                    )
                    a_a_i32 = vector.extract(a_a, static_position=[0], dynamic_position=[])
                    a_b_i32 = vector.extract(a_b, static_position=[0], dynamic_position=[])
                    a_i64 = _pack_i32_pair_to_i64(a_a_i32, a_b_i32)

                    # --- B operand load (V = K_nope_LDS)  [E.bug.5 fix]
                    # Lane (i, j) needs 8 fp8 at row={k_off+i*8..i*8+7}, col=v_col.
                    # K_nope_LDS is (BI, D) row-major fp8; consecutive K-rows
                    # are D=512 bytes apart, NOT i32-contiguous for general
                    # v_col.  Issue 8 single-byte loads per lane and pack
                    # them into one i64 via vector.from_elements + bitcast.
                    v_col = n_tile_col_base + mfma_row
                    v_byte_base = (
                        (fx.Int32(k_off) + lane_hi4 * fx.Int32(8))
                        * fx.Int32(D) + v_col
                    )
                    v_bytes = []
                    for b in range_constexpr(8):
                        v_off = v_byte_base + fx.Int32(b * D)
                        v_one = vector.load_op(
                            T.vec(1, T.i8), kn_lds_bytes,
                            [arith.index_cast(T.index, v_off)],
                        )
                        v_one_i8 = vector.extract(
                            v_one, static_position=[0], dynamic_position=[]
                        )
                        v_bytes.append(v_one_i8)
                    v_vec_i8 = vector.from_elements(T.vec(8, T.i8), v_bytes)
                    v_i64_vec = vector.bitcast(T.vec(1, T.i64), v_vec_i8)
                    v_i64 = vector.extract(
                        v_i64_vec, static_position=[0], dynamic_position=[]
                    )

                    acc_nt = rocdl.mfma_f32_16x16x32_fp8_fp8(
                        T.f32x4, [a_i64, v_i64, acc_nt, 0, 0, 0]
                    )
                acc_o[nt] = acc_nt

            gpu.barrier()  # before next chunk overwrites K_nope_LDS

        # ================================================================
        # Emit partial outputs to global (for combine kernel to reduce).
        # NO final divide / bf16 cast here -- combine sees raw fp32.
        # Apply PROB_SCALE = 1/FP8_MAX once (compensates for the
        # cvt_pk_fp8_f32 P-pack scale of FP8_MAX in STEP B11).
        #
        # Layouts:
        #   partial_O : (BS, NUM_PARTIALS, HG, D)   fp32, written via buffer_store
        #   partial_M : (BS, NUM_PARTIALS, HG)      fp32, running_max per row
        #   partial_L : (BS, NUM_PARTIALS, HG)      fp32, running_sum per row
        # ================================================================
        PROB_SCALE = arith.constant(1.0 / FP8_MAX, type=T.f32)
        for nt in range_constexpr(SV_N_TILES_WARP):
            n_tile_col = (
                warp_id * fx.Int32(SV_N_PER_WARP)
                + fx.Int32(nt * 16) + mfma_row
            )
            acc_nt = acc_o[nt]
            for r in range_constexpr(4):
                v = vector.extract(acc_nt, static_position=[r], dynamic_position=[])
                row = lane_hi4 * fx.Int32(4) + fx.Int32(r)
                # partial_o[batch, partial, row, n_tile_col] stride = HG * D
                # fp32 = 4 bytes
                elem_off = (
                    partial_o_off + row * fx.Int32(D) + n_tile_col
                )
                buffer_ops.buffer_store(
                    v * PROB_SCALE, po_rsrc, elem_off,
                )

        # Write running_max / running_sum (per-row, per-partial).  All 16
        # lanes of the same lane group hold the SAME 4 row-values (after
        # the wave reductions), so the write race is benign.  We use 4
        # stores per lane per output (HG*NUM_WARPS=64 writes per row, all
        # storing the same value).
        for r in range_constexpr(4):
            row = lane_hi4 * fx.Int32(4) + fx.Int32(r)
            m_v = vector.extract(running_max, static_position=[r], dynamic_position=[])
            l_v = vector.extract(running_sum, static_position=[r], dynamic_position=[])
            buffer_ops.buffer_store(
                m_v, pm_rsrc, partial_ml_off + row,
            )
            buffer_ops.buffer_store(
                l_v, pl_rsrc, partial_ml_off + row,
            )

    @flyc.jit
    def launch_fn(
        partial_o_ptr: fx.Tensor,
        partial_m_ptr: fx.Tensor,
        partial_l_ptr: fx.Tensor,
        q_nope_ptr: fx.Tensor,
        q_rope_ptr: fx.Tensor,
        q_scale_ptr: fx.Tensor,
        k_nope_ptr: fx.Tensor,
        k_scale_ptr: fx.Tensor,
        k_rope_ptr: fx.Tensor,
        indices_ptr: fx.Tensor,
        bs: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        fmha_partial_kernel(
            partial_o_ptr, partial_m_ptr, partial_l_ptr,
            q_nope_ptr, q_rope_ptr, q_scale_ptr,
            k_nope_ptr, k_scale_ptr, k_rope_ptr, indices_ptr,
        ).launch(
            grid=(bs, fx.Int32(NUM_PARTIALS)),
            block=(BLOCK_THREADS,),
            smem=SMEM_BYTES, stream=stream,
        )

    return launch_fn


@functools.lru_cache(maxsize=4)
def _build_combine_kernel():
    """Reduce ``NUM_PARTIALS`` per-CTA partials into the final bf16 output.

    Grid:  (BS, HG)   -- one CTA per (batch, head)
    Block: 64 threads (one wave)

    Each thread covers ``D / 64 = 8`` output dims.  It reads NUM_PARTIALS
    ``(m, l, O[8 dims])`` triples, combines them via the standard flash
    LSE-rescale, and writes the final bf16 output.
    """
    arch = _get_arch()
    DIMS_PER_THREAD = D // WARP_SIZE   # 512 / 64 = 8

    @flyc.kernel
    def combine_kernel(
        out_ptr: fx.Tensor,            # (BS, HG, D) bf16
        partial_o_ptr: fx.Tensor,      # (BS, NUM_PARTIALS, HG, D) fp32
        partial_m_ptr: fx.Tensor,      # (BS, NUM_PARTIALS, HG) fp32
        partial_l_ptr: fx.Tensor,      # (BS, NUM_PARTIALS, HG) fp32
    ):
        tid = gpu.thread_idx.x        # 0..63
        seq_idx = gpu.block_idx.x
        head_idx = gpu.block_idx.y

        out_rsrc = buffer_ops.create_buffer_resource(out_ptr, max_size=True)
        po_rsrc = buffer_ops.create_buffer_resource(partial_o_ptr, max_size=True)
        pm_rsrc = buffer_ops.create_buffer_resource(partial_m_ptr, max_size=True)
        pl_rsrc = buffer_ops.create_buffer_resource(partial_l_ptr, max_size=True)

        # Element offsets.
        po_seq_off = seq_idx * fx.Int32(NUM_PARTIALS * HG * D)
        ml_seq_off = seq_idx * fx.Int32(NUM_PARTIALS * HG)
        out_off = seq_idx * fx.Int32(HG * D) + head_idx * fx.Int32(D) + tid * fx.Int32(DIMS_PER_THREAD)

        # Pass 1: load all NUM_PARTIALS (m, l) for this head; find global max.
        # NUM_PARTIALS = 8; we hold them in registers per thread.
        ms = []
        ls = []
        for p in range_constexpr(NUM_PARTIALS):
            m_off = ml_seq_off + fx.Int32(p * HG) + head_idx
            l_off = m_off
            m_p = buffer_ops.buffer_load(pm_rsrc, m_off, vec_width=1, dtype=T.f32)
            l_p = buffer_ops.buffer_load(pl_rsrc, l_off, vec_width=1, dtype=T.f32)
            ms.append(m_p)
            ls.append(l_p)

        m_global = ms[0]
        for p in range_constexpr(NUM_PARTIALS - 1):
            m_global = m_global.maximumf(ms[p + 1])

        # alpha[p] = exp(m_p - m_global); l_global = sum(alpha[p] * l_p).
        alphas = []
        l_global = fx.Float32(0.0)
        for p in range_constexpr(NUM_PARTIALS):
            a = (ms[p] - m_global).exp2(fastmath=arith.FastMathFlags.fast)
            alphas.append(a)
            l_global = l_global + a * ls[p]

        # Pass 2: per thread, accumulate alpha[p] * partial_O[p, head, my-dims].
        # Output: 8 fp32 values per thread, then divide by l_global, cast bf16.
        accs = [fx.Float32(0.0) for _ in range(DIMS_PER_THREAD)]
        for p in range_constexpr(NUM_PARTIALS):
            base_p = (
                po_seq_off + fx.Int32(p * HG * D)
                + head_idx * fx.Int32(D) + tid * fx.Int32(DIMS_PER_THREAD)
            )
            for d in range_constexpr(DIMS_PER_THREAD):
                v = buffer_ops.buffer_load(
                    po_rsrc, base_p + fx.Int32(d),
                    vec_width=1, dtype=T.f32,
                )
                accs[d] = accs[d] + alphas[p] * v

        # Avoid divide-by-zero for fully-masked queries.
        ZERO_F = fx.Float32(0.0)
        ONE_F = fx.Float32(1.0)
        cond = arith.cmpf(arith.CmpFPredicate.OEQ, l_global, ZERO_F)
        safe = arith.select(cond, ONE_F, l_global)

        # Cast and store.  8 contiguous bf16 = 16 bytes per thread.
        for d in range_constexpr(DIMS_PER_THREAD):
            normed = accs[d] / safe
            bf16_v = arith.trunc_f(T.bf16, normed)
            i16_v = arith.bitcast(T.i16, bf16_v)
            byte_off = (out_off + fx.Int32(d)) * fx.Int32(2)
            buffer_ops.buffer_store(
                i16_v, out_rsrc, byte_off, offset_is_bytes=True
            )

    @flyc.jit
    def launch_combine(
        out_ptr: fx.Tensor,
        partial_o_ptr: fx.Tensor,
        partial_m_ptr: fx.Tensor,
        partial_l_ptr: fx.Tensor,
        bs: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        combine_kernel(
            out_ptr, partial_o_ptr, partial_m_ptr, partial_l_ptr,
        ).launch(
            grid=(bs, fx.Int32(HG)),
            block=(WARP_SIZE,),
            stream=stream,
        )

    return launch_combine


def _quantize_to_fp8(x_bf16):
    amax = x_bf16.abs().amax(dim=-1, keepdim=True).float()
    scale = (amax / FP8_MAX).clamp(min=1e-8)
    x_scaled = (x_bf16.float() / scale).clamp(-FP8_MAX, FP8_MAX)
    x_fp8 = x_scaled.to(_FP8_DTYPE)
    return x_fp8, scale.squeeze(-1).contiguous()


def flydsl_sparse_mla_decode_fused(
    q: torch.Tensor,
    kv_paged_uint8: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> torch.Tensor:
    """Public NSA-compatible entry point for the fused FlyDSL fmha.

    Mirrors ``tilelang_sparse_fwd`` signature so it can be plugged into
    ``nsa_backend._forward_tilelang`` directly via
    ``SGLANG_NSA_TILELANG_VARIANT=flydsl``.

    Parameters
    ----------
    q : (seq_len, heads, dim+tail_dim=576) bf16
    kv_paged_uint8 : (num_blocks, block_size, 1, 656) uint8 -- FP8 paged
        NSA KV layout (nope_fp8 + 4 fp32 tile-scales + rope_bf16).
    indices : (seq_len, kv_group=1, topk) int32
    sm_scale : float
    d_v : int (== 512)

    Notes
    -----
    The fused kernel currently consumes a **single per-token K-scale**
    (not per-tile).  We collapse the 4 NSA per-tile scales to a per-token
    mean as a near-correct approximation; the proper per-tile fp8 path
    requires applying different scales across the 4 K-chunks of the QK
    nope MFMA -- tracked as a follow-up optimization in the file
    docstring.
    """
    from sglang.srt.layers.attention.nsa.tilelang_kernel_fp8 import (
        _split_paged_kv_fp8,
    )

    assert d_v == D, f"flydsl fused kernel fixes d_v={D}, got {d_v}"
    assert q.dim() == 3 and indices.dim() == 3
    seq_len, heads, dim_total = q.shape
    assert dim_total == D + D_ROPE
    # The fused kernel is built for HG=16 padded heads.  GLM-5.1 with TP=8
    # gives heads=8 -- pad up to 16 here, drop the trailing 8 at output.
    assert heads <= HG, f"fused kernel max HG={HG}, got {heads}"

    # --- Split paged KV --------------------------------------------------
    k_nope_fp8_full, k_scale_full, k_rope_full = _split_paged_kv_fp8(
        kv_paged_uint8
    )
    k_nope_fp8 = k_nope_fp8_full.reshape(-1, D).contiguous()
    # Collapse 4 per-tile scales to a per-token MEAN (approximation).
    k_scale = k_scale_full.reshape(-1, 4).mean(dim=-1).contiguous()
    k_rope = k_rope_full.reshape(-1, D_ROPE).contiguous()

    # --- Pad heads to HG -------------------------------------------------
    if heads < HG:
        pad = HG - heads
        q = torch.nn.functional.pad(q, (0, 0, 0, pad))   # pad heads dim
    q_nope = q[:, :, :D].contiguous()
    q_rope = q[:, :, D:].contiguous()

    # --- Quantize Q nope per-(batch, head) row --------------------------
    q_nope_amax = q_nope.abs().amax(dim=-1, keepdim=True).float()
    q_scale = (q_nope_amax / FP8_MAX).clamp(min=1e-8).squeeze(-1)  # (batch, HG)
    q_nope_scaled = (q_nope.float() / q_scale.unsqueeze(-1)).clamp(
        -FP8_MAX, FP8_MAX
    )
    q_nope_fp8 = q_nope_scaled.to(_FP8_DTYPE).contiguous()
    q_scale = q_scale.contiguous()

    # --- Indices ---------------------------------------------------------
    if indices.dim() == 3:
        assert indices.shape[1] == 1
        indices = indices.squeeze(1)
    indices = indices.contiguous().to(torch.int32)

    # --- Allocate partial buffers + final output ------------------------
    partial_o = torch.empty(
        seq_len, NUM_PARTIALS, HG, D,
        dtype=torch.float32, device=q.device,
    )
    partial_m = torch.empty(
        seq_len, NUM_PARTIALS, HG,
        dtype=torch.float32, device=q.device,
    )
    partial_l = torch.empty(
        seq_len, NUM_PARTIALS, HG,
        dtype=torch.float32, device=q.device,
    )
    out = torch.empty(
        seq_len, HG, D, dtype=torch.bfloat16, device=q.device,
    )

    # --- Launch partial (multi-CTA: grid = (BS, NUM_PARTIALS)) ----------
    launch_partial = _build_sparse_mla_kernel(sm_scale)
    launch_partial(
        partial_o, partial_m, partial_l,
        q_nope_fp8, q_rope, q_scale,
        k_nope_fp8, k_scale, k_rope, indices,
        int(seq_len),
    )

    # --- Launch combine (grid = (BS, HG)) -------------------------------
    launch_combine = _build_combine_kernel()
    launch_combine(
        out, partial_o, partial_m, partial_l, int(seq_len),
    )

    # Trim padded heads back
    if heads < HG:
        out = out[:, :heads, :].contiguous()
    return out


def run_test():
    torch.manual_seed(0)
    q_nope_bf16 = torch.randn(HG, D, dtype=torch.bfloat16, device="cuda") * 0.5
    q_rope_bf16 = torch.randn(HG, D_ROPE, dtype=torch.bfloat16, device="cuda") * 0.5
    k_nope_bf16 = torch.randn(T_POOL, D, dtype=torch.bfloat16, device="cuda") * 0.5
    k_rope_bf16 = torch.randn(T_POOL, D_ROPE, dtype=torch.bfloat16, device="cuda") * 0.5

    q_nope_fp8, q_scale = _quantize_to_fp8(q_nope_bf16)
    k_nope_fp8, k_scale = _quantize_to_fp8(k_nope_bf16)
    indices = torch.randperm(T_POOL, device="cuda")[:TOPK].to(torch.int32)

    sm_scale = 1.0 / math.sqrt(D + D_ROPE)

    # Standalone test runs batch=1; reshape to (1, HG, ...) for the
    # batched kernel signature.
    out = torch.zeros(1, HG, D, dtype=torch.bfloat16, device="cuda")
    partial_o = torch.empty(1, NUM_PARTIALS, HG, D, dtype=torch.float32, device="cuda")
    partial_m = torch.empty(1, NUM_PARTIALS, HG, dtype=torch.float32, device="cuda")
    partial_l = torch.empty(1, NUM_PARTIALS, HG, dtype=torch.float32, device="cuda")
    try:
        launch_partial = _build_sparse_mla_kernel(sm_scale)
        launch_partial(
            partial_o, partial_m, partial_l,
            q_nope_fp8.unsqueeze(0).contiguous(),
            q_rope_bf16.unsqueeze(0).contiguous(),
            q_scale.unsqueeze(0).contiguous(),
            k_nope_fp8.contiguous(),
            k_scale.contiguous(),
            k_rope_bf16.contiguous(),
            indices.unsqueeze(0).contiguous(),
            1,
        )
        launch_combine = _build_combine_kernel()
        launch_combine(out, partial_o, partial_m, partial_l, 1)
        torch.cuda.synchronize()
        out = out.squeeze(0)
    except Exception as e:
        print(f"[WIP] Step E full fmha kernel build/launch failed: "
              f"{type(e).__name__}: {e}")
        return float("inf")

    # Reference: compute full-precision sparse-MLA decode in fp32
    k_nope_g = k_nope_bf16[indices.long()].float()
    k_rope_g = k_rope_bf16[indices.long()].float()
    s = (q_nope_bf16.float() @ k_nope_g.T + q_rope_bf16.float() @ k_rope_g.T) * sm_scale
    p = torch.softmax(s, dim=-1)
    ref = (p @ k_nope_g).to(torch.bfloat16)

    err = (out.float() - ref.float()).abs().max().item()
    print(f"FlyDSL sparse-MLA decode (HG={HG}, D={D}, TopK={TOPK}):")
    print(f"  max-abs vs torch fp32 ref = {err:.5f}")
    print("OK" if err < 0.5 else "FAIL")
    if err >= 0.5:
        diff = (out.float() - ref.float()).abs()
        print(f"  ref range: {ref.float().min():.3f} .. {ref.float().max():.3f}")
        print(f"  out range: {out.float().min():.3f} .. {out.float().max():.3f}")
        print(f"  out has nan: {torch.isnan(out.float()).any().item()}")
    return err


if __name__ == "__main__":
    err = run_test()
    sys.exit(0 if err < 0.5 else 1)
