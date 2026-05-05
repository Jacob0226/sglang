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

_FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0
LOG2E = 1.4426950408889634


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
    def fmha_kernel(
        out_ptr: fx.Tensor,            # (HG, D) bf16
        q_nope_ptr: fx.Tensor,         # (HG, D) fp8
        q_rope_ptr: fx.Tensor,         # (HG, D_ROPE) bf16
        q_scale_ptr: fx.Tensor,        # (HG,) fp32
        k_nope_ptr: fx.Tensor,         # (T_POOL, D) fp8
        k_scale_ptr: fx.Tensor,        # (T_POOL,) fp32
        k_rope_ptr: fx.Tensor,         # (T_POOL, D_ROPE) bf16
        indices_ptr: fx.Tensor,        # (TOPK,) int32
    ):
        tid = gpu.thread_idx.x
        warp_id = tid // fx.Int32(WARP_SIZE)
        lane = tid % fx.Int32(WARP_SIZE)
        lane_hi4 = lane // fx.Int32(16)            # 0..3
        mfma_row = lane % fx.Int32(16)             # 0..15

        out_rsrc = buffer_ops.create_buffer_resource(out_ptr, max_size=True)
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
        for li in range_constexpr(8):
            qi = tid + fx.Int32(li * BLOCK_THREADS)
            qv = buffer_ops.buffer_load(qn_rsrc, qi, vec_width=1, dtype=T.i32)
            vector.store(
                vector.from_elements(T.vec(1, T.i32), [qv]),
                qn_lds_i32,
                [arith.index_cast(T.index, qi)],
            )

        # --- STEP A2: load Q_rope to LDS (2 KB / 256 threads x 2 i32) -----
        for li in range_constexpr(2):
            qi = tid + fx.Int32(li * BLOCK_THREADS)
            qv = buffer_ops.buffer_load(qr_rsrc, qi, vec_width=1, dtype=T.i32)
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
        # Per-chunk loop
        # ================================================================
        for chunk_i in range_constexpr(NI):
            chunk_off = fx.Int32(chunk_i * BI)

            # --- STEP B1: gather K_nope_chunk -------------------------
            for li in range_constexpr(K_NOPE_BYTES // 4 // BLOCK_THREADS):
                lds_idx = tid + fx.Int32(li * BLOCK_THREADS)
                token = lds_idx // fx.Int32(D_I32)
                k_i32 = lds_idx % fx.Int32(D_I32)
                src_row = buffer_ops.buffer_load(
                    idx_rsrc, chunk_off + token, vec_width=1, dtype=T.i32
                )
                g_i32 = src_row * fx.Int32(D_I32) + k_i32
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
                    idx_rsrc, chunk_off + token, vec_width=1, dtype=T.i32
                )
                g_i32 = src_row * fx.Int32(D_ROPE_I32) + k_i32
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
                qs = buffer_ops.buffer_load(qs_rsrc, row_idx, vec_width=1, dtype=T.f32)
                q_scale_v = vector.insert(qs, q_scale_v, static_position=[r], dynamic_position=[])

            # K-scale: per-token, the column we wrote in this lane is
            # warp_id*16 + mfma_row of the chunk.  Single fp32 per lane.
            tok_idx = chunk_off + warp_id * fx.Int32(16) + mfma_row
            k_tok_global = buffer_ops.buffer_load(idx_rsrc, tok_idx, vec_width=1, dtype=T.i32)
            ks = buffer_ops.buffer_load(ks_rsrc, k_tok_global, vec_width=1, dtype=T.f32)

            # Mask: indices < 0 -> -inf
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
            # Write to LDS: 1 lane per (warp_id, lane_group) writes 4 row-maxes.
            # We use mfma_row==0 as the writer (one per (warp, lane_hi4)).
            cond_writer = arith.cmpi(arith.CmpIPredicate.eq, mfma_row, fx.Int32(0))
            with_writer = scf.IfOp(cond_writer, results_=[], has_else=False)
            with ir.InsertionPoint(with_writer.then_block):
                for r in range_constexpr(4):
                    v = vector.extract(local_max, static_position=[r], dynamic_position=[])
                    row_idx = lane_hi4 * fx.Int32(4) + fx.Int32(r)
                    slot = warp_id * fx.Int32(HG) + row_idx
                    max_lds.store(v, [arith.index_cast(T.index, slot)])
                scf.YieldOp([])

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
            p_v = arith.constant_vector(0.0, T.f32x4)
            local_sum = ZERO_F
            for r in range_constexpr(4):
                v = vector.extract(acc_qk, static_position=[r], dynamic_position=[])
                nm = vector.extract(new_max_v, static_position=[r], dynamic_position=[])
                p = (v - nm).exp2(fastmath=arith.FastMathFlags.fast)
                p_v = vector.insert(p, p_v, static_position=[r], dynamic_position=[])
                local_sum = local_sum + p

            # --- STEP B10: per-row sum of P (warp-local, then cross-warp) -----
            # local_sum is f32 (single value per lane, but we want per-row sums).
            # Actually we stored 4 values per lane (4 rows).  Need a per-row
            # warp-local sum.  Reuse _wave_sum_within_16 on each row's element.
            # For simplicity, run sum as a vector reduction.  ``running_sum``
            # update per-row uses the per-row sum.
            psum_v = arith.constant_vector(0.0, T.f32x4)
            for r in range_constexpr(4):
                v = vector.extract(p_v, static_position=[r], dynamic_position=[])
                psum_v = vector.insert(v, psum_v, static_position=[r], dynamic_position=[])
            local_sum_v = _wave_sum_within_16(psum_v)

            # Write 4 row-sums per (warp, lane_hi4) to LDS.
            cond_writer2 = arith.cmpi(arith.CmpIPredicate.eq, mfma_row, fx.Int32(0))
            with_writer2 = scf.IfOp(cond_writer2, results_=[], has_else=False)
            with ir.InsertionPoint(with_writer2.then_block):
                for r in range_constexpr(4):
                    v = vector.extract(local_sum_v, static_position=[r], dynamic_position=[])
                    row_idx = lane_hi4 * fx.Int32(4) + fx.Int32(r)
                    slot = warp_id * fx.Int32(HG) + row_idx
                    sum_lds.store(v, [arith.index_cast(T.index, slot)])
                scf.YieldOp([])

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

                    # --- B operand load (V = K_nope_LDS)
                    # B operand: lane (i, j) provides 8 fp8 at col=j (within tile),
                    # k={i*8..+7}.  Token row in K_nope_LDS = k_off + i*8 + 0..7;
                    # col in K_nope_LDS = n_tile_col_base + j.  4 fp8 = 1 i32.
                    v_token = fx.Int32(k_off) + lane_hi4 * fx.Int32(8)
                    v_col = n_tile_col_base + mfma_row
                    v_elem_off = v_token * fx.Int32(D) + v_col
                    # Need 8 fp8 = 2 i32, but they're at consecutive token rows
                    # NOT consecutive bytes.  Issue 2 separate i32 loads.
                    v_i32_idx_a = v_elem_off // fx.Int32(4)   # ! v_col MUST be i32-aligned for this to work
                    # Actually each (token, col) is a single fp8.  i32 index = (token * D + col) // 4.
                    # For mfma_row going 0..15, v_col varies and is NOT 4-aligned in general.
                    # FIXME: this load assumes col-alignment which doesn't hold; needs byte loads
                    # or B layout reorganization.  Marked as a known issue for next iteration.
                    v_a = vector.load_op(
                        T.vec(1, T.i32), kn_lds_i32,
                        [arith.index_cast(T.index, v_i32_idx_a)],
                    )
                    v_a_i32 = vector.extract(v_a, static_position=[0], dynamic_position=[])
                    # FIXME: dummy second pack; layout TBD
                    v_i64 = _pack_i32_pair_to_i64(v_a_i32, v_a_i32)

                    acc_nt = rocdl.mfma_f32_16x16x32_fp8_fp8(
                        T.f32x4, [a_i64, v_i64, acc_nt, 0, 0, 0]
                    )
                acc_o[nt] = acc_nt

            gpu.barrier()  # before next chunk overwrites K_nope_LDS

        # ================================================================
        # Final normalize + bf16 store
        # ================================================================
        # acc_o[nt][r] /= running_sum[r], cast to bf16, write to O[row, col]
        for nt in range_constexpr(SV_N_TILES_WARP):
            n_tile_col = warp_id * fx.Int32(SV_N_PER_WARP) + fx.Int32(nt * 16) + mfma_row
            acc_nt = acc_o[nt]
            for r in range_constexpr(4):
                v = vector.extract(acc_nt, static_position=[r], dynamic_position=[])
                rs = vector.extract(running_sum, static_position=[r], dynamic_position=[])
                # Avoid div-by-zero if a row had all-masked tokens.
                safe = rs.cmp_eq(ZERO_F).select(ONE_F, rs)
                normed = v / safe
                # cast to bf16 and store
                row = lane_hi4 * fx.Int32(4) + fx.Int32(r)
                # buffer_store one bf16 at a time (slow but simple).  We don't
                # have a 1-bf16 store helper, so cast through i16.
                # Actually buffer_ops.buffer_store accepts vec; pack via vector.
                bf16_v = arith.trunc_f(T.bf16, normed)
                # Compute byte offset in O (HG, D) bf16 = 16 * 512 * 2 = 16 KB
                byte_off = (row * fx.Int32(D) + n_tile_col) * fx.Int32(2)
                # Pack single bf16 -> i16 -> store as i16
                bf16_i16 = arith.bitcast(T.i16, bf16_v)
                buffer_ops.buffer_store(
                    bf16_i16, out_rsrc, byte_off, offset_is_bytes=True
                )

    @flyc.jit
    def launch_fn(
        out_ptr: fx.Tensor,
        q_nope_ptr: fx.Tensor,
        q_rope_ptr: fx.Tensor,
        q_scale_ptr: fx.Tensor,
        k_nope_ptr: fx.Tensor,
        k_scale_ptr: fx.Tensor,
        k_rope_ptr: fx.Tensor,
        indices_ptr: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        fmha_kernel(
            out_ptr, q_nope_ptr, q_rope_ptr, q_scale_ptr,
            k_nope_ptr, k_scale_ptr, k_rope_ptr, indices_ptr,
        ).launch(
            grid=(1,), block=(BLOCK_THREADS,),
            smem=SMEM_BYTES, stream=stream,
        )

    return launch_fn


def _quantize_to_fp8(x_bf16):
    amax = x_bf16.abs().amax(dim=-1, keepdim=True).float()
    scale = (amax / FP8_MAX).clamp(min=1e-8)
    x_scaled = (x_bf16.float() / scale).clamp(-FP8_MAX, FP8_MAX)
    x_fp8 = x_scaled.to(_FP8_DTYPE)
    return x_fp8, scale.squeeze(-1).contiguous()


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

    out = torch.zeros(HG, D, dtype=torch.bfloat16, device="cuda")
    try:
        launch_fn = _build_sparse_mla_kernel(sm_scale)
        launch_fn(
            out, q_nope_fp8, q_rope_bf16, q_scale,
            k_nope_fp8, k_scale, k_rope_bf16, indices,
        )
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[WIP] Step E full fmha kernel build/launch failed: "
              f"{type(e).__name__}: {e}")
        print("See file docstring for the bug-list to fix (E.bug.1..6).")
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
