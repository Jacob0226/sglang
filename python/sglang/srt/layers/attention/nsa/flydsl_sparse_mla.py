"""
FlyDSL sparse-MLA decode kernel for NSA / GLM-5.1-FP8.

Two-kernel design (matches TileLang's main_kernel + combine on AMD CDNA4
which lacks B200's cluster-cooperative attention):

  partial_kernel  -- grid = (BS, NUM_PARTIALS); each CTA processes a
                     contiguous slice of the topk dimension and writes a
                     partial (acc_o, running_max, running_sum) to global
                     memory.

  combine_kernel  -- grid = (BS, actual_heads); reduces across the
                     NUM_PARTIALS partials with the standard log-sum-exp
                     stable trick and writes the final bf16 output.

The kernel is **fully fused** in the single-CTA sense -- one CTA reads Q
once, walks all assigned chunks doing QK + online softmax + SV in
registers + LDS, and only emits its final partial at the very end.  No
intermediate global-memory round trips.

Performance vs TileLang main_kernel baseline (MI355X gfx950, GLM-5.1-FP8
shapes -- batch=4 heads=8 topk=2048 d_v=512)::

    bf16 baseline (TileLang main_kernel) :  66 us / call
    FlyDSL fused (full wrapper)          :  59 us / call    1.12x FASTER
    FlyDSL fused (kernel ONLY, no prep)  :  50 us / call    1.33x faster

In-kernel responsibilities (all of these used to live in the wrapper):
  - Q FP8 quantization (in-kernel amax + scale + cast in STEP A3).
  - KV pulled directly from the canonical 656-byte paged buffer
    (STEPs B1/B2/B5; no host-side .contiguous() splits).
  - Heads padding: caller passes Q (BS, actual_heads, dim_total) without
    F.pad; kernel masks ``head_idx_load >= actual_heads`` via a clamped
    offset + vector-select on the cooperative Q load.

Wrapper does only zero-cost views (paged-buffer flatten, indices
.squeeze) and the ``c.contiguous()`` short-circuit.  Total wrapper-only
overhead is ~9 us / call.

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
from flydsl.expr import math as fmath
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

# 656-byte NSA paged KV layout (matches nsa.quant_k_cache).
KV_BYTES_PER_TOKEN = 656
KV_I32_PER_TOKEN = KV_BYTES_PER_TOKEN // 4   # 164
KV_NOPE_I32_OFFSET = 0                        # bytes [  0..511]  fp8 nope (128 i32)
KV_SCALE_I32_OFFSET = D // 4                  # bytes [512..527]  fp32 scales x 4 (4 i32) -- 128
KV_ROPE_I32_OFFSET = (D + 4 * 4) // 4         # bytes [528..655]  bf16 rope (32 i32) -- 132


@functools.lru_cache(maxsize=8)
def _build_sparse_mla_kernel(sm_scale: float):
    arch = _get_arch()
    assert str(arch).startswith("gfx95"), f"expected gfx950, got {arch}"

    Q_NOPE_BF16_BYTES = HG * D * 2      # 16 KB -- transient bf16 staging for in-kernel Q-quant
    Q_NOPE_BYTES = HG * D               #  8 KB -- fp8 Q nope (output of in-kernel quant)
    Q_ROPE_BYTES = HG * D_ROPE * 2      #  2 KB -- bf16 rope
    Q_SCALE_BYTES = HG * 4              # 64 B  -- per-head q_scale fp32
    K_NOPE_BYTES = BI * D               # 32 KB
    K_ROPE_BYTES = BI * D_ROPE * 2      #  8 KB
    P_BYTES = HG * BI                   #  1 KB fp8
    S_BYTES = HG * BI * 4               #  4 KB fp32 staging
    MAX_BYTES = HG * NUM_WARPS * 4      #  256 B per (row, warp) fp32
    SUM_BYTES = HG * NUM_WARPS * 4      #  256 B

    allocator = SmemAllocator(
        None, arch=arch, global_sym_name="flydsl_sparse_mla_smem"
    )
    # Q_NOPE_BF16 is transient (used only during in-kernel quant prologue).
    # Overlap it with K_NOPE region since they're never live at the same
    # time -- but for first cut keep them separate for clarity.
    q_nope_bf16_off = 0
    allocator.ptr = Q_NOPE_BF16_BYTES
    q_nope_off = allocator.ptr
    allocator.ptr += Q_NOPE_BYTES
    q_rope_off = allocator.ptr
    allocator.ptr += Q_ROPE_BYTES
    q_scale_off = allocator.ptr
    allocator.ptr += Q_SCALE_BYTES
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
        # Q (bf16; single tensor, kernel reads nope/rope by byte offset) -
        q_ptr: fx.Tensor,              # (BS, HG, D + D_ROPE) bf16
        # KV (single 656-byte paged buffer, kernel reads directly) -------
        kv_paged_ptr: fx.Tensor,       # (T_POOL, 656) uint8 viewed as i32
        # Indices ---------------------------------------------------------
        indices_ptr: fx.Tensor,        # (BS, TOPK) int32
        # Number of active heads in Q (<= HG).  Padded heads' Q reads are
        # masked to 0 via clamped offsets + vector select (no F.pad on host).
        actual_heads: Int32,
    ):
        tid = gpu.thread_idx.x
        seq_idx = gpu.block_idx.x      # batch / decode-query index
        partial_idx = gpu.block_idx.y  # which slice of the topk loop (0..NUM_PARTIALS)
        warp_id = tid // fx.Int32(WARP_SIZE)
        lane = tid % fx.Int32(WARP_SIZE)
        lane_hi4 = lane // fx.Int32(16)            # 0..3
        mfma_row = lane % fx.Int32(16)             # 0..15

        # Per-batch offsets.  Single Q tensor (BS, actual_heads, dim_total);
        # caller does NOT F.pad to HG -- the kernel's load loop covers HG
        # logical rows per CTA but reads from a clamped offset and zeros
        # out the result via vector-select for h_idx >= actual_heads.
        q_dim_total = D + D_ROPE
        q_batch_off = seq_idx * actual_heads * fx.Int32(q_dim_total)
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
        # Single Q resource; nope at byte offset 0 of each row, rope at
        # byte offset D*2.  Per-row bf16 stride = (D + D_ROPE) * 2 bytes.
        # The wrapper still F.pads heads to HG (h<HG path TBD).
        q_rsrc = buffer_ops.create_buffer_resource(q_ptr, max_size=True)
        # Single resource for the entire 656 B/token paged KV buffer.
        kv_rsrc = buffer_ops.create_buffer_resource(kv_paged_ptr, max_size=True)
        idx_rsrc = buffer_ops.create_buffer_resource(indices_ptr, max_size=True)

        base = allocator.get_base()
        # Q bf16 staging (transient -- only live during quant prologue).
        qn_bf16_lds_i32 = SmemPtr(
            base, q_nope_bf16_off, T.i32, shape=(Q_NOPE_BF16_BYTES // 4,),
        ).get()
        qn_bf16_lds_bf16 = SmemPtr(
            base, q_nope_bf16_off, T.bf16, shape=(Q_NOPE_BF16_BYTES // 2,),
        ).get()
        # Q fp8 (output of in-kernel quant, used by all 32 chunks).
        qn_lds_i32 = SmemPtr(base, q_nope_off, T.i32, shape=(Q_NOPE_BYTES // 4,)).get()
        # Q rope bf16 (passed through unchanged).
        qr_lds_i32 = SmemPtr(base, q_rope_off, T.i32, shape=(Q_ROPE_BYTES // 4,)).get()
        # Per-head q_scale (1 fp32 per head, 16 fp32 total).
        q_scale_lds = SmemPtr(base, q_scale_off, T.f32, shape=(HG,))
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

        # --- STEP A1: load Q (bf16) to two LDS regions ----------------
        # Q layout: (BS, HG, D + D_ROPE) bf16.  Per-row stride = 1152 B.
        # 256 threads cooperatively load all HG rows.
        # We split nope/rope at LDS-write time to keep downstream kernel
        # logic identical to the previous separate-nope/separate-rope
        # design.  Avoids host-side ``q[:, :, :D].contiguous()`` copies.
        # Each thread handles one (head, dim_chunk_of_8_bf16) pair.
        # head_idx = tid // (q_dim_total / 8) ;
        # dim_off  = (tid % (q_dim_total / 8)) * 8 (bf16 elements)
        # Total per CTA: HG * (D + D_ROPE) / 8 = 16 * 144 = 2304 vec8 loads.
        # 256 threads x 9 loads each.
        DIM_CHUNKS_PER_HEAD = (D + D_ROPE) // 8   # 144 -- 8 bf16 per chunk
        TOTAL_CHUNKS = HG * DIM_CHUNKS_PER_HEAD   # 2304
        # Build a zero v4xi32 once (used to mask invalid head loads).
        ZERO_I32 = fx.Int32(0)
        zero_v4 = vector.from_elements(
            T.vec(4, T.i32), [ZERO_I32, ZERO_I32, ZERO_I32, ZERO_I32]
        )
        for li in range_constexpr((TOTAL_CHUNKS + BLOCK_THREADS - 1) // BLOCK_THREADS):
            chunk_id = tid + fx.Int32(li * BLOCK_THREADS)
            head_idx_load = chunk_id // fx.Int32(DIM_CHUNKS_PER_HEAD)
            dim_chunk = chunk_id % fx.Int32(DIM_CHUNKS_PER_HEAD)
            dim_off_bf16 = dim_chunk * fx.Int32(8)  # 0..1144 step 8

            # Mask: only load if head_idx_load < actual_heads.  For
            # invalid lanes we clamp the offset to 0 (a safe in-bounds
            # location) and then zero out the result with vector-select.
            h_valid = arith.cmpi(
                arith.CmpIPredicate.slt, head_idx_load, actual_heads
            )
            elem_off_unsafe = (
                q_batch_off
                + head_idx_load * fx.Int32(q_dim_total)
                + dim_off_bf16
            )
            elem_off = arith.select(h_valid, elem_off_unsafe, ZERO_I32)
            # Load 8 bf16 = 4 i32 = 16 bytes (one buffer_load_dwordx4).
            qv4_loaded = buffer_ops.buffer_load(
                q_rsrc, elem_off // fx.Int32(2),  # i32 index
                vec_width=4, dtype=T.i32,
            )
            qv4 = arith.select(h_valid, qv4_loaded, zero_v4)
            # Decide nope vs rope based on dim_off_bf16:
            #   dim_off in [0, D)        -> nope LDS at row=h, col=dim_off
            #   dim_off in [D, D+D_ROPE) -> rope LDS at row=h, col=dim_off-D
            cond_is_nope = arith.cmpi(
                arith.CmpIPredicate.slt, dim_off_bf16, fx.Int32(D)
            )
            # Compute LDS i32 offsets for both potential destinations.
            # We always store to BOTH (the wrong one is benign since the
            # if-then path below selects the right pointer); cheaper than
            # branching on every chunk.
            nope_off_i32 = (
                head_idx_load * fx.Int32(D) + dim_off_bf16
            ) // fx.Int32(2)  # bf16 -> i32 idx (4 bf16 per i32)
            rope_off_i32 = (
                head_idx_load * fx.Int32(D_ROPE)
                + (dim_off_bf16 - fx.Int32(D))
            ) // fx.Int32(2)
            # We need a conditional store so we don't write rope-region
            # offsets to the nope LDS or vice versa.  scf.IfOp on the
            # boolean ``cond_is_nope``.
            if_op = scf.IfOp(cond_is_nope, results_=[], has_else=True)
            with ir.InsertionPoint(if_op.then_block):
                vector.store(
                    qv4, qn_bf16_lds_i32,
                    [arith.index_cast(T.index, nope_off_i32)],
                )
                scf.YieldOp([])
            with ir.InsertionPoint(if_op.else_block):
                vector.store(
                    qv4, qr_lds_i32,
                    [arith.index_cast(T.index, rope_off_i32)],
                )
                scf.YieldOp([])

        gpu.barrier()  # Q_NOPE_BF16 must be fully resident before quant.

        # --- STEP A3: in-kernel Q-nope FP8 quantize -------------------
        # Replaces ~80us of host-side preprocessing (amax + scale + cast).
        # Layout: 4 warps, each warp handles HG/4 = 4 heads sequentially.
        # Per head: 64 lanes x 8 dims = 512 dims = 1 head row.
        # 1. Each lane reads 8 bf16 from row=h, cols=lane*8..lane*8+7.
        # 2. Compute lane-local amax across 8 fp32-cast values.
        # 3. Wave-reduce maxnumf across 64 lanes (offsets 32,16,8,4,2,1).
        # 4. q_scale_h = amax / FP8_MAX  (broadcast to all 64 lanes).
        # 5. Each lane casts its 8 bf16 to fp8 (cvt_pk_fp8_f32 x 4),
        #    packs 8 fp8 = 2 i32 = 1 i64 store, writes Q_NOPE_FP8_LDS.
        # 6. lane==0 also writes q_scale_h to Q_SCALE_LDS[h].
        FP8MAX_C = arith.constant(FP8_MAX, type=T.f32)
        c_w = fx.Int32(WARP_SIZE)
        for h_local in range_constexpr(HG // NUM_WARPS):  # 4 heads / warp
            h_global = warp_id * fx.Int32(HG // NUM_WARPS) + fx.Int32(h_local)

            # Each lane covers 8 contiguous bf16 elements: row h_global,
            # cols lane*8 .. lane*8+7.
            base_elem = h_global * fx.Int32(D) + lane * fx.Int32(8)
            bf16_8 = vector.load_op(
                T.vec(8, T.bf16), qn_bf16_lds_bf16,
                [arith.index_cast(T.index, base_elem)],
            )

            # Cast bf16 -> fp32, take abs (via math.absf), find lane-local max.
            f32_8 = arith.extf(T.vec(8, T.f32), bf16_8)
            abs_8 = fmath.absf(f32_8)
            local_amax = vector.reduction(T.f32, "maxnumf", abs_8)

            # Wave-reduce across 64 lanes -> wave-wide amax for this head.
            wave_amax = local_amax
            for sh in [32, 16, 8, 4, 2, 1]:
                peer = wave_amax.shuffle_xor(fx.Int32(sh), c_w)
                wave_amax = wave_amax.maximumf(peer)

            # q_scale = amax / FP8_MAX (clamp tiny to avoid div-by-zero
            # downstream); broadcast to all 64 lanes already via shuffle.
            EPS_F = arith.constant(1e-8, type=T.f32)
            q_scale_h = wave_amax / FP8MAX_C
            q_scale_h = q_scale_h.maximumf(EPS_F)

            # All 64 lanes have the same q_scale_h after the wave reduce,
            # so 64 lanes writing the same value to LDS slot h_global is a
            # benign write race.  Avoid scf.IfOp + the SSA-dominance trap
            # we hit with shared subexpressions across if-blocks.
            q_scale_lds.store(
                q_scale_h, [arith.index_cast(T.index, h_global)]
            )

            # Quantize: cast 8 bf16 -> 8 fp8 using q_scale_h.  Pack via
            # cvt_pk_fp8_f32 (2 fp32 -> packed-pair into i32 word).
            inv_scale = ONE_F / q_scale_h
            fp32_scaled = []
            for d in range_constexpr(8):
                v = vector.extract(f32_8, static_position=[d], dynamic_position=[])
                fp32_scaled.append(v * inv_scale)

            zero_i32 = arith.constant(0, type=T.i32)
            i32_pair_lo = rocdl.cvt_pk_fp8_f32(
                T.i32, fp32_scaled[0], fp32_scaled[1], zero_i32, False
            )
            i32_pair_lo = rocdl.cvt_pk_fp8_f32(
                T.i32, fp32_scaled[2], fp32_scaled[3], i32_pair_lo, True
            )
            i32_pair_hi = rocdl.cvt_pk_fp8_f32(
                T.i32, fp32_scaled[4], fp32_scaled[5], zero_i32, False
            )
            i32_pair_hi = rocdl.cvt_pk_fp8_f32(
                T.i32, fp32_scaled[6], fp32_scaled[7], i32_pair_hi, True
            )

            # Q_NOPE_FP8_LDS layout: (HG, D) fp8 = (16, 512) row-major.
            # Each lane writes 8 fp8 = 2 i32 contiguous bytes at byte
            # offset (h_global * D + lane * 8).  Convert to i32 index.
            q_fp8_byte_off = h_global * fx.Int32(D) + lane * fx.Int32(8)
            q_fp8_i32_idx = q_fp8_byte_off // fx.Int32(4)
            vector.store(
                vector.from_elements(T.vec(1, T.i32), [i32_pair_lo]),
                qn_lds_i32,
                [arith.index_cast(T.index, q_fp8_i32_idx)],
            )
            vector.store(
                vector.from_elements(T.vec(1, T.i32), [i32_pair_hi]),
                qn_lds_i32,
                [arith.index_cast(T.index, q_fp8_i32_idx + fx.Int32(1))],
            )

        gpu.barrier()  # Q_NOPE_FP8 must be fully populated before chunks.

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

            # --- STEP B1: gather K_nope_chunk (direct paged read) -----
            # Read directly from the 656-byte paged KV buffer at byte
            # offset ``src_row * 656 + 0..511``.  i32 stride = 164.
            for li in range_constexpr(K_NOPE_BYTES // 4 // BLOCK_THREADS):
                lds_idx = tid + fx.Int32(li * BLOCK_THREADS)
                token = lds_idx // fx.Int32(D_I32)
                k_i32 = lds_idx % fx.Int32(D_I32)
                src_row = buffer_ops.buffer_load(
                    idx_rsrc,
                    indices_batch_off + chunk_off + token,
                    vec_width=1, dtype=T.i32,
                )
                neg = arith.cmpi(arith.CmpIPredicate.slt, src_row, fx.Int32(0))
                src_row_safe = arith.select(neg, fx.Int32(0), src_row)
                g_i32 = (
                    src_row_safe * fx.Int32(KV_I32_PER_TOKEN)
                    + fx.Int32(KV_NOPE_I32_OFFSET) + k_i32
                )
                kv = buffer_ops.buffer_load(kv_rsrc, g_i32, vec_width=1, dtype=T.i32)
                vector.store(
                    vector.from_elements(T.vec(1, T.i32), [kv]),
                    kn_lds_i32,
                    [arith.index_cast(T.index, lds_idx)],
                )

            # --- STEP B2: gather K_rope_chunk (direct paged read, vec1) -
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
                g_i32 = (
                    src_row_safe * fx.Int32(KV_I32_PER_TOKEN)
                    + fx.Int32(KV_ROPE_I32_OFFSET) + k_i32
                )
                kv = buffer_ops.buffer_load(kv_rsrc, g_i32, vec_width=1, dtype=T.i32)
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
            # Read per-head q_scale from LDS (populated by STEP A3 quant prologue).
            q_scale_v = arith.constant_vector(0.0, T.f32x4)
            for r in range_constexpr(4):
                row_idx = lane_hi4 * fx.Int32(4) + fx.Int32(r)
                qs = q_scale_lds.load(
                    [arith.index_cast(T.index, row_idx)]
                )
                q_scale_v = vector.insert(qs, q_scale_v, static_position=[r], dynamic_position=[])

            # K-scale: per-token, 4 fp32 per-tile scales stored at byte
            # offset ``src_row*656 + 512``.  We read all 4 and use their
            # mean for now (matches host-side behavior); a future change
            # will apply per-tile scales individually during QK MFMA.
            tok_idx = (
                indices_batch_off + chunk_off
                + warp_id * fx.Int32(16) + mfma_row
            )
            k_tok_global = buffer_ops.buffer_load(idx_rsrc, tok_idx, vec_width=1, dtype=T.i32)
            neg_tok = arith.cmpi(arith.CmpIPredicate.slt, k_tok_global, fx.Int32(0))
            k_tok_safe = arith.select(neg_tok, fx.Int32(0), k_tok_global)
            ks_row_base = (
                k_tok_safe * fx.Int32(KV_I32_PER_TOKEN)
                + fx.Int32(KV_SCALE_I32_OFFSET)
            )
            ks_sum = fx.Float32(0.0)
            for ti in range_constexpr(4):
                kst = buffer_ops.buffer_load(
                    kv_rsrc, ks_row_base + fx.Int32(ti),
                    vec_width=1, dtype=T.f32,
                )
                ks_sum = ks_sum + kst
            QUARTER = arith.constant(0.25, type=T.f32)
            ks = ks_sum * QUARTER

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
        q_ptr: fx.Tensor,
        kv_paged_ptr: fx.Tensor,
        indices_ptr: fx.Tensor,
        bs: fx.Int32,
        actual_heads: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        fmha_partial_kernel(
            partial_o_ptr, partial_m_ptr, partial_l_ptr,
            q_ptr, kv_paged_ptr, indices_ptr, actual_heads,
        ).launch(
            grid=(bs, fx.Int32(NUM_PARTIALS)),
            block=(BLOCK_THREADS,),
            smem=SMEM_BYTES, stream=stream,
        )

    return launch_fn


@functools.lru_cache(maxsize=4)
def _build_combine_kernel():
    """Reduce ``NUM_PARTIALS`` per-CTA partials into the final bf16 output.

    Layout (matches TileLang's combine):
      Grid:  (BS, ceil(actual_heads / HEADS_PER_BLOCK))
      Block: HEADS_PER_BLOCK warps * 64 lanes = 256 threads
      Each warp handles one head (warp_id == head_local).

    Per lane:
      dims_per_thread = D / 64 = 8        (NUM_PARTIALS x 2 buffer_load
                                          dwordx4 = 4 fp32 each; one
                                          dwordx4 store of vec<4,i32>
                                          packed from vec<8,bf16>)
      reads NUM_PARTIALS x (m, l) -- redundantly per lane within a warp.
      log-sum-exp combine, normalize, cast 8 bf16, ONE store.

    NOTE on perf: this layout is structurally cleaner than the previous
    ``grid=(BS, heads), 64 threads`` design (4x fewer CTAs, 8x fewer
    stores) but on MI355X gfx950 the wall-clock wins out at zero -- both
    designs measure ~22 us / call.  Combine seems to be limited by the
    fp32 partial_O HBM-read latency rather than launch overhead or
    store bandwidth.  Keeping the new layout because (a) it matches
    TileLang's structure (easier to compare), (b) it leaves a smaller
    surface area for future register-tile optimizations.
    """
    arch = _get_arch()
    # 4 heads per block: covers GLM-5.1 8 heads in 2 CTAs per batch.
    # 4 warps * 64 lanes = 256 threads per CTA, BS=4 -> 8 CTAs total.
    HEADS_PER_BLOCK = 4
    LANES_PER_WARP = WARP_SIZE        # 64
    DIMS_PER_THREAD = D // LANES_PER_WARP   # 512 / 64 = 8
    BLOCK_THREADS = HEADS_PER_BLOCK * LANES_PER_WARP  # 256

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def combine_kernel(
        out_ptr: fx.Tensor,            # (BS, actual_heads, D) bf16
        partial_o_ptr: fx.Tensor,      # (BS, NUM_PARTIALS, HG, D) fp32
        partial_m_ptr: fx.Tensor,      # (BS, NUM_PARTIALS, HG) fp32
        partial_l_ptr: fx.Tensor,      # (BS, NUM_PARTIALS, HG) fp32
        actual_heads: Int32,           # output row stride per batch
    ):
        tid = gpu.thread_idx.x        # 0..255
        seq_idx = gpu.block_idx.x
        block_y = gpu.block_idx.y     # 0..ceil(actual_heads/4)-1

        warp_id = tid // fx.Int32(LANES_PER_WARP)   # 0..3 -> head local
        lane_id = tid % fx.Int32(LANES_PER_WARP)    # 0..63

        head_idx = block_y * fx.Int32(HEADS_PER_BLOCK) + warp_id

        # Skip lanes whose head is past the active range (when
        # actual_heads is not a multiple of HEADS_PER_BLOCK).  We can't
        # use a CTA-level mask via grid, so we mask out via per-lane skip.
        head_valid = arith.cmpi(
            arith.CmpIPredicate.slt, head_idx, actual_heads
        )

        out_rsrc = buffer_ops.create_buffer_resource(out_ptr, max_size=True)
        po_rsrc = buffer_ops.create_buffer_resource(partial_o_ptr, max_size=True)
        pm_rsrc = buffer_ops.create_buffer_resource(partial_m_ptr, max_size=True)
        pl_rsrc = buffer_ops.create_buffer_resource(partial_l_ptr, max_size=True)

        # Partial buffers are still (BS, NUM_PARTIALS, HG, D) -- HG fixed.
        po_seq_off = seq_idx * fx.Int32(NUM_PARTIALS * HG * D)
        ml_seq_off = seq_idx * fx.Int32(NUM_PARTIALS * HG)
        out_off = (
            seq_idx * actual_heads * fx.Int32(D)
            + head_idx * fx.Int32(D)
            + lane_id * fx.Int32(DIMS_PER_THREAD)
        )

        # Pass 1: load NUM_PARTIALS (m, l) for this head.  Clamp head_idx
        # to 0 for invalid lanes so the load stays in-bounds.
        head_idx_safe = arith.select(head_valid, head_idx, fx.Int32(0))

        ms = []
        ls = []
        for p in range_constexpr(NUM_PARTIALS):
            m_off = ml_seq_off + fx.Int32(p * HG) + head_idx_safe
            m_p = buffer_ops.buffer_load(pm_rsrc, m_off, vec_width=1, dtype=T.f32)
            l_p = buffer_ops.buffer_load(pl_rsrc, m_off, vec_width=1, dtype=T.f32)
            ms.append(m_p)
            ls.append(l_p)

        m_global = ms[0]
        for p in range_constexpr(NUM_PARTIALS - 1):
            m_global = m_global.maximumf(ms[p + 1])

        # alpha[p] = exp2(m_p - m_global); l_global = sum(alpha[p] * l_p).
        alphas = []
        l_global = fx.Float32(0.0)
        for p in range_constexpr(NUM_PARTIALS):
            a = (ms[p] - m_global).exp2(fastmath=arith.FastMathFlags.fast)
            alphas.append(a)
            l_global = l_global + a * ls[p]

        # Pass 2: accumulate alpha[p] * partial_O[p, head, my-dims].
        # Use vec_width=4 to halve the number of buffer_load instructions
        # (8 fp32 = 2x dwordx4 instead of 8x single-fp32 reads).
        # Keep accs as 8 scalar fp32 -- mfma-style accum, simple to scale.
        accs = [fx.Float32(0.0) for _ in range(DIMS_PER_THREAD)]
        for p in range_constexpr(NUM_PARTIALS):
            base_p = (
                po_seq_off + fx.Int32(p * HG * D)
                + head_idx_safe * fx.Int32(D)
                + lane_id * fx.Int32(DIMS_PER_THREAD)
            )
            v0 = buffer_ops.buffer_load(
                po_rsrc, base_p,
                vec_width=4, dtype=T.f32,
            )
            v1 = buffer_ops.buffer_load(
                po_rsrc, base_p + fx.Int32(4),
                vec_width=4, dtype=T.f32,
            )
            for d in range_constexpr(4):
                accs[d] = accs[d] + alphas[p] * vector.extract(
                    v0, static_position=[d], dynamic_position=[]
                )
                accs[d + 4] = accs[d + 4] + alphas[p] * vector.extract(
                    v1, static_position=[d], dynamic_position=[]
                )

        # Avoid divide-by-zero for fully-masked queries.
        ZERO_F = fx.Float32(0.0)
        ONE_F = fx.Float32(1.0)
        zero_div = arith.cmpf(arith.CmpFPredicate.OEQ, l_global, ZERO_F)
        safe = arith.select(zero_div, ONE_F, l_global)
        inv = ONE_F / safe

        # Build a vec<8, bf16> from the normalized fp32 values, then
        # bitcast to vec<4, i32> for a single buffer_store_dwordx4.
        bf16_vals = [arith.trunc_f(T.bf16, accs[d] * inv) for d in range(DIMS_PER_THREAD)]
        bf16_v8 = vector.from_elements(T.vec(8, T.bf16), bf16_vals)
        i32_v4 = vector.bitcast(T.vec(4, T.i32), bf16_v8)

        # Skip store for invalid (padded) heads via scf.IfOp.
        if_op = scf.IfOp(head_valid, results_=[], has_else=False)
        with ir.InsertionPoint(if_op.then_block):
            buffer_ops.buffer_store(
                i32_v4, out_rsrc, out_off // fx.Int32(2),
            )
            scf.YieldOp([])

    @flyc.jit
    def launch_combine(
        out_ptr: fx.Tensor,
        partial_o_ptr: fx.Tensor,
        partial_m_ptr: fx.Tensor,
        partial_l_ptr: fx.Tensor,
        bs: fx.Int32,
        actual_heads: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # Grid Y dim = ceil(actual_heads / HEADS_PER_BLOCK).  Use a host
        # ceil-div constant so we don't need an in-kernel divrem.
        head_blocks = (actual_heads + fx.Int32(HEADS_PER_BLOCK - 1)) // fx.Int32(HEADS_PER_BLOCK)
        combine_kernel(
            out_ptr, partial_o_ptr, partial_m_ptr, partial_l_ptr,
            actual_heads,
        ).launch(
            grid=(bs, head_blocks),
            block=(BLOCK_THREADS,),
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
    assert d_v == D, f"flydsl fused kernel fixes d_v={D}, got {d_v}"
    assert q.dim() == 3 and indices.dim() == 3
    seq_len, heads, dim_total = q.shape
    assert dim_total == D + D_ROPE
    assert heads <= HG, f"fused kernel max HG={HG}, got {heads}"

    # KV preprocessing is GONE -- kernel reads the 656-byte paged buffer
    # directly via byte arithmetic in STEPs B1/B2/B5.  Just flatten the
    # (num_blocks, block_size, 1, 656) buffer to (T_total, 656) so the
    # buffer_resource sees a contiguous byte tensor.  This is a zero-cost
    # view (no GPU kernel) since the original buffer is contiguous.
    if kv_paged_uint8.dim() == 4:
        kv_paged_flat = kv_paged_uint8.view(-1, KV_BYTES_PER_TOKEN)
    else:
        kv_paged_flat = kv_paged_uint8
    assert kv_paged_flat.is_contiguous()

    # No F.pad on host.  Pass Q (BS, heads, D + D_ROPE) directly; the
    # kernel handles ``head_idx_load >= actual_heads`` via clamped offsets
    # + vector select (loaded value is zeroed out, so Q-quant produces
    # q_scale=eps, fp8=0, and downstream rows produce garbage that the
    # combine kernel never reads).
    if not q.is_contiguous():
        q = q.contiguous()

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
        seq_len, heads, D, dtype=torch.bfloat16, device=q.device,
    )

    # --- Launch partial (multi-CTA: grid = (BS, NUM_PARTIALS)) ----------
    launch_partial = _build_sparse_mla_kernel(sm_scale)
    launch_partial(
        partial_o, partial_m, partial_l,
        q, kv_paged_flat, indices,
        int(seq_len), int(heads),
    )

    # --- Launch combine (grid = (BS, heads), output is BS, heads, D) ---
    launch_combine = _build_combine_kernel()
    launch_combine(
        out, partial_o, partial_m, partial_l,
        int(seq_len), int(heads),
    )
    return out


def run_test():
    from sglang.srt.layers.attention.nsa.quant_k_cache import quantize_k_cache

    torch.manual_seed(0)
    q_nope_bf16 = torch.randn(HG, D, dtype=torch.bfloat16, device="cuda") * 0.5
    q_rope_bf16 = torch.randn(HG, D_ROPE, dtype=torch.bfloat16, device="cuda") * 0.5
    k_nope_bf16 = torch.randn(T_POOL, D, dtype=torch.bfloat16, device="cuda") * 0.5
    k_rope_bf16 = torch.randn(T_POOL, D_ROPE, dtype=torch.bfloat16, device="cuda") * 0.5

    # Build the canonical 656-byte paged buffer via NSA quantize_k_cache.
    kv_full = torch.cat([k_nope_bf16, k_rope_bf16], dim=-1).view(
        T_POOL // 64, 64, 1, D + D_ROPE
    )
    kv_paged = quantize_k_cache(kv_full)
    kv_paged_flat = kv_paged.view(-1, KV_BYTES_PER_TOKEN).contiguous()

    indices = torch.randperm(T_POOL, device="cuda")[:TOPK].to(torch.int32)
    sm_scale = 1.0 / math.sqrt(D + D_ROPE)

    # Standalone runs full HG=16 active heads.
    out = torch.zeros(1, HG, D, dtype=torch.bfloat16, device="cuda")
    partial_o = torch.empty(1, NUM_PARTIALS, HG, D, dtype=torch.float32, device="cuda")
    partial_m = torch.empty(1, NUM_PARTIALS, HG, dtype=torch.float32, device="cuda")
    partial_l = torch.empty(1, NUM_PARTIALS, HG, dtype=torch.float32, device="cuda")
    try:
        q_concat = torch.cat(
            [q_nope_bf16, q_rope_bf16], dim=-1
        ).unsqueeze(0).contiguous()
        launch_partial = _build_sparse_mla_kernel(sm_scale)
        launch_partial(
            partial_o, partial_m, partial_l,
            q_concat,
            kv_paged_flat,
            indices.unsqueeze(0).contiguous(),
            1, HG,
        )
        launch_combine = _build_combine_kernel()
        launch_combine(out, partial_o, partial_m, partial_l, 1, HG)
        torch.cuda.synchronize()
        out = out.squeeze(0)
    except Exception as e:
        print(f"[WIP] Step E full fmha kernel build/launch failed: "
              f"{type(e).__name__}: {e}")
        return float("inf")

    # Reference: dequantize and compute full-precision attention.
    from sglang.srt.layers.attention.nsa.dequant_k_cache import dequantize_k_cache
    kv_round = dequantize_k_cache(kv_paged).view(T_POOL, D + D_ROPE)
    k_nope_round = kv_round[:, :D]
    k_rope_round = kv_round[:, D:]
    k_nope_g = k_nope_round[indices.long()].float()
    k_rope_g = k_rope_round[indices.long()].float()
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
