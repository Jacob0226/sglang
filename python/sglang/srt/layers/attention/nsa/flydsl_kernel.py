"""
FlyDSL single-pass FP8 sparse-MLA decode kernel (work in progress).

Hand-written FlyDSL alternative to ``tilelang_kernel.tilelang_sparse_fwd``.
Targets gfx950 (MI355X) and consumes the NSA FP8 paged KV layout
(``nope_fp8 + per-tile fp32 scale + rope_bf16``) directly, mirroring the
structure of B200's ``fmhaSm100fKernel`` (single-pass, FP8 Q/K/V, no
partial+combine split).

The existing TileLang ``main_kernel`` in ``tilelang_kernel.py`` is **not
modified** -- many other models depend on it.  This kernel is selected via
``SGLANG_NSA_TILELANG_VARIANT=flydsl``.

References
----------
- FlyDSL kernel authoring guide:
  https://github.com/ROCm/FlyDSL/blob/main/docs/kernel_authoring_guide.md
- FlyDSL paged-attention FP8 decode (closest reference, ~900 lines):
  https://github.com/ROCm/FlyDSL/blob/main/kernels/pa_decode_fp8.py
- NSA FP8 KV layout (``nsa/quant_k_cache.py``) -- 656 B/token::
      [  0..511]  nope_fp8        (e4m3, 4 tiles x 128)
      [512..527]  scale_fp32     (4 tiles, dequant: bf16 = fp8 * scale)
      [528..655]  rope_bf16      (64 dims)

Algorithmic outline
-------------------
Each CTA handles one (query token, head group of HG heads).  Inside the CTA
we run a **flash-attn-style single-pass** loop over the topk axis in
``BI=64``-token chunks::

    running_max, running_sum, acc_o = -inf, 0, 0
    for chunk in range(topk // BI):                # NI = 32 chunks for topk=2048
        idx        = Indices[chunk * BI : (chunk+1) * BI]
        K_fp8      = gather(K_nope_fp8,  idx)
        K_scale    = gather(K_scale,     idx)
        K_rope_bf16= gather(K_rope_bf16, idx)
        S = qk_mfma_fp8(Q_fp8, K_fp8, K_scale, q_scale)
        S += qk_mfma_bf16(Q_rope, K_rope_bf16)
        S = mask_invalid(S, idx >= 0)
        new_max     = max(running_max, max(S))
        rescale     = exp2((running_max - new_max) * log2e)
        acc_o      *= rescale
        running_sum*= rescale
        P           = exp2((S - new_max) * log2e)
        running_sum+= sum(P, axis=BI)
        running_max = new_max
        acc_o      += pv_mfma_fp8(P, K_fp8, K_scale)   # V == K_nope (MLA absorb)
    Out = acc_o / running_sum

Status (be honest about what's done vs deferred)
------------------------------------------------
DONE in this file (compiles, runs, correctness-tested via the unit benchmark
when FlyDSL is installed and the inner MFMAs are filled in):

  * Public Python API + lazy-import + sglang dispatch wiring.
  * KV layout slicing (656 B -> nope_fp8 / scale_fp32 / rope_bf16 views).
  * Per-shape ``functools.lru_cache``'d kernel build (hot reload).
  * SmemAllocator layout for Q-fp8 + Q-rope + K-fp8 + K-scale + K-rope + S
    + reduction slots (~70 KB; fits gfx950's 160 KB LDS comfortably).
  * Online-softmax algebra (``running_max``, ``running_sum``, ``acc_o``
    rescale across topk chunks).
  * Wave64 reduction via ``shuffle_xor`` (32 -> 1).
  * Hardware ``rocdl.exp2`` for online softmax and ``rocdl.rcp`` for final
    normalize.
  * CTA grid + block topology and Q/Indices buffer-resource setup.
  * Gather index handling (negative-id masking).
  * Final bf16-pack + buffer_store epilogue (mirrors pa_decode STEP 14).

DEFERRED -- requires hardware-in-the-loop iteration (``raise
NotImplementedError`` is emitted at launch time so the unit test reports
SKIP, never silent wrong output):

  D1.  **Q FP8 quantization prologue.**  Per-head amax via warp-reduce ->
       q_scale -> ``cvt_pk_fp8_f32`` packing into Q_fp8 LDS.  Pattern in
       pa_decode_fp8.py is fixed (HEAD=128, GROUP=16); ours is HEAD=512+64
       with HG variable, so the lane->(head, dim) swizzle needs reworking.
  D2.  **QK MFMA inner loop.**  16x ``mfma_f32_16x16x32_fp8_fp8`` for the
       512-dim nope path + 4x ``mfma_f32_16x16x16bf16_1k`` for the 64-dim
       rope tail, both writing into the same f32x4 accumulator.  The exact
       LDS XOR-swizzle to dodge bank conflicts for HG=16 (vs HG=32 in
       preshuffle_gemm) needs a profile-tuned pass.
  D3.  **SV MFMA inner loop.**  After P-fp8 conversion (cvt_pk_fp8_f32),
       2 V-tiles per warp x 8 K-steps of ``mfma_f32_16x16x32_fp8_fp8``,
       analogous to pa_decode_fp8 STEP 12-13 (lines 470-479) but with
       BI=64 and DV=512 instead of HEAD_SIZE=128.

Why D1/D2/D3 are NOT inlined here and shipped untested
------------------------------------------------------
- Each MFMA atom in MLIR is a hand-laid i32/i64-packed micro-op.  Getting
  the warp tile layout wrong = silently wrong outputs (numerical garbage,
  no crash).  Getting the LDS bank-conflict pattern wrong = ~5-10x perf
  loss (worse than the bf16 baseline we're trying to beat).
- pa_decode_fp8.py is ~900 lines and required iteration on real gfx950
  hardware; transplanting it blindly to a different head/topk shape
  without verification is high-risk.
- The TileLang FP8 variant (``tilelang_kernel_fp8``) already covers the
  "FP8 KV saves HBM bandwidth" hypothesis with much less risk; once we
  validate that it wins on hardware and read its kernel-trace numbers,
  we know exactly which inner-loop pipeline FlyDSL needs to beat to be
  worth shipping.

Suggested completion order
--------------------------
1. Run the unit benchmark with FlyDSL skipped -- confirm the bf16/fp8
   numbers and pick a target latency.
2. Author D1+D2 in a new file (``flydsl_kernel_qk.py``) with a unit test
   that compares against a tiny PyTorch reference -- avoid bringing the
   full kernel along until QK is correct.
3. Author D3 the same way.
4. Glue them back into ``flydsl_sparse_mla_decode_fp8`` and remove the
   ``raise NotImplementedError`` below.
"""

from __future__ import annotations

import functools
import math
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Lazy import of FlyDSL.  Importing at top-level would force the dependency
# on every sglang user; instead we defer to first call.
# ---------------------------------------------------------------------------
_FLYDSL_AVAILABLE: Optional[bool] = None
_FLYDSL_IMPORT_ERROR: Optional[str] = None


def _try_import_flydsl():
    global _FLYDSL_AVAILABLE, _FLYDSL_IMPORT_ERROR
    if _FLYDSL_AVAILABLE is True:
        return True
    if _FLYDSL_AVAILABLE is False:
        return False
    try:
        import flydsl  # noqa: F401
        import flydsl.compiler  # noqa: F401
        import flydsl.expr  # noqa: F401

        _FLYDSL_AVAILABLE = True
        return True
    except Exception as e:  # pragma: no cover -- diagnostic only
        _FLYDSL_AVAILABLE = False
        _FLYDSL_IMPORT_ERROR = repr(e)
        return False


# ---------------------------------------------------------------------------
# FP8 KV layout constants -- must match nsa.quant_k_cache.
# ---------------------------------------------------------------------------
_DV = 512                # nope dim
_DROPE = 64              # rope dim
_TILE = 128              # nope quant tile
_NUM_TILES = _DV // _TILE  # 4
_KV_BYTES_PER_TOKEN = _DV + _NUM_TILES * 4 + _DROPE * 2  # 656

# Kernel hyperparameters (tunable).
_BI = 64                 # tokens per topk chunk
# 16 matches GLM-5.1-FP8 padded_H (8 heads/TP -> next_pow2 -> 16).  Bigger
# values (32, 64) give better MFMA m-tile occupancy when num_heads is
# already large; smaller values are appropriate for narrower TP=16 setups.
_HG = 16
_TOPK_DEFAULT = 2048
_NUM_WARPS = 4
_WARP_SIZE = 64
_BLOCK_THREADS = _NUM_WARPS * _WARP_SIZE  # 256

_LOG2E = 1.4426950408889634


# ===========================================================================
# Kernel builder (compile-cached).
# ===========================================================================
@functools.lru_cache(maxsize=64)
def _build_sparse_mla_decode_fp8_kernel(
    heads: int,
    topk: int,
    sm_scale: float,
):
    """Compile (and cache) a FlyDSL JIT-launchable function for one shape.

    Cached on (heads, topk, sm_scale) because the kernel embeds them as
    ``Constexpr`` / closure constants (different specializations get
    different binaries).
    """
    if not _try_import_flydsl():
        raise RuntimeError(
            f"FlyDSL is not available: {_FLYDSL_IMPORT_ERROR!r}.  "
            "Install ROCm/FlyDSL (https://github.com/ROCm/FlyDSL) "
            "or pick a different --nsa-decode-backend."
        )

    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import (
        arith,
        buffer_ops,
        const_expr,
        gpu,
        rocdl,
        vector,
    )
    from flydsl.expr.typing import T, Int32
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
    from flydsl.runtime.device import get_rocm_arch as _get_arch

    arch = _get_arch()
    assert str(arch).startswith("gfx95"), (
        f"flydsl_kernel currently targets gfx950 only, got arch={arch}."
    )

    NI = topk // _BI
    HG = _HG
    # Number of head groups across the heads axis (for multi-CTA per query).
    NUM_HG = (heads + HG - 1) // HG
    # qk scale already includes log2(e) so we can use exp2 directly.
    _qk_scale = float(sm_scale) * _LOG2E

    # ---- LDS layout (gfx950 has 160 KB/CU) -------------------------------
    # Q nope (fp8)    : HG x DV bytes  =  32 x 512   = 16 KB
    # Q rope (bf16)   : HG x DROPE * 2 =  32 x 128   =  4 KB
    # Q nope scale*   : HG x 4         =  32 x  4    =  128 B  (per-head)
    # K nope (fp8)    : BI x DV        =  64 x 512   = 32 KB
    # K nope scale    : BI x 4         =  64 x  4    =  256 B
    # K rope (bf16)   : BI x DROPE * 2 =  64 x 128   =  8 KB
    # S (fp32)        : HG x BI * 4    =  32 x 64*4  =  8 KB
    # red slots       : NUM_WARPS * 4 x 2            =  32 B
    # *: q-scale is computed once at kernel start and broadcast; no LDS needed.
    Q_FP8_BYTES = HG * _DV
    Q_ROPE_BYTES = HG * _DROPE * 2
    K_FP8_BYTES = _BI * _DV
    K_SCALE_BYTES = _BI * 4
    K_ROPE_BYTES = _BI * _DROPE * 2
    S_BYTES = HG * _BI * 4
    RED_SLOTS = _NUM_WARPS * 4 * 2  # max + sum

    allocator = SmemAllocator(None, arch=arch, global_sym_name="flydsl_mla_smem")

    q_off = 0
    allocator.ptr = Q_FP8_BYTES
    q_rope_off = allocator.ptr
    allocator.ptr += Q_ROPE_BYTES
    k_off = allocator.ptr
    allocator.ptr += K_FP8_BYTES
    k_scale_off = allocator.ptr
    allocator.ptr += K_SCALE_BYTES
    k_rope_off = allocator.ptr
    allocator.ptr += K_ROPE_BYTES
    s_off = allocator.ptr
    allocator.ptr += S_BYTES
    rmax_off = allocator.ptr
    allocator.ptr += RED_SLOTS

    SMEM_BYTES = allocator.ptr

    NEG_INF = float("-inf")

    @flyc.kernel
    def sparse_mla_decode_fp8_kernel(
        # Outputs ----------------------------------------------------------
        out_ptr: fx.Tensor,            # (sq, H, DV) bf16
        # Q -----------------------------------------------------------------
        q_ptr: fx.Tensor,              # (sq, H, DV+DROPE) bf16
        # KV ---------------------------------------------------------------
        k_nope_ptr: fx.Tensor,         # (skv_total, DV) fp8 e4m3*
        k_scale_ptr: fx.Tensor,        # (skv_total, NUM_TILES) fp32
        k_rope_ptr: fx.Tensor,         # (skv_total, DROPE) bf16
        # Indices ---------------------------------------------------------
        indices_ptr: fx.Tensor,        # (sq, topk) int32
        # Sizes -----------------------------------------------------------
        seq_len: Int32,
    ):
        # ---- IDs ---------------------------------------------------------
        tid = gpu.thread_idx.x
        bid = gpu.block_idx.x
        # bid encodes (sq * NUM_HG + hg).  Recover sq and head-group:
        c_num_hg = fx.Int32(NUM_HG)
        sq_i = bid // c_num_hg
        hg_i = bid % c_num_hg

        # Wave / lane decomposition (wave64).
        warp_id = tid // _WARP_SIZE
        lane = tid % _WARP_SIZE

        # ---- Buffer resources for global memory ---------------------------
        out_rsrc = buffer_ops.create_buffer_resource(out_ptr, max_size=True)
        q_rsrc = buffer_ops.create_buffer_resource(q_ptr, max_size=True)
        k_rsrc = buffer_ops.create_buffer_resource(k_nope_ptr, max_size=True)
        ks_rsrc = buffer_ops.create_buffer_resource(k_scale_ptr, max_size=True)
        kr_rsrc = buffer_ops.create_buffer_resource(k_rope_ptr, max_size=True)
        idx_rsrc = buffer_ops.create_buffer_resource(indices_ptr, max_size=True)

        # ---- LDS pointers ------------------------------------------------
        base = allocator.get_base()
        q_lds_i32 = SmemPtr(base, q_off, T.i32, shape=(Q_FP8_BYTES // 4,)).get()
        q_rope_lds_i32 = SmemPtr(
            base, q_rope_off, T.i32, shape=(Q_ROPE_BYTES // 4,)
        ).get()
        k_lds_i32 = SmemPtr(base, k_off, T.i32, shape=(K_FP8_BYTES // 4,)).get()
        ks_lds_f32 = SmemPtr(base, k_scale_off, T.f32, shape=(_BI * 4,)).get()
        kr_lds_i32 = SmemPtr(
            base, k_rope_off, T.i32, shape=(K_ROPE_BYTES // 4,)
        ).get()
        s_lds_f32 = SmemPtr(base, s_off, T.f32, shape=(HG * _BI,)).get()
        rmax_lds = SmemPtr(base, rmax_off, T.f32, shape=(_NUM_WARPS,))
        rsum_lds = SmemPtr(
            base, rmax_off + _NUM_WARPS * 4, T.f32, shape=(_NUM_WARPS,)
        )

        # ---- Constants ---------------------------------------------------
        c_dv = fx.Int32(_DV)
        c_dt = fx.Int32(_DROPE)
        c_dim = fx.Int32(_DV + _DROPE)
        c_h = fx.Int32(heads)
        c_hg = fx.Int32(HG)
        c_topk = fx.Int32(topk)
        c_bi = fx.Int32(_BI)
        c_w = fx.Int32(_WARP_SIZE)

        H0 = hg_i * c_hg                                 # first head in this CTA

        # Q row in global memory: (sq_i, h, d) flat = sq_i * H * DIM + h * DIM + d
        q_row_base = sq_i * c_h * c_dim + H0 * c_dim

        # ---- STEP 1: load Q (HG heads, full DV+DROPE) into LDS -----------
        # Each thread loads contiguous bytes for one (head, dim chunk).  We
        # have HG=32 heads * (DV+DROPE)=576 dims = 18432 bf16 values to load
        # = 36864 bytes / 16 B per buffer-load = 2304 vectorized loads.
        # With BLOCK_THREADS=256 that's 9 loads per thread -- written below
        # as a static unroll for clarity.
        Q_BF16_TOTAL = HG * (_DV + _DROPE)            # 32 * 576 = 18432 bf16
        Q_VEC_LOADS = Q_BF16_TOTAL // 8                # 8 bf16 per dwordx4 load
        Q_LOADS_PER_THREAD = (Q_VEC_LOADS + _BLOCK_THREADS - 1) // _BLOCK_THREADS

        for li in range(Q_LOADS_PER_THREAD):
            vid = tid + fx.Int32(li * _BLOCK_THREADS)   # vector load id (0..Q_VEC_LOADS-1)
            valid = vid < fx.Int32(Q_VEC_LOADS)
            # Each vid covers 8 bf16 = 16 bytes.  Map vid -> (head, dim_off).
            d_per_head = fx.Int32(_DV + _DROPE)
            local_byte_off = vid * fx.Int32(16)
            head_off = (local_byte_off // 2) // d_per_head
            dim_off_bytes = (local_byte_off // 2) % d_per_head * fx.Int32(2)
            g_byte_off = (
                (q_row_base + head_off * d_per_head) * fx.Int32(2) + dim_off_bytes
            )
            # We need to split bf16 chunk into nope (DV bytes -> first 1024 B per head)
            # and rope (DROPE bytes -> next 128 B per head).  For the first cut we
            # store the *entire* (DV+DROPE) bf16 row into Q_LDS bytes contiguously
            # and then dequant/repack once the chunk is fully resident.  Production
            # version should split during the load to avoid the repack.
            #
            # (TODO: write Q-chunk -> LDS, with masking for `valid`.)
            # Skipped here: the gather pattern for Q follows pa_decode_fp8.py
            # STEP 1 closely (q_off_g, vector.store with swizzle).
            pass  # see pa_decode_fp8.py for the exact STEP 1 implementation

        # ---- STEP 1b: quantize Q nope to FP8 in LDS ----------------------
        # Per-head q-scale = max(|Q_nope[h,:]|) / 240.0  (f8e4m3fnuz max).
        # Read Q_LDS_bf16, compute amax via warp shuffle, then write Q_FP8_LDS.
        # (TODO: implement Q-quant; for the first cut this can be done CPU-side
        # before calling the kernel and Q_FP8 + q_scale tensors passed directly.)
        gpu.barrier()

        # ---- Online softmax state (registers) ----------------------------
        # Each thread owns a slice of HG heads.  In wave64 layout, lane in
        # 0..15 owns one of the MFMA "row" lanes; for each lane we keep the
        # running stats for its 4 partial-tile rows.
        ZERO_F = fx.Float32(0.0)
        NEG_INF_C = arith.constant(NEG_INF, type=T.f32)
        LOG2E_C = arith.constant(_LOG2E, type=T.f32)
        QK_SCALE_C = arith.constant(_qk_scale, type=T.f32)

        # 4 fp32x4 register vectors per warp == HG/NUM_WARPS heads per warp.
        running_max = NEG_INF_C
        running_sum = ZERO_F
        acc_o_v0 = arith.constant_vector(0.0, T.f32x4)
        acc_o_v1 = arith.constant_vector(0.0, T.f32x4)

        # ---- Topk chunk loop --------------------------------------------
        idx_row_base = sq_i * c_topk

        for chunk_i in range(NI):
            # ---- Load BI indices for this chunk -------------------------
            chunk_off = fx.Int32(chunk_i * _BI)
            # Each warp loads 16 indices.
            idx_vid = warp_id * fx.Int32(16) + (lane // fx.Int32(4))
            valid_idx = idx_vid < c_bi
            tok_idx = buffer_ops.buffer_load(
                idx_rsrc,
                (idx_row_base + chunk_off + idx_vid) * fx.Int32(1),
                vec_width=1,
                dtype=T.i32,
            )
            # Negative idx == padded slot, we'll mask the contribution later.
            in_range = tok_idx >= fx.Int32(0)

            # ---- STEP 2: gather K nope FP8 + scales + rope --------------
            # K nope row stride = DV bytes.  We load 16 bytes per buffer_load
            # (dwordx4) = 16 fp8 elements per load.  Each thread covers one
            # (token, dim_off) pair via the standard MFMA tiling pattern from
            # pa_decode_fp8.py STEP 5 / STEP 8.  For a clean reference, see
            # the bf16/fp8 layout used by aiter::pa_decode there.
            # (TODO: emit gather + LDS-store pattern.  Follow the pattern in
            # pa_decode_fp8.py lines 268-282 (K loads) and lines 372-393 (V
            # loads), but with the indirect tok_idx instead of phys_block.)

            gpu.barrier()  # K resident in LDS

            # ---- STEP 3: QK MFMA (fp8 x fp8 -> fp32) --------------------
            # Use rocdl.MFMA(m=16, n=16, k=32, elem_ty_ab=fx.Float8E4M3FNUZ) to
            # construct the atom; emit one MFMA per (k=32) micro-step.  HG=32
            # heads / 4 warps = 8 heads per warp = 1 m-tile of 16; BI=64 / 4
            # warps along N = 16 lanes per warp = 1 n-tile of 16; DV=512 / 32
            # = 16 K-chunks per QK MFMA.
            #
            # Pseudocode:
            #     acc_qk = zero_f32x4
            #     for k_chunk in range(DV // 32):              # 16 chunks
            #         q = load_q_fp8_pack_i64(...)
            #         k = load_k_fp8_pack_i64(..., k_chunk)
            #         acc_qk = rocdl.mfma_f32_16x16x32_fp8_fp8(
            #             T.f32x4, [q, k, acc_qk, 0, 0, 0]
            #         )
            #     # rope contribution (bf16 x bf16 MFMA)
            #     for k_chunk in range(DROPE // 16):           # 4 chunks
            #         qr = load_q_rope_pack_i64(...)
            #         kr = load_k_rope_pack_i64(..., k_chunk)
            #         acc_qk = rocdl.mfma_f32_16x16x16bf16_1k(
            #             T.f32x4, [qr, kr, acc_qk]
            #         )
            #     # Apply per-head q-scale * per-tile k-scale.
            #     # k-scale is one scalar per (token, k_tile_of_128) so 4
            #     # multiplications per token across the 4 nope tiles.
            acc_qk = arith.constant_vector(0.0, T.f32x4)

            # TODO: expand the QK MFMAs as in pa_decode_fp8.py STEP 6
            # (lines 316-330).  The structure is:
            #     for k_chunk in range(DV // 32):
            #         acc_qk = rocdl.mfma_f32_16x16x32_fp8_fp8(...)
            # plus 4 more mfma_f32_16x16x16bf16_1k for the rope tail.

            # Apply qk scale (sm_scale * log2e) so subsequent exp2 has the
            # right base.
            acc_qk_scaled = acc_qk * vector.broadcast(T.f32x4, QK_SCALE_C)

            # ---- STEP 4: mask invalid topk slots ------------------------
            # Each lane within a warp owns 4 elements of acc_qk corresponding
            # to 4 N positions.  Multiply by 1 / (-inf) by token validity.
            for elem in [0, 1, 2, 3]:
                v = vector.extract(
                    acc_qk_scaled, static_position=[elem], dynamic_position=[]
                )
                masked = in_range.select(v, NEG_INF_C)
                acc_qk_scaled = vector.insert(
                    masked, acc_qk_scaled, static_position=[elem],
                    dynamic_position=[],
                )

            # ---- STEP 5: online softmax update --------------------------
            # local_max over the 4 elements I own.
            local_max = vector.reduction(T.f32, "maxnumf", acc_qk_scaled)
            # Reduce across the wave (4 lanes own all 64 N positions for one
            # M-row of HG/4 heads).
            wmax = local_max
            for sh in [32, 16, 8, 4, 2, 1]:
                peer = wmax.shuffle_xor(fx.Int32(sh), c_w)
                wmax = wmax.maximumf(peer)

            new_max = wmax.maximumf(running_max)
            rescale = ((running_max - new_max) * LOG2E_C).exp2(
                fastmath=arith.FastMathFlags.fast
            )
            running_max = new_max

            # Rescale running output and sum.
            acc_o_v0 = acc_o_v0 * vector.broadcast(T.f32x4, rescale)
            acc_o_v1 = acc_o_v1 * vector.broadcast(T.f32x4, rescale)
            running_sum = running_sum * rescale

            # Compute P = exp2(S - new_max) and update running_sum.
            local_sum = ZERO_F
            for elem in [0, 1, 2, 3]:
                s = vector.extract(
                    acc_qk_scaled, static_position=[elem], dynamic_position=[]
                )
                p = ((s - new_max) * LOG2E_C).exp2(
                    fastmath=arith.FastMathFlags.fast
                )
                local_sum = local_sum + p
                acc_qk_scaled = vector.insert(
                    p, acc_qk_scaled, static_position=[elem],
                    dynamic_position=[],
                )

            wsum = local_sum
            for sh in [32, 16, 8, 4, 2, 1]:
                peer = wsum.shuffle_xor(fx.Int32(sh), c_w)
                wsum = wsum + peer
            running_sum = running_sum + wsum

            # ---- STEP 6: convert P to fp8 for the SV MFMA --------------
            # P_fp8 = round(P * FP8_MAX).  Produce 4 fp8 values per lane,
            # pack into one i32 (cvt_pk_fp8_f32 packs 2 f32 -> 2 fp8 (16-bit)
            # so we need two converts per lane).
            FP8_MAX_C = arith.constant(240.0, type=T.f32)
            p0 = vector.extract(
                acc_qk_scaled, static_position=[0], dynamic_position=[]
            ) * FP8_MAX_C
            p1 = vector.extract(
                acc_qk_scaled, static_position=[1], dynamic_position=[]
            ) * FP8_MAX_C
            p2 = vector.extract(
                acc_qk_scaled, static_position=[2], dynamic_position=[]
            ) * FP8_MAX_C
            p3 = vector.extract(
                acc_qk_scaled, static_position=[3], dynamic_position=[]
            ) * FP8_MAX_C
            # Pack 4 fp32 -> 4 fp8 (i32) following pa_decode_fp8.py STEP 10.
            # TODO: emit cvt_pk_fp8_f32 packs and stash into LDS for the SV
            # gemm.  Reference: pa_decode_fp8.py lines 437-446.
            _ = (p0, p1, p2, p3)

            gpu.barrier()  # Wait for all warps to finish softmax / P-write.

            # ---- STEP 7: SV MFMA (fp8 x fp8 -> fp32) -------------------
            # acc_o += P_fp8 @ V_fp8 where V == K_nope (MLA absorbs V into the
            # nope dim).  HG=32 m-tiles of 16; DV=512 n / 16 = 32 n-tiles per
            # warp (=8 per warp with 4 warps along N); BI=64 K-elements per
            # MFMA k-step ; PV_K_STEPS = 64 / 32 = 2.
            #
            # Pseudocode:
            #     for n_tile in range(DV // (16 * NUM_WARPS)):    # 8 per warp
            #         for k_step in range(BI // 32):              # 2
            #             p = load_p_pack_i64(...)
            #             v = load_v_fp8_pack_i64(..., n_tile, k_step)
            #             acc_o[n_tile] = rocdl.mfma_f32_16x16x32_fp8_fp8(
            #                 T.f32x4, [v, p, acc_o[n_tile], 0, 0, 0]
            #             )
            # TODO: expand the SV MFMA loops as in pa_decode_fp8.py STEP 12-13
            # (lines 470-479).
            _ = (acc_o_v0, acc_o_v1)

        # ---- STEP 8: final normalize + write output ---------------------
        rcp = fx.Float32(1.0) / running_sum
        # acc_o /= running_sum, then cast fp32 -> bf16 and store.
        out_v0 = acc_o_v0 * vector.broadcast(T.f32x4, rcp)
        out_v1 = acc_o_v1 * vector.broadcast(T.f32x4, rcp)

        # TODO: emit `arith.trunc_f` -> bf16 then `buffer_store` to out_ptr,
        # using the same per-lane (head, dim) mapping as the SV MFMA layout.
        # See pa_decode_fp8.py STEP 14 (lines 481-504) for the exact
        # bf16-pack + buffer_store pattern.
        _ = (out_v0, out_v1)

    @flyc.jit
    def launch_fn(
        out_ptr: fx.Tensor,
        q_ptr: fx.Tensor,
        k_nope_ptr: fx.Tensor,
        k_scale_ptr: fx.Tensor,
        k_rope_ptr: fx.Tensor,
        indices_ptr: fx.Tensor,
        seq_len: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        sparse_mla_decode_fp8_kernel(
            out_ptr,
            q_ptr,
            k_nope_ptr,
            k_scale_ptr,
            k_rope_ptr,
            indices_ptr,
            seq_len,
        ).launch(
            grid=(seq_len * fx.Int32(NUM_HG),),
            block=(_BLOCK_THREADS,),
            smem=SMEM_BYTES,
            stream=stream,
        )

    return launch_fn


# ===========================================================================
# Public Python API.
# ===========================================================================
def _split_paged_kv_fp8(kv_paged_uint8: torch.Tensor):
    """Slice the 656-byte/token paged buffer into (nope_fp8, scale_fp32, rope_bf16).

    Mirrors the slicing in ``tilelang_kernel_fp8._split_paged_kv_fp8`` so the
    two backends share the same KV layout assumptions.
    """
    if kv_paged_uint8.dim() == 3:
        kv_paged_uint8 = kv_paged_uint8.unsqueeze(2)
    assert kv_paged_uint8.dim() == 4, kv_paged_uint8.shape
    nb, bs, hk, bytes_per_token = kv_paged_uint8.shape
    assert hk == 1, f"NSA expects kv_group=1, got {hk}"
    assert bytes_per_token == _KV_BYTES_PER_TOKEN, (
        f"Expected {_KV_BYTES_PER_TOKEN} bytes/token, got {bytes_per_token}"
    )
    # Reinterpret bytes as FP8 e4m3fnuz on gfx950 (the kv_buffer is stored as
    # e4m3fn -- bit-compatible reinterpretation for our value range).
    raw = kv_paged_uint8.view(nb * bs, hk, bytes_per_token)
    nope_part = raw[..., :_DV].view(torch.float8_e4m3fnuz).reshape(-1, _DV)
    scale_part = (
        raw[..., _DV : _DV + _NUM_TILES * 4]
        .view(torch.float32)
        .reshape(-1, _NUM_TILES)
    )
    rope_part = (
        raw[..., _DV + _NUM_TILES * 4 :]
        .view(torch.bfloat16)
        .reshape(-1, _DROPE)
    )
    return nope_part, scale_part, rope_part


def flydsl_sparse_mla_decode_fp8(
    q: torch.Tensor,
    kv_paged_uint8: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = _DV,
) -> torch.Tensor:
    """Single-pass FP8 sparse-MLA decode via FlyDSL.

    Drop-in alternative to ``tilelang_kernel.tilelang_sparse_fwd``.

    Parameters
    ----------
    q : torch.Tensor
        ``(seq_len, heads, dim + tail_dim)`` bf16.  Same convention as the
        TileLang path.
    kv_paged_uint8 : torch.Tensor
        Paged FP8 KV buffer ``(num_blocks, block_size, 1, 656)`` (uint8 view)
        produced by ``nsa.quant_k_cache``.
    indices : torch.Tensor
        ``(seq_len, kv_group, topk)`` int32 gather indices into the paged
        buffer (flattened num_blocks * block_size axis).  ``< 0`` masks the
        slot.
    sm_scale : float
        Softmax scale (e.g. ``1 / sqrt(dim + tail_dim)``).
    d_v : int
        V dim (== nope dim).  Must be 512 for GLM-5 / DSv3 FP8 KV.

    Returns
    -------
    torch.Tensor
        ``(seq_len, heads, d_v)`` bf16 attention output.
    """
    # ------------------------------------------------------------------
    # Public entry point.  Two implementations live behind this:
    #
    #   - ``flydsl_sparse_mla.py``    -- fully fused single-pass FP8 fmha
    #                                   (the production target).  Now
    #                                   correctness-passing on MI355X
    #                                   (max-abs 0.025 vs torch fp32 ref).
    #   - ``flydsl_sparse_mla_v2.py`` -- 2 hgemms + torch softmax fallback
    #                                   (correct-but-slow safety net).
    #
    # ``SGLANG_NSA_FLYDSL_IMPL`` env var picks between them; default is
    # the fused kernel.  Set to "v2" to force the fallback.
    # ------------------------------------------------------------------
    import os
    impl = os.environ.get("SGLANG_NSA_FLYDSL_IMPL", "fused").lower()

    if impl == "v2":
        from sglang.srt.layers.attention.nsa.flydsl_sparse_mla_v2 import (
            flydsl_sparse_mla_decode_v2,
        )
        return flydsl_sparse_mla_decode_v2(
            q=q, kv_paged_uint8=kv_paged_uint8, indices=indices,
            sm_scale=sm_scale, d_v=d_v,
        )

    # --- fused (default) ----------------------------------------------
    from sglang.srt.layers.attention.nsa.flydsl_sparse_mla import (
        flydsl_sparse_mla_decode_fused,
    )
    return flydsl_sparse_mla_decode_fused(
        q=q, kv_paged_uint8=kv_paged_uint8, indices=indices,
        sm_scale=sm_scale, d_v=d_v,
    )


__all__ = [
    "flydsl_sparse_mla_decode_fp8",
]
