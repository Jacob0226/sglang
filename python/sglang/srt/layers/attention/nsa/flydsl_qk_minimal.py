"""
Step 1 of the FlyDSL sparse-MLA decode kernel: a **minimal bf16 QK GEMM**.

Goal
----
Validate that the FlyDSL MFMA + LDS + warp-tile machinery works on the
target gfx950 hardware **before** layering on FP8, indirect gather, online
softmax, and SV.  This file deliberately does NONE of those -- it only
proves that we can author a tile-level QK GEMM in FlyDSL and get a
numerically correct result.

Shapes (all small, fixed)
-------------------------
- Q : (M=16, K=128) bf16     -- one MFMA m-tile
- K : (N=64, K=128) bf16     -- four MFMA n-tiles of 16
- S : (M=16, N=64) fp32      -- raw QK product (no scale, no softmax)

Algorithm
---------
- Single CTA, 256 threads (4 wave64).
- Each warp owns 1 N-tile of 16 tokens (so warp_id maps directly to n-tile id).
- All warps share the same M-tile of 16 rows (since M=16 is exactly one MFMA
  m-tile).
- For each warp:
    - Load Q (16 x K) into LDS (cooperative, all warps).
    - Load this warp's slice of K (16 x K) into LDS.
    - Loop k_chunk in range(K // 16):
          a = load_q_pack(k_chunk)               # 4 x bf16 = 8 bytes = i64
          b = load_k_pack(k_chunk)               # 4 x bf16 = 8 bytes = i64
          acc = mfma_f32_16x16x16bf16_1k(a, b, acc)
    - Store acc (16 x 16 fp32) back to global S[:, n_tile_off : n_tile_off+16].

Once this is correct, Step 2 swaps K to fp8 (mfma_f32_16x16x32_fp8_fp8),
Step 3 adds indirect gather, Step 4 adds online softmax + SV.

Usage
-----
    python -m sglang.srt.layers.attention.nsa.flydsl_qk_minimal
"""

from __future__ import annotations

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


# ---- Compile-time constants ----
M = 16          # one MFMA m-tile
N = 64          # 4 MFMA n-tiles of 16
K = 128         # 8 K-chunks of 16
MFMA_M = 16
MFMA_N = 16
MFMA_K = 16     # bf16 MFMA k-chunk
NUM_WARPS = 4   # one per N-tile
WARP_SIZE = 64
BLOCK_THREADS = NUM_WARPS * WARP_SIZE   # 256
N_PER_WARP = N // NUM_WARPS              # 16
K_CHUNKS = K // MFMA_K                   # 8


def _build_qk_minimal_kernel():
    arch = _get_arch()
    assert str(arch).startswith("gfx95"), f"expected gfx950, got {arch}"

    # --- LDS layout ----
    # Q LDS:  M  x K  bf16 = 16 x 128 x 2 = 4 KB
    # K LDS:  N  x K  bf16 = 64 x 128 x 2 = 16 KB
    Q_LDS_BYTES = M * K * 2
    K_LDS_BYTES = N * K * 2

    allocator = SmemAllocator(None, arch=arch, global_sym_name="flydsl_qk_smem")
    q_off = 0
    allocator.ptr = Q_LDS_BYTES
    k_off = allocator.ptr
    allocator.ptr += K_LDS_BYTES
    SMEM_BYTES = allocator.ptr

    @flyc.kernel
    def qk_kernel(
        q_ptr: fx.Tensor,        # (M, K) bf16
        k_ptr: fx.Tensor,        # (N, K) bf16
        s_ptr: fx.Tensor,        # (M, N) fp32
    ):
        tid = gpu.thread_idx.x
        warp_id = tid // fx.Int32(WARP_SIZE)    # 0..3
        lane = tid % fx.Int32(WARP_SIZE)        # 0..63

        # --- buffer resources -------------------------------------------
        q_rsrc = buffer_ops.create_buffer_resource(q_ptr, max_size=True)
        k_rsrc = buffer_ops.create_buffer_resource(k_ptr, max_size=True)
        s_rsrc = buffer_ops.create_buffer_resource(s_ptr, max_size=True)

        # --- LDS pointers -----------------------------------------------
        base = allocator.get_base()
        # Q/K live in i32 view for the cooperative load.
        q_lds_i32 = SmemPtr(base, q_off, T.i32, shape=(Q_LDS_BYTES // 4,)).get()
        k_lds_i32 = SmemPtr(base, k_off, T.i32, shape=(K_LDS_BYTES // 4,)).get()
        # bf16 view for the MFMA pack loads (each pack = 4 bf16 = 8 bytes).
        q_lds_bf16 = SmemPtr(
            base, q_off, T.bf16, shape=(Q_LDS_BYTES // 2,)
        ).get()
        k_lds_bf16 = SmemPtr(
            base, k_off, T.bf16, shape=(K_LDS_BYTES // 2,)
        ).get()

        # --- STEP 1: cooperative load Q from global -> LDS --------------
        # Q is M x K = 16 x 128 = 2048 bf16 = 4096 bytes = 1024 i32.
        # 256 threads, each thread loads 4 i32 (= 8 bf16) to cover 1024 i32.
        for li in range_constexpr(4):
            q_idx = tid + fx.Int32(li * BLOCK_THREADS)
            q_v = buffer_ops.buffer_load(
                q_rsrc, q_idx, vec_width=1, dtype=T.i32
            )
            vector.store(
                vector.from_elements(T.vec(1, T.i32), [q_v]),
                q_lds_i32,
                [arith.index_cast(T.index, q_idx)],
            )

        # --- STEP 2: cooperative load K from global -> LDS --------------
        # K is N x K = 64 x 128 = 8192 bf16 = 16384 bytes = 4096 i32.
        # 256 threads each load 16 i32 (= 64 bf16).
        for li in range_constexpr(16):
            k_idx = tid + fx.Int32(li * BLOCK_THREADS)
            k_v = buffer_ops.buffer_load(
                k_rsrc, k_idx, vec_width=1, dtype=T.i32
            )
            vector.store(
                vector.from_elements(T.vec(1, T.i32), [k_v]),
                k_lds_i32,
                [arith.index_cast(T.index, k_idx)],
            )

        gpu.barrier()

        # --- STEP 3: MFMA QK GEMM ---------------------------------------
        # MFMA m=16, n=16, k=16, bf16 -> f32x4.
        # Each warp owns 1 n-tile of 16 columns of K.
        # Lane layout for mfma_f32_16x16x16bf16_1k input operand:
        #   A operand: 4 bf16 elements per lane, so 64 lanes * 4 = 256 = 16*16 elements.
        #   B operand: same.
        # Standard layout: lane k = i + j*16 covers row k%16, col k//16 of the MFMA tile.
        # For our QK gemm, A = Q[16 x 16 chunk of K], B = K[16 x 16 chunk of K]^T (transposed K).
        # We use m=16 lanes for M-rows and 16 lanes for K-positions.
        #
        # For load layout we follow pa_decode_fp8.py STEP 5:
        #   A vec from Q LDS:  base = (lane%16)*K + lane//16 * 4 (bytes after dividing properly)
        #   B vec from K LDS:  base = (warp_id*16 + lane%16)*K + lane//16 * 4
        # Each lane loads 4 bf16 = 1 i64.

        acc = arith.constant_vector(0.0, T.f32x4)
        # Per-K-chunk loop
        for kc in range_constexpr(K_CHUNKS):
            kc_off = kc * MFMA_K  # 0, 16, 32, ..., 112 (in bf16 elements)
            # A = Q[lane%16, kc_off + (lane//16)*4 ... +4]  (4 bf16 per lane)
            q_row = lane % fx.Int32(16)         # M-row 0..15
            q_col = (lane // fx.Int32(16)) * fx.Int32(4) + fx.Int32(kc_off)
            q_elem_off = q_row * fx.Int32(K) + q_col   # bf16 element index
            q_vec = vector.load_op(
                T.vec(4, T.bf16),
                q_lds_bf16,
                [arith.index_cast(T.index, q_elem_off)],
            )
            # rocdl.mfma_f32_16x16x16bf16_1k expects vector<4xi16>, not bf16.
            q_v_i16 = vector.bitcast(T.vec(4, T.i16), q_vec)

            # B = K[warp_id*16 + lane%16, kc_off + (lane//16)*4 ... +4]
            k_row = warp_id * fx.Int32(16) + lane % fx.Int32(16)
            k_col = (lane // fx.Int32(16)) * fx.Int32(4) + fx.Int32(kc_off)
            k_elem_off = k_row * fx.Int32(K) + k_col
            k_vec = vector.load_op(
                T.vec(4, T.bf16),
                k_lds_bf16,
                [arith.index_cast(T.index, k_elem_off)],
            )
            k_v_i16 = vector.bitcast(T.vec(4, T.i16), k_vec)

            acc = rocdl.mfma_f32_16x16x16bf16_1k(
                T.f32x4, [q_v_i16, k_v_i16, acc, 0, 0, 0]
            )

        # --- STEP 4: store acc to S -------------------------------------
        # MFMA 16x16 bf16 output: lane (i, j) where i = lane//16, j = lane%16
        # writes 4 f32 to {row = i*4 + r, col = j} for r in 0..3.
        out_row_base = (lane // fx.Int32(16)) * fx.Int32(4)
        out_col = warp_id * fx.Int32(16) + (lane % fx.Int32(16))
        for r in range_constexpr(4):
            v = vector.extract(acc, static_position=[r], dynamic_position=[])
            row = out_row_base + fx.Int32(r)
            byte_off = (row * fx.Int32(N) + out_col) * fx.Int32(4)
            buffer_ops.buffer_store(
                v, s_rsrc, byte_off, offset_is_bytes=True
            )

    @flyc.jit
    def launch_fn(
        q_ptr: fx.Tensor,
        k_ptr: fx.Tensor,
        s_ptr: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        # Finalize the LDS allocator (emit gpu.module global) -- required
        # whenever ``SmemAllocator`` is used outside the kernel scope.
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        qk_kernel(q_ptr, k_ptr, s_ptr).launch(
            grid=(1,), block=(BLOCK_THREADS,),
            smem=SMEM_BYTES, stream=stream,
        )

    return launch_fn


def run_test():
    """Validate the FlyDSL QK kernel against torch.matmul on a single shape."""
    torch.manual_seed(0)
    q = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.1
    k = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.1
    s = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    launch_fn = _build_qk_minimal_kernel()
    launch_fn(q, k, s)
    torch.cuda.synchronize()

    ref = (q.float() @ k.float().T)
    err = (s - ref).abs().max().item()
    print(f"FlyDSL minimal QK ({M}x{K} @ {N}x{K}^T -> {M}x{N}): "
          f"max-abs vs torch = {err:.5f}")
    print("OK" if err < 0.05 else "FAIL")
    if err >= 0.05:
        # Locate where the error is.
        diff = (s - ref).abs()
        max_idx = diff.argmax().item()
        max_row, max_col = max_idx // N, max_idx % N
        print(f"\nLargest diff at ({max_row}, {max_col}): "
              f"flydsl={s[max_row, max_col].item():.5f} "
              f"ref={ref[max_row, max_col].item():.5f}")
        # Heatmap of |diff| > 0.01 per (row, col)
        bad = diff > 0.01
        print(f"\nNum bad cells: {bad.sum().item()} / {M*N}")
        # Group by row
        bad_per_row = bad.sum(dim=1).cpu().numpy()
        print(f"Bad cells per row (M=0..{M-1}): {bad_per_row.tolist()}")
        bad_per_col = bad.sum(dim=0).cpu().numpy()
        print(f"Bad cells per col (N=0..{N-1}): {bad_per_col.tolist()}")
    return err


if __name__ == "__main__":
    err = run_test()
    sys.exit(0 if err < 0.05 else 1)
