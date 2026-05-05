"""
Step 3 of the FlyDSL sparse-MLA decode plan: FP8 QK with **indirect gather**.

Same single-CTA / single-tile layout as ``flydsl_qk_fp8.py``, but K is now a
**larger pool** of T=128 fp8 rows and we pick N=64 of them via an int32
``indices`` tensor.  This is the one new piece needed before we can scale up
to GLM-5.1-FP8 shapes (where K is the entire paged KV buffer and ``indices``
is the topk selection).

Shapes (small, fixed)
---------------------
- Q       : (M=16, K=128) bf16    -> host-quantized to fp8 e4m3 (OCP)
- K_full  : (T=128, K=128) fp8 e4m3 (OCP)
- indices : (N=64,) int32         -- which T-rows of K_full to use
- S       : (M=16, N=64) fp32

Gather pattern
--------------
Cooperative LDS load: each of 256 threads loads 8 i32, indirected through
``indices``::

    for li in 0..7:
        lds_idx  = tid + li * BLOCK_THREADS   # in [0 .. 2048)
        token    = lds_idx // 32              # 0..63
        k_i32    = lds_idx % 32               # 0..31 (each i32 = 4 fp8)
        src_row  = indices[token]
        K_LDS[lds_idx] = K_full[src_row * 32 + k_i32]

The MFMA inner loop is identical to Step 2 -- it reads the now-populated
K LDS without caring how it got there.

Usage
-----
    python -m sglang.srt.layers.attention.nsa.flydsl_qk_gather
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
M = 16
N = 64
K = 128
T_POOL = 128                # number of K rows in the pool
MFMA_K = 32                 # FP8 MFMA k-chunk
NUM_WARPS = 4
WARP_SIZE = 64
BLOCK_THREADS = NUM_WARPS * WARP_SIZE   # 256
K_CHUNKS = K // MFMA_K                   # 4

_FP8_DTYPE = torch.float8_e4m3fn         # gfx950 + FlyDSL OCP variant
FP8_MAX = 448.0
I32_PER_ROW = K // 4                     # 32 i32 per K row


def _build_qk_gather_kernel():
    arch = _get_arch()
    assert str(arch).startswith("gfx95"), f"expected gfx950, got {arch}"

    Q_LDS_BYTES = M * K              # 2 KB
    K_LDS_BYTES = N * K              # 8 KB

    allocator = SmemAllocator(
        None, arch=arch, global_sym_name="flydsl_qk_gather_smem"
    )
    q_off = 0
    allocator.ptr = Q_LDS_BYTES
    k_off = allocator.ptr
    allocator.ptr += K_LDS_BYTES
    SMEM_BYTES = allocator.ptr

    @flyc.kernel
    def qk_kernel(
        q_ptr: fx.Tensor,            # (M, K) fp8
        k_full_ptr: fx.Tensor,       # (T_POOL, K) fp8
        indices_ptr: fx.Tensor,      # (N,) int32
        s_ptr: fx.Tensor,            # (M, N) fp32
    ):
        tid = gpu.thread_idx.x
        warp_id = tid // fx.Int32(WARP_SIZE)
        lane = tid % fx.Int32(WARP_SIZE)

        q_rsrc = buffer_ops.create_buffer_resource(q_ptr, max_size=True)
        k_rsrc = buffer_ops.create_buffer_resource(k_full_ptr, max_size=True)
        idx_rsrc = buffer_ops.create_buffer_resource(indices_ptr, max_size=True)
        s_rsrc = buffer_ops.create_buffer_resource(s_ptr, max_size=True)

        base = allocator.get_base()
        q_lds_i32 = SmemPtr(base, q_off, T.i32, shape=(Q_LDS_BYTES // 4,)).get()
        k_lds_i32 = SmemPtr(base, k_off, T.i32, shape=(K_LDS_BYTES // 4,)).get()

        # --- STEP 1: cooperative load Q (contiguous, no gather) ---------
        # Q is M*K = 16*128 = 2048 fp8 = 512 i32; 256 threads x 2 loads.
        for li in range_constexpr(2):
            q_idx = tid + fx.Int32(li * BLOCK_THREADS)
            q_v = buffer_ops.buffer_load(
                q_rsrc, q_idx, vec_width=1, dtype=T.i32
            )
            vector.store(
                vector.from_elements(T.vec(1, T.i32), [q_v]),
                q_lds_i32,
                [arith.index_cast(T.index, q_idx)],
            )

        # --- STEP 2: indirect-gather K via indices ----------------------
        # K_LDS layout: row-major (N, K) bf16-equivalent fp8.
        # I32_PER_ROW = 32, so K_LDS has N*32 = 2048 i32.
        # 256 threads x 8 loads each = 2048 loads.  Each load:
        #   lds_idx = tid + li * 256
        #   token   = lds_idx // 32
        #   k_i32   = lds_idx % 32
        #   src_row = indices[token]                  (one i32 load)
        #   K_LDS[lds_idx] = K_full[src_row * 32 + k_i32]
        for li in range_constexpr(8):
            lds_idx = tid + fx.Int32(li * BLOCK_THREADS)
            token = lds_idx // fx.Int32(I32_PER_ROW)
            k_i32 = lds_idx % fx.Int32(I32_PER_ROW)

            # Load the gather index for this token.  buffer_load with
            # vec_width=1 returns a scalar i32 directly (no vector wrap),
            # mirroring pa_decode_fp8.py phys_block_v usage.
            src_row = buffer_ops.buffer_load(
                idx_rsrc, token, vec_width=1, dtype=T.i32
            )

            # Compute global K_full i32 index: src_row * 32 + k_i32
            g_i32_idx = src_row * fx.Int32(I32_PER_ROW) + k_i32

            k_v = buffer_ops.buffer_load(
                k_rsrc, g_i32_idx, vec_width=1, dtype=T.i32
            )
            vector.store(
                vector.from_elements(T.vec(1, T.i32), [k_v]),
                k_lds_i32,
                [arith.index_cast(T.index, lds_idx)],
            )

        gpu.barrier()

        # --- STEP 3: MFMA QK GEMM (FP8 x FP8) ---------------------------
        # Same as Step 2: pair-pack i32 -> i64, mfma_f32_16x16x32_fp8_fp8.
        def _pack_i32_pair_to_i64(a_i32, b_i32):
            v = vector.from_elements(T.vec(2, T.i32), [a_i32, b_i32])
            v1 = vector.bitcast(T.vec(1, T.i64), v)
            return vector.extract(v1, static_position=[0], dynamic_position=[])

        acc = arith.constant_vector(0.0, T.f32x4)
        for kc in range_constexpr(K_CHUNKS):
            kc_off = kc * MFMA_K  # 0, 32, 64, 96 (in fp8 elements)

            q_row = lane % fx.Int32(16)
            q_col_base = (lane // fx.Int32(16)) * fx.Int32(8) + fx.Int32(kc_off)
            q_elem_off = q_row * fx.Int32(K) + q_col_base
            q_i32_idx = q_elem_off // fx.Int32(4)
            q_a = vector.load_op(
                T.vec(1, T.i32), q_lds_i32,
                [arith.index_cast(T.index, q_i32_idx)],
            )
            q_b = vector.load_op(
                T.vec(1, T.i32), q_lds_i32,
                [arith.index_cast(T.index, q_i32_idx + fx.Int32(1))],
            )
            q_a_i32 = vector.extract(q_a, static_position=[0], dynamic_position=[])
            q_b_i32 = vector.extract(q_b, static_position=[0], dynamic_position=[])
            q_i64 = _pack_i32_pair_to_i64(q_a_i32, q_b_i32)

            k_row = warp_id * fx.Int32(16) + lane % fx.Int32(16)
            k_col_base = (lane // fx.Int32(16)) * fx.Int32(8) + fx.Int32(kc_off)
            k_elem_off = k_row * fx.Int32(K) + k_col_base
            k_i32_idx = k_elem_off // fx.Int32(4)
            k_a = vector.load_op(
                T.vec(1, T.i32), k_lds_i32,
                [arith.index_cast(T.index, k_i32_idx)],
            )
            k_b = vector.load_op(
                T.vec(1, T.i32), k_lds_i32,
                [arith.index_cast(T.index, k_i32_idx + fx.Int32(1))],
            )
            k_a_i32 = vector.extract(k_a, static_position=[0], dynamic_position=[])
            k_b_i32 = vector.extract(k_b, static_position=[0], dynamic_position=[])
            k_i64 = _pack_i32_pair_to_i64(k_a_i32, k_b_i32)

            acc = rocdl.mfma_f32_16x16x32_fp8_fp8(
                T.f32x4, [q_i64, k_i64, acc, 0, 0, 0]
            )

        # --- STEP 4: store acc to S -------------------------------------
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
        k_full_ptr: fx.Tensor,
        indices_ptr: fx.Tensor,
        s_ptr: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        qk_kernel(q_ptr, k_full_ptr, indices_ptr, s_ptr).launch(
            grid=(1,), block=(BLOCK_THREADS,),
            smem=SMEM_BYTES, stream=stream,
        )

    return launch_fn


def _quantize_to_fp8(x_bf16: torch.Tensor):
    amax = x_bf16.abs().amax(dim=-1, keepdim=True).float()
    scale = (amax / FP8_MAX).clamp(min=1e-8)
    x_scaled = (x_bf16.float() / scale).clamp(-FP8_MAX, FP8_MAX)
    x_fp8 = x_scaled.to(_FP8_DTYPE)
    return x_fp8, scale.squeeze(-1)


def run_test():
    """Validate the gather kernel against torch on a permuted-K reference."""
    torch.manual_seed(0)
    q_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.5
    k_full_bf16 = torch.randn(T_POOL, K, dtype=torch.bfloat16, device="cuda") * 0.5

    q_fp8, q_scale = _quantize_to_fp8(q_bf16)
    k_full_fp8, k_full_scale = _quantize_to_fp8(k_full_bf16)

    # Pick N=64 random rows out of T_POOL=128.
    perm = torch.randperm(T_POOL, device="cuda")[:N].to(torch.int32)
    # Insert a couple of duplicates to stress the gather (no-mask version).
    perm[5] = perm[10]
    perm[20] = perm[3]

    s_raw = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    launch_fn = _build_qk_gather_kernel()
    launch_fn(q_fp8, k_full_fp8, perm, s_raw)
    torch.cuda.synchronize()

    # Apply scales.  k_scale is per-row, so use the *gathered* scales.
    k_scale_gathered = k_full_scale[perm.long()]
    s = s_raw * q_scale.unsqueeze(1) * k_scale_gathered.unsqueeze(0)

    # Reference
    ref = q_bf16.float() @ k_full_bf16[perm.long()].float().T
    err = (s - ref).abs().max().item()
    print(f"FlyDSL FP8 QK + gather (M={M} N={N} K={K}, T_POOL={T_POOL}): "
          f"max-abs vs torch = {err:.5f}")
    print("OK" if err < 0.5 else "FAIL")
    if err >= 0.5:
        diff = (s - ref).abs()
        max_idx = diff.argmax().item()
        max_row, max_col = max_idx // N, max_idx % N
        print(f"\nLargest diff at ({max_row}, {max_col}): "
              f"flydsl={s[max_row, max_col].item():.5f} "
              f"ref={ref[max_row, max_col].item():.5f}")
        bad = diff > 0.5
        print(f"\nNum bad cells: {bad.sum().item()} / {M*N}")
        bad_per_col = bad.sum(dim=0).cpu().numpy()
        print(f"Bad cells per col (N=0..{N-1}): {bad_per_col.tolist()}")
    return err


if __name__ == "__main__":
    err = run_test()
    sys.exit(0 if err < 0.5 else 1)
