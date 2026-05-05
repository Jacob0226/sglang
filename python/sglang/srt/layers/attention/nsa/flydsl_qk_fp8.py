"""
Step 2 of the FlyDSL sparse-MLA decode plan: bf16 -> FP8 MFMA.

Same single-CTA / single-tile QK GEMM as ``flydsl_qk_minimal.py``, but with
both operands in FP8 (e4m3fnuz on gfx950) instead of bf16.  Validates that
the FP8 MFMA inner-loop pattern works before we layer indirect gather and
softmax on top.

Shapes (small, fixed)
---------------------
- Q : (M=16, K=128) FP8 e4m3fnuz   (host-side amax-scaled before launch)
- K : (N=64, K=128) FP8 e4m3fnuz
- S : (M=16, N=64) fp32             (raw FP8 product, x q_scale x k_scale)

The host-side scaling lets us avoid writing the in-kernel Q-quant prologue
(D1 in the master plan) for now -- once Step E lands and we have the full
fmha kernel, we'll either (a) move quant inside the kernel, or (b) leave it
in Python as a one-shot prologue, depending on which is faster.

MFMA op
-------
``rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [a_i64, b_i64, acc_f32x4])``

Operand: ``i64`` packing 8 FP8 elements per lane (vs ``vec<4xi16>`` for the
bf16 path).  K-chunks: K=128 / k=32 = 4 (vs 8 for bf16).

Usage
-----
    python -m sglang.srt.layers.attention.nsa.flydsl_qk_fp8
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
MFMA_M = 16
MFMA_N = 16
MFMA_K = 32          # FP8 MFMA k-chunk
NUM_WARPS = 4
WARP_SIZE = 64
BLOCK_THREADS = NUM_WARPS * WARP_SIZE   # 256
N_PER_WARP = N // NUM_WARPS              # 16
K_CHUNKS = K // MFMA_K                   # 4

# gfx950 + FlyDSL uses **e4m3 OCP (Float8E4M3FN)**, not FNUZ.  Verified:
# ``flydsl.expr.typing.default_f8_type() -> Float8E4M3FN.ir_type``.
# Mismatching the variant makes every MFMA produce NaN (FNUZ bit patterns
# can decode as NaN under FN interpretation).
_FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0   # e4m3 OCP max value


def _build_qk_fp8_kernel():
    arch = _get_arch()
    assert str(arch).startswith("gfx95"), f"expected gfx950, got {arch}"

    # Q LDS:  M  x K  fp8 = 16 x 128 = 2 KB
    # K LDS:  N  x K  fp8 = 64 x 128 = 8 KB
    Q_LDS_BYTES = M * K
    K_LDS_BYTES = N * K

    allocator = SmemAllocator(None, arch=arch, global_sym_name="flydsl_qk_fp8_smem")
    q_off = 0
    allocator.ptr = Q_LDS_BYTES
    k_off = allocator.ptr
    allocator.ptr += K_LDS_BYTES
    SMEM_BYTES = allocator.ptr

    @flyc.kernel
    def qk_kernel(
        q_ptr: fx.Tensor,        # (M, K) fp8
        k_ptr: fx.Tensor,        # (N, K) fp8
        s_ptr: fx.Tensor,        # (M, N) fp32
    ):
        tid = gpu.thread_idx.x
        warp_id = tid // fx.Int32(WARP_SIZE)
        lane = tid % fx.Int32(WARP_SIZE)

        # --- buffer resources -------------------------------------------
        q_rsrc = buffer_ops.create_buffer_resource(q_ptr, max_size=True)
        k_rsrc = buffer_ops.create_buffer_resource(k_ptr, max_size=True)
        s_rsrc = buffer_ops.create_buffer_resource(s_ptr, max_size=True)

        # --- LDS pointers -----------------------------------------------
        base = allocator.get_base()
        q_lds_i32 = SmemPtr(base, q_off, T.i32, shape=(Q_LDS_BYTES // 4,)).get()
        k_lds_i32 = SmemPtr(base, k_off, T.i32, shape=(K_LDS_BYTES // 4,)).get()

        # --- STEP 1: cooperative load Q from global -> LDS --------------
        # Q is M*K = 16*128 = 2048 fp8 = 2048 bytes = 512 i32.
        # 256 threads each load 2 i32 (= 8 fp8) to cover 512 i32.
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

        # --- STEP 2: cooperative load K from global -> LDS --------------
        # K is N*K = 64*128 = 8192 fp8 = 8192 bytes = 2048 i32.
        # 256 threads each load 8 i32 (= 32 fp8) to cover 2048 i32.
        for li in range_constexpr(8):
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

        # --- STEP 3: MFMA QK GEMM (FP8 x FP8) ---------------------------
        # mfma_f32_16x16x32_fp8_fp8: A and B operands are i64 (8 fp8 each).
        # Lane (i, j) where i=lane//16, j=lane%16 provides:
        #   - A: 8 fp8 at row=j, k-cols={i*8 .. i*8+7}
        #   - B: 8 fp8 at col=j, k-rows={i*8 .. i*8+7}
        # (Same lane-tile partition as bf16 MFMA but with k=32 instead of 16,
        # so 8 elements per lane instead of 4.)

        # Helper: pack 2 i32 -> 1 i64 (pa_decode_fp8.py STEP 6 pattern).
        def _pack_i32_pair_to_i64(a_i32, b_i32):
            v = vector.from_elements(T.vec(2, T.i32), [a_i32, b_i32])
            v1 = vector.bitcast(T.vec(1, T.i64), v)
            return vector.extract(v1, static_position=[0], dynamic_position=[])

        acc = arith.constant_vector(0.0, T.f32x4)
        for kc in range_constexpr(K_CHUNKS):
            kc_off = kc * MFMA_K  # 0, 32, 64, 96 (in fp8 elements)

            # A = Q[lane%16, kc_off + (lane//16)*8 ... +8]   = 8 fp8 = 2 i32
            q_row = lane % fx.Int32(16)
            q_col_base = (lane // fx.Int32(16)) * fx.Int32(8) + fx.Int32(kc_off)
            q_elem_off = q_row * fx.Int32(K) + q_col_base    # fp8 element index
            # i32 index = elem_off // 4 (4 fp8 per i32).
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

            # B = K[warp_id*16 + lane%16, kc_off + (lane//16)*8 ... +8]
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
        # Same output layout as bf16 MFMA: lane (i, j) writes
        #   {row = i*4 + r, col = warp_id*16 + j} for r in 0..3.
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
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        qk_kernel(q_ptr, k_ptr, s_ptr).launch(
            grid=(1,), block=(BLOCK_THREADS,),
            smem=SMEM_BYTES, stream=stream,
        )

    return launch_fn


def _quantize_to_fp8(x_bf16: torch.Tensor):
    """Per-row symmetric quant to e4m3 OCP, returns (fp8_tensor, scales)."""
    amax = x_bf16.abs().amax(dim=-1, keepdim=True).float()
    scale = (amax / FP8_MAX).clamp(min=1e-8)
    x_scaled = (x_bf16.float() / scale).clamp(-FP8_MAX, FP8_MAX)
    x_fp8 = x_scaled.to(_FP8_DTYPE)
    return x_fp8, scale.squeeze(-1)


def run_test():
    """Validate the FlyDSL FP8 QK kernel against torch fp32 reference."""
    torch.manual_seed(0)
    q_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.5
    k_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.5

    q_fp8, q_scale = _quantize_to_fp8(q_bf16)
    k_fp8, k_scale = _quantize_to_fp8(k_bf16)

    s_raw = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    launch_fn = _build_qk_fp8_kernel()
    launch_fn(q_fp8, k_fp8, s_raw)
    torch.cuda.synchronize()

    # Apply scales: S[m, n] = sum(q_fp8[m] * k_fp8[n]) * q_scale[m] * k_scale[n]
    s = s_raw * q_scale.unsqueeze(1) * k_scale.unsqueeze(0)

    # Reference: torch matmul on the original bf16 tensors (cast to fp32).
    ref = q_bf16.float() @ k_bf16.float().T
    err = (s - ref).abs().max().item()

    # FP8 quant introduces error proportional to FP8_MAX/256 ~ 1/256 of amax,
    # then the matmul scales it by sqrt(K). With K=128 random ~N(0, 0.5)
    # the quant error after sum should be ~0.05 ish.
    print(f"FlyDSL FP8 QK ({M}x{K} @ {N}x{K}^T -> {M}x{N}): "
          f"max-abs vs torch (after scale-up) = {err:.5f}")
    print("OK" if err < 0.5 else "FAIL")
    if err >= 0.5:
        diff = (s - ref).abs()
        max_idx = diff.argmax().item()
        max_row, max_col = max_idx // N, max_idx % N
        print(f"\nLargest diff at ({max_row}, {max_col}): "
              f"flydsl={s[max_row, max_col].item():.5f} "
              f"ref={ref[max_row, max_col].item():.5f}")
    return err


if __name__ == "__main__":
    err = run_test()
    sys.exit(0 if err < 0.5 else 1)
