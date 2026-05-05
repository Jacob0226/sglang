"""
Step 4 of the FlyDSL sparse-MLA decode plan: GLM-5.1-FP8 shapes + rope tail
+ multi-chunk topk loop.

Scales up Step 3 to the production decode shape::

    HG     = 16   head group (matches GLM-5.1 padded_H = 16)
    BI     = 64   tokens per topk chunk
    D      = 512  nope dim (FP8)
    D_ROPE = 64   rope dim (BF16)
    TopK   = 2048 sparse-attention topk
    T_POOL = 4096 KV pool (test only; real run uses paged buffer)

Output is raw QK product (HG, TopK) fp32, **without** softmax / SV / scale.
The host applies (q_scale * k_scale_per_token_per_tile) post-MFMA for the
correctness check; the in-kernel application is deferred to Step E (which
adds the online softmax that consumes acc_qk before it leaves registers).

Per-chunk loop body
-------------------
For each of NI=TopK/BI=32 chunks::

    1. Cooperative-gather K_nope_chunk (BI, D) fp8 into LDS via Indices
    2. Cooperative-gather K_rope_chunk (BI, D_ROPE) bf16 into LDS via Indices
    3. QK FP8 MFMA over D=512:   16 mfma_f32_16x16x32_fp8_fp8 per warp
    4. QK BF16 MFMA over D_ROPE: 4 mfma_f32_16x16x16bf16_1k     per warp
    5. Store S[:, chunk_off : chunk_off+BI] to global

Q stays resident in LDS across chunks (loaded once at kernel start).

LDS budget per CTA
------------------
    Q_nope_LDS :  HG * D       = 16 * 512  =  8 KB (fp8)
    Q_rope_LDS :  HG * D_ROPE  = 16 * 64*2 =  2 KB (bf16)
    K_nope_LDS :  BI * D       = 64 * 512  = 32 KB (fp8)
    K_rope_LDS :  BI * D_ROPE  = 64 * 64*2 =  8 KB (bf16)
                                          ----------
                                            50 KB / 160 KB on gfx950

Usage
-----
    python -m sglang.srt.layers.attention.nsa.flydsl_qk_glm5
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
HG = 16              # heads per CTA (GLM-5.1 padded)
BI = 64              # tokens per topk chunk
D = 512              # nope dim
D_ROPE = 64          # rope dim
TOPK = 2048
T_POOL = 4096        # KV pool size for test
NI = TOPK // BI      # 32 chunks

NUM_WARPS = 4
WARP_SIZE = 64
BLOCK_THREADS = NUM_WARPS * WARP_SIZE   # 256
N_PER_WARP = BI // NUM_WARPS             # 16

MFMA_K_FP8 = 32
MFMA_K_BF = 16
K_CHUNKS_FP8 = D // MFMA_K_FP8           # 16
K_CHUNKS_BF = D_ROPE // MFMA_K_BF        # 4

D_I32 = D // 4               # 128 i32 per row of D-fp8
D_ROPE_I32 = D_ROPE // 2     # 32 i32 per row of D_ROPE-bf16

_FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0


def _build_qk_glm5_kernel():
    arch = _get_arch()
    assert str(arch).startswith("gfx95"), f"expected gfx950, got {arch}"

    Q_NOPE_BYTES = HG * D                 # 8 KB
    Q_ROPE_BYTES = HG * D_ROPE * 2        # 2 KB
    K_NOPE_BYTES = BI * D                 # 32 KB
    K_ROPE_BYTES = BI * D_ROPE * 2        # 8 KB

    allocator = SmemAllocator(
        None, arch=arch, global_sym_name="flydsl_qk_glm5_smem"
    )
    q_nope_off = 0
    allocator.ptr = Q_NOPE_BYTES
    q_rope_off = allocator.ptr
    allocator.ptr += Q_ROPE_BYTES
    k_nope_off = allocator.ptr
    allocator.ptr += K_NOPE_BYTES
    k_rope_off = allocator.ptr
    allocator.ptr += K_ROPE_BYTES
    SMEM_BYTES = allocator.ptr

    @flyc.kernel
    def qk_kernel(
        q_nope_ptr: fx.Tensor,        # (HG, D) fp8
        q_rope_ptr: fx.Tensor,        # (HG, D_ROPE) bf16
        k_nope_ptr: fx.Tensor,        # (T_POOL, D) fp8
        k_rope_ptr: fx.Tensor,        # (T_POOL, D_ROPE) bf16
        indices_ptr: fx.Tensor,       # (TOPK,) int32
        s_ptr: fx.Tensor,             # (HG, TOPK) fp32
    ):
        tid = gpu.thread_idx.x
        warp_id = tid // fx.Int32(WARP_SIZE)
        lane = tid % fx.Int32(WARP_SIZE)

        qn_rsrc = buffer_ops.create_buffer_resource(q_nope_ptr, max_size=True)
        qr_rsrc = buffer_ops.create_buffer_resource(q_rope_ptr, max_size=True)
        kn_rsrc = buffer_ops.create_buffer_resource(k_nope_ptr, max_size=True)
        kr_rsrc = buffer_ops.create_buffer_resource(k_rope_ptr, max_size=True)
        idx_rsrc = buffer_ops.create_buffer_resource(indices_ptr, max_size=True)
        s_rsrc = buffer_ops.create_buffer_resource(s_ptr, max_size=True)

        base = allocator.get_base()
        qn_lds_i32 = SmemPtr(base, q_nope_off, T.i32, shape=(Q_NOPE_BYTES // 4,)).get()
        qr_lds_i32 = SmemPtr(base, q_rope_off, T.i32, shape=(Q_ROPE_BYTES // 4,)).get()
        kn_lds_i32 = SmemPtr(base, k_nope_off, T.i32, shape=(K_NOPE_BYTES // 4,)).get()
        kr_lds_i32 = SmemPtr(base, k_rope_off, T.i32, shape=(K_ROPE_BYTES // 4,)).get()
        # Element views for MFMA loads.
        qr_lds_bf16 = SmemPtr(base, q_rope_off, T.bf16, shape=(Q_ROPE_BYTES // 2,)).get()
        kr_lds_bf16 = SmemPtr(base, k_rope_off, T.bf16, shape=(K_ROPE_BYTES // 2,)).get()

        # --- STEP 1: load Q_nope (8 KB = 2048 i32, 256 threads x 8 loads) ---
        for li in range_constexpr(8):
            qi = tid + fx.Int32(li * BLOCK_THREADS)
            qv = buffer_ops.buffer_load(qn_rsrc, qi, vec_width=1, dtype=T.i32)
            vector.store(
                vector.from_elements(T.vec(1, T.i32), [qv]),
                qn_lds_i32,
                [arith.index_cast(T.index, qi)],
            )

        # --- STEP 2: load Q_rope (2 KB = 512 i32, 256 threads x 2 loads) ----
        for li in range_constexpr(2):
            qi = tid + fx.Int32(li * BLOCK_THREADS)
            qv = buffer_ops.buffer_load(qr_rsrc, qi, vec_width=1, dtype=T.i32)
            vector.store(
                vector.from_elements(T.vec(1, T.i32), [qv]),
                qr_lds_i32,
                [arith.index_cast(T.index, qi)],
            )

        # Helper: pack 2 i32 -> 1 i64 for fp8 MFMA.
        def _pack_i32_pair_to_i64(a, b):
            v = vector.from_elements(T.vec(2, T.i32), [a, b])
            v1 = vector.bitcast(T.vec(1, T.i64), v)
            return vector.extract(v1, static_position=[0], dynamic_position=[])

        # --- Multi-chunk topk loop -------------------------------------
        for chunk_i in range_constexpr(NI):
            chunk_off = chunk_i * BI                      # in tokens (0..2047 step 64)
            chunk_off_i32_kn = chunk_off * D_I32          # placeholder; not used
            del chunk_off_i32_kn

            # --- STEP 3: gather K_nope_chunk -----------------------------
            # K_nope_LDS layout: (BI, D) fp8 = (64, 512) = 32 KB = 8192 i32.
            # 256 threads x 32 loads each.
            #   lds_idx = tid + li * 256
            #   token   = lds_idx // D_I32  (D_I32 = 128 i32 per token)
            #   k_i32   = lds_idx % D_I32
            #   src_row = indices[chunk_off + token]
            #   K_nope_LDS[lds_idx] = K_nope_full[src_row * D_I32 + k_i32]
            for li in range_constexpr(K_NOPE_BYTES // 4 // BLOCK_THREADS):
                lds_idx = tid + fx.Int32(li * BLOCK_THREADS)
                token = lds_idx // fx.Int32(D_I32)
                k_i32 = lds_idx % fx.Int32(D_I32)
                src_row = buffer_ops.buffer_load(
                    idx_rsrc,
                    fx.Int32(chunk_off) + token,
                    vec_width=1, dtype=T.i32,
                )
                g_i32 = src_row * fx.Int32(D_I32) + k_i32
                kv = buffer_ops.buffer_load(kn_rsrc, g_i32, vec_width=1, dtype=T.i32)
                vector.store(
                    vector.from_elements(T.vec(1, T.i32), [kv]),
                    kn_lds_i32,
                    [arith.index_cast(T.index, lds_idx)],
                )

            # --- STEP 4: gather K_rope_chunk -----------------------------
            # K_rope_LDS layout: (BI, D_ROPE) bf16 = (64, 64) = 8 KB = 2048 i32.
            # 256 threads x 8 loads each.
            for li in range_constexpr(K_ROPE_BYTES // 4 // BLOCK_THREADS):
                lds_idx = tid + fx.Int32(li * BLOCK_THREADS)
                token = lds_idx // fx.Int32(D_ROPE_I32)
                k_i32 = lds_idx % fx.Int32(D_ROPE_I32)
                src_row = buffer_ops.buffer_load(
                    idx_rsrc,
                    fx.Int32(chunk_off) + token,
                    vec_width=1, dtype=T.i32,
                )
                g_i32 = src_row * fx.Int32(D_ROPE_I32) + k_i32
                kv = buffer_ops.buffer_load(kr_rsrc, g_i32, vec_width=1, dtype=T.i32)
                vector.store(
                    vector.from_elements(T.vec(1, T.i32), [kv]),
                    kr_lds_i32,
                    [arith.index_cast(T.index, lds_idx)],
                )

            gpu.barrier()

            # --- STEP 5: QK FP8 MFMA over D=512 -------------------------
            acc = arith.constant_vector(0.0, T.f32x4)
            for kc in range_constexpr(K_CHUNKS_FP8):
                kc_off = kc * MFMA_K_FP8                  # 0, 32, 64, ..., 480

                q_row = lane % fx.Int32(16)
                q_col = (lane // fx.Int32(16)) * fx.Int32(8) + fx.Int32(kc_off)
                q_elem_off = q_row * fx.Int32(D) + q_col
                q_i32_idx = q_elem_off // fx.Int32(4)
                q_a = vector.load_op(
                    T.vec(1, T.i32), qn_lds_i32,
                    [arith.index_cast(T.index, q_i32_idx)],
                )
                q_b = vector.load_op(
                    T.vec(1, T.i32), qn_lds_i32,
                    [arith.index_cast(T.index, q_i32_idx + fx.Int32(1))],
                )
                q_a_i32 = vector.extract(q_a, static_position=[0], dynamic_position=[])
                q_b_i32 = vector.extract(q_b, static_position=[0], dynamic_position=[])
                q_i64 = _pack_i32_pair_to_i64(q_a_i32, q_b_i32)

                k_row = warp_id * fx.Int32(16) + lane % fx.Int32(16)
                k_col = (lane // fx.Int32(16)) * fx.Int32(8) + fx.Int32(kc_off)
                k_elem_off = k_row * fx.Int32(D) + k_col
                k_i32_idx = k_elem_off // fx.Int32(4)
                k_a = vector.load_op(
                    T.vec(1, T.i32), kn_lds_i32,
                    [arith.index_cast(T.index, k_i32_idx)],
                )
                k_b = vector.load_op(
                    T.vec(1, T.i32), kn_lds_i32,
                    [arith.index_cast(T.index, k_i32_idx + fx.Int32(1))],
                )
                k_a_i32 = vector.extract(k_a, static_position=[0], dynamic_position=[])
                k_b_i32 = vector.extract(k_b, static_position=[0], dynamic_position=[])
                k_i64 = _pack_i32_pair_to_i64(k_a_i32, k_b_i32)

                acc = rocdl.mfma_f32_16x16x32_fp8_fp8(
                    T.f32x4, [q_i64, k_i64, acc, 0, 0, 0]
                )

            # --- STEP 6: QK BF16 MFMA over D_ROPE=64 ---------------------
            for kc in range_constexpr(K_CHUNKS_BF):
                kc_off = kc * MFMA_K_BF                   # 0, 16, 32, 48

                q_row = lane % fx.Int32(16)
                q_col = (lane // fx.Int32(16)) * fx.Int32(4) + fx.Int32(kc_off)
                q_elem_off = q_row * fx.Int32(D_ROPE) + q_col
                q_vec = vector.load_op(
                    T.vec(4, T.bf16), qr_lds_bf16,
                    [arith.index_cast(T.index, q_elem_off)],
                )
                q_v_i16 = vector.bitcast(T.vec(4, T.i16), q_vec)

                k_row = warp_id * fx.Int32(16) + lane % fx.Int32(16)
                k_col = (lane // fx.Int32(16)) * fx.Int32(4) + fx.Int32(kc_off)
                k_elem_off = k_row * fx.Int32(D_ROPE) + k_col
                k_vec = vector.load_op(
                    T.vec(4, T.bf16), kr_lds_bf16,
                    [arith.index_cast(T.index, k_elem_off)],
                )
                k_v_i16 = vector.bitcast(T.vec(4, T.i16), k_vec)

                acc = rocdl.mfma_f32_16x16x16bf16_1k(
                    T.f32x4, [q_v_i16, k_v_i16, acc, 0, 0, 0]
                )

            # --- STEP 7: store S[:, chunk_off : chunk_off+BI] -----------
            out_row_base = (lane // fx.Int32(16)) * fx.Int32(4)
            out_col_chunk = warp_id * fx.Int32(16) + (lane % fx.Int32(16))
            out_col = fx.Int32(chunk_off) + out_col_chunk
            for r in range_constexpr(4):
                v = vector.extract(acc, static_position=[r], dynamic_position=[])
                row = out_row_base + fx.Int32(r)
                byte_off = (row * fx.Int32(TOPK) + out_col) * fx.Int32(4)
                buffer_ops.buffer_store(
                    v, s_rsrc, byte_off, offset_is_bytes=True
                )

            # Wait for next chunk's K data to land before re-using LDS.
            gpu.barrier()

    @flyc.jit
    def launch_fn(
        q_nope_ptr: fx.Tensor,
        q_rope_ptr: fx.Tensor,
        k_nope_ptr: fx.Tensor,
        k_rope_ptr: fx.Tensor,
        indices_ptr: fx.Tensor,
        s_ptr: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        qk_kernel(
            q_nope_ptr, q_rope_ptr, k_nope_ptr, k_rope_ptr,
            indices_ptr, s_ptr,
        ).launch(
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
    """Validate the GLM-5.1-FP8-shape QK + gather kernel against torch."""
    torch.manual_seed(0)
    q_nope_bf16 = torch.randn(HG, D, dtype=torch.bfloat16, device="cuda") * 0.5
    q_rope_bf16 = torch.randn(HG, D_ROPE, dtype=torch.bfloat16, device="cuda") * 0.5
    k_nope_bf16 = torch.randn(T_POOL, D, dtype=torch.bfloat16, device="cuda") * 0.5
    k_rope_bf16 = torch.randn(T_POOL, D_ROPE, dtype=torch.bfloat16, device="cuda") * 0.5

    q_nope_fp8, q_scale = _quantize_to_fp8(q_nope_bf16)
    k_nope_fp8, k_scale = _quantize_to_fp8(k_nope_bf16)

    # Random topk indices.
    indices = torch.randperm(T_POOL, device="cuda")[:TOPK].to(torch.int32)

    s_raw = torch.zeros(HG, TOPK, dtype=torch.float32, device="cuda")
    launch_fn = _build_qk_glm5_kernel()
    launch_fn(
        q_nope_fp8, q_rope_bf16, k_nope_fp8, k_rope_bf16,
        indices, s_raw,
    )
    torch.cuda.synchronize()

    # --- Reconstruct expected S = (Q_nope_fp8 @ K_nope_fp8^T) * q_scale * k_scale
    #                             + Q_rope_bf16 @ K_rope_bf16^T  (gathered)
    # FP8 part: kernel produces raw fp8xfp8 sum; we apply per-row scales here.
    k_nope_gathered = k_nope_fp8[indices.long()]
    k_scale_gathered = k_scale[indices.long()]
    s_nope_raw = q_nope_fp8.float() @ k_nope_gathered.float().T
    s_nope_scaled = (
        s_nope_raw * q_scale.unsqueeze(1) * k_scale_gathered.unsqueeze(0)
    )

    # BF16 rope part:
    k_rope_gathered = k_rope_bf16[indices.long()]
    s_rope = q_rope_bf16.float() @ k_rope_gathered.float().T

    # The kernel stores raw fp8xfp8 + bf16xbf16 sum (no scale).  To compare
    # apples-to-apples against the kernel output, we compute the same:
    expect_kernel = s_nope_raw + s_rope

    # Reference (high-precision, what the model "should" produce):
    ref_full = (
        q_nope_bf16.float() @ k_nope_bf16[indices.long()].float().T
        + q_rope_bf16.float() @ k_rope_bf16[indices.long()].float().T
    )

    err_kernel = (s_raw - expect_kernel).abs().max().item()
    s_max = expect_kernel.abs().max().item()
    rel_kernel = err_kernel / max(s_max, 1.0)
    print(f"FlyDSL GLM5 QK ({HG}x{D+D_ROPE} @ {TOPK}x{D+D_ROPE}^T -> "
          f"{HG}x{TOPK}):")
    print(f"  s_raw vs kernel-expected               max-abs = {err_kernel:.4f}  "
          f"(relative = {rel_kernel:.2e})")

    s_scaled = s_nope_scaled + s_rope
    err_full = (s_scaled - ref_full).abs().max().item()
    print(f"  s (after fp8 scale) vs torch fp32 ref  max-abs = {err_full:.4f}")

    # Pass criteria
    # -------------
    # err_kernel: kernel-vs-host-fp32-on-the-same-fp8-bytes reduction.  The
    # kernel uses MFMA's hardware tree reduction; torch matmul uses a different
    # order; for K=576 fp8 sum building to ~5e5 magnitudes, ~1e-4 relative
    # is the f32-accumulator noise floor.  Threshold = 1e-3 relative.
    # err_full: scaled fp8 result vs torch bf16 ref.  FP8 quant introduces
    # ~1/256 of amax per element; with K=512 sum and per-row scales this
    # scales up to ~1.0 max-abs in the (-5..5) value range.  Threshold = 1.5.
    ok = (rel_kernel < 1e-3) and (err_full < 1.5)
    print("OK" if ok else "FAIL")
    return err_kernel, err_full, rel_kernel


if __name__ == "__main__":
    err_k, err_f, rel_k = run_test()
    sys.exit(0 if (rel_k < 1e-3 and err_f < 1.5) else 1)
