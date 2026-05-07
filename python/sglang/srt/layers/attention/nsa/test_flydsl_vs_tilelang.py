# SPDX-License-Identifier: Apache-2.0
"""Numerical correctness test: FlyDSL sparse-MLA decode vs TileLang reference.

Runs both ``flydsl_sparse_fwd`` and ``tilelang_sparse_fwd`` on the same
(Q, KV, Indices) tensors and diffs the BF16 outputs.

Usage (inside ``rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260503``):

    cd /home/jacchang/PR/sglang
    PYTHONPATH=python python -m \
        sglang.srt.layers.attention.nsa.test_flydsl_vs_tilelang
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.layers.attention.nsa.tilelang_kernel import tilelang_sparse_fwd
from sglang.srt.layers.attention.nsa.flydsl_kernel import flydsl_sparse_fwd
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz


# Pick the FP8 variant the kernels were built for. gfx94x (MI300) uses
# e4m3fnuz, gfx95x (MI350/MI355) uses e4m3fn.
_FP8_DTYPE = (
    torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
)


@dataclass
class Case:
    name: str
    seq_len: int
    num_pages: int
    expected_inner_iter: Optional[int] = None


def _gen_inputs(case, *, num_heads, d_v, d_tail, topk, fp8_dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q_bf16 = torch.randn(
        case.seq_len, num_heads, d_v + d_tail,
        dtype=torch.bfloat16, device="cuda", generator=g,
    ) * 0.1
    kv_bf16 = torch.randn(
        case.num_pages, 1, d_v + d_tail,
        dtype=torch.bfloat16, device="cuda", generator=g,
    ) * 0.1
    q_fp8 = q_bf16.to(fp8_dtype)
    kv_fp8 = kv_bf16.to(fp8_dtype)
    indices = torch.randint(
        0, case.num_pages, (case.seq_len, 1, topk),
        dtype=torch.int32, device="cuda", generator=g,
    )
    mask = (torch.rand(indices.shape, device="cuda", generator=g) < 0.05)
    indices[mask] = -1
    return q_fp8, kv_fp8, indices


def _diff_stats(a, b):
    a32 = a.float()
    b32 = b.float()
    diff = (a32 - b32).abs()
    denom = b32.abs().clamp_min(1e-6)
    rel = diff / denom
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": rel.max().item(),
        "mean_rel": rel.mean().item(),
    }


def run_case(case, *, num_heads=128, d_v=512, d_tail=64, topk=2048,
             atol=1e-2, rtol=1e-2, seed=0):
    fp8_dtype = _FP8_DTYPE
    sm_scale = 1.0 / (d_v + d_tail) ** 0.5

    print(f"\n=== {case.name}: seq_len={case.seq_len}, "
          f"num_pages={case.num_pages}, topk={topk} ===")

    q_fp8, kv_fp8, indices = _gen_inputs(
        case, num_heads=num_heads, d_v=d_v, d_tail=d_tail,
        topk=topk, fp8_dtype=fp8_dtype, seed=seed,
    )

    out_tl = tilelang_sparse_fwd(q_fp8, kv_fp8, indices, sm_scale, d_v=d_v)
    torch.cuda.synchronize()
    if out_tl.dim() == 4:
        out_tl = out_tl.squeeze(0)

    out_fl = flydsl_sparse_fwd(q_fp8, kv_fp8, indices, sm_scale, d_v=d_v)
    torch.cuda.synchronize()
    if out_fl.dim() == 4:
        out_fl = out_fl.squeeze(0)

    print(f"  out_tl shape={tuple(out_tl.shape)} dtype={out_tl.dtype}")
    print(f"  out_fl shape={tuple(out_fl.shape)} dtype={out_fl.dtype}")

    stats = _diff_stats(out_fl, out_tl)
    print(f"  max abs diff = {stats['max_abs']:.4g}")
    print(f"  mean abs diff = {stats['mean_abs']:.4g}")
    print(f"  max rel diff = {stats['max_rel']:.4g}")
    print(f"  mean rel diff = {stats['mean_rel']:.4g}")

    ok = (stats["max_abs"] <= atol) or (stats["max_rel"] <= rtol)
    print(f"  {'PASS' if ok else 'FAIL'} (atol={atol}, rtol={rtol})")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if os.environ.get("COMPILE_ONLY"):
        del os.environ["COMPILE_ONLY"]

    cases = [
        Case("fully_fused (inner_iter=32, n_groups=1)", seq_len=1,
             num_pages=4096, expected_inner_iter=32),
        Case("typical_split_K (inner_iter=4, n_groups=8)", seq_len=4,
             num_pages=4096, expected_inner_iter=4),
        Case("fine_split_K (inner_iter=1, n_groups=32)", seq_len=8,
             num_pages=4096, expected_inner_iter=1),
    ]

    all_ok = True
    for case in cases:
        try:
            ok = run_case(case, atol=args.atol, rtol=args.rtol, seed=args.seed)
            all_ok = all_ok and ok
        except Exception as exc:
            print(f"  EXCEPTION: {exc}")
            import traceback
            traceback.print_exc()
            all_ok = False

    print()
    print("=" * 60)
    print("OVERALL:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
