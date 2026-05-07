# SPDX-License-Identifier: Apache-2.0
"""Benchmark FlyDSL sparse-MLA decode vs TileLang reference.

Measures end-to-end latency of ``flydsl_sparse_fwd`` and ``tilelang_sparse_fwd``
on realistic NSA decode shapes. Uses CUDA events for timing and reports
ms / call plus tokens/sec.

Usage::

    cd /home/jacchang/PR/sglang
    PYTHONPATH=python python -m \\
        sglang.srt.layers.attention.nsa.bench_flydsl_vs_tilelang
"""

from __future__ import annotations

import argparse
import statistics

import torch

from sglang.srt.layers.attention.nsa.tilelang_kernel import tilelang_sparse_fwd
from sglang.srt.layers.attention.nsa.flydsl_kernel import flydsl_sparse_fwd
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz


_FP8 = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn


def _gen(seq_len, num_pages, num_heads=128, d_v=512, d_tail=64, topk=2048, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(seq_len, num_heads, d_v + d_tail,
                     dtype=torch.bfloat16, device="cuda", generator=g) * 0.1
         ).to(_FP8)
    kv = (torch.randn(num_pages, 1, d_v + d_tail,
                      dtype=torch.bfloat16, device="cuda", generator=g) * 0.1
          ).to(_FP8)
    indices = torch.randint(
        0, num_pages, (seq_len, 1, topk), dtype=torch.int32,
        device="cuda", generator=g,
    )
    return q, kv, indices


def _bench(fn, *args, warmup=10, iters=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn(*args)
        ends[i].record()
    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return {
        "mean": statistics.mean(times_ms),
        "median": statistics.median(times_ms),
        "min": min(times_ms),
        "p10": sorted(times_ms)[len(times_ms) // 10],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    sm_scale = 1.0 / (576) ** 0.5
    cases = [
        ("seq=64,  pages=8192",  64,  8192),
        ("seq=128, pages=16384", 128, 16384),
        ("seq=256, pages=32768", 256, 32768),
        ("seq=512, pages=65536", 512, 65536),
    ]

    print(f"{'case':<25} {'tilelang(ms)':>14} {'flydsl(ms)':>12} {'speedup':>8}")
    print("-" * 65)
    for name, seq_len, num_pages in cases:
        try:
            q, kv, indices = _gen(seq_len, num_pages)

            tl_stats = _bench(tilelang_sparse_fwd, q, kv, indices, sm_scale,
                              warmup=args.warmup, iters=args.iters)
            fl_stats = _bench(flydsl_sparse_fwd, q, kv, indices, sm_scale,
                              warmup=args.warmup, iters=args.iters)

            tl_med = tl_stats["median"]
            fl_med = fl_stats["median"]
            speedup = tl_med / fl_med
            print(f"{name:<25} {tl_med:>14.4f} {fl_med:>12.4f} {speedup:>7.2f}x")
        except Exception as exc:
            print(f"{name:<25} EXCEPTION: {exc}")


if __name__ == "__main__":
    main()
