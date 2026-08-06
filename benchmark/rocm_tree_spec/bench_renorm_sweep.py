"""Renorm cost as a function of distribution sharpness.

A single batch shape hides the thing that actually drives renorm cost. The
pivot search has a fast path bounded by a 1024-entry prefix and a full-sort
fallback when the nucleus does not fit, so the same code is either flat or
dominated by torch.sort depending only on how peaked the rows are. FlashInfer's
AOT kernel has the opposite sensitivity: its ternary pivot search needs more
rounds as a row gets peakier.

Sweeping a logit scale exposes both curves. Reporting the overflow fraction
alongside them is what makes two runs comparable: softmax(randn) over a 100K
vocabulary is nearly uniform and overflows every row, which is not a regime any
real model produces.
"""

from __future__ import annotations

import argparse

import torch

from sglang.kernels.ops.sampling.renorm import (
    _TOP_P_PREFIX,
    top_k_renorm_probs_torch,
    top_p_renorm_probs_torch,
)
from sglang.kernels.ops.sampling.renorm_triton import (
    top_k_renorm_probs_triton,
    top_p_renorm_probs_triton,
)

DEV = torch.device("cuda")


def load_aot():
    """Probe with a real call: the wrapper imports fine when the op is absent.
    Both spellings are tried because the wheel lags the in-tree rename."""
    for k_name, p_name in (
        ("top_k_renorm_probs", "top_p_renorm_probs"),
        ("top_k_renorm_prob", "top_p_renorm_prob"),
    ):
        try:
            import sgl_kernel

            fn_k = getattr(sgl_kernel, k_name)
            fn_p = getattr(sgl_kernel, p_name)
            probs = torch.softmax(torch.randn(2, 128, device=DEV), dim=-1)
            fn_k(probs, torch.full((2,), 8, dtype=torch.int64, device=DEV))
            fn_p(probs, torch.full((2,), 0.9, device=DEV))
            return fn_k, fn_p
        except Exception:
            continue
    return None


def timeit(fn, iters: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def describe(probs: torch.Tensor, top_p: float, sample_rows: int = 64):
    """Mean top-1 mass, median nucleus size, and the fraction of rows whose
    nucleus exceeds the prefix, which is what triggers the sort fallback."""
    sub = probs[:sample_rows]
    values, _ = torch.sort(sub, dim=-1, descending=True)
    cumsum = values.cumsum(dim=-1)
    nucleus = (cumsum < top_p).sum(dim=-1) + 1
    return (
        float(values[:, 0].mean()),
        int(nucleus.median()),
        float((nucleus > _TOP_P_PREFIX).float().mean()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=int, default=151936, help="GLM-5.2 vocabulary")
    parser.add_argument("--rows", type=int, default=1536, help="bs 256 x 6 draft tokens")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=(1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0),
        help="logit scale; 1.0 is the unscaled softmax(randn) that overflows every row",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    aot = load_aot()
    name = torch.cuda.get_device_name(0)
    print(f"{name}  vocab={args.vocab}  rows={args.rows}  top_p={args.top_p}  top_k={args.top_k}")
    print(f"AOT renorm kernel: {'available' if aot else 'unavailable (expected on ROCm)'}")
    print(f"prefix={_TOP_P_PREFIX} (nucleus above this falls back to a full sort)\n")

    header = (
        f"{'scale':>6} {'top1':>7} {'nucleus':>8} {'ovf':>6} "
        f"{'k_torch':>8} {'k_triton':>9} {'p_torch':>8} {'p_triton':>9}"
    )
    if aot is not None:
        header += f" {'k_aot':>7} {'p_aot':>7}"
    print(header)

    for scale in args.scales:
        probs = torch.softmax(
            torch.randn(args.rows, args.vocab, device=DEV) * scale, dim=-1
        )
        top1, nucleus, ovf = describe(probs, args.top_p)
        top_ks = torch.full((args.rows,), args.top_k, dtype=torch.int64, device=DEV)
        top_ps = torch.full((args.rows,), args.top_p, dtype=torch.float32, device=DEV)

        kt = timeit(lambda: top_k_renorm_probs_torch(probs, top_ks), args.iters)
        kr = timeit(lambda: top_k_renorm_probs_triton(probs, top_ks), args.iters)
        pt = timeit(lambda: top_p_renorm_probs_torch(probs, top_ps), args.iters)
        pr = timeit(lambda: top_p_renorm_probs_triton(probs, top_ps), args.iters)

        row = (
            f"{scale:>6.1f} {top1:>7.3f} {nucleus:>8} {ovf:>6.0%} "
            f"{kt:>8.3f} {kr:>9.3f} {pt:>8.3f} {pr:>9.3f}"
        )
        if aot is not None:
            ka = timeit(lambda: aot[0](probs, top_ks), args.iters)
            pa = timeit(lambda: aot[1](probs, top_ps), args.iters)
            row += f" {ka:>7.3f} {pa:>7.3f}"
        print(row)

        del probs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
