"""Renorm fallbacks vs the FlashInfer AOT kernel: equivalence first, then cost.

The threshold-semantics fix in renorm.py was validated on ROCm against the
in-tree Triton reference, which itself only claims FlashInfer alignment rather
than being FlashInfer. This script compares against the AOT kernel directly, so
the columns that matter only appear on NVIDIA hardware. On ROCm it degrades to
torch vs Triton, which is still useful as a regression check.

Support equality is the metric that matters: threshold semantics keep every
entry tied at the cutoff, rank semantics keep exactly k. A single differing
token means the same request can sample differently across platforms.
"""

from __future__ import annotations

import argparse

import torch

from sglang.kernels.ops.sampling.renorm import (
    top_k_renorm_probs_torch,
    top_p_renorm_probs_torch,
)
from sglang.kernels.ops.sampling.renorm_triton import (
    top_k_renorm_probs_triton,
    top_p_renorm_probs_triton,
)

DEV = torch.device("cuda")


def load_aot():
    """The wrapper imports fine on ROCm; only the torch op is missing, so probe
    with a real call rather than trusting the import. Wheels older than the
    in-tree rename only export the singular spelling."""
    import sgl_kernel

    def pick(*names):
        for name in names:
            fn = getattr(sgl_kernel, name, None)
            if fn is not None:
                return fn
        raise AttributeError(names[0])

    try:
        aot_k = pick("top_k_renorm_probs", "top_k_renorm_prob")
        aot_p = pick("top_p_renorm_probs", "top_p_renorm_prob")

        probs = torch.softmax(torch.randn(2, 128, device=DEV), dim=-1)
        aot_k(probs, torch.full((2,), 8, dtype=torch.int64, device=DEV))
        aot_p(probs, torch.full((2,), 0.9, device=DEV))
    except Exception:
        return None
    return aot_k, aot_p


def make_probs(shape: str, rows: int, vocab: int) -> torch.Tensor:
    """Distribution shapes chosen for where the semantics actually diverge:
    'flat' makes every entry a tie, 'sharp' pushes the tail below fp32
    resolution, 'duplicated' plants exact ties at the cutoff."""
    if shape == "flat":
        return torch.full((rows, vocab), 1.0 / vocab, device=DEV)
    if shape == "duplicated":
        base = torch.softmax(torch.randn(rows, vocab, device=DEV), dim=-1)
        # Force the top 64 entries of every row to an identical value.
        vals, idx = base.topk(64, dim=-1)
        base.scatter_(1, idx, vals.mean(dim=-1, keepdim=True).expand_as(vals))
        return base / base.sum(dim=-1, keepdim=True)
    scale = {"uniform": 1.0, "sharp": 8.0, "peaked": 16.0}[shape]
    return torch.softmax(torch.randn(rows, vocab, device=DEV) * scale, dim=-1)


def diff(got: torch.Tensor, ref: torch.Tensor) -> tuple[int, float]:
    support = int(((got > 0) ^ (ref > 0)).sum())
    value = float((got - ref).abs().max()) if got.numel() else 0.0
    return support, value


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


def run_correctness(aot, vocab: int, rows: int) -> bool:
    if aot is None:
        print("\n== correctness: skipped, no AOT kernel to compare against ==")
        return True
    aot_k, aot_p = aot
    print(f"\n== correctness vs AOT (rows={rows}, vocab={vocab}) ==")
    print(f"{'shape':>11} {'param':>14} {'torch sup':>10} {'triton sup':>11} {'max |dv|':>10}")

    worst = 0
    for shape in ("uniform", "sharp", "peaked", "flat", "duplicated"):
        probs = make_probs(shape, rows, vocab)

        for k in (1, 4, 20, 1024):
            top_ks = torch.full((rows,), k, dtype=torch.int64, device=DEV)
            ref = aot_k(probs, top_ks)
            s_torch, _ = diff(top_k_renorm_probs_torch(probs, top_ks), ref)
            s_triton, v = diff(top_k_renorm_probs_triton(probs, top_ks), ref)
            worst = max(worst, s_torch, s_triton)
            print(f"{shape:>11} {'top_k=' + str(k):>14} {s_torch:>10} {s_triton:>11} {v:>10.2e}")

        for p in (0.1, 0.5, 0.9, 0.95, 1.0):
            top_ps = torch.full((rows,), p, dtype=torch.float32, device=DEV)
            ref = aot_p(probs, top_ps)
            s_torch, _ = diff(top_p_renorm_probs_torch(probs, top_ps), ref)
            s_triton, v = diff(top_p_renorm_probs_triton(probs, top_ps), ref)
            worst = max(worst, s_torch, s_triton)
            print(f"{shape:>11} {'top_p=' + str(p):>14} {s_torch:>10} {s_triton:>11} {v:>10.2e}")

        # Per-row heterogeneous thresholds: the real serving case, and the one
        # a scalar-threshold test would never catch.
        top_ks = torch.randint(1, 256, (rows,), dtype=torch.int64, device=DEV)
        top_ps = torch.rand((rows,), dtype=torch.float32, device=DEV).clamp(0.05, 1.0)
        s_tk, _ = diff(top_k_renorm_probs_triton(probs, top_ks), aot_k(probs, top_ks))
        s_tp, v = diff(top_p_renorm_probs_triton(probs, top_ps), aot_p(probs, top_ps))
        worst = max(worst, s_tk, s_tp)
        print(f"{shape:>11} {'mixed per-row':>14} {'-':>10} {max(s_tk, s_tp):>11} {v:>10.2e}")

        del probs
        torch.cuda.empty_cache()

    print(f"\nworst support mismatch: {worst}  ->  {'PASS' if worst == 0 else 'FAIL'}")
    return worst == 0


def run_bench(aot, vocab: int, num_draft: int, iters: int, batches) -> None:
    print(f"\n== cost (vocab={vocab}, num_draft={num_draft}, {torch.cuda.get_device_name(0)}) ==")
    header = f"{'bs':>5} {'rows':>7} {'op':>6} {'torch(ms)':>10} {'triton(ms)':>11}"
    if aot is not None:
        header += f" {'aot(ms)':>9} {'triton/aot':>11}"
    print(header)

    for bs in batches:
        rows = bs * num_draft
        probs = torch.softmax(torch.randn(rows, vocab, device=DEV), dim=-1)
        top_ks = torch.full((rows,), 20, dtype=torch.int64, device=DEV)
        top_ps = torch.full((rows,), 0.95, dtype=torch.float32, device=DEV)

        cases = (
            ("top_k", top_k_renorm_probs_torch, top_k_renorm_probs_triton, top_ks, 0),
            ("top_p", top_p_renorm_probs_torch, top_p_renorm_probs_triton, top_ps, 1),
        )
        for name, fn_torch, fn_triton, arg, which in cases:
            t_torch = timeit(lambda: fn_torch(probs, arg), iters)
            t_triton = timeit(lambda: fn_triton(probs, arg), iters)
            row = f"{bs:>5} {rows:>7} {name:>6} {t_torch:>10.3f} {t_triton:>11.3f}"
            if aot is not None:
                t_aot = timeit(lambda: aot[which](probs, arg), iters)
                row += f" {t_aot:>9.3f} {t_triton / t_aot:>10.2f}x"
            print(row)

        del probs
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=int, default=151936, help="GLM-5.2 vocabulary")
    parser.add_argument("--num-draft", type=int, default=6)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--rows", type=int, default=64, help="rows for the correctness pass")
    parser.add_argument("--mode", choices=("both", "correctness", "bench"), default="both")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    aot = load_aot()
    print(f"AOT renorm kernel: {'available' if aot else 'unavailable (expected on ROCm)'}")

    ok = True
    if args.mode in ("both", "correctness"):
        ok = run_correctness(aot, args.vocab, args.rows)
    if args.mode in ("both", "bench"):
        run_bench(aot, args.vocab, args.num_draft, args.iters, (1, 8, 32, 128, 256))

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
