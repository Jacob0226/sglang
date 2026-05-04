"""
Unit benchmark for the three sparse-MLA decode kernels:

    1. Baseline   -- TileLang BF16 main_kernel
                     (``tilelang_kernel.tilelang_sparse_fwd``)
    2. Improved   -- TileLang FP8-KV variant
                     (``tilelang_kernel_fp8.tilelang_sparse_fwd_fp8``)
    3. FlyDSL     -- Single-pass FP8 fmha
                     (``flydsl_kernel.flydsl_sparse_mla_decode_fp8``)

All three are run on **the same synthetic Q / KV / topk indices**, with shapes
mirroring **GLM-5.1-FP8 with TP=8** (the production decode workload that
motivated this kernel work).  Defaults from
``/data/huggingface/hub/zai-org/GLM-5.1-FP8/config.json``::

    num_attention_heads = 64        -> 8 heads / TP-rank with --tp 8
    kv_lora_rank        = 512       == d_v   (after MLA absorb)
    qk_rope_head_dim    = 64        == d_rope
    index_topk          = 2048
    num_hidden_layers   = 78

Decode batch = 4 mirrors the ``conc=4`` profiling run; 16k KV tokens covers a
typical decode-time KV pool for the in=8192 input-length benchmark with
headroom.  Override anything via env vars (``NSA_TEST_*``) below.

Each kernel's output is compared against **two** references:

  * ``max-abs vs ref``  -- against a slow fp32 PyTorch reference
                          (``_reference_sparse_mla_decode``).  Loose tolerance;
                          catches catastrophic numerical bugs.
  * ``max-abs vs bf16`` -- against the **shipping bf16 baseline kernel
                           output** (whichever ``tilelang_sparse_fwd`` is
                           currently in production).  Strict comparison;
                           this is what guarantees a candidate kernel is
                           safe to ship.  ``bf16`` itself shows ``(self)``.

The status column says ``FAIL_REF`` if the kernel diverges from the fp32
reference, ``FAIL_BF16`` if it agrees with the reference but disagrees with
the bf16 baseline (e.g. drift from a regression), or ``OK`` if both pass
``cfg.tolerance``.  Latency is measured via ``torch.cuda.Event`` after
warmup; ``bf16`` is forced to run first so its output and timing are
available for the other rows.

This is a stand-alone script -- no pytest or sglang server required -- so you
can iterate on a single kernel without running e2e.

Usage
-----
    python -m sglang.srt.layers.attention.nsa.test_sparse_mla_decode_kernels

Environment knobs
-----------------
    NSA_TEST_NUM_TOKENS      total KV tokens in the paged buffer  (default: 16384)
    NSA_TEST_BLOCK_SIZE      tokens per page                     (default:    64)
    NSA_TEST_BATCH           number of decode queries (== conc)   (default:     4)
    NSA_TEST_HEADS           per-TP query heads                   (default:     8)
    NSA_TEST_TOPK            sparse-attention topk                (default:  2048)
    NSA_TEST_WARMUP          warmup iterations                    (default:     5)
    NSA_TEST_ITERS           timed iterations                     (default:    20)
    NSA_TEST_KERNELS         comma-separated subset of {bf16,fp8,flydsl}
                             (default: bf16,fp8,flydsl)
    NSA_TEST_TOLERANCE       max-abs tolerance vs reference       (default: 0.05)
"""

from __future__ import annotations

import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch


# ---------------------------------------------------------------------------
# Test config from env vars (so you can sweep without editing the file).
# ---------------------------------------------------------------------------
@dataclass
class TestConfig:
    """GLM-5.1-FP8 with TP=8 by default.  See module docstring for sources."""

    # KV pool: in=8192 + headroom -> 16k.  Storage tiled into 64-token blocks
    # (storage layout only; doesn't affect correctness).
    num_kv_tokens: int = int(os.getenv("NSA_TEST_NUM_TOKENS", "16384"))
    block_size: int = int(os.getenv("NSA_TEST_BLOCK_SIZE", "64"))

    # Decode shape: batch == concurrency=4 in production profile run.
    batch: int = int(os.getenv("NSA_TEST_BATCH", "4"))
    # GLM-5.1: 64 total attention heads / TP=8 = 8 heads per rank.
    # The kernel pads internally to padded_H=16 for full m=16 MFMA tiles.
    heads: int = int(os.getenv("NSA_TEST_HEADS", "8"))
    # Hard-coded shapes from GLM-5.1-FP8 config.json:
    #   kv_lora_rank=512 == d_v ; qk_rope_head_dim=64 == d_rope ;
    #   index_topk=2048
    topk: int = int(os.getenv("NSA_TEST_TOPK", "2048"))
    d_v: int = 512
    d_rope: int = 64

    warmup: int = int(os.getenv("NSA_TEST_WARMUP", "5"))
    iters: int = int(os.getenv("NSA_TEST_ITERS", "20"))
    tolerance: float = float(os.getenv("NSA_TEST_TOLERANCE", "0.05"))
    kernels: str = os.getenv("NSA_TEST_KERNELS", "bf16,fp8,flydsl")

    @property
    def dim_total(self) -> int:
        return self.d_v + self.d_rope

    @property
    def num_blocks(self) -> int:
        assert self.num_kv_tokens % self.block_size == 0
        return self.num_kv_tokens // self.block_size


# ---------------------------------------------------------------------------
# Reference: slow but correct sparse-MLA decode in pure PyTorch.
# ---------------------------------------------------------------------------
def _reference_sparse_mla_decode(
    q: torch.Tensor,                    # (batch, H, D + Drope) bf16
    kv_full_bf16: torch.Tensor,         # (num_tokens, D + Drope) bf16
    indices: torch.Tensor,              # (batch, topk) int32
    sm_scale: float,
    d_v: int,
) -> torch.Tensor:
    """Slow reference implementation -- gathers KV via indices and runs
    standard softmax(QK)·V in fp32."""
    batch, H, dim_total = q.shape
    topk = indices.shape[-1]

    out = torch.empty((batch, H, d_v), device=q.device, dtype=torch.float32)
    q_f32 = q.float()
    kv_f32 = kv_full_bf16.float()

    for b in range(batch):
        idx = indices[b].to(torch.long)              # (topk,)
        valid = idx >= 0
        # Replace negative indices by 0 for safe gather; mask in softmax.
        gathered_idx = torch.where(valid, idx, torch.zeros_like(idx))
        kv_b = kv_f32[gathered_idx]                  # (topk, D + Drope)

        # K = kv_b ; V = kv_b[:, :d_v] (MLA absorb -> V is the nope half).
        scores = torch.einsum("hd,td->ht", q_f32[b], kv_b) * sm_scale
        scores = scores.masked_fill(~valid.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)        # (H, topk)
        out[b] = torch.einsum("ht,td->hd", probs, kv_b[:, :d_v])

    return out.to(q.dtype)


# ---------------------------------------------------------------------------
# Test inputs (deterministic; same Q / KV layout for all kernels).
# ---------------------------------------------------------------------------
def make_inputs(cfg: TestConfig, device: str = "cuda", seed: int = 0):
    """Build (q, kv_bf16_paged, kv_fp8_paged, indices, kv_full_bf16).

    Shapes
    ------
    q                 : (batch, H, dim_total)              bf16
    kv_bf16_paged     : (num_blocks, block_size, 1, dim_total) bf16
    kv_fp8_paged      : (num_blocks, block_size, 1, 656)   uint8
    indices           : (batch, topk)                      int32
    kv_full_bf16      : (num_blocks*block_size, dim_total) bf16  (reference)
    """
    from sglang.srt.layers.attention.nsa.dequant_k_cache import (
        dequantize_k_cache,
    )
    from sglang.srt.layers.attention.nsa.quant_k_cache import (
        quantize_k_cache,
    )

    torch.manual_seed(seed)

    q = torch.randn(
        cfg.batch, cfg.heads, cfg.dim_total, device=device, dtype=torch.bfloat16
    )
    # KV scaled down a bit to keep softmax in a numerically reasonable range.
    kv_bf16_paged = (
        torch.randn(
            cfg.num_blocks,
            cfg.block_size,
            1,
            cfg.dim_total,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.5
    )
    # Quantize -> FP8 paged buffer (656 B/token).  Round-trip the BF16 path
    # too so the bf16 baseline operates on the *dequantized* version --
    # fair apples-to-apples comparison.
    kv_fp8_paged = quantize_k_cache(kv_bf16_paged)
    kv_bf16_paged_round = dequantize_k_cache(kv_fp8_paged)

    # Flat (token, dim) view used by the reference + by callers.
    kv_full_bf16 = kv_bf16_paged_round.view(
        cfg.num_kv_tokens, cfg.dim_total
    ).contiguous()

    # Topk indices: random subset within [0, num_kv_tokens).  Insert a small
    # number of -1 padding slots to exercise the masking path.
    indices = torch.empty(
        cfg.batch, cfg.topk, device=device, dtype=torch.int32
    )
    for b in range(cfg.batch):
        perm = torch.randperm(cfg.num_kv_tokens, device=device)[: cfg.topk]
        indices[b] = perm.to(torch.int32)
    # Mask ~1% of slots
    mask_pos = torch.rand_like(indices, dtype=torch.float32) < 0.01
    indices = torch.where(
        mask_pos, torch.full_like(indices, -1), indices
    )

    return q, kv_bf16_paged_round, kv_fp8_paged, indices, kv_full_bf16


# ---------------------------------------------------------------------------
# Kernel adapters.  Each takes the same canonical inputs and returns an
# (batch, H, d_v) bf16 tensor.
# ---------------------------------------------------------------------------
def _adapter_bf16(q, kv_bf16_paged_round, kv_fp8_paged, indices, sm_scale, d_v):
    from sglang.srt.layers.attention.nsa.tilelang_kernel import (
        tilelang_sparse_fwd,
    )

    # tilelang_sparse_fwd signature: kv ~ (seq_len_kv, kv_group=1, dim_total)
    # bf16 (the function asserts kv.dim() == 3 and unsqueezes a leading
    # batch dim internally).
    nb, bs, _, dim_total = kv_bf16_paged_round.shape
    kv_flat = (
        kv_bf16_paged_round.view(nb * bs, 1, dim_total).contiguous()
    )
    out = tilelang_sparse_fwd(
        q=q,
        kv=kv_flat,
        indices=indices.unsqueeze(1),    # (batch, 1, topk)
        sm_scale=sm_scale,
        d_v=d_v,
    )
    return out


def _adapter_fp8(q, kv_bf16_paged_round, kv_fp8_paged, indices, sm_scale, d_v):
    from sglang.srt.layers.attention.nsa.tilelang_kernel_fp8 import (
        tilelang_sparse_fwd_fp8,
    )

    out = tilelang_sparse_fwd_fp8(
        q=q,
        kv_paged_uint8=kv_fp8_paged,
        indices=indices.unsqueeze(1),
        sm_scale=sm_scale,
        d_v=d_v,
    )
    return out


def _adapter_flydsl(q, kv_bf16_paged_round, kv_fp8_paged, indices, sm_scale, d_v):
    from sglang.srt.layers.attention.nsa.flydsl_kernel import (
        flydsl_sparse_mla_decode_fp8,
    )

    out = flydsl_sparse_mla_decode_fp8(
        q=q,
        kv_paged_uint8=kv_fp8_paged,
        indices=indices.unsqueeze(1),
        sm_scale=sm_scale,
        d_v=d_v,
    )
    return out


# ---------------------------------------------------------------------------
# Bench helpers.
# ---------------------------------------------------------------------------
def _benchmark(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    """Returns avg ms per call."""
    for _ in range(warmup):
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


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


# ---------------------------------------------------------------------------
# Main runner.
# ---------------------------------------------------------------------------
def run(cfg: TestConfig) -> int:
    if not torch.cuda.is_available():
        print("[ERROR] CUDA / ROCm device not available.", file=sys.stderr)
        return 2

    device = "cuda"
    print(f"[cfg] {cfg}")

    # Build inputs once.
    q, kv_bf16_paged_round, kv_fp8_paged, indices, kv_full_bf16 = make_inputs(
        cfg, device=device
    )
    sm_scale = 1.0 / math.sqrt(cfg.dim_total)

    # Reference output (slow).
    print("[reference] computing PyTorch reference (this is slow)...")
    t0 = time.time()
    ref = _reference_sparse_mla_decode(
        q=q,
        kv_full_bf16=kv_full_bf16,
        indices=indices,
        sm_scale=sm_scale,
        d_v=cfg.d_v,
    )
    print(f"[reference] done in {(time.time() - t0) * 1000:.1f} ms")

    adapters: Dict[str, Callable] = {
        "bf16": _adapter_bf16,
        "fp8": _adapter_fp8,
        "flydsl": _adapter_flydsl,
    }
    selected = [k.strip() for k in cfg.kernels.split(",") if k.strip()]
    for name in selected:
        if name not in adapters:
            print(f"[ERROR] unknown kernel name: {name}; valid: {list(adapters)}",
                  file=sys.stderr)
            return 2

    # Always force bf16 to run first if it's in the selection -- we use its
    # output as the production-baseline reference for the "vs bf16" column.
    # This is a strictly tighter check than "vs PyTorch fp32 ref" because
    # any new variant must match what's already shipping.
    if "bf16" in selected and selected[0] != "bf16":
        selected = ["bf16"] + [k for k in selected if k != "bf16"]

    print()
    print(
        f"{'kernel':<8} | {'max-abs vs ref':<14} | {'max-abs vs bf16':<15} | "
        f"{'avg ms':>8} | {'speedup':>7} | status"
    )
    print("-" * 100)

    bf16_time: Optional[float] = None
    bf16_out: Optional[torch.Tensor] = None
    failures = 0
    for name in selected:
        adapter = adapters[name]

        def call():
            return adapter(
                q,
                kv_bf16_paged_round,
                kv_fp8_paged,
                indices,
                sm_scale,
                cfg.d_v,
            )

        # Correctness
        try:
            out = call()
            torch.cuda.synchronize()
            err_ref = _max_abs_diff(out, ref)
            if bf16_out is not None and name != "bf16":
                err_bf16 = _max_abs_diff(out, bf16_out)
            elif name == "bf16":
                # bf16 vs itself is trivially zero -- we'll display "(self)"
                # so the column unambiguously shows the baseline.
                err_bf16 = 0.0
            else:
                # bf16 not in selection or didn't run; no baseline available.
                err_bf16 = None

            # Verdict uses BOTH thresholds: vs PyTorch fp32 ref (loose
            # tolerance, catches all numerical issues) AND vs bf16 baseline
            # (strict: the new kernel must agree with shipping behavior).
            ok_ref = err_ref <= cfg.tolerance
            ok_bf16 = err_bf16 is None or err_bf16 <= cfg.tolerance
            ok = ok_ref and ok_bf16
            if ok:
                status = "OK"
            elif not ok_ref:
                status = f"FAIL_REF (>{cfg.tolerance})"
            else:
                status = f"FAIL_BF16 (>{cfg.tolerance})"
        except NotImplementedError as e:
            print(
                f"{name:<8} | {'-':<14} | {'-':<15} | {'-':>8} | {'-':>7} | "
                f"SKIP ({type(e).__name__}: {e})"
            )
            continue
        except Exception as e:
            failures += 1
            print(
                f"{name:<8} | {'-':<14} | {'-':<15} | {'-':>8} | {'-':>7} | "
                f"ERROR ({type(e).__name__}: {e})"
            )
            if os.getenv("NSA_TEST_VERBOSE_ERR"):
                traceback.print_exc()
            continue

        # Stash bf16 baseline output before benchmarking (so bench-only
        # failures don't affect downstream comparisons).
        if name == "bf16":
            bf16_out = out.detach().clone()

        # Bench
        try:
            avg_ms = _benchmark(call, cfg.warmup, cfg.iters)
        except Exception as e:
            failures += 1
            err_bf16_str = f"{err_bf16:.5f}" if err_bf16 is not None else "-"
            print(
                f"{name:<8} | {err_ref:<14.5f} | {err_bf16_str:<15} | "
                f"{'-':>8} | {'-':>7} | "
                f"BENCH-ERROR ({type(e).__name__}: {e})"
            )
            continue

        if name == "bf16":
            bf16_time = avg_ms
        speedup = (
            f"{bf16_time / avg_ms:.2f}x"
            if (bf16_time is not None and avg_ms > 0)
            else "-"
        )

        if name == "bf16":
            err_bf16_str = "(self)"
        elif err_bf16 is None:
            err_bf16_str = "-"
        else:
            err_bf16_str = f"{err_bf16:.5f}"

        print(
            f"{name:<8} | {err_ref:<14.5f} | {err_bf16_str:<15} | "
            f"{avg_ms:>8.3f} | {speedup:>7} | {status}"
        )

    print()
    if bf16_time is None:
        print("[note] no baseline timing -- enable 'bf16' in NSA_TEST_KERNELS for "
              "speedup numbers.")

    return 0 if failures == 0 else 1


def main() -> int:
    return run(TestConfig())


if __name__ == "__main__":
    sys.exit(main())
