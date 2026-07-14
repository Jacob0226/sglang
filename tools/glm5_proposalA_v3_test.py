#!/usr/bin/env python3
"""
Proposal A v3: dispatch-order variant of Proposal A.

Hypothesis
----------
HIP graph's replay scheduler picks the physical stream for each branch of
a fork based on DISPATCH ORDER, not on which `torch.cuda.Stream` we hand
it.  Concretely: at each fork point, the branch whose first captured
kernel appears FIRST in the capture stays on the same physical stream as
the predecessor (phys 0), and the other branch gets a new physical
stream (phys 4).

Evidence from earlier traces:
  Baseline UT                       Proposal A v2 (current)
  ─────────────────────             ─────────────────────────────────
  fork:                             fork:
    q_b_proj(cur)  FIRST  ────> p0    with stream(alt): indexer  FIRST  ────> p0
    with(alt):indexer  2nd ────> p4   kv_a_norm+W_kc+RoPE+Cat(cur) 2nd ────> p4
  observed: cur→p0, alt→p4          observed: indexer→p0, gap-fill→p4
                                    (labels swapped relative to Python dispatch)

This script tests three variants of Proposal A differing ONLY in the
order we dispatch work after the fork, to confirm / falsify the theory
and, if the theory holds, produce the "intended" physical layout:

  A_v2   :  with stream(alt): indexer   →  gap-fill(cur)        (original)
  A_v3   :  gap-fill(cur)                →  with stream(alt): indexer  (reorder)
  A_v3e  :  same as v3, but uses explicit record_event/wait_event
            instead of wait_stream (to rule out the sync primitive)

Run inside the MI355X container:
  python3 glm5_proposalA_v3_test.py --benchmark --profile
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import json
import os
import sys

os.environ.setdefault("SGLANG_USE_AITER", "1")

import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

base = importlib.import_module("glm5_decode_layer")
pa = importlib.import_module("glm5_proposalA_test_v2")

BF16 = torch.bfloat16
FP32 = torch.float32
FP8 = base.FP8


# ── Proposal A v3: reordered dispatch ─────────────────────────────────────

def decode_layer_forward_proposalA_v3(
    hidden, residual, attn_w, idx_w, moe_w, cfg,
    input_ln_w, post_attn_ln_w, kv_cache, dual_stream, alt_stream,
):
    """Same as Proposal A v2, but dispatch CUR's gap-fill BEFORE the
    `with torch.cuda.stream(alt)` block so cur's chain is captured first.
    """
    cur = torch.cuda.current_stream()

    h_normed, res_out, q_lat, kv_lat, h_fp8, h_scale = base.phase1_pre_attention(
        hidden, residual, attn_w, cfg, input_ln_w
    )

    if dual_stream and alt_stream is not None:
        q, q_lora = base.phase2_current(q_lat, attn_w, cfg)

        alt_stream.wait_stream(cur)

        # === REORDER: dispatch cur's gap-fill FIRST ==================
        kv_normed = base.aiter_rms_norm(kv_lat, attn_w.kv_a_ln_w, cfg.rms_norm_eps)
        q_all = pa.phase3_pre_wait(q, kv_normed, attn_w, cfg, kv_cache)

        # Then dispatch alt's indexer SECOND
        with torch.cuda.stream(alt_stream):
            topk_ids = pa.phase2_alt_proposalA(
                h_normed, h_fp8, h_scale, q_lora, idx_w, cfg
            )

        cur.wait_stream(alt_stream)
        attn_out = pa.phase3_post_wait(q_all, attn_w, cfg, kv_cache, topk_ids)
    else:
        q, q_lora = base.phase2_current(q_lat, attn_w, cfg)
        kv_normed = base.aiter_rms_norm(kv_lat, attn_w.kv_a_ln_w, cfg.rms_norm_eps)
        topk_ids = pa.phase2_alt_proposalA(
            h_normed, h_fp8, h_scale, q_lora, idx_w, cfg
        )
        q_all = pa.phase3_pre_wait(q, kv_normed, attn_w, cfg, kv_cache)
        attn_out = pa.phase3_post_wait(q_all, attn_w, cfg, kv_cache, topk_ids)

    # post-attn (same as baseline)
    mlp_normed = torch.empty_like(attn_out)
    res_out2 = torch.empty_like(attn_out)
    base.aiter_fused_add_rms_norm(
        mlp_normed, attn_out, res_out, res_out2, post_attn_ln_w, cfg.rms_norm_eps
    )

    if dual_stream and alt_stream is not None:
        alt_stream.wait_stream(cur)
        shared_out = base.phase4_shared(mlp_normed, moe_w, cfg)
        with torch.cuda.stream(alt_stream):
            routed_out = base.phase4_routed(mlp_normed, moe_w, cfg)
        cur.wait_stream(alt_stream)
    else:
        shared_out = base.phase4_shared(mlp_normed, moe_w, cfg)
        routed_out = base.phase4_routed(mlp_normed, moe_w, cfg)

    return shared_out + routed_out, res_out2


def decode_layer_forward_proposalA_v4(
    hidden, residual, attn_w, idx_w, moe_w, cfg,
    input_ln_w, post_attn_ln_w, kv_cache, dual_stream, alt_stream,
):
    """Proposal A v4 = v3 layout but wait_stream moved BEFORE q_b_proj.

    In v3 we did:
        q, q_lora = phase2_current(...)      # cur runs q_b_proj first
        alt_stream.wait_stream(cur)          # alt now waits for q_b_proj too
        kv_a_norm + W_kc + RoPE + Cat ...
        with stream(alt): indexer

    That makes alt start ~16μs late because indexer's first kernel (wq_b)
    only needs q_lora = q_lat (phase1 output), NOT q from q_b_proj.
    Baseline recognizes this and places wait_stream *before* q_b_proj:

        alt_stream.wait_stream(cur)          # alt waits only for phase1
        q, q_lora = phase2_current(...)      # cur's q_b_proj runs in parallel with alt
        with stream(alt): kv_a_norm + indexer

    v4 uses the same early wait_stream but keeps Proposal A's gap-fill
    (kv_a_norm + W_kc + RoPE + Cat) on cur BEFORE the `with stream(alt):`
    block to preserve the A_v3 physical layout (indexer on aux stream).
    """
    cur = torch.cuda.current_stream()

    h_normed, res_out, q_lat, kv_lat, h_fp8, h_scale = base.phase1_pre_attention(
        hidden, residual, attn_w, cfg, input_ln_w
    )

    if dual_stream and alt_stream is not None:
        # KEY: wait_stream BEFORE q_b_proj, so alt only waits for phase1.
        alt_stream.wait_stream(cur)

        q, q_lora = base.phase2_current(q_lat, attn_w, cfg)

        # cur-side gap fill, dispatched first (keeps cur chain on phys 0)
        kv_normed = base.aiter_rms_norm(kv_lat, attn_w.kv_a_ln_w, cfg.rms_norm_eps)
        q_all = pa.phase3_pre_wait(q, kv_normed, attn_w, cfg, kv_cache)

        # indexer on alt, dispatched second → aux stream
        with torch.cuda.stream(alt_stream):
            topk_ids = pa.phase2_alt_proposalA(
                h_normed, h_fp8, h_scale, q_lora, idx_w, cfg
            )

        cur.wait_stream(alt_stream)
        attn_out = pa.phase3_post_wait(q_all, attn_w, cfg, kv_cache, topk_ids)
    else:
        q, q_lora = base.phase2_current(q_lat, attn_w, cfg)
        kv_normed = base.aiter_rms_norm(kv_lat, attn_w.kv_a_ln_w, cfg.rms_norm_eps)
        topk_ids = pa.phase2_alt_proposalA(
            h_normed, h_fp8, h_scale, q_lora, idx_w, cfg
        )
        q_all = pa.phase3_pre_wait(q, kv_normed, attn_w, cfg, kv_cache)
        attn_out = pa.phase3_post_wait(q_all, attn_w, cfg, kv_cache, topk_ids)

    mlp_normed = torch.empty_like(attn_out)
    res_out2 = torch.empty_like(attn_out)
    base.aiter_fused_add_rms_norm(
        mlp_normed, attn_out, res_out, res_out2, post_attn_ln_w, cfg.rms_norm_eps
    )

    if dual_stream and alt_stream is not None:
        alt_stream.wait_stream(cur)
        shared_out = base.phase4_shared(mlp_normed, moe_w, cfg)
        with torch.cuda.stream(alt_stream):
            routed_out = base.phase4_routed(mlp_normed, moe_w, cfg)
        cur.wait_stream(alt_stream)
    else:
        shared_out = base.phase4_shared(mlp_normed, moe_w, cfg)
        routed_out = base.phase4_routed(mlp_normed, moe_w, cfg)

    return shared_out + routed_out, res_out2


def decode_layer_forward_proposalA_v3_events(
    hidden, residual, attn_w, idx_w, moe_w, cfg,
    input_ln_w, post_attn_ln_w, kv_cache, dual_stream, alt_stream,
):
    """Same as v3 but explicit record_event / wait_event instead of
    wait_stream.  Controls for the sync primitive."""
    cur = torch.cuda.current_stream()

    h_normed, res_out, q_lat, kv_lat, h_fp8, h_scale = base.phase1_pre_attention(
        hidden, residual, attn_w, cfg, input_ln_w
    )

    if dual_stream and alt_stream is not None:
        q, q_lora = base.phase2_current(q_lat, attn_w, cfg)

        ev_fork = cur.record_event()
        alt_stream.wait_event(ev_fork)

        kv_normed = base.aiter_rms_norm(kv_lat, attn_w.kv_a_ln_w, cfg.rms_norm_eps)
        q_all = pa.phase3_pre_wait(q, kv_normed, attn_w, cfg, kv_cache)

        with torch.cuda.stream(alt_stream):
            topk_ids = pa.phase2_alt_proposalA(
                h_normed, h_fp8, h_scale, q_lora, idx_w, cfg
            )

        ev_join = alt_stream.record_event()
        cur.wait_event(ev_join)
        attn_out = pa.phase3_post_wait(q_all, attn_w, cfg, kv_cache, topk_ids)
    else:
        q, q_lora = base.phase2_current(q_lat, attn_w, cfg)
        kv_normed = base.aiter_rms_norm(kv_lat, attn_w.kv_a_ln_w, cfg.rms_norm_eps)
        topk_ids = pa.phase2_alt_proposalA(
            h_normed, h_fp8, h_scale, q_lora, idx_w, cfg
        )
        q_all = pa.phase3_pre_wait(q, kv_normed, attn_w, cfg, kv_cache)
        attn_out = pa.phase3_post_wait(q_all, attn_w, cfg, kv_cache, topk_ids)

    mlp_normed = torch.empty_like(attn_out)
    res_out2 = torch.empty_like(attn_out)
    base.aiter_fused_add_rms_norm(
        mlp_normed, attn_out, res_out, res_out2, post_attn_ln_w, cfg.rms_norm_eps
    )

    if dual_stream and alt_stream is not None:
        ev_mf = cur.record_event()
        alt_stream.wait_event(ev_mf)
        shared_out = base.phase4_shared(mlp_normed, moe_w, cfg)
        with torch.cuda.stream(alt_stream):
            routed_out = base.phase4_routed(mlp_normed, moe_w, cfg)
        ev_mj = alt_stream.record_event()
        cur.wait_event(ev_mj)
    else:
        shared_out = base.phase4_shared(mlp_normed, moe_w, cfg)
        routed_out = base.phase4_routed(mlp_normed, moe_w, cfg)

    return shared_out + routed_out, res_out2


# ── Trace inspection: which kernels ended up on which stream? ────────────

def kernel_group_counts_by_stream(trace_path):
    """Bucket kernels into semantic groups and report per physical stream."""
    with gzip.open(trace_path, "rt") as f:
        t = json.load(f)
    evts = t if isinstance(t, list) else t.get("traceEvents", [])

    # Semantic buckets based on kernel name substrings
    groups = {
        "phase1_rms_quant":       ["fused_rms_fp8_group_quant"],
        "phase1_qkv_a_proj":      ["dynamic_per_group_scaled_quant",
                                   "add_rmsnorm_quant"],
        "q_b_proj_or_indexer_gemm": ["ck::kernel_gemm_xdl_cshuffle_v3"],
        "gap_kv_a_norm":          [],  # shares kernel w/ add_rmsnorm_quant
        "gap_W_kc_or_W_vc":       ["_batched_gemm_a8w8_a_per_token_group"],
        "gap_fused_qk_rope_cat":  ["fused_qk_rope_cat_and_cache_mla"],
        "indexer_k_norm":         ["Layernorm2dFwd"],
        "indexer_rope_2c":        ["kn_entry_2c_sbhd_cached_indirect_inplace"],
        "indexer_hadamard":       ["fast_hadamard_transform"],
        "indexer_act_quant":      ["act_quant_kernel"],
        "indexer_set_k":          ["_set_k_and_s_triton"],
        "indexer_weights_proj":   ["triton_poi_fused__to_copy_gemm",
                                   "Cijk_Alik_Bljk_S_B"],
        "indexer_paged_mqa":      ["_gluon_deepgemm_fp8_paged_mqa",
                                   "fast_topk_transform"],
        "attn_tilelang":          ["main_kernel"],
        "attn_flatten_fp8":       ["_fused_flatten_fp8_group_quant"],
        "allreduce":              ["cross_device_reduce"],
        "moe_shared":             [],  # shares CK gemm with phase1/qkv
        "moe_routed_gate":        ["wv_splitk_small_fp16_bf16"],
        "moe_routed_topk":        ["grouped_topk",
                                   "MoeSortingKernel"],
        "moe_routed_gemm":        ["kernel_moe_gemm"],
        "sgl_silu":               ["act_and_mul_kernel"],
        "pytorch_elementwise":    ["at::native::vectorized_elementwise",
                                   "at::native::"],
    }

    per_stream = {}
    for e in evts:
        if e.get("cat") != "kernel":
            continue
        sid = e.get("args", {}).get("stream")
        if sid is None:
            continue
        name = e.get("name", "")
        bucket = "other"
        for g, patterns in groups.items():
            if any(p in name for p in patterns):
                bucket = g
                break
        per_stream.setdefault(sid, {})
        per_stream[sid][bucket] = per_stream[sid].get(bucket, 0) + 1

    return per_stream


def print_bucket_table(per_stream, title):
    print(f"\n--- {title} ---")
    all_buckets = set()
    for d in per_stream.values():
        all_buckets.update(d.keys())
    all_buckets = sorted(all_buckets)
    streams = sorted(per_stream.keys())
    print(f"  {'bucket':<30s}  " + "  ".join(f"phys{s:>2d}" for s in streams))
    for b in all_buckets:
        row = "  ".join(f"{per_stream[s].get(b, 0):>6d}" for s in streams)
        print(f"  {b:<30s}  {row}")


# ── Benchmark driver ─────────────────────────────────────────────────────

def _capture(fn, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def _time_graph(g, iters):
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        g.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1000.0 / iters


def benchmark(cfg, *, warmup=20, iters=100, profile=False, seed=42,
              device="cuda"):
    print(f"\n{'='*72}\nBENCHMARK (HIP Graph)  M={cfg.batch_size}  "
          f"seq_len={cfg.seq_len}  warmup={warmup}  iters={iters}\n{'='*72}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    attn_w = base.AttentionWeights(cfg, device)
    idx_w = base.IndexerWeights(cfg, device)
    moe_w = base.MoEWeights(cfg, device)
    kv_cache = base.KVCache(cfg, cfg.batch_size, device)
    in_ln = torch.ones(cfg.hidden_size, dtype=FP32, device=device)
    post_ln = torch.ones(cfg.hidden_size, dtype=FP32, device=device)
    alt = torch.cuda.Stream()

    h = torch.randn(cfg.batch_size, cfg.hidden_size,
                    dtype=BF16, device=device) * 0.01
    r = torch.zeros(cfg.batch_size, cfg.hidden_size,
                    dtype=BF16, device=device)

    variants = {
        "Baseline":  lambda: base.decode_layer_forward(
            h, r, attn_w, idx_w, moe_w, cfg, in_ln, post_ln,
            kv_cache, True, alt),
        "A_v2":      lambda: pa.decode_layer_forward_proposalA(
            h, r, attn_w, idx_w, moe_w, cfg, in_ln, post_ln,
            kv_cache, True, alt),
        "A_v3":      lambda: decode_layer_forward_proposalA_v3(
            h, r, attn_w, idx_w, moe_w, cfg, in_ln, post_ln,
            kv_cache, True, alt),
        "A_v3e":     lambda: decode_layer_forward_proposalA_v3_events(
            h, r, attn_w, idx_w, moe_w, cfg, in_ln, post_ln,
            kv_cache, True, alt),
        "A_v4":      lambda: decode_layer_forward_proposalA_v4(
            h, r, attn_w, idx_w, moe_w, cfg, in_ln, post_ln,
            kv_cache, True, alt),
    }

    print("  capturing all graphs ...")
    graphs = {name: _capture(fn, warmup=warmup) for name, fn in variants.items()}

    for g in graphs.values():
        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()

    samples = {k: [] for k in variants}
    keys = list(variants.keys())
    orderings = [
        keys,
        list(reversed(keys)),
        ["A_v4", "A_v3", "A_v3e", "Baseline", "A_v2"],
    ]
    for order in orderings:
        for name in order:
            samples[name].append(_time_graph(graphs[name], iters))

    print()
    for name in variants:
        vals = sorted(samples[name])
        med = vals[len(vals) // 2]
        print(f"  {name:<10s}: samples={[f'{v:.1f}' for v in vals]}  median={med:.1f} μs")

    print()
    t_base = sorted(samples["Baseline"])[1]
    for k in ("A_v2", "A_v3", "A_v3e", "A_v4"):
        t = sorted(samples[k])[1]
        print(f"  Δ({k:<5s} vs base): {t_base - t:+.1f} μs "
              f"({(t_base - t) / t_base * 100:+.1f}%)")

    if profile:
        trace_dir = os.path.expanduser("~/SGLang-benchmarks/tmp/trace")
        os.makedirs(trace_dir, exist_ok=True)
        traces = {}
        for name, g in graphs.items():
            n_prof = 5
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
            ) as prof:
                for _ in range(n_prof):
                    g.replay()
                torch.cuda.synchronize()
            p = os.path.join(
                trace_dir,
                f"glm5_decode_M{cfg.batch_size}_{name}_v3diag.json.gz")
            prof.export_chrome_trace(p)
            traces[name] = p
            print(f"  TRACE: {p}")

        print(f"\n{'='*72}\nSTREAM LAYOUT COMPARISON\n{'='*72}")
        for name, p in traces.items():
            per = kernel_group_counts_by_stream(p)
            print_bucket_table(per, f"{name}")

    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = base.GLM5Config(batch_size=args.batch_size, seq_len=args.seq_len)
    benchmark(cfg, warmup=args.warmup, iters=args.iters,
              profile=args.profile, seed=args.seed)


if __name__ == "__main__":
    main()
