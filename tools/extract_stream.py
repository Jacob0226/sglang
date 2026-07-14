#!/usr/bin/env python3
"""
extract_stream.py

Extract kernels from a specific CUDA stream within a time range from a
PyTorch profiler trace (.json or .json.gz).

Usage examples:

    # List all streams and their kernel counts
    python extract_stream.py trace.json.gz --list-streams

    # Extract kernels from stream 8474, sorted by time
    python extract_stream.py trace.json.gz --stream 8474

    # Extract from stream 8474 between 1.0s and 1.1s
    python extract_stream.py trace.json.gz --stream 8474 --start 1000000 --end 1100000

    # Compare two streams side-by-side (e.g. default vs alt)
    python extract_stream.py trace.json.gz --stream 8474 --stream2 8475

    # Export to CSV
    python extract_stream.py trace.json.gz --stream 8474 --csv out.csv

    # Show all streams in a time range
    python extract_stream.py trace.json.gz --start 1000000 --end 1100000

Notes:
    - Times are in microseconds (us) as they appear in the trace.
    - Use --list-streams first to find stream IDs and pick a time range.
    - Kernel names are shortened for readability; use --full-name for raw names.
"""

import argparse
import gzip
import json
import csv
import sys
import io
from pathlib import Path
from collections import defaultdict


class Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, filepath):
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def load_trace(path: str):
    p = Path(path)
    if not p.exists():
        sys.exit(f"[ERROR] File not found: {path}")
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def short_name(name: str, max_len: int = 60) -> str:
    if not name:
        return ""
    s = name
    if "kernel_moe_gemm" in s:        return "kernel_moe_gemm (CK)"
    if "preshuffle" in s:              return "preshuffle_GEMM (CK)"
    if "ABScale" in s:                 return "ABScale_GEMM (CK)"
    if "sm100_fp8_gemm" in s:          return "deep_gemm_fp8"
    if "sm100_bf16_gemm" in s:         return "deep_gemm_bf16"
    if "sm100_fp8_paged_mqa" in s:     return "deep_gemm_fp8_paged_mqa"
    if "allreduce_fus" in s:           return "allreduce_fusion"
    if "RMSNormKernel" in s:           return "RMSNormKernel"
    if "per_token_group_quant" in s:   return "per_token_group_quant"
    if "fused_rms_fp8_group_quant" in s: return "fused_rms_fp8_group_quant"
    if "fused_flatten_fp8" in s:       return "fused_flatten_fp8_quant"
    if "MoeSorting" in s:              return "MoeSortingKernel"
    if "grouped_topk" in s:            return "grouped_topk"
    if "wv_splitk" in s:              return "wv_splitk (router)"
    if "mla_a8w8" in s:               return "mla_a8w8_decode"
    if "mla_reduce" in s:             return "mla_reduce"
    if "fmhaSm100" in s:              return "fmha_sm100_mla"
    if "batched_gemm_a8w8" in s:      return "batched_gemm_a8w8"
    if "fused_qk_rope" in s:          return "fused_qk_rope_cache"
    if "fast_hadamard" in s:          return "fast_hadamard"
    if "Layernorm2dFwd" in s:         return "Layernorm2dFwd"
    if "generalLayerNorm" in s:       return "generalLayerNorm"
    if "elementwise_kernel" in s:     return "elementwise"
    if "vectorized_elementwise" in s: return "vectorized_elementwise"
    if "CatArrayBatchedCopy" in s:    return "CatArrayBatchedCopy"
    if "reduce_scatter" in s:         return "reduce_scatter"
    if "cross_device_reduce" in s:    return "cross_device_reduce"
    if "topk_transform_decode" in s:  return "topk_transform_decode"
    if "fused_rope_kernel" in s:      return "fused_rope"
    if "act_and_mul" in s:            return "act_and_mul"
    if "bmm_E4m3" in s:              return "bmm_E4m3_MoE"
    if "bmm_Bfloat16" in s:          return "bmm_Bfloat16_MoE"
    if "routingIndices" in s:         return "routingIndices"
    if "activationDeepSeek" in s:     return "activationDeepSeek"
    if "finalizeKernel" in s:         return "finalizeKernel"
    if "splitKreduce" in s:           return "splitKreduce"
    if "nvjet_sm100" in s:            return s[s.find("nvjet_sm100"):s.find("nvjet_sm100")+35]
    if "triton_poi_fused" in s:       return s[:50]
    if "triton_" in s:
        m = s[:60]
        return m
    if "Cijk_SS" in s:                return "hipBLAS_GEMM"
    if "kn_entry_2c_sbhd" in s:      return "RoPE_cache_aiter"
    if "indexer_k_quant" in s:        return "indexer_k_quant_cache"
    if "dynamic_per_group_scaled_quant" in s: return "dynamic_per_group_quant"
    if "add_rmsnorm_quant" in s:      return "add_rmsnorm_quant"
    if "local_device_load_rmsnorm" in s: return "allreduce_rmsnorm"
    if "radix_topk" in s:             return "radix_topk"
    if "gluon_deepgemm_fp8" in s:     return "gluon_fp8_paged_mqa"
    if "set_mla_kv_buffer" in s:      return "set_mla_kv_buffer"
    if "RopeQuantize" in s:           return "RopeQuantize"
    if "convert_req_index" in s:      return "convert_req_index"
    if "_act_quant" in s:             return "act_quant"
    if "fused_store_indexer" in s:    return "fused_store_indexer"
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def extract_gpu_kernels(trace_data):
    """Extract all GPU kernel events with stream info."""
    events = trace_data if isinstance(trace_data, list) else trace_data.get("traceEvents", [])
    kernels = []
    for ev in events:
        cat = ev.get("cat", "")
        if cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        if "kernel" not in cat and "gpu" not in cat:
            continue
        name = ev.get("name", "")
        ts = ev.get("ts", 0)
        dur = ev.get("dur", 0)
        args = ev.get("args", {})
        stream = args.get("stream", ev.get("tid", "?"))
        kernels.append({
            "name": name,
            "ts": ts,
            "dur": dur,
            "stream": stream,
            "end": ts + dur,
        })
    kernels.sort(key=lambda x: x["ts"])
    return kernels


def list_streams(kernels):
    """Print summary of all streams."""
    stream_info = defaultdict(lambda: {"count": 0, "min_ts": float("inf"), "max_end": 0, "total_dur": 0})
    for k in kernels:
        s = k["stream"]
        stream_info[s]["count"] += 1
        stream_info[s]["min_ts"] = min(stream_info[s]["min_ts"], k["ts"])
        stream_info[s]["max_end"] = max(stream_info[s]["max_end"], k["end"])
        stream_info[s]["total_dur"] += k["dur"]

    print(f"\n{'Stream':>10s}  {'Kernels':>8s}  {'Start_us':>14s}  {'End_us':>14s}  {'Span_us':>12s}  {'Total_dur_us':>14s}")
    print("-" * 85)
    for sid in sorted(stream_info.keys(), key=lambda x: stream_info[x]["min_ts"]):
        info = stream_info[sid]
        span = info["max_end"] - info["min_ts"]
        print(f"{sid:>10}  {info['count']:>8d}  {info['min_ts']:>14.1f}  {info['max_end']:>14.1f}  {span:>12.1f}  {info['total_dur']:>14.1f}")
    print(f"\nTotal streams: {len(stream_info)}")
    return stream_info


def filter_kernels(kernels, stream=None, start=None, end=None):
    """Filter kernels by stream and time range."""
    result = kernels
    if stream is not None:
        result = [k for k in result if k["stream"] == stream]
    if start is not None:
        result = [k for k in result if k["end"] > start]
    if end is not None:
        result = [k for k in result if k["ts"] <= end]
    return result


def print_kernels(kernels, use_full_name=False, base_ts=None):
    """Print kernel list in a table."""
    if not kernels:
        print("  (no kernels found)")
        return

    if base_ts is None:
        base_ts = kernels[0]["ts"]

    print(f"\n  {'#':>4s}  {'RelStart_us':>12s}  {'Dur_us':>10s}  {'Stream':>8s}  {'Kernel'}")
    print("  " + "-" * 100)
    for i, k in enumerate(kernels):
        rel = k["ts"] - base_ts
        nm = k["name"] if use_full_name else short_name(k["name"])
        print(f"  {i:>4d}  {rel:>12.1f}  {k['dur']:>10.3f}  {k['stream']:>8}  {nm}")
    total_dur = sum(k["dur"] for k in kernels)
    span = kernels[-1]["end"] - kernels[0]["ts"]
    print(f"\n  Kernels: {len(kernels)}  |  Total dur: {total_dur:.1f} us  |  Span: {span:.1f} us")


def compare_streams(kernels, s1, s2, start=None, end=None, use_full_name=False):
    """Side-by-side comparison of two streams."""
    k1 = filter_kernels(kernels, stream=s1, start=start, end=end)
    k2 = filter_kernels(kernels, stream=s2, start=start, end=end)

    all_k = sorted(k1 + k2, key=lambda x: x["ts"])
    if not all_k:
        print("  (no kernels in either stream)")
        return

    base_ts = all_k[0]["ts"]

    print(f"\n  {'#':>4s}  {'RelStart':>10s}  {'Dur_us':>8s}  {'Stream':>8s}  {'Kernel'}")
    print("  " + "-" * 100)
    for i, k in enumerate(all_k):
        rel = k["ts"] - base_ts
        nm = k["name"] if use_full_name else short_name(k["name"])
        marker = "<<" if k["stream"] == s2 else ""
        print(f"  {i:>4d}  {rel:>10.1f}  {k['dur']:>8.3f}  {k['stream']:>8}  {nm}  {marker}")

    for sid, klist in [(s1, k1), (s2, k2)]:
        if klist:
            dur = sum(x["dur"] for x in klist)
            span = klist[-1]["end"] - klist[0]["ts"]
            print(f"  Stream {sid}: {len(klist)} kernels, total={dur:.1f}us, span={span:.1f}us")


def export_csv(kernels, path, use_full_name=False):
    """Export filtered kernels to CSV."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["#", "ts_us", "dur_us", "stream", "kernel_short", "kernel_full"])
        for i, k in enumerate(kernels):
            w.writerow([i, k["ts"], k["dur"], k["stream"],
                        short_name(k["name"]), k["name"]])
    print(f"Exported {len(kernels)} kernels to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract kernels from CUDA streams in a profiler trace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("trace", help="Path to trace file (.json or .json.gz)")
    parser.add_argument("--list-streams", action="store_true",
                        help="List all streams with summary stats")
    parser.add_argument("--stream", type=int, default=None,
                        help="Stream ID to extract")
    parser.add_argument("--stream2", type=int, default=None,
                        help="Second stream ID for side-by-side comparison")
    parser.add_argument("--start", type=float, default=None,
                        help="Start time in us (absolute timestamp from trace)")
    parser.add_argument("--end", type=float, default=None,
                        help="End time in us (absolute timestamp from trace)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Export results to CSV file")
    parser.add_argument("--full-name", action="store_true",
                        help="Show full kernel names instead of shortened")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of kernels to display")
    parser.add_argument("--time-unit", choices=["us", "ns"], default="us",
                        help="Unit for --start/--end values (default: us). "
                             "If 'ns', values are converted to us internally.")
    parser.add_argument("--output", type=str, default=None,
                        help="Also write output to this log file (tee)")

    args = parser.parse_args()

    tee = None
    if args.output:
        tee = Tee(args.output)
        sys.stdout = tee

    if args.time_unit == "ns":
        if args.start is not None:
            args.start /= 1000.0
        if args.end is not None:
            args.end /= 1000.0

    print(f"Loading trace: {args.trace}")
    trace_data = load_trace(args.trace)
    kernels = extract_gpu_kernels(trace_data)
    print(f"Found {len(kernels)} GPU kernels")

    if args.list_streams or (args.stream is None and args.stream2 is None):
        stream_info = list_streams(kernels)
        if args.start is not None or args.end is not None:
            print(f"\nFiltered by time: start={args.start}, end={args.end}")
            fk = filter_kernels(kernels, start=args.start, end=args.end)
            list_streams(fk)
        return

    if args.stream2 is not None:
        s1 = args.stream if args.stream is not None else None
        if s1 is None:
            sys.exit("--stream2 requires --stream")
        print(f"\nComparing stream {s1} vs {args.stream2}")
        compare_streams(kernels, s1, args.stream2,
                        start=args.start, end=args.end,
                        use_full_name=args.full_name)
        return

    fk = filter_kernels(kernels, stream=args.stream, start=args.start, end=args.end)
    print(f"\nStream {args.stream}: {len(fk)} kernels"
          + (f" (time range: {args.start}-{args.end} us)" if args.start or args.end else ""))

    if args.limit and len(fk) > args.limit:
        fk = fk[:args.limit]
        print(f"  (showing first {args.limit})")

    print_kernels(fk, use_full_name=args.full_name)

    if args.csv:
        all_fk = filter_kernels(kernels, stream=args.stream, start=args.start, end=args.end)
        export_csv(all_fk, args.csv, use_full_name=args.full_name)

    if tee:
        sys.stdout = tee.stdout
        tee.close()
        print(f"\nOutput also saved to: {args.output}")


if __name__ == "__main__":
    main()
