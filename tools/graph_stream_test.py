#!/usr/bin/env python3
"""
Test: does CUDA/HIP Graph preserve N independent stream assignments?
Works on both NVIDIA (CUDA) and AMD (HIP/ROCm) — auto-detected.

Each stream runs a long-running, low-CU kernel (vector add in a loop)
so they should all be visible and overlapping in the profiler trace.
"""

import torch
import os
import argparse

IS_HIP = torch.version.hip is not None
BACKEND = "HIP" if IS_HIP else "CUDA"

NUM_STREAMS = 10
N = 1024
REPEATS = 50


def make_work(device="cuda"):
    a = torch.randn(N, device=device)
    b = torch.randn(N, device=device)
    o = torch.zeros(N, device=device)
    return a, b, o


def run_on_streams(streams, data):
    cur = torch.cuda.current_stream()
    for i, s in enumerate(streams):
        s.wait_stream(cur)
        a, b, o = data[i]
        with torch.cuda.stream(s):
            for _ in range(REPEATS):
                torch.add(a, b, out=o)
    for s in streams:
        cur.wait_stream(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str,
                   default=os.path.expanduser("~/SGLang-benchmarks/tmp/trace"))
    p.add_argument("--tag", type=str, default="",
                   help="Extra tag appended to filenames")
    args = p.parse_args()

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0).replace(" ", "_")
    tag = f"_{args.tag}" if args.tag else ""
    prefix = f"stream_test_{BACKEND}_{gpu_name}{tag}"

    print(f"Backend: {BACKEND}  GPU: {gpu_name}  Streams: {NUM_STREAMS}")
    print(f"torch.version.{'hip' if IS_HIP else 'cuda'} = "
          f"{torch.version.hip if IS_HIP else torch.version.cuda}")

    streams = [torch.cuda.Stream(device=device) for _ in range(NUM_STREAMS)]
    data = [make_work(device) for _ in range(NUM_STREAMS)]

    os.makedirs(args.out_dir, exist_ok=True)

    # warmup
    for _ in range(5):
        run_on_streams(streams, data)
    torch.cuda.synchronize()

    # ── Eager profile ──
    print(f"[{gpu_name}] Profiling eager mode ({NUM_STREAMS} streams) ...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
        for _ in range(3):
            run_on_streams(streams, data)
        torch.cuda.synchronize()

    eager_path = os.path.join(args.out_dir, f"{prefix}_eager.json.gz")
    prof.export_chrome_trace(eager_path)
    print(f"  EAGER TRACE: {eager_path}")

    # ── Graph capture + profile ──
    print(f"[{gpu_name}] Capturing {BACKEND} Graph ...")
    for _ in range(5):
        run_on_streams(streams, data)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run_on_streams(streams, data)
    torch.cuda.synchronize()

    print(f"[{gpu_name}] Profiling graph replay ...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()

    graph_path = os.path.join(args.out_dir, f"{prefix}_graph.json.gz")
    prof.export_chrome_trace(graph_path)
    print(f"  GRAPH TRACE: {graph_path}")

    # ── Quick summary ──
    import json, gzip
    for label, path in [("EAGER", eager_path), ("GRAPH", graph_path)]:
        with gzip.open(path, "rt") as f:
            trace = json.load(f)
        events = trace if isinstance(trace, list) else trace.get("traceEvents", [])
        gpu_streams = set()
        n_kern = 0
        for e in events:
            if e.get("cat") == "kernel":
                n_kern += 1
                s = e.get("args", {}).get("stream")
                if s is not None:
                    gpu_streams.add(s)
        print(f"  {label}: {n_kern} kernels, {len(gpu_streams)} streams: {sorted(gpu_streams)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
