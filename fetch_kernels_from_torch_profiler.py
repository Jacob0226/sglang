#!/usr/bin/env python3
"""
fetch_torch_profiler_kernels.py

Parse a PyTorch profiler JSON trace (Chrome Trace Format) and display
GPU kernels with their start/end timestamps, optionally filtered by ts range.

The 'ts' field in the trace is a high-resolution timestamp in microseconds
(same unit used by Chrome Trace Format / CUDA profiler).

Usage:
    python fetch_torch_profiler_kernels.py <trace.json> [OPTIONS]

    --start FLOAT   Filter: ts >= START  (copy ts value from trace)
    --end   FLOAT   Filter: ts <= END    (copy ts value from trace)
    --sort  FIELD   Sort by: start (default), end, dur, name
    --top   N       Show only top N results (by duration, descending)
    --csv           Output as CSV
    --no-header     Suppress table header
    --list-all      Debug: list all event categories in the trace

Examples:
    # Show all GPU kernels
    python fetch_torch_profiler_kernels.py trace.json

    # Filter by ts range (paste the ts value directly from the trace)
    python fetch_torch_profiler_kernels.py trace.json \\
        --start 6195887545100 --end 6195887600000

    # Top 20 longest kernels
    python fetch_torch_profiler_kernels.py trace.json --top 20 --sort dur

    # Export to CSV
    python fetch_torch_profiler_kernels.py trace.json --csv > kernels.csv
"""

import argparse
import gzip
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# GPU kernel category heuristics (Chrome Trace Format emitted by PyTorch)
# ---------------------------------------------------------------------------
# Only actual GPU-side execution events.
# "cuda_runtime" (e.g. cudaLaunchKernelExC) is a CPU-side launch call — excluded.
GPU_CATEGORIES = {
    "kernel",      # actual GPU kernel execution  (pid=0, tid=stream_id)
    "gpu_memcpy",  # cudaMemcpy on GPU
    "gpu_memset",  # cudaMemset on GPU
}


def is_gpu_kernel(event: dict) -> bool:
    """Return True only for GPU-side execution events."""
    cat = event.get("cat", "").lower()
    return cat in GPU_CATEGORIES


# ---------------------------------------------------------------------------
# Loader: supports plain JSON and gzipped JSON
# ---------------------------------------------------------------------------

def load_trace(path: str) -> dict | list:
    p = Path(path)
    if not p.exists():
        sys.exit(f"[ERROR] File not found: {path}")

    opener = gzip.open if path.endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        sys.exit(f"[ERROR] Failed to parse JSON: {e}")


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_kernels(trace: dict | list) -> list[dict]:
    """
    Extract GPU kernel events and return:
        [ { name, ts, ts_end, dur, cat, pid, tid, args }, ... ]

    Field names mirror the trace JSON:
        ts     = event start (same as trace 'ts', microseconds)
        ts_end = ts + dur
        dur    = event duration (same as trace 'dur', microseconds)
    """
    events = trace if isinstance(trace, list) else trace.get("traceEvents", [])

    kernels = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("ph") != "X":          # only Complete events have ts+dur
            continue
        if not is_gpu_kernel(ev):
            continue

        ts  = float(ev.get("ts",  0))
        dur = float(ev.get("dur", 0))
        kernels.append({
            "name":   ev.get("name", "<unknown>"),
            "ts":     ts,
            "ts_end": ts + dur,
            "dur":    dur,
            "cat":    ev.get("cat", ""),
            "pid":    ev.get("pid", ""),
            "tid":    ev.get("tid", ""),
            "args":   ev.get("args", {}),
        })

    return kernels


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

# ts values are ~13-digit numbers; give them enough width
TS_WIDTH  = 20   # for ts / ts_end columns
DUR_WIDTH = 14   # for formatted duration
CAT_WIDTH = 16
NAME_WIDTH = 46


def fmt_dur(value: float) -> str:
    """Human-readable duration, always showing the raw us value too."""
    if value >= 1_000_000:
        return f"{value/1_000_000:.3f} s"
    if value >= 1_000:
        return f"{value/1_000:.3f} ms"
    return f"{value:.3f} us"


def print_table(kernels: list[dict], no_header: bool = False) -> None:
    # Name column width adapts to the longest name in the result set
    name_w = max(len(k["name"]) for k in kernels)
    name_w = max(name_w, len("KERNEL NAME"))

    total_width = name_w + TS_WIDTH + TS_WIDTH + DUR_WIDTH + CAT_WIDTH + 6
    sep = "-" * total_width

    if not no_header:
        print(sep)
        print(
            f"{'KERNEL NAME':<{name_w}}"
            f"{'ts (start)':>{TS_WIDTH}}"
            f"{'ts_end':>{TS_WIDTH}}"
            f"{'dur':>{DUR_WIDTH}}"
            f"  {'cat':<{CAT_WIDTH}}"
        )
        print(sep)

    for k in kernels:
        print(
            f"{k['name']:<{name_w}}"
            f"{k['ts']:>{TS_WIDTH}.3f}"
            f"{k['ts_end']:>{TS_WIDTH}.3f}"
            f"  {fmt_dur(k['dur']):>{DUR_WIDTH}}"
            f"  {k['cat']:<{CAT_WIDTH}}"
        )

    if not no_header:
        print(sep)
        print(f"Total: {len(kernels)} kernel(s)")


def print_csv(kernels: list[dict], path: str | None = None) -> None:
    import csv
    if path:
        fh = open(path, "w", newline="", encoding="utf-8")
    else:
        fh = sys.stdout
    try:
        writer = csv.writer(fh)
        writer.writerow(["name", "ts", "ts_end", "dur", "cat", "pid", "tid"])
        for k in kernels:
            writer.writerow([
                k["name"], k["ts"], k["ts_end"], k["dur"],
                k["cat"], k["pid"], k["tid"],
            ])
    finally:
        if path:
            fh.close()
            print(f"[INFO] CSV written to: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Extract GPU kernels from a PyTorch profiler JSON trace.\n"
            "--start / --end accept the raw 'ts' value from the trace (microseconds)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("trace", help="Path to profiler JSON (or .json.gz)")
    p.add_argument(
        "--start", type=float, default=None, metavar="TS",
        help="Keep kernels where ts >= START  (microseconds, same as trace JSON)",
    )
    p.add_argument(
        "--end", type=float, default=None, metavar="TS",
        help="Keep kernels where ts <= END    (microseconds, same as trace JSON)",
    )
    p.add_argument(
        "--ns", action="store_true",
        help="Interpret --start/--end as nanoseconds (Perfetto format). "
             "Values are divided by 1000 before filtering.",
    )
    p.add_argument(
        "--sort", choices=["start", "end", "dur", "name"], default="start",
        help="Sort by: start (default), end, dur, name",
    )
    p.add_argument(
        "--top", type=int, default=None, metavar="N",
        help="Keep only top N kernels by duration (longest first)",
    )
    p.add_argument("--csv", metavar="FILE", nargs="?", const="-",
                   help="Output as CSV. Optionally specify a file path (e.g. --csv out.csv). "
                        "Without a path, prints to stdout.")
    p.add_argument("--no-header", action="store_true", help="Suppress header row")
    p.add_argument(
        "--list-all", action="store_true",
        help="Debug: list all unique event categories in the trace",
    )
    return p


SORT_KEY = {
    "start": lambda k: k["ts"],
    "end":   lambda k: k["ts_end"],
    "dur":   lambda k: k["dur"],
    "name":  lambda k: k["name"],
}


def main() -> None:
    args = build_parser().parse_args()
    trace = load_trace(args.trace)

    # List all unique GPU kernel names (cat: "kernel" only)
    if args.list_all:
        kernels = extract_kernels(trace)
        names = sorted({k["name"] for k in kernels if k["cat"] == "kernel"})
        print(f"All GPU kernel names ({len(names)} unique):")
        for n in names:
            print(f"  {n}")
        return

    kernels = extract_kernels(trace)

    if not kernels:
        print("[WARN] No GPU kernels found. Try --list-all to inspect categories.")
        sys.exit(0)

    # Print ts range info so user knows what values to pass
    if not args.csv:
        ts_min = min(k["ts"] for k in kernels)
        ts_max = max(k["ts"] for k in kernels)
        print(f"[INFO] ts range in trace: {ts_min:.3f} ~ {ts_max:.3f}")

    # Convert Perfetto ns → trace us if --ns given
    start_us = args.start / 1000 if (args.start is not None and args.ns) else args.start
    end_us   = args.end   / 1000 if (args.end   is not None and args.ns) else args.end
    if args.ns and (args.start is not None or args.end is not None):
        print(f"[INFO] --ns: converted to us  start={start_us}  end={end_us}")

    # Filter by ts
    if start_us is not None:
        kernels = [k for k in kernels if k["ts"] >= start_us]
    if end_us is not None:
        kernels = [k for k in kernels if k["ts"] <= end_us]
    
    # print(f"[DEBUG] {kernels[0]['ts']}")

    if not kernels:
        print("[INFO] No kernels match the given --start/--end range.")
        sys.exit(0)

    # Sort
    reverse = args.sort == "dur"
    kernels.sort(key=SORT_KEY[args.sort], reverse=reverse)

    # Top-N by duration, then re-sort
    if args.top is not None:
        kernels = sorted(kernels, key=lambda k: k["dur"], reverse=True)[:args.top]
        kernels.sort(key=SORT_KEY[args.sort], reverse=reverse)

    print_table(kernels, no_header=args.no_header)
    if args.csv is not None:
        csv_path = None if args.csv == "-" else args.csv
        print_csv(kernels, path=csv_path)


if __name__ == "__main__":
    main()



# Example on Windows:
# python .\fetch_torch_profiler_kernels.py `
#     C:\Users\jacchang\Downloads\GLM\tmp_m15-21\MI355X_GLM4.7_FP8_i128_o128_c4_p8_step10_disableGraph-TP-0-DECODE.trace.json.gz `
#     --ns --start 6181131327395776 --end 6181131327399776 `
#     --csv C:\Users\jacchang\Desktop\MI355X_GLM4.7_FP8_1_Decode.csv
#     --csv C:\Users\jacchang\Downloads\GLM\tmp_m15-21\MI355X_GLM4.7_FP8_1_Decode.csv
