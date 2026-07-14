#!/usr/bin/env python3
"""
Quantify the real overlap (in microseconds) achieved by the ProposalA
unit test under HIP graph capture, regardless of which physical stream
label HIP assigns.

Rationale:
- The HTML visualization suggested Proposal A "remapped" streams, but the
  timing improvement (~7 μs) is what's physically achieved.
- Physical stream id in a HIP graph trace is NOT the same as the
  pytorch-level stream we dispatched to.  HIP graph is allowed to
  re-pool physical streams at replay time as long as it respects the
  DAG.
- What matters for performance is: "while something is running on physical
  stream X, is something else running on physical stream Y?"  That's the
  true concurrency.

This script consumes a torch.profiler trace (json.gz) from a single
decode iteration and reports:
  - active busy time on each physical stream
  - concurrency histogram (#streams busy at each μs)
  - per-iter critical-path length (wall clock)
  - effective parallel work = sum(busy) - critical_path

Usage:
  python3 glm5_proposalA_overlap_analysis.py <trace.json.gz> \
      [--iter-len AUTO] [--show-layout]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict


def load_kernels(path):
    with gzip.open(path, "rt") as f:
        t = json.load(f)
    evts = t if isinstance(t, list) else t.get("traceEvents", [])
    kevts = []
    for e in evts:
        if e.get("cat") != "kernel":
            continue
        sid = e.get("args", {}).get("stream")
        if sid is None:
            continue
        kevts.append({
            "ts": e.get("ts", 0),
            "dur": e.get("dur", 0),
            "sid": sid,
            "name": e.get("name", ""),
        })
    kevts.sort(key=lambda e: e["ts"])
    return kevts


def per_stream_stats(kernels):
    """Sum busy time and (ts_min, ts_max) per stream."""
    s = {}
    for k in kernels:
        info = s.setdefault(k["sid"], {"busy": 0, "min_ts": float("inf"),
                                       "max_end": 0, "count": 0})
        info["busy"] += k["dur"]
        info["min_ts"] = min(info["min_ts"], k["ts"])
        info["max_end"] = max(info["max_end"], k["ts"] + k["dur"])
        info["count"] += 1
    return s


def concurrency_profile(kernels, resolution=1.0):
    """Walk the timeline and, at each time step, count how many streams
    have a kernel running.  Returns (bins, hist) where hist[k] = fraction
    of time with exactly k streams busy."""
    if not kernels:
        return {}, 0
    t_min = min(k["ts"] for k in kernels)
    t_max = max(k["ts"] + k["dur"] for k in kernels)
    # Sweep line algorithm
    events = []
    for k in kernels:
        events.append((k["ts"], +1, k["sid"]))
        events.append((k["ts"] + k["dur"], -1, k["sid"]))
    events.sort()
    active_per_stream = defaultdict(int)
    hist = defaultdict(float)  # concurrency -> total duration
    prev_t = t_min
    for t, delta, sid in events:
        busy_streams = sum(1 for v in active_per_stream.values() if v > 0)
        hist[busy_streams] += max(0.0, t - prev_t)
        active_per_stream[sid] += delta
        prev_t = t
    return dict(hist), (t_max - t_min)


def critical_path(kernels, streams):
    """Lower bound on wall-clock time if we had perfect parallelism:
    max busy on any single stream.  Actual wall-clock is t_max - t_min."""
    if not kernels:
        return 0, 0
    t_min = min(k["ts"] for k in kernels)
    t_max = max(k["ts"] + k["dur"] for k in kernels)
    wall = t_max - t_min
    cp = max(s["busy"] for s in streams.values())
    return wall, cp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", help="trace.json.gz")
    ap.add_argument("--iter-len", type=float, default=0,
                    help="If set, clip analysis to N microseconds from the "
                         "start of the 2nd iteration; 0 uses full trace.")
    ap.add_argument("--show-layout", action="store_true",
                    help="Print per-kernel placement (first 40 rows).")
    args = ap.parse_args()

    kernels = load_kernels(args.trace)
    if args.iter_len > 0 and len(kernels) > 10:
        # Clip to a single iteration: use the longest gap in kernels as
        # the iteration boundary.
        # Heuristic: pick a start midway and use iter_len as the window.
        total_span = kernels[-1]["ts"] + kernels[-1]["dur"] - kernels[0]["ts"]
        start = kernels[0]["ts"] + total_span / 3
        end = start + args.iter_len
        kernels = [k for k in kernels if start <= k["ts"] < end]

    print(f"Trace: {args.trace}")
    print(f"  kernels: {len(kernels)}")

    streams = per_stream_stats(kernels)
    wall, cp = critical_path(kernels, streams)

    print(f"\n  Per physical stream:")
    for sid in sorted(streams):
        s = streams[sid]
        print(f"    stream {sid:>3d}: count={s['count']:4d}  "
              f"busy={s['busy']:8.1f}μs  "
              f"span={s['max_end']-s['min_ts']:8.1f}μs")

    total_busy = sum(s["busy"] for s in streams.values())
    print(f"\n  Wall clock    : {wall:10.1f} μs")
    print(f"  Sum of busy   : {total_busy:10.1f} μs")
    print(f"  Critical path : {cp:10.1f} μs   (max single-stream busy)")
    if wall > 0:
        print(f"  Concurrency   : {total_busy/wall:10.2f} x")
        overlap_us = total_busy - wall
        print(f"  Effective //  : {overlap_us:10.1f} μs   "
              f"(= sum_busy - wall; work saved by multi-stream)")

    hist, total_t = concurrency_profile(kernels)
    if total_t > 0:
        print(f"\n  Concurrency histogram:")
        for level in sorted(hist):
            frac = hist[level] / total_t * 100
            bar = "#" * int(frac / 2)
            print(f"    {level} streams busy: {hist[level]:7.1f}μs  "
                  f"({frac:5.1f}%) {bar}")

    if args.show_layout:
        print(f"\n  First 40 kernels:")
        if kernels:
            t0 = kernels[0]["ts"]
            for k in kernels[:40]:
                n = k["name"][:60]
                print(f"    t={k['ts']-t0:7.1f} +{k['dur']:5.1f}  "
                      f"s={k['sid']:>3d}  {n}")


if __name__ == "__main__":
    main()
