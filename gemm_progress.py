#!/usr/bin/env python3
"""
Live tqdm progress bar for one backend's tuning pass.

Tracks progress by summing the latest "batch N/M" across all per-GPU compare
reports (these headers are written live, and count ATTEMPTED shapes incl.
no-solution/timeout ones -- unlike candidate.csv which only counts solved
shapes). Stops when the sentinel file appears (orchestrator touches it after
the backend's tuners finish) or when progress reaches total.

Usage:
  python3 gemm_progress.py --backend-dir <dir> --total <N> [--poll 2]
"""
import argparse, glob, os, re, time

BATCH_RE = re.compile(r"batch (\d+)/(\d+)")


def progress(backend_dir):
    """Sum of latest batch index across all per-GPU compare reports."""
    n = 0
    for f in glob.glob(os.path.join(backend_dir, "compare_tmp", "aiter_compare", "*.compare.txt")):
        try:
            txt = open(f).read()
        except OSError:
            continue
        m = BATCH_RE.findall(txt)
        if m:
            n += int(m[-1][0])
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend-dir", required=True)
    ap.add_argument("--total", type=int, required=True)
    ap.add_argument("--poll", type=float, default=2.0)
    args = ap.parse_args()
    try:
        from tqdm import tqdm
    except ImportError:
        return  # no tqdm -> silently skip (tuning still runs)

    sentinel = os.path.join(args.backend_dir, ".monitor_stop")
    desc = os.path.basename(args.backend_dir.rstrip("/"))
    total = max(args.total, 1)
    bar = tqdm(total=total, desc=f"{desc:>10}", unit="shape", dynamic_ncols=True)
    last = 0
    while True:
        n = min(progress(args.backend_dir), total)
        if n > last:
            bar.update(n - last)
            last = n
        if os.path.exists(sentinel) or n >= total:
            break
        time.sleep(args.poll)
    if last < total:
        bar.update(total - last)
    bar.close()


if __name__ == "__main__":
    main()
