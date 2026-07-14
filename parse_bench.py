import os
import re
import csv
import argparse
from statistics import mean


CONFIG_RE = re.compile(
    r"i(?P<i>\d+)-o(?P<o>\d+)-n(?P<n>\d+)-concurrency(?P<c>\d+)"
)

E2E_RE = re.compile(r"Median E2E Latency \(ms\):\s+([\d.]+)")
TTFT_RE = re.compile(r"Median TTFT \(ms\):\s+([\d.]+)")
ITL_RE = re.compile(r"Median ITL \(ms\):\s+([\d.]+)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse sglang bench logs and summarize median latencies."
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=".",
        help="Root directory containing benchmark folders (default: current directory)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="summary.csv",
        help="Output CSV file name (default: summary.csv)",
    )
    return parser.parse_args()


def parse_bench_dir(root_dir):
    results = []

    for entry in os.listdir(root_dir):
        entry_path = os.path.join(root_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        m = CONFIG_RE.fullmatch(entry)
        if not m:
            continue

        i = int(m.group("i"))
        o = int(m.group("o"))
        n = int(m.group("n"))
        c = int(m.group("c"))

        e2e_list, ttft_list, itl_list = [], [], []

        for fname in os.listdir(entry_path):
            if not fname.startswith("bench") or not fname.endswith(".log"):
                continue

            fpath = os.path.join(entry_path, fname)
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            e2e = E2E_RE.search(text)
            ttft = TTFT_RE.search(text)
            itl = ITL_RE.search(text)

            if e2e and ttft and itl:
                e2e_list.append(float(e2e.group(1)))
                ttft_list.append(float(ttft.group(1)))
                itl_list.append(float(itl.group(1)))

        if e2e_list:
            results.append({
                "config": entry,
                "i": i,
                "o": o,
                "n": n,
                "c": c,
                "E2E": mean(e2e_list),
                "TTFT": mean(ttft_list),
                "ITL": mean(itl_list),
            })

    return results


def write_csv(results, output_path):
    results.sort(key=lambda x: (x["i"], x["o"], x["n"], x["c"]))

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "E2E", "TTFT", "ITL"])
        for r in results:
            writer.writerow([
                r["config"],
                f"{r['E2E']:.2f}",
                f"{r['TTFT']:.2f}",
                f"{r['ITL']:.2f}",
            ])


def main():
    args = parse_args()
    results = parse_bench_dir(args.root_dir)
    write_csv(results, args.output)
    print(f"[OK] Saved summary to {args.output}")


if __name__ == "__main__":
    main()


'''
python parse_bench.py --root-dir ~/prof/0123_bench --output ~/prof/0123_bench/summary.csv
'''