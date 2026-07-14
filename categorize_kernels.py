#!/usr/bin/env python3
"""
categorize_kernels.py

Classify GPU kernels by functionality and compare MI355X vs B200 performance.

Takes kernel CSVs from auto_detect_layer.py (or fetch_kernels_from_torch_profiler.py)
and a kernel_categories.yaml config to produce a categorized comparison table.

Usage:
    # Single platform analysis
    python categorize_kernels.py --mi355x mi355x_layer.csv --config kernel_categories.yaml

    # Cross-platform comparison
    python categorize_kernels.py \
        --mi355x mi355x_layer.csv \
        --b200 b200_layer.csv \
        --config kernel_categories.yaml \
        --csv comparison.csv

    # Use with auto_detect_layer.py directly from traces
    python categorize_kernels.py \
        --mi355x-trace mi355x.trace.json.gz \
        --b200-trace b200.trace.json.gz \
        --config kernel_categories.yaml
"""

import argparse
import csv
import re
import sys
import json
from pathlib import Path
from collections import OrderedDict

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load kernel_categories.yaml config."""
    if HAS_YAML:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    else:
        # Minimal YAML parser for our simple format
        return _parse_simple_yaml(path)


def _parse_simple_yaml(path: str) -> dict:
    """Fallback parser for simple YAML without pyyaml dependency."""
    config = {"categories": OrderedDict()}
    current_cat = None

    with open(path, "r") as f:
        for line in f:
            stripped = line.rstrip()
            if not stripped or stripped.startswith("#"):
                continue

            # Top-level keys
            if not line.startswith(" ") and ":" in stripped:
                key, _, val = stripped.partition(":")
                key = key.strip()
                val = val.strip()
                if key == "model":
                    config["model"] = val
                elif key == "layer_module":
                    config["layer_module"] = val
                elif key == "categories":
                    pass  # Categories block starts
                continue

            # Category-level (2-space indent)
            indent = len(line) - len(line.lstrip())
            content = stripped.strip()

            if indent == 2 and content.endswith(":"):
                current_cat = content[:-1]
                config["categories"][current_cat] = {
                    "module": "",
                    "description": "",
                    "patterns": [],
                }
            elif indent == 4 and current_cat:
                if content.startswith("module:"):
                    config["categories"][current_cat]["module"] = content.split(":", 1)[1].strip()
                elif content.startswith("description:"):
                    desc = content.split(":", 1)[1].strip().strip('"')
                    config["categories"][current_cat]["description"] = desc
                elif content.startswith("patterns:"):
                    pass  # patterns list follows
            elif indent == 6 and current_cat and content.startswith("- "):
                pattern = content[2:].strip().strip('"')
                config["categories"][current_cat]["patterns"].append(pattern)

    return config


# ---------------------------------------------------------------------------
# Kernel CSV loading
# ---------------------------------------------------------------------------

def load_kernel_csv(path: str) -> list[dict]:
    """Load kernel CSV (name, ts, ts_end, dur, cat, pid, tid)."""
    kernels = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kernels.append({
                "name": row["name"],
                "ts": float(row["ts"]),
                "ts_end": float(row["ts_end"]),
                "dur": float(row["dur"]),
                "cat": row.get("cat", "kernel"),
                "pid": row.get("pid", ""),
                "tid": row.get("tid", ""),
            })
    return kernels


# ---------------------------------------------------------------------------
# Categorization
# ---------------------------------------------------------------------------

def categorize_kernel(name: str, categories: dict) -> str:
    """Match a kernel name to the first matching category."""
    for cat_name, cat_info in categories.items():
        for pattern in cat_info.get("patterns", []):
            if re.search(pattern, name):
                return cat_name
    return "Other"


def categorize_all(kernels: list[dict], categories: dict) -> dict[str, list[dict]]:
    """Group kernels by category. Returns {category: [kernel, ...]}."""
    result = OrderedDict()
    # Initialize all categories (preserve config order)
    for cat_name in categories:
        result[cat_name] = []

    for k in kernels:
        cat = categorize_kernel(k["name"], categories)
        if cat not in result:
            result[cat] = []
        result[cat].append(k)

    return result


def aggregate_by_category(categorized: dict[str, list[dict]]) -> OrderedDict:
    """Compute total duration per category."""
    agg = OrderedDict()
    for cat, kernels in categorized.items():
        if not kernels:
            continue
        total_dur = sum(k["dur"] for k in kernels)
        agg[cat] = {
            "count": len(kernels),
            "total_dur": total_dur,
            "kernels": kernels,
        }
    return agg


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def fmt_dur(us: float) -> str:
    if us >= 1_000_000:
        return f"{us/1_000_000:.3f} s"
    if us >= 1_000:
        return f"{us/1_000:.3f} ms"
    return f"{us:.1f} us"


def print_single_platform(agg: dict, categories: dict, platform: str = "MI355X") -> None:
    total = sum(v["total_dur"] for v in agg.values())
    print(f"\n{'='*80}")
    print(f" {platform} Kernel Breakdown (1 DecoderLayer)")
    print(f"{'='*80}")

    current_module = None
    for cat_name in categories:
        if cat_name not in agg:
            continue
        info = agg[cat_name]
        module = categories[cat_name].get("module", "")

        if module != current_module:
            current_module = module
            print(f"\n  [{module}]")

        pct = info["total_dur"] / total * 100 if total > 0 else 0
        print(f"    {cat_name:<22} {info['count']:>3} kernels  "
              f"{fmt_dur(info['total_dur']):>12}  ({pct:5.1f}%)")

        # List individual kernels
        for k in info["kernels"]:
            short = k["name"][:60] + "..." if len(k["name"]) > 60 else k["name"]
            print(f"      {fmt_dur(k['dur']):>10}  {short}")

    print(f"\n  {'TOTAL':<22} {sum(v['count'] for v in agg.values()):>3} kernels  "
          f"{fmt_dur(total):>12}")
    print(f"{'='*80}\n")


def print_comparison(mi355x_agg: dict, b200_agg: dict, categories: dict) -> None:
    all_cats = list(categories.keys())
    mi_total = sum(v["total_dur"] for v in mi355x_agg.values())
    b2_total = sum(v["total_dur"] for v in b200_agg.values())

    print(f"\n{'='*100}")
    print(f" MI355X vs B200 Comparison (1 DecoderLayer)")
    print(f"{'='*100}")
    print(f"  {'Category':<22} {'Module':<14} "
          f"{'MI355X':>10} {'B200':>10} {'B200/MI355X':>12} {'Speedup':>8}")
    print(f"  {'-'*22} {'-'*14} {'-'*10} {'-'*10} {'-'*12} {'-'*8}")

    current_module = None
    module_mi = 0
    module_b2 = 0

    for cat_name in all_cats:
        mi = mi355x_agg.get(cat_name, {"total_dur": 0, "count": 0})
        b2 = b200_agg.get(cat_name, {"total_dur": 0, "count": 0})
        module = categories[cat_name].get("module", "")
        mi_dur = mi["total_dur"] if isinstance(mi, dict) else 0
        b2_dur = b2["total_dur"] if isinstance(b2, dict) else 0

        if module != current_module:
            if current_module is not None and module_mi > 0:
                ratio = module_b2 / module_mi if module_mi > 0 else float("inf")
                print(f"  {'  Subtotal':<22} {'':<14} "
                      f"{fmt_dur(module_mi):>10} {fmt_dur(module_b2):>10} "
                      f"{ratio:>11.0%} {module_mi/module_b2 if module_b2>0 else 0:>7.2f}x")
                print()
            current_module = module
            module_mi = 0
            module_b2 = 0
            print(f"  [{module}]")

        module_mi += mi_dur
        module_b2 += b2_dur

        if mi_dur == 0 and b2_dur == 0:
            continue

        if mi_dur > 0 and b2_dur > 0:
            ratio = b2_dur / mi_dur
            speedup = mi_dur / b2_dur
            ratio_s = f"{ratio:>11.0%}"
            speedup_s = f"{speedup:>7.2f}x"
        else:
            ratio_s = f"{'N/A':>12}"
            speedup_s = f"{'N/A':>8}"

        print(f"    {cat_name:<22} {module:<14} "
              f"{fmt_dur(mi_dur):>10} {fmt_dur(b2_dur):>10} "
              f"{ratio_s} {speedup_s}")

    # Last module subtotal
    if module_mi > 0:
        ratio = module_b2 / module_mi if module_mi > 0 else float("inf")
        print(f"  {'  Subtotal':<22} {'':<14} "
              f"{fmt_dur(module_mi):>10} {fmt_dur(module_b2):>10} "
              f"{ratio:>11.0%} {module_mi/module_b2 if module_b2>0 else 0:>7.2f}x")

    print(f"\n  {'TOTAL':<22} {'':<14} "
          f"{fmt_dur(mi_total):>10} {fmt_dur(b2_total):>10} "
          f"{b2_total/mi_total if mi_total>0 else 0:>11.0%} "
          f"{mi_total/b2_total if b2_total>0 else 0:>7.2f}x")
    print(f"{'='*100}\n")


def write_comparison_csv(mi355x_agg: dict, b200_agg: dict,
                         categories: dict, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Category", "Module",
            "MI355X_Kernels", "MI355X_Time_us",
            "B200_Kernels", "B200_Time_us",
            "B200/MI355X", "Potential_Speedup",
            "MI355X_Kernel_Names", "B200_Kernel_Names",
        ])

        for cat_name in categories:
            mi = mi355x_agg.get(cat_name, {"total_dur": 0, "count": 0, "kernels": []})
            b2 = b200_agg.get(cat_name, {"total_dur": 0, "count": 0, "kernels": []})
            module = categories[cat_name].get("module", "")
            mi_dur = mi["total_dur"] if isinstance(mi, dict) else 0
            b2_dur = b2["total_dur"] if isinstance(b2, dict) else 0
            mi_count = mi["count"] if isinstance(mi, dict) else 0
            b2_count = b2["count"] if isinstance(b2, dict) else 0

            if mi_dur == 0 and b2_dur == 0:
                continue

            ratio = b2_dur / mi_dur if mi_dur > 0 else ""
            speedup = mi_dur / b2_dur if b2_dur > 0 else ""

            mi_names = "; ".join(k["name"] for k in mi.get("kernels", []))
            b2_names = "; ".join(k["name"] for k in b2.get("kernels", []))

            writer.writerow([
                cat_name, module,
                mi_count, f"{mi_dur:.1f}",
                b2_count, f"{b2_dur:.1f}",
                f"{ratio:.4f}" if ratio else "N/A",
                f"{speedup:.2f}" if speedup else "N/A",
                mi_names, b2_names,
            ])

    print(f"[INFO] Comparison CSV written to: {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Optional: extract from trace directly
# ---------------------------------------------------------------------------

def extract_layer_from_trace(trace_path: str, layer_index: int = 2) -> list[dict]:
    """Run auto_detect_layer logic inline."""
    from auto_detect_layer import (
        load_trace, extract_kernels, find_busiest_stream,
        short_name, detect_layer_pattern, find_layer_boundaries,
    )
    trace = load_trace(trace_path)
    stream = find_busiest_stream(trace)
    kernels = extract_kernels(trace, stream=stream)
    names = [short_name(k["name"]) for k in kernels]
    pattern, offset = detect_layer_pattern(names)
    layers = find_layer_boundaries(kernels, pattern, offset)

    if layer_index >= len(layers):
        sys.exit(f"[ERROR] Layer index {layer_index} out of range "
                 f"(found {len(layers)} layers)")

    start, end = layers[layer_index]
    return kernels[start:end + 1]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Categorize GPU kernels and compare MI355X vs B200.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", required=True,
                   help="Path to kernel_categories.yaml")

    # Input: CSV or trace
    g = p.add_argument_group("Input (CSV)")
    g.add_argument("--mi355x", metavar="CSV", help="MI355X kernel CSV")
    g.add_argument("--b200", metavar="CSV", help="B200 kernel CSV")

    g2 = p.add_argument_group("Input (trace, auto-detect)")
    g2.add_argument("--mi355x-trace", metavar="TRACE",
                    help="MI355X trace JSON (auto-detect layer)")
    g2.add_argument("--b200-trace", metavar="TRACE",
                    help="B200 trace JSON (auto-detect layer)")
    g2.add_argument("--layer-index", type=int, default=2,
                    help="Layer index for auto-detect (default=2)")

    # Output
    p.add_argument("--csv", metavar="FILE",
                   help="Write comparison CSV")
    p.add_argument("--json", metavar="FILE",
                   help="Write comparison JSON (for optimization_projection.py)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    categories = config.get("categories", {})

    # Load MI355X kernels
    mi355x_kernels = None
    if args.mi355x:
        mi355x_kernels = load_kernel_csv(args.mi355x)
    elif args.mi355x_trace:
        mi355x_kernels = extract_layer_from_trace(
            args.mi355x_trace, args.layer_index)

    # Load B200 kernels
    b200_kernels = None
    if args.b200:
        b200_kernels = load_kernel_csv(args.b200)
    elif args.b200_trace:
        b200_kernels = extract_layer_from_trace(
            args.b200_trace, args.layer_index)

    if mi355x_kernels is None and b200_kernels is None:
        sys.exit("[ERROR] Provide at least one input: --mi355x/--mi355x-trace "
                 "or --b200/--b200-trace")

    # Categorize
    if mi355x_kernels:
        mi355x_cat = categorize_all(mi355x_kernels, categories)
        mi355x_agg = aggregate_by_category(mi355x_cat)
    else:
        mi355x_agg = {}

    if b200_kernels:
        b200_cat = categorize_all(b200_kernels, categories)
        b200_agg = aggregate_by_category(b200_cat)
    else:
        b200_agg = {}

    # Display
    if mi355x_kernels and b200_kernels:
        print_comparison(mi355x_agg, b200_agg, categories)
        if args.csv:
            write_comparison_csv(mi355x_agg, b200_agg, categories, args.csv)
    elif mi355x_kernels:
        print_single_platform(mi355x_agg, categories, "MI355X")
    elif b200_kernels:
        print_single_platform(b200_agg, categories, "B200")

    # JSON output for pipeline
    if args.json:
        output = {
            "model": config.get("model", ""),
            "categories": {},
        }
        for cat_name in categories:
            mi = mi355x_agg.get(cat_name, {"total_dur": 0, "count": 0})
            b2 = b200_agg.get(cat_name, {"total_dur": 0, "count": 0})
            output["categories"][cat_name] = {
                "module": categories[cat_name].get("module", ""),
                "mi355x_time": mi["total_dur"] if isinstance(mi, dict) else 0,
                "mi355x_count": mi["count"] if isinstance(mi, dict) else 0,
                "b200_time": b2["total_dur"] if isinstance(b2, dict) else 0,
                "b200_count": b2["count"] if isinstance(b2, dict) else 0,
            }

        with open(args.json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[INFO] JSON written to: {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
