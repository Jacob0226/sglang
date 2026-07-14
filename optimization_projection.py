#!/usr/bin/env python3
"""
optimization_projection.py

Compute optimization projections: if MI355X kernels matched B200 speed,
how much overall speedup would we get?

Reads the categorized comparison JSON from categorize_kernels.py and
computes per-category and cumulative optimization projections.

Usage:
    python optimization_projection.py \
        --input comparison.json \
        --moe-layers 89 \
        --ffn-layers 3 \
        --allreduce-mi355x 1806668 \
        --allreduce-b200 17580 \
        --e2e-b200 13959760

    # Or use directly with traces:
    python optimization_projection.py \
        --mi355x-trace mi355x.trace.json.gz \
        --b200-trace b200.trace.json.gz \
        --config kernel_categories.yaml \
        --moe-layers 89
"""

import argparse
import csv
import json
import sys
from collections import OrderedDict


def fmt_dur(us: float) -> str:
    if us >= 1_000_000:
        return f"{us/1_000_000:.3f} s"
    if us >= 1_000:
        return f"{us/1_000:.3f} ms"
    return f"{us:.1f} us"


def load_comparison_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def compute_projections(data: dict,
                        moe_layers: int = 89,
                        ffn_layers: int = 3,
                        allreduce_mi355x: float = 0,
                        allreduce_b200: float = 0,
                        e2e_b200: float = 0) -> dict:
    """
    Compute optimization projections.

    For each category:
    - potential_speedup = mi355x_time / b200_time (if MI355X is slower)
    - weight = mi355x_time * moe_layers (total time in the model)
    - savings = weight - (b200_time * moe_layers) (time saved if optimized)

    Cumulative: accumulate savings to see total improvement.
    """
    categories = data.get("categories", {})

    # Total 1-layer time
    mi_layer_total = sum(c["mi355x_time"] for c in categories.values())
    b2_layer_total = sum(c["b200_time"] for c in categories.values())

    # Full model time
    mi_total = mi_layer_total * moe_layers
    b2_total = b2_layer_total * moe_layers

    # Add allreduce and FFN (if provided)
    if allreduce_mi355x > 0:
        mi_total += allreduce_mi355x
    if allreduce_b200 > 0:
        b2_total += allreduce_b200

    results = OrderedDict()
    cumulative_savings = 0

    for cat_name, info in categories.items():
        mi_time = info["mi355x_time"]
        b2_time = info["b200_time"]

        if mi_time <= 0:
            results[cat_name] = {
                "module": info.get("module", ""),
                "mi355x_time": mi_time,
                "b200_time": b2_time,
                "potential_speedup": None,
                "mi355x_weight": 0,
                "savings": 0,
                "cumulative_savings": cumulative_savings,
                "overall_speedup_pct": None,
                "note": "No MI355X data",
            }
            continue

        if b2_time <= 0:
            results[cat_name] = {
                "module": info.get("module", ""),
                "mi355x_time": mi_time,
                "b200_time": b2_time,
                "potential_speedup": None,
                "mi355x_weight": mi_time * moe_layers,
                "savings": 0,
                "cumulative_savings": cumulative_savings,
                "overall_speedup_pct": None,
                "note": "No B200 data",
            }
            continue

        potential_speedup = mi_time / b2_time
        weight = mi_time * moe_layers

        if potential_speedup > 1.0:
            # MI355X is slower → room for optimization
            savings = (mi_time - b2_time) * moe_layers
        else:
            # MI355X is already faster or equal → no savings (or negative = B200 is faster here)
            savings = 0

        cumulative_savings += savings

        # Overall speedup if this category is optimized to B200 level
        if mi_total > 0:
            overall_speedup_pct = savings / mi_total * 100
        else:
            overall_speedup_pct = 0

        results[cat_name] = {
            "module": info.get("module", ""),
            "mi355x_time": mi_time,
            "b200_time": b2_time,
            "potential_speedup": potential_speedup,
            "mi355x_weight": weight,
            "savings": savings,
            "cumulative_savings": cumulative_savings,
            "overall_speedup_pct": overall_speedup_pct,
            "note": "" if potential_speedup > 1.0 else "B200 slower or equal",
        }

    # Final projections
    optimized_mi_total = mi_total - cumulative_savings
    gap_ratio = optimized_mi_total / b2_total if b2_total > 0 else float("inf")
    gap_vs_b200 = (e2e_b200 / optimized_mi_total * 100) if (e2e_b200 > 0 and optimized_mi_total > 0) else None

    summary = {
        "mi355x_1layer": mi_layer_total,
        "b200_1layer": b2_layer_total,
        "b200_over_mi355x_1layer": b2_layer_total / mi_layer_total if mi_layer_total > 0 else 0,
        "moe_layers": moe_layers,
        "mi355x_total": mi_total,
        "b200_total": b2_total,
        "cumulative_savings": cumulative_savings,
        "optimized_mi355x_total": optimized_mi_total,
        "gap_ratio": gap_ratio,
        "gap_vs_b200_e2e": gap_vs_b200,
    }

    return {
        "categories": results,
        "summary": summary,
    }


def print_projection_table(projections: dict) -> None:
    categories = projections["categories"]
    summary = projections["summary"]

    print(f"\n{'='*110}")
    print(f" Optimization Projection")
    print(f"{'='*110}")
    print(f"  {'Category':<22} {'Module':<12} {'KernelSpd':>10} "
          f"{'Savings':>12} {'OverallSpd%':>12} "
          f"{'CumSavings':>12} {'CumSpd%':>10}")
    print(f"  {'-'*22} {'-'*12} {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    mi_total = summary["mi355x_total"]

    for cat_name, info in categories.items():
        module = info["module"]
        speedup = info["potential_speedup"]
        savings = info["savings"]
        cum_savings = info["cumulative_savings"]
        note = info.get("note", "")

        if speedup is not None and speedup > 1.0:
            spd_s = f"{speedup:.2f}x"
        elif speedup is not None:
            spd_s = f"{speedup:.2f}x"
        else:
            spd_s = "N/A"

        savings_s = fmt_dur(savings) if savings > 0 else "-"
        overall_pct = info["overall_speedup_pct"]
        overall_s = f"{overall_pct:.1f}%" if overall_pct is not None else "N/A"

        cum_pct = cum_savings / mi_total * 100 if mi_total > 0 else 0

        marker = " ***" if speedup is not None and speedup > 2.0 else ""
        if note:
            marker += f"  ({note})"

        print(f"    {cat_name:<22} {module:<12} {spd_s:>10} "
              f"{savings_s:>12} {overall_s:>12} "
              f"{fmt_dur(cum_savings):>12} {cum_pct:>9.1f}%{marker}")

    print(f"\n  {'─'*100}")
    print(f"  Summary:")
    print(f"    1 Layer MI355X:    {fmt_dur(summary['mi355x_1layer'])}")
    print(f"    1 Layer B200:     {fmt_dur(summary['b200_1layer'])}")
    print(f"    B200/MI355X:      {summary['b200_over_mi355x_1layer']:.0%}")
    print(f"    {summary['moe_layers']} layers MI355X:  {fmt_dur(summary['mi355x_total'])}")
    print(f"    {summary['moe_layers']} layers B200:    {fmt_dur(summary['b200_total'])}")
    print(f"    Max savings:      {fmt_dur(summary['cumulative_savings'])}")
    print(f"    Optimized MI355X: {fmt_dur(summary['optimized_mi355x_total'])}")
    print(f"    Gap ratio:        {summary['gap_ratio']:.2f}x "
          f"({'MI355X still slower' if summary['gap_ratio'] > 1 else 'MI355X faster'})")
    if summary["gap_vs_b200_e2e"] is not None:
        print(f"    vs B200 E2E:      {summary['gap_vs_b200_e2e']:.1f}%")
    print(f"{'='*110}\n")


def write_projection_csv(projections: dict, path: str) -> None:
    categories = projections["categories"]
    summary = projections["summary"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Category", "Module",
            "MI355X_1Layer_us", "B200_1Layer_us",
            "Kernel_Speedup", "MI355X_Weight_us",
            "Savings_us", "Overall_Speedup_Pct",
            "Cumulative_Savings_us", "Cumulative_Speedup_Pct",
            "Note",
        ])

        mi_total = summary["mi355x_total"]
        for cat_name, info in categories.items():
            if info["mi355x_time"] == 0 and info["b200_time"] == 0:
                continue
            cum_pct = info["cumulative_savings"] / mi_total * 100 if mi_total > 0 else 0
            writer.writerow([
                cat_name, info["module"],
                f"{info['mi355x_time']:.1f}", f"{info['b200_time']:.1f}",
                f"{info['potential_speedup']:.2f}" if info["potential_speedup"] else "N/A",
                f"{info['mi355x_weight']:.0f}",
                f"{info['savings']:.0f}",
                f"{info['overall_speedup_pct']:.1f}" if info["overall_speedup_pct"] is not None else "N/A",
                f"{info['cumulative_savings']:.0f}",
                f"{cum_pct:.1f}",
                info.get("note", ""),
            ])

        # Summary row
        writer.writerow([])
        writer.writerow(["Summary"])
        writer.writerow(["1 Layer MI355X (us)", f"{summary['mi355x_1layer']:.1f}"])
        writer.writerow(["1 Layer B200 (us)", f"{summary['b200_1layer']:.1f}"])
        writer.writerow(["B200/MI355X", f"{summary['b200_over_mi355x_1layer']:.2f}"])
        writer.writerow([f"{summary['moe_layers']} layers MI355X (us)", f"{summary['mi355x_total']:.0f}"])
        writer.writerow([f"{summary['moe_layers']} layers B200 (us)", f"{summary['b200_total']:.0f}"])
        writer.writerow(["Max Savings (us)", f"{summary['cumulative_savings']:.0f}"])
        writer.writerow(["Optimized MI355X (us)", f"{summary['optimized_mi355x_total']:.0f}"])
        writer.writerow(["Gap Ratio", f"{summary['gap_ratio']:.2f}"])

    print(f"[INFO] Projection CSV written to: {path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute optimization projections for MI355X vs B200.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Input from categorize_kernels.py JSON
    p.add_argument("--input", metavar="JSON",
                   help="Comparison JSON from categorize_kernels.py --json")

    # Or directly from traces
    p.add_argument("--mi355x-trace", metavar="TRACE",
                   help="MI355X trace (auto-detect + categorize)")
    p.add_argument("--b200-trace", metavar="TRACE",
                   help="B200 trace (auto-detect + categorize)")
    p.add_argument("--config", metavar="YAML",
                   help="kernel_categories.yaml (required with --*-trace)")
    p.add_argument("--layer-index", type=int, default=2,
                   help="Layer index for auto-detect (default=2)")

    # Model config
    p.add_argument("--moe-layers", type=int, default=89,
                   help="Number of MoE layers (default=89 for GLM-4.7)")
    p.add_argument("--ffn-layers", type=int, default=3,
                   help="Number of FFN layers (default=3)")
    p.add_argument("--allreduce-mi355x", type=float, default=0,
                   help="Total allreduce time for MI355X (us)")
    p.add_argument("--allreduce-b200", type=float, default=0,
                   help="Total allreduce time for B200 (us)")
    p.add_argument("--e2e-b200", type=float, default=0,
                   help="B200 end-to-end time (us) for gap calculation")

    # Output
    p.add_argument("--csv", metavar="FILE", help="Write projection CSV")
    p.add_argument("--json-out", metavar="FILE", help="Write projection JSON")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.input:
        data = load_comparison_json(args.input)
    elif args.mi355x_trace or args.b200_trace:
        if not args.config:
            sys.exit("[ERROR] --config is required when using --*-trace")
        from categorize_kernels import (
            load_config, extract_layer_from_trace,
            categorize_all, aggregate_by_category,
        )
        config = load_config(args.config)
        categories = config.get("categories", {})

        data = {"model": config.get("model", ""), "categories": {}}

        mi355x_agg = {}
        b200_agg = {}
        if args.mi355x_trace:
            mi_kernels = extract_layer_from_trace(args.mi355x_trace, args.layer_index)
            mi355x_cat = categorize_all(mi_kernels, categories)
            mi355x_agg = aggregate_by_category(mi355x_cat)
        if args.b200_trace:
            b2_kernels = extract_layer_from_trace(args.b200_trace, args.layer_index)
            b200_cat = categorize_all(b2_kernels, categories)
            b200_agg = aggregate_by_category(b200_cat)

        for cat_name in categories:
            mi = mi355x_agg.get(cat_name, {"total_dur": 0, "count": 0})
            b2 = b200_agg.get(cat_name, {"total_dur": 0, "count": 0})
            data["categories"][cat_name] = {
                "module": categories[cat_name].get("module", ""),
                "mi355x_time": mi["total_dur"] if isinstance(mi, dict) else 0,
                "mi355x_count": mi["count"] if isinstance(mi, dict) else 0,
                "b200_time": b2["total_dur"] if isinstance(b2, dict) else 0,
                "b200_count": b2["count"] if isinstance(b2, dict) else 0,
            }
    else:
        sys.exit("[ERROR] Provide --input JSON or --mi355x-trace/--b200-trace")

    projections = compute_projections(
        data,
        moe_layers=args.moe_layers,
        ffn_layers=args.ffn_layers,
        allreduce_mi355x=args.allreduce_mi355x,
        allreduce_b200=args.allreduce_b200,
        e2e_b200=args.e2e_b200,
    )

    print_projection_table(projections)

    if args.csv:
        write_projection_csv(projections, args.csv)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(projections, f, indent=2)
        print(f"[INFO] Projection JSON written to: {args.json_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
