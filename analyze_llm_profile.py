#!/usr/bin/env python3
"""
analyze_llm_profile.py

Full pipeline orchestrator for LLM profiling analysis.
Combines auto_detect_layer → categorize_kernels → optimization_projection.

Usage:
    # Full cross-platform analysis
    python analyze_llm_profile.py \
        --mi355x-trace mi355x_decode.trace.json.gz \
        --b200-trace b200_decode.trace.json.gz \
        --config kernel_categories.yaml \
        --moe-layers 89 \
        --output-dir ./analysis_results/

    # MI355X only
    python analyze_llm_profile.py \
        --mi355x-trace mi355x_decode.trace.json.gz \
        --config kernel_categories.yaml \
        --output-dir ./analysis_results/

    # Use pre-extracted CSVs
    python analyze_llm_profile.py \
        --mi355x-csv mi355x_layer.csv \
        --b200-csv b200_layer.csv \
        --config kernel_categories.yaml \
        --output-dir ./analysis_results/
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Import from our modules
from auto_detect_layer import (
    load_trace, extract_kernels, find_busiest_stream,
    short_name, detect_layer_pattern, find_layer_boundaries,
    write_csv as write_kernel_csv,
    print_table as print_kernel_table,
)
from categorize_kernels import (
    load_config, load_kernel_csv, categorize_all, aggregate_by_category,
    print_single_platform, print_comparison, write_comparison_csv,
)
from optimization_projection import (
    compute_projections, print_projection_table, write_projection_csv,
)


def extract_layer(trace_path: str, platform: str,
                  layer_index: int = 2) -> list[dict]:
    """Extract one layer's kernels from a trace file."""
    print(f"\n{'='*60}", file=sys.stderr)
    print(f" Stage 1: Auto-detect layer ({platform})", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"[INFO] Loading: {trace_path}", file=sys.stderr)

    trace = load_trace(trace_path)
    stream = find_busiest_stream(trace)
    print(f"[INFO] Busiest GPU stream: {stream}", file=sys.stderr)

    kernels = extract_kernels(trace, stream=stream)
    print(f"[INFO] Total kernels on stream: {len(kernels)}", file=sys.stderr)

    names = [short_name(k["name"]) for k in kernels]
    pattern, offset = detect_layer_pattern(names)
    print(f"[INFO] Layer pattern: {len(pattern)} kernels, offset={offset}",
          file=sys.stderr)

    layers = find_layer_boundaries(kernels, pattern, offset)
    print(f"[INFO] Found {len(layers)} layer instances", file=sys.stderr)

    if layer_index >= len(layers):
        sys.exit(f"[ERROR] Layer index {layer_index} out of range "
                 f"(found {len(layers)} layers)")

    start, end = layers[layer_index]
    layer_kernels = kernels[start:end + 1]
    total_dur = sum(k["dur"] for k in layer_kernels)
    print(f"[INFO] Extracted layer {layer_index}: {len(layer_kernels)} kernels, "
          f"total={total_dur:.1f} us", file=sys.stderr)

    return layer_kernels


def run_pipeline(args) -> None:
    config = load_config(args.config)
    categories = config.get("categories", {})

    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Stage 1: Extract layer kernels ───
    mi355x_kernels = None
    b200_kernels = None

    if args.mi355x_trace:
        mi355x_kernels = extract_layer(args.mi355x_trace, "MI355X", args.layer_index)
        csv_path = output_dir / "mi355x_layer_kernels.csv"
        write_kernel_csv(mi355x_kernels, str(csv_path))
    elif args.mi355x_csv:
        mi355x_kernels = load_kernel_csv(args.mi355x_csv)
        print(f"[INFO] Loaded MI355X kernels from CSV: {len(mi355x_kernels)}", file=sys.stderr)

    if args.b200_trace:
        b200_kernels = extract_layer(args.b200_trace, "B200", args.layer_index)
        csv_path = output_dir / "b200_layer_kernels.csv"
        write_kernel_csv(b200_kernels, str(csv_path))
    elif args.b200_csv:
        b200_kernels = load_kernel_csv(args.b200_csv)
        print(f"[INFO] Loaded B200 kernels from CSV: {len(b200_kernels)}", file=sys.stderr)

    if not mi355x_kernels and not b200_kernels:
        sys.exit("[ERROR] No input provided.")

    # ─── Stage 2: Categorize ───
    print(f"\n{'='*60}", file=sys.stderr)
    print(f" Stage 2: Categorize Kernels", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    mi355x_agg = {}
    b200_agg = {}

    if mi355x_kernels:
        mi355x_cat = categorize_all(mi355x_kernels, categories)
        mi355x_agg = aggregate_by_category(mi355x_cat)

    if b200_kernels:
        b200_cat = categorize_all(b200_kernels, categories)
        b200_agg = aggregate_by_category(b200_cat)

    if mi355x_kernels and b200_kernels:
        print_comparison(mi355x_agg, b200_agg, categories)
        write_comparison_csv(mi355x_agg, b200_agg, categories,
                             str(output_dir / "categorized_comparison.csv"))
    elif mi355x_kernels:
        print_single_platform(mi355x_agg, categories, "MI355X")
    elif b200_kernels:
        print_single_platform(b200_agg, categories, "B200")

    # Write JSON for projection
    comparison_data = {
        "model": config.get("model", ""),
        "categories": {},
    }
    for cat_name in categories:
        mi = mi355x_agg.get(cat_name, {"total_dur": 0, "count": 0})
        b2 = b200_agg.get(cat_name, {"total_dur": 0, "count": 0})
        comparison_data["categories"][cat_name] = {
            "module": categories[cat_name].get("module", ""),
            "mi355x_time": mi["total_dur"] if isinstance(mi, dict) else 0,
            "mi355x_count": mi["count"] if isinstance(mi, dict) else 0,
            "b200_time": b2["total_dur"] if isinstance(b2, dict) else 0,
            "b200_count": b2["count"] if isinstance(b2, dict) else 0,
        }

    json_path = output_dir / "categorized_comparison.json"
    with open(json_path, "w") as f:
        json.dump(comparison_data, f, indent=2)

    # ─── Stage 3: Optimization Projection ───
    if mi355x_kernels and b200_kernels:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f" Stage 3: Optimization Projection", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        projections = compute_projections(
            comparison_data,
            moe_layers=args.moe_layers,
            ffn_layers=args.ffn_layers,
            allreduce_mi355x=args.allreduce_mi355x,
            allreduce_b200=args.allreduce_b200,
            e2e_b200=args.e2e_b200,
        )

        print_projection_table(projections)
        write_projection_csv(projections, str(output_dir / "optimization_projection.csv"))

        with open(output_dir / "optimization_projection.json", "w") as f:
            json.dump(projections, f, indent=2)

    print(f"\n[INFO] All outputs saved to: {output_dir}/", file=sys.stderr)
    print(f"  Files:", file=sys.stderr)
    for p in sorted(output_dir.iterdir()):
        print(f"    {p.name}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Full pipeline: auto-detect layer → categorize → optimize.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Input
    g1 = p.add_argument_group("Input (from trace)")
    g1.add_argument("--mi355x-trace", metavar="TRACE",
                    help="MI355X trace JSON (.json.gz)")
    g1.add_argument("--b200-trace", metavar="TRACE",
                    help="B200 trace JSON (.json.gz)")

    g2 = p.add_argument_group("Input (from CSV)")
    g2.add_argument("--mi355x-csv", metavar="CSV",
                    help="Pre-extracted MI355X kernel CSV")
    g2.add_argument("--b200-csv", metavar="CSV",
                    help="Pre-extracted B200 kernel CSV")

    # Config
    p.add_argument("--config", required=True,
                   help="Path to kernel_categories.yaml")
    p.add_argument("--layer-index", type=int, default=2,
                   help="Layer index for auto-detect (default=2)")

    # Model parameters
    g3 = p.add_argument_group("Model parameters")
    g3.add_argument("--moe-layers", type=int, default=89,
                    help="Number of MoE decoder layers (default=89)")
    g3.add_argument("--ffn-layers", type=int, default=3,
                    help="Number of FFN layers (default=3)")
    g3.add_argument("--allreduce-mi355x", type=float, default=0,
                    help="Total MI355X allreduce overhead (us)")
    g3.add_argument("--allreduce-b200", type=float, default=0,
                    help="Total B200 allreduce overhead (us)")
    g3.add_argument("--e2e-b200", type=float, default=0,
                    help="B200 end-to-end time for gap calc (us)")

    # Output
    p.add_argument("--output-dir", default="./analysis_results",
                   help="Output directory (default: ./analysis_results)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
