#!/usr/bin/env python3
"""Analyze CUDA stream overlap and bubble time from a Torch Profiler trace.

Mental model:
  Wall time     = Bubble + Busy
  Busy          = time with >=1 kernel running (union of all kernel intervals)
  Overlap       = sum_of_all_kernel_durations - Busy  (time "saved" by parallelism)
                  When N kernels overlap for dt, this contributes (N-1)*dt
  Bubble        = Wall - Busy  (idle GPU)

Usage:
    python3 analyze_trace_overlap.py trace.json.gz
    python3 analyze_trace_overlap.py trace.json.gz --exclude "nccl|memcpy"
    python3 analyze_trace_overlap.py dual.trace.json.gz single.trace.json.gz
    python3 analyze_trace_overlap.py trace.json.gz --filter "moe|expert"
"""

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


# ──────────────────── Data ────────────────────

class KernelEvent:
    __slots__ = ("name", "stream", "start", "dur", "cat")

    def __init__(self, name, stream, start, dur, cat):
        self.name = name
        self.stream = stream
        self.start = start
        self.dur = dur
        self.cat = cat

    @property
    def end(self):
        return self.start + self.dur


# ──────────────────── I/O ────────────────────

def load_trace(path: str) -> dict:
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def extract_gpu_kernels(trace_data: dict) -> List[KernelEvent]:
    events = trace_data if isinstance(trace_data, list) else trace_data.get("traceEvents", [])
    kernels = []
    for ev in events:
        if ev.get("ph") != "X" or ev.get("dur", 0) <= 0:
            continue
        cat = ev.get("cat", "")
        if cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        kernels.append(KernelEvent(
            name=ev.get("name", ""),
            stream=f"pid{ev.get('pid', '?')}/tid{ev.get('tid', '?')}",
            start=ev["ts"],
            dur=ev["dur"],
            cat=cat,
        ))
    return kernels


# ──────────────────── Core math ────────────────────

def compute_stats(kernels: List[KernelEvent]):
    """
    Sweep-line over all kernel start/end points.

    Returns:
        wall:       wall time (first kernel start → last kernel end)
        busy:       time with ≥1 kernel running  (= union of all kernel intervals)
        bubble:     time with 0 kernels running   (= wall − busy)
        overlap:    parallelism savings            (= sum_all − busy)
        sum_all:    sum of every kernel's duration (counts concurrency multiple times)
        conc_hist:  {concurrency_level: duration_us}
    """
    if not kernels:
        return 0, 0, 0, 0, 0, {}

    sum_all = sum(k.dur for k in kernels)

    # sweep line: +1 at each kernel start, −1 at each kernel end
    pts = []
    for k in kernels:
        pts.append((k.start, +1))
        pts.append((k.end, -1))
    pts.sort()

    wall_start = pts[0][0]
    wall_end = max(k.end for k in kernels)
    wall = wall_end - wall_start

    conc = 0
    prev_t = pts[0][0]
    hist = defaultdict(float)
    busy = 0.0

    for t, d in pts:
        dt = t - prev_t
        if dt > 0:
            hist[conc] += dt
            if conc >= 1:
                busy += dt
        conc += d
        prev_t = t

    bubble = wall - busy
    overlap = sum_all - busy  # weighted: N concurrent kernels for dt → (N-1)*dt saved

    return wall, busy, bubble, overlap, sum_all, dict(hist)


def merge_intervals(intervals):
    if not intervals:
        return []
    s = sorted(intervals)
    m = [s[0]]
    for a, b in s[1:]:
        if a <= m[-1][1]:
            m[-1] = (m[-1][0], max(m[-1][1], b))
        else:
            m.append((a, b))
    return m


def compute_stream_concurrency(kernels: List[KernelEvent]):
    """Like compute_stats but counts distinct STREAMS active, not individual kernels."""
    if not kernels:
        return {}

    streams = defaultdict(list)
    for k in kernels:
        streams[k.stream].append((k.start, k.end))

    pts = []
    for s, ivs in streams.items():
        for a, b in merge_intervals(ivs):
            pts.append((a, +1))
            pts.append((b, -1))
    pts.sort()

    wall = max(k.end for k in kernels) - min(k.start for k in kernels)
    conc = 0
    prev_t = pts[0][0]
    hist = defaultdict(float)
    for t, d in pts:
        dt = t - prev_t
        if dt > 0:
            hist[conc] += dt
        conc += d
        prev_t = t
    return dict(hist)


def pairwise_overlap(kernels: List[KernelEvent], min_us: float = 100) -> Dict[Tuple[str, str], float]:
    streams = defaultdict(list)
    for k in kernels:
        streams[k.stream].append((k.start, k.end))
    merged = {s: merge_intervals(ivs) for s, ivs in streams.items()}
    names = sorted(merged)
    result = {}
    for i, s1 in enumerate(names):
        for s2 in names[i + 1:]:
            iv1, iv2 = merged[s1], merged[s2]
            ov = 0.0
            j, k = 0, 0
            while j < len(iv1) and k < len(iv2):
                lo = max(iv1[j][0], iv2[k][0])
                hi = min(iv1[j][1], iv2[k][1])
                if lo < hi:
                    ov += hi - lo
                if iv1[j][1] < iv2[k][1]:
                    j += 1
                else:
                    k += 1
            if ov >= min_us:
                result[(s1, s2)] = ov
    return result


# ──────────────────── Formatting ────────────────────

def fmt(us: float) -> str:
    return f"{us / 1_000:.3f} ms"


def pct(part, total):
    return f"{100 * part / total:.1f}%" if total else "—"


def bar(part, total, width=40):
    if not total:
        return ""
    n = max(0, int(width * part / total))
    return "█" * n


def hdr(title):
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


# ──────────────────── Analysis for one trace ────────────────────

def analyze_one(kernels: List[KernelEvent], label: str, top_n: int,
                min_stream_us: float, show_pairs: bool, show_streams: bool = False):
    n_streams = len(set(k.stream for k in kernels))
    wall, busy, bubble, overlap, sum_all, conc_hist = compute_stats(kernels)
    stream_hist = compute_stream_concurrency(kernels)

    # ── Main summary (the user's colored-box model) ──
    hdr(f"{label}  ({len(kernels)} kernels, {n_streams} streams)")

    W = 40
    def row(label, val_s, pct_s=""):
        return f"  │  {label:<{W}s}{val_s:>12s}  {pct_s:>6s}"
    def sep():
        return f"  │  {'':<{W}s}{'─'*12}"

    print()
    print(f"  ┌─ Wall time {'─'*44}")
    print(f"  │")
    print(row("Sum of all kernel durations:", fmt(sum_all)))
    print(row("Overlap saved (parallelism):", f"- {fmt(overlap)}", pct(overlap, wall)))
    print(sep())
    print(row("Busy time (>=1 active):", fmt(busy), pct(busy, wall)))
    print(row("Bubble time (0 active):", fmt(bubble), pct(bubble, wall)))
    print(sep())
    print(row("Wall time:", fmt(wall), "100.0%"))
    print()

    if wall > 0:
        print(f"\n  Speedup from overlap:  {sum_all / wall:.2f}x  (Sum / Wall)")

    # ── Kernel concurrency histogram ──
    hdr(f"{label}: KERNEL CONCURRENCY (how many kernels at the same instant)")
    for lv in sorted(conc_hist):
        d = conc_hist[lv]
        lbl = "bubble" if lv == 0 else f"{lv} kernel(s)"
        ov_note = ""
        if lv >= 2:
            saved = (lv - 1) * d
            ov_note = f"  overlap saved {fmt(saved)}"
        print(f"  {lbl:>14s}: {fmt(d):>12s}  {pct(d, wall):>6s}  {bar(d, wall)}{ov_note}")

    # ── Stream concurrency histogram ──
    hdr(f"{label}: STREAM CONCURRENCY (how many distinct streams active)")
    for lv in sorted(stream_hist):
        d = stream_hist[lv]
        lbl = "bubble" if lv == 0 else f"{lv} stream(s)"
        print(f"  {lbl:>14s}: {fmt(d):>12s}  {pct(d, wall):>6s}  {bar(d, wall)}")

    # ── Per-stream breakdown (opt-in via --show-streams) ──
    if show_streams:
        by_stream = defaultdict(lambda: {"n": 0, "dur": 0.0, "by_name": defaultdict(float)})
        for k in kernels:
            s = by_stream[k.stream]
            s["n"] += 1
            s["dur"] += k.dur
            s["by_name"][k.name] += k.dur

        sorted_streams = sorted(by_stream.items(), key=lambda x: -x[1]["dur"])
        shown = [(sn, si) for sn, si in sorted_streams if si["dur"] >= min_stream_us]

        if shown:
            hdr(f"{label}: TOP STREAMS (≥ {fmt(min_stream_us)})")
            for sname, info in shown:
                print(f"\n  {sname}  ({info['n']} kernels, {fmt(info['dur'])}  {pct(info['dur'], sum_all)} of total)")
                top = sorted(info["by_name"].items(), key=lambda x: -x[1])[:top_n]
                for kn, kd in top:
                    print(f"    {fmt(kd):>12s}  {pct(kd, info['dur']):>6s}  {kn[:90]}")

    # ── Pairwise overlap ──
    if show_pairs:
        pairs = pairwise_overlap(kernels, min_us=min_stream_us)
        sig = {k: v for k, v in pairs.items() if v >= min_stream_us}
        if sig:
            hdr(f"{label}: PAIRWISE STREAM OVERLAP (≥ {fmt(min_stream_us)})")
            for (s1, s2), ov in sorted(sig.items(), key=lambda x: -x[1])[:30]:
                print(f"  {fmt(ov):>12s}  {pct(ov, wall):>6s}  {s1}  ↔  {s2}")

    return wall, busy, bubble, overlap, sum_all, conc_hist, stream_hist


# ──────────────────── Main ────────────────────

def main():
    p = argparse.ArgumentParser(description="Trace overlap analyzer",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("traces", nargs="*", help="trace.json / trace.json.gz (positional)")
    p.add_argument("--trace1", type=str, default=None, help="First trace file (named arg)")
    p.add_argument("--trace2", type=str, default=None, help="Second trace file (named arg, for comparison)")
    p.add_argument("--filter", type=str, default=None, help="Regex: keep only matching kernels")
    p.add_argument("--exclude", type=str, default=None, help="Regex: remove matching kernels")
    p.add_argument("--top", type=int, default=8, help="Top N kernels per stream")
    p.add_argument("--min-stream-ms", type=float, default=0.5, help="Min stream time to display (ms)")
    p.add_argument("--no-pairs", action="store_true", help="Skip pairwise overlap table")
    p.add_argument("--show-streams", action="store_true", help="Show per-stream breakdown (off by default)")
    p.add_argument("--csv", type=str, default=None, help="Export CSV")
    p.add_argument("--html", type=str, default=None, help="Export HTML report")
    args = p.parse_args()

    min_us = args.min_stream_ms * 1000

    trace_files = list(args.traces)
    if args.trace1:
        trace_files.insert(0, args.trace1)
    if args.trace2:
        trace_files.append(args.trace2)
    if not trace_files:
        p.error("No trace files provided. Use positional args or --trace1/--trace2.")

    results = []

    for i, path in enumerate(trace_files):
        label = Path(path).stem
        if len(trace_files) > 1:
            label = f"[{i+1}] {label}"

        print(f"\nLoading: {path}")
        data = load_trace(path)
        kernels = extract_gpu_kernels(data)
        print(f"  Raw GPU kernel events: {len(kernels)}")

        if args.filter:
            pre = len(kernels)
            kernels = [k for k in kernels if re.search(args.filter, k.name, re.I)]
            print(f"  After --filter '{args.filter}': {len(kernels)}/{pre}")
        if args.exclude:
            pre = len(kernels)
            kernels = [k for k in kernels if not re.search(args.exclude, k.name, re.I)]
            print(f"  After --exclude '{args.exclude}': {len(kernels)}/{pre}")

        if not kernels:
            print("  No kernels matched.")
            continue

        r = analyze_one(kernels, label, args.top, min_us, not args.no_pairs, args.show_streams)
        results.append((label, *r))

    # ── Side-by-side comparison ──
    if len(results) == 2:
        hdr("COMPARISON")
        a, b = results[0], results[1]
        # (label, wall, busy, bubble, overlap, sum_all, conc_hist, stream_hist)
        metrics = [
            ("Wall time",    1),
            ("Busy",         2),
            ("Bubble",       3),
            ("Overlap",      4),
            ("Sum all kern", 5),
        ]
        print(f"  {'Metric':<18s}  {a[0]:>14s}  {b[0]:>14s}  {'Delta':>14s}")
        print(f"  {'─'*18}  {'─'*14}  {'─'*14}  {'─'*14}")
        for name, idx in metrics:
            v0, v1 = a[idx], b[idx]
            d = v1 - v0
            sign = "+" if d >= 0 else ""
            print(f"  {name:<18s}  {fmt(v0):>14s}  {fmt(v1):>14s}  {sign}{fmt(d):>13s}")
        print()

    # ── HTML ──
    if args.html and results:
        generate_html(results, args.html)

    # ── CSV ──
    if args.csv and results:
        with open(args.csv, "w") as f:
            f.write("trace,wall_us,busy_us,bubble_us,overlap_us,sum_all_us\n")
            for label, wall, busy, bubble, overlap, sum_all, *_ in results:
                f.write(f"{label},{wall:.1f},{busy:.1f},{bubble:.1f},{overlap:.1f},{sum_all:.1f}\n")
        print(f"  CSV → {args.csv}")


def _html_single(r, html_mod):
    """Generate HTML card for a single trace."""
    label, wall, busy, bubble, overlap, sum_all, conc_hist, stream_hist = r
    speedup = sum_all / wall if wall > 0 else 0
    overlap_pct = 100 * overlap / wall if wall else 0
    busy_pct = 100 * busy / wall if wall else 0
    bubble_pct = 100 * bubble / wall if wall else 0
    safe_label = html_mod.escape(label)

    def hist_rows(hist, unit):
        rows = ""
        for lv in sorted(hist):
            d = hist[lv]
            w = 100 * d / wall if wall else 0
            lbl = "bubble" if lv == 0 else f"{lv} {unit}"
            cls = "bg-bubble" if lv == 0 else ("bg-busy" if lv == 1 else "bg-t1")
            note = ""
            if lv >= 2:
                saved = (lv - 1) * d
                note = f'<span class="hist-note">overlap saved {fmt(saved)}</span>'
            rows += f"""
          <div class="hist-row">
            <span class="hist-label">{lbl}</span>
            <div class="hist-bar-wrap">
              <div class="hist-bar {cls}" style="width:{w:.2f}%"></div>
            </div>
            <span class="hist-val">{fmt(d)}</span>
            <span class="hist-pct">{100*d/wall:.1f}%</span>
            {note}
          </div>"""
        return rows

    return f"""
    <div class="card">
      <h2>{safe_label}</h2>
      <div class="metric-grid">
        <div class="metric"><span class="metric-val">{fmt(wall)}</span><span class="metric-lbl">Wall time</span></div>
        <div class="metric"><span class="metric-val">{fmt(busy)}</span><span class="metric-lbl">Busy ({busy_pct:.1f}%)</span></div>
        <div class="metric"><span class="metric-val">{fmt(bubble)}</span><span class="metric-lbl">Bubble ({bubble_pct:.1f}%)</span></div>
        <div class="metric"><span class="metric-val">{speedup:.2f}x</span><span class="metric-lbl">Speedup (Sum/Wall)</span></div>
      </div>
      <h3>Wall time breakdown</h3>
      <div class="waterfall">
        <div class="wf-row"><span class="wf-label">Sum of all kernels</span><div class="wf-bar-wrap"><div class="wf-bar bg-sum" style="width:100%">{fmt(sum_all)}</div></div></div>
        <div class="wf-row"><span class="wf-label">Overlap saved</span><div class="wf-bar-wrap"><div class="wf-bar bg-overlap" style="width:{100*overlap/sum_all:.1f}%">-{fmt(overlap)}</div></div></div>
        <div class="wf-row"><span class="wf-label">Busy time</span><div class="wf-bar-wrap"><div class="wf-bar bg-busy" style="width:{100*busy/sum_all:.1f}%">{fmt(busy)}</div></div></div>
        <div class="wf-row"><span class="wf-label">Bubble time</span><div class="wf-bar-wrap"><div class="wf-bar bg-bubble" style="width:{100*bubble/sum_all:.1f}%">{fmt(bubble)}</div></div></div>
        <div class="wf-row"><span class="wf-label">Wall time</span><div class="wf-bar-wrap"><div class="wf-bar bg-wall" style="width:{100*wall/sum_all:.1f}%">{fmt(wall)}</div></div></div>
      </div>
      <h3>GPU utilization</h3>
      <div class="stacked-bar">
        <div class="seg bg-busy" style="width:{busy_pct:.2f}%" title="Busy {busy_pct:.1f}%"></div>
        <div class="seg bg-bubble" style="width:{bubble_pct:.2f}%" title="Bubble {bubble_pct:.1f}%"></div>
      </div>
      <div class="stacked-legend">
        <span><i class="dot bg-busy"></i>Busy {busy_pct:.1f}%</span>
        <span><i class="dot bg-overlap"></i>Overlap saved {overlap_pct:.1f}%</span>
        <span><i class="dot bg-bubble"></i>Bubble {bubble_pct:.1f}%</span>
      </div>
      <h3>Kernel concurrency</h3>
      <div class="hist">{hist_rows(conc_hist, "kernel(s)")}</div>
      <h3>Stream concurrency</h3>
      <div class="hist">{hist_rows(stream_hist, "stream(s)")}</div>
    </div>"""


def _html_comparison(results, html_mod):
    """Generate a single unified comparison card for 2 traces."""
    a_label, a_wall, a_busy, a_bubble, a_overlap, a_sum, a_conc, a_stream = results[0]
    b_label, b_wall, b_busy, b_bubble, b_overlap, b_sum, b_conc, b_stream = results[1]
    a_spd = a_sum / a_wall if a_wall else 0
    b_spd = b_sum / b_wall if b_wall else 0
    al = html_mod.escape(a_label)
    bl = html_mod.escape(b_label)

    # paired bar: two bars per metric, same scale (max of both)
    metrics = [
        ("Sum of all kernels", a_sum, b_sum, "bg-sum"),
        ("Overlap saved",      a_overlap, b_overlap, "bg-overlap"),
        ("Busy time",          a_busy, b_busy, "bg-busy"),
        ("Bubble time",        a_bubble, b_bubble, "bg-bubble"),
        ("Wall time",          a_wall, b_wall, "bg-wall"),
    ]
    global_max = max(a_sum, b_sum)
    bar_rows = ""
    for name, va, vb, cls in metrics:
        wa = 100 * va / global_max if global_max else 0
        wb = 100 * vb / global_max if global_max else 0
        d = vb - va
        sign = "+" if d >= 0 else ""
        dcls = "delta-pos" if d >= 0 else "delta-neg"
        bar_rows += f"""
      <div class="pair-group">
        <span class="pair-metric">{name}</span>
        <div class="pair-bars">
          <div class="pair-row"><span class="pair-tag t1">T1</span><div class="pair-bar-wrap"><div class="pair-bar bg-t1" style="width:{wa:.2f}%"></div></div><span class="pair-val">{fmt(va)}</span></div>
          <div class="pair-row"><span class="pair-tag t2">T2</span><div class="pair-bar-wrap"><div class="pair-bar bg-t2" style="width:{wb:.2f}%"></div></div><span class="pair-val">{fmt(vb)}</span></div>
        </div>
        <span class="pair-delta {dcls}">{sign}{fmt(d)}</span>
      </div>"""

    # delta table
    tbl_metrics = [("Wall time", a_wall, b_wall), ("Busy", a_busy, b_busy),
                   ("Bubble", a_bubble, b_bubble), ("Overlap", a_overlap, b_overlap),
                   ("Sum all kernels", a_sum, b_sum), ("Speedup (Sum/Wall)", a_spd, b_spd)]
    tbl_rows = ""
    for name, va, vb in tbl_metrics:
        if name.startswith("Speedup"):
            va_s, vb_s = f"{va:.2f}x", f"{vb:.2f}x"
            d = vb - va
            sign = "+" if d >= 0 else ""
            dcls = "delta-neg" if d >= 0 else "delta-pos"
            ds = f"{sign}{d:.2f}x"
        else:
            va_s, vb_s = fmt(va), fmt(vb)
            d = vb - va
            sign = "+" if d >= 0 else ""
            dcls = "delta-pos" if d >= 0 else "delta-neg"
            ds = f"{sign}{fmt(d)}"
        tbl_rows += f'<tr><td>{name}</td><td class="num">{va_s}</td><td class="num">{vb_s}</td><td class="num {dcls}">{ds}</td></tr>'

    # concurrency comparison (paired hist)
    def paired_hist(hist_a, hist_b, unit, ref_wall_a, ref_wall_b):
        all_levels = sorted(set(list(hist_a.keys()) + list(hist_b.keys())))
        rows = ""
        for lv in all_levels:
            da = hist_a.get(lv, 0)
            db = hist_b.get(lv, 0)
            pct_a = 100 * da / ref_wall_a if ref_wall_a else 0
            pct_b = 100 * db / ref_wall_b if ref_wall_b else 0
            lbl = "bubble" if lv == 0 else f"{lv} {unit}"
            rows += f"""
          <div class="pair-group narrow">
            <span class="pair-metric">{lbl}</span>
            <div class="pair-bars">
              <div class="pair-row"><span class="pair-tag t1">T1</span><div class="pair-bar-wrap"><div class="pair-bar bg-t1" style="width:{pct_a:.2f}%"></div></div><span class="pair-val">{pct_a:.1f}% ({fmt(da)})</span></div>
              <div class="pair-row"><span class="pair-tag t2">T2</span><div class="pair-bar-wrap"><div class="pair-bar bg-t2" style="width:{pct_b:.2f}%"></div></div><span class="pair-val">{pct_b:.1f}% ({fmt(db)})</span></div>
            </div>
          </div>"""
        return rows

    return f"""
    <div class="card">
      <h2>Comparison</h2>
      <div class="trace-legend">
        <span><i class="dot bg-t1"></i><b>T1</b> {al}</span>
        <span><i class="dot bg-t2"></i><b>T2</b> {bl}</span>
      </div>

      <div class="metric-grid metric-grid-cmp">
        <div class="metric"><span class="metric-val">{fmt(a_wall)}</span><span class="metric-lbl">T1 Wall</span></div>
        <div class="metric"><span class="metric-val">{fmt(b_wall)}</span><span class="metric-lbl">T2 Wall</span></div>
        <div class="metric"><span class="metric-val">{a_spd:.2f}x</span><span class="metric-lbl">T1 Speedup</span></div>
        <div class="metric"><span class="metric-val">{b_spd:.2f}x</span><span class="metric-lbl">T2 Speedup</span></div>
      </div>

      <h3>Wall time breakdown (same scale)</h3>
      <div class="pair-chart">{bar_rows}</div>

      <h3>Delta table</h3>
      <table class="cmp">
        <thead><tr><th>Metric</th><th class="num">T1</th><th class="num">T2</th><th class="num">Delta</th></tr></thead>
        <tbody>{tbl_rows}</tbody>
      </table>

      <h3>Kernel concurrency</h3>
      <div class="pair-chart">{paired_hist(a_conc, b_conc, "kernel(s)", a_wall, b_wall)}</div>

      <h3>Stream concurrency</h3>
      <div class="pair-chart">{paired_hist(a_stream, b_stream, "stream(s)", a_wall, b_wall)}</div>
    </div>"""


def generate_html(results, out_path):
    """Generate a self-contained HTML report with charts."""
    import html as html_mod

    if len(results) == 2:
        body = _html_comparison(results, html_mod)
    else:
        body = "".join(_html_single(r, html_mod) for r in results)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trace Overlap Report</title>
<style>
  :root {{
    --bg: #0f1117; --card: #1a1d27; --border: #2a2d3a;
    --text: #e4e6ed; --muted: #8b8fa3; --accent: #6c9eff;
    --busy: #e85d5d; --overlap: #4da6ff; --bubble: #a06cd5;
    --sum: #e8a45d; --wall: #5de8a0;
    --t1: #6c9eff; --t2: #ff9f6c;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
    background: var(--bg); color: var(--text);
    padding: 2rem; line-height: 1.6;
  }}
  h1 {{ font-size: 1.4rem; color: var(--accent); margin-bottom: 1.5rem; font-weight: 600; }}
  .card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
  }}
  .card h2 {{
    font-size: 1rem; color: var(--accent); margin-bottom: 1rem;
    padding-bottom: 0.5rem; border-bottom: 1px solid var(--border);
  }}
  .card h3 {{ font-size: 0.85rem; color: var(--muted); margin: 1.2rem 0 0.6rem; text-transform: uppercase; letter-spacing: 0.05em; }}

  .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1rem; }}
  .metric-grid-cmp {{ grid-template-columns: repeat(4, 1fr); }}
  .metric {{ text-align: center; }}
  .metric-val {{ display: block; font-size: 1.5rem; font-weight: 700; color: var(--text); }}
  .metric-lbl {{ display: block; font-size: 0.75rem; color: var(--muted); margin-top: 0.15rem; }}

  .trace-legend {{ display: flex; gap: 2rem; margin-bottom: 1rem; font-size: 0.85rem; }}
  .trace-legend b {{ margin-right: 0.3rem; }}

  /* single-trace waterfall */
  .waterfall {{ display: flex; flex-direction: column; gap: 6px; }}
  .wf-row {{ display: flex; align-items: center; gap: 12px; }}
  .wf-label {{ width: 160px; text-align: right; font-size: 0.8rem; color: var(--muted); flex-shrink: 0; }}
  .wf-bar-wrap {{ flex: 1; height: 28px; background: var(--bg); border-radius: 4px; overflow: hidden; }}
  .wf-bar {{
    height: 100%; border-radius: 4px; display: flex; align-items: center;
    padding: 0 10px; font-size: 0.75rem; font-weight: 600; color: #fff;
    min-width: fit-content; white-space: nowrap;
  }}

  .bg-sum {{ background: var(--sum); }}
  .bg-overlap {{ background: var(--overlap); }}
  .bg-busy {{ background: var(--busy); }}
  .bg-bubble {{ background: var(--bubble); }}
  .bg-wall {{ background: var(--wall); color: #111; }}
  .bg-t1 {{ background: var(--t1); }}
  .bg-t2 {{ background: var(--t2); }}

  .stacked-bar {{
    display: flex; height: 32px; border-radius: 6px; overflow: hidden;
    border: 1px solid var(--border); margin-top: 0.3rem;
  }}
  .seg {{ transition: width 0.3s; }}
  .stacked-legend {{ display: flex; gap: 1.2rem; margin-top: 0.5rem; font-size: 0.78rem; color: var(--muted); }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: middle; }}

  /* paired comparison bars */
  .pair-chart {{ display: flex; flex-direction: column; gap: 12px; }}
  .pair-group {{ display: grid; grid-template-columns: 160px 1fr 100px; align-items: center; gap: 8px; }}
  .pair-group.narrow {{ grid-template-columns: 100px 1fr; }}
  .pair-metric {{ text-align: right; font-size: 0.8rem; color: var(--muted); }}
  .pair-bars {{ display: flex; flex-direction: column; gap: 3px; }}
  .pair-row {{ display: flex; align-items: center; gap: 6px; }}
  .pair-tag {{ font-size: 0.65rem; font-weight: 700; width: 22px; text-align: center; border-radius: 3px; padding: 1px 0; }}
  .pair-tag.t1 {{ background: var(--t1); color: #111; }}
  .pair-tag.t2 {{ background: var(--t2); color: #111; }}
  .pair-bar-wrap {{ flex: 1; height: 18px; background: var(--bg); border-radius: 3px; overflow: hidden; }}
  .pair-bar {{ height: 100%; border-radius: 3px; }}
  .pair-val {{ font-size: 0.75rem; white-space: nowrap; min-width: 80px; }}
  .pair-delta {{ font-size: 0.78rem; font-weight: 600; text-align: right; }}

  .cmp {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  .cmp th, .cmp td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
  .cmp th {{ color: var(--muted); font-weight: 500; }}
  .cmp .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .delta-neg {{ color: var(--wall); }}
  .delta-pos {{ color: var(--busy); }}

  .hist {{ display: flex; flex-direction: column; gap: 4px; }}
  .hist-row {{ display: flex; align-items: center; gap: 8px; }}
  .hist-label {{ width: 100px; text-align: right; font-size: 0.78rem; color: var(--muted); flex-shrink: 0; }}
  .hist-bar-wrap {{ flex: 1; height: 22px; background: var(--bg); border-radius: 3px; overflow: hidden; }}
  .hist-bar {{ height: 100%; border-radius: 3px; }}
  .hist-val {{ width: 90px; text-align: right; font-size: 0.78rem; font-variant-numeric: tabular-nums; flex-shrink: 0; }}
  .hist-pct {{ width: 50px; text-align: right; font-size: 0.78rem; color: var(--muted); flex-shrink: 0; }}
  .hist-note {{ font-size: 0.72rem; color: var(--muted); font-style: italic; }}
</style>
</head>
<body>
  <h1>Trace Overlap Report</h1>
  {body}
  <div style="text-align:center; color:var(--muted); font-size:0.7rem; margin-top:2rem;">
    Generated by analyze_trace_overlap.py
  </div>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\n  HTML report -> {out_path}")


if __name__ == "__main__":
    main()
