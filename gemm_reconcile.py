#!/usr/bin/env python3
"""
Cross-backend GEMM tuning reconciler + ledger (production-gated).

Two modes:

  prepare   : split (master_input - ledger) into N per-GPU shard CSVs, so only
              NOT-yet-tuned shapes get tuned. Lets you add new GEMM sizes later
              without re-tuning everything.

  reconcile : after per-backend tuning runs, read every backend's --compare
              report (PRODUCTION gemm_a16w16 pre/post us, NOT sweep micro-us),
              pick the best backend per shape (must beat default by
              --min-improvement-pct), emit the winning rows into the final tuned
              CSV, and record EVERY tuned shape in the ledger -- including those
              where no backend beat default (so they are skipped next time).

Why production us: yesterday's mistake was selecting on sweep micro-benchmark
(call_hipb_mm) which looked great but regressed SGLang E2E. The --compare
report measures the real gemm_a16w16() op pre (default) vs post (tuned), which
is the right proxy; only shapes that improve that by >= threshold are kept.
"""
import argparse, csv, glob, os, re, sys, time
from collections import defaultdict
from statistics import median

KEY_COLS = ["M", "N", "K", "bias", "dtype", "outdtype", "scaleAB", "bpreshuffle"]
# compare-report line: "(M, N, K, <dtype>, bias=<...>) | pre | post | <ratio/pct> | <action>"
LINE_RE = re.compile(
    r"^\((\d+),\s*(\d+),\s*(\d+),[^)]*\)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|"
)


def read_input(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def shape_key(row):
    return (int(row["M"]), int(row["N"]), int(row["K"]))


def load_ledger_keys(ledger):
    keys = set()
    if os.path.exists(ledger):
        with open(ledger) as f:
            for row in csv.DictReader(f):
                try:
                    keys.add((int(row["M"]), int(row["N"]), int(row["K"])))
                except (KeyError, ValueError):
                    continue
    return keys


def cmd_prepare(args):
    master = read_input(args.input)
    done = load_ledger_keys(args.ledger)
    to_tune = [r for r in master if shape_key(r) not in done]
    os.makedirs(args.outdir, exist_ok=True)
    # round-robin split into nshards
    shards = [[] for _ in range(args.nshards)]
    for i, row in enumerate(to_tune):
        shards[i % args.nshards].append(row)
    header = master[0].keys() if master else KEY_COLS
    for g in range(args.nshards):
        p = os.path.join(args.outdir, f"shard_{g}.csv")
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(header))
            w.writeheader()
            for row in shards[g]:
                w.writerow(row)
    with open(os.path.join(args.outdir, "count"), "w") as f:
        f.write(str(len(to_tune)))
    print(
        f"[prepare] master={len(master)} already_tuned={len(done)} "
        f"to_tune={len(to_tune)} -> {args.nshards} shards in {args.outdir}"
    )


def parse_reports(report_glob):
    """shape (M,N,K) -> (median_pre_us, median_post_us) from compare reports."""
    pre_map, post_map = defaultdict(list), defaultdict(list)
    for fp in glob.glob(report_glob):
        try:
            txt = open(fp).read()
        except OSError:
            continue
        for line in txt.splitlines():
            m = LINE_RE.match(line.strip())
            if not m:
                continue
            k = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            pre, post = float(m.group(4)), float(m.group(5))
            if pre > 0:
                pre_map[k].append(pre)
            if post > 0:
                post_map[k].append(post)
    out = {}
    for k in set(pre_map) | set(post_map):
        pre = median(pre_map[k]) if pre_map.get(k) else None
        post = median(post_map[k]) if post_map.get(k) else None
        out[k] = (pre, post)
    return out


def load_backend_rows(shard_glob):
    """shape (M,N,K) -> full CSV row dict (winner row written by --update_improved)."""
    rows = {}
    header = None
    for fp in glob.glob(shard_glob):
        with open(fp) as f:
            r = csv.DictReader(f)
            header = r.fieldnames or header
            for row in r:
                try:
                    k = (int(row["M"]), int(row["N"]), int(row["K"]))
                except (KeyError, ValueError):
                    continue
                rows[k] = row
    return rows, header


def cmd_filter(args):
    """Write per-GPU shards of (to_tune - already-processed) for one backend.

    'already-processed' = shapes that appear in this backend's compare reports
    (every processed shape gets a pre/post line, INCLUDING timed-out shapes) or
    in its written tuned shards. This makes a backend resume at per-shape
    granularity: a re-run only tunes what is left, so an interrupted multi-hour
    flydsl run does not redo the shapes (incl. timeouts) it already covered.
    """
    to_tune = []
    for g in sorted(glob.glob(os.path.join(args.to_tune_dir, "shard_*.csv"))):
        to_tune.extend(read_input(g))
    processed = set()
    # PRIMARY incremental signal: candidate CSVs are appended per-batch as the
    # tuner processes each shape (compare.txt per-shape lines are only written at
    # the very end, so they are useless for mid-run resume).
    if args.candidates:
        crows, _ = load_backend_rows(args.candidates)
        processed |= set(crows.keys())
    # also honor any final compare report + written tuned shards if present
    if args.reports:
        processed |= set(parse_reports(args.reports).keys())
    if args.tuned:
        trows, _ = load_backend_rows(args.tuned)
        processed |= set(trows.keys())
    remaining = [r for r in to_tune if shape_key(r) not in processed]
    os.makedirs(args.outdir, exist_ok=True)
    shards = [[] for _ in range(args.nshards)]
    for i, row in enumerate(remaining):
        shards[i % args.nshards].append(row)
    header = to_tune[0].keys() if to_tune else KEY_COLS
    for g in range(args.nshards):
        p = os.path.join(args.outdir, f"shard_{g}.csv")
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(header))
            w.writeheader()
            for row in shards[g]:
                w.writerow(row)
    with open(os.path.join(args.outdir, "count"), "w") as f:
        f.write(str(len(remaining)))
    print(f"[filter] to_tune={len(to_tune)} processed={len(processed)} "
          f"remaining={len(remaining)}")


def cmd_reconcile(args):
    backends = [b for b in args.backends.split(",") if b and b != "all"]
    master = read_input(args.input)
    master_by_key = {shape_key(r): r for r in master}
    thr = args.min_improvement_pct / 100.0

    # gather per-backend production post_us + winning rows
    backend_post = {}   # backend -> {key: post_us}
    backend_rows = {}   # backend -> {key: row}
    default_us = defaultdict(list)
    csv_header = None
    for b in backends:
        rep = os.path.join(args.workdir, b, "compare_tmp", "aiter_compare", "*.compare.txt")
        rep_map = parse_reports(rep)
        backend_post[b] = {k: post for k, (pre, post) in rep_map.items() if post}
        for k, (pre, post) in rep_map.items():
            if pre:
                default_us[k].append(pre)
        rows, hdr = load_backend_rows(os.path.join(args.workdir, b, "shard_*.csv"))
        backend_rows[b] = rows
        if hdr:
            csv_header = hdr

    # torch native rows (libtype=torch, solidx=0, "native") for EVERY processed shape;
    # used to emit an explicit native row when no backend beats default.
    native_rows, nhdr = load_backend_rows(
        os.path.join(args.workdir, "torch", "compare_tmp", "aiter_compare", "*.candidate.csv")
    )
    if nhdr and not csv_header:
        csv_header = nhdr

    # per-shape cross-backend selection
    winners = {}     # key -> (backend, post_us, row)  (improved-over-default only)
    final_pick = {}  # key -> row to write to final CSV (winner OR explicit native)
    ledger_new = []  # rows for ledger
    tuned_keys = set()
    for b in backends:
        tuned_keys |= set(backend_post[b].keys())
    # also count shapes that were in shards but produced no valid post as tuned
    # (so we don't retry them forever); derive from prepare shard files if present
    for g in glob.glob(os.path.join(args.workdir, "shards", "shard_*.csv")):
        with open(g) as f:
            for row in csv.DictReader(f):
                try:
                    tuned_keys.add((int(row["M"]), int(row["N"]), int(row["K"])))
                except (KeyError, ValueError):
                    pass

    for k in sorted(tuned_keys):
        dflt = median(default_us[k]) if default_us.get(k) else None
        cands = {b: backend_post[b][k] for b in backends if k in backend_post[b]}
        best_b, best_post, improved = None, None, False
        if cands:
            best_b = min(cands, key=cands.get)
            best_post = cands[best_b]
            if dflt is not None and best_post < dflt * (1.0 - thr):
                improved = True
        if improved and k in backend_rows.get(best_b, {}):
            winners[k] = (best_b, best_post, backend_rows[best_b][k])
            final_pick[k] = backend_rows[best_b][k]
            chosen = best_b
        elif args.write_native and k in native_rows:
            # OPTIONAL: explicit torch-native row. OFF by default because, when this
            # CSV is merged ON TOP of aiter's bundled configs, a torch-native row
            # OVERRIDES aiter's already-tuned (often faster) kernel for that shape
            # -> regression. By omitting it, "no improvement" shapes fall through to
            # whatever the deployed base config (aiter's) provides.
            final_pick[k] = native_rows[k]
            chosen = "torch(native)"
        else:
            chosen = "default" if not improved else (best_b or "none")
        # ledger row (record regardless of improvement)
        base = master_by_key.get(k, {})
        ledger_new.append({
            "M": k[0], "N": k[1], "K": k[2],
            "bias": base.get("bias", "False"),
            "dtype": base.get("dtype", "torch.bfloat16"),
            "outdtype": base.get("outdtype", "torch.bfloat16"),
            "scaleAB": base.get("scaleAB", "False"),
            "bpreshuffle": base.get("bpreshuffle", "False"),
            "best_backend": chosen,
            "default_us": f"{dflt:.4f}" if dflt else "",
            "best_post_us": f"{best_post:.4f}" if (improved and best_post) else (f"{dflt:.4f}" if dflt else ""),
            "improved": "yes" if improved else "no",
            "all_backends_us": ";".join(f"{b}:{cands[b]:.3f}" for b in cands),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    # ---- write/merge FINAL tuned CSV ----
    final_rows = {}
    if os.path.exists(args.final) and csv_header:
        with open(args.final) as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    final_rows[(int(row["M"]), int(row["N"]), int(row["K"]))] = row
                except (KeyError, ValueError):
                    pass
    for k, row in final_pick.items():
        final_rows[k] = row
    if csv_header:
        os.makedirs(os.path.dirname(args.final), exist_ok=True)
        with open(args.final, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_header)
            w.writeheader()
            for k in sorted(final_rows):
                w.writerow({c: final_rows[k].get(c, "") for c in csv_header})

    # ---- append/merge ledger ----
    led_fields = ["M", "N", "K", "bias", "dtype", "outdtype", "scaleAB", "bpreshuffle",
                  "best_backend", "default_us", "best_post_us", "improved",
                  "all_backends_us", "updated_at"]
    existing = {}
    if os.path.exists(args.ledger):
        with open(args.ledger) as f:
            for row in csv.DictReader(f):
                try:
                    existing[(int(row["M"]), int(row["N"]), int(row["K"]))] = row
                except (KeyError, ValueError):
                    pass
    for row in ledger_new:
        existing[(row["M"], row["N"], row["K"])] = row
    with open(args.ledger, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=led_fields)
        w.writeheader()
        for k in sorted(existing, key=lambda x: (int(x[0]), int(x[1]), int(x[2]))):
            w.writerow({c: existing[k].get(c, "") for c in led_fields})

    # ---- summary ----
    bywin = defaultdict(int)
    for k, (b, post, row) in winners.items():
        bywin[b] += 1
    native_written = sum(1 for k in final_pick if k not in winners)
    print("===== RECONCILE SUMMARY =====")
    print(f"shapes tuned this run : {len(tuned_keys)}")
    print(f"shapes improved (kept): {len(winners)}")
    print(f"explicit native rows  : {native_written} (no backend beat default)")
    print("winners by backend    : " + (", ".join(f"{b}={n}" for b, n in sorted(bywin.items(), key=lambda x:-x[1])) or "none"))
    print(f"final tuned CSV       : {args.final} ({len(final_rows)} rows)")
    print(f"ledger                : {args.ledger} ({len(existing)} rows)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--input", required=True)
    p.add_argument("--ledger", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--nshards", type=int, default=8)
    p.set_defaults(func=cmd_prepare)
    r = sub.add_parser("reconcile")
    r.add_argument("--workdir", required=True)
    r.add_argument("--backends", required=True)
    r.add_argument("--input", required=True)
    r.add_argument("--ledger", required=True)
    r.add_argument("--final", required=True)
    r.add_argument("--min-improvement-pct", type=float, default=5.0)
    r.add_argument("--write-native", action="store_true", default=False,
                   help="Also emit explicit torch-native rows for shapes no backend "
                        "improved. OFF by default: such rows override a faster base "
                        "(e.g. aiter) config when merged on top of it.")
    r.set_defaults(func=cmd_reconcile)
    fl = sub.add_parser("filter")
    fl.add_argument("--to-tune-dir", required=True, dest="to_tune_dir")
    fl.add_argument("--candidates", default="")
    fl.add_argument("--reports", default="")
    fl.add_argument("--tuned", default="")
    fl.add_argument("--outdir", required=True)
    fl.add_argument("--nshards", type=int, default=8)
    fl.set_defaults(func=cmd_filter)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
