#!/usr/bin/env python3
"""
Fair, same-machine head-to-head between two GEMM tuned configs, then build a
no-regression merged config.

Why: aiter's runtime "merge" of tuned CSVs is a blind key-override (dedup
keep-last), NOT a per-shape "pick the faster". And the `us` already recorded in
each CSV was measured on different machines/versions, so it is not comparable
(same kernel can differ ~10%). So we RE-MEASURE both configs here, on this
machine, through the real production gemm_a16w16() op, and only keep our row
where it is genuinely faster than the baseline.

Usage (run inside the aiter container):
  # 1) benchmark one config (separate process => clean config cache)
  python3 gemm_headtohead.py bench --config <A.csv|default> --input shapes.csv --out usA.csv
  # 2) orchestrate everything (spawns bench twice, merges)
  python3 gemm_headtohead.py run --baseline <A.csv|default> --ours <ours.csv> \
      --input shapes.csv --final server.csv [--margin 0.0]
"""
import argparse, csv, os, subprocess, sys


def cmd_bench(args):
    # MUST set env before importing aiter so the config is picked up.
    if args.config and args.config != "default":
        os.environ["AITER_CONFIG_GEMM_BF16"] = args.config
    import torch
    from aiter.tuned_gemm import gemm_a16w16
    from aiter.test_common import run_perftest

    rows = list(csv.DictReader(open(args.input)))
    out = open(args.out, "w", newline="")
    w = csv.writer(out)
    w.writerow(["M", "N", "K", "us", "status"])
    dt = torch.bfloat16
    for r in rows:
        M, N, K = int(r["M"]), int(r["N"]), int(r["K"])
        try:
            A = torch.randn(M, K, dtype=dt, device="cuda")
            B = torch.randn(N, K, dtype=dt, device="cuda")
            _, us = run_perftest(
                gemm_a16w16, A, B, otype=dt,
                num_warmup=args.warmup, num_iters=args.iters,
            )
            w.writerow([M, N, K, round(float(us), 4), "ok"])
            out.flush()
            del A, B
            torch.cuda.empty_cache()
        except Exception as e:
            w.writerow([M, N, K, -1, f"err:{str(e)[:60]}"])
            out.flush()
    out.close()
    print(f"[bench] config={args.config} -> {args.out} ({len(rows)} shapes)")


def load_us(path):
    d = {}
    for r in csv.DictReader(open(path)):
        try:
            d[(int(r["M"]), int(r["N"]), int(r["K"]))] = float(r["us"])
        except (ValueError, KeyError):
            pass
    return d


def load_rows(path):
    """key -> full row dict (for building final CSV)."""
    d, hdr = {}, None
    if os.path.exists(path):
        rr = csv.DictReader(open(path))
        hdr = rr.fieldnames
        for r in rr:
            try:
                d[(int(r["M"]), int(r["N"]), int(r["K"]))] = r
            except (ValueError, KeyError):
                pass
    return d, hdr


def cmd_run(args):
    here = os.path.abspath(__file__)
    usA_path, usB_path = "/tmp/h2h_usA.csv", "/tmp/h2h_usB.csv"
    # baseline (A): merged aiter config (or "default" auto-merge)
    subprocess.run([sys.executable, here, "bench", "--config", args.baseline,
                    "--input", args.input, "--out", usA_path,
                    "--warmup", str(args.warmup), "--iters", str(args.iters)], check=True)
    # ours (B): the explicit merged csv that already contains aiter rows + our overrides
    subprocess.run([sys.executable, here, "bench", "--config", args.ours,
                    "--input", args.input, "--out", usB_path,
                    "--warmup", str(args.warmup), "--iters", str(args.iters)], check=True)

    usA, usB = load_us(usA_path), load_us(usB_path)
    ours_rows, hdr = load_rows(args.ours)
    base_rows, _ = load_rows(args.baseline) if args.baseline != "default" else ({}, None)

    shared = sorted(set(usA) & set(usB))
    win = lose = tie = 0
    final = {}
    report = []
    for k in shared:
        a, b = usA[k], usB[k]
        if a <= 0 or b <= 0:
            continue
        if b < a * (1.0 - args.margin / 100.0):      # ours faster
            win += 1
            if k in ours_rows:
                final[k] = ours_rows[k]
            report.append((k, a, b, "OURS", (a - b) / a * 100))
        elif b > a * (1.0 + args.margin / 100.0):    # ours slower -> keep baseline
            lose += 1
            if k in base_rows:
                final[k] = base_rows[k]
            report.append((k, a, b, "BASE", (a - b) / a * 100))
        else:
            tie += 1
            if k in base_rows:
                final[k] = base_rows[k]

    # write final no-regression CSV (only where we have a row to write)
    if hdr:
        with open(args.final, "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=hdr)
            wcsv.writeheader()
            for k in sorted(final):
                wcsv.writerow({c: final[k].get(c, "") for c in hdr})

    print("\n===== HEAD-TO-HEAD (same-machine production gemm_a16w16) =====")
    print(f"shapes compared = {len(shared)}")
    print(f"OURS faster = {win}   BASELINE faster = {lose}   tie(±{args.margin}%) = {tie}")
    # net speedup over compared shapes
    sa = sum(usA[k] for k in shared if usA[k] > 0 and usB[k] > 0)
    sb = sum(min(usA[k], usB[k]) for k in shared if usA[k] > 0 and usB[k] > 0)
    if sa:
        print(f"if pick-faster: total us {sa:.0f} -> {sb:.0f} ({(sa-sb)/sa*100:.1f}% faster)")
    print(f"final no-regression CSV -> {args.final} ({len(final)} rows)")
    print("\nbiggest OURS wins:")
    for k, a, b, who, p in sorted([r for r in report if r[3] == "OURS"], key=lambda x: -x[4])[:10]:
        print(f"  {k}: base={a:.2f} ours={b:.2f}us  +{p:.1f}%")
    print("\nbiggest cases we'd have REGRESSED (now reverted to baseline):")
    for k, a, b, who, p in sorted([r for r in report if r[3] == "BASE"], key=lambda x: x[4])[:10]:
        print(f"  {k}: base={a:.2f} ours={b:.2f}us  {p:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--config", required=True)
    b.add_argument("--input", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--warmup", type=int, default=10)
    b.add_argument("--iters", type=int, default=50)
    b.set_defaults(func=cmd_bench)
    r = sub.add_parser("run")
    r.add_argument("--baseline", required=True, help="merged aiter CSV, or 'default'")
    r.add_argument("--ours", required=True, help="merged aiter+ours CSV (env-set, no auto-merge)")
    r.add_argument("--input", required=True)
    r.add_argument("--final", required=True)
    r.add_argument("--warmup", type=int, default=10)
    r.add_argument("--iters", type=int, default=50)
    r.add_argument("--margin", type=float, default=0.0, help="%% tie band")
    r.set_defaults(func=cmd_run)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
