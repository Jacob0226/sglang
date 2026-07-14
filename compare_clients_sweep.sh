#!/usr/bin/env bash
# compare_clients_sweep.sh — sweep the client A/B across conc x seq-len.
# Calls compare_clients.sh per (isl:osl, conc) point and aggregates a CSV +
# a final side-by-side table. Each point restarts the server (faithful to IX:
# cuda-graph-max-bs=conc, context-length per seq).
#
# Usage:
#   ./compare_clients_sweep.sh                                  # default grid
#   ./compare_clients_sweep.sh --seqs "1024:1024 8192:1024" --conc "4 16 64 128"
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEQS="1024:1024 8192:1024"
CONC="4 16 64 128"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --seqs) SEQS="$2"; shift 2 ;;
    --conc) CONC="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift 1 ;;
  esac
done

TP_GUESS=4; [ -e /dev/kfd ] || TP_GUESS=8
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="${OUT_ROOT:-${SCRIPT_DIR}/results}"
export OUT_ROOT
CSV="${OUT_ROOT}/client_ab_sweep_${STAMP}.csv"
mkdir -p "${OUT_ROOT}"
echo "isl,osl,conc,A_total_tps,B_total_tps,A_pergpu,B_pergpu,B_over_A,A_itl_ms,B_itl_ms,A_dur_s,B_dur_s" > "$CSV"

grab() { grep -iE "$2" "$1" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | tail -n1; }

for seq in $SEQS; do
  isl="${seq%%:*}"; osl="${seq##*:}"
  for c in $CONC; do
    echo ">>> [$(date +%H:%M:%S)] point isl=$isl osl=$osl conc=$c"
    "${SCRIPT_DIR}/compare_clients.sh" --conc "$c" --isl "$isl" --osl "$osl" "${EXTRA_ARGS[@]}" \
        > "${OUT_ROOT}/_sweep_point_in${isl}_out${osl}_c${c}.log" 2>&1 || \
        echo "    point failed (see _sweep_point_in${isl}_out${osl}_c${c}.log)"

    d="${OUT_ROOT}/client_ab/mi355x_conc${c}_in${isl}_out${osl}_tp${TP_GUESS}_mtptrue"
    [ -d "$d" ] || d="${OUT_ROOT}/client_ab/b200_conc${c}_in${isl}_out${osl}_tp${TP_GUESS}_mtptrue"
    a_log="$d/A_sglang_client.log"; b_log="$d/B_ix_client.log"

    a_tot=$(grab "$a_log" "Total token throughput");  b_tot=$(grab "$b_log" "Total Token throughput")
    a_itl=$(grab "$a_log" "Mean ITL");                b_itl=$(grab "$b_log" "Mean ITL")
    a_dur=$(grab "$a_log" "Benchmark duration");      b_dur=$(grab "$b_log" "Benchmark duration")
    a_pg=$(awk -v x="${a_tot:-0}" -v g="$TP_GUESS" 'BEGIN{printf "%.1f", x/g}')
    b_pg=$(awk -v x="${b_tot:-0}" -v g="$TP_GUESS" 'BEGIN{printf "%.1f", x/g}')
    ratio=$(awk -v a="${a_tot:-0}" -v b="${b_tot:-0}" 'BEGIN{printf (a>0)?"%.3f":"NA", b/a}')

    echo "${isl},${osl},${c},${a_tot:-NA},${b_tot:-NA},${a_pg},${b_pg},${ratio},${a_itl:-NA},${b_itl:-NA},${a_dur:-NA},${b_dur:-NA}" >> "$CSV"
    echo "    A=${a_tot:-NA} B=${b_tot:-NA} (per-gpu A=${a_pg} B=${b_pg}, B/A=${ratio})"
  done
done

echo ""
echo "================== CLIENT A/B SWEEP (per-GPU total tok/s) =================="
column -s, -t "$CSV"
echo "==========================================================================="
echo "CSV: $CSV"
