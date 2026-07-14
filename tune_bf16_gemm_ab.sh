#!/usr/bin/env bash
# tune_bf16_gemm_ab.sh — aiter bf16 dense-GEMM tuning + before/after A/B profile.
#
# WHY: the server log shows
#   "[aiter] ... not found tuned config in /tmp/aiter_configs/bf16_tuned_gemm.csv,
#    will use default config! using torch solution:0"
# i.e. the dense bf16 GEMMs (28% of the CONC=4 decode step) run UNTUNED.
# This script: (1) collects the exact decode GEMM shapes, (2) tunes them with
# aiter's gemm_tuner, (3) re-profiles with the tuned CSV so you can measure the win.
#
# RUN THIS INSIDE THE SAME CONTAINER WHERE GLM.sh RUNS (needs GPUs + sglang + aiter).
#
# Usage (typical):
#   ./tune_bf16_gemm_ab.sh all          # collect -> tune -> A/B profile
#   ./tune_bf16_gemm_ab.sh collect      # only run the AITER_TUNE_GEMM=1 collection pass
#   ./tune_bf16_gemm_ab.sh tune         # only run the tuner on the collected/seeded csv
#   ./tune_bf16_gemm_ab.sh profile      # only run baseline + tuned --prof and compare
#
# Override paths/model via env, e.g.:
#   MODEL=amd/GLM-5.1-MXFP4 TP=4 ./tune_bf16_gemm_ab.sh all
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AITER_DIR="${AITER_DIR:-$(python3 -c 'import aiter,os;print(os.path.dirname(aiter.__file__))' 2>/dev/null)}"
TUNED_CSV="${AITER_CONFIG_GEMM_BF16:-/tmp/aiter_configs/bf16_tuned_gemm.csv}"
UNTUNED_CSV="${UNTUNED_CSV:-${AITER_DIR}/configs/bf16_untuned_gemm.csv}"
SEED_CSV="${SEED_CSV:-${HERE}/seed_bf16_untuned_gemm.csv}"
MODEL="${MODEL:-amd/GLM-5.1-MXFP4}"
TP="${TP:-4}"
GLM="${GLM:-${HERE}/GLM.sh}"

echo "aiter dir   : ${AITER_DIR:-<not found>}"
echo "untuned csv : ${UNTUNED_CSV}"
echo "tuned csv   : ${TUNED_CSV}"
mkdir -p "$(dirname "${TUNED_CSV}")"

collect() {
  # Run a short decode workload with AITER_TUNE_GEMM=1 so aiter records every
  # (M,N,K,dtype) it actually executes into ${UNTUNED_CSV}. The decode M = the
  # running batch size (= concurrency), which is exactly what we want to tune.
  echo ">>> [collect] starting server with AITER_TUNE_GEMM=1, running short bench..."
  # Seed first so the tuner always has the known log shapes even if auto-collect is sparse.
  if [ -f "${SEED_CSV}" ]; then
    { cat "${UNTUNED_CSV}" 2>/dev/null; tail -n +2 "${SEED_CSV}"; } | sort -u > /tmp/_u.csv && mv /tmp/_u.csv "${UNTUNED_CSV}"
  fi
  # Drive GLM.sh in plain (non-prof) bench mode; AITER_TUNE_GEMM is inherited by
  # the launched sglang server. Keep it short via a small bench (edit GLM.sh
  # concurrencies if you want fewer points).
  AITER_TUNE_GEMM=1 "${GLM}" --model "${MODEL}" --tp "${TP}" --tag gemmcollect \
      || echo "[warn] collect run exited non-zero (ok if traces/shapes already recorded)."
  echo ">>> [collect] untuned shapes now in ${UNTUNED_CSV}:"
  wc -l "${UNTUNED_CSV}" 2>/dev/null || true
}

tune() {
  echo ">>> [tune] running aiter gemm_tuner on ${UNTUNED_CSV} -> ${TUNED_CSV}"
  # gradlib's GemmTuner sweeps asm/ck/hipBLASLt solutions per shape, picks fastest.
  ( cd "${HERE}/../PR/aiter" 2>/dev/null || cd /home/jacchang/PR/aiter
    PYTHONPATH="gradlib:${PYTHONPATH:-}" python3 gradlib/gradlib/gemm_tuner.py \
        --input_file "${UNTUNED_CSV}" \
        --tuned_file "${TUNED_CSV}" \
        --indtype bf16 --outdtype bf16
  ) || { echo "[err] tuner failed"; return 1; }
  echo ">>> [tune] tuned rows:"; wc -l "${TUNED_CSV}" 2>/dev/null || true
}

profile() {
  echo ">>> [profile] BASELINE (no tuned csv)"
  AITER_CONFIG_GEMM_BF16="/tmp/_nonexistent_baseline.csv" \
      "${GLM}" --prof --model "${MODEL}" --tp "${TP}" --tag gemmBASE \
      || echo "[warn] baseline prof exited non-zero (profiler teardown is ok)."

  echo ">>> [profile] TUNED (AITER_CONFIG_GEMM_BF16=${TUNED_CSV})"
  AITER_CONFIG_GEMM_BF16="${TUNED_CSV}" \
      "${GLM}" --prof --model "${MODEL}" --tp "${TP}" --tag gemmTUNED \
      || echo "[warn] tuned prof exited non-zero (profiler teardown is ok)."

  echo ">>> [profile] done. Compare the two prof dirs with your analyze_trace.py /"
  echo "    compare_breakdown.py (look at the hgemm_bf16 / Cijk kernel us before vs after,"
  echo "    and confirm the 'not found tuned config ... torch solution:0' log is gone)."
}

case "${1:-all}" in
  collect) collect ;;
  tune)    tune ;;
  profile) profile ;;
  all)     collect; tune; profile ;;
  *) echo "usage: $0 {all|collect|tune|profile}"; exit 1 ;;
esac
