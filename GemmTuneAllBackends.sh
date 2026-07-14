#!/usr/bin/env bash
# Single-script multi-backend BF16 GEMM tuner for amd/GLM-5.1-MXFP4 (TP=4).
#
# Design (run INSIDE the container; uses /sgl-workspace/aiter 6/9+, supports opus):
#   1. Read the backend list dynamically from aiter's ALL_LIBTYPES (no hardcoding;
#      auto-picks up new backends like opus).
#   2. prepare: only tune shapes NOT already in the ledger (add new GEMM sizes later
#      without re-tuning everything).
#   3. Loop over each backend separately (isolates flydsl's timeout from the rest,
#      and prints per-backend wall-time so you see which backend is slow).
#   4. reconcile: pick the best backend per shape using PRODUCTION gemm_a16w16()
#      compare us (NOT sweep micro-us -> avoids "micro good but E2E bad"); keep only
#      shapes that beat default by >= MIN_IMPROVEMENT_PCT; record EVERY tuned shape
#      in the ledger (incl. no-improvement ones, so they are skipped next time).
#
# Final acceptance is still a real SGLang E2E benchmark after deploy.
set -euo pipefail

export HIP_FORCE_DEV_KERNARG=1
export ROCBLAS_LAYER=0
export HIPBLASLT_LOG_LEVEL=0
export GPU_MAX_HW_QUEUES=8

# Disable NUMA auto-balancing to avoid page-migration jitter during tuning.
# This is a HOST kernel setting; inside a container it may be read-only or denied,
# so this is best-effort and must not abort the run (|| true).
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing' 2>/dev/null \
  || sh -c 'echo 0 > /proc/sys/kernel/numa_balancing' 2>/dev/null \
  || echo "note: could not set numa_balancing (set it on the HOST as root); current=$(cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo n/a)"

GEMM_DIR="${HOME}/SGLang-benchmarks/amd_gemm_tuning/glm5.1_mxfp4_tp4"
MASTER_INPUT="${GEMM_DIR}/glm5.1_mxfp4_gemm_input.csv"
RECONCILE="${HOME}/SGLang-benchmarks/gemm_reconcile.py"
PROGRESS="${HOME}/SGLang-benchmarks/gemm_progress.py"

AITER_DIR="/sgl-workspace/aiter"
TUNER_DIRECT="${AITER_DIR}/csrc/gemm_a16w16/gemm_a16w16_tune.py"   # non-hipblaslt
TUNER_HIPB="${AITER_DIR}/csrc/gemm_a16w16/gemm_tuner.py"           # hipblaslt wrapper
TUNE_SRC="${TUNER_DIRECT}"   # ALL_LIBTYPES lives here

[[ -f "${MASTER_INPUT}" ]] || { echo "ERROR: master input ${MASTER_INPUT} missing" >&2; exit 1; }
[[ -f "${TUNE_SRC}"     ]] || { echo "ERROR: tuner ${TUNE_SRC} missing (recreate container / 6/9 aiter)" >&2; exit 1; }

# Version-stamp all outputs by aiter commit. solidx/kernelName are only valid for
# the exact aiter that produced them, so a different aiter = a fresh tree = correct
# re-tune (the ledger is per-version, so it won't wrongly skip shapes on a new aiter).
AITER_TAG="$(git -C "${AITER_DIR}" rev-parse --short HEAD 2>/dev/null || date +%Y%m%d)"
VERROOT="${GEMM_DIR}/aiter_${AITER_TAG}"
WORK="${VERROOT}/allbackends"
LEDGER="${VERROOT}/tuned_ledger.csv"
FINAL="${VERROOT}/tuned/glm5.1_mxfp4_bf16_tuned_gemm.csv"

NGPU=8
MIN_IMPROVEMENT_PCT=5
# NOTE: --timeout omitted on purpose -> aiter default = None (no per-task timeout),
# matching how aiter tuned its shipped configs. A 120s cap previously truncated the
# flydsl/asm sweeps and missed the deep split_k kernels (small-M was much worse than
# aiter). Trade-off: with no timeout a genuine GPU hang stalls the run (non-hipblaslt
# backends here have no crash-retry wrapper); if that happens, re-add --timeout 1200
# or run the hipblaslt-style subprocess wrapper.
COMMON_ARGS=( --splitK --warmup 10 --iters 50 --batch 1
              --compare --update_improved --min_improvement_pct "${MIN_IMPROVEMENT_PCT}" )

mkdir -p "${WORK}" "$(dirname "${FINAL}")"
TIMINGS="${VERROOT}/backend_timings.csv"
[[ -f "${TIMINGS}" ]] || echo "backend,seconds,status,processed_shapes,finished_at" > "${TIMINGS}"
# record provenance
{
    echo "aiter_dir: ${AITER_DIR}"
    echo "aiter_commit: $(git -C "${AITER_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "aiter_commit_date: $(git -C "${AITER_DIR}" log -1 --format=%ci 2>/dev/null || echo unknown)"
    echo "tuned_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host: $(hostname)"
} > "${VERROOT}/AITER_VERSION.txt"
echo "aiter tag: ${AITER_TAG}  ->  outputs under ${VERROOT}"

# --- 1. dynamic backend list from aiter ALL_LIBTYPES (drop "all") ---
mapfile -t BACKENDS < <(python3 - "${TUNE_SRC}" <<'PY'
import re, sys
src = open(sys.argv[1]).read()
m = re.search(r"ALL_LIBTYPES\s*=\s*\[(.*?)\]", src, re.S)
libs = re.findall(r'"([^"]+)"', m.group(1)) if m else []
for b in libs:
    if b != "all":
        print(b)
PY
)
[[ ${#BACKENDS[@]} -gt 0 ]] || { echo "ERROR: could not read ALL_LIBTYPES from ${TUNE_SRC}" >&2; exit 1; }
echo "Backends from aiter: ${BACKENDS[*]}"

# Reorder by predicted tuning speed (fast -> slow) so the bulk of the value lands
# early and the slow/crash-prone backends (hipblaslt, flydsl) run last:
#   torch(1 cfg) < skinny(1 cfg, often skipped) < triton(1 cfg+JIT) < opus(curated)
#   < asm(tens of tiles) < hipblaslt(2084 + wrapper/retry) < flydsl(6k-10k + JIT).
# Any backend not listed here is appended at the end in aiter's original order.
PRIORITY=( torch skinny triton opus asm hipblaslt flydsl )
ORDERED=()
for p in "${PRIORITY[@]}"; do
    for b in "${BACKENDS[@]}"; do
        [[ "${b}" == "${p}" ]] && ORDERED+=( "${b}" ) && break
    done
done
for b in "${BACKENDS[@]}"; do
    skip=0
    for o in "${ORDERED[@]}"; do [[ "${b}" == "${o}" ]] && skip=1 && break; done
    [[ "${skip}" -eq 0 ]] && ORDERED+=( "${b}" )
done
BACKENDS=( "${ORDERED[@]}" )
echo "Tuning order (fast->slow): ${BACKENDS[*]}"

# --- 2. prepare: only tune shapes not yet in ledger ---
python3 "${RECONCILE}" prepare \
    --input "${MASTER_INPUT}" --ledger "${LEDGER}" \
    --outdir "${WORK}/shards" --nshards "${NGPU}"
NSHAPES="$(cat "${WORK}/shards/count")"
if [[ "${NSHAPES}" -eq 0 ]]; then
    echo "No new shapes to tune (all in ledger). Nothing to do."
    exit 0
fi

# --- 3. per-backend loop (isolated; timed) ---
for b in "${BACKENDS[@]}"; do
    BDIR="${WORK}/${b}"
    mkdir -p "${BDIR}"
    # Crash/reboot resume: skip backends already fully finished in a previous run.
    if [[ -f "${BDIR}/.done" ]]; then
        echo "Backend ${b}: already done (found ${BDIR}/.done), skipping."
        continue
    fi
    export TMPDIR="${BDIR}/compare_tmp"      # keep compare reports on host disk
    mkdir -p "${TMPDIR}"
    args=( "${COMMON_ARGS[@]}" --libtype "${b}" )
    tuner="${TUNER_DIRECT}"
    if [[ "${b}" == "hipblaslt" ]]; then
        args+=( --with-hipblaslt )
        tuner="${TUNER_HIPB}"
    fi
    echo "=================================================="
    echo "Tuning backend: ${b}"
    start=${SECONDS}

    # Per-shape resume: tune only shapes this backend has NOT processed yet.
    # 'processed' = shapes already in this backend's candidate CSVs (appended
    # per-batch as each shape completes with a valid result). An interrupted
    # multi-hour run resumes from where it stopped. NOTE: shapes that produced
    # NO valid result (e.g. flydsl timeouts) are not recorded, so they are
    # retried on resume.
    python3 "${RECONCILE}" filter \
        --to-tune-dir "${WORK}/shards" \
        --candidates "${BDIR}/compare_tmp/aiter_compare/*.candidate.csv" \
        --reports "${BDIR}/compare_tmp/aiter_compare/*.compare.txt" \
        --tuned "${BDIR}/shard_*.csv" \
        --outdir "${BDIR}/remaining" --nshards "${NGPU}"
    REM="$(cat "${BDIR}/remaining/count")"
    if [[ "${REM}" -eq 0 ]]; then
        touch "${BDIR}/.done"
        echo "Backend ${b}: nothing remaining, marked done ($((SECONDS-start))s)."
        echo "${b},$((SECONDS-start)),already_complete,-,$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${TIMINGS}"
        continue
    fi

    # live tqdm progress bar for this backend (tracks attempted shapes via the
    # compare-report "batch N/M" headers across all GPUs). Best-effort.
    rm -f "${BDIR}/.monitor_stop"
    python3 "${PROGRESS}" --backend-dir "${BDIR}" --total "${REM}" &
    monitor_pid=$!

    pids=()
    for g in $(seq 0 $((NGPU-1))); do
        shard="${BDIR}/remaining/shard_${g}.csv"
        [[ -s "${shard}" ]] || continue
        [[ "$(wc -l < "${shard}")" -gt 1 ]] || continue   # skip header-only
        # Per-backend per-GPU stdout log: persists "timed out" / "No valid
        # solutions" / "candidate count: 0" so we can later tell, per shape,
        # whether a no-result was a TIMEOUT (recoverable with longer --timeout)
        # or a genuine NO-SOLUTION. Appended so resumes keep history.
        CUDA_VISIBLE_DEVICES=${g} python3 "${tuner}" \
            --tuned_file "${BDIR}/shard_${g}.csv" \
            --input_file "${shard}" \
            "${args[@]}" >> "${BDIR}/run_gpu${g}.log" 2>&1 &
        pids+=( "$!" )
    done
    pass_ok=1
    for pid in "${pids[@]}"; do wait "${pid}" || pass_ok=0; done
    touch "${BDIR}/.monitor_stop"; wait "${monitor_pid}" 2>/dev/null || true

    elapsed=$((SECONDS-start))
    nproc=$(cat "${BDIR}"/compare_tmp/aiter_compare/*.candidate.csv 2>/dev/null | tail -n +2 | cut -d, -f3-5 | sort -u | wc -l)
    # A backend is DONE when its tuner pass exits cleanly (exit 0). Shapes with NO
    # valid solution (skinny narrow-only, asm non-conforming, flydsl timeouts,
    # opus unsupported) also exit 0 -- they are legitimately "attempted, no tuned
    # config" and must NOT be retried every run. Only a real crash (non-zero exit)
    # leaves the backend resumable (per-shape filter skips already-solved shapes).
    if [[ "${pass_ok}" -eq 1 ]]; then
        touch "${BDIR}/.done"
        echo "Backend ${b} took ${elapsed}s (complete; ${nproc} shapes had a solution)"
        echo "${b},${elapsed},complete,${nproc},$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${TIMINGS}"
    else
        echo "Backend ${b} took ${elapsed}s (CRASHED: a tuner exited non-zero; rerun to resume)"
        echo "${b},${elapsed},crashed,${nproc},$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${TIMINGS}"
    fi
done

# --- 4. reconcile (production-gated cross-backend selection + ledger) ---
python3 "${RECONCILE}" reconcile \
    --workdir "${WORK}" \
    --backends "$(IFS=,; echo "${BACKENDS[*]}")" \
    --input "${MASTER_INPUT}" \
    --ledger "${LEDGER}" \
    --final "${FINAL}" \
    --min-improvement-pct "${MIN_IMPROVEMENT_PCT}"

echo "Done. Final tuned CSV: ${FINAL}"
echo "Ledger (tuned incl. no-improvement): ${LEDGER}"
echo "Per-backend outputs + compare reports under: ${WORK}/<backend>/"
