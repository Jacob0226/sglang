#!/usr/bin/env bash
# Watch GPU VRAM and, once >=4 GPUs are free, rerun the noisy GLM-5.2 i1k/conc8
# benchmark (opt1+elemfix+allreduce) on those 4 GPUs, then regenerate the CSV.
#
# Runs on the HOST; the actual bench runs via `docker exec` inside jacchang_GLM5
# (which mounts /home/jacchang, so logs land on the host directly).
set -uo pipefail

# Which i1k concurrency to rerun (default 8). Pass as first arg, e.g. `... 4`.
CONC="${1:-8}"
CONTAINER="jacchang_GLM5"
RESULT_DIR="/home/jacchang/SGLang-benchmarks/results/amd_GLM-5.2-MXFP4/rocm_sgl-dev-v0.5.14-rocm720-mi35x-20260628/bench-TP4_Opt1_ElemFix_AllReduce"
CONC_LOGNAME="bench_in1024_out1024_conc${CONC}.log"
FINISH_LOG="${RESULT_DIR}/Finish.log"
CSV_OUT="/home/jacchang/perf_GLM5.2_opt1_elemfix_allreduce_TP4.csv"

# A GPU counts as "free" if it uses < this many bytes (need ~255GB free for
# GLM-5.2 MXFP4 TP4 at mem-fraction 0.85). 45GB threshold.
FREE_THRESH=$((45 * 1024 * 1024 * 1024))
NEED_GPUS=4
POLL_SEC=30

log() { echo "[$(date '+%F %T')] $*"; }

# Print indices (0..7) of GPUs whose used VRAM < FREE_THRESH, space-separated.
free_gpus() {
    local i=0
    rocm-smi --showmeminfo vram 2>/dev/null \
        | grep -i 'Used Memory' \
        | grep -oE '[0-9]+$' \
        | while read -r used; do
            if [ "$used" -lt "$FREE_THRESH" ]; then echo -n "$i "; fi
            i=$((i + 1))
          done
}

log "Watcher started. Need ${NEED_GPUS} GPUs with <45GB used. Polling every ${POLL_SEC}s."

DEVS=""
while true; do
    # Prefer 0-3 for consistency with the original run; else take first 4 free.
    mapfile -t all_free < <(free_gpus | tr ' ' '\n' | grep -E '^[0-7]$')
    n=${#all_free[@]}
    log "Free GPUs (${n}): ${all_free[*]:-none}"

    if [ "$n" -ge "$NEED_GPUS" ]; then
        # confirm stable: re-check after a short pause
        sleep 15
        mapfile -t all_free2 < <(free_gpus | tr ' ' '\n' | grep -E '^[0-7]$')
        if [ "${#all_free2[@]}" -ge "$NEED_GPUS" ]; then
            if printf '%s\n' "${all_free2[@]}" | grep -qx 0 \
               && printf '%s\n' "${all_free2[@]}" | grep -qx 1 \
               && printf '%s\n' "${all_free2[@]}" | grep -qx 2 \
               && printf '%s\n' "${all_free2[@]}" | grep -qx 3; then
                DEVS="0,1,2,3"
            else
                DEVS=$(printf '%s,' "${all_free2[@]:0:4}"); DEVS="${DEVS%,}"
            fi
            log "Stable free set confirmed. Using HIP_VISIBLE_DEVICES=${DEVS}"
            break
        fi
        log "Free set not stable on recheck; keep polling."
    fi
    sleep "$POLL_SEC"
done

# --- Prepare clean rerun: drop stale log + its Finish.log entry ---
# Do it THROUGH the container (root) because result logs are root-owned.
log "Clearing stale i1k/conc${CONC} artifacts (via container root)."
docker exec "$CONTAINER" bash -lc "
    cd '$RESULT_DIR' &&
    rm -f '$CONC_LOGNAME' &&
    if [ -f Finish.log ]; then
        grep -v '$CONC_LOGNAME' Finish.log > Finish.log.tmp 2>/dev/null || true
        mv Finish.log.tmp Finish.log
    fi
"

# --- Launch the scoped rerun inside the container ---
log "Launching GLM.sh scoped to i1k/conc${CONC} on GPUs ${DEVS}..."
docker exec \
    -e HIP_VISIBLE_DEVICES="$DEVS" \
    -e IN_OUT_OVERRIDE="1024:1024" \
    -e CONC_OVERRIDE="$CONC" \
    "$CONTAINER" bash -lc '
        cd ~/SGLang-benchmarks &&
        ./GLM.sh \
            --model /data/huggingface/hub/amd/GLM-5.2-MXFP4 \
            --tp 4 \
            --docker rocm/sgl-dev:v0.5.14-rocm720-mi35x-20260628 \
            --tag TP4_Opt1_ElemFix_AllReduce
    '
rc=$?
log "GLM.sh finished (rc=${rc})."

# --- Regenerate CSV in the delivery format ---
log "Regenerating CSV: ${CSV_OUT}"
cd /home/jacchang/SGLang-benchmarks && python3 parse_perf_metrics_to_csv.py \
    --input_dir "$RESULT_DIR" \
    --output "$CSV_OUT" \
    --tp 4 --precision fp4 --framework SGLang \
    --hardware MI355X --machine smci355-ccs-aus-n11-13 \
    --image rocm/sgl-dev:v0.5.14-rocm720-mi35x-20260628 \
    --commit "https://github.com/Jacob0226/sglang/commit/a96febeb09c349dffc9af4cf6aab"

log "Done. New i1k/conc${CONC} row:"
grep -E "^1024,1024,${CONC}," "$CSV_OUT" || true
