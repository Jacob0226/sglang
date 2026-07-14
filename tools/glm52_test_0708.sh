#!/usr/bin/env bash
# Launch the 0708.sh server config (no tilelang NSA, no allreduce fusion,
# chunked-prefill 131072) and benchmark i8k/o1k/conc64 to compare Median TPOT
# against the GLM.sh run (56 ms). Runs inside jacchang_GLM5.
set -uo pipefail

CONTAINER="jacchang_GLM5"
PORT=8552
MODEL="/data/huggingface/hub/amd/GLM-5.2-MXFP4"
OUTDIR="/home/jacchang/SGLang-benchmarks/results/amd_GLM-5.2-MXFP4/rocm_sgl-dev-v0.5.14-rocm720-mi35x-20260628/test-0708_i8k_conc64"
SRV_LOG="${OUTDIR}/server.log"
BENCH_LOG="${OUTDIR}/bench_in8192_out1024_conc64.log"

log() { echo "[$(date '+%F %T')] $*"; }
mkdir -p "$OUTDIR" 2>/dev/null || docker exec "$CONTAINER" mkdir -p "$OUTDIR"

log "Starting 0708.sh server (port ${PORT})..."
docker exec -d -e MODEL="$MODEL" "$CONTAINER" bash -lc \
    "cd /home/jacchang/SGLang-benchmarks && bash 0708.sh > '${SRV_LOG}' 2>&1"

log "Waiting for server health..."
for i in $(seq 1 90); do
    code=$(docker exec "$CONTAINER" bash -lc "curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health" 2>/dev/null || echo 000)
    if [ "$code" = "200" ]; then log "Server ready after ~$((i*10))s."; break; fi
    if ! docker exec "$CONTAINER" bash -lc "pgrep -f 'sglang serve' >/dev/null"; then
        log "!!! server process died during startup — see ${SRV_LOG}"; exit 1
    fi
    sleep 10
done

log "Running bench i8k/o1k/conc64 (num-prompt 320)..."
docker exec "$CONTAINER" bash -lc "
    python3 -m sglang.bench_serving \
        --host localhost --port ${PORT} --model '${MODEL}' \
        --dataset-name random --random-input 8192 --random-output 1024 \
        --random-range-ratio 0.8 --max-concurrency 64 --num-prompt 320 \
        --output-file /dev/null 2>&1 | tee '${BENCH_LOG}'
"

log "Tearing down server..."
docker exec "$CONTAINER" bash -lc "pkill -9 -f 'sglang serve' || true; sleep 3"

log "=== Result summary ==="
grep -E 'Median TPOT|Mean TPOT|P90 TPOT|Median TTFT|Total token throughput|Max ITL|Concurrency:' "$BENCH_LOG" || true
