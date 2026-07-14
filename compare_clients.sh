#!/usr/bin/env bash
# compare_clients.sh — A/B the benchmark CLIENT, server held identical.
# ============================================================================
# Purpose: prove whether the gap vs InferenceX's published number comes from the
# benchmark *client* (the only remaining difference once docker image, TP, MTP
# and server args are matched).
#
# It starts ONE server (InferenceX GLM-5 FP8 MTP recipe), then drives it with
# BOTH clients at the same concurrency and prints their throughput side by side:
#   A) sglang.bench_serving            --backend sglang   (what GLM_IX.sh uses)
#   B) InferenceX benchmark_serving.py --backend vllm      (what the IX board uses)
#      (vendored into ./ix_bench_serving/, invoked exactly like benchmark_lib.sh)
#
# If B reports ~IX's low number while A reports your high number on the SAME
# server, the cause is the client/recipe -> PR-worthy. If both match A, the IX
# board number reflects their *environment* (CI contention etc.), not the recipe.
#
# Usage:
#   ./compare_clients.sh                       # conc4, 1024/1024, TP4, MTP, ROCm
#   ./compare_clients.sh --conc 32 --isl 8192 --osl 1024
#   ./compare_clients.sh --no-mtp --tp 8
#   ./compare_clients.sh --port 28553 --model /data/huggingface/hub/zai-org/GLM-5-FP8
# ----------------------------------------------------------------------------
set -uo pipefail
set -x

MODEL_PATH="/data/huggingface/hub/zai-org/GLM-5-FP8"
TP_SIZE="auto"
CONC=4
ISL=1024
OSL=1024
MTP_ENABLED="true"
HOST="0.0.0.0"
PORT="${PORT:-28553}"
RANDOM_RANGE_RATIO="0.8"
MEM_FRACTION_STATIC="0.85"
STREAM_INTERVAL="${STREAM_INTERVAL:-1}"
RUN_ONLY="${RUN_ONLY:-both}"   # both | a | b
A_BACKEND="${A_BACKEND:-sglang}"  # sglang (/generate) | sglang-oai (/v1/completions)
TAG="${TAG:-}"                 # optional suffix to keep variant results separate
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IX_CLIENT="${SCRIPT_DIR}/ix_bench_serving/benchmark_serving.py"

while [[ $# -gt 0 ]]; do
  case $1 in
    --model) MODEL_PATH="$2"; shift 2 ;;
    --tp)    TP_SIZE="$2"; shift 2 ;;
    --conc)  CONC="$2"; shift 2 ;;
    --isl)   ISL="$2"; shift 2 ;;
    --osl)   OSL="$2"; shift 2 ;;
    --no-mtp) MTP_ENABLED="false"; shift 1 ;;
    --port)  PORT="$2"; shift 2 ;;
    --stream-interval) STREAM_INTERVAL="$2"; shift 2 ;;
    --only) RUN_ONLY="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

is_rocm_gpu_env() { [ -e /dev/kfd ] || command -v rocm-smi >/dev/null 2>&1; }
if is_rocm_gpu_env; then PLATFORM="mi355x"; [ "$TP_SIZE" = "auto" ] && TP_SIZE=4
else PLATFORM="b200"; [ "$TP_SIZE" = "auto" ] && TP_SIZE=8; fi

# OUT_ROOT lets callers redirect heavy logs off a full / (e.g. OUT_ROOT=/data).
OUT_DIR="${OUT_ROOT:-${SCRIPT_DIR}/results}/client_ab/${PLATFORM}_conc${CONC}_in${ISL}_out${OSL}_tp${TP_SIZE}_mtp${MTP_ENABLED}_si${STREAM_INTERVAL}${TAG:+_$TAG}"
mkdir -p "$OUT_DIR"
SERVER_LOG="${OUT_DIR}/server.log"
A_LOG="${OUT_DIR}/A_sglang_client.log"
B_LOG="${OUT_DIR}/B_ix_client.log"

if [ ! -f "$IX_CLIENT" ]; then
    echo "ERROR: IX client not found at $IX_CLIENT" >&2
    echo "Copy InferenceX/utils/bench_serving/{benchmark_serving,backend_request_func,benchmark_utils,encoding_dsv4}.py into ${SCRIPT_DIR}/ix_bench_serving/" >&2
    exit 1
fi

# ---- start ONE server (InferenceX GLM-5 FP8 MTP recipe) ----
start_server() {
    echo ">>> starting server ($PLATFORM tp=$TP_SIZE mtp=$MTP_ENABLED)" | tee "$SERVER_LOG"
    local cmd
    if [ "$PLATFORM" = "mi355x" ]; then
        export SGLANG_ROCM_FUSED_DECODE_MLA=0 ROCM_QUICK_REDUCE_QUANTIZATION=INT4 SAFETENSORS_FAST_GPU=1
        cmd=(python3 -m sglang.launch_server --model-path "$MODEL_PATH" --host "$HOST" --port "$PORT"
            --tensor-parallel-size "$TP_SIZE" --trust-remote-code --tool-call-parser glm47 --reasoning-parser glm45
            --mem-fraction-static "$MEM_FRACTION_STATIC"
            --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}'
            --nsa-prefill-backend tilelang --nsa-decode-backend tilelang --kv-cache-dtype fp8_e4m3
            --tokenizer-worker-num $((TP_SIZE * 2)) --cuda-graph-max-bs "$CONC" --disable-radix-cache
            --stream-interval "$STREAM_INTERVAL")
        if [ "$MTP_ENABLED" = "true" ]; then
            export SGLANG_ENABLE_SPEC_V2=1
            cmd+=(--context-length $((ISL + OSL + 32)) --speculative-algorithm EAGLE
                --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
        else cmd+=(--max-running-requests "$CONC"); fi
    else
        export SGL_ENABLE_JIT_DEEPGEMM=1 PYTHONNOUSERSITE=1
        pip install --no-deps "transformers==5.2.0" "huggingface-hub==1.4.1" 2>/dev/null || true
        cmd=(python3 -m sglang.launch_server --model-path "$MODEL_PATH" --host "$HOST" --port "$PORT"
            --trust-remote-code --tensor-parallel-size "$TP_SIZE" --data-parallel-size 1 --expert-parallel-size 1
            --tool-call-parser glm47 --reasoning-parser glm45 --kv-cache-dtype fp8_e4m3 --quantization fp8
            --fp8-gemm-backend cutlass --attention-backend nsa --nsa-decode-backend trtllm --nsa-prefill-backend trtllm
            --moe-runner-backend flashinfer_trtllm --cuda-graph-max-bs "$CONC" --max-running-requests "$CONC"
            --mem-fraction-static "$MEM_FRACTION_STATIC" --chunked-prefill-size 32768 --max-prefill-tokens 32768
            --enable-flashinfer-allreduce-fusion --disable-radix-cache --stream-interval 30
            --model-loader-extra-config '{"enable_multithread_load": true}')
        if [ "$MTP_ENABLED" = "true" ]; then
            export SGLANG_ENABLE_SPEC_V2=1
            cmd+=(--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
        fi
    fi
    "${cmd[@]}" >> "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    echo ">>> waiting for health (pid=$SERVER_PID)..."
    until [ "$(curl -s -o /dev/null -w "%{http_code}" "http://${HOST}:$PORT/health")" -eq 200 ]; do
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died, see $SERVER_LOG"; exit 1; }
        sleep 5
    done
}
stop_server() {
    pkill -9 -P "${SERVER_PID:-0}" 2>/dev/null || true
    kill -9 "${SERVER_PID:-0}" 2>/dev/null || true
    pkill -9 -x python3 2>/dev/null || true   # name-exact: never matches this bash script
    sleep 10
}

NUM_PROMPTS=$((CONC * 10))
NUM_WARMUPS=$((CONC * 2))
CHAT_FLAG_SGL=(); CHAT_FLAG_IX=()
if [ "$MTP_ENABLED" = "true" ]; then CHAT_FLAG_SGL=(--apply-chat-template); CHAT_FLAG_IX=(--use-chat-template); fi

start_server

# ---- A) your client: sglang.bench_serving --backend sglang ----
if [ "$RUN_ONLY" != "b" ]; then
python3 -m sglang.bench_serving --backend "$A_BACKEND" --host "$HOST" --port "$PORT" --model "$MODEL_PATH" \
    --dataset-name random --random-input "$ISL" --random-output "$OSL" --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONC" --warmup-requests "$NUM_WARMUPS" \
    "${CHAT_FLAG_SGL[@]}" --output-file /dev/null 2>&1 | tee "$A_LOG"
fi

# ---- B) InferenceX client: benchmark_serving.py --backend vllm (exact recipe) ----
if [ "$RUN_ONLY" != "a" ]; then
B_WRAP=()
if [ "${PROFILE_B:-0}" = "1" ]; then
    B_WRAP=(py-spy record --idle --subprocesses --rate 100 --format raw \
        -o "${OUT_DIR}/ixprof.folded" --duration 9999 --)
fi
"${B_WRAP[@]}" python3 "$IX_CLIENT" --model "$MODEL_PATH" --backend vllm --base-url "http://${HOST}:$PORT" \
    --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONC" --request-rate inf --ignore-eos --save-result \
    --num-warmups "$NUM_WARMUPS" --percentile-metrics 'ttft,tpot,itl,e2el' \
    --result-dir "$OUT_DIR" --result-filename ix_result.json "${CHAT_FLAG_IX[@]}" 2>&1 | tee "$B_LOG"
fi

stop_server
set +x

# ---- side-by-side summary ----
grab() { grep -iE "$2" "$1" | tail -n1 | grep -oE '[0-9]+\.[0-9]+' | tail -n1; }
A_OUT=$(grab "$A_LOG" "Output token throughput")
A_TOT=$(grab "$A_LOG" "Total token throughput")
B_OUT=$(grab "$B_LOG" "Output token throughput")
B_TOT=$(grab "$B_LOG" "Total Token throughput")

echo ""
echo "================= CLIENT A/B (same server) ================="
echo "platform=$PLATFORM  tp=$TP_SIZE  mtp=$MTP_ENABLED  conc=$CONC  isl=$ISL  osl=$OSL"
printf "%-34s %12s %12s\n" "metric (tok/s)" "A:sglang" "B:IX-vllm"
printf "%-34s %12s %12s\n" "Output token throughput"        "${A_OUT:-NA}" "${B_OUT:-NA}"
printf "%-34s %12s %12s\n" "Total  token throughput"        "${A_TOT:-NA}" "${B_TOT:-NA}"
printf "%-34s %12s %12s\n" "per-GPU total (/$TP_SIZE)" \
    "$(awk -v x="${A_TOT:-0}" -v g="$TP_SIZE" 'BEGIN{printf (x>0)?"%.1f":"NA", x/g}')" \
    "$(awk -v x="${B_TOT:-0}" -v g="$TP_SIZE" 'BEGIN{printf (x>0)?"%.1f":"NA", x/g}')"
echo "==========================================================="
echo "logs: $A_LOG | $B_LOG | $SERVER_LOG"
