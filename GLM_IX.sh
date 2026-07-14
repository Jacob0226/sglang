#!/usr/bin/env bash
# GLM_IX.sh — InferenceX-faithful GLM-5 FP8 benchmark replica
# ============================================================================
# This script reproduces the *exact* server + benchmark settings used by
# SemiAnalysisAI/InferenceX for the GLM-5-FP8 single-node "fixed_seq_len"
# leaderboard entries, so your local numbers line up with the board.
#
# It mirrors, byte-for-byte where it matters, these InferenceX recipes:
#   benchmarks/single_node/fixed_seq_len/glm5_fp8_mi355x.sh       (ROCm, non-MTP)
#   benchmarks/single_node/fixed_seq_len/glm5_fp8_mi355x_mtp.sh   (ROCm, MTP)
#   benchmarks/single_node/fixed_seq_len/glm5_fp8_b200.sh         (NVIDIA, non-MTP)
#   benchmarks/single_node/fixed_seq_len/glm5_fp8_b200_mtp.sh     (NVIDIA, MTP)
# plus the sweep parameters from .github/configs/{amd,nvidia}-master.yaml
# and benchmark_lib.sh::run_benchmark_serving.
#
# WHY this differs from your GLM.sh (the cause of the score gap):
#   1. InferenceX relaunches the server FOR EACH concurrency, pinning
#      --cuda-graph-max-bs (and --max-running-requests / --context-length)
#      to that concurrency. GLM.sh starts ONE server and sweeps all conc.
#   2. The MI355X board is TP=4 + MTP. GLM.sh defaults FP8 to TP=8, no MTP.
#      (Per-GPU throughput on the board is total/4, so TP matters a lot.)
#   3. InferenceX does NOT pass --quantization on ROCm (model self-declares);
#      it DOES pass --quantization fp8 --fp8-gemm-backend cutlass on B200.
#   4. num-prompts = conc*10 (GLM.sh used conc*5), warmups = conc*2.
#   5. MTP runs apply the chat template; non-MTP runs do not.
#
# Usage:
#   ./GLM_IX.sh                       # auto: MI355X->TP4, B200->TP8 ; non-MTP
#   ./GLM_IX.sh --mtp                 # MTP (this is what the MI355X board shows)
#   ./GLM_IX.sh --model /data/huggingface/hub/zai-org/GLM-5-FP8
#   ./GLM_IX.sh --mtp --tp 4 --conc "4 8 16 32 64 128"
#   ./GLM_IX.sh --docker rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260503
# ----------------------------------------------------------------------------
set -euo pipefail
set -x
ulimit -n 65535
sh -c 'echo 0 > /proc/sys/kernel/numa_balancing' || true

MTP_ENABLED="false"
MTP_TAG=""
USER_TAG=""
TP_SIZE="auto"
MODEL_PATH="/data/huggingface/hub/zai-org/GLM-5-FP8"
DOCKER="untagged-docker"
# Concurrency sweep. InferenceX ranges (fixed_seq_len):
#   MI355X MTP : TP4 conc 4-128 (main), TP8 conc 4-8
#   B200  FP8  : TP8 conc 4-256 (both MTP and non-MTP)
# Override with --conc "4 8 ...". "auto" picks a per-platform default below.
CONC_LIST="auto"

while [[ $# -gt 0 ]]; do
  case $1 in
    --mtp)   MTP_ENABLED="true"; MTP_TAG="-MTP"; shift 1 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    --tp)    TP_SIZE="$2"; shift 2 ;;
    --conc)  CONC_LIST="$2"; shift 2 ;;
    --tag)   USER_TAG="-$2"; shift 2 ;;
    --port)  PORT="$2"; shift 2 ;;
    --docker) DOCKER="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done
MODEL_NAME=$(basename "${MODEL_PATH%/}")

is_rocm_gpu_env() { [ -e /dev/kfd ] || command -v rocm-smi >/dev/null 2>&1; }

# ===================== Platform-specific defaults (match InferenceX) ==========
if is_rocm_gpu_env; then
    PLATFORM="mi355x"
    [ "$TP_SIZE" = "auto" ] && TP_SIZE=4          # board = TP4
    [ "$CONC_LIST" = "auto" ] && CONC_LIST="4 8 16 32 64 128"
else
    PLATFORM="b200"
    [ "$TP_SIZE" = "auto" ] && TP_SIZE=8          # IX glm5-fp8-b200-sglang = TP8
    [ "$CONC_LIST" = "auto" ] && CONC_LIST="4 8 16 32 64 128 256"
fi
read -r -a CONCURRENCIES <<< "$CONC_LIST"

# ===================== Sweep parameters (match InferenceX) ====================
HOST="0.0.0.0"
# 8552 was already in use on this host; default to a high, unlikely-used port.
# Override anytime with --port <n>.
PORT="${PORT:-28552}"
in_out_tokens=("1024:1024" "8192:1024")   # InferenceX fixed_seq_len scenarios
RANDOM_RANGE_RATIO="0.8"                   # benchmark-tmpl.yml: RANDOM_RANGE_RATIO=0.8
MEM_FRACTION_STATIC="0.85"
NUM_PROMPT_MULTIPLIER=10                   # run_benchmark_serving: num-prompts=conc*10
WARMUP_MULTIPLIER=2                        # run_benchmark_serving: --num-warmups=2*conc

DOCKER_FILENAME=$(echo "$DOCKER" | sed 's/\//_/g; s/:/-/g')
LOG_DIR="$HOME/SGLang-benchmarks/results/$DOCKER_FILENAME/IX_${PLATFORM}_${MODEL_NAME}${MTP_TAG}-tp${TP_SIZE}${USER_TAG}"
FINISH_LOG="$LOG_DIR/Finish.log"
mkdir -p "$LOG_DIR"
touch "$FINISH_LOG"

log_command() {
    local logfile=$1; shift
    echo ">>>Executing command:" | tee -a "$logfile"
    echo "$*" | tee -a "$logfile"
    echo "---" | tee -a "$logfile"
    "$@" 2>&1 | tee -a "$logfile"
}

# ===================== Server launch (per concurrency) =======================
start_server() {
    local conc=$1 isl=$2 osl=$3
    local logfile="$4"
    echo ">>> Starting SGLang server (conc=$conc isl=$isl osl=$osl)" | tee "$logfile"

    local cm
    if [ "$PLATFORM" = "mi355x" ]; then
        # ---- InferenceX glm5_fp8_mi355x[_mtp].sh (ROCm) ----
        # NOTE: NO --quantization flag on ROCm — GLM-5-FP8 weights self-declare.
        export SGLANG_ROCM_FUSED_DECODE_MLA=0
        export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
        export SAFETENSORS_FAST_GPU=1
        cmd=(
            python3 -m sglang.launch_server
                --model-path "$MODEL_PATH"
                --host "$HOST" --port "$PORT"
                --tensor-parallel-size "$TP_SIZE"
                --trust-remote-code
                --tool-call-parser glm47
                --reasoning-parser glm45
                --mem-fraction-static "$MEM_FRACTION_STATIC"
                --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}'
                --nsa-prefill-backend tilelang
                --nsa-decode-backend tilelang
                --kv-cache-dtype fp8_e4m3
                --tokenizer-worker-num $((TP_SIZE * 2))
                --cuda-graph-max-bs "$conc"
                --disable-radix-cache
        )
        if [ "$MTP_ENABLED" = "true" ]; then
            export SGLANG_ENABLE_SPEC_V2=1
            cmd+=(
                --context-length $((isl + osl + 32))
                --speculative-algorithm EAGLE
                --speculative-num-steps 3
                --speculative-eagle-topk 1
                --speculative-num-draft-tokens 4
            )
        else
            cmd+=(--max-running-requests "$conc")
        fi
    else
        # ---- InferenceX glm5_fp8_b200[_mtp].sh (NVIDIA) ----
        export SGL_ENABLE_JIT_DEEPGEMM=1
        # IX pins these to keep tokenizer/transformers behaviour reproducible.
        pip install --no-deps "transformers==5.2.0" "huggingface-hub==1.4.1" 2>/dev/null || true
        export PYTHONNOUSERSITE=1
        cmd=(
            python3 -m sglang.launch_server
                --model-path "$MODEL_PATH"
                --host "$HOST" --port "$PORT"
                --trust-remote-code
                --tensor-parallel-size "$TP_SIZE"
                --data-parallel-size 1 --expert-parallel-size 1
                --tool-call-parser glm47
                --reasoning-parser glm45
                --kv-cache-dtype fp8_e4m3 --quantization fp8
                --fp8-gemm-backend cutlass
                --attention-backend nsa
                --nsa-decode-backend trtllm --nsa-prefill-backend trtllm
                --moe-runner-backend flashinfer_trtllm
                --cuda-graph-max-bs "$conc" --max-running-requests "$conc"
                --mem-fraction-static "$MEM_FRACTION_STATIC"
                --chunked-prefill-size 32768 --max-prefill-tokens 32768
                --enable-flashinfer-allreduce-fusion --disable-radix-cache
                --stream-interval 30
                --model-loader-extra-config '{"enable_multithread_load": true}'
        )
        if [ "$MTP_ENABLED" = "true" ]; then
            export SGLANG_ENABLE_SPEC_V2=1
            cmd+=(
                --speculative-algorithm EAGLE
                --speculative-num-steps 3
                --speculative-eagle-topk 1
                --speculative-num-draft-tokens 4
            )
        fi
    fi

    echo ">>> Executing command:" | tee -a "$logfile"
    echo "${cmd[*]}" | tee -a "$logfile"
    echo "---" | tee -a "$logfile"
    "${cmd[@]}" 2>&1 | tee -a "$logfile" &
    SERVER_PID=$!

    echo ">>> Waiting for server health..." | tee -a "$logfile"
    until [ "$(curl -s -o /dev/null -w "%{http_code}" "http://${HOST}:$PORT/health")" -eq 200 ]; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Server died before becoming healthy. See $logfile" >&2
            return 1
        fi
        sleep 5
    done
}

# CRITICAL: never use `pkill -f <pattern>` here. `-f` matches the WHOLE command
# line, and THIS script's own argv contains "sglang" (via the --docker image
# name, e.g. lmsysorg/sglang-rocm:...). `pkill -f sglang` therefore matched and
# terminated the script itself right after the first conc — the real reason the
# sweep stopped after conc4. We kill ONLY:
#   - the server PID we launched and its direct children (TP workers), and
#   - processes whose NAME is exactly "python3" (-x), which can never match the
#     bash script (whose process name is "bash"). In a container's isolated PID
#     namespace this touches only our own python processes.
stop_server() {
    pkill -9 -P "${SERVER_PID:-0}" 2>/dev/null || true
    kill -9 "${SERVER_PID:-0}" 2>/dev/null || true
    pkill -9 -x python3 2>/dev/null || true
    sleep 10
}

# ===================== Benchmark (per concurrency) ===========================
# Mirrors InferenceX benchmark_lib.sh::run_benchmark_serving:
#   random dataset, range-ratio 0.8, num-prompts=conc*10, warmups=conc*2,
#   ignore_eos (sglang.bench_serving default), chat template only for MTP.
# NOTE: InferenceX uses its own vLLM-style benchmark_serving.py (--backend vllm).
#       We use sglang.bench_serving (--backend sglang) since that ships in the
#       SGLang container. Both stream with ignore_eos and identical ISL/OSL, so
#       throughput is comparable; this client is the only residual difference.
run_one() {
    local conc=$1 isl=$2 osl=$3
    local logfile="${LOG_DIR}/bench_in${isl}_out${osl}_conc${conc}.log"
    if grep -q "$logfile" "$FINISH_LOG"; then
        echo "Found $logfile in $FINISH_LOG. Skipping."
        return 0
    fi
    local cmd=(
        python3 -m sglang.bench_serving
        --backend sglang
        --host "$HOST" --port "$PORT"
        --model "$MODEL_PATH"
        --dataset-name random
        --random-input "$isl"
        --random-output "$osl"
        --random-range-ratio "$RANDOM_RANGE_RATIO"
        --num-prompts $((conc * NUM_PROMPT_MULTIPLIER))
        --max-concurrency "$conc"
        --warmup-requests $((conc * WARMUP_MULTIPLIER))
        --output-file /dev/null
    )
    if [ "$MTP_ENABLED" = "true" ]; then
        cmd+=(--apply-chat-template)
    fi
    log_command "$logfile" "${cmd[@]}"
    echo "$logfile" >> "$FINISH_LOG"
}

# ===================== Disk preflight =======================================
# A full disk makes `tee` fail mid-run; with `set -e` that silently aborts the
# whole sweep (this is exactly what killed the conc4->conc8 transition before).
# Fail loudly up front instead.
avail_kb=$(df -P "$LOG_DIR" | awk 'NR==2{print $4}')
if [ "${avail_kb:-0}" -lt 2097152 ]; then   # < 2 GiB free
    echo "ERROR: only $((avail_kb/1024)) MiB free on $(df -P "$LOG_DIR" | awk 'NR==2{print $6}'). Free up disk before benchmarking — a full disk corrupts logs and skews/aborts runs." >&2
    exit 1
fi

# ===================== Main sweep ===========================================
# Best-effort: free our port if a previous crashed run left a server bound to
# it. fuser targets only the process holding the socket (never this script).
if command -v fuser >/dev/null 2>&1; then
    fuser -k "${PORT}/tcp" 2>/dev/null || true
fi
sleep 2
echo ">>> Platform=$PLATFORM TP=$TP_SIZE MTP=$MTP_ENABLED PORT=$PORT conc=(${CONCURRENCIES[*]})"
for io_pair in "${in_out_tokens[@]}"; do
    IFS=":" read -r isl osl <<< "$io_pair"
    for conc in "${CONCURRENCIES[@]}"; do
        bench_log="${LOG_DIR}/bench_in${isl}_out${osl}_conc${conc}.log"
        if grep -q "$bench_log" "$FINISH_LOG"; then
            echo "Found $bench_log in $FINISH_LOG. Skipping (server not started)."
            continue
        fi
        server_log="${LOG_DIR}/server_in${isl}_out${osl}_conc${conc}.log"
        # Guard each iteration so one bad conc (server death, transient error)
        # does not abort the whole sweep. Calling in an `if` also suppresses
        # `set -e` inside the function bodies.
        if ! start_server "$conc" "$isl" "$osl" "$server_log"; then
            echo ">>> conc=$conc isl=$isl osl=$osl: server failed to start, skipping." >&2
            stop_server
            continue
        fi
        run_one "$conc" "$isl" "$osl" || echo ">>> conc=$conc bench failed, continuing." >&2
        stop_server
    done
done

echo ">>> All InferenceX-matched runs complete. Logs: ${LOG_DIR}"
