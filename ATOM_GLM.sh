#!/usr/bin/env bash
# ATOM (rocm/atom-dev) benchmark + profile driver for GLM-5-FP8.
# Concise ATOM counterpart of GLM.sh. Run inside container jacchang_GLM5_ATOM.
#
# Usage:
#   ./ATOM_GLM.sh                                  # benchmark only
#   ./ATOM_GLM.sh --prof                           # profile (cuda-graph + no-cuda-graph)
#   ./ATOM_GLM.sh --model /data/huggingface/hub/zai-org/GLM-5-FP8 --tp 8
#   ./ATOM_GLM.sh --tag MyRun --docker rocm/atom-dev:sglang-v0.5.12-nightly_20260630
#   ./ATOM_GLM.sh --port 8888                      # HTTP server-port (default 8888)
set -euo pipefail
set -x
ulimit -n 65535

PROF="false"; MODEL="/data/huggingface/hub/zai-org/GLM-5-FP8"; TP=8
TAG=""; DOCKER="untagged-docker"; SERVER_PORT="${PORT:-8888}"; ENGINE_PORT=5678
while [[ $# -gt 0 ]]; do case $1 in
  --prof)   PROF="true"; shift;;
  --model)  MODEL="$2"; shift 2;;
  --tp)     TP="$2"; shift 2;;
  --tag)    TAG="-$2"; shift 2;;
  --docker) DOCKER="$2"; shift 2;;
  --port)   SERVER_PORT="$2"; shift 2;;
  *) echo "Unknown option: $1"; exit 1;;
esac; done

_M="${MODEL%/}"; MODEL_NAME="$(basename "$(dirname "$_M")")_$(basename "$_M")"
DOCKER_FN=$(echo "$DOCKER" | sed 's/\//_/g; s/:/-/g')
HOST="localhost"; DATASET="random"; RANGE_RATIO=0.8
VENDOR_TAG="${VENDOR_TAG:-AMD}"   # inserted into GLM.sh-style trace filenames

export SAFETENSORS_FAST_GPU=1
# NOTE: do NOT set PYTORCH_ALLOC_CONF=expandable_segments:True here — it breaks
# AITER's IPC-based custom all-reduce on multi-GPU TP (hipIpcGetMemHandle fails
# with "invalid argument"), killing server init. Long-context OOM is instead
# handled by --gpu-memory-utilization 0.85 below (matches SGLang's headroom).

# GSM8K accuracy uses lm_eval against ATOM's OpenAI-compatible /v1/completions
# (the sglang native bench_sglang.py is incompatible with ATOM's server).
GSM8K_FEWSHOT="${GSM8K_FEWSHOT:-5}"
GSM8K_CONCURRENT="${GSM8K_CONCURRENT:-65}"

# Benchmark sweep. Override via env for smoke tests, e.g.
#   IN_OUT="512:64" CONC="4" ./ATOM_GLM.sh --prof
IFS=' ' read -ra in_out <<< "${IN_OUT:-1024:1024 8192:1024 70000:300}"
IFS=' ' read -ra concurrencies <<< "${CONC:-4 8 16 32 64}"
MULT=5
SPECIAL="-bench"
if [ "$PROF" == "true" ]; then
    SPECIAL="-prof"; MULT=2
    [ -z "${CONC:-}" ] && concurrencies=(4)   # profile: conc 4 only
    # ATOM has no --profile-num-steps: it profiles the WHOLE run, so a long
    # output-len => a giant trace (1024 decode steps ~= 770 MB/rank). Profile
    # i8192/i1024 with a small output-len (default 16) to keep traces small,
    # akin to SGLang's --profile-num-steps. Override with IN_OUT / PROF_OUT.
    [ -z "${IN_OUT:-}" ] && in_out=("8192:${PROF_OUT:-16}" "1024:${PROF_OUT:-16}")
fi

LEAF="${SPECIAL}${TAG}"; LEAF="${LEAF#-}"
BASE_LOG_DIR="$HOME/SGLang-benchmarks/results/${MODEL_NAME}/${DOCKER_FN}/${LEAF}"

log_cmd() { local f=$1; shift; echo ">>> $*" | tee -a "$f"; "$@" 2>&1 | tee -a "$f"; }

# Resume support: a config counts as done if its bench log is recorded in
# Finish.log (bench mode) or its profiler dir exists (profile mode).
is_done() { # args: logfile prof_dir
    if [ "$PROF" == "true" ]; then [ -d "$2" ]; else grep -qxF "$1" "$FINISH_LOG" 2>/dev/null; fi
}
# True if this mode still has any pending work (so we bother starting the server).
mode_has_work() {
    [ "$PROF" != "true" ] && ! grep -qxF "$LOG_DIR/Accuracy_GSM8K.log" "$FINISH_LOG" 2>/dev/null && return 0
    local io i o c np
    for io in "${in_out[@]}"; do IFS=":" read -r i o <<< "$io"
        for c in "${concurrencies[@]}"; do np=$((c * MULT))
            is_done "$LOG_DIR/bench_in${i}_out${o}_conc${c}.log" \
                    "$LOG_DIR/prof_in${i}_out${o}_conc${c}_p${np}" || return 0
        done
    done
    return 1
}

# ATOM's torch profiler writes one trace per TP rank into $LOG_DIR/rank_<N>/.
# Snapshot those files (list_traces) so we can attribute each bench run's new
# traces and move them into a per-config subdir, renamed GLM.sh-style:
#   in<in>_out<out>_conc<c>_p<np>-<VENDOR>-TP-<rank>[-NoGraph].trace.json.gz
# (ATOM can't split prefill/decode, so there is no -EXTEND/-DECODE suffix.)
list_traces() { find "$LOG_DIR" -path '*/rank_*/*.json.gz' 2>/dev/null | sort; }
collect_traces() { # args: dst in out c np before
    local dst=$1 in=$2 out=$3 c=$4 np=$5 before=$6
    local new; new=$(comm -13 <(printf '%s\n' "$before") <(printf '%s\n' "$(list_traces)"))
    [ -z "$new" ] && { echo "No new trace found"; return 0; }
    mkdir -p "$dst"; printf '%s\n' "$new" | while read -r f; do
        [ -f "$f" ] || continue
        local rank; rank=$(basename "$(dirname "$f")"); rank=${rank#rank_}
        mv "$f" "$dst/in${in}_out${out}_conc${c}_p${np}-${VENDOR_TAG}-TP-${rank}${NOGRAPH_SUFFIX}.trace.json.gz"
    done
}

# Wait until the TP GPUs (0..TP-1) have released memory before (re)starting a
# server. A just-killed server can hold GPU memory for a while; starting too
# soon causes a HIP OOM during engine init. Falls back to a fixed sleep if
# rocm-smi is unavailable.
wait_gpu_free() {
    command -v rocm-smi >/dev/null 2>&1 || { sleep 15; return 0; }
    local tries=0
    while [ $tries -lt 60 ]; do   # up to ~300s
        local busy
        busy=$(rocm-smi --showmeminfo vram 2>/dev/null | awk -v n="$TP" '
            /VRAM Total Used Memory/ { g=$1; sub(/.*\[/,"",g); sub(/\].*/,"",g);
                if (g+0 < n && $NF+0 > m) m=$NF+0 } END { print m+0 }')
        [ "${busy:-0}" -lt 5000000000 ] && return 0   # < 5 GB on all TP GPUs
        echo ">>> Waiting for GPU 0-$((TP-1)) to free (max used ${busy} B)..."
        sleep 5; tries=$((tries + 1))
    done
    echo "[warn] GPUs still busy after wait; proceeding anyway."
}

start_server() {
    local logfile="$LOG_DIR/server_${MODEL_NAME}.log"
    wait_gpu_free
    if (exec 3<>"/dev/tcp/${HOST}/${SERVER_PORT}") 2>/dev/null; then
        exec 3>&- 3<&-
        echo "!!! ${HOST}:${SERVER_PORT} in use. pkill atom/python or use --port." | tee -a "$logfile"; exit 1
    fi
    local cmd=(python -m atom.entrypoints.openai_server
        --model "$MODEL" -tp "$TP" --kv_cache_dtype fp8 --trust-remote-code
        --host "$HOST" --port "$ENGINE_PORT" --server-port "$SERVER_PORT")
    # Default 0.85 mirrors SGLang's --mem-fraction-static 0.85 (apples-to-apples)
    # and leaves ~25 GB/GPU headroom so long-context (70000-token) NSA indexer
    # allocations don't HIP-OOM. Override with GPU_MEM_UTIL=<x>.
    cmd+=(--gpu-memory-utilization "${GPU_MEM_UTIL:-0.85}")
    if [ "$PROF" == "true" ]; then
        cmd+=(--torch-profiler-dir "$LOG_DIR" --mark-trace)
        [ "${EAGER:-false}" == "true" ] && cmd+=(--enforce-eager)
    fi
    echo ">>> ${cmd[*]}" | tee "$logfile"
    "${cmd[@]}" 2>&1 | tee -a "$logfile" &
    echo ">>> Waiting for server ($logfile)..."
    until [ "$(curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${SERVER_PORT}/health" 2>/dev/null)" = "200" ]; do
        pgrep -f "atom.entrypoints.openai_server" >/dev/null || { echo "!!! server died; see $logfile"; exit 1; }
        sleep 5
    done
}

bench_one() {
    local in=$1 out=$2 c=$3 np=$((c * MULT))
    local logfile="$LOG_DIR/bench_in${in}_out${out}_conc${c}.log"
    local prof_dir="$LOG_DIR/prof_in${in}_out${out}_conc${c}_p${np}"
    if is_done "$logfile" "$prof_dir"; then echo "Skip (done): $logfile"; return 0; fi
    # No --save-result: metrics are captured in the .log; we keep logs only.
    local cmd=(python -m atom.benchmarks.benchmark_serving --backend vllm
        --base-url "http://${HOST}:${SERVER_PORT}" --model "$MODEL"
        --dataset-name "$DATASET" --random-input-len "$in" --random-output-len "$out"
        --random-range-ratio "$RANGE_RATIO" --num-prompts "$np" --max-concurrency "$c"
        --request-rate inf --ignore-eos
        --percentile-metrics "ttft,tpot,itl,e2el")
    [ "$PROF" == "true" ] && cmd+=(--profile)
    local before=""
    [ "$PROF" == "true" ] && before=$(list_traces)
    local ok=1
    log_cmd "$logfile" "${cmd[@]}" || { ok=0; echo "[warn] bench exited non-zero; continuing."; }
    if [ "$PROF" == "true" ]; then
        # Profiler teardown can crash after traces are flushed, so keep whatever
        # traces exist; prof_dir existence is the resume marker.
        collect_traces "$prof_dir" "$in" "$out" "$c" "$np" "$before"
    elif [ "$ok" == "1" ]; then
        echo "$logfile" >> "$FINISH_LOG"   # mark done only on success
    fi
}

run_all() {
    # warmup
    log_cmd "$LOG_DIR/warmup.log" python -m atom.benchmarks.benchmark_serving --backend vllm \
        --base-url "http://${HOST}:${SERVER_PORT}" --model "$MODEL" --dataset-name "$DATASET" \
        --random-input-len 2048 --random-output-len 256 --random-range-ratio "$RANGE_RATIO" \
        --num-prompts 8 --max-concurrency 4 --request-rate inf --ignore-eos || true
    # accuracy via lm_eval (skip in profile mode; resume-aware; record on success)
    if [ "$PROF" != "true" ]; then
        local acc="$LOG_DIR/Accuracy_GSM8K.log"
        if grep -qxF "$acc" "$FINISH_LOG" 2>/dev/null; then
            echo "Skip (done): $acc"
        elif log_cmd "$acc" lm_eval --model local-completions \
                --model_args "model=${MODEL},base_url=http://${HOST}:${SERVER_PORT}/v1/completions,num_concurrent=${GSM8K_CONCURRENT},max_retries=3,tokenized_requests=False,trust_remote_code=True" \
                --tasks gsm8k --num_fewshot "${GSM8K_FEWSHOT}"; then
            echo "$acc" >> "$FINISH_LOG"
        else
            echo "[warn] GSM8K failed; not recorded (will retry next run)."
        fi
    fi
    for io in "${in_out[@]}"; do IFS=":" read -r i o <<< "$io"
        for c in "${concurrencies[@]}"; do bench_one "$i" "$o" "$c"; done
    done
}

# Profile mode runs twice: cuda-graph then no-cuda-graph (enforce-eager).
if [ "$PROF" == "true" ]; then MODES=("graph" "no-cuda-graph"); else MODES=("default"); fi
for MODE in "${MODES[@]}"; do
    if [ "$MODE" == "no-cuda-graph" ]; then
        LOG_DIR="$BASE_LOG_DIR/no-cuda-graph"; EAGER="true"; NOGRAPH_SUFFIX="-NoGraph"
    else
        LOG_DIR="$BASE_LOG_DIR"; EAGER="false"; NOGRAPH_SUFFIX=""
    fi
    mkdir -p "$LOG_DIR"; FINISH_LOG="$LOG_DIR/Finish.log"; touch "$FINISH_LOG"
    if ! mode_has_work; then echo ">>> [$MODE] all configs done, skipping (see $FINISH_LOG)"; continue; fi
    ( echo ">>> [$MODE] start"; start_server; run_all ) \
        || echo "[warn] mode '$MODE' aborted; continuing."
    # Real profiles were moved into prof_*/ by collect_traces; the leftover
    # rank_*/ dirs only hold startup capture_graph traces (not useful) — drop them.
    [ "$PROF" == "true" ] && find "$LOG_DIR" -maxdepth 1 -type d -name 'rank_*' -exec rm -rf {} + 2>/dev/null || true
    pkill -9 python || true; pkill -9 atom || true; sleep 10
done
echo ">>> Done. Logs: $BASE_LOG_DIR"
