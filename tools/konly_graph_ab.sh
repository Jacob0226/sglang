#!/usr/bin/env bash
# M2a: cuda-graph decode k-only ceiling. Same branch (dense-decode-konly),
# tilelang decode, CUDA graph ON, i1k/o1k. A/B by SGLANG_DSA_DECODE_DENSE_GRAPH.
#   baseline = full indexer every decode layer
#   dense    = skip indexer (k-only) on full layers when kv_len<=index_topk
# out1024 -> decode-dominated -> TPOT reflects the real saving.
set -uo pipefail

PORT=8234
MODEL=/data/huggingface/hub/amd/GLM-5.2-MXFP4
CONCS=(4 8 16 32 64)
OUTDIR=~/SGLang-benchmarks/tmp/konly_graph_ab
mkdir -p "$OUTDIR"

export SAFETENSORS_FAST_GPU=1
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_TORCH_PROFILER_DIR=/tmp

run_mode () {
    local mode=$1 flag=$2
    echo "############### MODE=$mode (SGLANG_DSA_DECODE_DENSE_GRAPH=$flag) ###############"
    pkill -9 -f sglang.launch_server 2>/dev/null; sleep 3
    SGLANG_DSA_DECODE_DENSE_GRAPH=$flag python3 -m sglang.launch_server \
        --model "$MODEL" --tp 4 --host localhost --port $PORT \
        --trust-remote-code --tool-call-parser glm47 --reasoning-parser glm45 \
        --watchdog-timeout 1200 --mem-fraction-static 0.85 --kv-cache-dtype fp8_e4m3 \
        --disable-radix-cache \
        --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 32}' \
        --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
        --tokenizer-worker-num 8 --enable-aiter-allreduce-fusion \
        --chunked-prefill-size 16384 > "$OUTDIR/server_$mode.log" 2>&1 &
    local pid=$!
    until [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/health 2>/dev/null)" = "200" ]; do
        if ! kill -0 $pid 2>/dev/null; then echo "!!! $mode server died"; tail -20 "$OUTDIR/server_$mode.log"; return 1; fi
        sleep 5
    done
    echo ">>> $mode ready; warmup"
    python3 -m sglang.bench_serving --host localhost --port $PORT --model "$MODEL" \
        --dataset-name random --random-input 1024 --random-output 64 \
        --random-range-ratio 0.8 --max-concurrency 4 --num-prompt 4 --output-file /dev/null >/dev/null 2>&1
    for c in "${CONCS[@]}"; do
        echo ">>> $mode conc$c"
        python3 -m sglang.bench_serving --host localhost --port $PORT --model "$MODEL" \
            --dataset-name random --random-input 1024 --random-output 1024 \
            --random-range-ratio 0.8 --max-concurrency $c --num-prompt $((c*3)) \
            --output-file /dev/null > "$OUTDIR/bench_${mode}_conc${c}.log" 2>&1
    done
    kill $pid 2>/dev/null; sleep 5
}

run_mode baseline 0
run_mode dense 1
echo "############### M2a A/B DONE ###############"
