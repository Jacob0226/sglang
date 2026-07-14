#!/usr/bin/env bash
# Milestone-1 validation: eager (no cuda graph) correctness of the decode
# k-only skip-indexer fast path on jacob/dense-decode-konly.
# GSM8K responses stay < index_topk (2048) so decode exercises the k-only path;
# if the path were wrong (like the spike's arange identity) accuracy would crater.
set -uo pipefail

PORT=8234
MODEL=/data/huggingface/hub/amd/GLM-5.2-MXFP4
LOG=~/SGLang-benchmarks/tmp/konly_eager_server.log

export SAFETENSORS_FAST_GPU=1
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export SGLANG_OPT_USE_TOPK_V2=0
export SGLANG_KONLY_DEBUG=1
export SGLANG_TORCH_PROFILER_DIR=/tmp

echo ">>> launching EAGER server (--disable-cuda-graph, tilelang decode)"
python3 -m sglang.launch_server \
    --model "$MODEL" --tp 4 --host localhost --port $PORT \
    --trust-remote-code --tool-call-parser glm47 --reasoning-parser glm45 \
    --watchdog-timeout 1200 --mem-fraction-static 0.85 --kv-cache-dtype fp8_e4m3 \
    --disable-radix-cache \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 32}' \
    --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
    --tokenizer-worker-num 8 --enable-aiter-allreduce-fusion \
    --chunked-prefill-size 16384 \
    --disable-cuda-graph > "$LOG" 2>&1 &
SERVER_PID=$!

echo ">>> waiting for health..."
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/health 2>/dev/null)" = "200" ]; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "!!! server died. tail:"; tail -30 "$LOG"; exit 1
    fi
    sleep 5
done
echo ">>> server ready"

echo ">>> tiny generation to trigger decode k-only marker"
curl -s http://localhost:$PORT/generate -H 'Content-Type: application/json' \
    -d '{"text":"Q: What is 12+7? A:","sampling_params":{"max_new_tokens":8,"temperature":0}}' >/dev/null
sleep 2
echo ">>> [KONLY] markers seen in server log:"
grep -c "KONLY] decode k-only fired" "$LOG" || true
grep -m2 "KONLY] decode k-only fired" "$LOG" || true

echo ">>> GSM8K (400 questions, eager)"
python3 /sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py \
    --port $PORT --num-questions 400 --parallel 128 2>&1 | tee ~/SGLang-benchmarks/tmp/konly_eager_gsm8k.log

echo ">>> stopping server"
kill $SERVER_PID 2>/dev/null
echo "DONE"
