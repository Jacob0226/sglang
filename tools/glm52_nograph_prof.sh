#!/usr/bin/env bash
# Minimal standalone no-cuda-graph profile for GLM-5.2-MXFP4 TP4 (structure trace).
# Avoids GLM.sh's full sweep (which crashes on 8192 / no-graph teardown). Profiles
# ONLY 1024:1024 conc4 with --disable-cuda-graph, so you get a clean
# *-TP-*-DECODE.trace.json.gz for the layer-structure side of analyze_trace.
#
# Usage (inside container):
#   HIP_VISIBLE_DEVICES=0,1,2,3 bash tools/glm52_nograph_prof.sh
# Optional env: PORT, MEM_FRAC, TAG, NO_ALLREDUCE_FUSION=1 (drop the fusion flag
#   if the no-graph server crashes at startup).
set -uo pipefail
MODEL=/data/huggingface/hub/amd/GLM-5.2-MXFP4
PORT="${PORT:-8234}"
TAG="${TAG:-elemfix_nograph}"
OUT="/home/jacchang/SGLang-benchmarks/tmp/opt2_bench/${TAG}"
mkdir -p "$OUT"
export SAFETENSORS_FAST_GPU=1 SGLANG_ROCM_FUSED_DECODE_MLA=0 ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3}"
export SGLANG_TORCH_PROFILER_DIR="$OUT"

FUSION_FLAG=(--enable-aiter-allreduce-fusion)
[ "${NO_ALLREDUCE_FUSION:-0}" = "1" ] && FUSION_FLAG=()

pkill -9 -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
sleep 3
echo ">>> launching --disable-cuda-graph server on GPUs $HIP_VISIBLE_DEVICES"
python3 -m sglang.launch_server --model "$MODEL" --tp 4 --host localhost --port "$PORT" \
  --trust-remote-code --tool-call-parser glm47 --reasoning-parser glm45 --watchdog-timeout 1200 \
  --mem-fraction-static "${MEM_FRAC:-0.85}" --kv-cache-dtype fp8_e4m3 --disable-radix-cache \
  --nsa-prefill-backend tilelang --nsa-decode-backend tilelang --tokenizer-worker-num 8 \
  "${FUSION_FLAG[@]}" --disable-cuda-graph \
  > "$OUT/server.log" 2>&1 &
SVR=$!

for i in $(seq 1 150); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$PORT/health 2>/dev/null)" = "200" ] && break
  if ! kill -0 $SVR 2>/dev/null; then echo "!!! server died (see $OUT/server.log)"; tail -30 "$OUT/server.log"; exit 1; fi
  sleep 5
done
echo ">>> server ready (~$((i*5))s); warmup"
python3 -m sglang.bench_serving --host localhost --port "$PORT" --model "$MODEL" --dataset-name random \
  --random-input 1024 --random-output 128 --random-range-ratio 0.8 --max-concurrency 4 --num-prompt 4 \
  --output-file /dev/null > "$OUT/warmup.log" 2>&1 || true

# NOTE: keep --random-output SHORT. The eager (no-cuda-graph) DSA/tilelang decode
# deterministically hits a GPU memory-access fault once KV context per request
# grows to ~1900 tokens (decode batch #token ~7616 at bs=4). Short output keeps
# context ~1088 so the fault never triggers, while still giving >5 decode steps
# for the profiler. (ATOM's reference trace also uses out=16 for this reason.)
OSL="${OSL:-32}"
echo ">>> profiling 1024:${OSL} conc4 (no cuda graph, short output to dodge the eager DSA fault)"
python3 -m sglang.bench_serving --host localhost --port "$PORT" --model "$MODEL" --dataset-name random \
  --random-input 1024 --random-output "$OSL" --random-range-ratio 0.8 --max-concurrency 4 --num-prompt 4 \
  --profile --profile-num-steps 5 --profile-by-stage --output-file /dev/null 2>&1 | tee "$OUT/prof.log" || true

pkill -9 -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
sleep 3
echo ">>> DONE. DECODE trace(s):"
find "$OUT" -name "*-DECODE.trace.json.gz" 2>/dev/null
