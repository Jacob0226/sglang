#!/usr/bin/env bash
# GLM-5.2-MXFP4 TP4 MI355X decode benchmark for one NSA decode backend.
# Usage (inside container): glm52_decode_bench.sh <tilelang|aiter> <tag> [gsm8k_q]
set -uo pipefail
BACKEND="${1:?backend}"
TAG="${2:-$BACKEND}"
GSM8K_Q="${3:-200}"
MODEL=/data/huggingface/hub/amd/GLM-5.2-MXFP4
PORT="${PORT:-8399}"
OUT=/home/jacchang/SGLang-benchmarks/tmp/opt2_bench
mkdir -p "$OUT"
LOG="$OUT/server_${TAG}.log"

export SAFETENSORS_FAST_GPU=1
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-4,5,6,7}"

pkill -9 -f sglang.launch_server 2>/dev/null || true
sleep 3

echo ">>> starting server backend=$BACKEND port=$PORT"
python3 -m sglang.launch_server \
  --model "$MODEL" --tp 4 --host localhost --port "$PORT" --trust-remote-code \
  --tool-call-parser glm47 --reasoning-parser glm45 --watchdog-timeout 1200 \
  --mem-fraction-static "${MEM_FRAC:-0.85}" --kv-cache-dtype fp8_e4m3 --disable-radix-cache \
  --nsa-prefill-backend tilelang --nsa-decode-backend "$BACKEND" \
  --tokenizer-worker-num 8 --cuda-graph-max-bs "${GRAPH_BS:-256}" ${EXTRA_FLAGS:-} \
  > "$LOG" 2>&1 &
SVR=$!

echo ">>> waiting for server..."
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/health" 2>/dev/null || true)
  [ "$code" = "200" ] && break
  if ! kill -0 $SVR 2>/dev/null; then echo "!!! server died"; tail -30 "$LOG"; exit 1; fi
  sleep 5
done
echo ">>> server ready after ~$((i*5))s"

echo ">>> warmup"
python3 -m sglang.bench_serving --host localhost --port "$PORT" --model "$MODEL" \
  --dataset-name random --random-input 2048 --random-output 256 \
  --random-range-ratio 0.8 --max-concurrency 4 --num-prompt 8 --output-file /dev/null \
  > "$OUT/warmup_${TAG}.log" 2>&1 || true

# CONFIGS: space-separated list of "isl:osl:conc:nprompt"
CONFIGS="${CONFIGS:-1024:512:4:20}"
for cfg in $CONFIGS; do
  IFS=":" read -r ISL OSL CONC NP <<< "$cfg"
  echo ">>> decode bench isl${ISL}/osl${OSL} conc${CONC} np${NP}"
  python3 -m sglang.bench_serving --host localhost --port "$PORT" --model "$MODEL" \
    --dataset-name random --random-input "$ISL" --random-output "$OSL" \
    --random-range-ratio 0.8 --max-concurrency "$CONC" --num-prompt "$NP" --output-file /dev/null \
    2>&1 | tee "$OUT/bench_${TAG}_i${ISL}_o${OSL}_c${CONC}.log"
done

if [ "$GSM8K_Q" -gt 0 ]; then
  echo ">>> GSM8K ($GSM8K_Q q)"
  python3 /sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py --port "$PORT" \
    --num-questions "$GSM8K_Q" --parallel 32 2>&1 | tee "$OUT/gsm8k_${TAG}.log" || true
fi

pkill -9 -f sglang.launch_server 2>/dev/null || true
sleep 5
echo ">>> DONE $TAG"
