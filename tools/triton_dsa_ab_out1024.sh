#!/usr/bin/env bash
# A/B (REAL decode): triton vs tilelang DSA decode backend, GLM-5.2-MXFP4 MI355X.
# out1024 (i1k/o1k) so decode dominates -> TPOT reflects the decode main_kernel,
# comparable to the user's reference table (~32ms at conc64). Prefill backend
# fixed to tilelang for both, so only the DECODE kernel differs.
set -uo pipefail

MODEL=/data/huggingface/hub/amd/GLM-5.2-MXFP4
DOCKER=rocm/sgl-dev:v0.5.14-rocm720-mi35x-20260708
export IN_OUT_OVERRIDE="1024:1024"
export CONC_OVERRIDE="4 8 16 32 64"

cd ~/SGLang-benchmarks

echo "############### RUN 1/2: triton DSA decode (out1024) ###############"
DSA_DECODE_BACKEND=triton ./GLM.sh --model "$MODEL" --tp 4 \
    --tag AB1k-TP4-triton --docker "$DOCKER"

echo "############### RUN 2/2: tilelang DSA decode (out1024) ###############"
DSA_DECODE_BACKEND=tilelang ./GLM.sh --model "$MODEL" --tp 4 \
    --tag AB1k-TP4-tilelang --docker "$DOCKER"

echo "############### out1024 A/B DONE ###############"
