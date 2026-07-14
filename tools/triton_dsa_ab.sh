#!/usr/bin/env bash
# A/B: triton vs tilelang DSA decode backend for GLM-5.2-MXFP4 on MI355X.
# Same build (pr-30575), same config; only variable = --dsa-decode-backend.
# Runs GSM8K (correctness, target >=0.92) + i1k/out16 conc sweep (perf).
set -uo pipefail

MODEL=/data/huggingface/hub/amd/GLM-5.2-MXFP4
DOCKER=rocm/sgl-dev:v0.5.14-rocm720-mi35x-20260708
export IN_OUT_OVERRIDE="1024:16"
export CONC_OVERRIDE="4 8 16 32 64"

cd ~/SGLang-benchmarks

echo "############### RUN 1/2: triton DSA decode ###############"
DSA_DECODE_BACKEND=triton ./GLM.sh --model "$MODEL" --tp 4 \
    --tag AB-TP4-triton --docker "$DOCKER"

echo "############### RUN 2/2: tilelang DSA decode (baseline) ###############"
DSA_DECODE_BACKEND=tilelang ./GLM.sh --model "$MODEL" --tp 4 \
    --tag AB-TP4-tilelang --docker "$DOCKER"

echo "############### A/B DONE ###############"
