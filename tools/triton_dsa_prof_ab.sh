#!/usr/bin/env bash
# Profile A/B: triton vs tilelang DSA decode backend, conc64 in1024/out16.
# Produces DECODE graph-ON traces to compare the decode attention kernel
# (triton sparse-MLA vs tilelang main_kernel partial+combine).
set -uo pipefail

MODEL=/data/huggingface/hub/amd/GLM-5.2-MXFP4
DOCKER=rocm/sgl-dev:v0.5.14-rocm720-mi35x-20260708
export IN_OUT_OVERRIDE="1024:16"
export CONC_OVERRIDE="64"

cd ~/SGLang-benchmarks

echo "############### PROF 1/2: triton DSA decode ###############"
DSA_DECODE_BACKEND=triton ./GLM.sh --model "$MODEL" --tp 4 --prof \
    --tag PROF-AB-TP4-triton --docker "$DOCKER"

echo "############### PROF 2/2: tilelang DSA decode ###############"
DSA_DECODE_BACKEND=tilelang ./GLM.sh --model "$MODEL" --tp 4 --prof \
    --tag PROF-AB-TP4-tilelang --docker "$DOCKER"

echo "############### PROF A/B DONE ###############"
