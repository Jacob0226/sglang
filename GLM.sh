#!/usr/bin/env bash
# Usage:
# ./GLM.sh
# ./GLM.sh --mtp --prof
# ./GLM.sh --prof-combined           # profile without splitting prefill/decode
# ./GLM.sh --dual-stream-rocm        # disable shared-experts-fusion for dual stream on ROCm
# ./GLM.sh --model /data/huggingface/hub/zai-org/GLM-5-FP8
# ./GLM.sh --prof --dual-stream-rocm --tag DualStream
# ./GLM.sh --tp 4 --tag 0507_TP4    # tensor parallel size (auto: TP=4 for FP4 models, TP=8 for FP8)
# ./GLM.sh --docker rocm/sgl-dev:v0.5.10rc0-rocm720-mi35x-20260412   # tag results dir with docker image
# ./GLM.sh --port 8600              # change server port (default 8552; also: PORT=8600 ./GLM.sh)
#
# GLM-5 / GLM-5.1 FP4 examples (auto-detects quant scheme from model name;
# mirrors InferenceX recipes — see SemiAnalysisAI/InferenceX benchmarks/
# single_node/glm5.1_fp4_mi355x.sh and glm5_fp4_b200.sh):
# ./GLM.sh --model amd/GLM-5.1-MXFP4               # MI355X (MXFP4 self-declares)
# ./GLM.sh --model nvidia/GLM-5-NVFP4              # B200   (NV official ModelOpt quant)
set -euo pipefail
set -x
ulimit -n 65535
sh -c 'echo 0 > /proc/sys/kernel/numa_balancing' 2>/dev/null || echo "[warn] cannot disable numa_balancing (need root); continuing"

MTP_ENABLED="false"
PROF_ENABLED="false"
PROF_COMBINED="false"   # if true: single combined trace (no --profile-by-stage)
DUAL_STREAM_ROCM="false"
MTP_TAG=""
USER_TAG=""
# TP_SIZE="auto" means: pick from MODEL_NAME after --model is parsed.
# FP4 models (MXFP4 / NVFP4) default to TP=4 so MI355X and B200 profiles
# are directly comparable at the same TP. FP8 / unrecognized fall back
# to TP=8. Override anytime with --tp.
TP_SIZE="auto"
MODEL_PATH="/data/huggingface/hub/zai-org/GLM-5-FP8"
# DOCKER labels the results directory so different docker images don't clobber
# each other. Override with --docker <image>. Known-good images:
#   rocm/sgl-dev:v0.5.10rc0-rocm720-mi35x-20260412   # MI355
#   lmsysorg/sglang:v0.5.9-cu130-runtime              # B200
DOCKER="untagged-docker"
CURRENT_DIR=$(pwd)
while [[ $# -gt 0 ]]; do
  case $1 in
    --mtp)
        MTP_ENABLED="true"
        MTP_TAG="-MTP"
        shift 1
        ;;
    --prof)
        PROF_ENABLED="true"
        shift 1
        ;;
    --prof-combined)
        PROF_ENABLED="true"
        PROF_COMBINED="true"
        shift 1
        ;;
    --dual-stream-rocm)
        DUAL_STREAM_ROCM="true"
        shift 1
        ;;
    --model)
        MODEL_PATH="$2"
        shift 2
        ;;
    --tp)
        TP_SIZE="$2"
        shift 2
        ;;
    --tag)
        USER_TAG="-$2"
        shift 2
        ;;
    --docker)
        DOCKER="$2"
        shift 2
        ;;
    --port)
        PORT="$2"
        shift 2
        ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done
# Use the last two path components joined with '_' so the org is kept, e.g.
#   /data/huggingface/hub/amd/GLM-5.1-MXFP4    -> amd_GLM-5.1-MXFP4
#   /data/huggingface/hub/nvidia/GLM-5-NVFP4   -> nvidia_GLM-5-NVFP4
#   /data/huggingface/hub/zai-org/GLM-5.1-FP8  -> zai-org_GLM-5.1-FP8
_MODEL_PATH_TRIMMED="${MODEL_PATH%/}"
_MODEL_LEAF=$(basename "${_MODEL_PATH_TRIMMED}")
_MODEL_ORG=$(basename "$(dirname "${_MODEL_PATH_TRIMMED}")")
MODEL_NAME="${_MODEL_ORG}_${_MODEL_LEAF}"

# ===================== Quantization auto-detection (matches InferenceX) =====================
# Pick --quantization and --mem-fraction-static based on the model name.
# Mirrors SemiAnalysisAI/InferenceX recipes in benchmarks/single_node/
# {glm5.1_fp4_mi355x.sh, glm5_fp4_b200.sh, glm5_fp8_b200.sh}.
#
#   *MXFP4*  -> AMD MXFP4 (e.g. amd/GLM-5.1-MXFP4): model files self-declare
#               quant, so no --quantization flag is passed. AMD's Quark recipe
#               quantizes the shared experts to MXFP4 too (model card:
#               "MOE-only (shared experts quantized), OCP MXFP4"), so fusion
#               works (no --disable-shared-experts-fusion needed). Matches
#               InferenceX glm5.1_fp4_mi355x.sh.
#   *NVFP4*  -> NVIDIA NVFP4 (e.g. nvidia/GLM-5-NVFP4): needs
#               --quantization modelopt_fp4 and --mem-fraction-static 0.9
#               (matches InferenceX glm5_fp4_b200.sh's KV-pool budget; NV's
#               HF card uses 0.80). No --disable-shared-experts-fusion (NV's
#               HF launch command and InferenceX glm5_fp4_b200.sh both omit
#               it — sglang's modelopt_fp4 path handles it correctly).
#   *FP8*    -> default GLM-5-FP8: --quantization fp8 on B200, none on ROCm.
QUANT_ARGS=()
MEM_FRACTION_STATIC="0.85"
# Set to "true" by --dual-stream-rocm (see start_server()) — needed so MoE.forward
# takes forward_normal_dual_stream instead of the fused shared-expert path.
NEED_DISABLE_SHARED_FUSION="false"
case "${MODEL_NAME}" in
    *NVFP4*)
        QUANT_ARGS=(--quantization modelopt_fp4)
        MEM_FRACTION_STATIC="0.8"
        case "${MODEL_NAME}" in
            *GLM-5.2*)
                # GLM-5.2 (glm_moe_dsa): NVIDIA's HF card launches at TP=8
                # But I need TP4
                # (https://huggingface.co/nvidia/GLM-5.2-NVFP4).
                [ "$TP_SIZE" = "auto" ] && TP_SIZE=4
                ;;
            *)
                # GLM-5 NVFP4 default TP=4 — InferenceX runs both TP=4 and TP=8
                # for B200 NVFP4 (yaml: { tp: 4, conc 4–256 } is the main sweep);
                # we pick TP=4 so it cross-compares cleanly with MI355X MXFP4 at TP=4.
                [ "$TP_SIZE" = "auto" ] && TP_SIZE=4
                ;;
        esac
        ;;
    *MXFP4*)
        # MXFP4 self-declares; shared experts are also MXFP4 -> fusion OK.
        # InferenceX MI355X main sweep is TP=2 but its TP=4 entry is the
        # one that lines up with B200's TP=4 NVFP4 sweep, so default to 4
        # here for direct MI355X-vs-B200 comparison. Override with --tp 2
        # to match InferenceX's MXFP4 main sweep.
        [ "$TP_SIZE" = "auto" ] && TP_SIZE=4
        ;;
    *FP8*)
        QUANT_ARGS=(--quantization fp8)
        ;;
esac
# Fallback for FP8 / unrecognized models: keep the historical TP=8 default.
[ "$TP_SIZE" = "auto" ] && TP_SIZE=8

# ===================== Server and Benchmark Setting =====================
# InferenceMax tuning (from InferenceX/glm5_fp8_mi355x.sh)
export SAFETENSORS_FAST_GPU=1
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
# 0708 docker issue: the GLM-5.2 DSA decode path now wires in the topk_v2 JIT
# kernel (dsa_backend._build_topk_v2_plan -> jit_kernel/dsv4/topk.py), whose
# topk_impl.cuh #includes <cooperative_groups.h> -- a CUDA header not shipped by
# ROCm 7.2 -> hipcc "ninja exited with status 1 / cooperative_groups.h not found"
# during CUDA-graph capture, so the server won't start. Disable topk_v2 until the
# kernel is hipified. (0628 docker never hit this: its dsa_backend didn't call
# topk_v2 for GLM-5.2 at all.)
export SGLANG_OPT_USE_TOPK_V2=0
# export AITER_ONLINE_TUNE=1
HOST="localhost"
# Override with `--port <n>` or `PORT=<n> ./GLM.sh`. Default 8552.
PORT="${PORT:-8234}"
DATASET="random"
in_out_tokens=("1024:1024" "8192:1024"  "70000:300")
random_range_ratio=0.8
concurrencies=(4 8 16 32 64) # 128 256
# Optional env overrides (space-separated), e.g. for a targeted single-config
# rerun without editing this file:
#   IN_OUT_OVERRIDE="1024:1024" CONC_OVERRIDE="8" ./GLM.sh ...
if [ -n "${IN_OUT_OVERRIDE:-}" ]; then read -ra in_out_tokens <<< "$IN_OUT_OVERRIDE"; fi
if [ -n "${CONC_OVERRIDE:-}" ]; then read -ra concurrencies <<< "$CONC_OVERRIDE"; fi
# in_out_tokens=("1024:1024")
# concurrencies=(32 64)
PROMPT_MULTIPLIER=5
if [ "$PROF_COMBINED" == "true" ]; then
    PROF_CMD=(--profile --profile-num-steps 5)
    COMBINED_SUFFIX="_Combined"
else
    PROF_CMD=(--profile --profile-num-steps 5 --profile-by-stage)
    COMBINED_SUFFIX=""
fi

# Vendor tag inserted into profiler trace filenames (e.g. ..._p8-AMD-TP-0-...).
# Auto-detect: AMD on ROCm, NV otherwise. Override with: VENDOR_TAG=AMD ./GLM.sh ...
if [ -z "${VENDOR_TAG:-}" ]; then
    if [ -e /dev/kfd ] || command -v rocm-smi >/dev/null 2>&1; then
        VENDOR_TAG="AMD"
    else
        VENDOR_TAG="NV"
    fi
fi

# ===================== Argument  =====================
SPECIAL_TAG="-bench"
if [ "$PROF_ENABLED" == "true" ]; then
    SPECIAL_TAG="-prof"
    concurrencies=(4)
    PROMPT_MULTIPLIER=2 # Faster for no cuda graph profiling

    # Debug
    in_out_tokens=("1024:16" "8192:16")
    concurrencies=(4 64)
fi
DOCKER_FILENAME=$(echo "$DOCKER" | sed 's/\//_/g; s/:/-/g')
# Layout: results/<model>/<docker-image>/<tags>
# The leaf is the combined tag portion (MTP/bench-or-prof/user tag) with the
# leading '-' stripped so it doesn't start with a dash.
LEAF_TAG="${MTP_TAG}${SPECIAL_TAG}${USER_TAG}"
LEAF_TAG="${LEAF_TAG#-}"
LOG_DIR="$HOME/SGLang-benchmarks/results/${MODEL_NAME}/$DOCKER_FILENAME/${LEAF_TAG}"
FINISH_LOG="$LOG_DIR/Finish.log"
mkdir -p "$LOG_DIR"
touch "$FINISH_LOG"
if [ "$PROF_ENABLED" == "true" ]; then
    export SGLANG_TORCH_PROFILER_DIR=$LOG_DIR
fi

log_command() {
    local logfile=$1
    shift 
    
    # Record command
    echo ">>>Executing command:" | tee -a "$logfile"
    echo "$*" | tee -a "$logfile"  
    echo "---" | tee -a "$logfile"

    # Execute command
    "$@" 2>&1 | tee -a "$logfile"
}

is_rocm_gpu_env() {
    [ -e /dev/kfd ] || command -v rocm-smi >/dev/null 2>&1
}

list_profiler_dirs() {
    find "${LOG_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | grep -E '^[0-9]+(\.[0-9]+)?$' || true
}

rename_profiler_artifacts() {
    local input_tokens=$1
    local output_tokens=$2
    local c=$3
    local num_prompts=$4
    local before_dirs=$5
    local after_dirs=$6
    local target_dir_name="prof_in${input_tokens}_out${output_tokens}_conc${c}_p${num_prompts}${COMBINED_SUFFIX}"
    local target_dir_path="${LOG_DIR}/${target_dir_name}"
    local new_dirs

    new_dirs=$(comm -13 <(printf '%s\n' "${before_dirs}" | sort) <(printf '%s\n' "${after_dirs}" | sort))
    if [ -z "${new_dirs}" ]; then
        echo "No new profiler directory found under ${LOG_DIR}"
        return 0
    fi

    local src_dir src_dir_path
    src_dir=$(printf '%s\n' "${new_dirs}" | tail -n 1)
    src_dir_path="${LOG_DIR}/${src_dir}"
    if [ "${src_dir}" != "${target_dir_name}" ]; then
        if [ -e "${target_dir_path}" ]; then
            target_dir_path="${LOG_DIR}/${target_dir_name}_$(date +%s)"
            echo "Target directory exists. Using ${target_dir_path}"
        fi
        mv "${src_dir_path}" "${target_dir_path}"
        echo "Renamed profiler dir: ${src_dir} -> $(basename "${target_dir_path}")"
    fi

    local trace_file filename tp_rank new_name
    for trace_file in "${target_dir_path}"/*-TP-*.trace.json.gz; do
        [ -f "${trace_file}" ] || continue
        filename=$(basename "${trace_file}")
        tp_rank=$(sed -E 's/^.*-TP-([0-9]+)\.trace\.json\.gz$/\1/' <<< "${filename}")
        new_name="in${input_tokens}_out${output_tokens}_conc${c}_p${num_prompts}${COMBINED_SUFFIX}-${VENDOR_TAG}-TP-${tp_rank}${NOGRAPH_SUFFIX}.trace.json.gz"
        mv "${trace_file}" "${target_dir_path}/${new_name}"
        echo "Renamed trace: ${filename} -> ${new_name}"
    done
}

rename_profiler_artifacts_by_stage() {
    local input_tokens=$1
    local output_tokens=$2
    local c=$3
    local num_prompts=$4
    local before_dirs=$5
    local after_dirs=$6
    local target_dir_name="prof_in${input_tokens}_out${output_tokens}_conc${c}_p${num_prompts}${COMBINED_SUFFIX}"
    local target_dir_path="${LOG_DIR}/${target_dir_name}"
    local new_dirs

    new_dirs=$(comm -13 <(printf '%s\n' "${before_dirs}" | sort) <(printf '%s\n' "${after_dirs}" | sort))
    if [ -z "${new_dirs}" ]; then
        echo "No new profiler directory found under ${LOG_DIR}"
        return 0
    fi

    local src_dir src_dir_path
    src_dir=$(printf '%s\n' "${new_dirs}" | tail -n 1)
    src_dir_path="${LOG_DIR}/${src_dir}"
    if [ "${src_dir}" != "${target_dir_name}" ]; then
        if [ -e "${target_dir_path}" ]; then
            target_dir_path="${LOG_DIR}/${target_dir_name}_$(date +%s)"
            echo "Target directory exists. Using ${target_dir_path}"
        fi
        mv "${src_dir_path}" "${target_dir_path}"
        echo "Renamed profiler dir: ${src_dir} -> $(basename "${target_dir_path}")"
    fi

    local trace_file filename tp_rank stage new_name
    for trace_file in "${target_dir_path}"/*-TP-*.trace.json.gz; do
        [ -f "${trace_file}" ] || continue
        filename=$(basename "${trace_file}")
        tp_rank=$(sed -E 's/^.*-TP-([0-9]+)-(EXTEND|DECODE)\.trace\.json\.gz$/\1/' <<< "${filename}")
        stage=$(sed -E 's/^.*-TP-([0-9]+)-(EXTEND|DECODE)\.trace\.json\.gz$/\2/' <<< "${filename}")

        if [ "${tp_rank}" = "${filename}" ] || [ "${stage}" = "${filename}" ]; then
            echo "Skip unmatched trace name: ${filename}"
            continue
        fi

        new_name="in${input_tokens}_out${output_tokens}_conc${c}_p${num_prompts}${COMBINED_SUFFIX}-${VENDOR_TAG}-TP-${tp_rank}-${stage}${NOGRAPH_SUFFIX}.trace.json.gz"
        mv "${trace_file}" "${target_dir_path}/${new_name}"
        echo "Renamed trace: ${filename} -> ${new_name}"
    done
}

prof_cmd_has_profile_by_stage() {
    local arg
    for arg in "${PROF_CMD[@]}"; do
        if [ "${arg}" = "--profile-by-stage" ]; then
            return 0
        fi
    done
    return 1
}

start_server() {
    local logfile="${LOG_DIR}/server_${MODEL_NAME}.log"
    echo ">>> Starting SGLang server" | tee "$logfile"

    local cmd=(
        python3 -m sglang.launch_server
            --model $MODEL_PATH
            --tp $TP_SIZE
            --host $HOST
            --port $PORT
            --trust-remote-code
            --tool-call-parser glm47
            --reasoning-parser glm45
            --watchdog-timeout 1200
            --mem-fraction-static "$MEM_FRACTION_STATIC"
            --kv-cache-dtype fp8_e4m3
            --disable-radix-cache
            --model-loader-extra-config "{\"enable_multithread_load\": true, \"num_threads\": ${WEIGHT_LOAD_THREADS:-32}}"
    )
    if [ ${#QUANT_ARGS[@]} -gt 0 ]; then
        cmd+=("${QUANT_ARGS[@]}")
    fi

    if is_rocm_gpu_env; then
        # Match InferenceX glm5.1_fp4_mi355x.sh: tilelang NSA backends,
        # tokenizer-worker-num scales with TP.
        # DSA prefill/decode backend overridable via env (default tilelang).
        # e.g. DSA_DECODE_BACKEND=triton to test Clint's PR #30575 triton
        # sparse-MLA decode kernel. Uses the modern --dsa-*-backend flag names.
        cmd+=(
            --dsa-prefill-backend "${DSA_PREFILL_BACKEND:-tilelang}"
            --dsa-decode-backend "${DSA_DECODE_BACKEND:-tilelang}"
            --tokenizer-worker-num $((TP_SIZE * 2))
        )
        # opt#3: aiter fused allreduce(+residual+rmsnorm) replaces the unfused
        # cross_device_reduce + add_rmsnorm_quant pair (~12us/layer on MI355X,
        # matching ATOM's allreduce_fusion_kernel_1stage). gfx950-only path; enable
        # only for the AMD MXFP4 GLM-5.1/5.2 recipes it was validated against.
        case "${MODEL_NAME}" in
            amd_GLM-5.1-MXFP4|amd_GLM-5.2-MXFP4)
                cmd+=(--enable-aiter-allreduce-fusion)
                ;;
        esac
        if [ "$DUAL_STREAM_ROCM" == "true" ]; then
            # Two independent toggles must both be set for full ROCm dual-stream:
            #   (a) --disable-shared-experts-fusion
            #         Forces num_fused_shared_experts=0 so DeepseekV2MoE.forward
            #         takes forward_normal_dual_stream (shared ∥ routed overlap)
            #         instead of forward_normal which would use the fused
            #         _fused_append_shared_experts_kernel.
            #   (b) SGLANG_ENABLE_HIP_DUAL_STREAM=1
            #         Required to actually create alt_stream on ROCm. Without
            #         it, alt_stream=None on HIP and *both* the NSA-decode A_v4
            #         layout and the MoE forward_normal_dual_stream are skipped.
            #         (Default OFF because the layout regresses on MI355X — see
            #         tools/dual_stream_regression_analysis.md for full analysis.)
            NEED_DISABLE_SHARED_FUSION="true"
            export SGLANG_ENABLE_HIP_DUAL_STREAM=1
        fi
    elif [[ "${MODEL_NAME}" == *GLM-5.2* ]]; then
        # GLM-5.2 (glm_moe_dsa) on B200: use NVIDIA's OFFICIAL HF launch settings
        # verbatim (https://huggingface.co/nvidia/GLM-5.2-NVFP4):
        #     --tp 8 --quantization modelopt_fp4 --tool-call-parser glm47
        #     --reasoning-parser glm45 --trust-remote-code
        #     --chunked-prefill-size 16384 --mem-fraction-static 0.80
        # TP (8) / quant (modelopt_fp4) / parsers / trust-remote-code / mem-fraction
        # (0.80) are already set above; here we only add the 16K prefill chunk.
        # We deliberately do NOT apply the GLM-5 / GLM-5.1 B200 tuning block below
        # (--attention-backend nsa, --enable-flashinfer-allreduce-fusion,
        # --stream-interval 30, --moe-runner-backend, 32K chunking, --cuda-graph-max-bs):
        # that combo triggered a reproducible TP all-gather (vocab-sized) deadlock on
        # the DSA path. Letting sglang auto-pick backends matches NV's command and
        # ran stably in testing.
        cmd+=(
            --chunked-prefill-size 16384
        )
    else
        # NVIDIA (B200) specific optimizations. Matches InferenceX
        # glm5_fp4_b200.sh / glm5_fp8_b200.sh: trtllm NSA, flashinfer MoE,
        # 32K prefill chunking, allreduce fusion, fixed stream-interval.
        cmd+=(
            --attention-backend nsa
            --nsa-prefill-backend trtllm
            --nsa-decode-backend trtllm
            --moe-runner-backend flashinfer_trtllm
            --chunked-prefill-size 32768
            --max-prefill-tokens 32768
            --enable-flashinfer-allreduce-fusion
            --stream-interval 30
            --tokenizer-worker-num 6
        )
        # NVFP4-specific: cap CUDA-graph BS and lower scheduler poll
        # interval to match InferenceX's glm5_fp4_b200.sh exactly.
        case "${MODEL_NAME}" in
            *NVFP4*)
                cmd+=(
                    --cuda-graph-max-bs 256
                    --scheduler-recv-interval 10
                )
                ;;
        esac
    fi

    # Auto-add --disable-shared-experts-fusion only when ROCm dual-stream is
    # enabled (forces num_fused_shared_experts=0 so MoE.forward takes
    # forward_normal_dual_stream instead of the fused path).
    # NOT added for nvidia/GLM-5-NVFP4 or amd/GLM-5.1-MXFP4: NV's official
    # sglang command, InferenceX glm5_fp4_b200.sh, and glm5.1_fp4_mi355x.sh
    # all omit it (sglang's modelopt_fp4 path handles NVFP4's mixed-precision
    # shared expert correctly; AMD's MXFP4 quantizes shared experts too).
    if [ "$NEED_DISABLE_SHARED_FUSION" = "true" ]; then
        cmd+=(--disable-shared-experts-fusion)
    fi

    if [ "$MTP_ENABLED" == "true" ]; then
        # EAGLE chain matches InferenceX glm5_fp8_mi355x_mtp.sh: (steps=3, topk=1, draft=4).
        # SGLANG_ENABLE_SPEC_V2=1 enables sglang's new spec scheduler (also set by InferenceX).
        # On MI355X we keep --nsa-{prefill,decode}-backend tilelang from above; do NOT
        # override --attention-backend (InferenceX doesn't either, tilelang NSA + EAGLE works).
        echo ">>> Speculative Decoding (MTP) is ENABLED." | tee -a "$logfile"
        export SGLANG_ENABLE_SPEC_V2=1
        cmd+=(
            --speculative-algorithm EAGLE
            --speculative-num-draft-tokens 4
            --speculative-num-steps 3
            --speculative-eagle-topk 1
        )
    fi

    if [ ${#EXTRA_SERVER_ARGS[@]} -gt 0 ]; then
        cmd+=("${EXTRA_SERVER_ARGS[@]}")
    fi

    # Preflight: fail fast if the port is already taken (otherwise the model
    # loads for ~2 min and only then dies with "[Errno 98] Address already in
    # use"). Common cause: a stale/orphaned sglang server from a previous run,
    # or another server sharing this host. Stop it or pick another --port.
    if (exec 3<>"/dev/tcp/${HOST}/${PORT}") 2>/dev/null; then
        exec 3>&- 3<&-
        echo "!!! ERROR: ${HOST}:${PORT} is already in use. Stop the existing server " \
             "(e.g. 'pkill -9 -f sglang.launch_server') or run with '--port <free-port>'." | tee -a "$logfile"
        exit 1
    fi

    # Start server in background
    echo ">>> Executing command:" | tee -a "$logfile"
    echo "${cmd[*]}" | tee -a "$logfile"
    echo "---" | tee -a "$logfile"
    "${cmd[@]}" 2>&1 | tee -a "$logfile" &

    echo ">>> Waiting for server to be ready (checking: '${logfile}')..." | tee -a "$logfile"
    until [ "$(curl -s -o /dev/null -w "%{http_code}" "http://${HOST}:$PORT/health" 2>/dev/null)" = "200" ]; do
        # Detect server death during startup (port bind failure, OOM, GPU fault)
        # so we don't poll forever. pgrep is scoped to this server's port.
        if ! pgrep -f "sglang.launch_server.*--port $PORT" >/dev/null 2>&1; then
            echo "!!! ERROR: server process died during startup. See '${logfile}' " \
                 "(look for 'Address already in use', OOM, or 'Memory access fault')." | tee -a "$logfile"
            exit 1
        fi
        echo "Waiting for server to be ready at http://${HOST}:$PORT/health..."
        sleep 5
    done
}

warmup() {
    local warmup_log="${LOG_DIR}/warmup.log"
    local warmup_cmd=(
        python3 -m sglang.bench_serving 
        --host $HOST 
        --port "${PORT}" 
        --model "${MODEL_PATH}" 
        --dataset-name "${DATASET}" 
        --random-input 1024
        --random-output 16
        --random-range-ratio "${random_range_ratio}"
        --max-concurrency 4 
        --num-prompt 4 
        --output-file /dev/null
    )
    log_command "$warmup_log" "${warmup_cmd[@]}"
}

accuracy_test() {
    # GSM8K
    gsm8k_logfile=$LOG_DIR/Accuracy_GSM8K.log
    if ! grep -q "$gsm8k_logfile" "$FINISH_LOG"; then
        echo ">>> Running Accuracy check (GSM8K)..."
        # --parallel caps how many GSM8K requests run concurrently. A very high
        # value (e.g. 1200) floods the server into one giant batch (#running-req
        # ~1197) which can trip a GPU memory-access fault in the MXFP4/tilelang
        # NSA kernels on MI355X. Accuracy is unaffected by lowering it (same 1200
        # questions, fewer in flight). Override with GSM8K_PARALLEL=<n>.
        gsm8k_cmd=(
            python3 /sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py 
                --port "$PORT" 
                --num-questions 1200 
                --parallel "${GSM8K_PARALLEL:-1200}"
        )
        log_command "$gsm8k_logfile" "${gsm8k_cmd[@]}"
        echo "$gsm8k_logfile" >> "$FINISH_LOG"
    else
        echo "Found Accuracy_GSM8K.log in ${FINISH_LOG}. Skipping."
    fi
}

run_benchmarks() {
    # Benchmark Loop
    for io_pair in "${in_out_tokens[@]}"; do
        IFS=":" read -r input_tokens output_tokens <<< "$io_pair"
        for c in "${concurrencies[@]}"; do
            local num_prompts=$((c * PROMPT_MULTIPLIER))
            local logfile="${LOG_DIR}/bench_in${input_tokens}_out${output_tokens}_conc${c}.log"

            local cmd=(
                python3 -m sglang.bench_serving
                --host "${HOST}"
                --port "${PORT}"
                --model "${MODEL_PATH}"
                --dataset-name "${DATASET}"
                --random-input "${input_tokens}"
                --random-output "${output_tokens}"
                --random-range-ratio "${random_range_ratio}"
                --max-concurrency "${c}"
                --num-prompt "${num_prompts}"
                --output-file /dev/null
            )
            
            # Add profiling args
            if [ "$PROF_ENABLED" == "true" ]; then
                cmd+=("${PROF_CMD[@]}")
            fi

            # Determine skip condition:
            # - Profiling mode: skip if the profile output directory already exists
            # - Benchmark mode: skip if logfile is recorded in Finish.log
            local skip="false"
            local prof_dir="${LOG_DIR}/prof_in${input_tokens}_out${output_tokens}_conc${c}_p${num_prompts}${COMBINED_SUFFIX}"
            if [ "$PROF_ENABLED" == "true" ]; then
                if [ -d "${prof_dir}" ]; then
                    echo "Found profile dir ${prof_dir}. Skipping."
                    skip="true"
                fi
            else
                if grep -q "$logfile" "$FINISH_LOG"; then
                    echo "Found $logfile in ${FINISH_LOG}. Skipping."
                    skip="true"
                fi
            fi

            if [ "$skip" == "false" ]; then
                echo "Running: $logfile"
                local profiler_dirs_before=""
                local profiler_dirs_after=""
                if [ "$PROF_ENABLED" == "true" ]; then
                    profiler_dirs_before=$(list_profiler_dirs) # Get the current folders under $LOG_DIR
                fi
                # Don't let a profiler-teardown crash (server segfault / NCCL
                # heartbeat / known ROCm torch-profiler teardown fault) abort the
                # whole run via `set -e`. The trace is typically already flushed
                # to disk before the crash, so we still want to fall through to
                # the rename step below and continue with the next iteration.
                if ! log_command "$logfile" "${cmd[@]}"; then
                    echo "[warn] command exited non-zero for ${logfile} (likely profiler teardown crash); traces may still be present — continuing to rename."
                fi
                echo "$logfile" >> "$FINISH_LOG"

                if [ "$PROF_ENABLED" == "true" ]; then
                    profiler_dirs_after=$(list_profiler_dirs) # Get the current folders under $LOG_DIR. This time will have another torch profiler folder
                    echo ">>> Processing profiler traces..."
                    if prof_cmd_has_profile_by_stage; then
                        rename_profiler_artifacts_by_stage "${input_tokens}" "${output_tokens}" "${c}" "${num_prompts}" "${profiler_dirs_before}" "${profiler_dirs_after}" \
                            || echo "[warn] rename_profiler_artifacts_by_stage failed for ${logfile}; raw trace dir left in place — continuing."
                    else
                        rename_profiler_artifacts "${input_tokens}" "${output_tokens}" "${c}" "${num_prompts}" "${profiler_dirs_before}" "${profiler_dirs_after}" \
                            || echo "[warn] rename_profiler_artifacts failed for ${logfile}; raw trace dir left in place — continuing."
                    fi
                fi
            fi
        done
    done
}


# ===================== Package Setup =====================
if ! is_rocm_gpu_env; then
    export SGL_ENABLE_JIT_DEEPGEMM=1
    # NV image ships a PEP 668 "externally managed" Python; install the
    # missing 'distro' dep (imported by sglang.bench_serving) with the
    # override flag. ROCm image already has it, so skip there.
    if ! python3 -c "import distro" >/dev/null 2>&1; then
        python3 -m pip install --user --break-system-packages distro
    fi
    # GLM-5.2 (nvidia/GLM-5.2-NVFP4) uses the glm_moe_dsa architecture, which the
    # transformers pinned in lmsysorg/sglang images (==5.8.1) is too old to load
    # ("layer_types entries must be in ... deepseek_sparse_attention"). Per the NV
    # model card (https://huggingface.co/nvidia/GLM-5.2-NVFP4), upgrade transformers
    # before launching. (--break-system-packages: same PEP 668 override as above.)
    # Gated to GLM-5.2 only so other models (GLM-5 / GLM-5.1) keep the pinned version.
    #
    # IMPORTANT: only upgrade if transformers is actually too old (< 5.3.0). Do NOT
    # blindly `pip install -U`: the purpose-built dev-glm52 image already ships a
    # compatible transformers (e.g. 5.12.1), and forcing it to the newest (5.13.0)
    # pulls in a version whose built-in `qwen3_asr` config collides with sglang's
    # own qwen3_asr registration ("'qwen3_asr' is already used by a Transformers
    # config"), which kills server startup at import time.
    case "${MODEL_NAME}" in
        *GLM-5.2*)
            if python3 -c "import transformers; from packaging.version import Version; import sys; sys.exit(0 if Version(transformers.__version__) >= Version('5.3.0') else 1)" 2>/dev/null; then
                echo "[info] transformers $(python3 -c 'import transformers; print(transformers.__version__)') already satisfies >=5.3.0; skipping upgrade."
            else
                python3 -m pip install --break-system-packages "transformers>=5.3.0"
            fi
            ;;
    esac
fi



prof_mode_complete() {
    # True (0) if every expected profile output for the CURRENT mode (LOG_DIR +
    # PROMPT_MULTIPLIER already set for this mode) already exists on disk. Lets a
    # re-run skip a whole mode — server launch + warmup + GSM8K included — and jump
    # straight to the mode that still needs data (e.g. cuda-graph already profiled
    # -> go directly to no-cuda-graph). Only meaningful in --prof mode.
    [ "$PROF_ENABLED" == "true" ] || return 1
    local io input_tokens output_tokens c num_prompts prof_dir
    for io in "${in_out_tokens[@]}"; do
        IFS=":" read -r input_tokens output_tokens <<< "$io"
        for c in "${concurrencies[@]}"; do
            num_prompts=$((c * PROMPT_MULTIPLIER))
            prof_dir="${LOG_DIR}/prof_in${input_tokens}_out${output_tokens}_conc${c}_p${num_prompts}${COMBINED_SUFFIX}"
            [ -d "$prof_dir" ] || return 1
        done
    done
    return 0
}

# ------------------- Start -----------------
if [ "$PROF_ENABLED" == "true" ]; then
    PROF_SERVER_MODES=("default" "no-cuda-graph")
else
    PROF_SERVER_MODES=("default")
fi

BASE_LOG_DIR="$LOG_DIR"

for PROF_MODE in "${PROF_SERVER_MODES[@]}"; do
    EXTRA_SERVER_ARGS=()
    NOGRAPH_SUFFIX=""
    if [ "$PROF_MODE" == "no-cuda-graph" ]; then
        EXTRA_SERVER_ARGS=(--disable-cuda-graph)
        NOGRAPH_SUFFIX="-NoGraph"
        PROMPT_MULTIPLIER=1
        LOG_DIR="${BASE_LOG_DIR}/no-cuda-graph"
        mkdir -p "$LOG_DIR"
        FINISH_LOG="$LOG_DIR/Finish.log"
        touch "$FINISH_LOG"
        export SGLANG_TORCH_PROFILER_DIR=$LOG_DIR
    else
        LOG_DIR="${BASE_LOG_DIR}"
    fi

    # Smart skip: if this mode's profile dirs already exist, don't launch the
    # server / warmup / GSM8K at all — jump to the next mode. (e.g. cuda-graph
    # profiling already done -> go straight to no-cuda-graph.)
    if prof_mode_complete; then
        echo ">>> [${PROF_MODE}] all profile dirs already present under '${LOG_DIR}' — skipping server launch / warmup / gsm8k for this mode."
        continue
    fi

    # Run each profiling mode inside a subshell so any fatal error — server
    # death (start_server's `exit 1`), profiler teardown crash, etc. — only
    # aborts THIS mode rather than the whole script. The next mode (e.g.
    # no-cuda-graph) still runs, and the cleanup below always executes.
    (
        echo ">>> [${PROF_MODE}] Starting server and benchmarks..."
        start_server
        warmup
        if [ "$PROF_MODE" == "default" ]; then
            accuracy_test
        fi
        run_benchmarks
    ) || echo "[warn] profiling mode '${PROF_MODE}' aborted (exit $?); continuing to cleanup and next mode."

    echo "[${PROF_SERVER_MODES[@]}], now is the end of ${PROF_MODE}"
    pkill -9 python || true
    pkill -9 sglang || true
    sleep 10
done














# GLM-4.7 is not a VLM.
# MMMU test
# TOKEN_LIST=(512 1024 2048 4096 8192 16384 131072)
# TOKEN_LIST=(16384 131072)
# ITERATIONS=3
# mkdir -p "$LOG_DIR/MMMU"
# for iter in $(seq 1 $((ITERATIONS))); do
#     for token in "${TOKEN_LIST[@]}"; do
#         mmmu_logfile="$LOG_DIR/MMMU/MMMU_Token${token}_Iter${iter}.log"
#         mmmu_cmd=(
#             python3 /sgl-workspace/sglang/benchmark/mmmu/bench_sglang.py  
#                 --port "$PORT" 
#                 --concurrency 900 
#                 --parallel 900 
#                 --temperature 0
#                 --max-new-tokens "$token" 
#                 --result-file "$LOG_DIR/MMMU/MMMU_Token${token}_ResultFile_Iter${iter}.jsonl"
#                 --raw-result-file "$LOG_DIR/MMMU/MMMU_Token${token}_RawResultFile_Iter${iter}.jsonl"
#         )
        
#         if ! grep -q "$mmmu_logfile" "$FINISH_LOG" 2>/dev/null; then
#             echo ">>> Running MMMU: Token $token, Iteration $iter"

#             start_time=$(date +%s)
#             log_command "$mmmu_logfile" "${mmmu_cmd[@]}"
#             end_time=$(date +%s)
#             elapsed=$((end_time - start_time))
#             formatted_time=$(date -u -d "@$elapsed" +"%H:%M:%S")
#             echo "-------------------------------------------" >> "$mmmu_logfile"
#             echo "Execution Time $formatted_time (HH:MM:SS)" >> "$mmmu_logfile"

#             echo "$mmmu_logfile" >> "$FINISH_LOG"
#             mv $CURRENT_DIR/answer_sglang.json $LOG_DIR/MMMU/MMMU_Token${token}_answer_sglang_Iter${iter}.json
#         else
#             echo "Found $mmmu_logfile in Finish.log. Skipping."
#         fi
#     done
# done
# # stop_server
# echo ">>> All configurations completed. Logs: ${LOG_DIR}. Stop server..."
# pkill -9 python || true
# sleep 10
