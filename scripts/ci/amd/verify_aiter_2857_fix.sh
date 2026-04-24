#!/usr/bin/env bash
# verify_aiter_2857_fix.sh
# ------------------------------------------------------------------------------
# Purpose: drive the ROCm-7.2.2-base-image experiment for aiter#2857
# (https://github.com/ROCm/aiter/issues/2857).
#
# Subcommands:
#   build  - (done by CI, not by this script; see investigation doc)
#   push   - push the built image to jacchang/shared on Docker Hub
#   verify - run a smoke-test container that boots Qwen3-235B-MXFP4 with
#            --tp 4 --ep 2 and checks whether CUDA-graph capture completes
#            without the NCCL watchdog hipErrorCapturedEvent crash.
#
# Usage:
#   bash scripts/ci/amd/verify_aiter_2857_fix.sh push [IMAGE]
#   bash scripts/ci/amd/verify_aiter_2857_fix.sh verify [IMAGE]
#
# If IMAGE is omitted, defaults to the test tag below.
set -euo pipefail

DEFAULT_IMAGE="jacchang/shared:sglang-aiter2857-rocm722-mi35x-20260424"
IMAGE="${2:-$DEFAULT_IMAGE}"

log() { printf '[verify-2857] %s\n' "$*" >&2; }

cmd_push() {
  log "target image: $IMAGE"
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    log "ERROR: local image $IMAGE does not exist; build it first"
    exit 1
  fi
  if ! grep -q '"auths"' ~/.docker/config.json 2>/dev/null; then
    log "WARN: ~/.docker/config.json has no auth entries; run 'docker login' first"
  fi
  docker push "$IMAGE"
  log "pushed: $IMAGE"
}

cmd_verify() {
  local image="$IMAGE"
  local name="ci_sglang_aiter2857verify"
  log "verifying fix with image: $image"
  docker rm -f "$name" >/dev/null 2>&1 || true

  # Replicate sgl-dev's launch flags: kfd/dri devices, cap-add for py-spy,
  # shm big enough for the 4-GPU shards, group-add video for access.
  docker run -dt --user root --device=/dev/kfd --device /dev/dri \
    --ulimit nofile=65536:65536 \
    --group-add video --shm-size 32g --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HF_HOME="${HF_HOME:-/root/.cache/huggingface}" \
    -e HF_HUB_ETAG_TIMEOUT=300 \
    -e HF_HUB_DOWNLOAD_TIMEOUT=300 \
    -e MIOPEN_USER_DB_PATH="${HF_HOME:-/root/.cache}/miopen" \
    -e MIOPEN_CUSTOM_CACHE_DIR="${HF_HOME:-/root/.cache}/miopen" \
    --name "$name" \
    "$image" >/dev/null

  log "container $name up; running environment sanity checks"
  docker exec "$name" bash -lc '
    set -e
    echo "=== ROCm (expected 7.2.2) ==="
    cat /opt/rocm/.info/version
    /opt/rocm/bin/hipconfig --version; echo
    echo "=== torch ==="
    python3 -c "import torch; print(torch.__version__); print(\"hip:\", torch.version.hip); print(\"device_count:\", torch.cuda.device_count())"
    echo "=== aiter ==="
    python3 -c "import aiter; print(\"meta_size:\", aiter.meta_size())"
    echo "=== watchdog symbols (should be absent; ROCm 7.2.2 has runtime fix) ==="
    SO=$(python3 -c "import torch,os; print(os.path.dirname(torch.__file__) + \"/lib/libtorch_cpu.so\")")
    strings -a "$SO" | grep -iE "RocmWatchdog|queryEventWithRocm" || echo "  (none; expected)"
  '

  log "launching Qwen3-235B-MXFP4 server with --tp 4 --ep 2 (this is the aiter#2857 repro)"
  log "log tail will be shown; watch for:"
  log "  PASS indicator: '\''The server is fired up and ready to roll!'\''"
  log "  FAIL indicator: hipErrorStreamCaptureUnsupported / hipErrorCapturedEvent"
  docker exec "$name" bash -lc '
    set -e
    SGLANG_USE_AITER=1 timeout 600 python3 -m sglang.launch_server \
      --model-path amd/Qwen3-235B-A22B-Instruct-2507-mxfp4 \
      --tp 4 --ep 2 --trust-remote-code \
      --attention-backend aiter \
      --device cuda --host 127.0.0.1 --port 11000 \
      2>&1 | tee /tmp/qwen3-launch.log | tail -30
    echo "--- relevant log lines ---"
    grep -E "fired up|hipError|Registering .* cuda graph|Capture.*[Dd]one|RuntimeError|watchdog thread terminated" /tmp/qwen3-launch.log | head -20 || true
  '

  log "verify done; container left running as $name for further inspection"
  log "to stop: docker rm -f $name"
}

case "${1:-}" in
  push)    cmd_push "$@" ;;
  verify)  cmd_verify "$@" ;;
  *) echo "Usage: $0 {push|verify} [IMAGE]" >&2; exit 2 ;;
esac
