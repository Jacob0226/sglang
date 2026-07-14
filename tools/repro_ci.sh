#!/usr/bin/env bash
# repro_ci.sh
# ------------------------------------------------------------------------------
# Reproduce SGLang AMD CI jobs locally on this MI35x / MI325 box.
#
# This mirrors what GitHub Actions does for AMD CI in
# .github/workflows/{pr-test-amd,nightly-test-amd*}.yml: launch the same
# rocm/sgl-dev:* (or custom) image, set the same env vars (SGLANG_IS_IN_CI*,
# SGLANG_USE_AITER, GPU_ARCHS, MIOPEN/HF caches), and run test/run_suite.py
# with the same suite/partition. Useful for repro-ing a failing CI job
# before pushing a PR.
#
# By default, the image is treated as self-contained: tests run from the
# image's /sgl-workspace/sglang/test against the image's installed sglang;
# only the HF cache is mounted from the host. Pass --sglang-dir PATH if
# you've made local sglang code changes that you want to test against.
#
# ============================================================================
# QUICKSTART
# ============================================================================
# CI showed `python3 .../test_xxx.py` failed -> just rerun that one file.
# 90% of the time this is what you want; you do NOT need --partition-*.
#
#     bash repro_ci.sh \
#         --docker rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260211 \
#         --single-file registered/amd/test_qwen3_instruct_mxfp4.py
#
# Map "what CI showed me" -> "what to pass":
#
#   CI surface clue                     | Flags to pass
#   ------------------------------------+-------------------------------------
#   one test file failed                | --single-file PATH         (90%)
#   whole nightly job failed (1 runner) | --suite NAME --nightly
#   whole per-commit stage failed       | --suite NAME    (no partition)
#   one shard of a partitioned stage    | --suite NAME --partition-id N \
#     (e.g. "(...gpu-1, 5)")            |   --partition-size M
#   multimodal-gen-test-*-gpu-amd       | not supported -- use --shell, see
#                                       | MULTIMODAL-GEN NOTE below
#
# Add --sglang-dir $HOME/PR/sglang if you've edited host code and want those
# edits picked up; otherwise the image's /sgl-workspace/sglang is used.
#
# ============================================================================
# MULTIMODAL-GEN NOTE
# ============================================================================
# multimodal-gen-test-*-gpu-amd jobs use a different test driver
# (python/sglang/multimodal_gen/test/run_suite.py) with different flags and
# 8 extra diffusion env vars; this script does NOT wrap that driver. To repro
# such a CI failure, use --shell to set up the container, then paste the
# CI's docker exec command verbatim. Example for
# "multimodal-gen-test-2-gpu-amd (linux-mi325-gpu-2, 2)" failure:
#
#   # 1. Open the container with host code mounted (so /sglang-checkout/python
#   #    matches your branch) and a mi325 hostname for arch detection.
#   bash repro_ci.sh \
#       --docker rocm/sgl-dev:v0.5.8.post1-rocm700-mi30x-20260211 \
#       --sglang-dir "$HOME/PR/sglang" \
#       --hostname linux-mi325-gpu-2 \
#       --shell
#
#   # 2. Inside the container shell, paste this (mirrors pr-test-amd.yml's
#   #    "Run diffusion server tests" step). First time only, install the
#   #    diffusion extras (~5-10 min); skip on subsequent runs:
#   pip install --cache-dir=/sgl-data/pip-cache -e /sglang-checkout/python".[dev_hip,diffusion]"
#
#   # 3. Then run the failing partition (copy the CI command verbatim):
#   cd /sglang-checkout/python
#   SGLANG_E2E_TOLERANCE=0.3 \
#   SGLANG_STAGE_TIME_TOLERANCE=0.2 \
#   SGLANG_NON_DENOISE_STAGE_TIME_TOLERANCE=0.6 \
#   SGLANG_DENOISE_STEP_TOLERANCE=0.6 \
#   SGLANG_DENOISE_AGG_TOLERANCE=0.3 \
#   SGLANG_TEST_NUM_INFERENCE_STEPS=5 \
#   AITER_JIT_DIR=/sgl-data/aiter-kernels \
#   MIOPEN_USER_DB_PATH=/sgl-data/miopen-cache \
#   HF_HUB_ENABLE_HF_TRANSFER=1 \
#   HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
#   python3 sglang/multimodal_gen/test/run_suite.py \
#       --suite 2-gpu --partition-id 2 --total-partitions 3 --continue-on-error
#
# ============================================================================
# CONCEPT REFERENCE
# ============================================================================
#
#   suite           A named group of test files (e.g. stage-b-test-small-1-gpu-amd,
#                   nightly-8-gpu-mi35x-qwen3-235b-mxfp4). Defined in
#                   sglang/test/run_suite.py. Each test file declares which suite
#                   it belongs to via register_amd_ci(suite="...") in its body.
#
#   partition       (optional, niche) CI splits *some* per-commit suites across
#                   N runners using an LPT heuristic (run_suite.py::auto_partition).
#                   On GitHub the failure surface looks like
#                   "stage-b-test-small-1-gpu-amd (linux-mi325-gpu-1, 5)" where
#                   5 is the partition id; size = workflow matrix.part length
#                   (see pr-test-amd.yml). Most nightly jobs run on a single
#                   runner with NO partition. If you only care about the failing
#                   file, use --single-file instead and ignore partitions
#                   entirely.
#
#   --single-file   Skip run_suite.py and just `python3 <FILE>` directly. This
#                   is exactly what run_suite.py does per file internally, so
#                   it's the fastest fix-and-rerun loop.
#
#   --list-only     Print which files run_suite.py would execute for the given
#                   --suite (+ --partition-id/-size if given) without running.
#                   Useful before committing to a 30-minute test run.
#
#   --shell         Drop into bash with all CI env (SGLANG_IS_IN_CI=1,
#                   SGLANG_USE_AITER=1, GPU_ARCHS=...) preset, for manual
#                   debugging.
#
#   test tree       Where test files are read from inside the container.
#                     image (default) -> /sgl-workspace/sglang/test
#                     host (--sglang-dir + --use-host-tests)
#                                     -> /sglang-checkout/test
#                   Both trees have the same registered/ layout, so a single
#                   --single-file path (e.g. registered/amd/test_xxx.py)
#                   resolves correctly under either.
#
# ============================================================================
# EXAMPLES
# ============================================================================
#
#   # 1. (most common) Rerun ONE failing test file from the image.
#   bash repro_ci.sh \
#       --docker rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260211 \
#       --single-file registered/amd/test_qwen3_instruct_mxfp4.py
#
#   # 2. Same as (1) but pick up YOUR host sglang edits.
#   bash repro_ci.sh \
#       --docker rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260211 \
#       --sglang-dir "$HOME/PR/sglang" \
#       --single-file registered/amd/test_qwen3_instruct_mxfp4.py
#
#   # 3. A whole nightly suite (1 runner, no partition).
#   bash repro_ci.sh \
#       --docker rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260211 \
#       --suite nightly-8-gpu-mi35x-qwen3-235b-mxfp4 --nightly
#
#   # 4. (niche) Reproduce ONE partition of a per-commit stage as CI ran it.
#   #    GitHub UI: "stage-b-test-small-1-gpu-amd (linux-mi325-gpu-1, 5)"
#   #    -> matrix.part: [0..13] = 14 partitions, failed part is 5.
#   bash repro_ci.sh \
#       --docker rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260211-preview \
#       --suite stage-b-test-small-1-gpu-amd \
#       --partition-id 5 --partition-size 14
#
#   # 5. Peek at which files (4) would run before committing to it.
#   bash repro_ci.sh \
#       --docker rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260211-preview \
#       --suite stage-b-test-small-1-gpu-amd \
#       --partition-id 5 --partition-size 14 --list-only
#
#   # 6. Set env up but drop into a shell instead of running tests
#   #    (e.g. to start the server manually and poke at it).
#   bash repro_ci.sh \
#       --docker rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260211 \
#       --shell
#
#   # 7. You changed sglang core / sgl-kernel / aiter and need a full rebuild
#   #    before testing. Slow (~10-30 min).
#   bash repro_ci.sh \
#       --docker rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260211 \
#       --sglang-dir "$HOME/PR/sglang" --run-install \
#       --single-file registered/amd/test_qwen3_instruct_mxfp4.py
#
#   # 8. Repro on a mi325 runner (per-commit suites usually run on mi325):
#   bash repro_ci.sh \
#       --docker rocm/sgl-dev:v0.5.8.post1-rocm700-mi30x-20260211 \
#       --hostname linux-mi325-gpu-1 \
#       --single-file registered/amd/test_xxx.py
#
#   # 9. multimodal-gen CI shard: see MULTIMODAL-GEN NOTE above (use --shell
#   #    + paste CI command; this script doesn't wrap that driver).
#
# Required:
#   --docker IMAGE         Docker image to use (no default).
#   --suite NAME           run_suite.py --suite name. Required unless --single-file.
#                          OR
#   --single-file PATH     Run only one test file (path relative to the test
#                          tree, e.g. registered/amd/test_xxx.py). Mutually
#                          exclusive with --suite.
#
# Optional:
#   --nightly              Add --nightly to run_suite.py (suite mode only).
#   --partition-id N       Auto-partition id (use with --partition-size).
#   --partition-size N     Auto-partition size (e.g. 14 for stage-b-test-small-1-gpu-amd).
#   --timeout N            run_suite.py --timeout-per-file (default: 1800;
#                          3600 in --nightly mode).
#   --hostname HOSTNAME    Spoofed container hostname for GPU arch detection
#                          via amd_ci_exec.sh:
#                            linux-mi35x-gpu-N -> gfx950 (sets GPU_ARCHS=gfx950)
#                            linux-mi325-gpu-N -> gfx942
#                          Default: linux-mi35x-gpu-8.
#   --sglang-dir PATH      Mount this host sglang checkout as /sglang-checkout
#                          (default: NOT mounted; image's /sgl-workspace/sglang
#                          is used). Pass this only if you've made local
#                          changes to sglang code or test files that you want
#                          to test against. Setting --sglang-dir auto-implies
#                          --use-host-tests.
#   --hf-home PATH         HF cache mount target -> /sgl-data/hf-cache
#                          (default: /data/huggingface if it exists, else
#                          $HOME/.cache/huggingface).
#   --use-host-tests       Run tests from /sglang-checkout/test. Requires
#                          --sglang-dir.
#   --use-image-tests      Run tests from /sgl-workspace/sglang/test (image's
#                          installed sglang) [default].
#   --run-install          Run sglang/scripts/ci/amd/amd_ci_install_dependency.sh
#                          (rebuilds sgl-kernel + aiter from the host checkout,
#                          ~10-30 min). Default skips install (image is assumed
#                          self-contained). Requires --sglang-dir. Also
#                          implies --container-name ci_sglang because the
#                          install script hardcodes that name.
#   --container-name NAME  Container name (default: ci_sglang_repro).
#   --keep-container       Keep container running after test exits [default].
#   --rm-container         Stop and remove the container at the end.
#   --shell                After environment is up, drop into an interactive
#                          bash inside the container instead of running tests.
#   --list-only            Print the file list run_suite.py would execute and
#                          exit (no actual test run). Requires --suite.
#   -h | --help            Show this help.
#
# Environment variables honored:
#   HF_TOKEN               Forwarded into the container so weight downloads work.
#
# Exit codes:
#   0   tests passed (or --shell exited cleanly, or --list-only finished)
#   2   bad args
#   *   whatever run_suite.py / single-file test returned
# ------------------------------------------------------------------------------
set -euo pipefail

log() { printf '[repro-ci] %s\n' "$*" >&2; }
die() { log "ERROR: $1"; exit "${2:-1}"; }

usage() {
  # Print the leading comment block (lines 2..first '# ---' divider after it).
  awk '
    NR == 1 { next }
    /^# -+$/ { if (seen) exit; seen=1; next }
    /^#/ { sub(/^# ?/, ""); print; next }
    { exit }
  ' "$0"
}

# ---------------------------- defaults ----------------------------
IMAGE=""
SUITE=""
SINGLE_FILE=""
NIGHTLY=0
PARTITION_ID=""
PARTITION_SIZE=""
TIMEOUT=""
HOSTNAME_SPOOF="linux-mi35x-gpu-8"
SGLANG_DIR=""           # if empty -> don't mount host sglang; use image's
HF_HOME_HOST=""
TEST_TREE=""            # auto: image if no SGLANG_DIR, host if SGLANG_DIR set
RUN_INSTALL=0
CONTAINER_NAME="ci_sglang_repro"
KEEP_CONTAINER=1
DROP_TO_SHELL=0
LIST_ONLY=0

# ---------------------------- arg parsing ----------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --docker)          IMAGE="$2"; shift 2 ;;
    --suite)           SUITE="$2"; shift 2 ;;
    --single-file)     SINGLE_FILE="$2"; shift 2 ;;
    --nightly)         NIGHTLY=1; shift ;;
    --partition-id)    PARTITION_ID="$2"; shift 2 ;;
    --partition-size)  PARTITION_SIZE="$2"; shift 2 ;;
    --timeout)         TIMEOUT="$2"; shift 2 ;;
    --hostname)        HOSTNAME_SPOOF="$2"; shift 2 ;;
    --sglang-dir)      SGLANG_DIR="$2"; shift 2 ;;
    --hf-home)         HF_HOME_HOST="$2"; shift 2 ;;
    --use-host-tests)  TEST_TREE="host"; shift ;;
    --use-image-tests) TEST_TREE="image"; shift ;;
    --run-install)     RUN_INSTALL=1; shift ;;
    --container-name)  CONTAINER_NAME="$2"; shift 2 ;;
    --keep-container)  KEEP_CONTAINER=1; shift ;;
    --rm-container)    KEEP_CONTAINER=0; shift ;;
    --shell)           DROP_TO_SHELL=1; shift ;;
    --list-only)       LIST_ONLY=1; shift ;;
    -h|--help)         usage; exit 0 ;;
    *)                 die "unknown arg '$1' (try --help)" 2 ;;
  esac
done

# ---------------------------- validate ----------------------------
[[ -z "$IMAGE" ]] && die "--docker IMAGE is required (try --help)" 2

if [[ -n "$SUITE" && -n "$SINGLE_FILE" ]]; then
  die "--suite and --single-file are mutually exclusive" 2
fi
if [[ "$LIST_ONLY" -eq 1 && -z "$SUITE" ]]; then
  die "--list-only requires --suite" 2
fi
if [[ -z "$SUITE" && -z "$SINGLE_FILE" && "$DROP_TO_SHELL" -eq 0 ]]; then
  die "one of --suite, --single-file, --shell is required" 2
fi
if [[ -n "$PARTITION_ID" && -z "$PARTITION_SIZE" ]] || \
   [[ -z "$PARTITION_ID" && -n "$PARTITION_SIZE" ]]; then
  die "--partition-id and --partition-size must be passed together" 2
fi
if [[ -n "$SINGLE_FILE" && ( -n "$PARTITION_ID" || "$NIGHTLY" -eq 1 ) ]]; then
  log "WARN: --partition-* / --nightly are ignored in --single-file mode"
fi
if [[ "$RUN_INSTALL" -eq 1 ]]; then
  [[ -z "$SGLANG_DIR" ]] && die "--run-install requires --sglang-dir (rebuilds from host checkout)" 2
  if [[ "$CONTAINER_NAME" != "ci_sglang" ]]; then
    log "NOTE: --run-install requires container name 'ci_sglang'; overriding"
    CONTAINER_NAME="ci_sglang"
  fi
fi

# Auto-pick test tree: if user mounted a host sglang, default to host tests;
# otherwise default to the image's tests. User can still override via flags.
if [[ -z "$TEST_TREE" ]]; then
  if [[ -n "$SGLANG_DIR" ]]; then TEST_TREE="host"; else TEST_TREE="image"; fi
fi
if [[ "$TEST_TREE" == "host" && -z "$SGLANG_DIR" ]]; then
  die "--use-host-tests requires --sglang-dir to mount the host checkout" 2
fi

# Validate the mount source if user asked to mount.
if [[ -n "$SGLANG_DIR" ]]; then
  [[ -d "$SGLANG_DIR" ]] || die "sglang checkout not found: $SGLANG_DIR (set --sglang-dir)"
  [[ -f "$SGLANG_DIR/test/run_suite.py" ]] || \
    die "$SGLANG_DIR doesn't look like an sglang checkout (no test/run_suite.py)"
fi

if [[ -z "$HF_HOME_HOST" ]]; then
  if [[ -d /data/huggingface ]]; then
    HF_HOME_HOST=/data/huggingface
  else
    HF_HOME_HOST="$HOME/.cache/huggingface"
  fi
fi
mkdir -p "$HF_HOME_HOST"

if [[ -z "$TIMEOUT" ]]; then
  if [[ "$NIGHTLY" -eq 1 ]]; then TIMEOUT=3600; else TIMEOUT=1800; fi
fi

case "$TEST_TREE" in
  host)  TEST_DIR="/sglang-checkout/test" ;;
  image) TEST_DIR="/sgl-workspace/sglang/test" ;;
  *)     die "internal: bad TEST_TREE=$TEST_TREE" ;;
esac

# Single-file path resolution: if user passes an absolute path, leave it; if
# it's relative, the docker exec -w TEST_DIR makes it relative to the chosen
# tree (so registered/amd/foo.py works for both tree types).

# ---------------------------- cleanup hook ----------------------------
cleanup() {
  if [[ "$KEEP_CONTAINER" -eq 1 ]]; then
    log "container '$CONTAINER_NAME' kept running for inspection"
    log "  exec into it:  docker exec -it $CONTAINER_NAME bash"
    log "  remove it:     docker rm -f $CONTAINER_NAME"
  else
    log "removing container '$CONTAINER_NAME'..."
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  fi
}

# ---------------------------- preflight log ----------------------------
log "image:           $IMAGE"
if [[ "$DROP_TO_SHELL" -eq 1 ]]; then
  log "mode:            shell (interactive)"
elif [[ "$LIST_ONLY" -eq 1 ]]; then
  log "mode:            list-only ($SUITE${NIGHTLY:+, nightly})"
elif [[ -n "$SUITE" ]]; then
  log "mode:            suite ($SUITE${NIGHTLY:+, nightly})"
  if [[ -n "$PARTITION_ID" ]]; then
    log "partition:       $PARTITION_ID / $PARTITION_SIZE"
  fi
  log "timeout/file:    ${TIMEOUT}s"
elif [[ -n "$SINGLE_FILE" ]]; then
  log "mode:            single-file ($SINGLE_FILE)"
fi
if [[ -n "$SGLANG_DIR" ]]; then
  log "sglang checkout: $SGLANG_DIR  -> /sglang-checkout (host mount)"
else
  log "sglang checkout: (using image's /sgl-workspace/sglang, no host mount)"
fi
log "HF cache:        $HF_HOME_HOST    -> /sgl-data/hf-cache"
log "hostname spoof:  $HOSTNAME_SPOOF"
log "test tree:       $TEST_TREE ($TEST_DIR)"
log "container name:  $CONTAINER_NAME (keep=$KEEP_CONTAINER)"

# ---------------------------- launch container ----------------------------
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

# Mount every DRI render node so the container sees all GPUs.
device_flags=(--device=/dev/kfd)
shopt -s nullglob
for d in /dev/dri/renderD* /dev/dri/card*; do
  [[ -e "$d" ]] && device_flags+=(--device "$d")
done
shopt -u nullglob

mount_args=(-v "$HF_HOME_HOST:/sgl-data/hf-cache")
if [[ -n "$SGLANG_DIR" ]]; then
  mount_args+=(-v "$SGLANG_DIR:/sglang-checkout")
  workdir="/sglang-checkout"
else
  workdir="/sgl-workspace/sglang"
fi

log "starting container..."
docker run -dt --user root \
  "${device_flags[@]}" \
  --ulimit nofile=65536:65536 \
  "${mount_args[@]}" \
  --group-add video \
  --shm-size 32g \
  --cap-add=SYS_PTRACE \
  --hostname "$HOSTNAME_SPOOF" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HF_HOME=/sgl-data/hf-cache \
  -e HF_HUB_ETAG_TIMEOUT=300 \
  -e HF_HUB_DOWNLOAD_TIMEOUT=300 \
  -e MIOPEN_USER_DB_PATH=/sgl-data/hf-cache/miopen \
  -e MIOPEN_CUSTOM_CACHE_DIR=/sgl-data/hf-cache/miopen \
  -e PYTHONPATH="/opt/tilelang:" \
  --security-opt seccomp=unconfined \
  -w "$workdir" \
  --name "$CONTAINER_NAME" \
  "$IMAGE" >/dev/null

# ---------------------------- sanity check ----------------------------
log "container '$CONTAINER_NAME' running; sanity-checking environment"
docker exec "$CONTAINER_NAME" bash -lc '
  set -e
  echo "=== ROCm ===";    cat /opt/rocm/.info/version 2>/dev/null || echo "(no /opt/rocm/.info/version)"
  echo "=== torch ===";   python3 -c "import torch; print(torch.__version__, \"hip:\", torch.version.hip, \"devs:\", torch.cuda.device_count())" || true
  echo "=== aiter ===";   pip show amd-aiter 2>/dev/null | grep ^Version || echo "(amd-aiter not installed)"
  echo "=== hostname ==="; hostname
'

# ---------------------------- optional install ----------------------------
if [[ "$RUN_INSTALL" -eq 1 ]]; then
  log "running amd_ci_install_dependency.sh (rebuilds sgl-kernel + aiter; ~10-30 min)"
  bash "$SGLANG_DIR/scripts/ci/amd/amd_ci_install_dependency.sh"
fi

# ---------------------------- common ENV for test runs ----------------------------
# Mirror amd_ci_exec.sh's ENV_MAP. We don't actually call amd_ci_exec.sh
# because it hardcodes the container name 'ci_sglang' and we want the user
# to be able to use a different container name.
common_env=(
  -e SGLANG_IS_IN_CI=1
  -e SGLANG_IS_IN_CI_AMD=1
  -e SGLANG_USE_AITER=1
)
if [[ "$HOSTNAME_SPOOF" =~ mi35 ]]; then
  common_env+=(-e GPU_ARCHS=gfx950)
fi

# ---------------------------- shell mode ----------------------------
if [[ "$DROP_TO_SHELL" -eq 1 ]]; then
  log "entering interactive shell. CI env (SGLANG_IS_IN_CI=1, SGLANG_USE_AITER=1)"
  log "  is set. test tree: $TEST_DIR. exit shell to return."
  set +e
  docker exec -it -w "$TEST_DIR" "${common_env[@]}" "$CONTAINER_NAME" bash
  exit_code=$?
  set -e
  cleanup
  exit "$exit_code"
fi

# ---------------------------- list-only mode ----------------------------
if [[ "$LIST_ONLY" -eq 1 ]]; then
  log "listing files run_suite.py would execute (no test run):"
  # Use a heredoc so we don't have to escape complex quotes inside python -c.
  # Variables passed as env so we don't have to interpolate into python source.
  set +e
  docker exec -w "$TEST_DIR" \
    -e LIST_SUITE="$SUITE" \
    -e LIST_NIGHTLY="$NIGHTLY" \
    -e LIST_PART_ID="${PARTITION_ID:-}" \
    -e LIST_PART_SIZE="${PARTITION_SIZE:-}" \
    "${common_env[@]}" \
    "$CONTAINER_NAME" python3 - <<'PYEOF'
import glob, os
from sglang.test.ci.ci_register import collect_tests, HWBackend
from run_suite import auto_partition  # cwd is /sglang-checkout/test

suite = os.environ["LIST_SUITE"]
nightly = os.environ["LIST_NIGHTLY"] == "1"
part_id = os.environ.get("LIST_PART_ID") or ""
part_size = os.environ.get("LIST_PART_SIZE") or ""

files = [
    f for f in glob.glob("registered/**/*.py", recursive=True)
    if not f.endswith("/conftest.py") and not f.endswith("/__init__.py")
]
tests = collect_tests(files, sanity_check=True)
matching = [
    t for t in tests
    if t.backend == HWBackend.AMD and t.suite == suite and t.nightly == nightly
]

if part_id and part_size:
    matching = auto_partition(matching, int(part_id), int(part_size))

matching.sort(key=lambda t: (-t.est_time, t.filename))
total = sum(t.est_time for t in matching)
header = f"hw=amd suite={suite} nightly={nightly}"
if part_id:
    header += f" partition={part_id}/{part_size}"
print(f"Found {len(matching)} test(s) for {header} (est total {total:.0f}s):")
for t in matching:
    flag = " [DISABLED]" if t.disabled else ""
    print(f"  {t.est_time:6.0f}s  {t.filename}{flag}")
PYEOF
  exit_code=$?
  set -e
  cleanup
  exit "$exit_code"
fi

# ---------------------------- single-file mode ----------------------------
if [[ -n "$SINGLE_FILE" ]]; then
  log "running single test file: $SINGLE_FILE"
  log "  this is exactly what run_suite.py invokes per file (subprocess.Popen([python3, FILE]))"
  set +e
  docker exec -w "$TEST_DIR" "${common_env[@]}" \
    "$CONTAINER_NAME" python3 "$SINGLE_FILE"
  exit_code=$?
  set -e
  log "single-file exit code: $exit_code"
  cleanup
  exit "$exit_code"
fi

# ---------------------------- suite mode ----------------------------
log "running suite locally: $SUITE"
log "  this is the full repro of the GitHub Actions stage that runs"
log "  'python3 run_suite.py --hw amd --suite $SUITE ...'"

run_suite_args=(--hw amd --suite "$SUITE" --timeout-per-file "$TIMEOUT")
[[ "$NIGHTLY" -eq 1 ]] && run_suite_args+=(--nightly)
if [[ -n "$PARTITION_ID" ]]; then
  run_suite_args+=(--auto-partition-id "$PARTITION_ID" --auto-partition-size "$PARTITION_SIZE")
fi

set +e
docker exec -w "$TEST_DIR" "${common_env[@]}" \
  "$CONTAINER_NAME" python3 run_suite.py "${run_suite_args[@]}"
exit_code=$?
set -e

log "suite exit code: $exit_code"
cleanup
exit "$exit_code"
