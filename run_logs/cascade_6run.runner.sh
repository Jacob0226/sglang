#!/bin/bash
# Auto-generated runner for the 6-cascade-run sequence.
#
# IMPORTANT inside-container preconditions (set by docker exec wrapper):
#   --user 10055 (jacchang) — files land on /home/jacchang bind mount
#   -e HOME=/home/jacchang   — cascade_dsr1.sh's $HOME-derived LOG_DIR is right
#   /etc/passwd has jacchang — getpass.getuser() doesn't crash
# This script then activates /opt/venv (system /usr/bin/python3 doesn't have sglang).
#
# Each call to cascade_dsr1.sh has its own internal set -e; we explicitly
# do NOT exit-on-error here so a single mid-sequence failure (OOM, hung
# warmup, scheduler crash) doesn't kill the remaining runs.

set +e
. /opt/venv/bin/activate
cd /home/jacchang/SGLang-benchmarks

DOCKER_TAG="rocm/sgl-dev:v0.5.11-rocm720-mi35x-20260507"

# Aggressive cleanup between runs. cascade_dsr1.sh's own trap only does
# `pkill -f sglang.launch_server`, which misses the renamed scheduler
# subprocesses (sglang::schedul, sglang::detoken) — those keep ~230 GB
# of HBM held per GPU and starve the next run's weight load. Without
# this, run N+1 hits a half-allocated GPU and SGLang's post-startup
# warmup hangs silently with /health stuck at 503.
hard_cleanup_between_runs() {
  echo ">>> [cleanup] killing leftover sglang/python processes"
  pkill -9 -f 'sglang' 2>/dev/null
  pkill -9 python 2>/dev/null
  sleep 5
  pgrep -fa 'sglang|cascade_dsr1' || echo "  (no leftover processes)"

  echo ">>> [cleanup] waiting for GPU memory to drop below 2 GB / GPU (max 60s)"
  for i in $(seq 1 12); do
    used_max=$(rocm-smi --showmeminfo vram 2>/dev/null \
                 | awk '/VRAM Total Used Memory/ {n=$NF+0; if (n>m) m=n} END {print m+0}')
    if [ "${used_max:-0}" -lt 2147483648 ]; then  # 2 GiB
      printf "  GPU clean after %d sec (max usage %s bytes)\n" $((i*5)) "$used_max"
      break
    fi
    sleep 5
  done
  rocm-smi --showmeminfo vram 2>/dev/null \
    | awk '/VRAM Total Used Memory/ {printf "  GPU%d used = %.1f GB\n", c++, $NF/1024/1024/1024}'
}

run_one() {
  local n="$1" mode="$2" tag="$3"; shift 3
  echo
  echo "============================================================"
  echo "RUN $n/6: cache_mode=$mode tag=$tag extra='$*' @ $(date '+%F %T')"
  echo "============================================================"
  ./cascade_dsr1.sh --tag "$tag" --cache-mode "$mode" --gsm8k-precheck "$@" --docker "$DOCKER_TAG"
  local rc=$?
  echo "RUN $n/6 finished (rc=$rc) @ $(date '+%F %T')"
  hard_cleanup_between_runs
}

echo "============================================================"
echo "RUNNER ENV (sanity check before kicking off the 6 runs)"
echo "============================================================"
echo "  USER  = $(whoami)  (uid=$(id -u))"
echo "  HOME  = $HOME"
echo "  CWD   = $(pwd)"
echo "  python= $(which python3)"
python3 -c "import sglang; print('  sglang  =', sglang.__file__)"
hard_cleanup_between_runs   # also do an initial cleanup before the first run
echo

echo "============================================================"
echo "CASCADE 6-RUN START @ $(date '+%F %T')"
echo "============================================================"

# Batch A: chunked-prefill-size = 65536 (cascade_dsr1.sh default)
run_one 1 L1      MI355X_15rounds
run_one 2 L2      MI355X_15rounds
run_one 3 L3_file MI355X_15rounds

# Batch B: chunked-prefill-size = 131072 (override)
run_one 4 L1      MI355X_15rounds_cps131k --chunked-prefill-size 131072
run_one 5 L2      MI355X_15rounds_cps131k --chunked-prefill-size 131072
run_one 6 L3_file MI355X_15rounds_cps131k --chunked-prefill-size 131072

echo
echo "============================================================"
echo "ALL 6 RUNS DONE @ $(date '+%F %T')"
echo "============================================================"
