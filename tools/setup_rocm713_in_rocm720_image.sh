#!/usr/bin/env bash
# ============================================================================
# setup_rocm713_in_rocm720_image.sh
#
# Install TheRock ROCm 7.13 (pip) on top of `rocm/sgl-dev:*-rocm720-mi35x-*`
# docker image and rebuild the C++ extensions whose torch/HIP ABI broke with
# the 7.2 -> 7.13 upgrade (aiter, sgl-kernel, fast-hadamard-transform).
#
# Why this script exists (lessons learned in 0504 setup):
#   1. AMDGPU kernel driver in this image (amdgpu 6.16.13) is forward-compat
#      with ROCm 7.13 user-space (validated by `rocm-sdk test` 26/26 OK).
#   2. The torch 2.11+rocm7.13 wheel's libtorch_hip.so has RPATH baked to
#      /opt/rocm/lib (= 7.2 in this image). Without LD_LIBRARY_PATH override
#      torch loads 7.2 libs and `device_count() == 0`. Fix: prepend bundled
#      `_rocm_sdk_devel/lib` to LD_LIBRARY_PATH.
#   3. ROCm 7.13 headers use `__bf16` (Clang 15+ / GCC 13+ only). System g++
#      11.4 chokes. Fix: CXX/CC -> bundled `_rocm_sdk_devel/lib/llvm/bin/clang`
#      (AMD clang 23).
#   4. tilelang JIT + sgl-kernel build call `ld.lld`. Bundle's ld.lld is a
#      26KB launcher wrapper that fails to find its `amdllvm` exec helper at
#      runtime. Fix: replace with symlink to the real `lld` binary in same
#      directory (lld auto-detects GNU flavor via argv[0]).
#   5. tilelang/aiter binaries link against new torch ABI. aiter's compiled
#      .so files reference symbols like
#      `c10::hip::getCurrentHIPStream(signed char)` whose signature changed
#      between torch 2.9 and 2.11. Fix: rebuild all three extensions.
#
# Usage:
#   # Inside the container (recommended):
#   bash ~/SGLang-benchmarks/tools/setup_rocm713_in_rocm720_image.sh
#
#   # From host:
#   docker exec -it <container> bash \
#     ~/SGLang-benchmarks/tools/setup_rocm713_in_rocm720_image.sh
#
# Optional env vars:
#   GFX_TARGET     gfx950 (default) | gfx942
#   PREBUILD_AITER 1 (default) — pre-build aiter MoE kernels (5-15 min)
#
# After the script:
#   - Source the printed `eval "$(...)"` snippet OR re-source ~/.bashrc to
#     pick up the persistent env (LD_LIBRARY_PATH, ROCM_PATH, PATH, CXX, CC).
#   - Switch to your chosen sglang branch (e.g. dual-stream branch) and run
#     ./GLM.sh as usual.
# ============================================================================

set -euo pipefail

# -------- Config ------------------------------------------------------------
GFX_TARGET="${GFX_TARGET:-gfx950}"
PREBUILD_AITER="${PREBUILD_AITER:-1}"
INDEX_URL="https://rocm.nightlies.amd.com/v2/${GFX_TARGET}-dcgpu/"

SDK_DEVEL="/opt/venv/lib/python3.10/site-packages/_rocm_sdk_devel"
SDK_CORE="/opt/venv/lib/python3.10/site-packages/_rocm_sdk_core"

SGL_ROOT="/sgl-workspace/sglang"
AITER_ROOT="/sgl-workspace/aiter"
FHT_ROOT="/sgl-workspace/fast-hadamard-transform"

# -------- Helpers -----------------------------------------------------------
log() { printf "\n\033[1;36m=== %s ===\033[0m\n" "$*"; }
ok()  { printf "  \033[32m[OK]\033[0m %s\n" "$*"; }
warn(){ printf "  \033[33m[WARN]\033[0m %s\n" "$*"; }
die() { printf "  \033[31m[FAIL]\033[0m %s\n" "$*" >&2; exit 1; }

require_root() {
    [ "$(id -u)" -eq 0 ] || die "Must run as root (inside container)."
}

ensure_in_container() {
    [ -d /sgl-workspace ] || die "/sgl-workspace not found. Run inside the docker container."
    [ -d /opt/venv ] || die "/opt/venv not found. This script assumes the rocm/sgl-dev image layout."
}

detect_image_tag() {
    docker_tag=""
    if [ -f /etc/sglang_image_info ]; then
        docker_tag=$(grep -oE 'rocm[0-9-]+mi35x[0-9-]+' /etc/sglang_image_info 2>/dev/null || true)
    fi
    if [ -z "$docker_tag" ]; then
        # fall back: check torch baked-in HIP version (image's original is 7.2)
        docker_tag=$(python3 -c 'import torch; print(torch.version.hip)' 2>/dev/null || echo "unknown")
    fi
    echo "$docker_tag"
}

# -------- Phase 1: pip install ROCm 7.13 stack ------------------------------
phase1_pip_install() {
    log "Phase 1: pip install TheRock ROCm 7.13 stack ($INDEX_URL)"

    pip install --break-system-packages --force-reinstall \
        --index-url "$INDEX_URL" \
        torch torchvision

    pip install --break-system-packages \
        --index-url "$INDEX_URL" \
        "rocm[devel]"

    ok "pip install done"
}

# -------- Phase 2: env vars -------------------------------------------------
write_env() {
    log "Phase 2: persistent env vars (writing to ~/.bashrc)"

    if grep -q "TheRock pip ROCm 7.13" /root/.bashrc 2>/dev/null; then
        warn "~/.bashrc already has TheRock env block; skipping write"
    else
        cat >> /root/.bashrc <<EOF

# === TheRock pip ROCm 7.13 (managed by setup_rocm713_in_rocm720_image.sh) ===
export ROCM_PATH=$SDK_DEVEL
export ROCM_HOME=$SDK_DEVEL
export HIP_PATH=$SDK_DEVEL
export PATH=$SDK_DEVEL/lib/llvm/bin:$SDK_DEVEL/bin:\$PATH
export LD_LIBRARY_PATH=$SDK_DEVEL/lib:$SDK_CORE/lib:\${LD_LIBRARY_PATH:-}
export CXX=$SDK_DEVEL/lib/llvm/bin/clang++
export CC=$SDK_DEVEL/lib/llvm/bin/clang
EOF
        ok "wrote env block to /root/.bashrc"
    fi

    # also export in current shell
    export ROCM_PATH="$SDK_DEVEL"
    export ROCM_HOME="$SDK_DEVEL"
    export HIP_PATH="$SDK_DEVEL"
    export PATH="$SDK_DEVEL/lib/llvm/bin:$SDK_DEVEL/bin:$PATH"
    export LD_LIBRARY_PATH="$SDK_DEVEL/lib:$SDK_CORE/lib:${LD_LIBRARY_PATH:-}"
    export CXX="$SDK_DEVEL/lib/llvm/bin/clang++"
    export CC="$SDK_DEVEL/lib/llvm/bin/clang"

    # sanity
    python3 -c '
import torch
print(f"  torch        : {torch.__version__}")
print(f"  hip          : {torch.version.hip}")
print(f"  device_count : {torch.cuda.device_count()}")
assert torch.cuda.device_count() == 8, f"Expected 8 GPUs, got {torch.cuda.device_count()}"
print(f"  gfx          : {torch.cuda.get_device_properties(0).gcnArchName}")
'
    ok "torch sees all 8 GPUs"
}

# -------- Phase 3: fix ld.lld wrapper ---------------------------------------
phase3_fix_ld_lld() {
    log "Phase 3: replace broken ld.lld wrapper with symlink to lld"

    local llvm_bin="$SDK_DEVEL/lib/llvm/bin"
    local ldlld="$llvm_bin/ld.lld"

    if [ ! -e "$ldlld" ]; then
        die "ld.lld not found at $ldlld"
    fi

    if [ -L "$ldlld" ]; then
        ok "ld.lld already a symlink, skipping"
        return 0
    fi

    # original is a 26KB ELF launcher; real lld is ~8MB
    local size; size=$(stat -c '%s' "$ldlld")
    if [ "$size" -lt 100000 ]; then
        mv "$ldlld" "${ldlld}.broken_wrapper.$(date +%s)"
        ln -s "$llvm_bin/lld" "$ldlld"
        ok "ld.lld -> lld symlink installed"
    else
        warn "ld.lld is larger than 100KB ($size); may already be patched"
    fi

    # verify
    "$ldlld" --version | head -1 || die "ld.lld still broken after fix"
    ok "ld.lld --version works"
}

# -------- Phase 4: rebuild aiter --------------------------------------------
phase4_rebuild_aiter() {
    log "Phase 4: rebuild aiter (torch 2.9 -> 2.11 ABI break)"

    cd "$AITER_ROOT"
    rm -rf aiter/jit/build/ aiter/jit/*.so build/

    if [ "$PREBUILD_AITER" = "1" ]; then
        PREBUILD_KERNELS=1 GPU_ARCHS="$GFX_TARGET" python3 setup.py develop \
            2>&1 | tee /tmp/aiter_rebuild.log | tail -10
    else
        GPU_ARCHS="$GFX_TARGET" python3 setup.py develop \
            2>&1 | tee /tmp/aiter_rebuild.log | tail -10
    fi

    # numerics check
    python3 - <<'PY'
import torch, aiter
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
w = torch.ones(1024, device="cuda", dtype=torch.bfloat16)
y = aiter.rms_norm(x, w, 1e-6)
torch.cuda.synchronize()
ref = torch.nn.functional.rms_norm(x, [1024], w, 1e-6)
diff = (y - ref).abs().max().item()
assert diff < 0.01, f"FAIL: diff={diff}"
print(f"  aiter.rms_norm OK, max diff vs torch ref = {diff:.6f}")
PY
    ok "aiter rebuilt and verified"
}

# -------- Phase 5: rebuild sgl-kernel (ROCm path uses setup_rocm.py) --------
phase5_rebuild_sgl_kernel() {
    log "Phase 5: rebuild sgl-kernel via setup_rocm.py"

    pip uninstall --break-system-packages -y \
        sgl-kernel sglang-kernel sgl_kernel 2>&1 | tail -3 || true

    cd "$SGL_ROOT/sgl-kernel"
    rm -rf build/ dist/ *.egg-info/

    AMDGPU_TARGET="$GFX_TARGET" python3 setup_rocm.py build_ext \
        2>&1 | tee /tmp/sgl_kernel_build.log | tail -10

    # build_ext only builds; manually drop .so + python sources into site-packages
    local site="/opt/venv/lib/python3.10/site-packages/sgl_kernel"
    mkdir -p "$site"
    cp build/lib.*/sgl_kernel/common_ops*.so "$site/"
    cp -r python/sgl_kernel/* "$site/" 2>/dev/null || true

    python3 -c "import sgl_kernel; print(f'  sgl_kernel OK at {sgl_kernel.__file__}')"
    ok "sgl-kernel rebuilt"
}

# -------- Phase 6: rebuild fast-hadamard-transform --------------------------
phase6_rebuild_fast_hadamard() {
    log "Phase 6: rebuild fast-hadamard-transform"

    pip uninstall --break-system-packages -y fast-hadamard-transform \
        2>&1 | tail -3 || true

    cd "$FHT_ROOT"
    rm -rf build/ dist/ *.egg-info/

    pip install --break-system-packages --no-build-isolation -e . \
        2>&1 | tee /tmp/fast_hadamard_build.log | tail -10

    python3 - <<'PY'
import torch, fast_hadamard_transform
x = torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16)
y = fast_hadamard_transform.hadamard_transform(x)
torch.cuda.synchronize()
print(f"  fast_hadamard_transform OK, shape={y.shape}")
PY
    ok "fast-hadamard-transform rebuilt"
}

# -------- Phase 7: final verification ---------------------------------------
phase7_verify_all() {
    log "Phase 7: final integration check"

    python3 - <<'PY'
import torch
print(f"  torch    : {torch.__version__}")
print(f"  hip      : {torch.version.hip}")
print(f"  count    : {torch.cuda.device_count()}")
print(f"  gfx      : {torch.cuda.get_device_properties(0).gcnArchName}")

import aiter
print(f"  aiter    : import OK")

import sgl_kernel
print(f"  sgl_kernel : {sgl_kernel.__file__}")

import fast_hadamard_transform
print(f"  fast_hadamard_transform : {fast_hadamard_transform.__file__}")

# heavy smoke test (matmul + aiter rms + hadamard)
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()

w = torch.ones(2048, device="cuda", dtype=torch.bfloat16)
y2 = aiter.rms_norm(x, w, 1e-6)
torch.cuda.synchronize()

z = torch.randn(8, 1024, device="cuda", dtype=torch.bfloat16)
z2 = fast_hadamard_transform.hadamard_transform(z)
torch.cuda.synchronize()

print("  smoke    : matmul + aiter.rms_norm + hadamard_transform all OK")
PY
    ok "ALL ROCm 7.13 EXTENSIONS WORKING"
}

# ============================================================================
# main
# ============================================================================
main() {
    require_root
    ensure_in_container

    log "Detected image: $(detect_image_tag)"
    log "Target GFX    : $GFX_TARGET"
    log "Index URL     : $INDEX_URL"

    phase1_pip_install
    write_env
    phase3_fix_ld_lld
    phase4_rebuild_aiter
    phase5_rebuild_sgl_kernel
    phase6_rebuild_fast_hadamard
    phase7_verify_all

    log "Setup complete!"
    cat <<'EOF'

  Next steps:
  -----------
  1. Re-source env so subsequent shells inherit it:
       source ~/.bashrc

  2. (Optional) Switch to dual-stream sglang branch:
       cd /sgl-workspace/sglang
       git remote add fork https://github.com/Jacob0226/sglang.git 2>/dev/null || true
       git fetch fork jacob/glm5-rocm-nsa-on-thomas
       git checkout -B jacob/glm5-rocm-nsa-on-thomas \
           fork/jacob/glm5-rocm-nsa-on-thomas

  3. Run benchmark / profile:
       cd ~/SGLang-benchmarks
       ./GLM.sh \
         --model /data/huggingface/hub/zai-org/GLM-5.1-FP8 \
         --docker rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260503 \
         --tag 0504_DualStream_NewHIP \
         --dual-stream-rocm --prof
EOF
}

main "$@"
