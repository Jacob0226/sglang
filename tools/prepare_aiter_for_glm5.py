#!/usr/bin/env python3
"""
One-stop aiter prep for SGLang GLM-5 / GLM-5.1 stack on MI355X.

Two known issues need to be fixed after a fresh container (`rocm/sgl-dev:*`)
is launched and the SGL + aiter PR stack is merged in:

  1) check_args strict-isinstance bug
     `check_args()` runs AFTER develop=True ops convert torch.Tensor
     -> aiter_tensor_t, but still strict-checks isinstance(arg,
     torch.Tensor) and raises TypeError. Triggered by SGL PR #23562
     calling indexer_k_quant_and_cache.

  2) Stale prebuilt .so files missing _set_current_hip_stream
     Pre-baked .so artifacts in the docker image (e.g. module_cache,
     module_moe_sorting_opus) were compiled before AITER_SET_STREAM_PYBIND
     was added to their pybind sources (PR #2953 etc.). Triggered as
     `AttributeError: module 'aiter.jit.module_X' has no attribute
     '_set_current_hip_stream'`.

This script:
  (a) Applies the in-place check_args patch via `patch_aiter_check_args.py`.
  (b) For every module whose pybind source has `AITER_SET_STREAM_PYBIND`
      but whose installed .so doesn't expose `_set_current_hip_stream`,
      removes the stale .so + build cache and triggers a JIT rebuild.

Idempotent: rerunning is a fast no-op once both fixes are in place.

Usage (inside container):

    python3 ~/SGLang-benchmarks/tools/prepare_aiter_for_glm5.py

Optional flags:
    --aiter-root /sgl-workspace/aiter   # override aiter root
    --skip-patch                         # only do (b)
    --skip-rebuild                       # only do (a)
"""
from __future__ import annotations

import argparse
import glob
import importlib
import os
import pathlib
import shutil
import subprocess
import sys
import time

DEFAULT_AITER_ROOT = "/sgl-workspace/aiter"


def _run_patch_script(aiter_root: str) -> int:
    """Apply the check_args isinstance patch (idempotent)."""
    here = pathlib.Path(__file__).resolve().parent
    patch_script = here / "patch_aiter_check_args.py"
    if not patch_script.exists():
        print(f"[prepare] WARN: {patch_script} not found, skipping patch step")
        return 0
    core_py = pathlib.Path(aiter_root) / "aiter/jit/core.py"
    return subprocess.call([sys.executable, str(patch_script), "--core-py", str(core_py)])


def _find_stale_modules(aiter_root: str) -> list[str]:
    """Return module names whose pybind source has AITER_SET_STREAM_PYBIND
    but whose installed .so doesn't expose _set_current_hip_stream."""
    pybind_dir = pathlib.Path(aiter_root) / "csrc/pybind"
    src_with_macro: set[str] = set()
    for src in pybind_dir.glob("*.cu"):
        try:
            if "AITER_SET_STREAM_PYBIND" in src.read_text():
                src_with_macro.add(src.name.replace("_pybind.cu", ""))
        except OSError:
            pass

    so_dir = pathlib.Path(aiter_root) / "aiter/jit"
    candidates = sorted(
        os.path.splitext(p.name)[0]
        for p in so_dir.glob("module_*.so")
        if p.name.replace(".so", "").replace("module_", "") in src_with_macro
    )

    # Probe each candidate via Python import
    sys.path.insert(0, str(aiter_root))
    stale = []
    for m in candidates:
        try:
            mod = importlib.import_module(f"aiter.jit.{m}")
            if not hasattr(mod, "_set_current_hip_stream"):
                stale.append(m)
        except Exception as e:
            print(f"[prepare] {m}: import failed ({e}), will rebuild")
            stale.append(m)
    sys.path.pop(0)
    return stale


def _rebuild_module(aiter_root: str, md_name: str) -> None:
    """Remove stale artifacts and trigger JIT rebuild."""
    aiter_root_p = pathlib.Path(aiter_root)
    so_path = aiter_root_p / "aiter/jit" / f"{md_name}.so"
    bd_path = aiter_root_p / "aiter/jit/build" / md_name
    if so_path.exists():
        so_path.unlink()
        print(f"[prepare]   removed {so_path}")
    if bd_path.exists():
        shutil.rmtree(bd_path)
        print(f"[prepare]   removed {bd_path}")

    sys.path.insert(0, str(aiter_root))
    try:
        from aiter.jit import core  # type: ignore
        d = core.get_args_of_build(md_name)
        t0 = time.time()
        core.build_module(
            md_name,
            d["srcs"],
            d["flags_extra_cc"],
            d["flags_extra_hip"],
            d["blob_gen_cmd"],
            d["extra_include"],
            d["extra_ldflags"],
            d["verbose"],
            d["is_python_module"],
            d["is_standalone"],
            d["torch_exclude"],
            d.get("third_party", []),
        )
        print(f"[prepare]   built {md_name} in {time.time() - t0:.1f}s")
    finally:
        sys.path.pop(0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aiter-root", default=DEFAULT_AITER_ROOT)
    ap.add_argument("--skip-patch", action="store_true")
    ap.add_argument("--skip-rebuild", action="store_true")
    args = ap.parse_args()

    if not pathlib.Path(args.aiter_root).is_dir():
        print(f"[prepare] ERROR: aiter root not found: {args.aiter_root}", file=sys.stderr)
        return 2

    rc = 0
    if not args.skip_patch:
        print("[prepare] === Step 1/2: patching check_args ===")
        rc = _run_patch_script(args.aiter_root)
        if rc != 0:
            print(f"[prepare] patch step failed (rc={rc})", file=sys.stderr)
            return rc

    if not args.skip_rebuild:
        print("[prepare] === Step 2/2: detecting stale .so files ===")
        stale = _find_stale_modules(args.aiter_root)
        if not stale:
            print("[prepare] all modules have _set_current_hip_stream — nothing to rebuild")
        else:
            print(f"[prepare] stale modules to rebuild: {stale}")
            for m in stale:
                print(f"[prepare] rebuilding {m} ...")
                _rebuild_module(args.aiter_root, m)
            # Re-verify
            still_stale = _find_stale_modules(args.aiter_root)
            if still_stale:
                print(f"[prepare] WARNING: still stale after rebuild: {still_stale}", file=sys.stderr)
                rc = 1
            else:
                print("[prepare] all stale modules rebuilt successfully")

    print("[prepare] done.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
