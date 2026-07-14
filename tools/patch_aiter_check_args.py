#!/usr/bin/env python3
"""
Patch aiter's check_args to accept aiter_tensor_t where torch.Tensor is annotated.

Background
----------
Ops in `aiter/ops/cache.py` decorated with `@compile_ops(..., develop=True)`
(e.g. `indexer_k_quant_and_cache`, `cp_gather_indexer_k_quant_cache`,
`fused_qk_rope_concat_and_cache_mla`) have their `torch.Tensor` args auto-
converted to `aiter_tensor_t` *before* `check_args()` runs (see core.py
~line 1610: "develop=True: torch.Tensor -> pybind aiter_tensor_t before C++").

But `check_args()` then does a strict `isinstance(arg, torch.Tensor)` and
raises:

    TypeError: indexer_k_quant_and_cache: k needs to be <class 'torch.Tensor'>
               but got <class 'aiter.jit.module_aiter_core.aiter_tensor_t'>

This bug exists in upstream `ROCm/aiter` `main` (verified at d35d35b40,
2026-05). It only fires once a downstream caller (here: sgl-project/sglang
PR #23562 "preshuffle paged MQA + page_size=64" → calls
`indexer_k_quant_and_cache(key, kv_cache, ..., preshuffle=True)`) actually
exercises a develop=True op.

This patch adds two helpers (`_matches`, `_matches_tuple`) inside
`check_args` that treat `aiter_tensor_t` as a valid stand-in for
`torch.Tensor`, used in both the single-type (`origin is None`) and
Optional/Union branches.

Usage
-----
Inside the container:

    python3 ~/SGLang-benchmarks/tools/patch_aiter_check_args.py

Or pin a custom path:

    python3 ~/SGLang-benchmarks/tools/patch_aiter_check_args.py \
        --core-py /sgl-workspace/aiter/aiter/jit/core.py

Idempotent: running twice is a no-op. Exits 0 on success / already-patched,
non-zero on failure (e.g. target block not found because upstream changed).
"""
from __future__ import annotations

import argparse
import pathlib
import sys

DEFAULT_CORE_PY = "/sgl-workspace/aiter/aiter/jit/core.py"

OLD = """                            if origin is None:
                                if not isinstance(arg, expected_type) and not (
                                    any(el in str(expected_type) for el in enum_types)
                                    and isinstance(arg, int)
                                ):
                                    raise TypeError(
                                        f\"{loadName}: {el} needs to be {expected_type} but got {got_type}\"
                                    )
                            elif origin is list:
                                if not isinstance(arg, list):
                                    raise TypeError(
                                        f\"{loadName}: {el} needs to be List[{sub_t}] but got {arg}\"
                                    )
                            elif origin is typing.Union or origin is types.UnionType:
                                if arg is not None and not isinstance(arg, sub_t):
                                    raise TypeError(
                                        f\"{loadName}: {el} needs to be Optional[{sub_t}] but got {arg}\"
                                    )"""

NEW = """                            # PATCH: in develop=True mode, args are converted torch.Tensor->aiter_tensor_t
                            # BEFORE check_args runs. Treat them as interchangeable here.
                            def _matches(arg, expected):
                                if isinstance(arg, expected):
                                    return True
                                if expected is torch.Tensor and aiter_tensor_t is not object and isinstance(arg, aiter_tensor_t):
                                    return True
                                return False
                            def _matches_tuple(arg, types_tuple):
                                if isinstance(arg, types_tuple):
                                    return True
                                if torch.Tensor in types_tuple and aiter_tensor_t is not object and isinstance(arg, aiter_tensor_t):
                                    return True
                                return False
                            if origin is None:
                                if not _matches(arg, expected_type) and not (
                                    any(el in str(expected_type) for el in enum_types)
                                    and isinstance(arg, int)
                                ):
                                    raise TypeError(
                                        f\"{loadName}: {el} needs to be {expected_type} but got {got_type}\"
                                    )
                            elif origin is list:
                                if not isinstance(arg, list):
                                    raise TypeError(
                                        f\"{loadName}: {el} needs to be List[{sub_t}] but got {arg}\"
                                    )
                            elif origin is typing.Union or origin is types.UnionType:
                                if arg is not None and not _matches_tuple(arg, sub_t):
                                    raise TypeError(
                                        f\"{loadName}: {el} needs to be Optional[{sub_t}] but got {arg}\"
                                    )"""

MARKER = "PATCH: in develop=True mode, args are converted torch.Tensor->aiter_tensor_t"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core-py", default=DEFAULT_CORE_PY,
                    help=f"Path to aiter/jit/core.py (default: {DEFAULT_CORE_PY})")
    args = ap.parse_args()

    path = pathlib.Path(args.core_py)
    if not path.is_file():
        print(f"[patch_aiter_check_args] ERROR: not a file: {path}", file=sys.stderr)
        return 2

    src = path.read_text()
    if MARKER in src:
        print(f"[patch_aiter_check_args] already patched: {path}")
        return 0

    if OLD not in src:
        print(f"[patch_aiter_check_args] ERROR: target block not found in {path}.", file=sys.stderr)
        print("  Upstream aiter likely changed check_args. Re-derive the patch:", file=sys.stderr)
        print("    grep -n 'def check_args' aiter/jit/core.py", file=sys.stderr)
        return 1

    path.write_text(src.replace(OLD, NEW))
    print(f"[patch_aiter_check_args] patched OK: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
