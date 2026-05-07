# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for ``flydsl_kernel.py``.

Run with::

    cd ~/PR/sglang
    PYTHONPATH=python COMPILE_ONLY=1 python -m \
        sglang.srt.layers.attention.nsa.test_flydsl_kernel

Inside the ``jacchang_GLM_FlyDSL`` Docker container::

    docker exec jacchang_GLM_FlyDSL bash -c \
        "cd /home/jacchang/PR/sglang && \
         PYTHONPATH=python COMPILE_ONLY=1 python -m \
         sglang.srt.layers.attention.nsa.test_flydsl_kernel"

Status (2026-05-07):
    * combine kernel: COMPILES end-to-end on gfx950.
    * partial-FP8 kernel: COMPILES end-to-end on gfx950.

Numerical correctness vs the TileLang reference is not yet verified -- the
next step is to run both backends side by side on the same Q/KV/Indices
tensors and diff the BF16 output.
"""

from __future__ import annotations

import os

import torch

from sglang.srt.layers.attention.nsa import flydsl_kernel as fk


def smoke_combine() -> None:
    print("[combine] Building kernel...")
    combine = fk.build_sparse_mla_fwd_decode_combine(
        num_heads=128,
        d_v=512,
        topk=2048,
        head_per_block=4,
        block_I=64,
        threads=256,
    )

    NI = 32
    seq_len = 1
    po = torch.zeros(
        1, seq_len, NI, 128, 512, dtype=torch.bfloat16, device="cuda"
    )
    plse = torch.zeros(
        1, seq_len, NI, 128, dtype=torch.float32, device="cuda"
    )
    out = torch.zeros(
        1, seq_len, 128, 512, dtype=torch.bfloat16, device="cuda"
    )
    print("[combine] Calling launcher...")
    combine(po, plse, out, seq_len)
    print("[combine] OK")


def smoke_partial() -> None:
    print("[partial] Building kernel...")
    partial = fk.build_sparse_mla_fwd_decode_partial_fp8(
        num_heads=128,
        d_v=512,
        d_tail=64,
        topk=2048,
        sm_scale=1.0 / 24.0,
        block_I=64,
        inner_iter=8,
        threads=256,
    )
    from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
    seq_len = 1
    n_groups = 4
    fp8_dt = (
        torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
    )
    q = torch.zeros(1, seq_len, 128, 576, dtype=fp8_dt, device="cuda")
    kv = torch.zeros(1, 4096, 1, 576, dtype=fp8_dt, device="cuda")
    indices = torch.zeros(1, seq_len, 1, 2048, dtype=torch.int32, device="cuda")
    po = torch.zeros(1, seq_len, n_groups, 128, 512, dtype=torch.bfloat16, device="cuda")
    plse = torch.zeros(1, seq_len, n_groups, 128, dtype=torch.float32, device="cuda")
    print("[partial] Calling launcher...")
    partial(q, kv, indices, po, plse, seq_len)
    print("[partial] OK")


if __name__ == "__main__":
    if not os.environ.get("COMPILE_ONLY"):
        os.environ["COMPILE_ONLY"] = "1"
    smoke_combine()
    smoke_partial()
