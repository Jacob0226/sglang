#!/usr/bin/env python3
"""
GLM-5.2 fp8 MLA decode microbench (MI355X / gfx950).

Isolates the cost of SGLang's opt#2 aiter fp8 decode path vs. ATOM's, to explain
the ~69% end-to-end regression. Reproduces the exact decode call from
dsa_backend.py::_forward_aiter with GLM-5.2 TP4 shapes (nhead=16, d=576, v=512,
topk=2048), and measures:

  1. pure mla_decode_fwd kernel time (metadata hoisted)      -> vs ATOM 13.7us
  2. metadata build time (get_mla_metadata_v1 + get_valid_kv_indices)
  3. per-DECODE-STEP cost, SGLang-style (build+kernel) x N_layers  [per-layer rebuild]
  4. per-DECODE-STEP cost, ATOM-style   (build x1) + kernel x N_layers [hoisted]
  5. whether the metadata build is CUDA-graph capturable at all.

All timings via cuda events; CUDA-graph captured where noted (that is the real
decode deployment mode).
"""
import argparse
import torch
import aiter
from aiter import dtypes
from aiter.mla import mla_decode_fwd
from aiter.ops.attention import get_mla_metadata_info_v1, get_mla_metadata_v1
from sglang.srt.layers.attention.dsa.triton_kernel import get_valid_kv_indices

FP8 = torch.float8_e4m3fn


def build_inputs(bs, seq_len, nhead, d, v, topk, device):
    # KV pool: page_size=1, [num_tokens, 1, 1, d], fp8
    pool_len = bs * (seq_len + 8) + 128
    kv = (torch.randn(pool_len, 1, 1, d, device=device, dtype=torch.bfloat16) * 0.1).to(FP8)

    # q_kernel: [bs, nhead, d] fp8 ; o: [bs, nhead, v] bf16
    q = (torch.randn(bs, nhead, d, device=device, dtype=torch.bfloat16) * 0.1).to(FP8)
    o = torch.empty(bs, nhead, v, device=device, dtype=torch.bfloat16)

    # page_table_1: [bs, topk] int32, first seq_len valid token ids then -1
    page_table = torch.full((bs, topk), -1, device=device, dtype=torch.int32)
    for b in range(bs):
        start = b * (seq_len + 8)
        n = min(seq_len, topk)
        page_table[b, :n] = torch.arange(start, start + n, device=device, dtype=torch.int32)

    qo_indptr = torch.arange(bs + 1, device=device, dtype=torch.int32)
    kv_last_page_lens = torch.ones(bs, device=device, dtype=torch.int32)
    return kv, q, o, page_table, qo_indptr, kv_last_page_lens


def alloc_meta_buffers(bs, nhead, num_kv_splits, intra, device):
    info = get_mla_metadata_info_v1(
        bs, 1, nhead, FP8, FP8, is_sparse=True, fast_mode=False,
        num_kv_splits=num_kv_splits, intra_batch_mode=intra,
    )
    bufs = [torch.empty(sz if isinstance(sz, tuple) else (sz,), dtype=dt, device=device)
            for (sz, dt) in info]
    return bufs  # work_meta, work_indptr, work_info, reduce_indptr, reduce_final, reduce_partial


def build_metadata(qo_indptr, kv_indptr, kv_last_page_lens, page_table, kv_indices,
                   bufs, nhead, num_kv_splits, intra, bs, topk):
    (work_meta, work_indptr, work_info, reduce_indptr, reduce_final, reduce_partial) = bufs
    # compact indices (get_valid_kv_indices) -- part of per-layer rebuild in SGLang
    non_minus1 = (page_table != -1).sum(dim=1)
    kv_indptr[1:bs + 1] = torch.cumsum(non_minus1, dim=0)
    get_valid_kv_indices(page_table, kv_indptr, kv_indices, bs)
    get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_lens, nhead, 1, False,
        work_meta, work_info, work_indptr, reduce_indptr, reduce_final, reduce_partial,
        page_size=1, kv_granularity=16, max_seqlen_qo=1, uni_seqlen_qo=1,
        fast_mode=False, max_split_per_batch=num_kv_splits,
        intra_batch_mode=intra, dtype_q=FP8, dtype_kv=FP8,
    )


def run_kernel(q, kv, o, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens, bufs,
               num_kv_splits, intra, q_scale, kv_scale, d):
    (work_meta, work_indptr, work_info, reduce_indptr, reduce_final, reduce_partial) = bufs
    mla_decode_fwd(
        q, kv.view(-1, 1, 1, d), o, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens,
        1, page_size=1, sm_scale=1.0 / (d ** 0.5), num_kv_splits=num_kv_splits,
        work_meta_data=work_meta, work_indptr=work_indptr, work_info_set=work_info,
        reduce_indptr=reduce_indptr, reduce_final_map=reduce_final,
        reduce_partial_map=reduce_partial, q_scale=q_scale, kv_scale=kv_scale,
        intra_batch_mode=intra,
    )


def timed(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=1400)
    ap.add_argument("--nhead", type=int, default=16)
    ap.add_argument("--layers", type=int, default=75)
    ap.add_argument("--num-kv-splits", type=int, default=16)
    ap.add_argument("--intra", action="store_true", default=False)
    args = ap.parse_args()

    device = "cuda"
    d, v, topk = 576, 512, 2048
    intra = args.intra
    print(f"gfx={aiter.jit.utils.chip_info.get_gfx()} bs={args.bs} seq_len={args.seq_len} "
          f"nhead={args.nhead} layers={args.layers} splits={args.num_kv_splits} intra={intra}")

    kv, q, o, page_table, qo_indptr, kv_last_page_lens = build_inputs(
        args.bs, args.seq_len, args.nhead, d, v, topk, device)
    kv_indptr = torch.zeros(args.bs + 1, device=device, dtype=torch.int32)
    kv_indices = torch.zeros(args.bs * topk, device=device, dtype=torch.int32)
    bufs = alloc_meta_buffers(args.bs, args.nhead, args.num_kv_splits, intra, device)
    q_scale = torch.ones((), dtype=torch.float32, device=device)
    kv_scale = torch.ones((), dtype=torch.float32, device=device)

    (work_meta, work_indptr, work_info, reduce_indptr, reduce_final, reduce_partial) = bufs

    def do_compact():  # get_valid_kv_indices — PER-LAYER (depends on layer's topk sel)
        non_minus1 = (page_table != -1).sum(dim=1)
        kv_indptr[1:args.bs + 1] = torch.cumsum(non_minus1, dim=0)
        get_valid_kv_indices(page_table, kv_indptr, kv_indices, args.bs)

    def do_schedule():  # get_mla_metadata_v1 — HOISTABLE (depends only on kv counts)
        get_mla_metadata_v1(
            qo_indptr, kv_indptr, kv_last_page_lens, args.nhead, 1, False,
            work_meta, work_info, work_indptr, reduce_indptr, reduce_final, reduce_partial,
            page_size=1, kv_granularity=16, max_seqlen_qo=1, uni_seqlen_qo=1,
            fast_mode=False, max_split_per_batch=args.num_kv_splits,
            intra_batch_mode=intra, dtype_q=FP8, dtype_kv=FP8,
        )

    def do_build():
        do_compact(); do_schedule()

    def do_kernel():
        run_kernel(q, kv, o, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens, bufs,
                   args.num_kv_splits, intra, q_scale, kv_scale, d)

    # correctness / no-fault first
    do_build(); do_kernel(); torch.cuda.synchronize()
    print("OK: build + kernel ran without fault")

    t_build = timed(do_build)
    t_compact = timed(do_compact)
    t_sched = timed(do_schedule)
    t_kernel = timed(do_kernel)
    print(f"[eager] metadata build (compact+sched): {t_build*1000:8.2f} us/call")
    print(f"[eager]   get_valid_kv_indices        : {t_compact*1000:8.2f} us/call  (per-layer, must stay)")
    print(f"[eager]   get_mla_metadata_v1         : {t_sched*1000:8.2f} us/call  (HOISTABLE)")
    print(f"[eager] kernel only                   : {t_kernel*1000:8.2f} us/call  (ATOM ref ~13.7us)")

    N = args.layers
    def per_layer_step():   # SGLang current: full rebuild every layer
        for _ in range(N):
            do_compact(); do_schedule(); do_kernel()
    def hoisted_step():     # ATOM: schedule once, compact+kernel per layer
        do_schedule()
        for _ in range(N):
            do_compact(); do_kernel()
    def fully_hoisted_step():  # upper bound: schedule+compact once
        do_compact(); do_schedule()
        for _ in range(N):
            do_kernel()

    t_pl = timed(per_layer_step, iters=10, warmup=3)
    t_ho = timed(hoisted_step, iters=10, warmup=3)
    t_fh = timed(fully_hoisted_step, iters=10, warmup=3)
    print(f"[eager] per-step CURRENT   (rebuild all x{N}) : {t_pl:8.3f} ms")
    print(f"[eager] per-step FIX       (sched x1)         : {t_ho:8.3f} ms  (save {t_pl-t_ho:.2f} ms)")
    print(f"[eager] per-step FULL-HOIST(sched+compact x1) : {t_fh:8.3f} ms  (save {t_pl-t_fh:.2f} ms)")

    # ---- CUDA graph capturability of the metadata build ----
    print("\n-- CUDA graph capture test (per-call) --")
    for name, fn in [("kernel-only", do_kernel), ("build-only", do_build),
                     ("build+kernel", lambda: (do_build(), do_kernel()))]:
        try:
            g = torch.cuda.CUDAGraph()
            do_build(); do_kernel(); torch.cuda.synchronize()  # warm
            with torch.cuda.graph(g):
                fn()
            torch.cuda.synchronize()
            t = timed(lambda: g.replay(), iters=50, warmup=10)
            print(f"  capture {name:14s}: OK   replay {t*1000:8.2f} us")
        except Exception as ex:
            print(f"  capture {name:14s}: FAILED  {type(ex).__name__}: {str(ex)[:160]}")

    # ---- Deployment-accurate: whole per-decode-step captured in a graph ----
    print("\n-- CUDA graph captured PER-DECODE-STEP (deployment-accurate) --")
    for name, fn in [(f"CURRENT rebuild x{N}", per_layer_step),
                     ("FIX sched-hoist x1", hoisted_step),
                     ("FULL-HOIST x1", fully_hoisted_step)]:
        try:
            g = torch.cuda.CUDAGraph()
            fn(); torch.cuda.synchronize()  # warm
            with torch.cuda.graph(g):
                fn()
            torch.cuda.synchronize()
            t = timed(lambda: g.replay(), iters=20, warmup=5)
            print(f"  {name:22s}: {t:8.3f} ms/step")
        except Exception as ex:
            print(f"  {name:22s}: FAILED  {type(ex).__name__}: {str(ex)[:120]}")


if __name__ == "__main__":
    main()
