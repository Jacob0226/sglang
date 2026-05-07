# FlyDSL Sparse-MLA Decode — Performance Optimization Plan

> Status: 2026-05-07. Correctness vs TileLang **PASS** on all three split-K
> granularities (`inner_iter ∈ {1, 4, 32}`); max abs diff ~1e-4 on BF16 output.
> Performance is **2-11x slower** than TileLang and the gap grows with
> sequence length. This document enumerates the concrete optimizations to
> close the gap, in priority order.

## Baseline (MI355X / gfx950, `rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260503`)

```
case                  TileLang   FlyDSL   FlyDSL/TileLang
seq=64,  pages=8192    0.179 ms  0.43 ms      0.42x
seq=128, pages=16384   0.315 ms  1.18 ms      0.27x
seq=256, pages=32768   0.601 ms  3.51 ms      0.17x
seq=512, pages=65536   1.172 ms 12.98 ms      0.09x
```

Run with:

```bash
docker exec jacchang_GLM_FlyDSL bash -c \
  "cd /home/jacchang/PR/sglang && PYTHONPATH=python python -m \
   sglang.srt.layers.attention.nsa.bench_flydsl_vs_tilelang"
```

---

## Optimization Backlog (ordered by expected impact / effort ratio)

### 1. ⭐ V LDS transpose — **biggest single win**

**Problem.** Per SV B-pack each lane reads 8 strided bytes (`(k_row+i)*GROUP_SIZE+n_col`,
i in 0..7) → 8 `ds_read_u8` ≈ 8x slower than 1 `ds_read_b64`. With `ks=2 *
ns=8 = 16` packs per SV GEMM and `inner_iter * num_iter` SV GEMMs per CTA,
this dominates the kernel.

**Fix.** Allocate a SECOND LDS region for V in column-major layout:

```
vt_smem  : [d_v, BI] FP8 (= [512, 64] = 32 KiB)
addr     : vt_smem[n * BI + k]
```

After the cooperative gather, V data is duplicated into both `kv_smem`
(row-major, used by QK GEMM) and `vt_smem` (col-major, used by SV GEMM).
SV B-pack becomes:

```python
b_off = (n_col * BI) + (ks * MFMA_K) + (lane_div_16 * MFMA_LANES_PER_K_PACK)
b_v8i8 = vector.load_op(v8i8_t, vt_smem, [b_off])  # 8 bytes contiguous!
b_i64  = _pack_v8i8_as_i64(b_v8i8)                # bitcast to i64
```

**Cost.** +32 KiB LDS per CTA. Total ~80 KiB. Fits gfx950 (160 KiB) but
**blows gfx942 (64 KiB)** — gate behind `is_gfx95_supported()` and keep the
slow-path for gfx942.

**Sub-issue: how to write `vt_smem` cheaply.** Each gather thread already
reads 16 K-bytes contiguously for one (page, n-col-block). Writing those 16
bytes to `vt_smem` requires 16 individual addresses (one per N-col).
Naive: 16 `ds_write_u8` per thread (~slow, possible bank conflicts).
Better options:

* **(a) In-wave shuffle transpose.** Use `gpu.ShuffleOp` (or `ds_bpermute`)
  to redistribute the 16 K-bytes across 16 lanes such that each lane ends
  up with 16 K-bytes for one fixed N-col, then write with 1 `ds_write_b128`.
  Best perf, ~30 lines of shuffle plumbing.
* **(b) LDS-to-LDS transpose pass.** After the row-major gather, do a
  transpose pass that reads 16x16 blocks from `kv_smem` (3x `ds_read_b32`),
  shuffles to transpose, writes col-major (3x `ds_write_b32`). Adds 1
  barrier per gather but works on existing data. Easier.
* **(c) Per-byte writes.** 16 `ds_write_u8` per thread. Simplest. Probably
  bank-conflict-bound to ~16 cycles per gather.

Recommend (b) for first cut, (a) if (b) is still slow.

**Expected impact.** ~3-5x speedup. SV becomes register-bound rather than
LDS-bound.

**Effort.** ~30-60 minutes including correctness re-test.

---

### 2. K/V double-buffering

**Problem.** Per KV iter we serialize: `gather → barrier → QK → barrier →
softmax → barrier → quantize → barrier → SV → barrier`. Each barrier is
~50-200 cycles on a busy CU and the gather is an external memory wait.

**Fix.** Two LDS buffers for KV (`kv_smem_a`, `kv_smem_b`) and a software
pipeline:

```
Iter 0: gather → buf_a
Iter 1: gather → buf_b   ||   QK+softmax+SV(buf_a)
Iter 2: gather → buf_a   ||   QK+softmax+SV(buf_b)
...
```

**Cost.** Doubles KV LDS (~32 KiB more for the V transposed too if we keep
both). Fits gfx950, tight on gfx942.

**Expected impact.** ~1.5-2x speedup, more for memory-bound configs (small
`inner_iter` / large `topk`).

**Effort.** ~1-2 hours (need to add `cur_buf_id` loop-carried state via
`scf_range(init=...)` / careful Python rebind).

---

### 3. `rocdl.sched_group_barrier` — instruction scheduling

**Problem.** The compiler can't see that QK MFMA can run while SV LDS reads
are in flight (or that gather VMEM can run while softmax MFMA executes).
Without sched hints it tends to produce a strict in-order schedule.

**Fix.** Add the `flash_attn_func.py`-style scheduling annotations:

```python
rocdl.sched_group_barrier(rocdl.mask_vmem_rd, 1, 0)   # 1 VMEM-read slot
rocdl.sched_group_barrier(rocdl.mask_mfma,    2, 0)   # 2 MFMA slots
rocdl.sched_group_barrier(rocdl.mask_dsrd,    1, 0)   # 1 LDS-read slot
```

placed at the boundary between the QK loop, softmax, SV loop, and the
next-iter gather (only useful with #2, double-buffering).

**Expected impact.** ~1.1-1.3x.

**Effort.** ~30 minutes. Best done after #2.

---

### 4. XOR LDS swizzle for KV

**Problem.** Row-major KV LDS at `addr(row, col) = row * GROUP_SIZE + col`
collides on the same bank when 4 lanes within a wave's 16-lane block read
the same column bytes (e.g. when ks varies but lane_mod_16 is fixed). On
gfx950 LDS has 32 banks of 4 bytes each → up to 4-way conflict on B-pack
reads.

**Fix.** Apply `col ^ ((row & 7) << 4)` swizzle (16-element granularity) on
both write and read sides — same pattern as `flash_attn_func.py`.

**Expected impact.** ~1.1-1.3x for SV B-pack reads (less if #1 already
converted them to single `ds_read_b64`).

**Effort.** ~20 minutes — easy `_kv_swizzle(row, col)` helper plus update
both gather store and SV read addresses.

---

### 5. `buffer_load_dwordx4_lds` DMA path

**Problem.** Current cooperative gather loads global → VGPR → LDS (two-hop).
Costs 16 VGPRs per thread per gather and prevents overlapping VMEM with
next-iter compute (registers are pinned).

**Fix.** Use `rocdl.raw_ptr_buffer_load_lds` (= `buffer_load_dwordx4_lds`)
which DMAs directly global → LDS. Available on gfx950+. See
`flash_attn_func.py::coop_dma_k`.

**Expected impact.** Frees ~16 VGPRs per thread → potentially higher
occupancy. Modest direct latency win, but enables more aggressive
double-buffering (#2).

**Effort.** ~30-45 minutes — slightly tricky LDS-byte-offset arithmetic.

---

### 6. Reduce-scratch barrier merging

**Problem.** Cross-wave softmax does two LDS dances (one for max, one for
sum) with 2 `gpu.barrier()` each. That's 4 barriers per KV iter.

**Fix.** Pack max and sum into the same scratch slot, e.g.:

```python
# Each lane writes BOTH max and sum into one i64 (= 2 fp32 packed).
packed = vector.from_elements(v2f32_t, [max_val, sum_val])
vector.store(packed, reduce_smem, [off])
gpu.barrier()
# Read back, unpack, reduce both at once.
```

Cuts barriers from 4 to 2 per iter.

**Expected impact.** ~1.05-1.15x (saves a few hundred cycles per CTA).

**Effort.** ~20 minutes.

---

### 7. Use `ds_read_tr8_b64` (HW transpose) instead of LDS V transpose (#1 alternative)

**Problem.** Same as #1 — strided V reads in SV GEMM.

**Alternative fix.** Use `rocdl.ds_load_tr8_b64` (FP8 hardware transpose
LDS read, gfx950+). Each lane provides an address; HW reads 8 bytes per
lane and applies a 4×4 transpose within 16-lane blocks so that the result
matches the MFMA B operand layout for FP8.

**Trade-offs vs #1.**

* ✅ No extra LDS (vs +32 KiB for vt_smem)
* ✅ No second LDS write in the gather
* ❌ Per-lane address pattern is non-obvious and ISA-specific
* ❌ Only available on gfx950+ (same gating as #1's gfx942 fallback)

**Effort.** ~1-2 hours if you have to derive the address pattern from
scratch; ~30 minutes if you copy from `flash_attn_func.py::ds_read_tr_v4f16`
and adapt.

**Recommendation.** Do #1 first (more straightforward), revisit #7 if #1's
gather-side transposed write is the bottleneck.

---

## Recommended sequencing

1. **Sprint 1 (~2-3 hours):** #1 (V LDS transpose, option (b) LDS-to-LDS
   transpose pass) + #4 (XOR swizzle). Measure.
2. **Sprint 2 (~2-3 hours):** #2 (double-buffering) + #3 (sched barriers).
   Measure.
3. **Sprint 3 (~1-2 hours):** #6 (barrier merging) + #5 (DMA path). Measure.
4. **Optional:** #7 if #1 is still bottlenecked on the transposed write.

After all sprints, expect FlyDSL within ~1.0-1.3x of TileLang. Beyond that
requires deeper LLVM scheduler tuning (`waves_per_eu`, `flat_work_group_size`,
`enable-post-misched`, `lsr-drop-solution` -- see the
`_fmha_compile_hints` block in `flash_attn_func.py`).

---

## Tooling tips

* **Dump IR.** `FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=/tmp/flydsl_dump python ...`
  produces `00_origin.mlir` ... `18_reconcile_unrealized_casts.mlir` per
  kernel under `/tmp/flydsl_dump/<kernel_name>_0/`. The `08_convert_fly_to_rocdl.mlir`
  stage is the most readable for spotting LDS / VMEM access patterns.
* **Disable disk cache while iterating.** `FLYDSL_RUNTIME_ENABLE_CACHE=0`
  forces every change to recompile.
* **Compile-only sanity check.** `COMPILE_ONLY=1 python -m
  sglang.srt.layers.attention.nsa.test_flydsl_kernel` traces + lowers + emits
  the GFX binary but doesn't launch -- useful when GPU is busy.
* **Side-by-side correctness.** `python -m
  sglang.srt.layers.attention.nsa.test_flydsl_vs_tilelang` (no `COMPILE_ONLY`).
  Tolerance defaults are `atol=1e-2, rtol=1e-2`; tighten for regressions.
* **Benchmark harness.** `python -m
  sglang.srt.layers.attention.nsa.bench_flydsl_vs_tilelang` measures median
  latency over 50 iters.

---

## Reference implementations to crib from

* **`/home/jacchang/PR/aiter/aiter/ops/flydsl/kernels/flash_attn_func.py`**
  — production FlyDSL FMHA on gfx950. Includes XOR swizzle, DMA loads,
  ds_read_tr16, ping-pong buffers, sched annotations, `waves_per_eu`,
  ROCm LLVM passthrough attrs. The single best reference for a
  performance-tuned FlyDSL attention kernel.
* **`/home/jacchang/PR/aiter/aiter/ops/flydsl/kernels/preshuffle_gemm.py`**
  — production FlyDSL FP8 GEMM. Reference for MFMA-FP8 i64 packing,
  CShuffle epilog, LDS-stage management.
* **`/home/jacchang/PR/aiter/aiter/ops/flydsl/kernels/pa_decode_fp8.py`**
  — production FlyDSL paged-attention decode (FP8). Closest analog to
  our sparse MLA decode -- same Q-in-LDS / KV-gather / online-softmax
  pattern, just no sparse indexing.
