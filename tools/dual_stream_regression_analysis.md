# GLM-5-FP8 NSA Decode Dual-Stream Regression Analysis (MI355X / ROCm)

**Date:** Apr 2026

**Baseline:** SGLang PR #21511 + aiter PR #2879.

**Configuration under test:** dual-stream layout enabled via `SGLANG_ENABLE_HIP_DUAL_STREAM=1` + `--dual-stream-rocm` (the latter implies `--disable-shared-experts-fusion`). The optimization overlaps the NSA indexer chain on alt-stream with `[q_b_proj + bmm w_kc + fused_qk_rope_cat]` on the current stream.

**TLDR:** Dual-stream layout **loses ~+49 μs / layer** of wall-clock vs Baseline on MI355X (271 → 320 μs / layer; bench TPOT regression of 3.24 ms / token corresponds to ~+50 μs / layer at TPOT level — these match). The four roughly-equal contributors are:
1. **HIP-graph dual-stream bubble** ~+18 μs / layer of which **+16.5 μs is ≥2 μs real CPU stalls** (3 distinct stall regions, 5–6 μs each — verified not launch overhead by cross-platform comparison; see "Bubble decomposition" below)
2. **AllReduce slowdown** ~+16 μs / layer (+7.98 μs / call mean × 2 calls)
3. **MoE block widens** ~+9 μs wall-clock (forced unfused shared experts; +46 μs kernel-duration mostly hidden by overlap)
4. **Indexer chain HBM contention** ~+7 μs / layer (per-fire normalized)

**Cross-platform validation:** B200 GLM-5-FP8 dual-stream (CUDA-graph stream-pool replay) has **0 μs of ≥2 μs stalls** as well — its dual-stream layout is clean. The two platforms differ in *another* dimension that turns out to be a profiler artifact: B200 (CUPTI) reports ~23 μs / layer of sub-1 μs inter-kernel gaps, while MI355X (roctracer) reports ~0 μs. We initially read this as "B200 has launch overhead, MI355X doesn't"; that's wrong — see "Bubble decomposition" below. The actually-meaningful comparison is the ≥2 μs stall count: **MI355X dual-stream has 3 real stalls / layer, B200 dual-stream has 0**.

---

## Bench Numbers (8k1k conc4, GLM-5.1-FP8, TP=8, fp8 KV cache)

| Variant | Median TPOT | Δ vs Baseline |
|---|---|---|
| Baseline (SGLang PR #21511, ROCm 7.2) | **21.21 ms** | — |
| Dual stream (ROCm 7.2) | **24.45 ms** | **+15.3% (regression)** |
| Dual stream + ROCm 7.13 (TheRock pip; **NEW**) | **28.32 ms** | **+33.5% (worsens further)** |

The ROCm 7.2 dual-stream layout regresses by 3.24 ms / token = ~50.6 μs / layer at TPOT level. **Upgrading to ROCm 7.13 user-space (via TheRock pip stack) makes it worse, not better** — additional +3.87 ms / token, with the kernel timeline now using **4 active GPU streams** instead of 2.

---

## Per-Block Comparison (avg over 15 layers, trace-direct measurement)

Layer windowing: `(end of MoE-AR_i) → (end of MoE-AR_{i+1})` — one full transformer layer. Each window covers `attn block + attn-AR + MoE block + MoE-AR`.

| bucket | Baseline (PR #21511) | Dual stream | Δ wall-clock |
|---|---|---|---|
| AllReduce per call (mean over 781 calls / trace) | 10.26 μs | **18.24 μs** † | +7.98 μs / call |
| AllReduce / layer (× 2 calls) | 20.5 μs | 36.5 μs | **+16.0 μs / layer** |
| Indexer chain (9 kernels per layer, per-fire normalized) ‖ | 63.3 μs | 70.3 μs | **+7.0 μs / layer** |
| MoE block (wall-clock width) ‡ | 78.6 μs | 87.4 μs | **+8.8 μs / layer** |
| **GPU bubble (idle, no kernel on any stream) §** | **0.04 μs** | **18.19 μs** | **+18.2 μs / layer** |
| **total layer dur (real, wall-clock)** | **271.0 μs** | **320.0 μs** | **+49.0 μs / layer** |

So real-layer regression is `320 − 271 ≈ +49 μs / layer` (~18%) at the trace level, which matches the bench TPOT regression at the per-layer level (3.24 ms / token / num_layers ≈ +50 μs).

† Same `aiter::cross_device_reduce_1stage<bf16, 8>` kernel — yet much slower under dual-stream HIP-graph capture. Distribution shifts uniformly right (no bimodal): see "AllReduce slowdown" below.

‖ Indexer chain = 9 kernels per layer (`indexer_layernorm`, `hadamard ×2`, `act_quant`, `indexer_k_quant_and_cache`, `wv_splitk`, `paged_mqa_logits_preshuffle`, `topk_transform_decode`, `triton_poi_fused_mul_unsqueeze`, `cached_indirect_inplace`). Per-fire deltas: `indexer_layernorm +0.73 μs`, `hadamard +0.96 μs ×2`, `act_quant +0.72`, `indexer_k_quant +0.82`, `wv_splitk +0.95`, `paged_mqa_logits +0.88`, `topk_transform +0.32`, `fused_mul_unsqueeze +0.72`, `cached_indirect −0.07`. Memory-bound kernels each pay 0.7–1.0 μs in dual-stream; compute-bound `topk_transform` and `cached_indirect` pay ~0. See "HBM bandwidth contention" below.

§ Bubble = layer wall-clock − union of busy intervals across all streams. Measured directly on 15 consecutive layer-passes; samples in dual-stream were 11.8, 18.4, 18.0, 19.0, 17.6, 17.9, 24.1, 17.5, 19.2, 18.2, 18.7, 18.8, 18.0, 17.3, 18.2 μs (stable, not outlier-driven). Baseline is uniformly ~0 μs — the GPU is queued back-to-back. **Crucially, decomposing this bubble by gap-size shows it is real CPU stalls, not launch overhead** — see the "Bubble decomposition (kind classification)" section below for why this matters.

‡ MoE block decomposition (avg over 20 layers):

| metric | Baseline | Dual stream |
|---|---|---|
| MoE block wall-clock | **78.6 μs** | **87.4 μs** |
| sum of kernel durations | 79.0 μs | 124.9 μs |
| overlap factor (sum/wall) | 1.01× (serial) | 1.43× (concurrent) |

The `--disable-shared-experts-fusion` flag (implied by `--dual-stream-rocm`) does change which kernels run:
- **Baseline (fused)**: a single `_fused_append_shared_experts_kernel` (~4.4 μs) handles the entire shared-expert pipeline (gate_up + silu + down + add-residual).
- **Dual stream (unfused)**: aiter splits the chain into 4 kernels — `ck_gemm_preshuffle` (gate_up, ~37 μs avg over 20 blocks; long because it runs on alt-stream concurrent with cur-stream router/sort/routed-gate_up under HBM contention) + `sgl_hip::act_and_mul` (~7 μs) + `ck_gemm_preshuffle` (down, ~5 μs) + `vectorized_*_add` (~4 μs) ≈ **~53 μs of kernel-duration time**.

Naively this looks like **+49 μs of extra shared-expert work per layer**, but the trace shows the MoE block only widens by **+8.8 μs wall-clock** because dual-stream overlaps the slow shared gate_up with the routed-expert dispatch on cur stream (overlap factor 1.43×). The "cost" of unfused shared experts is therefore mostly hidden — the actual MoE wall-clock penalty is only ~+9 μs / layer, not +43 / +49 μs.

---

## ROCm 7.13 + 4-stream regression (NEW, May 2026)

After upgrading to ROCm 7.13 user-space (via [TheRock](https://github.com/ROCm/TheRock) pip — torch 2.11.0+rocm7.13, rocm-sdk 7.13.0a20260426 with bundled libamdhip64.so.7.13.26162) and running the same dual-stream branch on a **fresh container** with rebuilt aiter / sgl-kernel / fast-hadamard-transform, the dual-stream layout uses **4 active GPU streams** (cur + 3 alt streams) instead of the 2 streams seen on ROCm 7.2 — yet wall-clock per layer **gets worse, not better**.

### Direct measurement (8k1k conc4, GLM-5.1-FP8, TP=8)

The trace has high layer-dur variance (min 290 / median 366 / max 636 μs across 391 layer windows). For a fair like-for-like comparison we use a "typical 4-stream layer" (#43, dur 337 μs, all 4 streams active) for kernel-level numbers, but quote the median 366 μs for the bench-aligned wall comparison.

| Variant | Layer wall (this layer / median) | AR mean | All-stream-idle bubble | Real-stall ≥2 μs / layer | Streams |
|---|---|---|---|---|---|
| MI355X single-stream (ROCm 7.2) | 271 / 271 μs | 10.26 μs | 0 μs | 0 μs | 1 |
| MI355X dual-stream HIP (ROCm 7.2) | 312 / 320 μs | 18.24 μs | 18.2 μs | 16.5 μs (3) | 2 |
| **MI355X dual-stream HIP (ROCm 7.13)** | **337 / 366 μs** | **22.26 μs** (this) / **25.72 μs** (median) | **18.2 μs** | **16.3 μs (3)** | **4** |
| B200 dual-stream CUDA-graph (reference) | 303 μs | n/a | 23.4 μs sub-1µs (artifact) | 0 μs | 4 (pool) |

### Surprise: bubble didn't grow, kernels did

**Trace-direct decomposition of the +25 μs / layer regression vs ROCm 7.2 dual-stream (using #43 layer):**

| component | ROCm 7.2 dual | ROCm 7.13 dual (4-stream) | Δ |
|---|---|---|---|
| AllReduce per layer (× 2 calls) | 36.5 μs | **44.5 μs** (mean 22.26 × 2) | **+8.0 μs** |
| Indexer chain | 70.3 μs | **91.3 μs** (15 kernels) | **+21.0 μs** |
| GEMMs (9 ck_gemm + a8w8) | ~96 μs | **105.8 μs** | **+9.8 μs** |
| All-stream-idle bubble | 18.2 μs | **18.2 μs** | **±0** |
| Other (norm, silu, mqa, add, misc) | rest | rest | balance ~−14 μs |
| **total layer wall** | **312 μs** | **337 μs** | **+25 μs** |

The most striking finding: **the all-stream-idle bubble didn't get worse**. It's still ~18 μs / layer with 3 real stalls — exactly the same pattern as ROCm 7.2 dual-stream. **Adding 2 more streams (2 → 4) did NOT amplify HIP-graph fence cost** the way one might have feared. So this isn't a "more streams → more fences" story.

The +25 μs / layer regression is entirely from **kernels running slower**:

1. **AR per call slows another +4 μs** (18.24 → 22.26 μs at this layer; +7.5 μs at the median). Same `aiter::cross_device_reduce_1stage` kernel, same TP=8 topology. The kernel itself is taking longer in ROCm 7.13. With 2 AR per layer, **+8 μs / layer** here, more at median.

2. **Indexer chain +21 μs** — memory-bound kernels (Hadamard, act_quant, indexer_layernorm, paged_mqa_logits, topk_transform) each pay 0.5–2 μs more. With 4 concurrent streams competing for the same 8 TB/s HBM pipe, contention is worse than 2-stream ROCm 7.2.

3. **GEMMs +9.8 μs** — aiter rebuilt with `PREBUILD_KERNELS=1` but `/tmp/aiter_configs/*.csv` is empty after the rebuild. First-touch shapes hit "torch solution:0" / "skinny solution:2" defaults instead of tuned solutions (visible as warnings in server log). aiter autotune would likely recover most of this.

### Comparison summary

| Hypothesis | ROCm 7.2 dual analysis (original) | ROCm 7.13 dual (this run) |
|---|---|---|
| HIP-graph fence in critical path → bubble | ✓ confirmed | ✓ same magnitude (no worse) |
| HBM contention slows memory-bound kernels | possible (+7 μs indexer) | ✓ amplified (+21 μs indexer; 4 streams worse than 2) |
| AR fence slows AR kernel | ✓ +8 μs / call | ✓ another +4 μs / call (cumulative +12 vs baseline) |
| Untuned kernels (aiter cache wiped) | n/a | **new contributor** (+10 μs in GEMMs) |

### What's needed for ROCm 7.13 dual-stream to win

- **Aiter autotune** on the fresh ROCm 7.13 stack: regenerate `/tmp/aiter_configs/*.csv` for all GEMM and quant shapes used by GLM-5.1-FP8 decode. Likely recovers 9–15 μs / layer (the "GEMMs +9.8" + part of "indexer +21" buckets).
- **AR kernel tuning for ROCm 7.13**: the `aiter::cross_device_reduce_1stage` kernel's slowdown across versions deserves investigation — it's the same kernel with the same workload but +12 μs / call vs ROCm 7.2 single-stream. Either the kernel needs new tuning or the ROCm 7.13 runtime adds per-call setup.
- **Memory bandwidth with 4 streams**: HBM3E saturation profile under 4 concurrent streams probably worse than 2. Could be measured with HIP profiler counters.
- **Bench warm-up**: aiter JIT triggers many "first-touch" compilations during early server life. Forcing a longer pre-prof warmup (e.g., 3-5 min of varied request shapes) before the prof window should eliminate the "skinny solution:2" cold-cache penalty.

### Validates the original hypothesis (with one caveat)

The original ROCm 7.2 analysis blamed the regression on HIP-graph cross-stream fence cost in dual-stream replay. The ROCm 7.13 4-stream measurement **partially confirms and partially refines** that:
- ✓ HIP-graph fence cost is real (~18 μs bubble per layer in BOTH 2-stream and 4-stream).
- ✗ But fence cost did NOT scale linearly with stream count — adding 2 more streams kept the bubble flat. So "more streams → more fences in critical path" is wrong.
- ✓ HBM contention IS worse with more streams. This is consistent with the original hypothesis but more pronounced.
- New: **untuned aiter kernels** add a one-time penalty after stack rebuild that's separable from the dual-stream layout itself.

The dual-stream layout regression is therefore mostly **kernel-level cost** (AR + memory-bound + untuned), not **layout-level cost** (fences). A future ROCm release that speeds up `aiter::cross_device_reduce_1stage` and memory-bound NSA kernels under HBM contention would close more of the gap than reducing fence cost would.

---

## Bubble decomposition (kind classification) — what kind of GPU idle?

The +18 μs / layer bubble in MI355X dual-stream is significant only if it's *real CPU stall*. To distinguish real stall from per-kernel dispatch latency, we classify each inter-kernel gap inside a layer-pass:

- **`launch_micro`** (< 1 μs): per-kernel CP→SM dispatch latency. **Important caveat: see "Profiler-accounting caveat" below.**
- **`small_fence`** (1–2 μs): minor scheduling fence (e.g., a single `cudaEventSynchronize`).
- **`real_stall`** (≥ 2 μs): something on the host actually blocked the GPU dispatch queue. This is what the dual-stream regression analysis cares about.

Direct trace measurement, 15 consecutive layer-passes per trace:

| trace | layer wall | total bubble | sub-1 μs (see caveat) | 1–2 μs (small fence) | ≥ 2 μs (real stall) | verdict |
|---|---|---|---|---|---|---|
| MI355X · single-stream HIP (ROCm 7.2) | 271 μs | **0 μs** | 0 μs | 0 μs | 0 μs | clean |
| MI355X · dual-stream HIP (ROCm 7.2, 2 streams) | 320 μs | 18.2 μs | ~0 μs | 1.7 μs (1 fence) | **16.5 μs (3 stalls)** | real HIP-runtime overhead |
| B200 · CUDA-graph dual-stream | 303 μs | 23.4 μs | 23.4 μs (65 gaps) | 0 μs | **0 μs** | clean |
| **MI355X · dual-stream HIP (ROCm 7.13, 4 streams)** | 337 μs (this) / 366 μs (median) | 18.2 μs | ~0 μs | 1.9 μs (1 fence) | **16.3 μs (3 stalls)** | **bubble unchanged but kernel costs grew (AR / indexer / untuned GEMM); see "ROCm 7.13 + 4-stream regression" section** |

**The signal in this table is the right-most column** (≥ 2 μs real stall): MI355X dual-stream has 16.5 μs of real CPU stall per layer; both MI355X single-stream and B200 dual-stream have zero. The middle "sub-1 μs" column is misleading at first glance — see the caveat below.

### Profiler-accounting caveat (sub-1 μs column)

Initial reading: "B200 has 23 μs / layer of launch micro-gaps, MI355X has 0 — MI355X is more efficient at packing kernels back-to-back". **This is wrong.** It's a profiler accounting artifact, not a real GPU behavior difference.

Direct evidence from the raw timestamps in each trace:

| metric | MI355X (roctracer) | B200 (CUPTI) |
|---|---|---|
| total intra-stream gaps | 16,479 (baseline) | 38,074 |
| **negative gaps** (next-kernel start_ts < prev-kernel end_ts) | **1 outlier** | **4,296 (11.28%)** |
| smallest gap | −0.124 μs | **−6.656 μs** |
| sub-50 ns gaps | 99.47% | 0.71% |
| p50 gap | **0.001 μs (1 ns)** | **0.352 μs (~350 ns)** |
| ts fractional precision | 0.00098 / 0.00195 increments visible | 0.001 increments visible |

The smoking gun is **B200's 4,296 negative gaps** (gap as small as −6.66 μs). A negative gap means the next kernel's reported start ts is *before* the previous kernel's reported end ts. This is physically impossible if both timestamps reflect the same kind of physical event. The only explanation is that CUPTI uses different events for start vs end:

| | start ts source | end ts source |
|---|---|---|
| **CUDA / CUPTI** | first warp begins on SM | last warp completes on SM |
| **ROCm / roctracer** | AQL dispatch packet processed (CP layer) | kernel completion signal |

CP→warp scheduler→SM dispatch latency is ~200–500 ns on both platforms. CUPTI exposes this latency as inter-kernel gap (because end-of-CP-pipelined-dispatch can already be in the future relative to start-of-SM-execution); roctracer absorbs it into the kernel's reported duration. So:

- **MI355X kernels appear to pack with 0 gap** because their reported duration includes dispatch latency.
- **B200 kernels appear to have 350 ns gaps between them** because CUPTI separates SM-execution duration from CP-dispatch latency.

This means the "MI355X has 0 launch residue, B200 has 23 μs" finding is **not a GPU-utilization difference**. The actual per-kernel dispatch latency is similar on both platforms; only the bookkeeping differs. **Sub-1 μs gaps are not actionable** on either platform without kernel fusion (same per-kernel bookkeeping cost on both); they are not what dual-stream-regression analysis is about.

### What actually matters: ≥ 2 μs real stalls

The conclusion above (HIP-runtime overhead) does *not* depend on the sub-1 μs column. It depends on the ≥ 2 μs column:

1. **MI355X dual-stream's 16.5 μs is real ≥2 μs stalls.** Three distinct stall regions per layer:
   - **5.4 μs** stall before `q_b_proj` on cur, while `indexer_layernorm` is queued on alt — cross-stream `wait_event` fence.
   - **6.1 μs** stall after `main_kernel` (NSA paged_mqa); before MoE-block dispatch.
   - **5.8 μs** stall between MoE-AR end and next-layer dispatch — graph reentry overhead.
   
   These gap durations (5–6 μs each, consistently) are **far too long to be dispatch latency** (which is ~200–500 ns) and they don't show up in MI355X single-stream. They look like the cost of `hipStreamWaitEvent` flushing the dispatch queue before the next kernel can start.

2. **B200 dual-stream has 0 ≥2 μs stalls** (and no 1–2 μs fences either). Its dual-stream layout is clean — the CUDA-graph runtime / NCCL doesn't put cross-stream synchronization fences in the critical path.

3. **B200 has 4 lanes from CUDA-graph stream pool, MI355X dual has 2 lanes (cur + alt fixed).** B200's "dual-stream" is actually 4 stream-pool slots that the graph alternately dispatches into; lanes interleave (lane 0 → lane 1 → lane 2 → lane 3 → lane 0) rather than running concurrently in critical path. So a B200 layer has many more inter-kernel boundaries than an MI355X-dual layer — and CUPTI exposes each boundary as a sub-1 μs "gap", inflating the apparent total bubble. None of those are real stalls.

The conclusion stays: **HIP graph in dual-stream replay puts cross-stream synchronization fences in the critical path; CUDA / NCCL does not.** This is the same root cause class as the AR slowdown. A future ROCm release that reduces dual-stream graph fence cost would recover both simultaneously.

---

## Why dual-stream loses on MI355X

> **Honesty note**: The bench / per-kernel-duration deltas below are **measured**.
> The proposed *mechanisms* (HBM bandwidth, CU split, AR fence) are **hypotheses
> consistent with the data but not directly verified** with hardware counters.
> See "Verification needed" at the bottom.

### Hardware refresher (MI355X, from CDNA 4 whitepaper)

```
1 GPU package = 8 XCDs (TSMC 3nm) + 2 IODs (TSMC 6nm) + 8 HBM3E stacks
              = 256 active CUs (32 per XCD, 4 disabled per XCD for yield)
              = 32 MB L2 cache (4MB per XCD)
              = 256 MB Infinity Cache (in IODs, shared across all XCDs)
              = 288 GB HBM3E, 8 TB/s aggregate bandwidth
```

(Note: previously this analysis quoted 304 CUs / 6 TB/s — both wrong, MI300X
numbers misapplied. MI355X is 256 active CUs / 8 TB/s.)

GPU streams **share** all of the above. When two streams have kernels running
concurrently they compete for these finite resources.

### 1. HBM bandwidth contention (HYPOTHESIS — likely contributor to indexer slowdown)

Memory-bound kernels in the indexer chain (Hadamard, act_quant, paged_mqa,
topk_transform — all do little compute relative to data movement) read from
HBM. When dual-stream runs cur's q_b_proj GEMM concurrently with alt's wk
GEMM (also reading weights from HBM), aggregate demand may approach or exceed
the 8 TB/s pipe.

But "cache thrashing" (an earlier claim) is **probably not the dominant cause**
for GLM-5 decode: weights are multi-GB and don't fit in the 256 MB Infinity
Cache regardless of single/dual-stream. KV cache + activations are GB-scale
too. The cache mostly handles small intermediates, which fit either way.

**What we actually measured** (cur+alt running concurrently on stream 113/106 vs Baseline's single-stream phys 8):

```
                    Baseline (single)  Dual stream    Δ
indexer_layernorm      4.7 μs           5.8 μs    +1.1
hadamard               4.2              4.6      +0.4
hadamard               4.1              4.9      +0.8
act_quant              4.7              6.0      +1.3
indexer_k_quant        4.4              4.6      +0.2
wv_splitk (NSA score)  5.4              6.0      +0.6
paged_mqa              4.1              4.9      +0.8
topk_transform        15.2             17.6      +2.4
                                  ───────────
                                       +8.2 μs total
```

GEMMs (q_b_proj, wq_b, wk) are compute-bound and basically don't slow down (+0~0.1 μs); only memory-bound indexer kernels each pay 0.5-2.4 μs. The slowdown pattern is *consistent with* HBM contention but could also be e.g. memory-controller queue depth or HBM channel-bank conflicts. Verification needs hardware counters.

### 2. Infinity Cache thrashing (LIKELY NOT the cause, retracted)

An earlier draft of this doc claimed "dual-stream's two working sets evict
each other from the 256 MB Infinity Cache". On reflection that's likely
wrong for GLM-5 decode: weights are multi-GB and don't fit in cache anyway,
and per-kernel intermediates at decode batch=4 are small (KB to single-MB
scale). Single-stream and dual-stream both have the same cache miss profile
for these workloads.

### 3. Compute-Unit (CU) split (HYPOTHESIS — probably weak contributor)

MI355X has **256 active CUs** across 8 XCDs (32 each, 4 disabled). When
two kernels run concurrently, ROCm scheduler distributes workgroups across
CUs. *In principle* this could halve the CUs available per kernel.

In practice, small kernels (B=4 decode) likely use only a fraction of the
CUs anyway — they're launch / latency dominated, not CU-throughput
dominated. So the +1.5 μs main_kernel slowdown observed in the trace
**probably is not** caused by CU split.

What might it be instead? Some possibilities:
- L1/LDS contention between concurrent waves
- Shared instruction cache pressure
- Wavefront-level scheduling artifacts
- Cumulative effect of (1) — main_kernel does some HBM access too

Honestly, **this needs `rocprof-v3` HW counters to attribute properly**.

### 4. AllReduce slowdown (+7.98 μs / call mean) — VERIFIED, mechanism is HYPOTHESIS

**What was directly measured**:
- Same `aiter::cross_device_reduce_1stage<bf16, 8 ranks>` kernel
  (byte-identical mangled name)
- Same MI355X hardware, same aiter build (PR #2879)
- Same TP=8 topology, same bench input sizes
- Single-stream HIP graph: AR mean ≈ 10.3 μs (median 10.0, p10–p90 7.4–13.4)
- Dual-stream HIP graph:  AR mean ≈ 18.2 μs (median 17.5, p10–p90 12.6–24.4)
- → **+7.98 μs / call (mean) duration difference is real**, and the entire
  distribution shifts right (no bimodal — every AR is slower, not just a tail).
  The slowdown disappears when alt_stream is set to None (i.e., dual-stream off).

**What was NOT verified, just hypothesized**:

The proposed mechanism — that "the AR's peer-fence has to drain alt's
KV-cache writes, costing extra time attributed to the AR kernel" — is
**guess-work**. Other plausible mechanisms include:

- HIP-graph dual-stream replay scheduler emits extra event signal/wait
  nodes that gate the AR start, padding its measured duration.
- aiter's signal-buffer poll loop in the cross_device_reduce inner loop
  takes longer when peer GPUs are busier (e.g., still finishing alt-stream
  kv_cache writes).
- Some ROCm runtime quirk specific to graph capture under dual-stream that
  `rocprof` would reveal but I haven't run.

To verify the actual mechanism:
1. Run dual-stream + `--disable-cuda-graph` → if AR is fast, mechanism is
   graph-specific. If still slow, mechanism is dual-stream-runtime specific.
2. Profile AR with `rocprof-v3 --hsa-trace` to see kernel start/end vs
   stream queue depth.
3. Read the aiter `cross_device_reduce_1stage` source to see fence
   sequence and check whether there's a per-stream barrier.

This is two ARs per layer (attn AR + MoE AR), so 2 × +7.98 μs ≈ **+16 μs/layer**
that disappears when dual-stream is off.

### 5. GPU bubble (+18.2 μs / layer) — VERIFIED, mechanism is HYPOTHESIS

**What was directly measured** (15 consecutive layer-passes per trace, layer wall − union of busy intervals across all streams):

- Baseline (single-stream HIP graph): **~0 μs / layer** (0.04 μs over 15 layers — GPU queued back-to-back, essentially saturated)
- Dual stream (HIP graph): **+18.2 μs / layer** (samples 11.8 … 24.1 μs, stable, not outlier-driven)
- Bubble alone is **5.7%** of the dual-stream layer wall-clock — the GPU is literally idle for that fraction of every layer

**Cross-platform reference (B200, NVIDIA)**:

To rule out "this 18 μs is just launch-overhead surfacing", we ran the same per-layer bubble analysis on a B200 GLM-5-FP8 decode trace (TP=8, conc=4, in1000/out1000, CUDA graph, single-stream) over 173 steady-state layers in a middle window. Critically, we also broke the bubble down by gap size:

| Variant | layer wall | GPU bubble | sub-1 μs gaps (launch micro-overhead) | ≥2 μs gaps (real CPU stall) |
|---|---|---|---|---|
| MI355X single-stream + HIP graph | 271.0 μs | 0.04 μs (~0%) | ~0 μs | 0 μs |
| MI355X dual-stream + HIP graph | 320.0 μs | **18.2 μs (5.7%)** | n/a (not separately decomposed) | n/a |
| B200 single-stream + CUDA graph | 302.7 μs | 23.4 μs (7.7%) | **23.4 μs (100%)** | **0 μs** |

**Key finding from the B200 trace: the entire 23.4 μs / layer bubble is sub-1 μs micro-gaps between consecutive kernels — there are zero ≥2 μs gaps anywhere in steady-state decode.** That is, B200's bubble is purely the residue of CUDA graph kernel launches not being perfectly back-to-back; the CPU is never stalling the GPU.

This recasts the MI355X result:

- HIP single-stream packs kernels even tighter than CUDA does (~0 μs vs B200's 23 μs of micro-gaps), so on single-stream both platforms are essentially saturated.
- The MI355X dual-stream **+18.2 μs bubble is therefore not generic launch overhead** — single-stream HIP demonstrates the GPU *can* run truly back-to-back. Something about dual-stream HIP-graph capture introduces wall-clock idle that single-stream HIP does not.

**Mechanism (HYPOTHESIS)**:

HIP-graph dual-stream replay likely emits cross-stream `wait_event` nodes at every fork/join. Each fence flushes the dispatch queue for ~1–6 μs depending on what the other stream is doing. A representative dual-stream layer has 4 distinct bubble regions, each 1.6–6.1 μs, summing to ~19 μs:

- Bubble #1 (~1.6 μs): between attn-block kernels
- Bubble #2 (~5.4 μs): before `q_b_proj` on cur, while `indexer_layernorm` queued on alt — cross-stream `wait_event`
- Bubble #3 (~6.1 μs): after NSA `paged_mqa`, before MoE-block dispatch
- Bubble #4 (~5.8 μs): between MoE-AR end and next-layer dispatch — graph reentry overhead

This is not directly verified. To verify:
1. `--disable-cuda-graph` ablation on dual-stream — if bubble disappears, mechanism is graph-specific.
2. `rocprof-v3 --hsa-trace` to inspect HIP graph node dispatches and event waits at μs resolution.
3. The bubble likely shares root cause with §4 (AR slowdown) — both surface only under dual-stream HIP graph capture.

---

## Total accounting (Dual stream vs Baseline) — wall-clock

All numbers below are **trace-direct measurements** averaged over 15 consecutive layer-passes from each trace.

| source | μs / layer | how measured |
|---|---|---|
| GPU bubble (HIP-graph dual-stream artifact, GPU idle) | **+18.2** | wall − union(busy intervals) over 15 layers |
| AR slowdown (2 ARs / layer × +7.98 μs / call mean) | **+16.0** | 781-call mean from each trace |
| MoE block wall-clock widens (unfused shared experts, +46 μs kernel-duration mostly hidden by overlap) | **+8.8** | per-block wall-clock over 20 MoE blocks |
| Indexer chain HBM contention (memory-bound kernels) | **+7.0** | per-fire normalized over 9 kernels × 15 layers |
| Other attention kernels (main_kernel, bmm_w_vc, …) | +3 to +5 | not separately measured here |
| Theoretical dual-stream saving (gap-fill ∥ indexer overlap) | already reflected in MoE-block & indexer wall numbers above | — |
| **Sub-total** | **~+50 μs** | |
| **Real-layer regression measured in trace** | **+49 μs** ✓ | layer wall delta over 15 layer-passes |

The four near-equal contributors (bubble +18, AR +16, MoE +9, indexer +7) sum to +50, which closely matches the observed +49 μs / layer wall-clock regression. No "unaccounted CPU overhead" needed — the bubble IS the CPU/scheduler overhead, and it's directly visible in the trace.

The bench TPOT regression of 3.24 ms / token corresponds to **~+50 μs / layer at TPOT level**, which now matches the trace-measured +49 μs at the GPU level — no separate CPU-overhead term needed.

Note that the **kernel-duration sum delta** (sum every kernel's individual duration, ignoring overlap) is ~**+46 μs in the MoE block alone**, and roughly **+95 μs total over the layer**. The wall-clock regression is smaller (+49 μs) precisely because dual-stream is overlapping work — that's the optimization's intended effect. But the AR slowdown and the bubble are fixed-cost and don't overlap, so the layout still loses on net.

---

## Recommendation

Dual-stream layout should remain **opt-in via env var** (default OFF) on HIP, not enabled by default. Four near-equal contributors (each ~+7 to +18 μs / layer) cost ~+49 μs total:

1. **GPU bubble +18 μs / layer** — HIP graph capture under dual-stream introduces ~5.7% GPU-idle time per layer (Baseline GPU is 100% saturated). Mechanism likely cross-stream event fences during graph replay. Verifying with `--disable-cuda-graph` ablation would confirm.
2. **AllReduce slowdown +16 μs / layer** — `aiter::cross_device_reduce_1stage` is +7.98 μs / call slower under dual-stream HIP-graph capture (10.3 → 18.2 μs / call, 2 calls / layer). Fixed cost.
3. **MoE block wall-clock +9 μs** — `--dual-stream-rocm` implies `--disable-shared-experts-fusion`, forcing aiter to split shared experts into 4 kernels totaling ~+46 μs of kernel-duration. Most is overlapped with the routed-expert chain, so only ~+9 μs surfaces at wall-clock — but it still adds.
4. **Indexer chain HBM contention +7 μs** — 9 memory-bound NSA kernels each pay 0.7–1.0 μs in dual-stream when sharing HBM with concurrent cur-stream GEMMs.

The theoretical dual-stream gain (gap-fill ∥ indexer ≈ −10 μs of overlap saving) is already reflected in the wall-clock numbers above; it does land but is dwarfed by the four cost contributors.

The layout should be revisited when:
- A future ROCm release fixes the HIP-graph AR fence cost AND/OR reduces the dual-stream bubble. These two together explain ~70% of the regression and are pure runtime artifacts.
- aiter ships a fused shared-expert kernel that's compatible with the dual-stream MoE path (keep `_fused_append_shared_experts_kernel` even when `--disable-shared-experts-fusion`).

---

## Verification needed

The claims in §1, §3, §4-mechanism above are *hypotheses* consistent with the
observed kernel slowdowns but **not** verified with hardware counters or
source inspection. Concrete next steps:

1. **rocprof-v3 hardware counters** on Baseline vs dual stream for one decode step:
   - HBM bytes / cycle per kernel (verifies §1)
   - CU active rate / wavefront occupancy per kernel (verifies §3)
   - LDS / L1 / L2 hit rates (sanity check §2)
2. **`--disable-cuda-graph` ablation** on dual-stream variant — if AR is fast
   in eager mode, §4 is graph-specific; otherwise dual-stream-runtime specific.
3. **aiter source inspection** — read `cross_device_reduce_1stage` to see
   fence sequence, peer signal protocol, etc.
4. **Larger batch ablation** — at conc=64 indexer is longer absolutely, dual-
   stream might recover ROI even with the contention costs. Not measured.
5. **Future ROCm releases** (≥ 7.3?) may fix HIP graph dual-stream scheduling.
   Re-run periodically.
