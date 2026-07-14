# GLM-5.2 DSA Dense-Decode (skip-indexer) — Handoff for Design A

**Goal of the next step (Design A):** make the decode "skip-indexer / k-only" fast
path work **under CUDA graph for arbitrary/unknown context length** by capturing
**two graph variants per bs bucket** (`dense` = k-only, `sparse` = full indexer)
and selecting which to replay on the host, per forward pass, from `max_kv_len`.

---

## 0. Which branch to continue on

- **Repo (fork):** `Jacob0226/sglang`
  - SSH: `git@github.com:Jacob0226/sglang.git`
  - HTTPS (machines without SSH keys): `https://github.com/Jacob0226/sglang.git`
- **Continue on branch:** `jacob/dense-decode-konly`  ← START HERE
  - Tip commit: `5a17402ce` `[dense-decode][M2a] static SGLANG_DSA_DECODE_DENSE_GRAPH flag ...`
  - Contains M1 (eager-correct decode k-only) + M2a (static cuda-graph flag). This
    handoff doc lives at repo root of this branch.
- **Base lineage** (already merged into this branch, `4a109dee1`):
  `jacob/glm-moe-shared-fusion` + `jacob/glm-mla-fp8-absorbed-bmm` (= PR #30519).
- **Benchmark harness** (GLM.sh + tools + analysis scripts) is on a separate
  snapshot branch: `jacob/sglang-benchmarks-snapshot` (unrelated history; contains
  `GLM.sh`, `analyze_trace.py`, `compare_breakdown.py`, `tools/konly_*`,
  `tools/triton_dsa_*`, etc.). Fetch it into `~/SGLang-benchmarks/` on the new box.
- **Do NOT reuse** `jacob/spike-i1k-dense-decode` — that is the OLD broken
  `arange()` identity approach (GSM8K 0.02). Superseded by this branch.

```bash
git clone https://github.com/Jacob0226/sglang.git
cd sglang && git checkout jacob/dense-decode-konly
```

---

## 1. Background & the key correctness lesson

GLM-5.2 = DeepSeek Sparse Attention (DSA / NSA), `index_topk = 2048`, page_size=64,
78 layers (full layers run the indexer; ~1/4 are "shared" layers that reuse the
previous full layer's top-k indices). When a request's `kv_len <= index_topk`, the
top-k "selects all" valid positions, so the indexer work (fp8 index GEMM +
`paged_mqa_logits` + `topk_transform`) is **wasted** — decode can skip it.

**CRITICAL correctness lesson (why the earlier spike failed):**
With the default `SGLANG_DSA_FUSE_TOPK=true`, `topk_indices` are **physical page
slots**, NOT logical positions. The old spike synthesized `arange(0..kv_len-1)`
(logical) → wrong slots when KV sits at a page offset (e.g. `[64..70]` not
`[0..6]`) → GSM8K 0.02.
**The correct mechanism** is to route the indexer through its existing
`_forward_cuda_k_only()`, which builds select-all indices via
`topk_transform(dummy_logits)` → returns the correct **physical** slots
(offset-safe). Do NOT synthesize indices in the attention backend.

---

## 2. What is already DONE (M1 + M2a)

All changes are in ONE file: `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`.

### M1 — decode k-only, eager-correct  (commits `9009008f9`, `0a2c6c585`)
- `_should_skip_logits_computation(...)` (~line 536): now returns True for
  `is_decode_or_idle()` too, gated on `max_kv_len <= index_topk`.
  - **Eager**: per-step host check on `seq_lens_cpu` (or `seq_lens.max()` fallback)
    → correct for BOTH `<=2048` (k-only) and `>2048` (falls through to full
    indexer / sparse).
  - **Capture mode** (`get_is_capture_mode()`): returns the M2a env flag (below);
    does NOT host-sync (that would break capture).
- `_forward_cuda_k_only(...)` (~line 1339): relaxed the `assert is_extend...` to
  also accept `is_decode_or_idle()`.
- Env-gated debug marker `SGLANG_KONLY_DEBUG=1` logs `[KONLY] decode k-only fired`
  (layer 0, once per decode step) to confirm the path is active.

**M1 validated (eager, `--disable-cuda-graph`):** GSM8K 400q = **0.943**, marker
fired 1964×. Correct for variable length.

### M2a — static cuda-graph flag  (commit `5a17402ce`)
- Env `SGLANG_DSA_DECODE_DENSE_GRAPH=1`: under capture, force the decode k-only
  branch (captures the dense graph). **Only safe when the deployment guarantees
  every request's kv_len <= index_topk** (e.g. i1k/o1k). WRONG for mixed >2K
  traffic (a captured k-only graph replays k-only for all lengths). Default 0.

**M2a measured (cuda graph, i1k/o1k, tilelang decode, A/B by the flag):**

| conc | baseline TPOT | dense TPOT | Δ |
|---|---|---|---|
| 4  | 12.05 | 11.43 | −5.1% |
| 8  | 13.88 | 13.16 | −5.2% |
| 16 | 17.95 | 17.16 | −4.4% |
| 32 | 22.93 | 22.11 | −3.6% |
| 64 | 31.12 | 30.19 | −3.0% (thr +2.7%) |

Note: the real gain is a modest ~3–5% TPOT (NOT the spike's inflated/broken ~17%),
because (a) only ~1/4 full layers skip the indexer, (b) k-only still does K-store +
dummy `topk_transform`, (c) the indexer is partly overlapped on `alt_stream`.

---

## 3. TASK: Design A — dual-variant cuda graph with host dispatch

**Why not a single self-switching graph:** a captured graph is static; you cannot
put a data-dependent `if kv_len>2048` inside it. **The switch must happen on the
host, choosing WHICH pre-captured graph to replay.** We do NOT need to know the
length at capture time — only at replay time, and the host already knows the
current batch's `max_kv_len` then.

**Design A:** capture two variants per bs bucket and dispatch:
```
per decode step (host, before replay):
    m = max_kv_len(this batch)          # from seq_lens_cpu
    replay(dense_graph[bs])  if m <= index_topk   # k-only, saves indexer
    else replay(sparse_graph[bs])                 # full indexer, correct
```
- **Rejected: Design B** (single dense graph + eager fallback for >2K). User
  rejected it: they run input-8K cases (kv_len ≫ 2048) that would fall to eager =
  far too slow.
- With Design A, the **8K case always replays the sparse graph** (full graph speed,
  correct, never eager). 1K case replays dense (saves indexer). Both stay on graph.

### Correctness requirements
1. **Batch-max, not per-request.** A decode batch mixes lengths; dispatch on
   `max(seq_lens)`. If any request > index_topk → use sparse graph (dense would be
   wrong for the long one). Short requests in a mixed batch temporarily lose the
   dense speedup — acceptable. (Optional later: length-bucketing in the scheduler
   for homogeneous batches.)
2. **Capture both variants.** Capture dense with the k-only branch active and
   sparse with the full indexer, per bs bucket.
3. **`seq_lens_cpu` must be available at decode replay.** DSA backend currently
   sets `needs_cpu_seq_lens=False` (`dsa_backend.py` ~306-309). Either set it True
   (a cheap per-step d2h copy of the seq_lens vector) or reuse whatever host-side
   length info the runner already has for bucketing.

### Hooks to modify (file:line approximate)
- `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`
  - `variant_label` (~384-388, 735-743): today only `"lora"`/`"nolora"`. Add
    `"dense"`/`"sparse"`. Graph key is `(bs, stream_idx, variant_label)`
    (`shape_key.py: _make_graph_key`).
  - `capture_one_shape` (~745-847): capture a `dense` variant (with
    `SGLANG_DSA_DECODE_DENSE_GRAPH` effectively on / the k-only branch forced) AND
    a `sparse` variant (full indexer) per bs.
  - `can_run_graph` (~400-469): select variant from `max(seq_lens_cpu)` vs
    `index_topk`; ensure a matching captured graph exists, else fall back
    gracefully (should not happen if both variants captured for all bs).
- `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`
  - `_should_skip_logits_computation` capture-mode branch (~556-566): instead of a
    single global env flag, the k-only-vs-full decision under capture must be
    driven by **which variant is currently being captured** (thread a
    capture-variant signal, e.g. a context var / attr on the runner, rather than
    the static env). During "dense" capture → return True; during "sparse"
    capture → return False.
- The attention backend (`dsa_backend.py`) needs **no** identity synthesis; it
  already consumes the indexer's `topk_indices` correctly (`_get_fused_topk_page_table`
  passthrough when `SGLANG_DSA_FUSE_TOPK=true`).

### Memory / capture-time cost (GLM-5.2 FP4, TP4) — estimated
- Decode bs buckets today = **52** (`bs=[1,2,4,8,12,...,512]`). Dual variant →
  **104** graphs (+52).
- Extra memory is **small (~tens to ~150 MB, negligible vs 192/288 GB)** because:
  - Model weights / KV cache: graph-external, unchanged.
  - Activation/capture **memory pool is shared** across all decode graphs; the
    dense path's peak ⊆ the sparse path's, so the shared pool does not grow.
  - Static input/metadata buffers (`page_table`, `cache_seqlens`, input_ids, …)
    are allocated once and shared by all bs/variants.
  - The only real increase is the **captured kernel-DAG metadata** for +52 graphs.
- Secondary cost: **capture time ~2×** (~33s → ~66s), one-time at startup.
- ACTION on new box: measure exactly via `torch.cuda.memory_reserved()` before/
  after capture, or the server log `available_gpu_mem` delta.

### Validation plan
1. Eager already proves per-step correctness (GSM8K 0.943). Keep as regression.
2. After Design A: **GSM8K under CUDA graph** (mixed lengths incl. >2K) must stay
   ≥ 0.92 — this proves variant selection routes >2K to sparse.
3. Perf A/B under graph: i1k/o1k (dense wins ~3–5% TPOT) AND i8k (must match
   baseline exactly — always sparse, no regression, never eager).

---

## 4. Environment / how to run (reproduce on new box)

- Docker image used: `rocm/sgl-dev:v0.5.14-rocm720-mi35x-20260708` (MI355X gfx950).
- Model: `/data/huggingface/hub/amd/GLM-5.2-MXFP4`, TP4.
- Required env (see `GLM.sh`): `SAFETENSORS_FAST_GPU=1`,
  `SGLANG_ROCM_FUSED_DECODE_MLA=0`, `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`,
  `SGLANG_OPT_USE_TOPK_V2=0` (0708 image: topk_v2 JIT needs `cooperative_groups.h`
  not in ROCm 7.2 → disable).
- Server args (MI355X): `--dsa-prefill-backend tilelang --dsa-decode-backend tilelang
  --kv-cache-dtype fp8_e4m3 --enable-aiter-allreduce-fusion --chunked-prefill-size 16384
  --mem-fraction-static 0.85 --disable-radix-cache --tp 4`.
- Benchmark harness (`~/SGLang-benchmarks`, branch `jacob/sglang-benchmarks-snapshot`):
  - `GLM.sh` has an added `DSA_DECODE_BACKEND` / `DSA_PREFILL_BACKEND` env override.
  - `tools/konly_eager_val.sh` — eager GSM8K correctness (M1).
  - `tools/konly_graph_ab.sh` — cuda-graph A/B by `SGLANG_DSA_DECODE_DENSE_GRAPH`.
  - `SGLANG_KONLY_DEBUG=1` prints `[KONLY] decode k-only fired` to confirm activation.
  - `analyze_trace.py` / `compare_breakdown.py` for kernel-level trace analysis.

⚠️ ROCm note: do NOT `pkill -9` GPU server processes — it can trigger 100–200 GB
gpucore dumps that fill the shared disk. Use graceful kill (SIGTERM) or the driver
scripts' cleanup.

---

## 5. TL;DR for the next machine
1. `git checkout jacob/dense-decode-konly` (M1+M2a already there, correct approach).
2. Implement Design A: dual `dense`/`sparse` graph variants in
   `decode_cuda_graph_runner.py` + host dispatch on `max_kv_len` vs `index_topk`;
   thread the capture-variant signal into `_should_skip_logits_computation`.
3. Ensure `seq_lens_cpu` available at decode (needs_cpu_seq_lens).
4. Validate: GSM8K under graph ≥0.92 (mixed lengths) + i1k dense ~3–5% TPOT + i8k
   no regression (always sparse, never eager).
5. Expected extra cost: ~tens–150 MB, capture ~2×; both negligible/one-time.
