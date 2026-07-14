# SGLang GLM-5.2 (DSA) fp8 MLA decode — port spec

Goal: make SGLang's GLM-5.2-MXFP4 decode use the aiter **fp8 MLA decode** kernel
(`aiter.mla.mla_decode_fwd`), matching ATOM, instead of the bf16 tilelang
`main_kernel` (×2). This is the remaining gap to ATOM after opt#1.

Measured (MI355X TP4, GLM-5.2-MXFP4, i1024/o1024 conc4, median TPOT):
- SGLang baseline (tilelang decode): 15.66 ms, GSM8K 0.940
- SGLang + opt#1 (fp8 absorbed bmm): 14.89 ms, GSM8K 0.929
- ATOM (aiter fp8 decode): 11.73 ms  ← target

Env: container `jacchang_GLM5`, image `rocm/sgl-dev:v0.5.14-rocm720-mi35x-20260628`,
SGLang editable at `/sgl-workspace/sglang` (branch main), aiter at `/sgl-workspace/aiter`.
Repro test: `~/run_sgl_test.sh <tag>` (server GLM.sh MI355X args → GSM8K 1319 → conc4 TPOT).
Trigger the path with `--nsa-decode-backend aiter` (aka `--dsa-decode-backend aiter`).

--------------------------------------------------------------------------------
## Status: aiter DSA decode path is unfinished/broken in SGLang

`dsa_backend.py::DSAAttnBackend._forward_aiter` GPU-memory-faults (both cuda-graph
capture AND eager `--disable-cuda-graph`). Instrumented args right before the
`mla_decode_fwd` call (decode warmup, bs=8, 1 tok/req):

    q_kernel = (8,16,576) float8_e4m3fn
    o_kernel = (8,16,512) float8_e4m3fn        # BUG: should be bf16
    kv_view  = (2634752,1,1,576) float8_e4m3fn
    cu_seqlens_q = (9,): [0,1,2,3,4,5,...]
    kv_indptr    = (2049,): [0,7,14,21,28,35,...]   # COMPACTED, variable length
    kv_indices   = (4194304,) , kv_max=518 , pool=2634752
    persist keys = work_meta_data, work_indptr, work_info_set, reduce_indptr,
                   reduce_final_map, reduce_partial_map, intra_batch_mode, num_kv_splits=64
    topk = 2048 , max_seq_len_q = 1
    q_scale = None      # BUG
    kv_scale = nan      # BUG (should be 1.0)

Blocker chain observed while patching:
1. q is fp8 but `q_scale=None` → aiter asm aborts: "fp8 Q requires q_scale and kv_scale".
2. Set q_scale=ones → `o=empty_like(q)`=fp8 → "kn_mla_reduce_v1 doesn't support output type Float8_e4m3fn".
3. Force o=bf16 + q_scale=ones → GPU **memory access fault** inside mla_decode_fwd
   (eager too), i.e. the index/metadata layout does not match the kernel contract.

--------------------------------------------------------------------------------
## The two incompatible schemes

### SGLang (current, faults) — dsa_backend.py
- `_forward_aiter` (~L2160-2245):
  - q_all fp8 → q_kernel fp8; `o = torch.empty_like(q)` (inherits **fp8**).
  - `q_scale=None`; `kv_scale=torch.ones(())` when fp8 (empirically prints nan).
  - `get_valid_kv_indices(page_table_1, kv_indptr, kv_indices, bs)` →
    **compacted variable-length** kv_indices + kv_indptr (drops -1).
  - `_prepare_aiter_dsa_decode_metadata` (~L513-566) → `get_mla_metadata_v1(...,
    topk=dsa_index_topk, max_split_per_batch=64, page_size=1, kv_granularity=16,
    intra_batch_mode=True)`; buffers from `_make_aiter_dsa_decode_metadata_buffer`
    (~L429) via `get_mla_metadata_info_v1`.
  - `mla_decode_fwd(q_kernel, kv.view(-1,1,1,head_dim), o_kernel, cu_seqlens_q,
    kv_indptr, kv_indices, kv_last_page_lens, max_seq_len_q, q_scale=None,
    kv_scale, **persistent)` (~L2229).

### ATOM (working) — atom/model_ops/attention_mla.py + plugin/sglang/attention_backend/sparse_mla_indexer.py
- Indexer `sparse_attn_indexer_sglang_plugin_mode` (sparse_mla_indexer.py L386):
  fills **fixed `topk_indices_buffer[:bs, :2048]` (-1 padded)** via
  `deepgemm_fp8_paged_mqa_logits` + `top_k_per_row_decode` (L497-505);
  `seq_lens_i32 = forward_batch.seq_lens[:bs]`.
- Decode `_forward_decode` (attention_mla.py L954-1119):
  - `o = torch.empty(B, padded_num_heads, kv_lora_rank, dtype=self.dtype)` → **bf16**.
  - `_q_scale = _k_scale = one_scale = torch.tensor(1.0)` (L238-240).
  - sparse: `paged_kv_indices = self.sparse_kv_indices_buffer` (L869-872) — the
    **fixed [tokens, 2048]** buffer (NOT compacted); `paged_kv_indptr = sparse_kv_indptr`.
  - `mla_decode_fwd(q, kv.view(-1,page_size,1,dim), o, cu_seqlens_q, kv_indptr,
    sparse_kv_indices_buffer, kv_last_page_lens, max_q_len, page_size=page_size,
    num_kv_splits=None if seg else 16, sm_scale, work_meta_data/work_indptr/
    work_info_set/reduce_indptr/reduce_final_map/reduce_partial_map,
    q_scale=one, kv_scale=one)` (L1092-1114).

--------------------------------------------------------------------------------
## Incompatibility points (root causes)

| # | Item | SGLang (faults) | ATOM (works) | Fix |
|---|------|-----------------|--------------|-----|
| 1 | **sparse index layout** | compacted var-len flat + indptr (`get_valid_kv_indices`) | **fixed `[tokens, 2048]` (-1 padded)** + per-token `seq_len` | pass the fixed topk buffer (SGLang `page_table_1` is already `[bs, topk]` -1-padded, used raw by tilelang) instead of compacting |
| 2 | **metadata builder** | `get_mla_metadata_v1` from compacted indptr, splits=64 | `get_mla_metadata` from fixed buffer + seq_len, splits=16 | build metadata from the fixed buffer/seq_len the way the kernel expects |
| 3 | **output dtype** | `o = empty_like(q)` → **fp8** | `o` = **bf16** (`self.dtype`) | force o bf16 (`kn_mla_reduce_v1` rejects fp8 out) |
| 4 | **q_scale** | **None** (never set) → asm abort | `one_scale` (1.0) | pass q_scale=1.0 |
| 5 | **kv_scale** | empirically **nan** | 1.0 | ensure kv_scale=1.0 |
| 6 | num_kv_splits | 64 | 16 (non-seg) | match kernel/buffer sizing |

**CORRECTED primary suspect (after reading ATOM `attention_mla.py::_forward_decode`
and `attentions/aiter_mla.py`):** the earlier "fixed `[bs,topk]` vs compacted
buffer" theory is **WRONG**. ATOM's sparse fp8 decode also feeds the aiter
`mla_decode_fwd` a **compacted CSR** buffer — `sparse_kv_indptr` +
`sparse_kv_indices` (`self.sparse_kv_indices_buffer`) — the *same* ragged format
SGLang produces via `get_valid_kv_indices`. Both call the exact same
`aiter.mla.mla_decode_fwd(q, kv, o, qo_indptr, kv_indptr, kv_indices,
kv_last_page_lens, ...)`.

The real divergence is the **persistent work-scheduling metadata**:
- ATOM builds `work_meta_data / work_indptr / work_info_set / reduce_indptr /
  reduce_final_map / reduce_partial_map` via aiter `get_mla_metadata_v1` /
  `get_mla_metadata_info_v1` (see `atom/model_ops/attentions/aiter_mla.py`),
  with `num_kv_splits=16` on the page_size=1 persistent path (or `None` on the
  seg / `ATOM_MLA_PAGE_SIZE>1` path), and `q_scale=kv_scale=1.0`.
- SGLang builds its own in `_prepare_aiter_dsa_decode_metadata`. When forced onto
  the fp8 branch it produced an OOB — i.e. its split/worker-buffer sizing for the
  **topk-clipped** sparse seqlens (`dsa_cache_seqlens_int32`) does not match what
  the asm kernel indexes. That mismatch (not the index buffer) is the fault.

--------------------------------------------------------------------------------
## Recommended fix

Keep SGLang's compacted `get_valid_kv_indices` (kv_indptr + kv_indices) — it is
already the right format. Only replace the **metadata build**:
1. Replace `_prepare_aiter_dsa_decode_metadata` with a faithful port of ATOM's
   `aiter_mla.py` persistent-buffer build: call `get_mla_metadata_v1` /
   `get_mla_metadata_info_v1` with the compacted `kv_indptr`, the topk-clipped
   `dsa_cache_seqlens_int32`, and `num_kv_splits=16` (page_size=1 path).
2. Pass the resulting `work_meta_data / work_indptr / work_info_set /
   reduce_indptr / reduce_final_map / reduce_partial_map` into `mla_decode_fwd`.
3. `q_scale=kv_scale=torch.ones((), fp32)`; `o` bf16.
4. Gate on `ATOM_MLA_PAGE_SIZE` (=1 → persistent 16-split path; >1 → seg path
   which ATOM uses with `num_kv_splits=None` and padded q row stride).

**Needs HIP-level debugging** (rocgdb / AMD_LOG_LEVEL / bounds) to confirm the
kernel's exact index contract, since the fault is an opaque asm OOB with no line
info. Reference: `atom/model_ops/attention_mla.py::_forward_decode` and
`atom/plugin/sglang/attention_backend/sparse_mla_indexer.py`.

--------------------------------------------------------------------------------
## GLM-5.1 vs 5.2 indexer / layer design (from config.json, verified)

The DSA (NSA) sparse indexer is the biggest architectural difference between the
two model revisions, and it directly changes how many topk-selection kernels run
per decode step.

| key | GLM-5.1 | GLM-5.2 | meaning |
|---|---|---|---|
| `indexer_types` | (absent → all `full`) | `['full','full','full','shared','shared','shared','full','shared', ...]` len 78 | 5.2 marks most layers `shared` |
| `index_topk_freq` | (absent → every step) | `4` | 5.2 recomputes topk only every 4th decode step |
| `index_skip_topk_offset` | (absent) | `3` | phase offset for the freq schedule |
| `index_share_for_mtp_iteration` | (absent) | `True` | MTP/nextn reuses the base step's topk |
| `mlp_layer_types` | (implicit via `first_k_dense_replace=3`) | `['dense','dense','dense','sparse', ...]` | layers 0–2 dense, 3–77 MoE (same in both) |
| `index_topk` | 2048 | 2048 | selected KV budget per query, unchanged |

**`full` vs `shared` indexer layer:**
- `full`  = the layer runs the indexer itself (wq_b / weights_proj →
  `deepgemm_fp8_paged_mqa_logits` → `top_k_per_row`/radix topk) to *produce* the
  `[bs, topk]` selection.
- `shared` = the layer does **not** recompute topk; it **reuses** the selection
  produced by the nearest preceding `full` layer. So the indexer-selection
  kernels are simply absent from a `shared` layer's decode step.

**Net effect (5.1 → 5.2):**
- 5.1: every one of the 75 sparse layers runs a full indexer topk **every decode
  step** → 75 × topk selections/step.
- 5.2: only `full` layers run it, and only every `index_topk_freq=4` steps →
  roughly an order of magnitude fewer `deepgemm_fp8_paged_mqa_logits` / topk
  kernels per step. This is the main reason 5.2's decode is cheaper on the
  indexer side while keeping the same `index_topk=2048` accuracy budget.

**Consequence for the side-by-side excel:** the profiled layer is **layer 3**,
which is `indexer_types[3]='shared'` and `mlp_layer_types[3]='sparse'` — i.e. a
**MoE layer with a shared (non-recomputing) indexer**. That is why its MLA
section shows no indexer-topk kernels. A `full` layer (e.g. layer 6) on a
recompute step would additionally show the wq_b / paged_mqa_logits / topk
kernels, adding cost that is *not* visible in the current sheet. For a port,
both paths must be handled: `full` (produce + write selection) and `shared`
(read selection).

--------------------------------------------------------------------------------
## Pinpointed divergence (the actual next experiment)

SGLang and ATOM decode call the **same** `aiter.mla.get_mla_metadata_v1` + the
**same** `mla_decode_fwd` on **compacted CSR** (`clip(context_len, topk)` cumsum).
Diffing the two runtime `get_mla_metadata_v1` invocations:

| arg | SGLang `_prepare_aiter_dsa_decode_metadata` | ATOM decode (`aiter_mla.py:1331`) |
|---|---|---|
| `intra_batch_mode` | **`True`** | **not passed (False)** |
| `topk` | **`self.dsa_index_topk` (2048)** | **not passed** |
| `nhead_kv` (arg 5) | `1` | `1` |
| `page_size` | `1` | `self.block_size` |
| `kv_granularity` | `16` | `max(block_size,16)` |
| `max_split_per_batch` | `aiter_dsa_max_split_per_batch` | `16` |
| `num_kv_splits` returned | `max_split_per_batch` | `16` (kernel arg) |

**Hypothesis:** `intra_batch_mode=True` + `topk=2048` puts the asm kernel into a
sparse "intra-batch" schedule that assumes a **topk-strided** work layout, but
the indices/indptr handed in are **compacted** → out-of-bounds. ATOM's working
decode does NOT use that mode.

**First experiment (small, not a rewrite):** in SGLang's
`_prepare_aiter_dsa_decode_metadata`, set `intra_batch_mode=False` and drop
`topk=` (match ATOM), keep `num_kv_splits=16`, `q_scale=kv_scale=1.0`, `o` bf16;
enable the fp8 decode branch and validate with GSM8K + TPOT. If the kernel needs
`intra_batch_mode=True`, then instead feed a **topk-strided** kv_indices buffer
(`page_table_1` flattened, size `tokens*topk`, `-1`→0 padded) with a fixed
`kv_indptr` stride of `topk` — the layout `intra_batch_mode` actually expects.

**Reframing:** SGLang already contains ATOM's MLA kernel + metadata builder, so
this is NOT a subsystem port — it is aligning ~2 metadata args on the fp8 sparse
decode branch. Much smaller than the spec's original "replace everything".

--------------------------------------------------------------------------------
## Done in this investigation
- opt#1 (fp8 MLA absorbed bmm for GLM) — landed on branch
  `jacob/glm-mla-fp8-absorbed-bmm` (Jacob0226/sglang). 15.66→14.89 ms, GSM8K 0.929.
- opt#3 (fused allreduce) — **CORRECTION: not a valid skip.** The earlier "SGLang
  norm/comm ≤ ATOM" was read off the excel *section subtotal* (norm/comm ATOM 37 vs
  SGLang 22.1), which is **miscategorised** (one of SGLang's two
  `cross_device_reduce` got bucketed into MoE/MLP, and the residual-add+rmsnorm
  around the MLA output is counted under MLA_attention). At the **kernel level** ATOM
  is faster: per allreduce+norm event ATOM fuses `allreduce_fusion_kernel_1stage` =
  **11.19us**, vs SGLang unfused `26cross_device_reduce_1stage` (12.21) +
  `24add_rmsnorm_quant_kernel` (4.94) = **17.15us**; ×2/layer → ATOM **22.4us** vs
  SGLang **34.3us**, i.e. ATOM ~**12us/layer faster** on comm+norm. opt#3 is real.
  SGLang already ships the equivalent: `--enable-aiter-allreduce-fusion`
  (`communicator.py::apply_aiter_all_reduce_fusion`, gated only off CP/deterministic),
  which routes to aiter's fused allreduce(+residual+rmsnorm) — the same
  `allreduce_fusion_kernel_1stage` family ATOM uses. So opt#3 ≈ *enable that flag*
  (and confirm the GLM-5.2 model tags `hidden_states._sglang_needs_allreduce_fusion`),
  not a new kernel. NOT overlapping with PR #30195 (that fuses the *indexer q/k RoPE*
  `kn_entry_2c_...` → `apply_rope_inplace`, a different kernel).
- opt#2 diagnosis corrected: not a buffer-format port; both use compacted CSR +
  the same aiter kernel/metadata fn. Divergence isolated to `intra_batch_mode` /
  `topk` args in `_prepare_aiter_dsa_decode_metadata`. See table above.

## opt#2 RESOLVED — aiter fp8 decode now works, but is SLOWER than tilelang

The GPU-fault blocker is fixed. Enabling the aiter fp8 MLA decode core
(`--nsa-decode-backend aiter --kv-cache-dtype fp8_e4m3`) required 4 changes in
`dsa_backend.py` (saved as `opt2_aiter_fp8_decode.patch`):
1. `intra_batch_mode=True -> False` (buffer sizing, runtime `get_mla_metadata_v1`,
   and the kwarg passed into `mla_decode_fwd`) + drop `topk=` — fixes the OOB.
2. Add `q_scale=ones` when Q is fp8 (asm kernel asserts `fp8 Q requires q_scale`).
3. Force MLA output `o`/`o_kernel` to bf16 (`kn_mla_reduce_v1` rejects fp8 output).
4. `aiter_dsa_max_split_per_batch 64 -> 16` (match ATOM; no measurable effect).

**Validation (GLM-5.2-MXFP4, TP4, MI355X, isl1024/osl512, conc4):**

| decode backend | GSM8K (200q) | Median TPOT | Output tok/s |
|---|---|---|---|
| tilelang (baseline) | — | **14.52 ms** | 259.9 |
| aiter fp8 (opt#2)   | **0.945** | 24.45 ms | 158.7 |

**Conclusion (SUPERSEDED — see below):** correctness is perfect (GSM8K 0.945) but
the aiter fp8 decode core was ~69% **slower** than tilelang. The earlier verdict
("do not merge; tilelang wins; would need seg-MLA") was **wrong about the cause**.

--------------------------------------------------------------------------------
## opt#2 FIXED — the 69% was per-layer metadata rebuild, now hoisted → parity

**Root cause (confirmed by microbench + CUDA-graph capture, `tools/glm52_aiter_decode_microbench.py`):**
The aiter kernel itself was never the problem. It is the *correct* ATOM kernel
`mla_a8w8_qh16_qseqlen1_gqaratio16_ps` and runs at **16us captured** — matching
ATOM's ~13.7us (mla+reduce). The regression was that SGLang's `_forward_aiter`
rebuilt the aiter decode **work-schedule** (`get_mla_metadata_v1`, ~85us) +
compacted indices (`get_valid_kv_indices`, ~31us) **once per layer = 75x/step**,
all captured *inside* the decode CUDA graph. ATOM builds the schedule **once per
step** and shares it.

Deployment-accurate per-decode-step MLA cost, captured in a CUDA graph (bs=4,
seq≈1400, nhead=16, 75 layers):

| strategy | ms/step |
|---|---|
| CURRENT opt#2 (rebuild schedule ×75) | **9.03** |
| FIX: hoist `get_mla_metadata_v1` ×1/step | **2.14** (−6.9 ms) |
| upper bound: hoist schedule + compaction ×1/step | 0.96 |

The ~9 ms/step rebuild ≈ the +9.9 ms TPOT regression measured end-to-end.

**Fix (`opt2_aiter_fp8_decode_hoisted.patch`, on top of the 4 opt#2 correctness
edits):** hoist the schedule build out of the per-layer path into the metadata-prep
hooks — `init_forward_metadata` (eager), `_build_forward_metadata_cuda_graph`
(capture), and `_apply_cuda_graph_metadata` (replay) — via a new
`_build_aiter_dsa_decode_schedule(...)`. It writes once per step into the
persistent `self.aiter_dsa_work_*` buffers (built *outside* the captured graph,
same pattern SGLang already uses for the DeepGEMM paged-MQA schedule), keyed on
`dsa_cu_seqlens_k` (the topk-clipped KV counts, identical across all MLA layers in
a step). `_forward_aiter` now only *consumes* those buffers + runs the per-layer
index compaction. CUDA-graph-safe: control flow / kernel pointers are stable; only
buffer contents are refreshed per replay.

**Validation (GLM-5.2-MXFP4, TP4, MI355X, isl1024/osl512, conc4, same box/flags,
`--cuda-graph-max-bs 64`, GPUs 4-7):**

| decode backend | Median TPOT | GSM8K(200) |
|---|---|---|
| tilelang (tuned MI355X default) | 16.12 ms | 0.955 |
| aiter fp8 opt#2 (before hoist)  | 24.45 ms | 0.945 |
| **aiter fp8 opt#2 + hoist**     | **16.82 ms** | **0.960** |

The fix turns a **69% regression into ~4% (parity)** with the tuned tilelang
default, correctness preserved, and validates the aiter fp8 decode core runs at
ATOM speed. A follow-up experiment hoisting/parallelising `get_valid_kv_indices`
too (front-packed fast copy, `tools`) was **correct (GSM8K 0.940) but gave no
speedup** at bs=4 (16.83 ms) — the compaction overlaps other work and is not the
bottleneck. Remaining gap to ATOM (11.73 ms) is ATOM's *fused projections + whole
attention subsystem*, not the decode core — a separate, larger effort.

Artifacts: `opt2_aiter_fp8_decode_hoisted.patch`,
`tools/glm52_aiter_decode_microbench.py`, `tools/glm52_decode_bench.sh`,
bench logs under `tmp/opt2_bench/`.

--------------------------------------------------------------------------------
## Why the excel's "aiter core is faster" does NOT make the decode faster

The excel (conc4) shows the isolated aiter core `mla_a8w8...ps` (8.01) + `kn_mla_
reduce_v1_ps` (5.69) = 13.7us BEATING tilelang `main_kernel`×2 = 18.6us. But the
aiter decode *path* is still not faster end-to-end. Two reasons, confirmed by
adding the **i1k-o1k conc64** config:

**1. Extra work the excel's single-kernel view hides.** The aiter path must, per
decode step, also (a) build the persistent work-schedule `get_mla_metadata_v1`
and (b) compact the sparse selection into CSR `get_valid_kv_indices`. tilelang's
`main_kernel` consumes the `[bs, topk]` padded page-table *directly* and pays
neither. The schedule build **scales ~O(bs·CUs)**: microbench captured cost
85us @bs4 -> **545us @bs64**. Even hoisted to once/step it is pure overhead
tilelang doesn't have.

**2. conc64 measurement (same box/flags, `--cuda-graph-max-bs 64`):**

| config | metric | tilelang | aiter+hoist |
|---|---|---|---|
| conc4  | median TPOT   | 16.12 ms | 16.89 ms |
| conc64 | median TPOT   | 33.18 ms | 38.02 ms |
| conc64 | **median ITL**(steady state) | 28.30 ms | **29.11 ms** |
| conc64 | mean ITL      | 32.95 ms | 61.11 ms |
| conc64 | P99 ITL       | 128 ms   | 324 ms |
| conc64 | **max ITL**   | 2.7 s    | **31.1 s** |
| conc64 | output tok/s  | 1827     | 1010 |

**The conc64 gap was a cuda-graph-cap artifact I introduced, not a kernel deficit
and NOT a real tuning requirement.** SGLang's *default* decode `max_bs` on MI355X
is **512** (`server_args.py` gpu_mem>160GB branch), which already covers conc64
(peak running bs ~72). My bench script had capped `--cuda-graph-max-bs 64` only to
dodge a capture-time OOM at `--mem-fraction-static 0.85` (the default 512-bucket
capture ran the GPU out of memory: "Capturing batches bs=496 avail 12.47GB, tried
16GB"). That 64 cap (< peak 72) forced some decode steps to **eager**, and eager
hurts the aiter path far more than tilelang (its per-step schedule build ~545us@bs64
+ index compaction + more launches run un-captured) → the 31s max-ITL tail. The
correct fix is to keep the default max_bs (or any value ≥ peak) and instead lower
`--mem-fraction-static` so the capture fits; setting 96 just happened to clear the
peak. Re-running with `--cuda-graph-max-bs 96` (≥ peak) collapses the tail:

| conc64 (i1k-o1k) | tilelang g64 | aiter g64 | **aiter g96** |
|---|---|---|---|
| output tok/s | 1827 | 1010 | **1798** |
| mean TPOT | 32.89 | 61.00 | **33.40** |
| median ITL | 28.30 | 29.11 | 29.08 |
| P99 ITL | 128 | 324 | **127** |
| max ITL | 2.7 s | 31.1 s | **2.71 s** |

**Final verdict:** with the graph cap set above peak concurrency, aiter fp8 decode
**matches tilelang at conc64** (throughput 1798 vs 1827, identical max ITL) and is
at parity at conc4. So the excel's isolated core-kernel win (13.7 vs 18.6us) is
real but does NOT make the *path* faster: aiter pays a per-step schedule build +
per-layer CSR compaction that tilelang's kernel avoids (it consumes the padded
`[bs,topk]` table directly), which nets out to parity. To make aiter actually
*beat* tilelang would require removing that overhead — feed the padded table to
the kernel directly (the `intra_batch_mode` topk-strided path, currently faulting)
and/or fold the schedule build into the graph — plus ATOM's fused projections for
the rest of the layer. **Operational takeaway: the default decode max_bs (512 on
MI355X) already covers these workloads; do NOT cap it below peak running batch for
the aiter DSA decode (my 64 cap caused the tail). If capture OOMs, lower
`--mem-fraction-static`, don't shrink max_bs below peak.**

--------------------------------------------------------------------------------
## ATOM vs SGLang: who runs get_mla_metadata_v1 / the CSR compaction (verified)

Both schemes feed the aiter `mla_decode_fwd` a persistent work-schedule + a
compacted CSR kv-index buffer — the difference is *where/how often*.

| step | ATOM | SGLang (pre-fix) | SGLang (post-fix) |
|---|---|---|---|
| `get_mla_metadata_v1` (schedule) | **once/step** in `prepare_decode` → `_set_ubatch_mla_buffers` (`aiter_mla.py:1331`), keyed on `sparse_kv_indptr` | **per layer ×78** in `_forward_aiter` | **once/step** in metadata-prep (matches ATOM) |
| CSR `kv_indices` compaction | **not a separate per-layer pass** — the indexer writes the compacted `sparse_kv_indices_buffer` directly, and `sparse_kv_indptr` is built once/step (`aiter_mla.py:598-603`); every attention layer *reads* that shared buffer | **per layer** via `get_valid_kv_indices` (compacts the `[bs,topk]` padded `page_table_1`) | still **per layer** (unchanged) |

**So does ATOM "do" them?** Yes — ATOM calls the *same* `get_mla_metadata_v1`,
but once per step and shared across all 78 layers. It has *no* per-layer
`get_valid_kv_indices` because its **indexer emits the compacted CSR directly**
into a buffer shared by every layer.

**Why did SGLang "need" the per-layer versions?** Not a hard requirement — it is
SGLang's DSA representation. SGLang's sparse selection is a `-1`-padded
`[bs, topk]` page table (`transform_index_page_table_decode`), the format its
*other* decode backends (tilelang, flashmla) consume **directly**. The aiter
kernel instead wants compacted CSR, so `_forward_aiter` runs `get_valid_kv_indices`
as an adapter — per layer, because each layer's topk selection differs. The
schedule was *also* rebuilt per layer purely by omission (it was written inside
`_forward_aiter`); nothing required that, which is why hoisting it to once/step
(this fix) is safe and matches ATOM. Fully matching ATOM on the compaction too
would require SGLang's indexer to emit compacted CSR directly (bigger change);
measured, it is not the bottleneck at the tested batch sizes.

--------------------------------------------------------------------------------
## Provenance of the "16us@bs4 → 24us@bs64" kernel numbers (NOT a server profile)

These are from the **microbench** `tools/glm52_aiter_decode_microbench.py`, the
`-- CUDA graph capture test (per-call) --` → `capture kernel-only` line, which
times `aiter.mla.mla_decode_fwd()` **in isolation** (stage1 asm + `kn_mla_reduce`
bundled) via `torch.cuda.Event` over 50 CUDA-graph replays at GLM-5.2 decode
shapes (nhead=16, d=576, v=512, topk=2048, seq≈1400):
- `--bs 4`  → `capture kernel-only: replay 15.6–16.2 us`
- `--bs 64` → `capture kernel-only: replay 24.4 us`

This is an isolated-kernel microbench, **not** a torch-profiler trace of the live
server like the ATOM excel. A real per-kernel profile of the running SGLang
conc64 decode (via `--profile` + trace parse) has **not** been captured yet; that
is the correct next step to state per-kernel decode times with excel-level rigor.
