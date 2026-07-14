# opt#2 — Porting ATOM's fp8 MLA decode core into SGLang: conclusion

**Date:** 2026-07-06  **HW:** MI355X (gfx950), ROCm  **Model:** GLM-5.2-MXFP4, TP4
**Container:** `jacchang_GLM5` (SGLang)  **Branch:** `Jacob0226/sglang`
`jacob/glm-mla-fp8-aiter-decode-wip` (commit `1461f11d`)

--------------------------------------------------------------------------------
## TL;DR

- The long-standing GPU fault when enabling SGLang's **aiter fp8 MLA decode core**
  (`--nsa-decode-backend aiter` + `--kv-cache-dtype fp8_e4m3`) is **fixed**. It now
  runs correctly: **GSM8K 0.945**, 0 invalid.
- But it is **~69% slower** than the current tilelang decode on MI355X:
  **median TPOT 24.45 ms vs 14.52 ms**.
- Root cause (revised — see correction below): NOT seg-MLA. Both ATOM and this
  SGLang path run the `page_size=1` decode. The most plausible cost is that
  SGLang rebuilds the aiter decode metadata (`get_mla_metadata_v1`) and compacts
  indices (`get_valid_kv_indices`) **per layer, every step (78x/step)**, while
  ATOM builds it **once per step** and shares it across layers; plus tilelang is
  the tuned MI355X default. Needs a kernel trace to confirm.
- => opt#2 is a **correctness / enablement** result, **not** a perf win on MI355X.
  Kept on a separate WIP branch; the clean +5% opt#1 branch was not touched.

--------------------------------------------------------------------------------
## What was changed (4 edits in `dsa_backend.py`)

Saved as `opt2_aiter_fp8_decode.patch`.

1. **`intra_batch_mode=True → False`** in all three spots (buffer sizing
   `get_mla_metadata_info_v1`, runtime `get_mla_metadata_v1`, and the kwarg passed
   into `mla_decode_fwd`) **and drop `topk=`**. The `True + topk=2048` path put the
   asm kernel into a sparse "intra-batch" schedule that assumes a **topk-strided**
   work layout, while the indices/indptr handed in are **compacted CSR** → the
   kernel addressed past the buffer → the opaque GPU OOB. ATOM's decode uses
   neither flag.
2. **`q_scale = ones` when Q is fp8** — `asm_mla.cu:227 mla_decode_stage1_asm_fwd:
   fp8 Q requires q_scale and kv_scale`. SGLang only set `kv_scale`.
3. **MLA output `o` forced bf16** — `kn_mla_reduce_v1 doesn't support output type
   Float8_e4m3fn`. `o` had inherited fp8 from `q`.
4. **`aiter_dsa_max_split_per_batch 64 → 16`** (match ATOM). No measurable effect
   (see below), i.e. splitting is not the bottleneck.

--------------------------------------------------------------------------------
## Validation (GLM-5.2-MXFP4, TP4, MI355X, isl1024 / osl512, conc4)

| decode backend | GSM8K (200q) | Median TPOT | Output tok/s |
|---|---|---|---|
| tilelang (baseline)                | —     | **14.52 ms** | 259.9 |
| aiter fp8 opt#2 (64 spl)           | 0.945 | 24.55 ms     | 158.8 |
| aiter fp8 opt#2 (16 spl)           | 0.945 | 24.45 ms     | 158.7 |
| aiter fp8 opt#2 (`--page-size 64`) | —     | 24.53 ms     | —     |

Identical launch flags except `--nsa-decode-backend {tilelang|aiter}`.

### `--page-size 64` experiment (empirical, per user request)
Tested explicitly. Result: **no effect** on the aiter DSA decode.
- `--page-size 64` → TPOT 24.53 ms (identical to the implicit run).
- `--page-size 1` → server log still prints **"Setting page size to 64 for
  DeepSeek DSA"** and `server_args.page_size=64`. **SGLang forces page_size=64 for
  DeepSeek DSA** regardless of the flag, so the knob is not adjustable for GLM-5.2,
  and `_forward_aiter` runs its internal `page_size=1` kernel layout either way.
- (page_size 64 *does* help on some non-DSA models, where the flag is honored and
  changes KV paging / the decode kernel — but that path is pinned here.)

--------------------------------------------------------------------------------
## Why ATOM's MLA is faster, but the SGLang integration got 69% slower

### CORRECTION: it is NOT seg-MLA (an earlier draft wrongly said so)
Verified from the ATOM source + run logs:
- `atom/utils/envs.py`: `ATOM_MLA_PAGE_SIZE` **defaults to 1**, and `ATOM_GLM.sh`
  does **not** set it. So `use_seg_mla = (not use_triton_mla) and
  ATOM_MLA_PAGE_SIZE>1` is **False** — ATOM ran the **`page_size=1`** decode too.
- ATOM run logs show `kv_cache_block_size=16` (the KV *pool* paging), which is
  unrelated to the MLA decode kernel's `page_size`.
- SGLang's `_forward_aiter` **hardcodes** `page_size=1`
  (`kv_cache.view(-1,1,1,head_dim)`, metadata `page_size=1`), independent of the
  server `--page-size`. The opt#2 run already had server `page_size=64` (KV pool)
  and the decode still ran `page_size=1` internally.

=> Setting `--page-size 64` cannot invoke seg-MLA for the aiter DSA decode, and
ATOM does not use seg-MLA either. So seg-MLA is not the explanation.

### The real leading hypothesis: per-layer metadata rebuild overhead
The `page_size=1` `mla_decode_fwd` **is the same kernel family** ATOM calls, so
the core kernel is unlikely to be 69% slower on its own. What differs is
**how often SGLang rebuilds the decode metadata**:

- SGLang's `_forward_aiter` (called **once per layer, per decode step**) runs, on
  the fp8 branch, both:
  - `get_valid_kv_indices(...)` — compacts the `[bs, topk]` page table into CSR;
  - `_prepare_aiter_dsa_decode_metadata(...)` → `get_mla_metadata_v1(...)` — the
    persistent work-scheduling build.
  With 78 layers that is **~78 rebuilds per decode step** (a GPU launch, and
  possibly a sync, each time).
- ATOM builds its decode metadata **once per step** in `prepare_decode` and
  **shares** the work buffers across all layers (`_set_mla_persistent_worker_buffers`),
  so the per-step scheduling cost is paid once, not 78x.

At conc4/bs≈4 the actual attention math is tiny, so this fixed per-layer launch
overhead can easily dominate and explain the ~10 ms/token gap.

### Supporting evidence
`num_kv_splits 64 → 16` (ATOM's value) moved TPOT by <0.1 ms (24.55 → 24.45). If
the decode kernel's KV-split/reduce work were the cost, 16 would have helped a lot
at bs≈4; it didn't — pointing away from the kernel's inner loop and toward
fixed per-call overhead (metadata/index rebuild + launch), which split count does
not change.

### Also contributing
- **tilelang is the tuned MI355X default** (InferenceX). The aiter `page_size=1`
  path is a generic fallback, not shape/HW-tuned for gfx950 GLM decode.
- SGLang keeps its own (unfused) projections; ATOM's `q_proj_and_k_up` /
  `v_up_proj_and_o` are fused — but that affects the projection GEMMs, not the
  decode-vs-decode delta measured here.

### Net / how to actually make opt#2 faster
Not seg-MLA. The tractable levers, in order:
1. **Hoist the aiter decode metadata build out of the per-layer path** — compute
   `get_mla_metadata_v1` + `get_valid_kv_indices` once per step in
   `init_forward_metadata` and reuse across layers (mirror ATOM). This is the most
   likely big win and is a real code change, not a launch flag.
2. Confirm first with a **kernel trace** of the aiter-in-SGLang decode: if
   `get_mla_metadata` / `get_valid_kv_indices` appear ~78x/step, (1) is confirmed.
3. Only if the kernel itself is slow, consider tuning configs for aiter
   `mla_decode_fwd` on gfx950.

The 4 correctness fixes remain valuable: they make the aiter fp8 decode path
*functionally usable* in SGLang (it GPU-faulted before) and document the exact
ATOM↔SGLang contract mismatch.

--------------------------------------------------------------------------------
## Caveats / not yet done
- Only one bench point (conc4, isl1024/osl512). Higher concurrency / longer
  contexts could shift the tilelang-vs-aiter gap, but tilelang is expected to keep
  the lead on MI355X.
- Did **not** yet capture a kernel trace of the aiter-in-SGLang run; the
  per-layer-metadata-rebuild hypothesis is inferred from the code + the split-count
  experiment, not from a per-kernel profile. A trace is the next step to confirm.
- seg-MLA (`page_size>1`) is a dead end here: neither ATOM nor SGLang's aiter DSA
  decode uses it (both `page_size=1`); `--page-size 64` does not change the decode
  kernel path.

--------------------------------------------------------------------------------
## Artifacts
- Patch: `opt2_aiter_fp8_decode.patch`
- Full spec / diagnosis: `GLM52_fp8_decode_port_spec.md`
- WIP branch: `Jacob0226/sglang` `jacob/glm-mla-fp8-aiter-decode-wip` (`1461f11d`)
- opt#1 (landed, +5%): `Jacob0226/sglang` `jacob/glm-mla-fp8-absorbed-bmm`
