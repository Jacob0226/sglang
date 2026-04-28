# MI35x Qwen3-235B-MXFP4 CUDA-graph-capture crash since aiter v0.1.12.post1

**Upstream issue:** [ROCm/aiter#2857](https://github.com/ROCm/aiter/issues/2857)
**Regression commit in sglang:** `213027951` — `[AMD] Upgrade Aiter (#22264)` (2026-04-11), bumping `AITER_COMMIT_DEFAULT` from `v0.1.11.post1` → `v0.1.12.post1`.
**Affected CI job:** `nightly-8-gpu-mi35x-qwen3-235b-mxfp4` (MI35x, `--tp 4 --ep 2`, `amd/Qwen3-235B-A22B-Instruct-2507-mxfp4`).

## TL;DR

- The literal failure is **`hipEventQuery` raising `hipErrorStreamCaptureUnsupported` / `hipErrorCapturedEvent` inside PyTorch's NCCL watchdog thread** (`ProcessGroupNCCL.cpp:2055`).
- The underlying bug is a **HIP runtime bug**, tracked in [pytorch/pytorch#177309](https://github.com/pytorch/pytorch/issues/177309) and [ROCm/rocm-systems#3176](https://github.com/pytorch/pytorch/pull/176251): on ROCm, `hipEventQuery` does **not** honor `hipStreamCaptureModeThreadLocal`. A query from a non-capturing thread still consults the *global* capture list and can both (a) fail on the watchdog and (b) invalidate an active capture on another thread.
- aiter v0.1.12.post1 does not contain any new bug; it only **widens the race window** by doubling the per-capture bookkeeping (`graph_unreg_output_buffers_`, TCP-store + `pickle.dumps` IPC exchange, blocking `hipMemcpy` H2D). Qwen3-235B MXFP4 (`--tp 4 --ep 2`, MoE, ~96 layers, ~150 AR/forward) is the unique config that stretches the window wide enough to trigger the watchdog race consistently. (Note: `--tp 4 --ep 2` produces only **1** CustomAllreduce-bearing PG per rank — the TP group — but **3–4** NCCL-bearing PGs whose watchdog threads are all polling concurrently inside the capture window. See §4 for the precise count.)
- **Fixes**, in order of preference:
  1. **Bump the ROCm base image from 7.2.0 → 7.2.1+** (e.g. `rocm/pytorch:rocm7.2.2_ubuntu22.04_py3.10_pytorch_release_2.9.1`). The HIP runtime bug is fixed in **ROCm 7.2.1+** itself; same torch 2.9.1 ABI, **zero rebuild** of aiter / sgl-kernel / etc. **Empirically verified to fix the crash** — see "Empirical verification" section below.
  2. Short-term: revert `AITER_COMMIT_DEFAULT` to `v0.1.11.post1` (i.e. revert sglang commit `213027951`). Hides the race by going back to a narrower window. Caveat: also reverts whatever else v0.1.12.post1 brought.
  3. **PyTorch-side software workaround** (cherry-pick into a local torch 2.9.1 rebuild): the only path that keeps the system on ROCm 7.2.0. The relevant upstream patch ([pytorch/pytorch#176251](https://github.com/pytorch/pytorch/pull/176251)) was **merged twice and reverted twice** (latest revert 2026-03-31), so it's currently NOT in main. Off the upstream-supported path. See Fallback section.
  4. (Optional, speculative) patch aiter to deduplicate `graph_unreg_*_buffers_` by base ptr before calling `hipIpcGetMemHandle` to shrink the race window. Band-aid; does not actually fix the runtime bug.

## Evidence trail

### 1. What kernel/HIP API actually throws

The stack in the issue ends at
`custom_all_reduce.py:403 CustomAllreduce.capture().__exit__ [finally block] → flush_graph_buffers() → _gather_ipc_meta() [pickle.dumps(shard_data)]`.
But the exception that aborts the process is raised from a *different* thread:

```
[rank3]:[E421 22:56:09 ProcessGroupNCCL.cpp:2055] [PG ID 5 PG GUID 23 Rank 1]
Process group watchdog thread terminated with exception:
HIP error: operation not permitted when stream is capturing
Search for `hipErrorStreamCaptureUnsupported'

[rank3]:[E421 22:56:09 ProcessGroupNCCL.cpp:2055] [PG ID 6 PG GUID 27 Rank 1]
Process group watchdog thread terminated with exception:
HIP error: operation not permitted on an event last recorded in a capturing stream
Search for `hipErrorCapturedEvent'
```

`ProcessGroupNCCL.cpp:2055` is inside PyTorch's `ncclCommWatchdog` → per-work `CUDAEvent::query()` → **`hipEventQuery`**. That's the HIP call that fails. No aiter compute kernel (allreduce / rmsnorm / etc.) is the culprit.

### 2. Why `hipEventQuery` fails

Confirmed by PyTorch and AMD in:

- Issue: [pytorch/pytorch#177309 — Cross-thread stream-capture mode restrictions in hipEventQuery/hipEventSynchronize cause false watchdog failures](https://github.com/pytorch/pytorch/issues/177309)
- Author statement on [PR #176251](https://github.com/pytorch/pytorch/pull/176251) (`@chinmaydk99`, 2026-03-10):
  > "current HIP runtimes don't honor THREAD_LOCAL mode for `hipEventQuery`. Even with the mode switch, `hipEventQuery` still checks the global capture list and returns `hipErrorStreamCaptureUnsupported` if another thread has GLOBAL capture active, **which invalidates the session**."

PyTorch already sets `cudaStreamCaptureModeThreadLocal` around the watchdog query — on NVIDIA CUDA that is sufficient; on current ROCm/HIP it is not.

Two error variants in the log:


| Error                              | Trigger                                                                                    |
| ---------------------------------- | ------------------------------------------------------------------------------------------ |
| `hipErrorStreamCaptureUnsupported` | Watchdog queries an event while some thread has an active GLOBAL capture.                  |
| `hipErrorCapturedEvent`            | Watchdog queries an event that was originally recorded on a stream that was then captured. |


Both are emitted by the HIP runtime from inside `hipEventQuery`.

### 3. Why v0.1.11.post1 worked but v0.1.12.post1 does not

> **Common misreading:** "v0.1.12 added the fused allreduce+rmsnorm+quant kernel and that's why output buffers need IPC tracking now". This is not quite right — **both v0.1.11.post1 and v0.1.12.post1 ship the fused kernel** (PR #1990 lands in both chains, with different SHAs because of the history rewrite; the precompiled `hsa/gfx942/allreduce_rmsnorm_qnt_N8192.co` is in both releases).
> The actual change: v0.1.11.post1 required the **output buffer to be explicitly pre-registered** (one-time setup, no per-call IPC bookkeeping); v0.1.12.post1 introduces a **dynamic in-graph output-buffer registration** path (`is_broadcast_reg_outptr` → `get_output_buffer_RD`) so the user can pass arbitrary tensors. That's what doubles the per-AR host-side bookkeeping during capture.

Diff between the two aiter Python files (`aiter/dist/device_communicators/custom_all_reduce.py`) and the C++ backend (`csrc/include/custom_all_reduce.cuh`, `csrc/kernels/custom_all_reduce.cu`):


| Change                                                                                                                                                                             | Location                                                   | Effect                                                                                                                                      |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| New `graph_unreg_output_buffers_` vector + `register_output_buffer` + `get_output_buffer_RD`                                                                                       | `custom_all_reduce.cuh:1874`, `get_output_buffer_RD` 2065- | Each `allreduce()` during capture now pushes **both input and output** base pointers (old code pushed only input). Doubles the bookkeeping. |
| New `is_broadcast_reg_outptr = (reg_inp_ptr == 0)` path                                                                                                                            | `custom_all_reduce.cu :: _all_reduce`                      | Activated in graph mode: every AR with registered input also treats output as an IPC buffer → goes through `get_output_buffer_RD`.          |
| `flush_graph_buffers` now loops `hipPointerGetAttribute` + `hipIpcGetMemHandle` over `input + output` buffers                                                                      | `custom_all_reduce.cuh :: get_graph_buffer_ipc_meta` 1932- | 2× host-side sync work compared to v0.1.11.                                                                                                 |
| IPC metadata exchange switched from `dist.broadcast_object_list(..., device="cpu")` (gloo) to `TCPStore.set/get` + `pickle.dumps`                                                  | `custom_all_reduce.py :: IPCBufferPool._gather_ipc_meta`   | Each rank now does sequential `set`+`world_size × get` on the TCP store. Slower than a gloo broadcast for small messages.                   |
| `register_graph_buffers` ends with a synchronous `hipMemcpy(d_rank_data_base_, ..., hipMemcpyHostToDevice)`                                                                        | `custom_all_reduce.cuh :: register_graph_buffers` 2157-    | Blocking H2D copy that serializes with any in-flight work and lengthens the flush. (Same API was also used in v0.1.11.)                     |
| `max_size` raised from ~~128 MB to 1 GB, plus `"meta"` buffer allocation via `hipExtMallocWithFlags(hipDeviceMallocUncached)` sized `meta_size + 2*max_size` (~~2 GB) at init time | `custom_all_reduce.py :: CustomAllreduce.__init__`         | Larger init allocation; not on the hot path during capture but contributes to memory pressure.                                              |


Net effect on timing: the capture-exit flush in v0.1.12.post1 is materially longer than in v0.1.11.post1, which gives the NCCL watchdog more polling opportunities while captured NCCL events still exist.

### 4. Why Qwen3-235B MXFP4 specifically, not GLM5 / DeepSeek-R1 MXFP4

Qwen3-235B-MXFP4 is the **only** nightly job on MI35x that uses `--tp 4 --ep 2`. Every other large-model job uses simpler `--tp 8` (or `--dp 8`). That single configuration difference is the dominant variable.

#### Per-job parallelism on the MI35x nightly suite

| Test (suite name) | Model | `--tp` | `--ep` | `--dp` | `--attention-backend` | Result |
| --- | --- | ---: | ---: | ---: | --- | --- |
| **`nightly-8-gpu-mi35x-qwen3-235b-mxfp4`** | `amd/Qwen3-235B-A22B-Instruct-2507-mxfp4` | **4** | **2** | — | aiter | **CRASH** (since aiter v0.1.12.post1) |
| `nightly-8-gpu-mi35x-deepseek-r1-mxfp4` | `amd/DeepSeek-R1-MXFP4-Preview` | 8 | — | — | (default) | pass |
| `nightly-8-gpu-mi35x-kimi-k25` | `amd/Kimi-K2.5-MXFP4` | 8 | — | — | aiter | pass |
| `nightly-perf-8-gpu-mi35x-deepseek-v32-basic` | `deepseek-ai/DeepSeek-V3.2` | — | — | 8 | (default) | pass |
| (others, dense models) | various | 8 | — | — | (default) | pass |

(extracted from `test/registered/amd/test_*.py` `other_args` lists; verified empirically.)

#### What the `--tp 4 --ep 2` configuration costs per capture

Tracing `init_distributed_environment()` in `python/sglang/srt/distributed/parallel_state.py` line-by-line:

`GroupCoordinator.__init__` only allocates a `self.ca_comm` (a real `CustomAllreduce` object with its own IPC pool) when **both** are true:

```380:392:python/sglang/srt/distributed/parallel_state.py
self.ca_comm: Optional[Any] = None
self.qr_comm: Optional[QuickAllReduce] = None
if use_custom_allreduce and self.world_size > 1:
    ...
    self.ca_comm = CAClass(group=self.cpu_group, device=self.device)
```

For `--tp 4 --ep 2 --pp 1 --dp 1 --attn-cp 1 --moe-dp 1` (8 GPUs):

| Group | `use_custom_allreduce` arg | per-rank `world_size` in this group | `ca_comm` allocated? |
| --- | --- | ---: | --- |
| `_WORLD` | False (explicit) | 8 | No |
| `_TP` | True (default) | **4** | **Yes — the only one** |
| `_ATTN_CP` | True (default) | 1 (trivial groups) | No (size-1) |
| `_ATTN_TP` | (alias = `_TP`, same object) | 4 | (same object as `_TP`) |
| `_MOE_DP` | True (default) | 1 (trivial groups) | No (size-1) |
| `_MOE_EP` | **False (explicit)** | 2 | No |
| `_MOE_TP` | **False (explicit)** | 4 | No |
| `_PP` | **False (explicit)** | 1 | No |

→ **Per rank, exactly 1 group owns a `ca_comm`: `_TP`.**

Earlier versions of this doc said "4 CustomAllreduce-bearing PGs per rank" — that was wrong. The "4" is a different count: the number of `GroupCoordinator`s that `parallel_state.graph_capture()` pushes onto its `ExitStack`:

```1543:1551:python/sglang/srt/distributed/parallel_state.py
with get_tp_group().graph_capture(stream=stream) as context, \
     get_pp_group().graph_capture(context):
    with contextlib.ExitStack() as stack:
        seen = {id(_TP)}
        for group in (_MOE_EP, _MOE_TP):
            if group is not None and id(group) not in seen:
                seen.add(id(group))
                stack.enter_context(group.graph_capture(context))
        yield context
```

That's `_TP, _PP, _MOE_EP, _MOE_TP` = 4 stacked groups. But each `GroupCoordinator.graph_capture()` does:

```508:510:python/sglang/srt/distributed/parallel_state.py
ca_comm = self.ca_comm
maybe_ca_context = nullcontext() if ca_comm is None else ca_comm.capture()
```

Three of the four (`_PP`, `_MOE_EP`, `_MOE_TP`) have `ca_comm is None` → fall through to `nullcontext()` and do **zero** IPC work. **Only `_TP.graph_capture()` actually runs `flush_graph_buffers` at exit.**

#### Why the crash log nonetheless shows 4 distinct PG GUIDs

`ProcessGroupNCCL::watchdogHandler` is a per-PG background thread that polls every NCCL `CUDAEvent` in its `workMetaList_` via `hipEventQuery`. It does **not** care whether the PG has a `CustomAllreduce` attached — only whether NCCL collectives have been issued on it. With `--tp 4 --ep 2`, several PGs have active NCCL traffic during capture (TP allreduces, MOE_EP dispatch, MOE_TP combine, plus collectives on `_WORLD` for sync), so multiple watchdog threads are concurrently polling. The HIP runtime bug invalidates whichever poll happens to land in the active capture window, and several PG GUIDs surface in the crash log because they all hit failing `hipEventQuery` calls within the same race window.

So "4 PG GUIDs in the crash log" ≠ "4 CA-bearing PGs". It's "4 NCCL watchdog threads happened to poll inside the capture window simultaneously".

#### Compounding factors in the cumulative race window

The width of the race window is set by the *single* TP-`ca_comm` doing a lot of bookkeeping inside one capture:

```
race_window_width  ≈  (#allreduce per forward_pass)
                   ×  (#captured batch sizes)
                   ×  (per-AR host-busy from aiter v0.1.12)
```

| Factor | dense `--tp 8` model | Qwen3-235B `--tp 4 --ep 2` |
| --- | ---: | ---: |
| #PGs with `ca_comm` (per rank) | 1 | 1 |
| #PGs with NCCL traffic during capture (watchdog polling targets) | 1–2 | **3–4** (TP + MOE_EP + MOE_TP + WORLD) |
| #allreduce / forward_pass (model-shape dependent) | ~80–120 | **~150** (large MoE — every expert dispatch + combine) |
| #captured batch sizes (cuda_graph_bs) | ~52 | ~52 |
| per-AR host-busy in v0.1.12 (out-of-place IPC tracking, 2N entries) | small | small per-AR but **summed over ~150 AR × 52 batches** |
| Watchdog poll cadence | every few seconds | every few seconds |

Multiply across the row and the `--tp 4 --ep 2` capture phase is **host-busy for materially longer per batch capture, with more watchdog threads polling concurrently**. On the same hardware `--tp 4 --ep 2` deterministically overlaps at least one watchdog poll, while a single-PG `--tp 8` dense job most often does not.

#### Other models are not "immune", they're just statistically lucky

The bug is not Qwen3-specific — it's a probabilistic interaction between the host-busy span during capture and the watchdog poll interval. Any future model that adds a second model-parallel dimension (TP × EP, TP × PP, or DP × TP-attn / EP-moe split) on ROCm ≤ 7.2.0 can re-trigger this. **The deterministic fix is the runtime fix in ROCm 7.2.1+, not a per-model parallelism patch.**

#### Historical precedent and why its workaround doesn't transfer

This is not the first time this exact HIP runtime quirk has bitten sglang on AMD. The earlier hit was on a `dp_attention` workload:

- [sgl-project/sglang#10434](https://github.com/sgl-project/sglang/pull/10434) "Temporary work-around for rocm 7.0.0 alpha with enabling data-parallel issue" — **merged 2025-09-16** (the fix that actually lives in main). It introduced the `SGLANG_USE_ROCM700A` env-var gate and made `DpPaddingMode.get_default_mode_in_cuda_graph` return `SUM_LEN` on ROCm 7 instead of `MAX_LEN`.
- [sgl-project/sglang#11184](https://github.com/sgl-project/sglang/pull/11184) "[AMD] Serialize cross-ProcessGroup collectives for dp_attention" — **closed 2026-01-06 with the `DO NOT MERGE` label, never landed**. It tried to keep the original `MAX_LEN` (multi-PG `reduce_scatter` → `all_gather`) path and just synchronize across PGs with `work.wait()` + `TORCH_NCCL_BLOCKING_WAIT=1`. The PR thread shows reviewers preferred the simpler avoidance approach already in main.

The PR #11184 writeup explicitly cites the [ROCm 7 HIP stream-capture updates](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/hip-7-changes.html#stream-capture-updates) as the root cause — **same underlying bug class as aiter#2857**.

The actual fix that landed in main (#10434) takes an "avoidance" approach. Looking at the relevant code in `python/sglang/srt/layers/dp_attention.py`:

```python
# DP gather has two implementations:
def _dp_gather_via_all_gather(...):
    # MAX_LEN mode: TWO process groups in sequence
    get_attention_tp_group().reduce_scatter_tensor(...)   # PG #1
    get_tp_group().all_gather_into_tensor(...)            # PG #2  ← cross-PG sequence triggers race

def _dp_gather_via_all_reduce(...):
    # SUM_LEN mode: ONE process group only
    inplace_all_reduce(global_tokens, group_name=get_tp_group().unique_name)   # single PG, race-safe

# The work-around just forces SUM_LEN on ROCm 7+:
@classmethod
def get_default_mode_in_cuda_graph(cls) -> DpPaddingMode:
    if _USE_ROCM700A_WA:
        return cls.SUM_LEN              # avoid the cross-PG sequence
    else:
        return cls.MAX_LEN
```

The strategy is "**dodge the race-prone code path entirely**" — the multi-PG `reduce_scatter` + `all_gather` sequence becomes a single-PG `all_reduce`, so cross-PG capture-state interaction never happens.

**That avoidance strategy does not transfer to aiter#2857.** sglang's DP gather had the luxury of two algorithmic paths to choose between (`SUM_LEN` and `MAX_LEN`); switching to the single-PG path was a one-line change. aiter#2857 has no equivalent: there is exactly one CA-bearing PG (`_TP`) and it can't be removed, the per-AR `flush_graph_buffers` work in v0.1.12 simply takes longer, and the watchdog threads on every other NCCL-bearing PG (MOE_EP, MOE_TP, WORLD, …) are still polling concurrently — there is no "single-PG version of aiter MoE allreduce" to fall back to and no way to silence those other watchdog threads. The race is therefore inherent to any ROCm ≤ 7.2.0 deployment whose capture phase runs long enough for at least one of the concurrent watchdog polls to land inside it — only the runtime fix (ROCm 7.2.1+) closes it.

### 5. Cross-thread race sketch

```text
main thread                                            NCCL watchdog thread(s)
─────────────────────────────────────────────          ───────────────────────────────────────────
parallel_state.graph_capture() entered                 [one watchdog per PG, polling every few sec]
  TP.graph_capture()
    self.ca_comm.capture()  ← only TP enters this        TP-watchdog:    set hipStreamCaptureModeThreadLocal
  PP.graph_capture()                                                     ↓ (HIP ROCm ≤ 7.2.0 ignores it)
    nullcontext  (PP.ca_comm is None)                                    hipEventQuery(nccl_work.event)
  MOE_EP.graph_capture()                                MOE_EP-watchdog: same poll on its events
    nullcontext  (MOE_EP.ca_comm is None)               MOE_TP-watchdog: same poll on its events
  MOE_TP.graph_capture()                                WORLD-watchdog:  same poll on its events
    nullcontext  (MOE_TP.ca_comm is None)                                ↓
  torch.cuda.graph begin (GLOBAL capture active)        any one of these threads:
    forward_pass over 52 captured batch sizes:            sees GLOBAL capture on another thread
      PyNccl allreduce on NCCL stream                     → hipErrorStreamCaptureUnsupported
        (each AR enqueues an event the watchdog            or hipErrorCapturedEvent
         will eventually poll)                          → invalidates the main-thread capture
      ca_comm.all_reduce  (~150 calls × 52 batches)     → watchdog thread re-throws → process aborts
        push input + output ptr into
        graph_unreg_*_buffers_   (2N per AR in v0.1.12)
  torch.cuda.graph end
TP.capture().__exit__:
  flush_graph_buffers   ← single, expensive flush on the one TP ca_comm:
    hipPointerGetAttribute × 2N
    hipIpcGetMemHandle    × 2N
    TCPStore.set/get + pickle.dumps  (slow; touches at::Tensor → may sync GPU)
    hipMemcpy HtoD (blocking)
PP / MOE_EP / MOE_TP capture().__exit__:  no-op (nullcontext)
```

Two columns to read: the **left** is "host-busy minutes during one capture batch on TP", the **right** is "all watchdog threads polling concurrently — every PG has its own thread". v0.1.12 lengthens the left column (2N entries per AR plus the slower `pickle.dumps` exchange); the multi-PG layout (`--tp 4 --ep 2`) widens the right column (more watchdog threads polling = more chances for the unsafe HIP path to fire).

## Recommended fix path

Root cause recap: the HIP runtime is the buggy layer. Fix the runtime and every consumer works; avoid the runtime and you only move the symptoms around.

Per upstream statement on [pytorch/pytorch#179780](https://github.com/pytorch/pytorch/pull/179780):

> "The runtime fix was introduced in **rocm 7.2.1**, but to ensure backwards compatibility, we are adding a version guard so that the workaround only takes effect in ROCm versions without the runtime fix."

That is, ROCm 7.2.1+ already has the fix in the runtime — no PyTorch workaround needed at all. ROCm ≤ 7.2.0 needs a PyTorch-side workaround.

**Important caveat (verified 2026-04-27):** None of the PyTorch-side workaround PRs (#175377, #176251, #176942, #178943, #179780) have been merged upstream. The workaround code only exists in still-open or already-closed PR diffs. Any "torch nightly with the watchdog fix" claim should be confirmed by `strings -a libtorch_*.so | grep RocmWatchdog` — empirically the upstream PyTorch nightly index, the AMD `rocm.nightlies.amd.com/v2/gfx950-dcgpu/torch/` index, and the `rocm/pytorch:rocm7.2_..._2.9.1` base image **all lack this symbol**.

Versions currently in play:


| Image                                                               | ROCm      | PyTorch                                                     | Status                                                                                            |
| ------------------------------------------------------------------- | --------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `rocm/sgl-dev:v0.5.10rc0-rocm720-mi35x-20260412` (dev container)    | **7.2.0** | `2.9.1+rocm7.2.0.lw.git7e1940d4` (`7e1940d4` is 2025-12-09) | **Hits the bug** — ROCm < 7.2.1 and torch predates PR #175377 / #176251 by ~3 months.             |
| `rocm/sgl-dev:v0.5.10rc0-rocm700-mi35x-20260421` (nightly CI image) | ~7.0.x    | similar-aged torch                                          | Hits the bug.                                                                                     |
| `rocm/pytorch:rocm7.2.2_ubuntu22.04_py3.10_pytorch_release_2.9.1`   | **7.2.2** | `2.9.1+rocm7.2.2.git36be91cf`                               | **Drop-in safe** — same Ubuntu 22.04 / py3.10 / torch 2.9.1 ABI, but ROCm 7.2.2 runtime is fixed. |
| `rocm/pytorch:rocm7.2.1_ubuntu22.04_py3.10_pytorch_release_2.9.1`   | 7.2.1     | 2.9.1                                                       | Also safe; 7.2.1 is the first version with the runtime fix.                                       |
| `rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0`  | 7.2.2     | 2.10.0 (`40d237bf`, 2026-04-03)                             | ROCm is fixed, but Py 3.12 / Ubuntu 24.04 would force a full C++-extension rebuild.               |


### Preferred — bump the ROCm base image from 7.2 → 7.2.2 (or 7.2.1)

One line in `docker/rocm.Dockerfile`:

```diff
-ARG BASE_IMAGE_950_ROCM720="rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.1"
+ARG BASE_IMAGE_950_ROCM720="rocm/pytorch:rocm7.2.2_ubuntu22.04_py3.10_pytorch_release_2.9.1"
```

(same idea for `BASE_IMAGE_942_ROCM720`).

Why this is the cleanest path:

- Ubuntu 22.04, Python 3.10, **PyTorch 2.9.1** — all identical to the current stack. C++ ABI unchanged.
- **Zero rebuild** of `aiter`, `sgl-kernel`, `fast_hadamard_transform`, `triton`, `tilelang`, `mori`. HIP is backward compatible across 7.2.x patch releases, so kernels built for 7.2.0 run on 7.2.2.
- Fixes the **actual** HIP runtime bug; does not rely on a PyTorch-side workaround. Once everything runs on ≥ 7.2.1, PyTorch's `RocmWatchdogEventQueryContextGuard` becomes a no-op anyway ([PR #179780](https://github.com/pytorch/pytorch/pull/179780)).
- The CI `Hot patch: torch-ROCm` block in `rocm.Dockerfile` (the `TORCH_ROCM_FILE="torch-2.9.1+rocm7.2.0.lw.git7e1940d4..."` override) can be left as-is; at container boot it runs `python3 -c "import torch"` against whatever is installed, and `rocm/pytorch:rocm7.2.2_..._2.9.1` already ships 2.9.1. If a rebuild of that hot-patch block is still desired, it can be removed entirely for rocm720 since the upstream rocm/pytorch image already provides the right triton metadata.

The rocm700 path (`BASE_IMAGE_950 = rocm/sgl-dev:rocm7-vllm-20250904`) does not have a rocm/pytorch counterpart with the fix; a separate base-image refresh is needed, or the matching rocm720 job used instead.

### Fallback 1 — ~~upgrade PyTorch to a nightly that includes #176251~~ (does not exist)

> **DO NOT TRY THIS PATH AS WRITTEN PREVIOUSLY.** Earlier versions of this doc claimed
> PR #176251 was "merged 2026-03-17" — incomplete. The PR was actually **merged twice
> and reverted twice**, with the latest revert on 2026-03-31 leaving the workaround
> code OUT of main:
>
> | Date | Event | Commit |
> | --- | --- | --- |
> | 2026-03-06 | jeffdaily merges #176251 | `a346446` |
> | 2026-03-10 | jeffdaily reverts ("needs additional reviews, blocking revert of another PR") | `6c047fe` |
> | 2026-03-17 | jeffdaily re-merges with `merge -f` flag | `5ae3a6f` |
> | 2026-03-31 | chinmaydk99 reverts again | `23a4016` |
>
> **PyTorch upstream's stance** (synthesized from the PR thread):
> - PyTorch core reviewers (`@galv`, `@ngimel`) objected to querying CUDAGraph capture
>   state from `ProcessGroupNCCL` on layer-composability grounds (`@galv`: "the source
>   of truth for 'is a capture currently happening?' would have to be cuda or hip").
> - AMD (`@pragupta`, `@jeffdaily`) committed to fixing the bug in the next ROCm
>   release ("Next ROCm release is tentatively planned 3/26, and we're pushing to get
>   this fix as part of that delivery"). That release shipped as ROCm 7.2.1 with the
>   runtime fix.
> - Follow-up PRs reflect "remove workaround / add version guard so workaround only
>   activates on ROCm < 7.2.1": [#178943](https://github.com/pytorch/pytorch/pull/178943)
>   "Remove workaround", [#179780](https://github.com/pytorch/pytorch/pull/179780)
>   "Add version guard". Both **open, not merged** as of 2026-04-27.
>
> **Net effect:** PyTorch upstream's strategy is "ROCm 7.2.1+ is the proper baseline;
> we will not long-term carry a software workaround for ROCm ≤ 7.2.0 inside
> ProcessGroupNCCL." If you must stay on ROCm 7.2.0 you are explicitly off the
> upstream-supported path and must carry the workaround out-of-tree.
>
> Empirical check on representative wheels (2026-04-27):
>
> | Source | Wheel checked | `RocmWatchdogEventQueryContextGuard` symbol? |
> | --- | --- | --- |
> | Upstream `download.pytorch.org/whl/nightly/rocm7.2` | latest nightly | absent (latest revert was 2026-03-31; main doesn't have it) |
> | AMD `rocm.nightlies.amd.com/v2/gfx950-dcgpu/torch/` | `torch-2.9.1+rocm7.13.0a20260424-cp310` | **absent** |
> | AMD `rocm.nightlies.amd.com/v2/gfx950-dcgpu/torch/` | `torch-2.10.0+rocm7.13.0a20260424-cp310` | **absent** |
> | `rocm/pytorch:rocm7.2_..._2.9.1` (current sgl-dev base) | `torch-2.9.1+rocm7.2.0.git7e1940d4` | absent (predates the PR by 3 months anyway) |
>
> Conclusion: there is **no public torch wheel** anywhere that contains the watchdog
> workaround as of 2026-04-27. If you need a PyTorch-side fix on ROCm 7.2.0, you must
> apply the (currently-reverted-from-main) PR diff to a torch source tree yourself
> and rebuild — see Fallback 2.

### Fallback 2 — cherry-pick the watchdog patch into a local torch 2.9.1 rebuild

Pick one of the unmerged PRs and apply its diff to a torch v2.9.1 source checkout:

| PR | State | Files | LOC | Self-contained? | Notes |
| --- | --- | --- | --- | --- | --- |
| [#176251](https://github.com/pytorch/pytorch/pull/176251) | closed | 3 | +117/-6 | **No, partial.** Reads `_currently_capturing_graphs` map "under its mutex" — that map's lifecycle was NOT fully present in v2.9.1. Per #176942's own description, you also need *partial* of #168912 to make `is_graph_capture_active()` semantically correct. | Cleanest single-PR diff but standalone-on-2.9.1 is risky. |
| [#176942](https://github.com/pytorch/pytorch/pull/176942) | open | 3 | +72/-5 | **Yes — explicitly self-declared.** Body: "Content sources: #175377 (full), #176251 (full), #168912 (partial)". | The pragmatic choice if you are doing F2; carries the risk that the PR may still change before merging. |

Build path (same regardless of which PR you pick):

```bash
# inside an isolated build host (1-2 hr; can use gfx950 hardware to accelerate
# but the binary is generic for any gfx950 user)
git clone --branch v2.9.1 https://github.com/pytorch/pytorch.git
cd pytorch
git submodule update --init --recursive
# Apply the patch (use 176942 as a self-contained superset; download .patch from GitHub)
curl -L https://github.com/pytorch/pytorch/pull/176942.patch | git apply
PYTORCH_ROCM_ARCH="gfx950"  USE_ROCM=1  python3 setup.py bdist_wheel
ls dist/torch-2.9.1*.whl
```

Result: a `torch-2.9.1+...` wheel with `RocmWatchdogEventQueryContextGuard` baked in.
ABI is identical to the original torch 2.9.1 (the PR only edits `.cpp` / `.h` body, not
public headers / templates / ABI symbols), so **aiter / sgl-kernel / fast-hadamard /
triton-custom / tilelang do not need to be rebuilt** — the existing sgl-dev binaries
just keep working with the new wheel.

In the sglang Dockerfile, this just replaces the existing "Hot patch: torch-ROCm"
block's wheel source. No other changes needed.

Caveats:

- The patch is from an unmerged-and-still-evolving PR. Treat the chosen commit hash
  of `pull/176942.patch` as a **pinned input** so reproducible builds don't silently
  drift when the PR author rebases.
- Once ROCm is upgraded to 7.2.1+ (Preferred path), this workaround becomes a no-op
  but still safe to keep — PR #179780 (also unmerged) only adds a runtime version
  guard that auto-skips the workaround on ROCm ≥ 7.2.1.

### Fallback 1.5 — pin a specific upstream PyTorch nightly date (NOT VIABLE for this fix)

Conceptually attractive: "find a torch nightly built right after the watchdog PR
merged, pin that wheel, done." Marked here only to spell out why it doesn't apply:
the watchdog PR was never merged, so no upstream nightly exists with the fix. This
option is left as a placeholder in case a future upstream PR (a successor of #176942)
does merge.

### Optional — shrink the race window from aiter side (lower priority, EMPIRICALLY INSUFFICIENT)

All three of these are **patches against the aiter repo** ([ROCm/aiter](https://github.com/ROCm/aiter)),
specifically `aiter/dist/device_communicators/custom_all_reduce.py` and the C++ side
in `csrc/`. None of them is a PyTorch change. They would not fix the underlying HIP
runtime bug but would reduce exposure to it. They are ordered roughly by how much
they shrink the window for the same Qwen3-235B `--tp 4 --ep 2` workload.

> **Empirical result (2026-04-27):** the gloo-broadcast revert (the strongest
> of the three below) was applied inside `rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260426`
> (ROCm 7.2.0 + aiter v0.1.12.post1) and re-tested. The race **still fires** —
> just at a different point in the capture session:
>
> | Configuration | When does the watchdog race fire? | Verdict |
> | --- | --- | --- |
> | unpatched aiter v0.1.12.post1 | inside server init / first PG `ca_comm.capture` | CRASH |
> | aiter v0.1.12.post1 + gloo revert (Patch B below) | inside the very first `Capturing batches (bs=512)` iteration | CRASH |
>
> The gloo patch demonstrably ran (`broadcast_object_list` shows up in the
> crash stack), but the outer `torch.cuda.graph(...)` capture is open throughout
> the entire forward-pass kernel dispatch — not just during
> `flush_graph_buffers`. Shrinking the flush-side window only shifts when the
> watchdog poll lands inside that capture window; the race is still
> probabilistically reachable, and Qwen3-235B `--tp 4 --ep 2`'s long
> forward-pass + many PGs makes it deterministic. **Conclusion: no
> aiter-side patch can fix this; it is the runtime bug that has to go.**

- **In `IPCBufferPool._gather_ipc_meta`, revert to the v0.1.11.post1 gloo `broadcast_object_list` exchange** instead of v0.1.12.post1's `TCPStore.set` + `world_size × get` + `pickle.dumps`. The new TCP-store path adds a `pickle.dumps(tensor_payload)` on the host that touches `at::Tensor::data_ptr()`, which can trigger implicit GPU sync during graph-capture exit; the gloo path is pure CPU message-passing and never touches a CUDA stream. (Both `broadcast_object_list` and `TCPStore` are PyTorch APIs — the choice is in aiter source. The patch target is aiter, not PyTorch.) Largest impact of the three, since this is the slowest single step inside the capture-exit flush.
- **Deduplicate `graph_unreg_input_buffers_` / `graph_unreg_output_buffers_` by base allocation pointer** before calling `hipIpcGetMemHandle`. Same allocator slab often reuses the same base, so unique handle count collapses from thousands to tens.
- **`stream.synchronize()` before the IPC exchange** — this is the suggestion #2 in [aiter#2857](https://github.com/ROCm/aiter/issues/2857)'s "Suggested Fix Direction":
  ```python
  # in aiter/dist/device_communicators/custom_all_reduce.py :: IPCBufferPool.flush_graph_buffers
  torch.cuda.current_stream().synchronize()   # drain pending GPU work before host-side IPC exchange
  # ... existing _gather_ipc_meta + register_graph_buffers logic ...
  ```
  Caveat: this only drains the **calling thread's** stream. The HIP runtime bug is in
  the **other thread** (NCCL watchdog) calling `hipEventQuery` against the global
  capture list. Adding the sync makes the local capture state quiesce a bit faster but
  does not stop the watchdog poll from racing into it. Treat as a probabilistic
  band-aid, not a deterministic fix.

If you ship more than one of these together, you compound the window-shrinking effect
but you still are not fixing the runtime bug — the deterministic fix is either ROCm
7.2.1+ runtime (Preferred) or a torch-side watchdog skip (Fallback 2).

## Order-of-operations cheat sheet

For **Fallback 2** (cherry-pick + local torch 2.9.1 rebuild), since the rebuilt wheel
is still torch 2.9.1 ABI, **no C++ extension rebuilds are required**. Just swap the
torch wheel:

1. Build the patched torch wheel once on a build host (1-2 hr; see Fallback 2 for
   the exact `git apply` + `setup.py bdist_wheel` recipe). Pin the patch source by
   hash (e.g. download `https://github.com/pytorch/pytorch/pull/176942.patch` and
   stash it locally), so reproducible builds don't drift if the PR is rebased.
2. In the sglang Dockerfile, replace the `TORCH_ROCM_FILE` wheel that the existing
   "Hot patch: torch-ROCm" block reinstalls with the patched wheel. No other
   Dockerfile changes needed.
3. Verify inside a built image: `strings -a $(python3 -c "import torch,os; print(os.path.dirname(torch.__file__))")/lib/libtorch_cpu.so | grep RocmWatchdog` should return hits.
4. Smoke test: launch Qwen3-235B-MXFP4 with `--tp 4 --ep 2` and confirm graph
   capture completes past the previous crash point. (Or just run
   `scripts/ci/amd/verify_aiter_2857_fix.sh ci` against the new image.)

For the **Preferred** base-image upgrade, the cheat sheet collapses to: change the
single `BASE_IMAGE_950_ROCM720` ARG line in `docker/rocm.Dockerfile` and rebuild.
The existing sgl-dev build already works against torch 2.9.1 and that exact ABI is
preserved.

A full torch-replacement-with-ABI-break flow (e.g. switching to torch 2.10 or 2.12
nightly) would additionally need `aiter` / `sgl-kernel` / `fast_hadamard_transform` /
`triton-custom` / `mori` / `tilelang` rebuilt against the new torch ABI — but this is
not a path that ships the watchdog fix today (per "Fallback 1" section: no such
nightly exists), and is mentioned only for completeness.

## Empirical verification (this branch)

Recorded for posterity so future readers don't have to redo the experiment.

### Test image

Built a one-off image to validate the **Preferred** path:

```
jacchang/shared:sglang-aiter2857-rocm722-mi35x-20260424
   = rocm.Dockerfile  with  BASE_IMAGE_950_ROCM720 bumped to
     rocm/pytorch:rocm7.2.2_ubuntu22.04_py3.10_pytorch_release_2.9.1
   + everything else unchanged (aiter v0.1.12.post1, sgl-kernel, ...)
```

`scripts/ci/amd/verify_aiter_2857_fix.sh ci` runs the actual GitHub-CI suite
(`nightly-8-gpu-mi35x-qwen3-235b-mxfp4`) locally on an MI35x box against this image.

### Results

| Configuration | GSM8K accuracy (1319 q) | bs=1 speed | Graph-capture race? | Verdict |
| --- | ---: | ---: | --- | --- |
| ROCm 7.2.0 + aiter v0.1.12.post1 (current sgl-dev base) | (test never reaches eval) | n/a | **CRASH** at capture exit (`hipErrorCapturedEvent`) | original CI failure mode |
| ROCm 7.2.0 + aiter v0.1.12.post1 **+ Patch B** (`_gather_ipc_meta` reverted to gloo `broadcast_object_list`) | (test never reaches eval) | n/a | **STILL CRASH** — race shifted from server init to inside `Capturing batches (bs=512)` iter 0/52 | aiter-side window-shrink is **insufficient**; runtime bug still surfaces during the forward-pass capture body |
| **ROCm 7.2.2** + aiter v0.1.12.post1 | **0.909** | (test fails at acc threshold before reaching speed) | **PASS** — capture 52/52 batches clean, server up, 1319 q served with `cuda graph: True` | watchdog race **fixed**, but acc < 0.93 threshold (regression masked by the crash) |
| **ROCm 7.2.2** + aiter v0.1.11.post1 | **0.9401** | 95.89 tok/s | PASS | full pass, used as the pre-regression baseline |

Two independent issues, originally both surfacing as the same CI failure:

1. **Graph-capture race** — fixed by ROCm 7.2.2 base bump (verified above).
2. **GSM8K accuracy regression** — aiter v0.1.12.post1 drops accuracy by ~3pp vs v0.1.11.post1 on this exact stack. **Not** the watchdog bug; was previously masked because the crash at (1) happened first. Tracked separately; see "Caveat: aiter v0.1.11 → v0.1.12 history rewrite" below for why this is hard to bisect.

### Caveat — `git bisect` on aiter v0.1.11.post1 → v0.1.12.post1 does not work

If you try to bisect the accuracy regression:

```bash
cd /sgl-workspace/aiter
git bisect start
git bisect bad  v0.1.12.post1   # 7b570738c
git bisect good v0.1.11.post1   # 417de6df4
```

it will eagerly try commits dated 2024-11-11 (the very start of the repo). That is
because the two tags **share zero git ancestry**:

```
$ git merge-base v0.1.11.post1 v0.1.12.post1
(empty -- no common ancestor)

$ git rev-list v0.1.11.post1 | wc -l
1405
$ git rev-list v0.1.12.post1 | wc -l
1665
$ git rev-list v0.1.11.post1 v0.1.12.post1 | sort | uniq -d | wc -l
0   # zero shared commit SHAs
```

Both chains contain commits as far back as 2024-11-11 ("Initial commit"), but the
SHAs differ — i.e. the upstream aiter repo received a `git filter-repo` /
force-push history rewrite somewhere between 2026-03-05 (v0.1.11.post1 tag) and
2026-04-10 (v0.1.12.post1 tag). v0.1.10.post3 (Feb 2026) sits cleanly on the OLD
chain (`merge-base v0.1.10.post3 v0.1.11.post1` = v0.1.10.post3) but is **disjoint**
from the new chain (`merge-base v0.1.10.post3 v0.1.12.post1` = empty).

Net effect: bisecting requires either talking to aiter maintainers about the
pre-rewrite commit graph, or doing file-level delta-debugging on the 1029-file diff
(82740 insertions / 29491 deletions) between the two trees rather than commit-level
bisect. The latter was done — see "Module-level delta-debugging" below.

### Module-level delta-debugging on `fused_moe.py` (2026-04-28)

Since commit-level `git bisect` is impossible (above), did file-level swap
A/B testing instead. All experiments start from
`jacchang/shared:sglang-aiter2857-rocm722-mi35x-20260424` (v0.1.12.post1
baseline, GSM8K ≈ 0.918 ± sampling noise), then overlay v0.1.11.post1's
version of one or more files at `/sgl-workspace/aiter/aiter/...` and re-run
the GSM8K eval. Helper script `/tmp/repro_2857/exp.sh` automates fresh
container spawn + Triton cache flush + test launch.

| Exp | What was changed (v0.1.11 → v0.1.12 image) | GSM8K | bs=1 speed | Verdict |
| --- | --- | ---: | ---: | --- |
| baseline | (none, pure v0.1.12.post1) | 0.918 | 100.6 | FAIL — re-confirms doc Results above |
| **A** | `aiter/utility/fp4_utils.py` only (Triton FP4 quant kernel; the diff includes a rewrite of `_round_fp32_to_fp4` rounding semantics from round-half-up to round-to-nearest-even) | 0.912 | 93.4 | NO IMPROVEMENT — `fp4_utils.py` is **not** the regression source. The rounding-rewrite hypothesis was wrong. |
| **B** | `aiter/fused_moe.py` only (the dispatcher; 280-line diff) | **0.952** | 99.2 | **PASS, recovered** — regression rides with `fused_moe.py` changes |
| **C** | `aiter/fused_moe.py` line 20 import only: redirect `mxfp4_moe_sort_fwd` and `fused_dynamic_mxfp4_quant_moe_sort` to their Triton-path equivalents (with M-dim padding to multiple of 32 to match HIP's `out_scale` shape contract) | — | — | **CRASH** — `ck_moe_stage1` HIP kernel SIGABRTs on the first forward (Memory access fault by GPU on `0x14208000` etc.) |
| **D** | `aiter/fused_moe.py` `get_padded_M` body only (revert v0.1.11's down-cap policy: M<2048→1024, M<16384→2048, else→16384 vs v0.1.12's nextPow2/32768-cap) | 0.908–0.913 | 100.2 | NO IMPROVEMENT — `get_padded_M` policy is **not** the regression source |

#### What this localizes

**Confirmed by Exp B + Exp D**:
- The regression rides with changes inside `fused_moe.py` (Exp B recovered).
- It is **not** the Triton MXFP4 quant kernel in `fp4_utils.py` (Exp A).
- It is **not** the `get_padded_M` policy change (Exp D).

**Strongly suggested by Exp C's crash, but not proven by single-variable
isolation**:
v0.1.12 added `mxfp4_moe_sort_hip` and `fused_dynamic_mxfp4_quant_moe_sort_hip`
public wrappers in `aiter/ops/quant.py` (decorated `@compile_ops("module_quant")`,
bound to HIP/CK kernels in `module_quant.so`), and rewired `fused_moe.py`
line 20 + 4 call sites to call them. Replacing only those Python wrappers
with Triton-path equivalents (while keeping v0.1.12's `ck_moe_stage1` /
`ck_moe_stage2` HIP MoE GEMM kernels downstream) makes the HIP GEMM SIGABRT
on the first forward — even after padding the Triton sort output's M dim
to match HIP's `(M_o + 31) // 32 * 32` allocator size.

This means v0.1.12's MoE 2-stage HIP pipeline appears to be **co-designed**:
the sort kernel output and the GEMM kernel input share assumptions about
layout / strides / dtype / scale-tile alignment beyond what the public
Python signatures expose. Exp B succeeds because v0.1.11's `fused_moe.py`
calls v0.1.11's Triton sort + Triton/CK GEMM pair — the Python side
sidesteps the entire v0.1.12 HIP MoE 2-stage path.

So the real answer to "which change in `fused_moe.py` causes the
regression?" is: it's **not** any single Python edit visible in the 280-line
diff in isolation. The full Python rewire (line 20 import + 4 call sites)
is needed to dispatch to the new HIP `mxfp4_moe_sort_hip` /
`fused_dynamic_mxfp4_quant_moe_sort_hip` kernels, and the *numerical
regression* is in those C++/HIP kernels (compiled into `module_quant.so` /
`module_moe_*.so` from `csrc/`).

Bisecting any further requires diffing v0.1.11 vs v0.1.12 `csrc/` and
rebuilding `module_quant.so` (and the matching `module_moe_*.so`) with one
csrc-level change reverted at a time. Each iteration is a ~30–60 min CK +
GEMM template build, and is best done by an aiter maintainer with full
csrc context.

### Trying aiter v0.1.13.dev0 — does **not** drop in cleanly (2026-04-28)

[ROCm/aiter](https://github.com/ROCm/aiter) tagged `v0.1.13.dev0` (==
`v0.1.13-rc1`, commit `930c94120 revert gptoss tuned config #2904`) on top
of the v0.1.12 line. Tested whether it might already have fixed the GSM8K
regression upstream:

```bash
# inside the v0.1.12.post1 container (jacchang/shared:...20260424)
cd /sgl-workspace/aiter
git fetch --tags origin
git checkout v0.1.13.dev0                  # -> 930c94120
git submodule update --init --recursive    # CK bumped to fdf4bb7fcc98
pip uninstall -y amd-aiter
GPU_ARCHS=gfx950 PREBUILD_KERNELS=1 python3 setup.py develop
```

Build succeeded (`pip show amd-aiter` → `Version: 0.1.13rc1`). Server
launched, model loaded, CUDA graph capture completed cleanly
(`The server is fired up and ready to roll!`). But the **first GSM8K
inference request** trips a HIP memory fault on multiple ranks:

```
Memory access fault by GPU node-3 ... on address 0x90208000. Reason: Unknown.
Memory access fault by GPU node-2 ... on address 0x38208000. Reason: Unknown.
Memory access fault by GPU node-4 ... on address 0x14208000. Reason: Unknown.
GPU coredump: handler exited with error (status: 1)
Fatal Python error: Aborted
```

The three faulting addresses share the same low-order pattern
(`0x...208000`), suggesting a stride / pointer-arithmetic bug rather than
random memory corruption.

Three plausible causes, none individually verified:

1. Real regression in v0.1.13.dev0 csrc/ kernels (the cleanest interpretation).
2. `setup.py develop` doesn't fully overwrite the v0.1.12 pre-built
   `/sgl-workspace/aiter/aiter/jit/*.so` files in the image; mixing
   v0.1.12 cached ABI with v0.1.13 newly-compiled symbols.
3. Submodule `composable_kernel` was bumped to `fdf4bb7fcc98` but
   `sgl-kernel` and other torch C++ extensions in the image still expect
   the older CK ABI.

Bottom line: **v0.1.13.dev0 is not a drop-in fix** for the v0.1.12.post1
GSM8K regression on this CI configuration. Worth retesting once
v0.1.13.post0 / v0.1.13 final ships, or when AMD provides a pre-built
`rocm/sgl-dev:*` image rebased on v0.1.13.

`v0.1.12.post2` (cherry-picks
[#2645 multi-arch CK GEMM dispatch](https://github.com/ROCm/aiter/pull/2645)
+ a SynchronizedCache backport on top of v0.1.12.post1; on the
`release/v0.1.12` branch and so shares git history with v0.1.12.post1)
is a much safer drop-in candidate to test next — its scope is narrower and
the ABI risk is much lower than v0.1.13.dev0. Untested as of 2026-04-28.

### Snapshot images for further A/B work

For other contributors who want to reproduce or A/B without rebuilding aiter:

| Image | Contents | Empirically observed |
| --- | --- | --- |
| `jacchang/shared:sglang-aiter2857-rocm722-mi35x-aiter011post1-20260427` | Preferred image with aiter rebuilt to v0.1.11.post1 inside | GSM8K 0.9401, speed 95.9 tok/s — **PASS** |
| `jacchang/shared:sglang-aiter2857-rocm722-mi35x-aiter012post1-20260427` | Preferred image with aiter v0.1.12.post1 (the original built-in pin) | GSM8K 0.909 — fails accuracy assertion only; capture is clean |

## References

- Upstream aiter issue: [ROCm/aiter#2857](https://github.com/ROCm/aiter/issues/2857)
- HIP runtime bug: [pytorch/pytorch#177309](https://github.com/pytorch/pytorch/issues/177309) (issue), [ROCm/rocm-systems#3176](https://github.com/pytorch/pytorch/pull/176251) (referenced from the issue body)
- PyTorch-side workaround attempts (**not in main** as of 2026-04-27):
  - [pytorch/pytorch#175377](https://github.com/pytorch/pytorch/pull/175377) — closed 2026-03-11, "Handle capture-time HIP event query errors in NCCL watchdog". Earlier attempt, superseded by #176251.
  - [pytorch/pytorch#176251](https://github.com/pytorch/pytorch/pull/176251) — "Avoid watchdog event queries during graph capture". **Merged twice and reverted twice**: merged 2026-03-06 (`a346446`), reverted 2026-03-10 (`6c047fe`), re-merged 2026-03-17 (`5ae3a6f`), reverted again 2026-03-31 (`23a4016`). The PyTorch core team's objection is layer composability — querying CUDAGraph capture state from `ProcessGroupNCCL`. Code is currently NOT in main.
  - [pytorch/pytorch#176942](https://github.com/pytorch/pytorch/pull/176942) — open, supersedes #175377 + #176251 + partial of #168912 — but **not merged**.
  - [pytorch/pytorch#178943](https://github.com/pytorch/pytorch/pull/178943) — open, "Remove workaround" (depends on the watchdog fix landing first).
  - [pytorch/pytorch#179780](https://github.com/pytorch/pytorch/pull/179780) — open, "Add ROCm version guard" so workaround only activates on ROCm < 7.2.1.

  Synthesis: PyTorch upstream's strategy is to make ROCm 7.2.1+ the supported baseline (where the runtime is fixed natively) and **not** carry a software workaround for ROCm ≤ 7.2.0 inside ProcessGroupNCCL long-term.
- sglang regression commit: `213027951a31342400aa6f489691ca0626bf76c7` — `[AMD] Upgrade Aiter (#22264)`
- Same bug-class precedent in sglang (different surface):
  - [sgl-project/sglang#10434](https://github.com/sgl-project/sglang/pull/10434) "Temporary work-around for rocm 7.0.0 alpha with enabling data-parallel issue" — **the fix that landed in main** (merged 2025-09-16). Adds `SGLANG_USE_ROCM700A` gate; switches `DpPaddingMode` from `MAX_LEN` to `SUM_LEN` to avoid the multi-PG `reduce_scatter`+`all_gather` sequence.
  - [sgl-project/sglang#11184](https://github.com/sgl-project/sglang/pull/11184) "Serialize cross-ProcessGroup collectives for dp_attention" — **closed 2026-01-06 with `DO NOT MERGE` label, never landed**. Tried to keep the multi-PG path and serialize with `work.wait()` instead. Useful as documentation of the alternative approach considered.
- ROCm 7 stream-capture behavior change (cited by sglang#11184 as root cause): [hip-7-changes / stream-capture-updates](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/hip-7-changes.html#stream-capture-updates)
- Other related aiter bugs in the same period: [ROCm/aiter#2061](https://github.com/ROCm/aiter/issues/2061) "Custom all-reduce IPC buffers use fixed VA…" (open) — different mechanism, but signals aiter custom_all_reduce had multiple bugs in this window
- aiter v0.1.12.post1 Python path: `aiter/dist/device_communicators/custom_all_reduce.py` (`IPCBufferPool`, `flush_graph_buffers`, `_gather_ipc_meta`)
- aiter v0.1.12.post1 C++ path: `csrc/include/custom_all_reduce.cuh` (`get_graph_buffer_ipc_meta`, `register_graph_buffers`), `csrc/kernels/custom_all_reduce.cu` (`allocate_meta_buffer`, `_all_reduce`)

