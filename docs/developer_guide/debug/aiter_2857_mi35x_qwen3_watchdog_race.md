# MI35x Qwen3-235B-MXFP4 CUDA-graph-capture crash since aiter v0.1.12.post1

**Upstream issue:** [ROCm/aiter#2857](https://github.com/ROCm/aiter/issues/2857)
**Regression commit in sglang:** `213027951` — `[AMD] Upgrade Aiter (#22264)` (2026-04-11), bumping `AITER_COMMIT_DEFAULT` from `v0.1.11.post1` → `v0.1.12.post1`.
**Affected CI job:** `nightly-8-gpu-mi35x-qwen3-235b-mxfp4` (MI35x, `--tp 4 --ep 2`, `amd/Qwen3-235B-A22B-Instruct-2507-mxfp4`).

## TL;DR

- The literal failure is **`hipEventQuery` raising `hipErrorStreamCaptureUnsupported` / `hipErrorCapturedEvent` inside PyTorch's NCCL watchdog thread** (`ProcessGroupNCCL.cpp:2055`).
- The underlying bug is a **HIP runtime bug**, tracked in [pytorch/pytorch#177309](https://github.com/pytorch/pytorch/issues/177309) and [ROCm/rocm-systems#3176](https://github.com/pytorch/pytorch/pull/176251): on ROCm, `hipEventQuery` does **not** honor `hipStreamCaptureModeThreadLocal`. A query from a non-capturing thread still consults the *global* capture list and can both (a) fail on the watchdog and (b) invalidate an active capture on another thread.
- aiter v0.1.12.post1 does not contain any new bug; it only **widens the race window** by doubling the per-capture bookkeeping (`graph_unreg_output_buffers_`, TCP-store + `pickle.dumps` IPC exchange, blocking `hipMemcpy` H2D). Qwen3-235B MXFP4 (`--tp 4 --ep 2`, MoE, many layers, many PGs) is the unique config that stretches the window wide enough to trigger the watchdog race consistently.
- **Fixes**, in order of preference:
  1. **Bump the PyTorch used in the ROCm image to include [pytorch/pytorch#176251](https://github.com/pytorch/pytorch/pull/176251)** (merged 2026-03-17). The workaround (`RocmWatchdogEventQueryContextGuard`) makes the watchdog skip `hipEventQuery` during active capture.
  2. Short-term: revert `AITER_COMMIT_DEFAULT` to `v0.1.11.post1` (i.e. revert sglang commit `213027951`).
  3. (Optional, speculative) patch aiter to deduplicate `graph_unreg_*_buffers_` by base ptr before calling `hipIpcGetMemHandle` to shrink the race window.

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

| Error                                 | Trigger                                                                                    |
| ------------------------------------- | ------------------------------------------------------------------------------------------ |
| `hipErrorStreamCaptureUnsupported`    | Watchdog queries an event while some thread has an active GLOBAL capture.                  |
| `hipErrorCapturedEvent`               | Watchdog queries an event that was originally recorded on a stream that was then captured. |

Both are emitted by the HIP runtime from inside `hipEventQuery`.

### 3. Why v0.1.11.post1 worked but v0.1.12.post1 does not

Diff between the two aiter Python files (`aiter/dist/device_communicators/custom_all_reduce.py`) and the C++ backend (`csrc/include/custom_all_reduce.cuh`, `csrc/kernels/custom_all_reduce.cu`):

| Change                                                                                                                                                                           | Location                                                   | Effect                                                                                                                                     |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| New `graph_unreg_output_buffers_` vector + `register_output_buffer` + `get_output_buffer_RD`                                                                                     | `custom_all_reduce.cuh:1874`, `get_output_buffer_RD` 2065- | Each `allreduce()` during capture now pushes **both input and output** base pointers (old code pushed only input). Doubles the bookkeeping. |
| New `is_broadcast_reg_outptr = (reg_inp_ptr == 0)` path                                                                                                                          | `custom_all_reduce.cu :: _all_reduce`                      | Activated in graph mode: every AR with registered input also treats output as an IPC buffer → goes through `get_output_buffer_RD`.        |
| `flush_graph_buffers` now loops `hipPointerGetAttribute` + `hipIpcGetMemHandle` over `input + output` buffers                                                                    | `custom_all_reduce.cuh :: get_graph_buffer_ipc_meta` 1932- | 2× host-side sync work compared to v0.1.11.                                                                                                |
| IPC metadata exchange switched from `dist.broadcast_object_list(..., device="cpu")` (gloo) to `TCPStore.set/get` + `pickle.dumps`                                                | `custom_all_reduce.py :: IPCBufferPool._gather_ipc_meta`   | Each rank now does sequential `set`+`world_size × get` on the TCP store. Slower than a gloo broadcast for small messages.                  |
| `register_graph_buffers` ends with a synchronous `hipMemcpy(d_rank_data_base_, ..., hipMemcpyHostToDevice)`                                                                      | `custom_all_reduce.cuh :: register_graph_buffers` 2157-    | Blocking H2D copy that serializes with any in-flight work and lengthens the flush. (Same API was also used in v0.1.11.)                    |
| `max_size` raised from ~128 MB to 1 GB, plus `"meta"` buffer allocation via `hipExtMallocWithFlags(hipDeviceMallocUncached)` sized `meta_size + 2*max_size` (~2 GB) at init time | `custom_all_reduce.py :: CustomAllreduce.__init__`         | Larger init allocation; not on the hot path during capture but contributes to memory pressure.                                             |

Net effect on timing: the capture-exit flush in v0.1.12.post1 is materially longer than in v0.1.11.post1, which gives the NCCL watchdog more polling opportunities while captured NCCL events still exist.

### 4. Why Qwen3-235B MXFP4 specifically, not GLM5 / DeepSeek-R1 MXFP4

Qwen3-235B is the only job in the suite using `--tp 4 --ep 2`. That configuration creates more process groups (TP=4, EP=2, plus the usual PP group) each carrying its own `CustomAllreduce` instance. The outer `parallel_state.graph_capture()` stacks them via `ExitStack` so they all exit in sequence, each running `flush_graph_buffers`. Combined with Qwen3's large MoE depth (many AR calls per layer × many captured batch sizes), the accumulated `graph_unreg_*_buffers_` length and TCP-store exchange time per rank is much larger than other models on the same hardware, pushing timing past the watchdog poll interval.

### 5. Cross-thread race sketch

```text
main thread                            NCCL watchdog thread
─────────────────────────────────      ──────────────────────────────────────
sglang graph_capture enters
  tp_group.graph_capture
    ca_comm_tp.capture (IS_CAPTURING)  [polls every few seconds]
  pp_group.graph_capture                 set hipStreamCaptureModeThreadLocal
    ca_comm_pp.capture (IS_CAPTURING)    ↓ (HIP ignores THREAD_LOCAL)
  torch.cuda.graph begin (GLOBAL)        hipEventQuery(nccl_work.event)
    forward_pass                           ↓
    PyNccl allreduce on NCCL stream      sees GLOBAL capture active on
    ca_comm.all_reduce (AR)              another thread
      push input+output ptr to           → returns hipErrorStreamCaptureUnsupported
      graph_unreg_*_buffers_              or hipErrorCapturedEvent
  torch.cuda.graph end                    → invalidates the main-thread capture
ca_comm_pp.capture.__exit__               → watchdog throws → process dies
  flush_graph_buffers:
    hipPointerGetAttribute × 2N
    hipIpcGetMemHandle × 2N
    TCPStore.set/get + pickle.dumps
    hipMemcpy HtoD (blocking)
ca_comm_tp.capture.__exit__ [same]
```

The left column grows significantly in v0.1.12.post1; the right column is unchanged. More iterations on the left ⇒ more chance for the right to land on the unsafe HIP path.

## Recommended fix path

### Preferred — bump PyTorch to include #176251

[`pytorch/pytorch#176251` "[ROCm] Avoid watchdog event queries during graph capture"](https://github.com/pytorch/pytorch/pull/176251) was merged on 2026-03-17 as commit `686aba0196bd2458beaf9abc097fbb4d1c90f4fe`. It adds:

- `RocmWatchdogEventQueryContextGuard` (thread-local guard set only on the watchdog thread)
- `queryEventWithRocmWatchdogCaptureWorkaround()` — skips `hipEventQuery` entirely when any capture is active; otherwise maps `hipErrorCapturedEvent` / `hipErrorStreamCaptureUnsupported` from the watchdog to "not ready" instead of fatal.

Any PyTorch build after 2026-03-17 that is ROCm-compatible with 7.2 will include this workaround. (The follow-up PRs [#178943 / #179780](https://github.com/pytorch/pytorch/pull/178943) are version guards to remove the workaround once HIP runtime ships the proper fix.)

The image we use in CI is pinned to a specific ROCm-patched torch wheel:

```
TORCH_ROCM_FILE="torch-2.9.1+rocm7.2.0.lw.git7e1940d4-cp310-cp310-linux_x86_64.whl"
```

This is a 2.9.1-based build that predates 2026-03-17 and does **not** contain the workaround. Fixing this properly requires rebasing that wheel on a newer PyTorch commit, or switching to a nightly ROCm 7.2 wheel.

### Short term — revert `AITER_COMMIT_DEFAULT` to `v0.1.11.post1`

Effectively a revert of sglang commit `213027951` in `docker/rocm.Dockerfile`. This eliminates the widened race window. Trade-off: we lose any v0.1.12 functional improvements until the torch bump lands.

### Optional — shrink the race window from aiter side

Lower-risk patches that can be suggested upstream (aiter):

- Deduplicate `graph_unreg_input_buffers_` / `graph_unreg_output_buffers_` by base allocation pointer before calling `hipIpcGetMemHandle` — same allocator slab often reuses the same base, so the unique handle count can collapse from thousands to tens.
- Replace the `TCPStore.set` + `world_size × get` loop with a single gloo `broadcast_object_list` (old behaviour) so the per-rank exchange is O(1) small messages instead of O(world_size) blocking RTTs.

Neither fixes the HIP runtime bug — they only reduce how wide the capture window is.

## References

- Upstream aiter issue: [ROCm/aiter#2857](https://github.com/ROCm/aiter/issues/2857)
- HIP runtime bug: [pytorch/pytorch#177309](https://github.com/pytorch/pytorch/issues/177309), [ROCm/rocm-systems#3176](https://github.com/pytorch/pytorch/pull/176251)
- PyTorch workaround: [pytorch/pytorch#175377](https://github.com/pytorch/pytorch/pull/175377) (reverted), [pytorch/pytorch#176251](https://github.com/pytorch/pytorch/pull/176251) (merged 2026-03-17)
- Follow-ups: [pytorch/pytorch#178943](https://github.com/pytorch/pytorch/pull/178943), [pytorch/pytorch#179780](https://github.com/pytorch/pytorch/pull/179780)
- sglang regression commit: `213027951a31342400aa6f489691ca0626bf76c7` — `[AMD] Upgrade Aiter (#22264)`
- aiter v0.1.12.post1 Python path: `aiter/dist/device_communicators/custom_all_reduce.py` (`IPCBufferPool`, `flush_graph_buffers`, `_gather_ipc_meta`)
- aiter v0.1.12.post1 C++ path: `csrc/include/custom_all_reduce.cuh` (`get_graph_buffer_ipc_meta`, `register_graph_buffers`), `csrc/kernels/custom_all_reduce.cu` (`allocate_meta_buffer`, `_all_reduce`)
