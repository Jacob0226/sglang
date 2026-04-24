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

Root cause recap: the HIP runtime is the buggy layer. Fix the runtime and every consumer works; avoid the runtime and you only move the symptoms around.

Per upstream statement on [pytorch/pytorch#179780](https://github.com/pytorch/pytorch/pull/179780):

> "The runtime fix was introduced in **rocm 7.2.1**, but to ensure backwards compatibility, we are adding a version guard so that the workaround only takes effect in ROCm versions without the runtime fix."

That is, ROCm 7.2.1+ already has the fix in the runtime — no PyTorch workaround needed at all. ROCm ≤ 7.2.0 needs PR #176251 in PyTorch.

Versions currently in play:

| Image                                                                 | ROCm     | PyTorch                            | Status                                                                                          |
| --------------------------------------------------------------------- | -------- | ---------------------------------- | ----------------------------------------------------------------------------------------------- |
| `rocm/sgl-dev:v0.5.10rc0-rocm720-mi35x-20260412` (dev container)      | **7.2.0** | `2.9.1+rocm7.2.0.lw.git7e1940d4` (`7e1940d4` is 2025-12-09) | **Hits the bug** — ROCm < 7.2.1 and torch predates PR #175377 / #176251 by ~3 months.            |
| `rocm/sgl-dev:v0.5.10rc0-rocm700-mi35x-20260421` (nightly CI image)   | ~7.0.x   | similar-aged torch                 | Hits the bug.                                                                                   |
| `rocm/pytorch:rocm7.2.2_ubuntu22.04_py3.10_pytorch_release_2.9.1`     | **7.2.2** | `2.9.1+rocm7.2.2.git36be91cf`      | **Drop-in safe** — same Ubuntu 22.04 / py3.10 / torch 2.9.1 ABI, but ROCm 7.2.2 runtime is fixed. |
| `rocm/pytorch:rocm7.2.1_ubuntu22.04_py3.10_pytorch_release_2.9.1`     | 7.2.1    | 2.9.1                              | Also safe; 7.2.1 is the first version with the runtime fix.                                     |
| `rocm/pytorch:rocm7.2.2_ubuntu24.04_py3.12_pytorch_release_2.10.0`    | 7.2.2    | 2.10.0 (`40d237bf`, 2026-04-03)    | ROCm is fixed, but Py 3.12 / Ubuntu 24.04 would force a full C++-extension rebuild.              |

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

### Fallback 1 — keep ROCm 7.2.0, upgrade PyTorch to a nightly that includes #176251

Works if for any reason the base image cannot be moved. Use a PyTorch nightly published after 2026-03-17 (and before PR #178943 / #179780 land in a way that guards the workaround off for ROCm 7.2.0):

```bash
# inside the container
pip uninstall -y torch torchvision torchaudio
pip install --pre torch==2.12.0.dev20260415+rocm7.2 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/rocm7.2
```

Then rebuild all C++ extensions that link against torch:

```bash
# aiter (editable, already in /sgl-workspace/aiter)
cd /sgl-workspace/aiter && pip uninstall -y amd-aiter
GPU_ARCHS=gfx950 PREBUILD_KERNELS=1 python3 setup.py develop

# sgl-kernel
cd /sgl-workspace/sglang/sgl-kernel
pip uninstall -y sglang-kernel
rm -f pyproject.toml && cp pyproject_rocm.toml pyproject.toml
AMDGPU_TARGET=gfx950 python3 setup_rocm.py install

# fast-hadamard-transform
cd /sgl-workspace/fast-hadamard-transform && python3 setup.py install

# triton (custom build): usually survives minor torch bumps, re-run if import fails
cd /sgl-workspace/triton-custom && pip install -e .

# sglang itself: pure Python, editable — no rebuild needed
```

Caveats:

- Torch nightly `2.12.x` / `2.13.x` has had API drift from 2.9; aiter / sgl-kernel source may hit deprecation or signature changes. If build fails, the diff is usually small and patchable, but it is not zero.
- torchvision / torchaudio must match the new torch; nightly index has them.
- `RocmWatchdogEventQueryContextGuard` is still in main as of writing (PR #178943 draft, PR #179780 open-approved but not merged); confirm with `strings -a $(python3 -c "import torch,os; print(os.path.dirname(torch.__file__))")/lib/libtorch_hip.so | grep RocmWatchdog` after install.

### Fallback 2 — cherry-pick PR #176251 into a local torch 2.9.1 rebuild

Only three files change ([`ProcessGroupNCCL.cpp`, `CUDAGraph.cpp`, `CUDAGraph.h`](https://github.com/pytorch/pytorch/pull/176251)). Build yields a drop-in torch 2.9.1 wheel with the workaround, zero ABI drift, zero C++-extension rebuild. Costs 1-2 hours of torch source build. Only worth it if Fallback 1's nightly breaks something and the base image cannot be swapped.

### Optional — shrink the race window from aiter side (lower priority)

These would not fix the underlying runtime bug but would reduce exposure:

- Deduplicate `graph_unreg_input_buffers_` / `graph_unreg_output_buffers_` by base allocation pointer before calling `hipIpcGetMemHandle`. Same allocator slab often reuses the same base, so unique handle count collapses from thousands to tens.
- Replace the `TCPStore.set` + `world_size × get` loop in `IPCBufferPool._gather_ipc_meta` with a single gloo `broadcast_object_list` (the v0.1.11.post1 behaviour). O(1) small messages vs O(world_size) blocking RTTs.

## Order-of-operations cheat sheet

When doing a torch-level upgrade (Fallback 1), the only thing that depends on torch's C++ ABI is the set of C++ extensions. Python-only packages (sglang itself, sglang-router, py tools) do not need any rebuild.

1. Backup: `pip freeze > /tmp/env.pre.txt`, snapshot `/opt/venv/lib/python3.10/site-packages/torch/lib/` if space allows.
2. Uninstall `torch`, `torchvision`, `torchaudio` together.
3. Install new `torch` + matching `torchvision` + `torchaudio` from the same index.
4. Rebuild C++ extensions — order among them does not matter:
   - `amd-aiter` (editable → `python setup.py develop`)
   - `sglang-kernel` (wheel → rebuild from `/sgl-workspace/sglang/sgl-kernel`)
   - `fast_hadamard_transform`
   - `triton-custom` (usually survives, but `pip install -e .` if import fails)
   - `mori`, `tilelang` if present (editable → rerun their build)
5. `sglang` is pure Python editable install — re-import to pick up the new torch, no rebuild.
6. Verify: `strings -a …/libtorch_hip.so | grep RocmWatchdog` should return hits (PR #176251 symbols present).
7. Smoke test: launch Qwen3-235B-MXFP4 with `--tp 4 --ep 2` and confirm graph capture completes past the previous crash point.

For the base-image upgrade (Preferred), none of steps 4-7 are required — the existing sgl-dev build already works against torch 2.9.1 and that exact ABI is preserved.

## References

- Upstream aiter issue: [ROCm/aiter#2857](https://github.com/ROCm/aiter/issues/2857)
- HIP runtime bug: [pytorch/pytorch#177309](https://github.com/pytorch/pytorch/issues/177309), [ROCm/rocm-systems#3176](https://github.com/pytorch/pytorch/pull/176251)
- PyTorch workaround: [pytorch/pytorch#175377](https://github.com/pytorch/pytorch/pull/175377) (reverted), [pytorch/pytorch#176251](https://github.com/pytorch/pytorch/pull/176251) (merged 2026-03-17)
- Follow-ups: [pytorch/pytorch#178943](https://github.com/pytorch/pytorch/pull/178943), [pytorch/pytorch#179780](https://github.com/pytorch/pytorch/pull/179780)
- sglang regression commit: `213027951a31342400aa6f489691ca0626bf76c7` — `[AMD] Upgrade Aiter (#22264)`
- aiter v0.1.12.post1 Python path: `aiter/dist/device_communicators/custom_all_reduce.py` (`IPCBufferPool`, `flush_graph_buffers`, `_gather_ipc_meta`)
- aiter v0.1.12.post1 C++ path: `csrc/include/custom_all_reduce.cuh` (`get_graph_buffer_ipc_meta`, `register_graph_buffers`), `csrc/kernels/custom_all_reduce.cu` (`allocate_meta_buffer`, `_all_reduce`)
