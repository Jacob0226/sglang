# tools/

Profiling, analysis, and micro-benchmark scripts for GLM-5 decode-layer optimization on B200 / MI355X.

## Trace analysis

| Script | Description | Example |
|--------|-------------|---------|
| `analyze_trace_overlap.py` | Sweep-line analysis of CUDA kernel overlap, bubble time, and stream concurrency from Torch Profiler traces. Supports HTML reports and two-trace comparison. | `python3 analyze_trace_overlap.py --trace1 dual.trace.json.gz --trace2 single.trace.json.gz --html cmp.html` |
| `extract_stream.py` | Extract and inspect kernels from specific CUDA streams within a trace. List streams, filter by time range, or compare two streams side-by-side. | `python3 extract_stream.py trace.json.gz --list-streams` |

## Micro-benchmarks

| Script | Description | Target |
|--------|-------------|--------|
| `glm5_decode_layer.py` | Single decode-layer micro-benchmark using real aiter/CK/TileLang/Triton kernels matching the actual SGLang decode trace. | MI355X |
| `glm5_proposalA_test.py` | Proposal A: overlap kv_a_norm / W_kc / RoPE / Cat with NSA indexer to fill the idle gap between fork and join. | MI355X |
| `glm5_proposalA_test_v2.py` | Proposal A v2: same idea, outputs traces to `trace_v2/` for comparison. | MI355X |

## Stream & overlap tests

| Script | Description | Target |
|--------|-------------|--------|
| `graph_stream_test.py` | Verify that CUDA/HIP Graph capture preserves N independent stream assignments. Auto-detects NVIDIA vs AMD. | B200 / MI355X |
| `test_dual_stream_sweep.py` | GLM5 MoE dual-stream unit test — replicates shared vs routed expert work on two streams. Compares single-stream vs dual-stream graph capture. | B200 / MI355X |
| `test_gemm_vs_elemwise_overlap.py` | Diagnostic: measures dual-stream overlap with correct fork-join pattern for GEMM+GEMM, GEMM+elementwise, and elem+elem combos. | B200 / MI355X |
| `test_graph_multi_stream_nv.py` | Test multi-stream GEMM overlap at various token counts (1–128). Each path simulates a full MLP: gate_up GEMM → mul → down GEMM. | B200 |

## CI reproduction

| Script | Description | Example |
|--------|-------------|---------|
| `repro_ci.sh` | Reproduce SGLang AMD CI jobs locally on this MI35x/MI325 box. Mirrors `.github/workflows/{pr-test-amd,nightly-test-amd*}.yml`: launches the same `rocm/sgl-dev:*` image, sets the same env vars (`SGLANG_IS_IN_CI*`, `SGLANG_USE_AITER`, `GPU_ARCHS`), and runs `test/run_suite.py` with the same suite/partition. By default uses the image's self-contained `/sgl-workspace/sglang` (no host mount); pass `--sglang-dir PATH` to test local sglang code changes. Generic version of `sglang/scripts/ci/amd/verify_aiter_2857_fix.sh` with no aiter#2857-specific bits. Run `bash repro_ci.sh --help` for full options. | `bash repro_ci.sh --docker rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260211 --suite stage-b-test-small-1-gpu-amd --partition-id 5 --partition-size 14` |

## Environment setup

| Script | Description | Example |
|--------|-------------|---------|
| `setup_rocm713_in_rocm720_image.sh` | Install TheRock ROCm 7.13 (pip) on top of the `rocm/sgl-dev:*-rocm720-mi35x-*` docker image and rebuild the dependent C++ extensions (aiter, sgl-kernel via `setup_rocm.py`, fast-hadamard-transform). Handles all the gotchas: `LD_LIBRARY_PATH` override for torch wheel RPATH bug, `CXX/CC` switch to bundled AMD Clang 23 (system g++ 11.4 can't compile `__bf16`), `ld.lld` wrapper symlink fix. Run inside the container as root. | `docker exec -it <container> bash ~/SGLang-benchmarks/tools/setup_rocm713_in_rocm720_image.sh` |
