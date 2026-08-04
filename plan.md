---
name: ROCm Tree Sampling
overview: Replace ROCm’s silent greedy speculative-verify fallback with a faithful Triton port of the CUDA target-only tree sampler. Land correctness first with centralized exact renormalization fallbacks, then optimize top-k/top-p renormalization only if profiling shows it is material.
todos:
  - id: tree-kernel
    content: Implement and register a faithful Triton target-only tree sampler
    status: completed
  - id: portable-renorm
    content: Centralize exact ROCm top-k/top-p renormalization fallbacks
    status: completed
  - id: runtime-dispatch
    content: Wire ROCm EAGLE and DFLASH to real stochastic tree verification
    status: completed
  - id: correctness-tests
    content: Add CUDA-oracle, ROCm-oracle, distribution, determinism, and dispatch tests
    status: completed
  - id: accuracy-performance
    content: Benchmark kernels and validate the customer GLM-5.2-FP8 accuracy workload
    status: cancelled
isProject: false
---

# ROCm tree speculative sampling

## 1. Port the target-only tree verifier to Triton
- Add [`python/sglang/kernels/ops/speculative/tree_sampling.py`](/home/macui/squidward-glm51/python/sglang/kernels/ops/speculative/tree_sampling.py) with a drop-in `tree_speculative_sampling_target_only_triton` wrapper and one Triton program per request.
- Reproduce the CUDA oracle in [`speculative_sampling.cuh`](/home/macui/squidward-glm51/python/sglang/kernels/aot/csrc/speculative/speculative_sampling.cuh): child/sibling traversal, cumulative sibling probability, `threshold_single`/`threshold_acc`, rejected-candidate scratch mutation, and bonus-token sampling from `relu(target_probs - draft_probs)`.
- Reuse the blocked two-pass vocabulary reduction/CDF pattern from [`reject_sampling.py`](/home/macui/squidward-glm51/python/sglang/kernels/ops/speculative/reject_sampling.py), but pass independent strides for `accept_index`, tree topology, target probabilities, and draft scratch.
- Preserve the frozen external kwargs `retrive_*` and `accept_token_num`; use correctly spelled `retrieve_*`, `num_correct_drafts`, and `bonus_token` internally.
- Register the Triton kernel in [`python/sglang/kernels/ops/speculative/__init__.py`](/home/macui/squidward-glm51/python/sglang/kernels/ops/speculative/__init__.py). Keep CUDA/MUSA on the existing AOT kernel by default and use it as the parity oracle.

## 2. Make probability renormalization portable
- Add centralized backend wrappers under [`python/sglang/kernels/ops/sampling/`](/home/macui/squidward-glm51/python/sglang/kernels/ops/sampling/) for top-k and top-p renormalization.
- Preserve CUDA/MUSA dispatch to `sgl_kernel`; on ROCm use exact Torch sort/mask/cumsum/scatter fallbacks equivalent to the regular sampler, including per-row thresholds, `k >= vocab`, `p >= 1`, ties, and zero-sum protection.
- Do not duplicate PR 32922’s private helpers inside `eagle_utils.py`; share one implementation between speculative verification and sampling tests.
- Treat performant Triton renorm as a follow-up optimization: benchmark the exact fallback first. If material, implement chunked multi-pass pivot/reduction kernels with distribution-level—not bitwise—parity at cutoff ties.

## 3. Replace the ROCm greedy workaround with real tree dispatch
- Update [`python/sglang/srt/speculative/eagle_utils.py`](/home/macui/squidward-glm51/python/sglang/srt/speculative/eagle_utils.py): remove `_is_hip` from the greedy gate, use Triton tree verification for ROCm target-only sampling, retain `chain_speculative_sampling_triton` only when rejection sampling is explicitly requested, and route top-k/top-p through the portable wrappers.
- Preserve greedy behavior for genuinely greedy requests. Ensure stochastic ROCm results enter the existing rank-0 TP broadcast so all ranks receive identical `predict`, `accept_index`, and `num_correct_drafts`.
- Update capability checks in [`spec_utils.py`](/home/macui/squidward-glm51/python/sglang/srt/speculative/spec_utils.py) and speculative startup validation so HIP advertises tree sampling only when the Triton implementation is importable; fail loudly instead of silently forcing argmax.
- Reuse the Triton tree implementation in [`dflash_utils.py`](/home/macui/squidward-glm51/python/sglang/srt/speculative/dflash_utils.py) after EAGLE parity is established, removing the separate CUDA/MUSA-only tree gate.

## 4. Prove semantics and reproduce the accuracy bug
- Add Triton kernel correctness tests under [`test/registered/jit/speculative/`](/home/macui/squidward-glm51/test/registered/jit/speculative/) and register both CUDA kernel CI and AMD CI.
- On NVIDIA, compare Triton against `sgl_kernel.tree_speculative_sampling_target_only` using fixed coins across: existing golden trees, branched `topk > 1` trees, random valid trees, threshold sweeps, full acceptance, sibling rejection, residual bonus sampling, draft scratch mutation, non-contiguous strides, and vocab sizes through production scale.
- On ROCm, compare against a small explicit Python/Torch oracle and run the same golden/random matrix. Add seeded repeatability and verify coins remain in `[0,1)`.
- Extend PR 32922’s distribution test from a chain to branched trees: non-greedy tree verification must preserve the target distribution while the old greedy fallback must demonstrably collapse to argmax.
- Add renorm support/sum/distribution tests for scalar and per-row top-k/top-p, boundary ties, `k=1`, `k=vocab`, `p=1`, and degenerate rows. Compare CUDA AOT and portable paths by total variation near cutoff ties.
- Add a dispatch test proving ROCm stochastic EAGLE selects the Triton tree kernel, rejection mode selects the chain kernel, and greedy traffic remains unchanged.

## 5. Validate performance and customer accuracy
- Add a focused benchmark under [`test/registered/jit/benchmark/`](/home/macui/squidward-glm51/test/registered/jit/benchmark/) covering batch size, draft-tree width/depth, vocab size, branching factor, and threshold modes; compare Triton with CUDA AOT and separately measure renorm fallback cost.
- Re-run the reported GLM-5.2-FP8 long-horizon workload on ROCm with no speculator, EAGLE target-only tree, and explicit chain rejection. Record pass rate, generated-length distribution, cap hits, accept length, and latency.
- Require tree mode with `speculative_eagle_topk > 1` to match the no-speculator accuracy distribution; PR 32922’s chain-only result is not sufficient.
- Land as two reviewable changes: first the Triton tree verifier + exact renorm fallback + dispatch/tests; second, only if justified by profiling, Triton top-k/top-p renorm optimization.