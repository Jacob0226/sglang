# ROCm tree speculative sampling validation

Branch `RM/rocm-tree-spec-sampling-triton`, Jacob stack tip `1ea4bd1ce5`.

## Triton tree verifier

| Platform | Passed | Subtests | Skipped | CUDA AOT oracle |
| -------- | -----: | -------: | ------: | --------------- |
| B200     |     36 |       41 |       0 | Passed          |

## Renormalization correctness

| Case | Previous fallback | Corrected fallback | Observed difference | Reference |
| ---- | ----------------- | ------------------ | ------------------: | --------- |
| Top-k cutoff ties | Kept exactly `k`; ties broken by sort order | Keeps every probability at or above the kth-value cutoff | 429 support differences / 1536 rows | FlashInfer cutoff semantics |
| Top-p `p=1.0` | Compared float32 cumsum against literal `1.0` | Compares against each row's own total probability | FlashInfer kept 4 / 133177 nonzero tokens on a peaked row | Preserve full row support |

## Compatibility policy

| Path | Policy |
| ---- | ------ |
| Top-k | Match FlashInfer value-cutoff support semantics |
| Top-p `p<1.0` | Match nucleus cutoff semantics |
| Top-p `p=1.0` | Preserve the full normalized row; do not regress to AOT behavior |

## Renormalization implementation

| Change | Before | After | Result |
| ------ | ------ | ----- | ------ |
| Pivot selection | `torch.sort`, `O(V log V)` | Exact `torch.topk`, `O(V)` selection | Same cutoff semantics |
| Apply + normalize | Materialized masked copy | Fused HIP Triton kernel; one vocabulary read | 3.7 GB → 2.8 GB traffic |
| Top-p prefix | 1024 entries | 4096 entries | Top-k selection remains flat through 4096 |
| Nucleus 1024–4096 | 13.4 ms sort fallback | 2.88 ms prefix path | 16.95 ms → 2.88 ms |
| Rows already within prefix | Baseline | Wider prefix | +0.19 ms |

## Shipped platform paths

| Platform | Tree verification | Top-k / top-p renormalization |
| -------- | ----------------- | ----------------------------- |
| MI355X / ROCm | `tree_speculative_sampling_target_only_triton` | `top_k_renorm_probs_triton`, `top_p_renorm_probs_triton` |
| B200 / CUDA | `sgl_kernel.tree_speculative_sampling_target_only` | FlashInfer `top_k_renorm_probs`, `top_p_renorm_probs` |

## Cross-platform performance

1536 rows, vocabulary 151936; probability distributions swept by top-1 mass.

| Operation | Distribution / batch | Faster platform | Margin |
| --------- | -------------------- | --------------- | -----: |
| Tree verification | bs=256 | B200 AOT | 1.20x / 0.038 ms |
| Top-k renormalization | all rows | B200 AOT | 1.81x |
| Top-p renormalization | top1 ≤ 0.07 | B200 AOT | 5.2–6.5x |
| Top-p renormalization | top1 0.20–0.60 | B200 AOT | 1.31–1.45x |
| Top-p renormalization | top1 ≥ 0.76 | MI355X Triton | 1.37–1.40x |

## Remaining performance boundary

| Path | Trigger | Cost |
| ---- | ------- | ---- |
| `torch.sort` fallback | Nucleus exceeds 4096 entries | Approximately 5x vs B200 AOT |
