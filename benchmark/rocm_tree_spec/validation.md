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
