# PR25742 GLM-5.1-MXFP4 Accuracy Bisect Notes

Last updated: 2026-05-25

## TL;DR

For `amd/GLM-5.1-MXFP4` on MI355X, GSM8K accuracy is healthy at TP=8 across images, but TP=2 regresses from ~0.94 to ~0.58-0.62 starting between the 2026-04-30 and 2026-05-01 docker images.

The regression is confirmed to come from the baked `aiter` version, not from SGLang Python code:

- 2026-04-30 image: SGLang `aa7491144` + aiter `v0.1.12.post1` -> TP=2 GOOD, ~0.939
- 2026-05-01 image: aiter upgraded to `0.1.12.post2.dev150+ga6bb49937` -> TP=2 BAD, ~0.60
- Swap test: 2026-04-30 SGLang `aa7491144` + 2026-05-01 aiter `a6bb499` -> TP=2 BAD, ~0.623

Current aiter sub-bisect status: the first-bad commit is `47b096643` (index 40), which added and retuned BF16 GEMM model configs. The regression is caused by tuned config selection, not by a kernel source change in that commit.

High-level root cause:

- `267a450be` (index 39) is GOOD: GSM8K TP=2 accuracy `0.934`, Invalid `0.001`.
- `47b096643` (index 40) is BAD: GSM8K TP=2 accuracy `0.598`, Invalid `0.336`.
- Reverting the `model_configs` changes from `47b096643` back to index 39 restores GOOD: accuracy `0.938`, Invalid `0.000`.
- Shape overlap analysis shows GLM-5.1 BF16 GEMM shapes only overlap `glm5_bf16_tuned_gemm.csv` among the files added/changed by `47b096643`.
- Reverting all changed model config inputs to the index 39 state restores accuracy.
- Removing `glm5_bf16_tuned_gemm.csv` does not materially hurt performance on the later 2026-05-24 image.
- The 2026-05-24 image is also accuracy-healthy with its original baked `glm5_bf16_tuned_gemm.csv`: TP=2 GSM8K accuracy `0.938`, Invalid `0.001`.
- Important correction: the 2026-05-01 BAD aiter endpoint `a6bb499` already contains PR #2803 / `0dad4342d`, which removed GLM-5 Triton tuned GEMM rows. Therefore the persistent 2026-05-01 accuracy drop is not explained by Triton rows alone; the current leading suspects are the a6bb499-era tuned CSV row choices and/or FlyDSL split-K behavior that was later changed/fixed.

## Docker Run Template

Use the user's container launch template and replace `DOCKER_NAME` with the target image:

```bash
[ -d /data2 ] && export DATA=/data2 || export DATA=/data
docker run -it --rm --privileged --name=$(whoami)_HiCache --network=host --device=/dev/kfd \
    --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --ipc=host --shm-size=32g -e HOME=$HOME -w $HOME/SGLang-benchmarks/ \
    -v $DATA/:/data/ -v /home:/home -v $HOME:$HOME -e USER=$HOME -e TERM=xterm -v /home:/home \
    DOCKER_NAME
```

For `v0.5.10rc0` and related older images, non-interactive `docker exec bash -c` does not source `/etc/bash.bashrc`, so set:

```bash
export PYTHONPATH=/sgl-workspace/sglang/python:/sgl-workspace/aiter:${PYTHONPATH:-}
```

## Accuracy Command

The primary bisect signal uses the GSM8K command extracted from `GLM.sh`, not PR #25742's `lm_eval` chat-completions path:

```bash
python3 /sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py \
    --port 8552 \
    --num-questions 1200 \
    --parallel 1200
```

This path uses SGLang's native `/generate` flow. It avoids the separate `lm_eval` issue where `--reasoning-parser glm45` routes text into `reasoning_content` while lm-eval reads only `message.content`.

## Initial Docker Matrix

All rows use `amd/GLM-5.1-MXFP4` and the MI355X launch flags from `GLM.sh`.

| Docker image | Date | TP=8 | TP=2 | Verdict |
|---|---:|---:|---:|---|
| `lmsysorg/sglang-rocm:v0.5.10rc0-rocm720-mi35x-20260415` | 2026-04-15 | 0.941, Invalid 0.001 | 0.943, Invalid 0.000 | GOOD |
| `rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260503` | 2026-05-03 | 0.932, Invalid 0.005 | 0.623, Invalid 0.295 | BAD at TP=2 |
| `rocm/sgl-dev:v0.5.11-rocm720-mi35x-20260510` | 2026-05-10 | 0.927, Invalid 0.002 | 0.590, Invalid 0.357 | BAD at TP=2 |
| `lmsysorg/sglang-rocm:v0.5.12-rocm720-mi35x-20260517` | 2026-05-17 | 0.917, Invalid 0.004 | 0.579, Invalid 0.337 | BAD at TP=2 |

Conclusion from this matrix:

- TP=8 remains healthy.
- TP=2 was healthy on 2026-04-15 but bad by 2026-05-03.
- The issue is a real regression, not an inherent TP=2 limitation.

## Docker Date Bisect

Daily image bisect narrowed the regression onset to 2026-04-30 -> 2026-05-01:

| Image date | TP=2 accuracy | Invalid | Verdict |
|---:|---:|---:|---|
| 2026-04-22 | 0.936 | 0.000 | GOOD |
| 2026-04-23 | 0.933 | n/a | GOOD |
| 2026-04-28 | 0.932 | 0.002 | GOOD |
| 2026-04-30 | 0.939 | 0.000 | GOOD |
| 2026-05-01 | 0.601 | 0.338 | BAD |
| 2026-05-24 | 0.938 | 0.001 | GOOD with original baked `glm5_bf16_tuned_gemm.csv` |

The 2026-04-30 image contains SGLang `aa74911448`. The 2026-05-01 image contains SGLang `4a50cd781e`.

Testing SGLang Python commits inside the 2026-05-01 container did not flip the result back to GOOD:

| Checked SGLang commit | Reason | TP=2 result |
|---|---|---:|
| `0acc569ed` | parent of SGLang PR #23597 | 0.623 BAD |
| `dc395bc05` | parent of SGLang PR #23654 | 0.593 BAD |
| `cf4f46209` | parent of SGLang PR #23811 | 0.609 BAD |

This pointed away from SGLang Python and toward build-time dependencies.

## Aiter Root-Cause Confirmation

Aiter versions:

| Image | SGLang HEAD | Aiter version |
|---|---|---|
| 2026-04-30 GOOD | `aa7491144` | `v0.1.12.post1` |
| 2026-05-01 BAD | `4a50cd781` | `0.1.12.post2.dev150+ga6bb49937` |

Swap test:

| Config | TP=2 accuracy | Invalid | Verdict |
|---|---:|---:|---|
| 2026-04-30 SGLang `aa7491144` + original aiter `v0.1.12.post1` | 0.939 | 0.000 | GOOD |
| 2026-04-30 SGLang `aa7491144` + swapped 2026-05-01 aiter `a6bb499` | 0.623 | 0.311 | BAD |

Only aiter changed. This confirms the regression lives in the aiter window:

```text
(v0.1.12.post1, a6bb499]
```

Do not claim `a6bb499` / PR #2879 itself is first-bad yet. It is only the bad window endpoint.

## Aiter Sub-Bisect Progress

The aiter window contains about 150 commits. Full rebuild with `PREBUILD_KERNELS=1` is too slow because it compiles many kernels that GLM-5.1 TP=2 does not use. The current working method is source checkout plus selective JIT rebuild of modules used by the run.

Known results:

| Index | Aiter commit | Notes | flydsl | TP=2 accuracy | Invalid | Verdict |
|---:|---|---|---|---:|---:|---|
| baseline | `v0.1.12.post1` | original 2026-04-30 aiter | `0.1.2` | 0.930-0.939 | ~0.000 | GOOD |
| 14 | `2f7b13fc5` | parent of PR #2693 | `0.1.2` | 0.939 | n/a | GOOD |
| 15 | `5282d715` | PR #2693, `fused_dynamic_mxfp4_quant_moe_sort` dispatch fix | `0.1.2` | 0.932 | 0.002 | GOOD |
| 21 | `b025f094` | PR #2700, optimize `fused_dynamic_mxfp4_quant_moe_sort_hip` small-M path | `0.1.2` | 0.932 | n/a | GOOD |
| 31 | `d87e5991` | PR #2717, replace CK/CK_TILE in MLA Reduce/Metadata with OPUS | `0.1.2` | 0.935 | 0.001 | GOOD |
| 38 | `e2ab3041` | PR #2741, add FlyDSL GEMM AOT precompile support | `0.1.3.1` | 0.948 | 0.000 | GOOD |
| 39 | `267a450be` | PR #2739, gather support qk_nope_head_dim != v_head_dim | `0.1.3.1` | 0.934 | 0.001 | GOOD |
| 40 | `47b096643` | PR #2733, add/retune BF16 GEMM model configs | `0.1.3.1` | 0.598 | 0.336 | BAD / first-bad |
| 40 + good configs | `47b096643` | `model_configs` reverted to index 39 state | `0.1.3.1` | 0.938 | 0.000 | GOOD |
| 40, no GLM-5 BF16 config | `47b096643` | `glm5_bf16_tuned_gemm.csv` removed from `model_configs` | `0.1.3.1` | 0.645 | 0.293 | BAD in first-bad-era image |
| 41 | `f326553f` | PR #2742, hoist introspection out of per-call ctype dispatch | `0.1.3.1` | 0.607 | 0.318 | BAD |
| 44 | `fde73e19a` | PR #2746, rebase linear attention for new flydsl version | `0.1.3.1` | 0.627 | 0.308 | BAD |
| 58 | `42117777` | PR #2756, use `AITER_CONFIGS` for FlyDSL AOT defaults | `0.1.3.1` | 0.630 | 0.303 | BAD |
| 85 | `ca33757e4` | midpoint around PR #2775 / gptoss area | `0.1.3.1` | 0.582 | n/a | BAD |
| 148 | `c71075ced` | parent of PR #2927, after many later changes | mixed during earlier test | 0.608 | 0.323 | BAD |
| 149 | `da67a099` | PR #2927 / parent of PR #2879 | mixed during earlier test | 0.657 after JIT `module_cache` rebuild | n/a | BAD |
| 150 | `a6bb499` | endpoint in 2026-05-01 image / PR #2879 | image baked | ~0.60 | ~0.31-0.36 | BAD |

Final narrowed range:

```text
267a450be (GOOD, index 39) < first-bad == 47b096643 (BAD, index 40)
```

Latest notes:

- `d87e5991` was tested with `flydsl==0.1.2` and is GOOD, so PR #2717 is not the first bad commit.
- `42117777` is BAD, shrinking the upper bound from index 85 to index 58.
- `fde73e19a` initially failed to launch with `flydsl==0.1.2`: aiter expected `0.1.3.1`, disabled CK/HIP ops, then failed with missing `aiter.fmoe`.
- Retesting `fde73e19a` with `flydsl==0.1.3.1` succeeded and is BAD, shrinking the range to `(31, 44]`.
- `e2ab3041` is GOOD, shrinking the range to `(38, 44]`.
- `f326553f` is BAD, shrinking the range to `(38, 41]`.
- `267a450be` is GOOD and `47b096643` is BAD, identifying `47b096643` as first-bad.

## First-Bad Config Analysis

Commit `47b096643` only changes BF16 GEMM tuning CSVs:

```text
aiter/configs/model_configs/dsv3_bf16_tuned_gemm.csv
aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv
aiter/configs/model_configs/glm5_untuned_gemm_bf16.csv
aiter/configs/model_configs/gptoss_bf16_tuned_gemm.csv
aiter/configs/model_configs/kimik2_bf16_tuned_gemm.csv
aiter/configs/model_configs/llama405B_bf16_tuned_gemm.csv
aiter/configs/model_configs/llama70B_bf16_tuned_gemm.csv
aiter/configs/model_configs/qwen32B_bf16_tuned_gemm.csv
```

There is no C++/HIP/Triton kernel source change in this commit. The regression is caused by selecting tuned BF16 GEMM configs.

Shape overlap analysis against the GLM-5.1 BF16 GEMM shapes showed:

- `glm5_bf16_tuned_gemm.csv`: 88 overlapping shapes.
- `dsv3_bf16_tuned_gemm.csv`, `gptoss_bf16_tuned_gemm.csv`, `kimik2_bf16_tuned_gemm.csv`: 0 overlapping GLM shapes.
- Newly added Llama/Qwen configs: 0 overlapping GLM shapes.

`glm5_bf16_tuned_gemm.csv` backend distribution:

| Backend | Rows | GSM8K TP=2 result when isolated |
|---|---:|---|
| `asm` | 42 | GOOD, accuracy ~0.939 |
| `flydsl` | 30 | GOOD, accuracy ~0.936 |
| `triton` | 16 | GOOD, accuracy ~0.940 |
| `flydsl + triton` | 46 | GOOD, accuracy ~0.930 |
| `asm + flydsl` | 72 | GOOD, accuracy ~0.929-0.948, but slower |
| full config | 88 | BAD, accuracy ~0.598, Invalid ~0.336 |

Interpretation:

- The bad behavior is in the GLM-5 BF16 GEMM tuned config contents.
- It is not explained by a single backend family being universally wrong.
- Single-backend and two-backend subsets tested so far are GOOD, while the full config is BAD. The remaining likely cause is a specific row or combination of rows in `glm5_bf16_tuned_gemm.csv`.
- On the first-bad-era image, simply moving only `glm5_bf16_tuned_gemm.csv` out was not enough to recover accuracy until all changed model configs were restored to the index 39 state. This suggests cached/merged config state must be cleared carefully (`/tmp/aiter_configs/bf16_tuned_gemm.csv*`) and all model config inputs must be controlled during tests.

## 2026-05-24 Image Accuracy and Performance Check

The later image `rocm/sgl-dev:v0.5.12.post1-rocm720-mi35x-20260524` was tested with `GLM.sh` using:

```bash
./GLM.sh \
  --model /data/huggingface/hub/amd/GLM-5.1-MXFP4 \
  --tp 2 \
  --docker rocm/sgl-dev:v0.5.12.post1-rocm720-mi35x-20260524
```

Two variants were compared:

- Original image contents with baked `glm5_bf16_tuned_gemm.csv`.
- Same image with `/sgl-workspace/aiter/aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv` deleted and `/tmp/aiter_configs/bf16_tuned_gemm.csv*` cleared.

Result directories:

```text
/home/jacchang/SGLang-benchmarks/results/rocm_sgl-dev-v0.5.12.post1-rocm720-mi35x-20260524/GLM-5.1-MXFP4-bench-orig_glm5_bf16_csv
/home/jacchang/SGLang-benchmarks/results/rocm_sgl-dev-v0.5.12.post1-rocm720-mi35x-20260524/GLM-5.1-MXFP4-bench-no_glm5_bf16_csv
```

Accuracy summary:

| Case | TP=2 GSM8K accuracy | Invalid | Verdict |
|---|---:|---:|---|
| Original baked `glm5_bf16_tuned_gemm.csv` | 0.938 | 0.001 | GOOD |
| No GLM-5 BF16 CSV | 0.937 | 0.000 | GOOD |

This means the 2026-05-24 image does not reproduce the TP=2 accuracy drop even with its original baked GLM-5 BF16 tuned config.

Performance summary:

| Case | Original output tok/s | No GLM-5 BF16 CSV output tok/s | Delta |
|---|---:|---:|---:|
| 8192:1024, conc4 | 155.46 | 157.50 | +1.31% |
| 8192:1024, conc16 | 350.59 | 348.05 | -0.72% |
| 8192:1024, conc32 | 486.27 | 480.98 | -1.09% |
| 1024:1024, conc4 | 181.77 | 184.85 | +1.69% |
| 1024:1024, conc16 | 479.45 | 473.94 | -1.15% |
| 1024:1024, conc32 | 731.50 | 717.73 | -1.88% |

Conclusion from the 2026-05-24 performance check:

- The original baked `glm5_bf16_tuned_gemm.csv` is accuracy-healthy on this image.
- Removing `glm5_bf16_tuned_gemm.csv` has no meaningful performance cost in this benchmark sweep.
- All output-throughput deltas are within about +/-2%.
- Logs from the no-CSV run confirm fallback behavior: GLM BF16 GEMM shapes print `not found tuned config ... will use default config`.

## Current Fix-Window Hypothesis

Local aiter history shows:

```text
47b096643  2026-04-15  first-bad config commit; GLM-5 CSV has 88 rows: asm=42, flydsl=30, triton=16
0dad4342d  2026-04-20  PR #2803 removes GLM-5 Triton tuned GEMM rows; GLM-5 CSV has 72 rows: asm=47, flydsl=25
a6bb49937  2026-04-29  2026-05-01 image aiter endpoint; still 72 rows: asm=47, flydsl=25; TP=2 BAD
HEAD/0524 era          GLM-5 CSV has 72 rows: asm=47, flydsl=16, torch=9; TP=2 GOOD on 2026-05-24 image
```

PR #2803 is already included in the 2026-05-01 BAD endpoint, so the 2026-05-01 failure is not simply "Triton rows were selected." The next tests should determine whether the 2026-05-24/current CSV alone fixes the 2026-05-01 image, or whether a later aiter code fix such as FlyDSL split-K synchronization is required.

## Rebuild / JIT Methodology

Important lessons from failed methodology attempts:

1. Source-only checkout is not enough if prebuilt `.so` files remain.
2. `module_cache.so` was prebuilt in the image and masked changes from PR #2879 until deleted.
3. Some aiter commits require different `flydsl` versions:
   - commits through index 31 worked with `flydsl==0.1.2`
   - index 44 and later need `flydsl==0.1.3.1` in practice, even though index 44's `requirements.txt` still says `0.1.2`
   - index 85 needed `flydsl==0.1.3.1`
   - `flydsl==0.1.5` was not compatible with the old `v0.1.12.post1` baseline in one validation attempt
4. Full `PREBUILD_KERNELS=1` rebuild is not practical for each bisect step; it can take hours.

Modules observed in GLM-5.1 TP=2 logs and likely worth deleting before each aiter checkout/test:

```text
module_aiter_core
module_activation
module_cache
module_custom
module_custom_all_reduce
module_fused_qk_norm_rope_cache_quant_shuffle
module_mla_metadata
module_moe_asm
module_moe_ck2stages_fp4x2_fp4x2_preshuffle_*_silu_per_1x32_mulWeightStage2
module_moe_cktile2stages
module_moe_sorting
module_norm
module_quant
module_rmsnorm_quant
module_rope_2c_cached_positions_fwd
```

Practical per-step flow:

```bash
# Inside the bisect container
cd /sgl-workspace/aiter
git checkout <aiter-commit>

# Match flydsl to the aiter era.
pip install 'flydsl==0.1.2'       # for old commits around index 14-31
# or:
pip install 'flydsl==0.1.3.1'     # for later commits around index 85

# Remove stale prebuilt/JIT modules that can mask the checkout.
rm -f aiter/jit/module_aiter_core*.so
rm -f aiter/jit/module_activation*.so
rm -f aiter/jit/module_cache*.so
rm -f aiter/jit/module_custom*.so
rm -f aiter/jit/module_custom_all_reduce*.so
rm -f aiter/jit/module_fused_qk_norm_rope_cache_quant_shuffle*.so
rm -f aiter/jit/module_mla_metadata*.so
rm -f aiter/jit/module_moe_asm*.so
rm -f aiter/jit/module_moe_ck2stages_fp4x2_fp4x2_preshuffle_*_silu_per_1x32_mulWeightStage2*.so
rm -f aiter/jit/module_moe_cktile2stages*.so
rm -f aiter/jit/module_moe_sorting*.so
rm -f aiter/jit/module_norm*.so
rm -f aiter/jit/module_quant*.so
rm -f aiter/jit/module_rmsnorm_quant*.so
rm -f aiter/jit/module_rope_2c_cached_positions_fwd*.so

export PYTHONPATH=/sgl-workspace/sglang/python:/sgl-workspace/aiter:${PYTHONPATH:-}
```

Then launch server with the same MI355X GLM-5.1-MXFP4 TP=2 arguments and run the GSM8K command above.

Polling policy requested by user: do not wait longer than 1 minute before checking whether server/bench is ready or finished.

## Current Next Step

The commit-level bisect is complete:

```text
first-bad = 47b096643 (index 40)
parent GOOD = 267a450be (index 39)
```

The next useful work is row-level isolation inside `glm5_bf16_tuned_gemm.csv`:

```text
1. Start from the full GLM-5 BF16 tuned config, which is BAD.
2. Bisect rows or shape groups inside that CSV.
3. Clear /tmp/aiter_configs/bf16_tuned_gemm.csv* before every run.
4. Run GSM8K TP=2 as the correctness signal.
```

Backend-level tests so far do not identify a single bad backend family; tested single-backend and two-backend subsets were GOOD, while the full config was BAD.

