# ROCm 樹狀推測解碼驗證器：從缺席到 Triton

本文記錄 ROCm 上「沒有 tree 驗證器」如何演變成現在的 Triton 實作，以及要與 NVIDIA 的 CUDA 版本做有意義的比較，還缺哪些東西。

分支：`rocm-tree-fixes`（追蹤 `max/RM/rocm-tree-spec-sampling-triton`）
驗證硬體：AMD Instinct MI355X (gfx950)，ROCm 7.2，torch 2.9.1，Triton 3.6.0
撰寫時間：2026-08-06

---

## 1. 起點：ROCm 為什麼沒有 tree

推測解碼的驗證階段有三種策略，彼此正交於 draft 演算法（EAGLE/EAGLE3/DFLASH…）和候選結構（chain/tree）：

| 驗證策略 | 需要 draft 分布 q？ | 適用結構 | 接受率 |
|---|---|---|---|
| greedy | 否 | 皆可 | 只接受 argmax 相符者 |
| target-only (SpecInfer) | 否 | tree 為主 | `Σᵢ p(xᵢ)`，隨寬度成長 |
| rejection sampling | **是** | 僅 chain | `1 − TV(p, q)`，寬度 1 時最優 |

CUDA 上這三種都有 kernel。ROCm 的問題出在 **sgl-kernel 的 ROCm build 沒有編進兩組來自 flashinfer 的 kernel**：

- `tree_speculative_sampling_target_only`（源自 `speculative_sampling.cuh`）
- `top_k_renorm_prob` / `top_p_renorm_prob`（源自 `renorm.cu`）

關鍵在於失敗方式很隱蔽：**Python wrapper 在 ROCm 上 import 得動，只有底層的 `torch.ops.sgl_kernel.*` 呼叫會在執行期丟 `AttributeError`。** 所以靜態檢查看不出問題。

當年 ROCm 剛支援推測解碼時（PR #17450），因為沒有可用的替代驗證器，`eagle_utils.py` 的 greedy 判定裡加上了 `_is_hip`：

```python
if sampling_info.is_all_greedy or _is_hip or ...:   # 舊版
    target_predict = torch.argmax(next_token_logits, dim=-1)
```

結果是 **ROCm 上所有請求都被靜默導向 greedy 驗證，無視使用者指定的 temperature / top_p / top_k**。後來 `chain_speculative_sampling_triton` 被加進來，但這個 gate 沒有跟著重新檢視，於是 bug 一直存在。這就是 PR #32922 修的東西。

---

## 2. 三段式修復

### 2.1 PR #32922 — 揭露問題

移除 greedy gate 裡的 `_is_hip`，讓非 greedy 請求真正走隨機驗證。但當時 ROCm 只有 chain kernel 可用，所以 `--speculative-eagle-topk > 1` 的樹狀候選仍無處可去。

### 2.2 Max 的 Triton port — 補上 tree

分支上 6 個 commit：

```
3cabcc5a8d  feat(spec): add Triton target-only tree sampler        (tree_sampling.py, 396 行)
eff981bd53  feat(sampling): add portable probability renormalization
3426ffbbf3  feat(spec): enable Triton tree verification on ROCm    (dispatcher)
a27ebb208e  test(spec): cover Triton tree sampling parity
fec2a0c512  test(spec): cover DFLASH Triton tree integration
3352e346e9  docs: add ROCm tree sampling plan
```

核心是 `tree_speculative_sampling_target_only_triton`，一個請求一個 Triton program，忠實重現 CUDA oracle 的行為：子節點/兄弟節點走訪、兄弟累積機率、`threshold_single` / `threshold_acc` 雙判準、被拒候選的 draft scratch 變更、以及從 `relu(target_probs − draft_probs)` 抽 bonus token。

dispatcher 變成三向分流：

```python
def _get_spec_sampling_verify_fn(use_rejection_sampling: bool):
    if use_rejection_sampling:
        return chain_speculative_sampling_triton      # 顯式要求，僅 topk=1
    if _is_hip:
        return tree_speculative_sampling_target_only_triton   # ROCm
    from sgl_kernel import tree_speculative_sampling_target_only
    return tree_speculative_sampling_target_only      # CUDA/MUSA，AOT
```

DFLASH 也接上同一個 Triton 實作，取代原本 CUDA/MUSA-only 的 gate。

### 2.3 renorm — 原本被刻意留下的洞

Max 的 plan §2 明講「Treat performant Triton renorm as a follow-up optimization: benchmark the exact fallback first」，而那個 benchmark 屬於 `accuracy-performance` todo，狀態是 **cancelled**，所以從未執行。

在 MI355X 上實測後發現這是整個改動裡最嚴重的效能問題：

```
vocab=151936  num_draft_tokens=6  MI355X
   bs    rows  top_k(ms)  top_p(ms)  tree(ms)  renorm/tree
    1       6      0.426      0.765     0.096        12.4x
    8      48      0.750      1.141     0.120        15.8x
   32     192      4.647      5.104     0.141        69.3x
  128     768     11.388     12.521     0.223       107.3x
  256    1536     19.183     20.679     0.334       119.3x
```

batch 256 時 renorm 要 **40ms**，而 tree kernel 只要 0.33ms。原因是兩支 fallback 都對整個 vocab 做完整排序（`torch.sort`，O(V log V)，還要搬 int64 索引），而 flashinfer 的 CUDA kernel 用的是 pivot 多輪縮減，從不排序。

同時發現一個獨立於效能的正確性問題。AOT wrapper 的 docstring 定義的是**閾值語意**：

> We mask out the probabilities less than `threshold` where the cumulative sum of `probs[probs >= threshold]` is `top_p`

也就是保留所有 `p >= threshold` 者，含閾值上全部並列的 token。但 Max 的兩支 fallback 都是**排名截斷**（保留剛好 k 個，以排序順序打破並列）。實測 1536 rows 有 429 筆 support 差異 —— **同一個請求在 CUDA 和 ROCm 上會選到不同的 token 集合**。

修法分兩層：

| 層 | 改動 | 效果 |
|---|---|---|
| 語意 | 排名截斷 → 閾值語意 | 與 flashinfer 對齊 |
| pivot 搜尋 | `torch.sort` → `torch.topk` | 40ms → 8.3ms |
| 套用 + 正規化 | 未融合 torch → Triton | 8.3ms → 5.0ms |

`topk(..., sorted=True)` 就是真正的遞減前綴，所以精確；nucleus 超出前綴長度（1024）的少數 row 才退回完整排序。

融合的部分沒有沿用 upstream `top_p_renorm_triton.py` 的結構。它是「寫出 masked 副本 + 部分和」再「原地正規化」，需 3.7GB 流量；改成「只求和不物化」再「把倒數摺進寫入」後降到 2.8GB，省掉一整趟寫入。

最終結果：

```
  rows  orig(ms)  torch(ms)  triton(ms)  vs orig
     6     1.514      0.388       0.331     4.6x
    48     2.216      0.563       0.448     5.0x
   192     9.863      1.289       0.811    12.2x
   768    23.642      4.543       2.835     8.3x
  1536    38.457      8.305       5.027     7.7x
```

**這張表量的是 logit scale 6.0 的分布**（`softmax(randn * 6.0)`，nucleus 約 40，不觸發 §6.6 說的 sort 退路）。當時沒有註明這件事，後來在 B200 上用未縮放的 `softmax(randn)` 重跑得到 17.78ms，一度被誤判為平台差異。renorm 的成本幾乎完全取決於 nucleus 是否超出 1024 前綴，**任何 renorm 數字都必須連同輸入分布一起記錄才有意義**。

**一個容易踩的浮點陷阱**值得記錄：`cumsum >= top_p` 這個判定不可靠，因為 fp32 的機率總和是 0.99999994 而非 1.0。尖銳分布下前幾項就會捨入到 1.0，導致 `top_p=1.0` 反而砍掉整條尾巴（第一版實作就是如此，測出 225 萬筆 support 差異）。正確做法是以該 row 的實際總和為基準判定捨棄質量：`prefix <= total − (1 − top_p)`。

### 2.4 為什麼是 Triton，而不是直接 hipify

CUDA 版就只是一份 `speculative_sampling.cuh`，很自然會問：ROCm 直接 hipify 不就好了？

**現況不是 hipify 失敗，是這兩顆從來沒進過 ROCm 的建置範圍。** 兩邊走完全不同的建置：`CMakeLists.txt`（557 行）會抓一個固定 commit 的 flashinfer，並把它的 `renorm.cu` 直接編進去；`setup_rocm.py`（130 行）則是一份手動維護的 13 檔案清單，完全沒有 flashinfer。`speculative_sampling.cu` 不在清單裡，因為它的 header 第 22 行 `#include <flashinfer/sampling.cuh>`。

容器內實測，模式很乾淨：

| op | 來源 | ROCm 上 |
|---|---|---|
| `build_tree_kernel_efficient` | `eagle_utils.cu`（在清單裡） | 有 |
| `tree_speculative_sampling_target_only` | 依賴 flashinfer | 無 |
| `top_k_renorm_probs` | flashinfer 自己的 `renorm.cu` | 無 |

**移植面其實不大。** kernel 本體對 hipify 友善：樹的走訪主迴圈是純量邏輯，沒有 `__shfl`、沒有 inline PTX、沒有 tensor core，而那些才是 hipify 真正會卡住的東西。它從 flashinfer 只用到四樣：`vec_t`（`vec_dtypes.cuh`）、`SamplingTempStorage` 與 `DeviceSamplingFromProb`（`sampling.cuh`），以及 `cub::BlockScan`/`BlockReduce`（有 hipCUB/rocPRIM 對應）。要搬的是三個 header，不是整個 library。

兩個要實測才能排除的風險：`DeviceSamplingFromProb` 內部是否假設 32-lane warp（flashinfer 原始碼是建置時才抓，不在磁碟上）；以及 LDS 用量 —— 第 233 行的 `cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize, ...)` 通常意味著超過 48KB 的預設上限，而 AMD 每個 workgroup 的硬上限是 64KB。以 `SamplingTempStorage<1024>` 的結構估算約 4–8KB，比較像是抄過來的防禦性樣板，傾向不是阻礙。

**但兩顆的性質不同。** tree kernel 是 sglang 自己的檔案，hipify 與否是我們能決定的事；renorm 是 flashinfer 自己的 `renorm.cu`，要它就等於要把 flashinfer 的 sampling 整塊帶上 ROCm。AMD 原本的移植（`ROCm/flashinfer`、`AMD-Ecosystem/flashinfer`）現在都已退役並上游化，但重心一直是 attention —— upstream `csrc` 的 188 個檔案裡沒有任何 HIP 命名的檔案，`renorm.cu` 與 `sampling.cu` 仍是 CUDA。所以 renorm 沒有現成的 ROCm 版本可以拿。

**優先序才是決定性的理由。** tree kernel 的 Triton 版是 0.12–0.24ms，renorm 是 5.0ms。就算把 tree kernel hipify 成原生移植並假設快一倍，省下的也是 0.1ms 等級 —— hipify 幫得上忙的恰好是那顆不是瓶頸的，而幫不上忙的 renorm 才是瓶頸。

必須註明的但書：**目前沒有 AOT renorm 的基準數字**，所以那 5.0ms 距離原生實作還差多少是未知的。這正是 §6 要在 B200 上量的東西；在那之前，「hipify 值不值得」沒有數據可以回答。

---

## 3. 目前的分流全貌

| 裝置 | tree 驗證 | chain 驗證（rejection） | renorm |
|---|---|---|---|
| CUDA | AOT (flashinfer) | Triton | AOT (flashinfer) |
| **ROCm** | **Triton** | Triton | **Triton**（pivot 仍為 torch） |
| MUSA | AOT | Triton | AOT |
| CPU / NPU | 各自的 greedy 路徑 | — | 純 torch |

注意 ROCm 的 renorm 並非全 Triton：**pivot 搜尋仍是 `torch.topk`，Triton 只負責套用閾值與正規化**。5.0ms 裡約 4ms 是兩次 topk，套用只剩約 1ms（已逼近 2.8GB 在 5.4 TB/s 下的理論值 0.52ms）。瓶頸已經從「套用」轉移到「選擇」。

---

## 4. 已驗證與未驗證

### 已驗證（MI355X, gfx950）

```
33 passed, 3 skipped, 30 subtests passed
```

- Triton tree kernel 對上獨立的 torch oracle：通過
- 分支樹（topk>1）保持 target 分布：通過
- 種子可重現性：通過
- DFLASH 整合：通過
- dispatch 選擇（HIP→Triton tree、rejection→chain、greedy 不變）：通過
- renorm：4 種分布形狀 × 5 個 top_p × 4 個 top_k、逐 row 異值閾值、人造並列、全零 row、空 batch，support 逐位元相同，最大 TV 距離 1e-7

值得注意的是 **gfx950 是首次驗證**。ROCm 的 sgl-kernel build 預設目標是 gfx942，Max 回報的「ROCm: 10 passed」極可能跑在 MI300 上。

### 未驗證（3 個跳過的測試）

| 測試 | 跳過原因 |
|---|---|
| `test_matches_cuda_aot_oracle` | `torch.version.hip is not None` — CUDA AOT oracle 在 ROCm 不存在 |
| `test_top_k_fallback_matches_kernel` | 探測 `sgl_kernel.top_k_renorm_prob` 失敗 |
| `test_top_p_fallback_matches_kernel` | 同上 |

三者的共同點：**都需要 CUDA AOT kernel 當比較基準，只能在 NVIDIA 機器上啟用**。這意味著「Triton port 與 CUDA 等價」這件事目前**在 ROCm 上無法自證**。

> 後續：這三支已在 B200 上實際執行並通過（`36 passed, skipped 0`），見 §6.5。

---

## 5. 要與 NVIDIA 的 tree 比較，還差什麼

### 5.1 等價性尚未證明

目前 Triton tree kernel 只對過「獨立的 torch oracle」，沒有對過真正的 CUDA AOT kernel。renorm 的情況更微妙：我今天的修改是**對著 in-tree 的 `top_p_renorm_triton.py` 驗證的**，而那支自己也只是宣稱對齊 flashinfer，並非 flashinfer 本身。

**需要的動作**：在一台 NVIDIA 機器上 checkout 同一個分支跑測試。屆時三個 skipped 測試會自動啟用，這是唯一的等價性證明。

### 5.2 完全沒有 tree kernel 的效能對照

我量到的 tree kernel 耗時（0.096–0.334ms）有兩個必須說明的限制：

1. **那是 chain 拓撲**（`retrive_next_sibling` 全為 -1）當佔位符量的，不是真正的樹。真實樹狀走訪會更慢，所以這是下界而非代表值。
2. **沒有 CUDA AOT 的對照數字**，因為 ROCm 上根本沒有那顆 kernel。

Max 的 plan §5 要求的 benchmark 矩陣（batch size、樹寬/深、vocab size、branching factor、threshold 模式）從未執行。

**需要的動作**：在 NVIDIA 機器上做 Triton vs CUDA AOT 的 microbenchmark（同一台機器，兩者都可用），才能知道 Triton port 的相對成本。

### 5.3 已知的實作偏差尚未量化

Triton 版與 CUDA 版有三處已知差異：

| 偏差 | 性質 | 影響 |
|---|---|---|
| `deterministic` 參數被接受後直接 `del` | 語意 | 該旗標在 ROCm 上無效，但未文件化 |
| FP32 累加順序在 CDF 邊界不同 | 數值 | 已在程式碼註解說明 |
| 未指定 `num_warps`（預設 4 = 256 threads，CUDA 為 1024） | 效能 | 未量化 |
| 第二趟 vocab 掃描在 `found_bonus` 後不提早退出 | 效能 | 未量化 |

後兩項是我原先建議優先處理的，但那個判斷是錯的 —— 那是在 0.33ms 裡省零頭，而旁邊有 40ms 在燒。現在 renorm 修完了，這兩項的相對重要性才會上升，但仍需先有 benchmark 才知道值不值得。

### 5.4 沒有 e2e 準確度驗證

Max 自己在 plan §5 訂的合併門檻：

> Require tree mode with `speculative_eagle_topk > 1` to match the no-speculator accuracy distribution; PR 32922's chain-only result is not sufficient.

這個 todo（`accuracy-performance`）狀態是 **cancelled**，從未執行。PR #32922 的實驗是 chain（topk=1）且樣本數 n=8，信賴區間過寬，按他自己的標準不足以證明樹狀模式正確。

### 5.5 沒有 accept length 對照

要判斷「樹到底值不值得」，關鍵指標不是 kernel 耗時而是 **acceptance length** —— 樹的價值在於接受率隨寬度成長（`Σᵢ p(xᵢ)`）。這需要同 model、同 workload、同 sampling params 在兩個平台上跑，記錄每次 verify 平均接受幾個 token。

### 小結：跨平台比較的前置條件

| # | 缺什麼 | 需要的硬體 | 產出 | 狀態 |
|---|---|---|---|---|
| 1 | 等價性證明 | NVIDIA | 3 個 skipped 測試轉為通過 | **已完成**（§6.5） |
| 2 | kernel 效能對照 | NVIDIA | Triton vs CUDA AOT microbenchmark | **已完成**（§6.5） |
| 3 | 真實樹拓撲的耗時 | 兩者 | 取代目前的 chain 佔位量測 | **已完成**（§6.5） |
| 4 | e2e 準確度 | ROCm | topk>1 對上 no-speculator 的分布 | 未做 |
| 5 | accept length | 兩者 | 同 workload 的接受長度對照 | 未做 |

第 1、2 項的瓶頸單純是**沒有 NVIDIA 機器跑這個分支**，2026-08-06 在 B200 上補完。第 4、5 項則是本地就能做的 —— 手上有 MI355X 和客戶的 GLM-5.2-FP8 workload，而這正好是 Max 缺的部分（他 plan 裡的路徑是 `/home/macui/...`，且該 todo 被他自己取消）。

---

## 6. B200 測試計畫

§5 的前置條件 1、2、3 卡在同一件事：沒有 NVIDIA 機器。這節是可以直接照著跑的版本。量測腳本已隨分支進 repo（`benchmark/rocm_tree_spec/`），pull 下去就有。

### 6.0 環境

```bash
git clone git@github.com:Jacob0226/sglang.git
cd sglang && git checkout rocm-tree-fixes
export PYTHONPATH=$PWD/python
```

需要 `sgl_kernel` wheel 含 AOT renorm 與 tree kernel（CUDA 版預設就有）。兩支腳本都會**用一次真實呼叫探測 AOT**，因為 Python wrapper 在缺 kernel 時仍然 import 得動；探測失敗會自動降級成只跑 fallback，不會整支掛掉。

### 6.1 等價性：三個 skipped 測試（前置條件 1）

```bash
python3 -m pytest test/registered/kernels/ops/speculative/ -v
```

ROCm 上的結果是 `33 passed, 3 skipped`。B200 上這三個應該轉為實際執行：

| 測試 | 驗什麼 |
|---|---|
| `test_matches_cuda_aot_oracle` | Triton tree kernel vs CUDA AOT |
| `test_top_k_fallback_matches_kernel` | top_k fallback vs flashinfer AOT |
| `test_top_p_fallback_matches_kernel` | top_p fallback vs flashinfer AOT |

**後兩個有可能失敗，而那會是有價值的結果。** §2.3 的閾值語意修正是對著 in-tree 的 `top_p_renorm_triton.py` 驗的，而那支自己也只是「宣稱」對齊 flashinfer，並非 flashinfer 本身。若失敗，代表我對 flashinfer 語意的解讀有誤，應以 AOT 為準回頭修 `renorm.py`。

### 6.2 renorm：等價性與成本（前置條件 1、2）

```bash
python3 benchmark/rocm_tree_spec/bench_renorm_aot.py
```

同時比較 torch fallback、Triton fallback、flashinfer AOT。correctness 段掃 5 種分布形狀 ×（4 個 top_k + 5 個 top_p + 逐 row 異值），看的是 **support 是否逐位元相同** —— 差一個 token 就代表同一個請求在兩個平台會抽到不同結果。`flat`（每一項都是並列）與 `duplicated`（在截斷點植入精確並列）兩種形狀專門用來壓並列邊界，正是排名截斷與閾值語意會分歧之處。

判準是 `worst support mismatch: 0`。數值差在 1e-7 量級屬 fp32 雜訊，可接受。

cost 段給出 MI355X 上量不到的關鍵數字：**Triton/AOT 比值**。這是判斷「5.0ms 還剩多少空間」的唯一依據，也直接決定 §2.4 的 hipify 值不值得做。

一個實作細節：Triton renorm 註冊時是 `CapabilityRequirement.HIP`，在 CUDA 上 registry 不會選它，所以腳本是直接 import 繞過 registry 才能做對照。

已在 MI355X 上以 torch 實作假扮 AOT 跑過完整路徑（50 組全部 support 相同、最大數值差 1.8e-7），所以到 B200 只會換掉參考實作，不會遇到腳本本身的問題。

### 6.3 tree kernel：Triton vs CUDA AOT（前置條件 2、3）

```bash
python3 benchmark/rocm_tree_spec/bench_tree_sampling.py --logit-scale 16 --iters 20
```

AOT 可用時會自動多出 `aot(ms)` 與 `triton/aot` 兩欄。這支用的是**真實 k-ary 樹拓樸**而非 §5.2 那個 chain 佔位，所以順帶補掉前置條件 3。

`--logit-scale 16` 是在 MI355X 上校準過的。`softmax(randn)` 在 151936 vocab 下近乎均勻，每個候選機率約 6.6e-6，會在根節點就全部被拒、accept length 恆為 0，等於完全沒測到接受路徑。scale 16 時 top-1 約 0.81，accept length 落在 1.2–1.7 的合理區間。要重新校準用 `--calibrate`。

### 6.4 要帶回來的數據

| # | 數據 | 用途 |
|---|---|---|
| 1 | 三個測試的 pass/fail | 等價性證明，或修正方向 |
| 2 | renorm 的 `worst support mismatch` | 閾值語意是否真的對齊 flashinfer |
| 3 | renorm 的 Triton/AOT 比值 | 決定 hipify 或進一步優化值不值得 |
| 4 | tree 的 Triton/AOT 比值 | Triton port 的相對成本 |
| 5 | 真實樹拓樸下的 tree 耗時 | 取代目前的 chain 佔位量測 |

第 3、4、5 項在 MI355X 上都量不到，因為那裡根本沒有 AOT kernel 可當基準。

### 6.5 執行結果（B200 × 8，CUDA 13.0，torch 2.11.0，sgl_kernel 0.4.5，2026-08-06）

容器 `jacchang_GLM`（`lmsysorg/sglang:v0.5.16-cu130-runtime`）。量測期間 8 張卡都空閒、無其他 process。

**環境上唯一的意外**：`bench_renorm_aot.py` 的 AOT 探測失敗，但不是缺 kernel，是**命名版本落差**。in-tree wrapper 已改名為複數 `top_k_renorm_probs`，而容器裡的 wheel（0.4.5）只匯出單數 `top_k_renorm_prob`；底層 op `torch.ops.sgl_kernel.top_k_renorm_probs` 兩者都是複數。探測已改成兩種拼法都試。這正好是 §6.0 那句「探測失敗會自動降級」的反面教材 —— 降級太安靜，會把版本落差誤報成硬體不支援。

#### §6.1 的結果 — 等價性：三個測試全過

```
36 passed, 41 subtests passed   (skipped 0)
```

`test_matches_cuda_aot_oracle`、`test_top_k_fallback_matches_kernel`、`test_top_p_fallback_matches_kernel` 全數實際執行並通過。**前置條件 1 成立：Triton tree kernel 與 CUDA AOT 等價。**

#### §6.2 的結果（一）— renorm 等價性：top_k 完全相同，top_p 只在 `top_p=1.0` 分歧

腳本的 `worst support mismatch: 0` 判準**沒過**（報 2671270）。但拆開看方向與質量後，結論與 §6.2 預期的「解讀有誤」相反：

| 類別 | support 差異 | 最大 TV | 判讀 |
|---|---|---|---|
| top_k（5 形狀 × 4 個 k，共 20 組） | **全部 0** | 3.1e-07 | 閾值語意與 flashinfer 逐位元一致 |
| top_p < 1.0 | 最多 **2** token（總量 9.7M） | 3.9e-04 | 恰好落在截斷點的並列，單 token 等級 |
| top_p = 1.0 | 最多 3493103 | **2.2e-07** | 見下 |

`top_p=1.0` 的差異方向才是關鍵。`support mismatch` 是 XOR 計數，只說有幾個 token 不一致，不說是誰丟的；拆成「只有 AOT 留」與「只有我們留」兩欄後（peaked，64 列）：

```
row  in_nonzero  keep_aot  keep_ours  aot_only  ours_only  pivot
  0      149998    132917     149998         0      17081  0.000e+00
  3      133177         4     133177         0     133173  0.000e+00
  7      150611    137594     150611         0      13017  0.000e+00

rows where AOT keeps fewer than we do: 63/64
rows where pivot > 0 (not a true no-op): 1/64
```

**63 列裡我們的 pivot 解出來是 0（真正的 no-op，保留全部非零 token），而 AOT 砍掉了尾巴** —— row 3 最極端，輸入有 133177 個非零 token，AOT 只留 4 個。這正是 §2.3 那個 fp32 陷阱：flashinfer 拿 cumsum 跟 `top_p` 比，而該列實際總和是 1.000000119，尖銳分布下前幾項就湊到 1.0，尾巴整條丟掉。我們改以該列實際總和為基準算捨棄質量（`prefix <= total − (1 − top_p)`），`top_p=1.0` 時右邊就是 `total − 0`，所以是真 no-op。

剩下那 1 列方向相反（我們的 pivot > 0，丟掉了 AOT 留的東西），它就是上表 `top_p = 1.0` 那格裡 `aot_only` 部分的來源。兩個方向涉及的質量都微不足道：AOT 留而我們丟的最大單一機率是 2.86e-08，每列獨有的總質量 ≤ 1.5e-07。所以 support 差了幾百萬個 token，TV 距離卻只有 2.2e-07 —— 全表最小。

**所以不需要照 §6.2 預案裡「以 AOT 為準回頭修 `renorm.py`」那條路走。** 我們的行為比 AOT 更正確；`top_p=1.0` 應該是 no-op，而 AOT 不是。也因為移動的質量微不足道，pytest 那兩支以分布距離（TV < 1e-2）比較的測試才會通過，而逐位元的 support 判準會失敗。這說明**判準本身該修**：support 相同適用於 top_k 與 top_p < 1，`top_p=1.0` 只能用 TV 判。

#### §6.2 的結果（二）— renorm 成本：瓶頸不是 Triton，是 overflow 退路

vocab=151936，1536 rows：

```
   op    torch(ms)  triton(ms)  aot(ms)  triton/aot
 top_k       5.434       3.086    1.323       2.33x
 top_p      20.129      17.778    3.447       5.16x
```

top_p 那個 5.16x 具誤導性，因為它量的是 overflow 退路而不是 kernel。`top_p_pivots` 只用 `torch.topk(probs, 1024)` 找 pivot；門檻落在 1024 名以外時，前綴看不到答案，只能退回對全部 151936 個 token 做完整 `torch.sort`（12.8ms）。而 benchmark 用的 `softmax(randn)` 在 151936 vocab 下近乎均勻（`top1 ≈ 0.0003`），要湊到 `top_p=0.95` 得收集約 13 萬個 token —— **每一列都走了退路**。17.78ms = 12.8ms 的 sort + 3.4ms 的實際工作。

那種 `top1 ≈ 0` 的分布在真實推論裡不存在。把 logit 乘上 scale 掃過尖銳度（1536 rows，iters=100，每個 AOT 點取 3 次試驗的最小值；三次都落在 0.01ms 內）：

```
 scale    top1  nucleus  aot(ms)  triton(ms)   winner   ratio
     1   0.000     1024    3.448      17.771      aot    5.15x
     2   0.008     1024    2.705      17.893      aot    6.61x
     4   0.197      975    2.201      16.542      aot    7.52x
     6   0.454       62    2.060       3.403      aot    1.65x
     8   0.580       12    1.983       3.402      aot    1.72x
    12   0.738        3    1.952       3.403      aot    1.75x
    16   0.804        2    3.940       3.401   triton    1.16x
    24   0.869        1    6.543       3.401   triton    1.92x
    32   0.901        1    4.880       3.401   triton    1.43x
```

兩件事：

1. **Triton 是水平線（3.40ms），成本與資料無關** —— `topk(1024)` 加上兩趟固定的 vocab 掃描，不管分布長什麼樣都一樣。overflow 一消失（scale ≥ 6）就再也不動。
2. **AOT 是 U 形曲線（1.95–6.54ms），成本與資料相關。** 分布越尖它反而越慢：scale 12 是 1.95ms，scale 24 到 6.54ms。這符合 pivot 多輪縮減的性質 —— top-1 佔 0.87、其餘極小時，要收斂到精確門檻需要更多輪。§2.3 說 flashinfer「用 pivot 多輪縮減，從不排序」，代價就是輪數隨分布變動。

所以交叉點在 `nucleus ≤ 2` / `top1 ≥ 0.80` 附近：比這平，AOT 快 1.65–7.5x；比這尖，Triton 快 1.16–1.92x。**整體而言 AOT 仍然較快，但不是全面較快，而那個最大的 5–7.5x 差距來自 `torch.sort` 退路，不是 Triton kernel。**

pivot 搜尋佔多少，完全看在哪個 regime。B200 上 `torch.topk(probs, 1024)` 是 2.69ms（1536 rows）：在 overflow regime 只佔 top_p 總時間的 15%，但在 fast path（3.40ms）佔了約 79%。**所以 §7 那個「兩次 topk 合併成一次」的技術債在真實分布下是有價值的** —— 那裡 topk 就是主要成本，§7 估的約 1.6x 站得住；只有在 flat 分布下它才被 `torch.sort` 蓋掉。

另外有一個待查的落差：同一份 Triton 程式在 B200 上是 20.9ms（top_k 3.09 + top_p 17.78），而 §2.3 在 MI355X 上記的是 5.0ms。兩邊都用 `softmax(randn)`，若 MI355X 當時也是每列 overflow，那 MI355X 的 `torch.sort` 得比 B200 快 6 倍才對得上，這不太合理；比較可能是 §2.3 的量測其實沒有觸發 overflow（該節寫「nucleus 超出前綴長度的**少數** row 才退回完整排序」，正暗示作者當時觀察到的 overflow 比例很低，而 B200 上是 100%）。需要在 MI355X 上重跑同一支腳本並印出 overflow 比例才能定論。**在釐清之前，§2.3 那個 5.0ms 不該當成與 B200 的 17.78ms 可直接比較的數字。**

#### §6.3 的結果 — tree kernel：真實 k-ary 樹拓樸下 1.1–1.2x

`--logit-scale 16 --iters 20`，摘 ndt=8 的兩端（ndt=16 的數字幾乎相同）：

```
   bs  ndt  width  depth  triton(ms)   aot(ms)  triton/aot  accept_len
    1    8      2      4      0.1072    0.0402       2.67x        0.00
    8    8      2      4      0.1372    0.0765       1.79x        1.25
   32    8      2      4      0.1654    0.0919       1.80x        1.47
  128    8      2      4      0.2080    0.1148       1.81x        1.10
  256    8      2      4      0.2228    0.1888       1.18x        1.20
```

**Triton/AOT 在 bs=1 是 2.0–2.7x，隨 batch 收斂到 bs=256 的 1.12–1.19x。** 絕對差距在 serving batch 下是 0.03ms。accept_len 落在 0.88–2.25，證實 scale 16 的校準在 B200 上同樣有效（bs=1 那列的 0.00 是 n=1 的取樣雜訊，不是失敗）。這同時補掉前置條件 3：真實樹拓樸取代了 §5.2 的 chain 佔位。

#### 結論：hipify 不值得做

§2.4 說「目前沒有 AOT renorm 的基準數字，所以那 5.0ms 距離原生實作還差多少是未知的」。現在知道了：

| | 誰快 | 倍數 | 判讀 |
|---|---|---|---|
| tree kernel（bs=256） | AOT | 1.12–1.19x | 移植只能省 0.03ms，不值得 |
| top_k renorm | AOT | 2.33x | 差距在 `topk`，不在 apply |
| top_p renorm，nucleus > 1024 | AOT | 5.2–7.5x | 差距在 `torch.sort` 退路，hipify 幫不上 |
| top_p renorm，nucleus 3–62 | AOT | 1.65–1.75x | 絕對差 1.4ms |
| top_p renorm，nucleus ≤ 2 | **Triton** | 1.16–1.92x | AOT 的多輪縮減在此變慢 |

原本 §2.4 的推論是「hipify 幫得上忙的恰好不是瓶頸」。B200 的數據把結論推得更遠：**兩顆都不值得 hipify。** tree kernel 移植的上限是 0.03ms；renorm 最大的那筆差距是 `torch.sort` 退路，而它是 torch 層的問題，兩個平台共通，與 CUDA/ROCm 無關。真正值得做的優化是讓 pivot 搜尋不要 overflow（例如前綴不足時改用二分搜尋門檻值，而非完整排序），那在兩個平台上都會贏。

#### 剩下的缺口

前置條件 1、2、3 已結案。4（e2e 準確度）、5（accept length 對照）仍未做，兩者都需要跑真實 model 而非 microbenchmark。

### 6.6 MI355X 對照（gfx950，ROCm 7.2，torch 2.9.1，2026-08-06）

§6.5 留下的待查落差已結案，而且答案不是平台差異。

#### 5.0ms 對 20.9ms：是輸入分布，不是硬體

同一支 `bench_renorm_aot.py` 在 MI355X 上跑出 top_k 2.427 + top_p 17.358 = **19.79ms**，B200 是 20.86ms。**兩邊幾乎相同，MI355X 還略快 5%。** 所以 §2.3 的 5.0ms 從來就不是跟 B200 的 17.78ms 同一個量測。

差別在 `verify_renorm_triton.py` 第 100 行：`torch.softmax(torch.randn(rows, VOCAB) * 6.0, dim=-1)`。**§2.3 用的是 logit scale 6.0，而 `bench_renorm_aot.py` 用未縮放的 `softmax(randn)`。** 新增的 `bench_renorm_sweep.py` 把 overflow 比例印出來後，交叉點一目了然：

```
 scale    top1  nucleus    ovf  k_torch  k_triton  p_torch  p_triton
     1   0.000   112545   100%    4.155     2.472   19.093    17.466
     2   0.007    54988   100%    4.027     2.328   19.047    17.395
     4   0.206     1842    84%    4.059     2.387   18.708    17.080
     6   0.462       40     0%    4.012     2.337    4.254     2.620
     8   0.618        8     0%    4.028     2.353    4.306     2.678
    12   0.722        3     0%    4.022     2.364    4.293     2.687
    16   0.817        2     0%    4.034     2.356    4.306     2.678
    24   0.829        2     0%    4.371     2.698    4.662     3.043
    32   0.881        1     0%    4.502     2.809    4.793     3.153
```

scale 4 到 6 之間 overflow 從 84% 掉到 0%，`p_triton` 跟著從 17.08ms 掉到 2.62ms。scale 6 時 top_k 2.337 + top_p 2.620 = **4.96ms**，精確重現 §2.3 記的 5.027ms。§6.5 猜「§2.3 的量測其實沒有觸發 overflow」是對的，只是原因不是巧合而是腳本明寫的 `* 6.0`。

**教訓是量測沒有記錄輸入分布。** renorm 的成本幾乎完全由 overflow 與否決定，不記 logit scale 的 renorm 數字沒有意義。`bench_renorm_sweep.py` 因此固定輸出 `ovf` 欄。

#### MI355X Triton vs B200 AOT

把 §6.5 的 B200 sweep 與上表疊起來（`p_triton` 對 `aot`）：

| scale | top1 | MI355X triton | B200 AOT | 誰快 | 倍數 |
|---|---|---|---|---|---|
| 1 | 0.000 | 17.466 | 3.448 | AOT | 5.07x |
| 4 | 0.206 | 17.080 | 2.201 | AOT | 7.76x |
| 6 | 0.462 | 2.620 | 2.060 | AOT | 1.27x |
| 12 | 0.722 | 2.687 | 1.952 | AOT | 1.38x |
| 16 | 0.817 | 2.678 | 3.940 | **Triton** | 1.47x |
| 24 | 0.829 | 3.043 | 6.543 | **Triton** | 2.15x |
| 32 | 0.881 | 3.153 | 4.880 | **Triton** | 1.55x |

**在 `top1 ≥ 0.80` 的尖銳分布下，MI355X 跑 Triton 比 B200 跑原生 AOT 還快 1.47–2.15x。** 原因是 §6.5 已經指出的兩條曲線性質：Triton 與資料無關（overflow 消失後就是水平線），AOT 的三分搜尋則越尖越慢。交叉點落在 `nucleus ≤ 2` 附近。

同時 MI355X 的 Triton 比 B200 的 Triton 快：fast path 是 2.62–3.15ms 對 B200 恆定的 3.40ms，約 1.25x。overflow regime 兩邊則幾乎相同（17.47 對 17.77），因為那裡量的是 `torch.sort` 而不是 kernel。

必須註明這是**跨硬體比較**，混了硬體與軟體兩個變因，不能單獨解讀成「Triton 實作優於 AOT 實作」。

#### tree kernel

`--logit-scale 16 --iters 20`，ndt=8 / width 2 / depth 4：

| bs | MI355X triton | B200 triton | B200 AOT |
|---|---|---|---|
| 128 | 0.1908 | 0.2080 | 0.1148 |
| 256 | 0.2269 | 0.2228 | 0.1888 |

bs=256 時 MI355X 與 B200 的 Triton 差 1.8%，形同相同。B200 AOT 在 bs=256 快 1.20x，絕對差 0.038ms。

#### 對結論的影響

§6.5 的「兩顆都不值得 hipify」不變，而且理由更強：tree kernel 的移植上限仍是 0.03ms 等級，renorm 最大的差距仍在 `torch.sort` 退路那個兩平台共通的 torch 層問題。新增的一點是**在真實推論的尖銳分布下，ROCm 的 Triton renorm 已經不輸 B200 的原生 kernel**，所以 renorm 的 hipify 動機比 §6.5 判斷的更低。

反過來，§7 那個「兩次 topk 合併成一次」的價值被進一步確認：no-overflow regime 下 `k_triton` 2.34ms 加 `p_triton` 2.62ms，兩者都由 topk 主導，合併掉一次就是省掉接近一半。這是目前檯面上最值得做的 renorm 優化，而且兩個平台都會受益。

---

## 7. overflow 退路：用更寬的前綴取代排序

§6.6 指出 renorm 最大的一筆差距來自 `torch.sort` 退路，而它是兩平台共通的 torch 程式碼。這節記錄實際處理的過程。

### 7.1 二分搜尋在純 torch 裡是死路

直覺的修法是照 flashinfer 那樣二分搜尋門檻值，避開排序。實測後不成立（1536 列，151936 vocab）：

```
    full sort (current path)   13.408 ms
         one masked-sum pass    0.886 ms
      one compare+count pass    1.429 ms
```

一趟 `torch.where(probs > t, probs, 0).sum(-1)` 要 0.886ms，因為 torch 會**物化中間張量**（讀 933MB、寫 933MB、再讀 933MB），而不是像 kernel 那樣單趟讀完就地累加。收斂需要十幾輪，5 輪已經 4.4ms、17 輪比排序還慢。要達到 flashinfer 的成本必須自己寫 Triton 的 masked-sum kernel，那是另一個層級的改動。

### 7.2 topk 的成本平台

換個角度：與其避開選擇，不如讓選擇涵蓋更多情況。

```
                      topk k       ms
                        1024    1.841
                        2048    1.859
                        4096    1.921   <- 平台末端
                        8192    2.781
                       16384    3.709
                       32768    5.799
```

**1024 到 4096 幾乎免費（+4%），8192 才開始付錢。** 所以把 `_TOP_P_PREFIX` 放寬到 4096，等於用 4% 的選擇成本換掉 nucleus 落在 1024–4096 的所有列的 13.4ms 排序。

這也解釋了 `bench_escalation.py` 當初為什麼被判死路：它每輪把 K 乘 8（1024 → 8192 → 65536 → vocab），一路衝出平台，末端比排序還慘。單步升到平台末端是性質不同的操作。

### 7.3 結果

`_TOP_P_PREFIX = 1024 → 4096`，top_p 耗時（ms）：

| scale | nucleus | 前 | 後 | |
|---|---|---|---|---|
| 2 | 54988 | 17.395 | 17.669 | 仍走排序 |
| 3 | 13774 | — | 17.671 | 仍走排序 |
| **4** | **~1900** | **16.952** | **2.875** | **5.9x** |
| 6 | 46 | 2.690 | 2.876 | +0.19ms |
| 16 | 2 | 2.678 | 2.881 | +0.19ms |

常見情況（nucleus < 1024）付出 +0.19ms，即 top_p 慢 7%、整條 renorm 慢 3.8%；換到的是 nucleus 1024–4096 這段從 19.3ms 降到 5.2ms。

**這個取捨的理由是消掉懸崖，不是改善平均值。** 以期望值算，只要有 1.4% 的機率落在 1024–4096 就回本；更重要的是 17ms 的尖峰是 p99 延遲來源，用固定的 0.19ms 換掉划算。沒有走「只在 overflow 時對子集再做一次 topk(4096)」那條路，是因為它只多省 0.19ms 卻要引入升級路徑與額外的 host 同步分支，複雜度不成比例。

### 7.4 正確性

`test/registered/kernels/ops/speculative/` 全過（33 passed、3 skipped、30 subtests），與改動前相同。

另外量了一件測試抓不到的事：nucleus 落在 1024–4096 的列從「遞增排序累加」改走「遞減前綴累加」，fp32 下兩種累加順序在邊界可能給出不同的 pivot。512 列實測：

```
 scale   nucleus                  path change  tok diff   max |dv|  mass moved
   3.5      5355                 sort -> sort         1   1.14e-05    1.09e-05
   4.0      1973            sort -> fast path         7   2.67e-05    2.54e-05
   5.0       264       fast path -> fast path         0   0.00e+00    0.00e+00
```

512 列裡差 7 個 token、涉及質量 2.5e-05，與 §6.5 量到的「我們與 AOT 在 top_p < 1.0 最多差 2 個 token」同一量級，屬既有的邊界雜訊而非新引入的問題。

### 7.5 還沒解決的

nucleus > 4096 的列仍走 13.4ms 排序（上表 scale 2、3），對應 top1 < 0.07 的極平分布。真實推論應該不會出現，但**我們沒有真實工作負載的 nucleus 分布數據**，所以無法斷言。要徹底消掉排序得寫 Triton 的 masked-sum 做二分搜尋（§7.1），在有數據證明它會發生之前不值得做。

---

## 8. 尚未處理的技術債

**兩次 topk 可以合併成一次。** sampler 是先 `top_k_renorm` 再 `top_p_renorm`，各自做一次選擇。但 top_k 只是把部分項歸零再等比縮放，所以 top_p 的 pivot 可以從同一個排序前綴推導。合併後約再 1.6x，但需新增 `top_k_top_p_renorm_probs` 進入點並改呼叫端，動到公開 API。

**新增了 host 同步。** `top_ks.max().item()` 和 `overflow.any()` 各一次 device→host 同步，原本的純張量實作沒有。因為 topk 成本對 K 在 4096 內幾乎不變，改用固定 K 可完全消除同步且不變慢，但會改變 API 行為。

**`top_p_renorm_triton.py` 與 `renorm_triton.py` 功能重疊。** 前者來自 upstream #32890（Kimi K3），未註冊到 dispatch registry，且自身仍用 `torch.sort` 找 pivot。沒有合併是因為它被 Kimi K3 測試以 `rtol=2e-6, atol=1e-8` 綁著，容差偏緊，風險不值得冒。長期應該收斂成一份。

---

## 附錄：本地重現

```bash
# 測試（容器內以 root 執行，避免 aiter cache 權限問題）
docker exec --user root -e PYTHONPATH=/home/jacchang/PR/sglang/python \
  -e PYTHONDONTWRITEBYTECODE=1 jacchang_GLM5_0728 \
  bash -c "cd /home/jacchang/PR/sglang && \
           python3 -m pytest test/registered/kernels/ops/speculative/ -v"
```

跨機器需要的腳本已進 repo（`benchmark/rocm_tree_spec/`），兩邊 pull 都拿得到：

| 腳本 | 用途 |
|---|---|
| `bench_tree_sampling.py` | 真實樹拓樸下的 tree kernel，AOT 可用時自動對照 |
| `bench_renorm_aot.py` | renorm 三方對照（torch / Triton / AOT）與 support 等價性 |
| `bench_renorm_sweep.py` | renorm 成本對分布尖銳度的曲線，附 overflow 比例 |

以下是當初探索用的一次性腳本，只留在 `/home/jacchang/`，沒有進 repo：

| 腳本 | 用途 |
|---|---|
| `bench_renorm.py` | renorm 對 tree kernel 的相對成本 |
| `bench_floor.py` | 記憶體頻寬地板與 topk 成本曲線 |
| `bench_realistic.py` | 真實形狀分布下的 top_p 比較 |
| `bench_escalation.py` | 自適應 K（已證實為死路） |
| `bench_tree_warps.py` | `num_warps` / `BLOCK_V` 調參（結論：BLOCK_V 才有影響） |
| `verify_renorm_triton.py` | 融合 Triton 路徑的完整驗證 |
| `diag_renorm_aot_mismatch.py` | §6.5 用：拆解 support 差異的方向與質量 |
| `diag_renorm_attribution.py` | §6.5 用：把 renorm 成本歸因到 pivot 搜尋與 overflow 退路 |
| `diag_aot_scale_sweep.py` | §6.5 用：AOT 對分布尖銳度的 U 形曲線（確認非雜訊）|
| `diag_top_p_one_direction.py` | §6.5 用：`top_p=1.0` 逐列拆解誰丟掉哪些 token |

跑完記得 `sudo chown -R jacchang:jacchang .`，容器以 root 執行會在 repo 留下 root 檔案。
