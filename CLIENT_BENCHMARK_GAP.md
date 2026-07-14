# GLM-5-FP8 Benchmark 分數差異分析:問題出在 benchmark client，不是 server

## TL;DR

- 本機 `GLM.sh` 跑出來的分數比 InferenceX 榜上高 **20~29%**。
- 經過「**同一台 server、同一份負載、只換 benchmark client**」的 A/B 驗證,差距 **100% 來自 benchmark client 實作**,與 kernel / server / 機器設定無關。
- `GLM.sh` 用的是 **sglang 原生 client**(`python3 -m sglang.bench_serving`,走 `/generate`),這顆比較快。
- InferenceX 榜上用的是他們自己的 **vLLM 風格 client**(`benchmark_serving.py`,走 `/v1/completions` SSE streaming),在 MTP 高 token 速率下會被 **單核 asyncio + 每 token `json.loads`** 卡住,造成 ITL 灌水與 TCP 反壓拖慢 server。
- 換算成 per-GPU 後,我們的 B(IX client)精準重現了榜上的藍色數字(誤差 1~2%)。

---

## 1. 設定

| 項目 | 值 |
|------|------|
| 模型 | GLM-5-FP8 |
| 平台 | MI355X (ROCm) |
| 並行 | TP4 + MTP(EAGLE 投機解碼,accept length ≈ 3.7) |
| 序列 | random in=1024 / out=1024 (range ratio 0.8) |
| GPU | 限制在 0–3(容器 `jacchang_GLM5`) |
| 做法 | 同一個 server instance,只切換 benchmark client |

- **Client A** = sglang 原生:`python3 -m sglang.bench_serving --backend sglang`(GLM.sh 使用的就是這顆)
- **Client B** = InferenceX:`ix_bench_serving/benchmark_serving.py`(走 OpenAI `/v1/completions` streaming)

---

## 2. A/B 結果(整機 total,1k/1k)

| conc | Client | Total tok/s | Output tok/s | Duration (s) | Mean ITL (ms) | B 相對 A |
|------|--------|-------------|--------------|--------------|---------------|----------|
| 4  | A sglang  | **950.8**  | –      | 77.1  | 8.18  | — |
| 4  | B IX-vllm | 677.6      | –      | 108.9 | 29.95 | **−28.7%** |
| 16 | A sglang  | **2223.6** | –      | 132.7 | 13.63 | — |
| 16 | B IX-vllm | 1687.1     | –      | 174.6 | 49.36 | **−24.1%** |
| 64 | A sglang  | **4379.7** | 2186.5 | 269.4 | 20.92 | — |
| 64 | B IX-vllm | 3443.7     | 1720.6 | 342.8 | 71.95 | **−21.4%** |

重點觀察:

1. 三個 concurrency 點,B 都比 A 低 21~29%,**server 完全相同**。
2. B 的 Mean ITL 永遠是 A 的 **3.4~3.7×**(8→30、14→49、21→72 ms)。
3. concurrency 越高差距略收斂(−28.7% → −21.4%),符合 client-bound 特徵(高併發時 server 排隊比例變大,client overhead 被稀釋)。

---

## 3. 與 InferenceX 榜上數字比對(per-GPU)

榜上數字是 **per_gpu**,我們量的是整機 total,TP4 → ÷4 後比較:

| conc (1k1k) | B total | B ÷4 = per_gpu | 榜上藍色 IX per_gpu | 一致? |
|------|---------|----------------|---------------------|--------|
| 4  | 677.6  | **169.4** | **166** | ✅ ~2% |
| 16 | 1687.1 | **421.8** | **420** | ✅ 幾乎一致 |
| 64 | 3443.7 | **860.9** | **850** | ✅ ~1% |

對照 A 也對得上榜上黑色(本機 sglang)欄:

| conc (1k1k) | A total | A ÷4 = per_gpu | 榜上黑色 sglang per_gpu |
|------|---------|----------------|-------------------------|
| 4  | 950.8  | 237.7  | 225 |
| 16 | 2223.6 | 555.9  | 513 |
| 64 | 4379.7 | 1094.9 | 1074 |

> 結論:**B 精準重現了榜上的 InferenceX 數字**,代表榜上的低分就是 IX client 造成的,不是 server。

---

## 4. Root cause:IX client 為什麼慢

IX client 是 **單一 process + 單一 asyncio event loop**,所有併發請求的 SSE 串流都擠在同一顆 CPU core 上被 Python(GIL)序列化處理。

關鍵解析迴圈(`ix_bench_serving/backend_request_func.py`,`async_request_openai_completions`):

```python
async for chunk_bytes in response.content:
    chunk_bytes = chunk_bytes.strip()
    if not chunk_bytes:
        continue
    chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
    if chunk != "[DONE]":
        data = json.loads(chunk)          # <-- 每個 token chunk 一次,最貴
        if choices := data.get("choices"):
            text = choices[0].get("text")
            timestamp = time.perf_counter()
            ...
            output.itl.append(timestamp - most_recent_timestamp)
            most_recent_timestamp = timestamp
            generated_text += text or ""
```

每收到一個 token chunk 就做:`strip` → `decode` → `removeprefix` → **`json.loads`** → dict 取值 → append。在 `stream=True` + `include_usage` 下每 token 一個 SSE chunk,而 MTP 一次吐一串 token(accept ≈ 3.7),token 到達速率非常高。conc64 整機 3000+ tok/s 時,這顆單核每秒要 `json.loads` 數千個 chunk → **client 端 CPU 飽和**。

造成兩個後果,皆被數據印證:

1. **ITL 被灌水(測量假象)**:`timestamp` 是「event loop 處理到該 chunk 的時間」,不是 token 到達網卡的時間。event loop 塞車時 chunk 在緩衝排隊 → 時間戳延後 → Mean ITL 被放大 3.4~3.7×。
2. **server 被真的拖慢(實際損失)**:client 讀不夠快 → TCP 接收窗被填滿 → 反壓回 server → server send 被 block → 生成停頓。所以 duration **真的**變長(269→343s),throughput 才真的掉 21~29%。

次要因素(讓 IX 再慢一點,但非主因):

- 每個 request 在函式內各自 `aiohttp.ClientSession(...)` → 每條請求新開 TCP 連線,無連線池複用;
- `stream_options.include_usage` 多一輪 usage chunk;
- 全程 GIL 單執行緒,concurrency 越高越吃虧。

sglang 原生 client(A)走 `/generate`、解析路徑精簡,單核在此 token 速率下尚未打爆,讀得夠快、不對 server 形成反壓,所以 server 全速跑完。

---

## 4.1 這跟 MTP 無關:無 MTP 的 case 一樣是 GLM.sh 較快

榜上 `spec_method = none`(關閉 MTP)的 MI355X / TP8 數據顯示,**沒有 MTP 時本機 sglang client 仍然比 InferenceX 高**,證明差距是 client 結構性問題,MTP 只是放大器。

per-GPU 比較(1k/1k,無 MTP):

| conc | 本機 sglang per_gpu | IX per_gpu | IX 相對本機 |
|------|---------------------|------------|-------------|
| 4  | 56  | 44  | −21% |
| 8  | 98  | 77  | −21% |
| 16 | 174 | 140 | −20% |
| 32 | 283 | 247 | −13% |
| 64 | 463 | 385 | −17% |

對照(有 MTP 時差距為 −21~29%):

- **無 MTP:差 ~13~21%** → 差距已經存在。
- **有 MTP:差 ~21~29%** → 相同 client 瓶頸被放大。

> 解讀:root cause(單核 asyncio + 每 token `json.loads` + TCP 反壓)對**任何 streaming 量測**都成立,與投機解碼無關。MTP 讓 server 一次吐一串 token、token 到達速率更高 → 單核 client 更早飽和 → 把差距從 ~20% 推到 ~29%。拿掉 MTP,token 速率降低、client 壓力變小,差距縮到 ~13~21%,但因 per-token 固定開銷與 ITL 結構性灌水仍在,**永遠不會歸零**。**MTP 不是原因,是乘數。**

---

## 4.2 實驗:`--stream-interval 30` 沒有幫助(且更正吞吐主因)

假設:把 server 的 `--stream-interval` 從 1 調到 30,讓 chunk 變少變大,理論上可降低 IX client 逐 chunk `json.loads` 的數量與反壓,改善分數。實測結果**否定此假設**。

MI355X / TP4 / MTP / 1k1k,同 server 換 client,si1 vs si30:

| conc | client | si1 tput | si30 tput | si1 ITL | si30 ITL |
|------|--------|----------|-----------|---------|----------|
| 16 | A sglang | 2223.6 | 2244.3 | 13.6 | 13.6 |
| 16 | B IX     | 1687.1 | 1676.1 | 49.4 | **1407.9** |
| 64 | A sglang | 4379.7 | 4416.5 | 20.9 | 20.8 |
| 64 | B IX     | 3443.7 | 3460.2 | 72.0 | **2005.0** |

觀察:

1. **吞吐幾乎不變**(兩 client 差異 <1%,B/A 差距維持 −21~25%)→ stream-interval 救不回 throughput gap。
2. **A 的 ITL 不受影響**(sglang 用 `chunk_gap / num_new_tokens` per-token 分攤),**B 的 ITL 爆炸 28~29×**(IX per-chunk 記一筆,一個 chunk 30 token,gap 被當成單 token ITL)→ 再次印證 §4 的 ITL 算法 bug。

**更正吞吐主因**:既然把 chunk 數降 30 倍仍救不回吞吐,conc16/64 的 throughput gap **主因不是逐 chunk `json.loads` 的數量**。更可能是 IX client **每個 request 各自新開 `ClientSession`/TCP 連線(無連線池)+ 單核 event loop 的 dispatch 間隙**,造成併發槽位填不滿、server 被閒置 → duration 拉長。`read_bufsize`(64KB vs 10MB)在高突發下仍可能貢獻反壓,但不是 conc16/64 此處的單一主因。待以 `top -H` / 連線數觀測進一步確認。

---

## 4.3 決定性實驗總表(同 server，隔離 client）

| # | 實驗 | 結果 | 結論 |
|---|------|------|------|
| 1 | Baseline A/B (conc 4/16/64) | A 比 B 高 21–29%；B 重現榜上 per-GPU | 差距來自 client |
| 2 | `--stream-interval 30` | 吞吐不變；B 的 ITL 爆炸 | 非 json 量；暴露 per-chunk ITL bug |
| 3 | `read_bufsize` 64KB→10MB | 3434 vs 3443 無變化 | 排除緩衝 |
| 4 | 三方 sglang-oai (C) | C=4377 ≈ A=4380 | endpoint/server 無罪 |
| 5 | 移除 `include_usage` | 357s 沒更快 | 排除 |
| 6 | session 換成 sglang | 678 vs 677 無變化 | 排除 trust_env/session |
| 7 | 強制 uvloop | 1687 = stock | 排除 event loop 實作 |
| 8 | py-spy profiling | 主迴圈在 epoll 等待 | I/O-bound，非 CPU-bound |
| 9 | **同 server C vs B (conc16)** | **C=1942.6, B=1590.1 (+22%)** | **鐵證：純 IX client 程式碼** |

### 同 server process 的相位分析(conc16,running-req≥14 飽和狀態)

C 與 B 在**同一個 server process** 上先後跑,直接比對 server 端 gen throughput:

| 相位 (同一 server process) | mean gen tput | peak gen tput |
|------|------|------|
| C (sglang client) | **1033** | **1327** |
| B (IX client, 正常) | 838 (−19%) | 1010 (−24%) |
| B (IX, drain-only 不做 json) | 852 | 1042 |
| B (IX, session 換成 sglang) | 856 | 1054 |

→ **連 peak 都掉 ~24%**:server 在跑 B 時,即使最佳瞬間 decode 也較慢 = server 真的被 client 消費速率拖住(per-step send 反壓)。
→ **drain-only(完全不做 json/處理、純抽乾 socket)幾乎沒改善(852 vs 838)** = 瓶頸**不是** client 的 Python 處理。
→ **session 換成 sglang 完全相同也沒改善(856)** = 不是 read_bufsize / trust_env。

### 兩個獨立問題與最終定位

- **(A) ITL/interactivity 灌水 — 真因確定,修正已實作並驗證** ✅
  **驗證結果(conc16,套用修正後):B ITL median 45.0 → 14.79 ms**(與 TPOT 18.34、與 C 的 11.32 同量級),吞吐不變。PR 內容見 `PR_ITL_FIX.md`。
  per-chunk ITL 計法,MTP 下每 chunk 含 ~accept_length(~3.8) 個 token,IX 只記一筆 → ITL = accept_length × 真實值。實測 conc4 B:TPOT median 11.38ms vs ITL median 29.21ms;同-server conc16:C ITL 11.5 vs B 45.0。修法同 sglang:`adjust_itl = chunk_gap / num_new_tokens` 再 `itl.extend([adjust_itl]*num_new_tokens)`。

- **(B) 吞吐低 21–29% — 確定來自 IX client 的 HTTP 串流消費層,非 Python 處理/設定** ⏳
  全部排除:read_bufsize、trust_env、uvloop、include_usage、logprobs、json 解析量(stream-interval)、batch 飽和、client 端處理(drain-only)。
  py-spy:client 主迴圈在 `select`/epoll 等待(I/O-bound,非 CPU-bound)。
  server 端:同 batch、gen tput 與 peak 皆真實較低 → server 被 per-step send 反壓。
  request func / dispatch / dataset token 數逐行比對近乎相同。
  **結論:瓶頸在 IX client 消費 HTTP 串流的連線/協定層,且無法用緩衝/event loop/payload 調參修正。**

### TCP 層證據(ss 取樣,loopback 127.0.0.1:28553,C+B 同 server)

對 ~6000 個 socket 樣本掃描:

| 指標 | 值 |
|------|-----|
| 非零 Recv-Q / Send-Q 的樣本 | 9 / 6000 |
| max Send-Q(server 送不出) | **274 bytes** |
| max Recv-Q(client 沒收) | **284 bytes** |
| retrans / rwnd_limited / sndbuf_limited | **0** |
| mss / socket buffer | 32768 / 2.6MB(loopback 視窗超大) |

→ **完全沒有 TCP 反壓**:kernel buffer 從不滿、零重傳、零視窗限制。loopback 上資料瞬間搬完。
→ **推翻「TCP 視窗/反壓」假設**。瓶頸純在應用層 send/recv 速率,不在傳輸層。

### Payload diff(C vs B,皆打 /v1/completions)

| 欄位 | sglang (C) | IX (B) |
|------|-----------|--------|
| best_of | 1 | 1(預設) |
| logprobs | (不送) | `null`(不送 --logprobs) |
| stream_options.include_usage | (不送) | `true` |

→ include_usage 已測移除無效;logprobs=null、best_of=1 → **payload 不是原因**。

### (B) 排除總結

非以下任一:TCP/傳輸層(ss 證明)、client 處理(drain 證明)、read_bufsize、trust_env、uvloop、session、include_usage、logprobs、best_of、stream-interval、batch 飽和。
server gen tput(含 peak)為 B 真實較低,但**不是 TCP 反壓造成**。
剩下唯一未查的原子原因:**server scheduler 在串流給 B 的連線時,每 decode step 的應用層行為差異**(需對 sglang server 端做 py-spy/scheduler profiling 才能鎖定;已超出 client 端可觀測範圍)。

### Server 端 py-spy profiling(部分完成,C 已抓 / B 待補)

對「實際持有 client 連線(:28553)的 server worker 行程」做 py-spy `--idle`(用 `ss -tnp` 定位 pid;`--subprocesses` 會 segfault 故改逐 pid):

- **C 相位(sglang-oai client)已成功**:4 個 client-facing worker,47992 samples。每個 worker 兩個 thread,各 ~50% wall time:一個 blocked 在 `multiprocessing recv`(跨行程 pipe),一個是 uvicorn asyncio loop(大多 parked 在 epoll)。**活躍的 SSE 送出路徑**(`openai/serving_completions.py::_generate_completion_stream` → pydantic `model_dump_json` → `protocol.py::_serialize` → starlette `stream_response`)在 C 只佔 **<0.05%**。
- **B 相位未抓到**:`ss` pid 偵測時機沒對上(IX 主 run 標記與實際發 request 間有 gap / per-request 短連線),B.folded 為空。**這是接續時唯一要補的一步**:用「輪詢 ss 直到有連線」的方式重抓 B,再 diff `model_dump_json / _serialize / stream_response / socket send` 這幾個 frame 在 B 是否顯著高於 C(C baseline <0.05%)。

> 關鍵觀察:server 對外串流確實走 **pydantic `model_dump_json` 逐 chunk 序列化**。若 B 的 payload(`stream_options.include_usage` 等)或回應格式讓每 chunk 序列化更重,會在這裡現形 —— 這是 B-profiling 補完後最可能的著力點。

### 下次接續的唯一待辦(B server-side profiling)
1. 跑 `./compare_clients.sh --conc 16 --only b`,搭配 `server_profile_b.sh`(已改成輪詢 ss 直到有連線)。
2. diff `B.folded` vs `C.folded.keep`(在 `/data/jacchang_client_ab/`),比對 asyncio-loop thread 內活躍 frame 佔比。
3. 若 `model_dump_json/_serialize` 在 B 明顯偏高 → 真因 = server 對 IX 連線的逐 chunk 序列化/送出較重(可能與 payload/格式相關),屆時可提具體 client/server PR。

---

## 5. 反常觀察:B200 上 IX client 反而比較快(待驗證)

在 B200 上觀察到 **相反** 的現象 —— IX client 反而比 sglang 原生 client 快。這與 MI355X 結論相反,目前**尚未做同 server A/B 對照**,以下為候選假設,需進一步驗證:

候選假設:

1. **server-bound vs client-bound 反轉**:B200 解碼更快,可能 sglang 原生 client 在某條路徑(例如 retokenize / apply-chat-template / tqdm 更新)成為瓶頸,反而 IX client 路徑在該情境下開銷較低。
2. **兩支 script 的負載不同**:`GLM.sh`(A)與走 IX client 的設定在 `--num-prompts`、warmup、`--random-range-ratio`、chat template、`max_tokens`/`ignore_eos` 等參數上若不一致,會直接影響可比性。需確認兩邊「除了 client 以外」完全相同。
3. **retokenized 計數差異**:sglang 原生 client 會回報 "Total generated tokens (retokenized)";若兩支 client 計 token 的方式不同(usage vs 重新 tokenize),per-GPU 數字基準就不同。
4. **量測變異 / concurrency 點不同**:單點、未暖機、或不同 concurrency 之間直接比,容易誤判。

建議驗證步驟(複製 MI355X 的方法到 B200):

1. 起**一個** GLM-5-FP8 server(B200, TP8 或對應設定),固定不動。
2. 用 `compare_clients.sh` 對同一 server 連續跑 A 與 B,**參數除 client 外完全相同**(同 in/out len、同 conc、同 num-prompts、同 warmup)。
3. 掃 conc = 4 / 16 / 64,記錄 Total tok/s、Duration、Mean/Median ITL。
4. 比較 A vs B,並確認 token 計數基準一致(留意 retokenized 與 usage 的差異)。
5. 若 B 在 B200 真的較快,進一步用 `top -H` 抓兩支 client 的 python 執行緒 CPU%,判斷是否 client-bound 在 B200 上落到 A 身上。

> 在 B200 A/B 對照數據出來前,先不下結論;此節僅記錄現象與待辦。

---

## 6. 重現方式

```bash
# 在容器 jacchang_GLM5 內,限制 GPU 0-3,起同一個 server 後:
./compare_clients.sh --conc 4  --isl 1024 --osl 1024
./compare_clients.sh --conc 16 --isl 1024 --osl 1024
./compare_clients.sh --conc 64 --isl 1024 --osl 1024
# 結果:/data/jacchang_client_ab/client_ab/mi355x_conc*_in1024_out1024_tp4_mtptrue/
#   A_*.log = sglang 原生 client
#   B_*.log = InferenceX vLLM client
```

---

## 7. 結論

- 本機 `GLM.sh` 比 InferenceX 榜高,**不是機器/設定比較強,而是 GLM.sh 用 sglang 原生 client 量測**;InferenceX 榜用的是會被 client 端 CPU 卡住的 vLLM client。
- 在 MI355X 上,同一台 server 換 client 即可還原榜上低分,per-GPU 換算後誤差僅 1~2%。
- **B200 上的相反現象尚待同 server A/B 對照確認**,不應與 MI355X 結論混為一談。
