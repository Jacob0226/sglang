# PR: Fix inter-token latency (ITL) over-counting under speculative decoding / MTP

## Problem

`benchmark_serving.py` records **one ITL sample per streamed SSE chunk**:

```python
# backend_request_func.py, async_request_openai_completions (decode phase)
output.itl.append(timestamp - most_recent_timestamp)
```

With speculative decoding / Multi-Token Prediction (MTP) a single streamed chunk
carries **several tokens** (≈ the acceptance length), not one. Counting one ITL
per chunk therefore over-states inter-token latency by roughly the acceptance
length, and the reported ITL no longer matches TPOT.

### Evidence (GLM-5-FP8, SGLang server with MTP/EAGLE, AMD MI355X, TP4)

Same server process, same `/v1/completions` endpoint, conc 16:

| client | TPOT median | ITL median |
|--------|-------------|------------|
| sglang `bench_serving` (sglang-oai) | 14.9 ms | **11.5 ms** (consistent) |
| this client (before fix) | 18.6 ms | **45.0 ms** (~3.9× inflated) |

conc 4: this client reports TPOT median 11.38 ms vs ITL median 29.21 ms — ITL
is ~2.6× the true per-token latency purely because of per-chunk accounting.

SGLang's `bench_serving` already handles this for the OpenAI-completions backend
(`use_retokenized_itl`): it stores each chunk's text and, in `calculate_metrics`,
re-tokenizes the chunk to spread its latency across its tokens.

## Fix

Mirror SGLang's approach (no extra request-time cost; re-tokenization happens
only in post-run metric calculation):

1. `RequestFuncOutput` gains a `text_chunks: List[str]` field.
2. `async_request_openai_completions` appends each decode chunk's text to
   `text_chunks` alongside the existing per-chunk ITL append.
3. `calculate_metrics` re-tokenizes each chunk and distributes that chunk's ITL
   evenly across its token count:

```python
chunk_texts = outputs[i].text_chunks
if chunk_texts and len(chunk_texts) == len(outputs[i].itl):
    for chunk_itl, chunk_text in zip(outputs[i].itl, chunk_texts):
        num_tokens = len(tokenizer(chunk_text, add_special_tokens=False).input_ids)
        if num_tokens <= 1:
            itls.append(chunk_itl)
        else:
            itls.extend([chunk_itl / num_tokens] * num_tokens)
else:
    itls += outputs[i].itl   # unchanged fallback
```

Behaviour is unchanged for non-speculative backends (1 token/chunk →
`num_tokens == 1` → identical to before).

## Files changed
- `backend_request_func.py`: add `text_chunks` field; append chunk text in
  `async_request_openai_completions`.
- `benchmark_serving.py`: re-distribute ITL per token in `calculate_metrics`.

## Test plan
- Re-run GLM-5-FP8 (MTP on) at conc 4 / 16: ITL median should converge to TPOT
  median (≈ within a few %), instead of ~2.6–3.9× higher.
- Run a non-speculative model: ITL unchanged (regression check).

## Note (separate issue)
This PR only addresses the **ITL metric correctness**. The separate ~21–29%
**throughput** deficit vs SGLang's client is tracked separately; it is isolated
to the client's HTTP-stream consumption layer (not processing/config) and is
under TCP-level investigation. See `CLIENT_BENCHMARK_GAP.md`.
