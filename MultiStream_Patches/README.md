# MI355X Dual-Stream Optimization Patches

Base commit: `5aa72a9b0` on branch `jacob/hip-dual-stream-glm5`

## Apply Order: C → A → B → D

```bash
cd ~/PR/sglang

# Patch C: Per-layer alt_stream (HIP only)
git apply ~/SGLang-benchmarks/patches/C_per_layer_alt_stream.patch

# Patch A: w_kc absorb + RoPE overlap with indexer
git apply ~/SGLang-benchmarks/patches/A_wkc_rope_overlap.patch

# Patch B: Gate before MoE fork (likely neutral — revert if negative)
git apply ~/SGLang-benchmarks/patches/B_gate_before_fork.patch

# Patch D: Delay attention join into forward_absorb_core (requires A)
git apply ~/SGLang-benchmarks/patches/D_delay_join.patch
```

## Dependencies

```
C ──────────────────────── (independent, deepseek_v2.py)
A ──────────────────────── (independent, forward_mla.py)
B ──────────────────────── (independent, deepseek_v2.py, different function than C)
D ── requires A applied ── (incremental on forward_mla.py)
```

B can be skipped or reverted without affecting D:
```bash
# C → A → D (skip B)
git apply C_per_layer_alt_stream.patch
git apply A_wkc_rope_overlap.patch
git apply D_delay_join.patch
```

## Benchmark Between Patches

```bash
cd ~/SGLang-benchmarks
./GLM.sh --prof --dual-stream-rocm
```

## What Each Patch Does

| Patch | File | Change | Expected Impact |
|-------|------|--------|-----------------|
| **C** | `deepseek_v2.py` | 1 shared alt_stream → per-layer streams on HIP | Inter-layer pipelining (B200 gets this free via CUDA Graph DAG opt) |
| **A** | `forward_mla.py` | w_kc absorb (6.2μs) + RoPE (5.1μs) run on alt_stream, overlapping with NSA indexer (141μs) | −11.3μs/layer × 68 = −0.77ms/token |
| **B** | `deepseek_v2.py` | MoE gate (8.7μs) runs before fork instead of on alt_stream | ~0μs (gate moves from overlapped to serial) |
| **D** | `forward_mla.py` | Cat/fused ops also run on alt_stream; join delayed to just before attn_mqa | −7.3μs/layer × 68 = −0.50ms/token (on top of A) |
