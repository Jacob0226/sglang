#!/usr/bin/env python3
"""
GLM-5.1-FP8 single decode-layer micro-benchmark on MI355X (gfx95) — v5.

Uses real aiter / CK / TileLang / sgl-kernel / Triton kernels matching
the actual SGLang decode trace with:
  --nsa-decode-backend tilelang --disable-shared-experts-fusion

Key kernels (matching real SGLang trace):
  - MLA decode: TileLang sparse_mla_fwd_decode_partial + combine (main_kernel ×2)
  - NSA RoPE: rope_cached_positions_2c_fwd_inplace (kn_entry_2c_sbhd)
  - weights_proj: F.linear → hipBLAS Cijk (FP32 weights)
  - Router gate: wv_splitk_small_fp16_bf16
  - kv_a_norm on alt stream (matching real stream placement)

Allreduce and torch.compile fusions are not benchmarked (single-card).

Usage:
    python glm5_decode_layer.py
    python glm5_decode_layer.py --profile --prof-tag v5
    python glm5_decode_layer.py --batch-size 4 --num-layers 3 --profile
"""

import os
import sys
import types

os.environ.setdefault("SGLANG_USE_AITER", "1")

import argparse
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

SGLANG_PATH = "/home/jacchang/PR/sglang/python"
if SGLANG_PATH not in sys.path:
    sys.path.insert(0, SGLANG_PATH)

BF16 = torch.bfloat16
FP32 = torch.float32

# ── Pre-mock sglang subpackages to avoid heavy __init__ chains ───────────
_sglang_pkgs = [
    "sglang", "sglang.srt", "sglang.srt.utils",
    "sglang.srt.layers", "sglang.srt.layers.quantization",
    "sglang.srt.layers.attention", "sglang.srt.layers.attention.nsa",
    "sglang.srt.layers.attention.triton_ops",
]
for _pkg in _sglang_pkgs:
    if _pkg not in sys.modules:
        _m = types.ModuleType(_pkg)
        _m.__path__ = [os.path.join(SGLANG_PATH, _pkg.replace(".", "/"))]
        sys.modules[_pkg] = _m

_is_hip = torch.version.hip is not None
sys.modules["sglang.srt.utils"].is_hip = lambda: _is_hip
sys.modules["sglang.srt.utils"].is_gfx95_supported = lambda: (
    _is_hip and "gfx95" in torch.cuda.get_device_properties(0).gcnArchName
)

_fp8k = types.ModuleType("sglang.srt.layers.quantization.fp8_kernel")
_fp8k.is_fp8_fnuz = lambda: (
    _is_hip and "gfx94" in torch.cuda.get_device_properties(0).gcnArchName
)
sys.modules["sglang.srt.layers.quantization.fp8_kernel"] = _fp8k

# ── aiter kernel imports ─────────────────────────────────────────────────
import aiter
from aiter import (
    ActivationType,
    QuantType,
    get_hip_quant,
    layernorm2d_fwd as aiter_layernorm2d,
    rmsnorm2d_fwd as aiter_rms_norm,
    rmsnorm2d_fwd_with_add as aiter_fused_add_rms_norm,
)
from aiter.fused_moe import fused_moe as aiter_fused_moe
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as batched_fp8_gemm,
)
from aiter.ops.triton.fused_fp8_quant import fused_rms_fp8_group_quant
from aiter import gemm_a8w8_blockscale as ck_gemm_a8w8_blockscale
from aiter.ops.triton.fused_fp8_quant import fused_flatten_fp8_group_quant
from aiter.ops.triton.fused_kv_cache import fused_qk_rope_cat_and_cache_mla
from sgl_kernel import silu_and_mul as sgl_silu_and_mul

per1x128_quant = get_hip_quant(aiter.QuantType.per_1x128)
FP8 = aiter.dtypes.fp8

from aiter import biased_grouped_topk as aiter_biased_grouped_topk
from fast_hadamard_transform import hadamard_transform
from sgl_kernel import fast_topk_transform_fused

# SGLang TileLang kernel (needs mocked sglang.srt.layers.quantization)
from sglang.srt.layers.attention.nsa.tilelang_kernel import act_quant

# Skinny BF16 GEMM: wv_splitk for router gate, F.linear for weights_proj
from aiter import wv_splitk_small_fp16_bf16 as aiter_wv_splitk
# TileLang sparse MLA decode attention kernel (matches --nsa-decode-backend tilelang)
from sglang.srt.layers.attention.nsa.tilelang_kernel import tilelang_sparse_fwd

# NSA indexer k cache store
from sglang.srt.layers.attention.nsa.index_buf_accessor import _set_k_and_s_triton

# ── Optional kernels for closer trace matching ────────────────────────────
HAS_PAGED_MQA = False
try:
    from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
    HAS_PAGED_MQA = True
except Exception:
    pass

HAS_AITER_ROPE = False
try:
    from aiter.ops.rope import rope_cached_positions_2c_fwd_inplace
    HAS_AITER_ROPE = True
except Exception:
    pass


# ── Config ────────────────────────────────────────────────────────────────
@dataclass
class GLM5Config:
    hidden_size: int = 6144
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    qk_head_dim: int = 256  # nope + rope
    v_head_dim: int = 256
    num_heads: int = 64
    tp_size: int = 8
    num_local_heads: int = 8  # 64 / 8
    fused_qkv_out: int = 2624  # q_lora + kv_lora + rope
    kv_cache_dim: int = 576  # kv_lora + rope

    # NSA indexer
    index_n_heads: int = 32
    index_head_dim: int = 128

    # MoE
    n_routed_experts: int = 256
    n_local_experts: int = 32  # 256 / 8
    num_experts_per_tok: int = 8
    n_shared_experts: int = 1
    moe_intermediate_size: int = 2048
    routed_scaling_factor: float = 2.5
    n_group: int = 1
    topk_group: int = 1

    rms_norm_eps: float = 1e-5

    # decode context
    batch_size: int = 1
    seq_len: int = 4096
    num_kv_splits: int = 32


# ── FP8 helpers ───────────────────────────────────────────────────────────
def _rand_fp8(shape, device="cuda"):
    return (torch.randn(shape, dtype=BF16, device=device) * 0.01).to(FP8)


def _blockscale(rows, cols, device="cuda"):
    n_r = (rows + 127) // 128
    n_c = (cols + 127) // 128
    return torch.ones(n_r, n_c, dtype=FP32, device=device)


def fp8_gemm(x, w, w_scale):
    """per-1×128 activation quant → CK gemm_a8w8_blockscale."""
    shape = x.shape
    x2d = x.reshape(-1, shape[-1])
    xq, xs = per1x128_quant(x2d, quant_dtype=FP8)
    o = ck_gemm_a8w8_blockscale(xq, w, xs, w_scale, dtype=BF16)
    return o.view(*shape[:-1], o.shape[-1])


# ── Weight containers ─────────────────────────────────────────────────────
class AttentionWeights:
    def __init__(self, cfg: GLM5Config, device="cuda"):
        H = cfg.hidden_size
        self.fused_qkv_w = _rand_fp8((cfg.fused_qkv_out, H), device)
        self.fused_qkv_s = _blockscale(cfg.fused_qkv_out, H, device)

        self.q_a_ln_w = torch.ones(cfg.q_lora_rank, dtype=FP32, device=device)
        self.kv_a_ln_w = torch.ones(cfg.kv_lora_rank, dtype=FP32, device=device)

        q_b_out = cfg.num_local_heads * cfg.qk_head_dim
        self.q_b_w = _rand_fp8((q_b_out, cfg.q_lora_rank), device)
        self.q_b_s = _blockscale(q_b_out, cfg.q_lora_rank, device)

        # MLA absorbed batched-GEMM weights (FP8, per-head scale)
        self.w_kc = _rand_fp8(
            (cfg.num_local_heads, cfg.qk_nope_head_dim, cfg.kv_lora_rank), device
        )
        self.w_vc = _rand_fp8(
            (cfg.num_local_heads, cfg.kv_lora_rank, cfg.v_head_dim), device
        )
        self.w_scale = torch.ones(cfg.num_local_heads, dtype=FP32, device=device)

        o_in = cfg.num_local_heads * cfg.v_head_dim
        self.o_proj_w = _rand_fp8((H, o_in), device)
        self.o_proj_s = _blockscale(H, o_in, device)

        # RoPE cos/sin tables for fused_qk_rope_cat_and_cache_mla
        max_pos = cfg.seq_len + 128
        self.cos_cache = torch.randn(max_pos, cfg.qk_rope_head_dim // 2, dtype=BF16, device=device)
        self.sin_cache = torch.randn(max_pos, cfg.qk_rope_head_dim // 2, dtype=BF16, device=device)
        self.k_scale = torch.tensor(1.0, dtype=FP32, device=device)


class IndexerWeights:
    def __init__(self, cfg: GLM5Config, device="cuda"):
        idx_out = cfg.index_n_heads * cfg.index_head_dim
        self.wq_b_w = _rand_fp8((idx_out, cfg.q_lora_rank), device)
        self.wq_b_s = _blockscale(idx_out, cfg.q_lora_rank, device)

        self.wk_w = _rand_fp8((cfg.index_head_dim, cfg.hidden_size), device)
        self.wk_s = _blockscale(cfg.index_head_dim, cfg.hidden_size, device)

        self.k_norm_w = torch.ones(cfg.index_head_dim, dtype=BF16, device=device)
        self.k_norm_b = torch.zeros(cfg.index_head_dim, dtype=BF16, device=device)

        # FP32 weights on HIP (matches real SGLang ReplicatedLinear params_dtype)
        self.weights_proj_w = (
            torch.randn(cfg.index_n_heads, cfg.hidden_size, dtype=FP32, device=device)
            * 0.01
        )

        INDEX_TOPK = 2048
        M = cfg.batch_size
        num_pages = max(cfg.seq_len, INDEX_TOPK)
        self.topk_logits = torch.randn(M, num_pages, dtype=FP32, device=device)
        self.topk_seq_lens = torch.full(
            (M,), num_pages, dtype=torch.int32, device=device
        )
        self.topk_page_table = (
            torch.arange(num_pages, dtype=torch.int32, device=device)
            .unsqueeze(0)
            .expand(M, -1)
            .contiguous()
        )
        self.topk_cu_seqlens_q = torch.arange(
            M + 1, dtype=torch.int32, device=device
        )
        self.topk_k = INDEX_TOPK

        # Paged FP8 KV cache for deepgemm_fp8_paged_mqa_logits
        # Layout: [num_pages, KVBlockSize=1, nkv=1, index_dim] where
        # index_dim = head_dim + 4 (FP8 k + packed FP32 scale)
        index_dim = cfg.index_head_dim + 4
        raw = torch.zeros(num_pages, 1, 1, index_dim, dtype=torch.uint8, device=device)
        k_fp8_data = (
            torch.randn(num_pages, 1, 1, cfg.index_head_dim, dtype=BF16, device=device) * 0.01
        ).to(FP8)
        raw[:, :, :, : cfg.index_head_dim] = k_fp8_data.view(torch.uint8)
        scale_bytes = torch.tensor([1.0], dtype=FP32).view(torch.uint8)
        raw[:, :, :, cfg.index_head_dim :] = scale_bytes.to(device)
        self.paged_k_cache = raw.view(FP8)

        self.paged_block_tables = (
            torch.arange(num_pages, dtype=torch.int32, device=device)
            .unsqueeze(0)
            .expand(M, -1)
            .contiguous()
        )
        self.paged_context_lens = torch.full(
            (M,), num_pages, dtype=torch.int32, device=device
        )
        self.paged_max_seq_len = num_pages

        # _set_k_and_s buffer: [num_pages, page_size=1, (128+4)] uint8
        buf_numel_per_page = 1 * (cfg.index_head_dim + 4)
        self.nsa_k_buf = torch.zeros(
            num_pages, buf_numel_per_page, dtype=torch.uint8, device=device
        )
        self.nsa_k_loc = torch.arange(M, dtype=torch.long, device=device)

        # RoPE cos/sin caches — matches DeepseekScalingRotaryEmbedding format
        # [max_pos, 1, 1, rope_dim//2].  The 2c kernel expects 4D cos/sin.
        self.rope_head_dim = cfg.qk_rope_head_dim  # 64
        max_pos = cfg.seq_len + 128
        rope_cos_dim = self.rope_head_dim // 2      # 32
        self.rope_cos = torch.ones(max_pos, 1, 1, rope_cos_dim, dtype=BF16, device=device)
        self.rope_sin = torch.zeros(max_pos, 1, 1, rope_cos_dim, dtype=BF16, device=device)
        self.rope_positions = torch.randint(
            0, cfg.seq_len, (M,), dtype=torch.long, device=device
        )


class MoEWeights:
    def __init__(self, cfg: GLM5Config, device="cuda"):
        H = cfg.hidden_size
        inter = cfg.moe_intermediate_size

        self.gate_w = (
            torch.randn(cfg.n_routed_experts, H, dtype=BF16, device=device) * 0.01
        )
        self.correction_bias = torch.zeros(
            cfg.n_routed_experts, dtype=BF16, device=device
        )
        # Pre-allocated router output buffer (wv_splitk outputs BF16)
        self.router_out = torch.zeros(
            cfg.batch_size, cfg.n_routed_experts, dtype=BF16, device=device
        )

        # shared expert (TP-sharded)
        s_inter = (inter * cfg.n_shared_experts) // cfg.tp_size
        self.shared_gate_up_w = _rand_fp8((s_inter * 2, H), device)
        self.shared_gate_up_s = _blockscale(s_inter * 2, H, device)
        self.shared_down_w = _rand_fp8((H, s_inter), device)
        self.shared_down_s = _blockscale(H, s_inter, device)

        # routed experts  [n_local, ...]
        E = cfg.n_local_experts
        w13 = _rand_fp8((inter * 2, H), device)
        w2 = _rand_fp8((H, inter), device)
        self.w13_weight = shuffle_weight(
            w13.unsqueeze(0).expand(E, -1, -1).contiguous(), (16, 16)
        )
        self.w2_weight = shuffle_weight(
            w2.unsqueeze(0).expand(E, -1, -1).contiguous(), (16, 16)
        )
        self.w13_scale = _blockscale(inter * 2, H, device).unsqueeze(0).expand(E, -1, -1).contiguous()
        self.w2_scale = _blockscale(H, inter, device).unsqueeze(0).expand(E, -1, -1).contiguous()


# ── KV cache (decode attention) ──────────────────────────────────────────
class KVCache:
    def __init__(self, cfg: GLM5Config, batch_size: int, device="cuda"):
        S = cfg.seq_len
        total = batch_size * S

        self.kv_buffer = (
            torch.randn(total, 1, cfg.kv_cache_dim, dtype=BF16, device=device) * 0.01
        )

        self.slot_mapping = torch.arange(batch_size, dtype=torch.long, device=device)
        self.positions = torch.randint(0, S, (batch_size,), dtype=torch.long, device=device)

        # Pre-allocated k_pe for fused_qk_rope_cat_and_cache_mla
        self.k_pe_buf = torch.randn(
            batch_size, 1, cfg.qk_rope_head_dim, dtype=BF16, device=device
        )

        # output (written by phase3_attn_core)
        self.attn_output = torch.zeros(
            batch_size, cfg.num_local_heads, cfg.kv_lora_rank, dtype=BF16, device=device
        )


# ── Phase implementations ─────────────────────────────────────────────────
def phase1_pre_attention(hidden, residual, attn_w, cfg, norm_w):
    """fused_rms_fp8_group_quant (Triton: add+norm+FP8 quant) → fused_qkv_a_proj (CK FP8 GEMM).
    Returns pre-quantized (normed_fp8, normed_scale) so alt stream can reuse for wk proj."""
    (normed_fp8, normed_scale), normed_bf16, _, res_out = fused_rms_fp8_group_quant(
        hidden, norm_w, cfg.rms_norm_eps,
        inp2=None, inp2_weight=None, inp2_epsilon=None,
        group_size=128, dtype_quant=FP8,
        res1=residual,
        output_unquantized_inp1=True,
    )

    qkv = ck_gemm_a8w8_blockscale(
        normed_fp8, attn_w.fused_qkv_w, normed_scale, attn_w.fused_qkv_s, dtype=BF16
    )
    q_lat = qkv[..., : cfg.q_lora_rank]
    kv_lat = qkv[..., cfg.q_lora_rank : cfg.q_lora_rank + cfg.kv_lora_rank]
    return normed_bf16, res_out, q_lat, kv_lat, normed_fp8, normed_scale


def phase2_current(q_lat, attn_w, cfg):
    """q_a_norm → FP8 quant → q_b_proj CK GEMM (kv_a_norm moved to alt stream)."""
    M = q_lat.shape[0]

    q_lora = aiter_rms_norm(q_lat, attn_w.q_a_ln_w, cfg.rms_norm_eps)

    q_fp8, q_scale = per1x128_quant(q_lora, quant_dtype=FP8)
    q = ck_gemm_a8w8_blockscale(q_fp8, attn_w.q_b_w, q_scale, attn_w.q_b_s, dtype=BF16)
    q = q.view(M, cfg.num_local_heads, cfg.qk_head_dim)
    return q, q_lora


def phase2_alt(h_normed, h_fp8, h_scale, q_lora, kv_lat, attn_w, idx_w, cfg):
    """kv_a_norm (moved here to match real trace) + NSA Indexer: wq_b → wk →
    k_norm → RoPE → hadamard → act_quant → weights_proj → fill(-inf) →
    paged_mqa → topk_transform_decode.
    h_fp8/h_scale: pre-quantized h_normed from phase1 (avoids re-quantizing)."""
    M = h_fp8.shape[0]

    # kv_a_norm on alt stream (matches real SGLang stream 106 placement)
    kv_normed = aiter_rms_norm(kv_lat, attn_w.kv_a_ln_w, cfg.rms_norm_eps)

    idx_q = fp8_gemm(q_lora, idx_w.wq_b_w, idx_w.wq_b_s)
    idx_q = idx_q.view(M, cfg.index_n_heads, cfg.index_head_dim)

    # Reuse pre-quantized h_normed from phase1 (no extra dynamic_per_group_quant)
    idx_k = ck_gemm_a8w8_blockscale(h_fp8, idx_w.wk_w, h_scale, idx_w.wk_s, dtype=BF16)
    idx_k = aiter_layernorm2d(idx_k, idx_w.k_norm_w, idx_w.k_norm_b, 1e-5)

    # RoPE via rope_cached_positions_2c_fwd_inplace (in-place on 4D tensors,
    # same kernel as real SGLang: kn_entry_2c_sbhd_cached_indirect_inplace).
    # Reshape to [1, M, heads, head_dim] for the 2c kernel, slice rope portion.
    if HAS_AITER_ROPE:
        rd = idx_w.rope_head_dim  # 64
        idx_q_4d = idx_q.unsqueeze(0)               # [1, M, n_heads, head_dim]
        idx_k_4d = idx_k.unsqueeze(0).unsqueeze(2)   # [1, M, 1, head_dim]
        pos = idx_w.rope_positions[:M].unsqueeze(0)   # [1, M]
        rope_cached_positions_2c_fwd_inplace(
            idx_q_4d[..., :rd], idx_k_4d[..., :rd],
            idx_w.rope_cos, idx_w.rope_sin, pos,
            0, True, False,
        )
        idx_q = idx_q_4d.squeeze(0)
        idx_k = idx_k_4d.squeeze(0).squeeze(1)

    scale = cfg.index_head_dim**-0.5
    idx_q = hadamard_transform(idx_q.contiguous(), scale=scale)
    idx_k = hadamard_transform(
        idx_k.unsqueeze(1).contiguous(), scale=scale
    ).squeeze(1)

    q_fp8, _q_s = act_quant(idx_q)
    _k_fp8, _k_s = act_quant(idx_k.unsqueeze(1))

    # Store k + scale into NSA paged buffer (produces _set_k_and_s_triton_kernel)
    _set_k_and_s_triton(
        buf=idx_w.nsa_k_buf,
        loc=idx_w.nsa_k_loc[:M],
        index_k=_k_fp8.squeeze(1),
        index_k_scale=_k_s.squeeze(1),
        page_size=1,
    )

    # weights_proj: F.linear → hipBLAS GEMM (matches real SGLang trace: Cijk + PostGSU)
    weights = F.linear(h_normed.to(idx_w.weights_proj_w.dtype), idx_w.weights_proj_w)
    weights = (weights.float() * (cfg.index_n_heads**-0.5)).unsqueeze(1)  # [M, 1, n_heads]

    # Fill logits with -inf, then overwrite via paged MQA
    idx_w.topk_logits[:M].fill_(float("-inf"))

    if HAS_PAGED_MQA:
        deepgemm_fp8_paged_mqa_logits(
            q_fp8.unsqueeze(1),               # [M, 1, n_heads, head_dim]
            idx_w.paged_k_cache,              # [pages, 1, 1, head_dim+4] FP8
            weights,                          # [M, 1, n_heads]
            idx_w.topk_logits[:M],            # [M, max_pages] (output)
            idx_w.paged_context_lens[:M],     # [M]
            idx_w.paged_block_tables[:M],     # [M, max_pages]
            idx_w.paged_max_seq_len,
            Preshuffle=False,
            KVBlockSize=1,
            ChunkK=128,
            TotalCuCount=256,
            WavePerEU=5,
        )

    topk_ids = fast_topk_transform_fused(
        idx_w.topk_logits[:M], idx_w.topk_seq_lens[:M],
        idx_w.topk_page_table[:M], idx_w.topk_cu_seqlens_q,
        idx_w.topk_k,
    )
    return topk_ids, kv_normed


def phase3_attn_core(q, kv_normed, attn_w, cfg, kv_cache, topk_ids):
    """w_kc batched GEMM → fused_qk_rope_cat → cat q_all → TileLang sparse MLA decode
    (partial + combine = 2× main_kernel) → w_vc batched GEMM →
    fused_flatten_fp8_group_quant → o_proj CK GEMM.
    """
    M = q.shape[0]
    H = cfg.num_local_heads
    q_nope = q[..., : cfg.qk_nope_head_dim]
    q_pe = q[..., cfg.qk_nope_head_dim :]

    k_nope = kv_normed.unsqueeze(1)  # [M, 1, kv_lora]
    k_pe = kv_cache.k_pe_buf[:M]     # [M, 1, rope_dim]

    # batched GEMM w_kc FIRST (matches SGLang order: absorb before RoPE)
    # [M, H, nope=192] → [H, M, kv_lora=512]
    q_nope_absorbed = batched_fp8_gemm(
        X=q_nope,
        WQ=attn_w.w_kc.transpose(-1, -2),
        w_scale=attn_w.w_scale,
        group_size=128,
        YQ=None,
        transpose_bm=False,
        transpose_bm_in=True,
        dtype=BF16,
    )
    q_nope_absorbed = q_nope_absorbed.transpose(0, 1)  # [M, H, kv_lora=512]

    # fused_qk_rope: RoPE on q_pe/k_pe + store k cache
    # q_nope_absorbed is 512-dim (already power-of-2, no padding needed)
    q_out, _, _, _ = fused_qk_rope_cat_and_cache_mla(
        q_nope_absorbed,
        q_pe,
        k_nope,
        k_pe,
        kv_cache.kv_buffer,
        kv_cache.slot_mapping,
        kv_cache.positions,
        attn_w.cos_cache,
        attn_w.sin_cache,
        attn_w.k_scale,
        True,  # is_neox
    )

    # cat q_all (matches SGLang CatArrayBatchedCopy)
    q_nope_out = q_out[..., : cfg.kv_lora_rank]   # [M, H, 512]
    q_rope_out = q_out[..., cfg.kv_lora_rank :]    # [M, H, 64]
    q_all = torch.cat([q_nope_out, q_rope_out], dim=-1)  # [M, H, 576]

    # TileLang sparse MLA decode attention (matches --nsa-decode-backend tilelang)
    # Produces 2× main_kernel: sparse_mla_fwd_decode_partial + combine
    sm_scale = cfg.qk_head_dim ** -0.5
    indices = topk_ids[:M].unsqueeze(1).to(torch.int32)  # [M, 1, 2048]
    attn_output = tilelang_sparse_fwd(
        q=q_all,                     # [M, H, 576]
        kv=kv_cache.kv_buffer,       # [total, 1, 576]
        indices=indices,             # [M, 1, 2048]
        sm_scale=sm_scale,
        d_v=cfg.kv_lora_rank,        # 512
    )
    # tilelang returns [1, M, H, d_v]; squeeze batch dim
    if attn_output.dim() == 4:
        attn_output = attn_output.squeeze(0)
    kv_cache.attn_output = attn_output  # [M, H, kv_lora_rank]

    # batched GEMM w_vc: [M, H, kv_lora] → [H, M, v_dim]
    attn_bmm = batched_fp8_gemm(
        X=kv_cache.attn_output,
        WQ=attn_w.w_vc.transpose(-1, -2),
        w_scale=attn_w.w_scale,
        group_size=128,
        YQ=None,
        transpose_bm=False,
        transpose_bm_in=True,
        dtype=BF16,
    )
    attn_bmm = attn_bmm.transpose(0, 1)  # [M, H, v_dim]

    # fused flatten + FP8 group quant before o_proj
    attn_flat_q, attn_flat_s = fused_flatten_fp8_group_quant(
        attn_bmm, group_size=128, dtype_quant=FP8,
    )

    o = ck_gemm_a8w8_blockscale(attn_flat_q, attn_w.o_proj_w, attn_flat_s, attn_w.o_proj_s, dtype=BF16)
    return o


def phase4_shared(mlp_in, moe_w, cfg):
    """Shared expert MLP: gate_up → SiLU*gate → down (all CK FP8 GEMM)."""
    gate_up = fp8_gemm(mlp_in, moe_w.shared_gate_up_w, moe_w.shared_gate_up_s)
    activated = sgl_silu_and_mul(gate_up)
    return fp8_gemm(activated, moe_w.shared_down_w, moe_w.shared_down_s)


def phase4_routed(mlp_in, moe_w, cfg):
    """Router (wv_splitk) → biased_grouped_topk → aiter fused_moe (CK MoE GEMM)."""
    M = mlp_in.shape[0]
    dev = mlp_in.device

    # Router gate: wv_splitk BF16 GEMM (matches real SGLang trace)
    router_out = moe_w.router_out[:M]
    aiter_wv_splitk(moe_w.gate_w, mlp_in, router_out, M, 256)
    router_logits = router_out.to(BF16)

    topk_w = torch.empty(M, cfg.num_experts_per_tok, dtype=FP32, device=dev)
    topk_ids = torch.empty(M, cfg.num_experts_per_tok, dtype=torch.int32, device=dev)

    aiter_biased_grouped_topk(
        router_logits,
        moe_w.correction_bias,
        topk_w,
        topk_ids,
        cfg.n_group,
        cfg.topk_group,
        True,
        cfg.routed_scaling_factor,
    )

    # n_local_experts == n_routed_experts (no TP sharding in benchmark), skip modulo
    # routed_scaling_factor is already applied inside aiter_biased_grouped_topk
    # (see its signature: routed_scaling_factor: float = 1.0  # mul to topk_weights)
    # SGLang also skips post-multiply when _use_aiter=True (deepseek_v2.py L921-924)
    topk_w = topk_w.float()

    out = aiter_fused_moe(
        mlp_in,
        moe_w.w13_weight,
        moe_w.w2_weight,
        topk_w,
        topk_ids,
        w1_scale=moe_w.w13_scale,
        w2_scale=moe_w.w2_scale,
        quant_type=QuantType.per_1x128,
        activation=ActivationType.Silu,
    )
    return out


# ── Decode layer ──────────────────────────────────────────────────────────
def decode_layer_forward(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    attn_w: AttentionWeights,
    idx_w: IndexerWeights,
    moe_w: MoEWeights,
    cfg: GLM5Config,
    input_ln_w: torch.Tensor,
    post_attn_ln_w: torch.Tensor,
    kv_cache: KVCache,
    dual_stream: bool,
    alt_stream: Optional[torch.cuda.Stream],
):
    cur = torch.cuda.current_stream()

    # ── Phase 1 ──
    h_normed, res_out, q_lat, kv_lat, h_fp8, h_scale = phase1_pre_attention(
        hidden, residual, attn_w, cfg, input_ln_w
    )

    # ── Phase 2: dual-stream (kv_a_norm runs on alt stream with indexer) ──
    if dual_stream and alt_stream is not None:
        alt_stream.wait_stream(cur)
        q, q_lora = phase2_current(q_lat, attn_w, cfg)
        with torch.cuda.stream(alt_stream):
            topk_ids, kv_normed = phase2_alt(
                h_normed, h_fp8, h_scale, q_lora, kv_lat, attn_w, idx_w, cfg
            )
        cur.wait_stream(alt_stream)
    else:
        q, q_lora = phase2_current(q_lat, attn_w, cfg)
        topk_ids, kv_normed = phase2_alt(
            h_normed, h_fp8, h_scale, q_lora, kv_lat, attn_w, idx_w, cfg
        )

    # ── Phase 3 ──
    attn_out = phase3_attn_core(q, kv_normed, attn_w, cfg, kv_cache, topk_ids)

    # prepare_mlp: residual add + RMSNorm
    mlp_normed = torch.empty_like(attn_out)
    res_out2 = torch.empty_like(attn_out)
    aiter_fused_add_rms_norm(
        mlp_normed, attn_out, res_out, res_out2, post_attn_ln_w, cfg.rms_norm_eps
    )

    # ── Phase 4: dual-stream MoE ──
    if dual_stream and alt_stream is not None:
        alt_stream.wait_stream(cur)
        shared_out = phase4_shared(mlp_normed, moe_w, cfg)
        with torch.cuda.stream(alt_stream):
            routed_out = phase4_routed(mlp_normed, moe_w, cfg)
        cur.wait_stream(alt_stream)
    else:
        shared_out = phase4_shared(mlp_normed, moe_w, cfg)
        routed_out = phase4_routed(mlp_normed, moe_w, cfg)

    moe_out = shared_out + routed_out
    return moe_out, res_out2


# ── Benchmark harness ─────────────────────────────────────────────────────
def benchmark(
    batch_size, num_layers, warmup, iters, dual_stream, profile, prof_tag, seq_len,
    use_graph,
):
    cfg = GLM5Config(batch_size=batch_size, seq_len=seq_len)
    device = "cuda"

    print(f"\n{'=' * 72}")
    print("GLM-5.1-FP8 Decode Layer v5  (TileLang sparse MLA + real kernels)")
    print(
        f"M={batch_size}  TP={cfg.tp_size}  layers={num_layers}  "
        f"seq_len={seq_len}  splits={cfg.num_kv_splits}"
    )
    print(
        f"dual_stream={dual_stream}  graph={use_graph}  "
        f"rope={HAS_AITER_ROPE}  paged_mqa={HAS_PAGED_MQA}"
    )
    print(f"warmup={warmup}  iters={iters}")
    print(f"{'=' * 72}")

    # allocate per-layer weights
    layers_w = []
    for i in range(num_layers):
        print(f"  alloc layer {i + 1}/{num_layers} …")
        layers_w.append(
            {
                "attn": AttentionWeights(cfg, device),
                "idx": IndexerWeights(cfg, device),
                "moe": MoEWeights(cfg, device),
                "in_ln": torch.ones(cfg.hidden_size, dtype=FP32, device=device),
                "post_ln": torch.ones(cfg.hidden_size, dtype=FP32, device=device),
            }
        )

    kv_caches = [KVCache(cfg, batch_size, device) for _ in range(num_layers)]
    alt = torch.cuda.Stream() if dual_stream else None

    static_h = torch.randn(batch_size, cfg.hidden_size, dtype=BF16, device=device) * 0.01
    static_r = torch.zeros(batch_size, cfg.hidden_size, dtype=BF16, device=device)

    def run_layers():
        h, r = static_h, static_r
        for i, lw in enumerate(layers_w):
            h, r = decode_layer_forward(
                h, r,
                lw["attn"], lw["idx"], lw["moe"], cfg,
                lw["in_ln"], lw["post_ln"],
                kv_caches[i], dual_stream, alt,
            )
        return h

    # warmup (also JIT-compiles Triton kernels)
    print("warmup …")
    for _ in range(warmup):
        run_layers()
    torch.cuda.synchronize()

    # HIP graph capture
    graph = None
    if use_graph:
        print("capturing HIP graph …")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _out = run_layers()
        torch.cuda.synchronize()

    # timed iterations
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        if graph is not None:
            graph.replay()
        else:
            run_layers()
    t1.record()
    torch.cuda.synchronize()

    total_us = t0.elapsed_time(t1) * 1000.0
    avg_us = total_us / iters
    avg_layer = avg_us / num_layers
    print(
        f"\nTotal {total_us:.0f} μs  |  avg/iter ({num_layers}L) {avg_us:.1f} μs  "
        f"|  avg/layer {avg_layer:.1f} μs"
    )

    # profiler
    if profile:
        n_prof = 5
        print(f"\nprofiling ({num_layers}L × {n_prof} iters, graph={'on' if graph else 'off'}) …")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            for _ in range(n_prof):
                if graph is not None:
                    graph.replay()
                else:
                    run_layers()
            torch.cuda.synchronize()

        trace_dir = os.path.expanduser("~/SGLang-benchmarks/tmp/trace")
        os.makedirs(trace_dir, exist_ok=True)
        trace_path = os.path.join(
            trace_dir,
            f"glm5_decode_M{batch_size}_L{num_layers}_v5_trace_{prof_tag}.json.gz",
        )
        prof.export_chrome_trace(trace_path)
        print(f"\n{'*' * 72}")
        print(f"TRACE SAVED: {trace_path}")
        print(f"{'*' * 72}")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))


# ── CLI ───────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="GLM-5.1-FP8 decode-layer v5 (TileLang sparse MLA, NSA indexer)"
    )
    p.add_argument("--batch-size", type=int, nargs="+", default=[1])
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--seq-len", type=int, default=4096,
                    help="KV cache sequence length per request")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--no-dual-stream", action="store_true")
    p.add_argument("--no-graph", action="store_true",
                    help="Skip HIP graph capture (kernels will NOT overlap)")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--prof-tag", type=str, default="baseline")
    args = p.parse_args()

    for bs in args.batch_size:
        benchmark(
            bs, args.num_layers, args.warmup, args.iters,
            dual_stream=not args.no_dual_stream,
            profile=args.profile,
            prof_tag=args.prof_tag,
            seq_len=args.seq_len,
            use_graph=not args.no_graph,
        )


if __name__ == "__main__":
    main()
