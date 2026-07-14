"""
GLM5 MoE dual-stream unit test — extracted from deepseek_v2.py forward_normal_dual_stream.

Replicates the EXACT same work on each stream:
  current_stream: shared expert  → gate_up FP8 GEMM → SiLU → down FP8 GEMM
  alt_stream:     routed expert  → gate GEMM → topk → expert up GEMM → act → expert down GEMM

Supports both MI355X (ROCm, float8_e4m3fnuz) and B200 (CUDA, float8_e4m3fn).
Compares single-stream vs dual-stream graph capture.
"""
import torch
import torch.nn.functional as F
import time

# ── Auto-detect platform ──
IS_HIP = hasattr(torch.version, "hip") and torch.version.hip is not None
FP8 = torch.float8_e4m3fnuz if IS_HIP else torch.float8_e4m3fn
DEVICE = "cuda"

# ── GLM5 / DeepSeek-V3 dimensions ──
HIDDEN = 7168
SHARED_INTER = 2048      # moe_intermediate_size * n_shared_experts
N_EXPERTS = 256
TOP_K = 8
ROUTED_INTER = 2048      # moe_intermediate_size

REPS = 50
WARMUP = 10


def to_fp8(t):
    """Convert BF16 tensor to FP8 with per-tensor scale."""
    amax = t.abs().amax().float().clamp(min=1e-12)
    scale = (torch.finfo(FP8).max / amax).float()
    return (t.float() * scale).clamp(
        min=torch.finfo(FP8).min, max=torch.finfo(FP8).max
    ).to(FP8), scale.reciprocal().view(1)


def fp8_linear(x_fp8, x_scale, w_fp8, w_scale):
    """FP8 matmul: x_fp8 [M,K] @ w_fp8.t() [K,N] → [M,N] BF16."""
    return torch._scaled_mm(
        x_fp8, w_fp8.t(),
        scale_a=x_scale, scale_b=w_scale,
        out_dtype=torch.bfloat16,
    )


class SharedExpert:
    """Shared expert path: gate_up → SiLU → down."""

    def __init__(self, M):
        self.w_gate_up_fp8, self.s_gate_up = to_fp8(
            torch.randn(SHARED_INTER * 2, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
        )
        self.w_down_fp8, self.s_down = to_fp8(
            torch.randn(HIDDEN, SHARED_INTER, device=DEVICE, dtype=torch.bfloat16)
        )
        # pre-allocate input
        self.x_fp8, self.x_scale = to_fp8(
            torch.randn(M, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
        )

    def __call__(self):
        # gate_up: [M, H] → [M, 2*inter]
        gate_up = fp8_linear(self.x_fp8, self.x_scale, self.w_gate_up_fp8, self.s_gate_up)
        # SiLU-and-mul: split into gate and up, apply silu(gate)*up
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        # down: [M, inter] → [M, H]
        h_fp8, h_scale = to_fp8(hidden)
        out = fp8_linear(h_fp8, h_scale, self.w_down_fp8, self.s_down)
        return out


class RoutedExperts:
    """Routed expert path: gate → topk → expert up → act → expert down."""

    def __init__(self, M):
        # gate weights: [N_EXPERTS, H]
        self.w_gate = torch.randn(N_EXPERTS, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
        # expert weights (simulate TOP_K experts)
        self.w_up_fp8, self.s_up = to_fp8(
            torch.randn(ROUTED_INTER * 2, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
        )
        self.w_down_fp8, self.s_down = to_fp8(
            torch.randn(HIDDEN, ROUTED_INTER, device=DEVICE, dtype=torch.bfloat16)
        )
        self.x_fp8, self.x_scale = to_fp8(
            torch.randn(M, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
        )
        self.x_bf16 = torch.randn(M, HIDDEN, device=DEVICE, dtype=torch.bfloat16)

    def __call__(self):
        # gate: [M, H] @ [H, N_EXPERTS] → [M, N_EXPERTS]
        logits = F.linear(self.x_bf16, self.w_gate)
        # topk
        scores, indices = torch.topk(logits, TOP_K, dim=-1)
        weights = torch.softmax(scores, dim=-1)
        # expert up: [M, H] → [M, 2*inter]
        gate_up = fp8_linear(self.x_fp8, self.x_scale, self.w_up_fp8, self.s_up)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        # expert down: [M, inter] → [M, H]
        h_fp8, h_scale = to_fp8(hidden)
        out = fp8_linear(h_fp8, h_scale, self.w_down_fp8, self.s_down)
        return out


def bench_graph(g, reps=REPS):
    for _ in range(WARMUP):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(reps):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter_ns() - t0) / reps


def test_overlap(M):
    shared = SharedExpert(M)
    routed = RoutedExperts(M)

    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    # warmup on all streams
    for _ in range(5):
        shared(); routed()
        with torch.cuda.stream(s1): shared(); routed()
        with torch.cuda.stream(s2): shared(); routed()
    torch.cuda.synchronize()

    # ── Single-stream: shared then routed, sequential ──
    g_single = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_single, stream=s1):
        shared()
        routed()
    torch.cuda.synchronize()

    # ── Dual-stream: SGLang fork-join pattern ──
    #   alt.wait(current)       ← fork
    #   shared on current       ← concurrent
    #   routed on alt           ← concurrent
    #   current.wait(alt)       ← join
    g_dual = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_dual, stream=s1):
        s2.wait_stream(s1)
        shared()
        with torch.cuda.stream(s2):
            routed()
        s1.wait_stream(s2)
    torch.cuda.synchronize()

    t_single = bench_graph(g_single)
    t_dual = bench_graph(g_dual)
    ratio = t_dual / t_single

    del g_single, g_dual
    torch.cuda.empty_cache()
    return t_single, t_dual, ratio


# ── Main ──
dev = torch.cuda.get_device_name(0)
platform = "ROCm (HIP)" if IS_HIP else "CUDA"
print(f"Device:   {dev}")
print(f"PyTorch:  {torch.__version__}")
print(f"Platform: {platform}")
print(f"FP8:      {FP8}")
print(f"Model:    GLM5 / DeepSeek-V3  (H={HIDDEN}, shared_inter={SHARED_INTER}, "
      f"n_experts={N_EXPERTS}, top_k={TOP_K})")

print(f"\n{'='*78}")
print(f"GLM5 MoE dual-stream: shared expert vs routed expert")
print(f"  current_stream: gate_up({HIDDEN}→{SHARED_INTER*2}) → SiLU → down({SHARED_INTER}→{HIDDEN})")
print(f"  alt_stream:     gate({HIDDEN}→{N_EXPERTS}) → topk → up({HIDDEN}→{ROUTED_INTER*2}) → SiLU → down({ROUTED_INTER}→{HIDDEN})")
print(f"{'='*78}")
print(f"  {'M':>6} | {'single':>12} | {'dual':>12} | {'ratio':>7} | {'saved':>7} | overlap?")
print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*7}-+-{'-'*7}-+-{'-'*10}")

for M in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
    try:
        t_s, t_d, r = test_overlap(M)
        saved_us = (t_s - t_d) / 1e3
        tag = "YES" if r < 0.85 else "no"
        print(f"  {M:>6} | {t_s/1e3:>10.1f}us | {t_d/1e3:>10.1f}us | {r:>7.3f} | {saved_us:>+6.1f}us | {tag}", flush=True)
    except Exception as e:
        print(f"  {M:>6} | ERROR: {e}", flush=True)

print(f"\n{'='*78}")
print("ratio < 0.85 → dual-stream overlap is effective")
print("ratio ≈ 1.0  → no overlap (kernels serialized)")
print("ratio > 1.0  → dual-stream is SLOWER (sync overhead > overlap benefit)")
print(f"{'='*78}")
