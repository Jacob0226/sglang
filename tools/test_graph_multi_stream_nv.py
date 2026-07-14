"""
Test multi-stream GEMM overlap at various token counts — NVIDIA version.
Each token count runs in a separate subprocess.

Each "path" simulates a full MLP: gate_up GEMM → mul → down GEMM (3 kernels),
matching the real shared/routed expert workload in SGLang.

Run on B200:
  python test_graph_multi_stream_nv.py
"""
import subprocess
import sys
import os

TOKEN_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128]

WORKER_SCRIPT = r'''
import torch
import time
import sys

TOKENS = int(sys.argv[1])
REPS = 50
HIDDEN = 7168
INTER = 2048

s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

# Shared expert MLP weights
w_up1 = torch.randn(HIDDEN, INTER * 2, device="cuda", dtype=torch.bfloat16)
w_dn1 = torch.randn(INTER, HIDDEN, device="cuda", dtype=torch.bfloat16)

# Routed expert MLP weights
w_up2 = torch.randn(HIDDEN, INTER * 2, device="cuda", dtype=torch.bfloat16)
w_dn2 = torch.randn(INTER, HIDDEN, device="cuda", dtype=torch.bfloat16)

x = torch.randn(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)

# Pre-allocated buffers
buf_up1 = torch.empty(TOKENS, INTER * 2, device="cuda", dtype=torch.bfloat16)
buf_h1 = torch.empty(TOKENS, INTER, device="cuda", dtype=torch.bfloat16)
buf_dn1 = torch.empty(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)
buf_up2 = torch.empty(TOKENS, INTER * 2, device="cuda", dtype=torch.bfloat16)
buf_h2 = torch.empty(TOKENS, INTER, device="cuda", dtype=torch.bfloat16)
buf_dn2 = torch.empty(TOKENS, HIDDEN, device="cuda", dtype=torch.bfloat16)

def shared_mlp():
    torch.mm(x, w_up1, out=buf_up1)
    g, v = buf_up1[:, :INTER], buf_up1[:, INTER:]
    torch.mul(g, v, out=buf_h1)
    torch.mm(buf_h1, w_dn1, out=buf_dn1)

def routed_mlp():
    torch.mm(x, w_up2, out=buf_up2)
    g, v = buf_up2[:, :INTER], buf_up2[:, INTER:]
    torch.mul(g, v, out=buf_h2)
    torch.mm(buf_h2, w_dn2, out=buf_dn2)

def bench(fn, reps=REPS):
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(reps):
        fn()
        torch.cuda.synchronize()
    return (time.perf_counter_ns() - t0) / reps

# Warmup
for _ in range(10):
    shared_mlp(); routed_mlp()
torch.cuda.synchronize()
for _ in range(5):
    with torch.cuda.stream(s1): shared_mlp(); routed_mlp()
    with torch.cuda.stream(s2): shared_mlp(); routed_mlp()
torch.cuda.synchronize()

# Eager
def eager_single():
    with torch.cuda.stream(s1):
        shared_mlp(); routed_mlp()
def eager_dual():
    with torch.cuda.stream(s1):
        shared_mlp()
    with torch.cuda.stream(s2):
        routed_mlp()

t_es = bench(eager_single)
t_ed = bench(eager_dual)

# Graph single
pool = torch.cuda.graph_pool_handle()

g_s = torch.cuda.CUDAGraph()
with torch.cuda.graph(g_s, pool=pool, stream=s1):
    shared_mlp()
    routed_mlp()
torch.cuda.synchronize()

# Graph dual
g_d = torch.cuda.CUDAGraph()
with torch.cuda.graph(g_d, pool=pool, stream=s1):
    shared_mlp()
    with torch.cuda.stream(s2):
        s2.wait_stream(s1)
        routed_mlp()
    s1.wait_stream(s2)
torch.cuda.synchronize()

for _ in range(5):
    g_s.replay(); torch.cuda.synchronize()
    g_d.replay(); torch.cuda.synchronize()

t_gs = bench(lambda: g_s.replay())
t_gd = bench(lambda: g_d.replay())

print(f"{TOKENS},{t_es},{t_ed},{t_gs},{t_gd}")
'''

# Print device info
dev_cmd = 'import torch; print(f"Device: {torch.cuda.get_device_name(0)}"); print(f"PyTorch: {torch.__version__}")'
info = subprocess.run([sys.executable, "-c", dev_cmd], capture_output=True, text=True)
print(info.stdout.strip())

print()
print("=" * 100)
print(f" {'tokens':>6} | {'eager_s':>10} | {'eager_d':>10} | {'e_ratio':>7} | {'graph_s':>10} | {'graph_d':>10} | {'g_ratio':>7} | overlap?")
print("-" * 100)

for tokens in TOKEN_COUNTS:
    try:
        result = subprocess.run(
            [sys.executable, "-c", WORKER_SCRIPT, str(tokens)],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")},
        )
    except subprocess.TimeoutExpired:
        print(f"  {tokens:>5} | TIMEOUT")
        continue

    if result.returncode != 0:
        err = result.stderr.strip().split("\n")[-1][:80]
        print(f"  {tokens:>5} | ERROR: {err}")
        continue

    line = result.stdout.strip().split("\n")[-1]
    tok, t_es, t_ed, t_gs, t_gd = line.split(",")
    t_es, t_ed, t_gs, t_gd = float(t_es), float(t_ed), float(t_gs), float(t_gd)
    e_ratio = t_ed / t_es
    g_ratio = t_gd / t_gs
    overlap = "YES" if g_ratio < 0.75 else "no"
    print(
        f"  {tokens:>5} | {t_es/1e3:>8.1f}us | {t_ed/1e3:>8.1f}us | {e_ratio:>7.3f} "
        f"| {t_gs/1e3:>8.1f}us | {t_gd/1e3:>8.1f}us | {g_ratio:>7.3f} | {overlap}"
    )

print("=" * 100)
print()
print("Each path = full MLP: gate_up GEMM [T,7168]×[7168,4096] → mul → down GEMM [T,2048]×[2048,7168]")
print("If g_ratio < 0.75 → cuBLAS GEMM overlaps across streams (dual-stream works)")
print("If g_ratio ≈ 1.0 → cuBLAS serializes or GPU fully saturated")
