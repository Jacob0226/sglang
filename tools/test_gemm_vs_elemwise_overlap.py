"""
Diagnostic: dual-stream overlap with CORRECT fork-join pattern.

Previous tests were WRONG — s2.wait_stream(s1) was placed AFTER s1's work,
creating a serial dependency. SGLang places it BEFORE all work (fork point),
allowing both streams to execute concurrently.

SGLang pattern:
  alt_stream.wait_stream(current)  ← fork: alt starts after prior layer
  shared_experts on current        ← concurrent
  routed_experts on alt            ← concurrent
  current.wait_stream(alt)         ← join

Tests:
  1. GEMM + GEMM
  2. GEMM + elementwise (asymmetric, most representative)
  3. elem + elem
"""
import subprocess, sys, os

WORKER = r'''
import torch, time, sys, json

LOG = "/tmp/debug-0c86e8.log"
def log(msg, **data):
    with open(LOG, "a") as f:
        f.write(json.dumps({"sessionId":"0c86e8","location":"test_diag","message":msg,"data":data,"timestamp":int(time.time()*1000)}) + "\n")

REPS = 50
HIDDEN = 7168
INTER = 2048
SIZE = 32 * 1024 * 1024

s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

# GEMM buffers
x = torch.randn(1, HIDDEN, device="cuda", dtype=torch.bfloat16)
w = torch.randn(HIDDEN, INTER, device="cuda", dtype=torch.bfloat16)
out_gemm = torch.empty(1, INTER, device="cuda", dtype=torch.bfloat16)

# Second GEMM buffers (separate to avoid conflicts)
w2 = torch.randn(HIDDEN, INTER, device="cuda", dtype=torch.bfloat16)
out_gemm2 = torch.empty(1, INTER, device="cuda", dtype=torch.bfloat16)

# Elementwise buffers
e1 = torch.randn(SIZE, device="cuda", dtype=torch.bfloat16)
e2 = torch.empty_like(e1)
e3 = torch.randn(SIZE, device="cuda", dtype=torch.bfloat16)
e4 = torch.empty_like(e3)

def do_gemm_a():
    torch.mm(x, w, out=out_gemm)
    torch.mm(x, w, out=out_gemm)
    torch.mm(x, w, out=out_gemm)

def do_gemm_b():
    torch.mm(x, w2, out=out_gemm2)
    torch.mm(x, w2, out=out_gemm2)
    torch.mm(x, w2, out=out_gemm2)

def do_elem_a():
    e2.copy_(e1).mul_(1.01).add_(0.01).relu_().mul_(1.01).add_(0.01)

def do_elem_b():
    e4.copy_(e3).mul_(1.01).add_(0.01).relu_().mul_(1.01).add_(0.01)

def bench(fn, reps=REPS):
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(reps):
        fn()
        torch.cuda.synchronize()
    return (time.perf_counter_ns() - t0) / reps

# Warmup all ops on all streams
for _ in range(10):
    do_gemm_a(); do_gemm_b(); do_elem_a(); do_elem_b()
torch.cuda.synchronize()
for _ in range(5):
    with torch.cuda.stream(s1): do_gemm_a(); do_gemm_b(); do_elem_a(); do_elem_b()
    with torch.cuda.stream(s2): do_gemm_a(); do_gemm_b(); do_elem_a(); do_elem_b()
torch.cuda.synchronize()

def capture_single_dual(work_a, work_b, label):
    """Capture single-stream and dual-stream graphs with CORRECT fork-join pattern."""
    # Single: both on s1, sequential
    g_s = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_s, stream=s1):
        work_a()
        work_b()
    torch.cuda.synchronize()

    # Dual: CORRECT SGLang fork-join pattern
    #   s2.wait_stream(s1)  ← fork (before any work)
    #   work_a on s1        ← concurrent
    #   work_b on s2        ← concurrent
    #   s1.wait_stream(s2)  ← join
    g_d = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_d, stream=s1):
        s2.wait_stream(s1)            # fork: s2 starts after prior ops (= nothing at graph start)
        work_a()                      # s1: work A (runs concurrently with B)
        with torch.cuda.stream(s2):
            work_b()                  # s2: work B (runs concurrently with A)
        s1.wait_stream(s2)            # join: s1 waits for s2
    torch.cuda.synchronize()

    # Warmup replay
    for _ in range(5):
        g_s.replay(); torch.cuda.synchronize()
        g_d.replay(); torch.cuda.synchronize()

    t_s = bench(lambda: g_s.replay())
    t_d = bench(lambda: g_d.replay())
    ratio = t_d / t_s

    # #region agent log
    log(f"result_{label}", single_ns=t_s, dual_ns=t_d, ratio=round(ratio, 4), label=label, hypothesisId="fork_join_fix")
    # #endregion

    del g_s, g_d
    return t_s, t_d, ratio

dev = torch.cuda.get_device_name(0)
ver = torch.__version__
print(f"Device: {dev}")
print(f"PyTorch: {ver}")
log("device_info", device=dev, pytorch=ver)
print()

tests = [
    ("gemm+gemm", do_gemm_a, do_gemm_b),
    ("gemm+elem", do_gemm_a, do_elem_a),
    ("elem+elem", do_elem_a, do_elem_b),
]

for label, wa, wb in tests:
    t_s, t_d, r = capture_single_dual(wa, wb, label)
    overlap = "YES" if r < 0.75 else "no"
    print(f"  {label:>12}: single={t_s/1e3:.1f}us  dual={t_d/1e3:.1f}us  ratio={r:.3f}  {overlap}")

print()
print("If ratio < 0.75 → overlap confirmed (dual-stream works)")
print("Previous tests showed ratio=1.0 due to WRONG wait_stream placement (bug in test)")
'''

print("Running diagnostic with CORRECT fork-join pattern...\n", flush=True)
try:
    result = subprocess.run(
        [sys.executable, "-c", WORKER],
        capture_output=True, text=True, timeout=120,
        env={**os.environ,
             "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES", "0"),
             "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")},
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-500:])
except subprocess.TimeoutExpired:
    print("TIMEOUT — graph capture or replay hung")
