// KDA target verify with the ReplaySSM ring on gfx950: causal conv1d (width 4) + safe-gate
// delta rule over T draft tokens, K = V = 128, one KDA head per (q, k, v) head.
//
// One block of 256 threads per (request, v head, BVC-wide V tile). The conv, gate and beta
// of the head are computed once per block and shared through LDS; the recurrence keeps a
// 128 x BVC fp32 state tile in registers, KSPLIT adjacent lanes per V column.
//
// The ring values (raw k / v, gate, beta) and the intermediate conv window match
// fused_kda_conv_gating_verify_kernel bit for bit: the conv products are rounded before
// the adds, and exp / divide follow the Triton gfx950 lowering. The output differs from it
// in reduction order and in the l2 norm, which uses v_rsq_f32.

#include <hip/hip_runtime.h>
#include <stdint.h>

#define KD 128
#define NTHREADS 256

typedef unsigned short bf16_t;
typedef float f2 __attribute__((ext_vector_type(2)));

struct Args {
    const bf16_t* x;          // [B * T, dim]
    const float* w;           // [dim, 4] contiguous
    const bf16_t* cs;         // conv state, dim contiguous
    const int32_t* cs_idx;    // [B]
    bf16_t* iw;               // intermediate conv window or null
    const int32_t* iw_idx;    // [B]
    const bf16_t* a;          // [B * T, HV * K]
    const bf16_t* b;          // [B * T, HV]
    const float* A_log;       // [HV]
    const float* dt_bias;     // [HV * K]
    const float* state;       // [slots, HV, V, K]
    const int32_t* h0_idx;    // [B]
    bf16_t* o;                // [B * T, HV, V]
    bf16_t* rawv;             // [slots, HV, L, V]
    bf16_t* rawk;             // [slots, H, L, K]
    float* ring_g;            // [slots, HV, L, K]
    float* ring_beta;         // [slots, HV, L]
    const int32_t* ring_idx;  // [B]
    int64_t x_stride, cs_line, cs_tok, iw_line, iw_step, iw_win;
    int64_t a_stride, b_stride, rawv_slot, rawk_slot, g_slot, beta_slot;
    int32_t L, H, HV;
    float lower_bound, scale;
};

__device__ __forceinline__ float bf2f(bf16_t v) { return __uint_as_float(((uint32_t)v) << 16); }

__device__ __forceinline__ bf16_t f2bf(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (bf16_t)(u >> 16);
}

// tl.exp on gfx950: v_exp_f32(fma(x, log2e, off)) * 2^-off, off = 64 below 2^-126.
__device__ __forceinline__ float tl_exp(float x) {
    const float l2e = __uint_as_float(0x3fb8aa3bu);
    const bool small = __fmul_rn(x, l2e) < -126.0f;
    const float r = __builtin_amdgcn_exp2f(__builtin_fmaf(x, l2e, small ? 64.0f : 0.0f));
    return __builtin_amdgcn_ldexpf(r, small ? -64 : 0);
}

// Width-4 conv tap sum then SiLU, rounded to bf16. hipcc contracts mul + add into FMA by
// default; the Triton kernel rounds every product.
__device__ __forceinline__ bf16_t conv_silu(bf16_t x0, bf16_t x1, bf16_t x2, bf16_t x3, const float* w) {
#pragma clang fp contract(off)
    float acc = 0.0f + bf2f(x0) * w[0];
    acc = acc + bf2f(x1) * w[1];
    acc = acc + bf2f(x2) * w[2];
    acc = acc + bf2f(x3) * w[3];
    return f2bf(acc / (1.0f + tl_exp(-acc)));
}

template <int CTRL>
__device__ __forceinline__ float dpp(float v) {
    return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), CTRL, 0xf, 0xf, false));
}

// Sum over aligned groups of 8 lanes, result in every lane: quad xor 1, quad xor 2, then
// row_half_mirror pairs the two quads.
__device__ __forceinline__ float sum8(float v) {
    v += dpp<0xb1>(v);
    v += dpp<0x4e>(v);
    return v + dpp<0x141>(v);
}

__device__ __forceinline__ float wave_sum(float v) {
    v = sum8(v);
    v += dpp<0x140>(v);
    return __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 0)) +
           __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 16)) +
           __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 32)) +
           __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), 48));
}

template <int T, int BVC>
__device__ __forceinline__ void kda_verify(const Args& p) {
    // Recurrence: 8 lanes split K (lane ks owns float4 groups ks, ks + 8, ...), so a V column's
    // dot products reduce over 8 lanes with DPP; each thread holds VPT columns, NCG apart.
    constexpr int KSPLIT = 8;
    constexpr int KG = KD / (4 * KSPLIT);
    constexpr int NCG = NTHREADS / KSPLIT;
    constexpr int VPT = BVC / NCG;
    static_assert(VPT >= 1 && BVC % NCG == 0, "BVC must be a multiple of 32");

    __shared__ float4 s_q[T][KD / 4];
    __shared__ float4 s_k[T][KD / 4];
    __shared__ float4 s_dec[T][KD / 4];
    __shared__ float s_v[T][BVC];
    __shared__ float s_beta[T];
    __shared__ float s_ss[NTHREADS / 64][T];

    const int tid = threadIdx.x;
    const int vt = blockIdx.x;
    const int n = blockIdx.y / p.HV;
    const int hv = blockIdx.y % p.HV;
    const int h = hv / (p.HV / p.H);
    const int v0 = vt * BVC;
    const int64_t bos = (int64_t)n * T;

    const int64_t cs_slot = p.cs_idx[n];
    if (cs_slot < 0) return;
    const int64_t h0 = p.h0_idx[n];
    const int64_t ring = p.ring_idx[n];
    const int64_t iw_slot = p.iw != nullptr ? p.iw_idx[n] : 0;
    const bool qk_owner = vt == 0 && hv % (p.HV / p.H) == 0;

    // Recurrence mapping. The state loads are issued after the prep loads: vmcnt retires in
    // order, so the prep would otherwise wait for the whole state tile. All global loads are
    // unconditional (clamped addresses) so no branch splits them into separate round trips.
    const int cg = tid / KSPLIT;
    const int ks = tid % KSPLIT;
    // hs[j][2i], hs[j][2i + 1]: the low / high halves of float4 group i of column j.
    f2 hs[VPT][2 * KG];
    auto load_state = [&]() {
        const int64_t slot = h0 > 0 ? h0 : 0;
#pragma unroll
        for (int j = 0; j < VPT; ++j) {
            const float4* src = reinterpret_cast<const float4*>(
                p.state + ((slot * p.HV + hv) * KD + v0 + cg + NCG * j) * KD);
#pragma unroll
            for (int i = 0; i < KG; ++i) {
                const float4 f = src[i * KSPLIT + ks];
                hs[j][2 * i] = f2{f.x, f.y};
                hs[j][2 * i + 1] = f2{f.z, f.w};
            }
        }
    };

    // Prep: threads 0..127 own q channel ch and gate channel ch; 128..255 own k channel ch,
    // v column ch (ch < BVC) and beta (ch < T).
    const int ch = tid & (KD - 1);
    const bool is_k = __builtin_amdgcn_readfirstlane(tid) >= KD;  // wave-uniform
    const int64_t qk_ch = (is_k ? (int64_t)p.H * KD : 0) + (int64_t)h * KD + ch;
    float* const sq = reinterpret_cast<float*>(s_q);
    float* const sk = reinterpret_cast<float*>(s_k);
    float* const sdec = reinterpret_cast<float*>(s_dec);
    float qk[T];
    const bf16_t* cs_base = p.cs + cs_slot * p.cs_line;
    bf16_t xw[T + 3];
#pragma unroll
    for (int j = 0; j < 3; ++j) xw[j] = cs_base[qk_ch + j * p.cs_tok];
#pragma unroll
    for (int t = 0; t < T; ++t) xw[t + 3] = p.x[(bos + t) * p.x_stride + qk_ch];
    const float4 wq4 = *reinterpret_cast<const float4*>(p.w + qk_ch * 4);
    const float wq[4] = {wq4.x, wq4.y, wq4.z, wq4.w};

    const bool has_v = is_k && ch < BVC;
    const int64_t v_ch = 2 * (int64_t)p.H * KD + (int64_t)hv * KD + v0 + (ch < BVC ? ch : BVC - 1);
    bf16_t vw[T + 3];
#pragma unroll
    for (int j = 0; j < 3; ++j) vw[j] = cs_base[v_ch + j * p.cs_tok];
#pragma unroll
    for (int t = 0; t < T; ++t) vw[t + 3] = p.x[(bos + t) * p.x_stride + v_ch];
    const float4 wv4 = *reinterpret_cast<const float4*>(p.w + v_ch * 4);
    const float wv[4] = {wv4.x, wv4.y, wv4.z, wv4.w};

    float av[T];
    const float dtb = p.dt_bias[(int64_t)hv * KD + ch];
    const float a_log = p.A_log[hv];
#pragma unroll
    for (int t = 0; t < T; ++t) av[t] = bf2f(p.a[(bos + t) * p.a_stride + (int64_t)hv * KD + ch]);
    const float bb = bf2f(p.b[(bos + (ch < T ? ch : T - 1)) * p.b_stride + hv]);
    load_state();

    // conv + silu, rounded to bf16 as the unfused path stores it
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const bf16_t r = conv_silu(xw[t], xw[t + 1], xw[t + 2], xw[t + 3], wq);
        qk[t] = bf2f(r);
        (is_k ? sk : sq)[t * KD + ch] = qk[t];
        if (is_k && qk_owner && ring >= 0 && t < p.L)
            p.rawk[ring * p.rawk_slot + ((int64_t)h * p.L + t) * KD + ch] = r;
    }
    if (has_v) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const bf16_t r = conv_silu(vw[t], vw[t + 1], vw[t + 2], vw[t + 3], wv);
            s_v[t][ch] = bf2f(r);
            if (ring >= 0 && t < p.L)
                p.rawv[ring * p.rawv_slot + ((int64_t)hv * p.L + t) * KD + v0 + ch] = r;
        }
    }
    if (!is_k) {
        const float ea = tl_exp(a_log);
#pragma unroll
        for (int t = 0; t < T; ++t) {
            const float y = __fmul_rn(ea, __fadd_rn(av[t], dtb));
            const float g = __fmul_rn(p.lower_bound, 1.0f / (1.0f + tl_exp(-y)));
            sdec[t * KD + ch] = tl_exp(g);
            if (vt == 0 && ring >= 0 && t < p.L)
                p.ring_g[ring * p.g_slot + ((int64_t)hv * p.L + t) * KD + ch] = g;
        }
    }
    if (is_k && ch < T) {
        const float beta = 1.0f / (1.0f + tl_exp(-bb));
        s_beta[ch] = beta;
        if (vt == 0 && ring >= 0 && ch < p.L) p.ring_beta[ring * p.beta_slot + (int64_t)hv * p.L + ch] = beta;
    }
    if (p.iw != nullptr) {
        bf16_t* iw_base = p.iw + iw_slot * p.iw_line;
        if (qk_owner) {
#pragma unroll
            for (int t = 0; t < T; ++t)
#pragma unroll
                for (int j = 0; j < 3; ++j) iw_base[t * p.iw_step + qk_ch + j * p.iw_win] = xw[t + 1 + j];
        }
        if (has_v) {
#pragma unroll
            for (int t = 0; t < T; ++t)
#pragma unroll
                for (int j = 0; j < 3; ++j) iw_base[t * p.iw_step + v_ch + j * p.iw_win] = vw[t + 1 + j];
        }
    }
    // l2 norm sums of squares: waves 0-1 hold the q rows, waves 2-3 the k rows. The norms are
    // applied inside the recurrence instead of rewriting q / k.
    {
        float ss[T];
#pragma unroll
        for (int t = 0; t < T; ++t) ss[t] = wave_sum(qk[t] * qk[t]);
        if ((tid & 63) == 0) {
#pragma unroll
            for (int t = 0; t < T; ++t) s_ss[tid >> 6][t] = ss[t];
        }
    }
    __syncthreads();

    if (h0 < 0) {
#pragma unroll
        for (int j = 0; j < VPT; ++j)
#pragma unroll
            for (int i = 0; i < 2 * KG; ++i) hs[j][i] = f2{0.0f, 0.0f};
    }
    float out[VPT][T];
#pragma unroll
    for (int t = 0; t < T; ++t) {
        const float rk = __builtin_amdgcn_rsqf(s_ss[2][t] + s_ss[3][t] + 1e-6f);
        const float rq = __builtin_amdgcn_rsqf(s_ss[0][t] + s_ss[1][t] + 1e-6f) * p.scale;
        f2 d[2 * KG], k[2 * KG], q[2 * KG];
#pragma unroll
        for (int i = 0; i < KG; ++i) {
            const float4 fd = s_dec[t][i * KSPLIT + ks];
            const float4 fk = s_k[t][i * KSPLIT + ks];
            const float4 fq = s_q[t][i * KSPLIT + ks];
            d[2 * i] = f2{fd.x, fd.y};
            d[2 * i + 1] = f2{fd.z, fd.w};
            k[2 * i] = f2{fk.x, fk.y};
            k[2 * i + 1] = f2{fk.z, fk.w};
            q[2 * i] = f2{fq.x, fq.y};
            q[2 * i + 1] = f2{fq.z, fq.w};
        }
        float pk[VPT];
#pragma unroll
        for (int j = 0; j < VPT; ++j) {
            f2 a0 = f2{0.0f, 0.0f}, a1 = f2{0.0f, 0.0f};
#pragma unroll
            for (int i = 0; i < 2 * KG; i += 2) {
                hs[j][i] *= d[i];
                hs[j][i + 1] *= d[i + 1];
                a0 = __builtin_elementwise_fma(hs[j][i], k[i], a0);
                a1 = __builtin_elementwise_fma(hs[j][i + 1], k[i + 1], a1);
            }
            const f2 a = a0 + a1;
            pk[j] = a.x + a.y;
        }
#pragma unroll
        for (int j = 0; j < VPT; ++j) pk[j] = sum8(pk[j]);
#pragma unroll
        for (int j = 0; j < VPT; ++j) {
            // h += (k * rk) * ((v - rk * (h . k)) * beta)
            const float u = (s_v[t][cg + NCG * j] - pk[j] * rk) * s_beta[t] * rk;
            const f2 uu = f2{u, u};
            f2 a0 = f2{0.0f, 0.0f}, a1 = f2{0.0f, 0.0f};
#pragma unroll
            for (int i = 0; i < 2 * KG; i += 2) {
                hs[j][i] = __builtin_elementwise_fma(k[i], uu, hs[j][i]);
                hs[j][i + 1] = __builtin_elementwise_fma(k[i + 1], uu, hs[j][i + 1]);
                a0 = __builtin_elementwise_fma(hs[j][i], q[i], a0);
                a1 = __builtin_elementwise_fma(hs[j][i + 1], q[i + 1], a1);
            }
            const f2 a = a0 + a1;
            out[j][t] = a.x + a.y;
        }
#pragma unroll
        for (int j = 0; j < VPT; ++j) out[j][t] = sum8(out[j][t]) * rq;
    }
    // Every lane of a column group holds the sums; lane ks stores token ks.
#pragma unroll
    for (int j = 0; j < VPT; ++j) {
        float v = out[j][0];
#pragma unroll
        for (int t = 1; t < T; ++t) v = ks == t ? out[j][t] : v;
        if (ks < T) p.o[((bos + ks) * p.HV + hv) * KD + v0 + cg + NCG * j] = f2bf(v);
    }
}

#define KDA_VERIFY(T, BVC)                                                                      \
    extern "C" __global__ void __launch_bounds__(NTHREADS) kda_verify_t##T##_v##BVC(Args p) { \
        kda_verify<T, BVC>(p);                                                                  \
    }

KDA_VERIFY(6, 32)
KDA_VERIFY(6, 64)
KDA_VERIFY(8, 32)
KDA_VERIFY(8, 64)
