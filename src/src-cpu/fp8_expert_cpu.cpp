// fp8_expert_cpu.cpp — see header for the design contract.
//
// Build: any of -O2/-O3; SIMD paths use per-function target attributes, so no
// global -mavx* flags are required (they don't hurt). Do NOT enable
// -ffast-math for this TU: FP8-E4M3 subnormals decode through an exact
// fp32-subnormal multiply, and FTZ/DAZ would flush them to zero.

#include "fp8_expert_cpu.h"

#include <immintrin.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace garlic_cpu {

// ===========================================================================
// Scalar numeric helpers
// ===========================================================================

float fp8_e4m3_to_float(uint8_t b) {
    const int s = b >> 7;
    const int e = (b >> 3) & 0xF;
    const int m = b & 0x7;
    float v;
    if (e == 0)                 v = std::ldexp((float)m, -9);        // subnormal: m * 2^-3 * 2^-6
    else if (e == 15 && m == 7) v = std::numeric_limits<float>::quiet_NaN();
    else                        v = std::ldexp(8.0f + (float)m, e - 10); // (1 + m/8) * 2^(e-7)
    return s ? -v : v;
}

static const float* fp8_lut() {
    static const std::vector<float> t = [] {
        std::vector<float> v(256);
        for (int i = 0; i < 256; ++i) v[i] = fp8_e4m3_to_float((uint8_t)i);
        return v;
    }();
    return t.data();
}

float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u)               // NaN: quiet, keep sign
        return (uint16_t)((u >> 16) | 0x0040u);
    u += 0x7FFFu + ((u >> 16) & 1u);                    // round to nearest even
    return (uint16_t)(u >> 16);
}

static inline float bf16r(float f) { return bf16_to_f32(f32_to_bf16(f)); }

static inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

// Matches the GPU epilogue exactly:
//   __float2bfloat16_rn(up) * __float2bfloat16_rn(silu(gate))   (bf16 __hmul)
static inline float swiglu_bf16(float gate, float up) {
    return bf16r(bf16r(up) * bf16r(silu(gate)));
}

// Scale-row index — identical integer arithmetic to mat-ops.cu
// (`s_col = n * scale_N / N`).
static inline int scale_row_index(int n, int scale_rows, int N) {
    return (int)(((int64_t)n * scale_rows) / N);
}

// ===========================================================================
// SIMD FP8 -> FP32 decode
// ---------------------------------------------------------------------------
// E4M3 byte  s|eeee|mmm  is placed straight into an fp32 as
//   sign<<31 | e<<23 | m<<20   ==  (1 + m/8) * 2^(e - 127)      (e >= 1)
// and multiplying by 2^120 yields (1 + m/8) * 2^(e - 7): the true value.
// e == 0 lands on an fp32 subnormal (m * 2^-129); * 2^120 = m * 2^-9, exact.
// Sole caveat: the NaN encodings (0x7F/0xFF) decode to ±480 — impossible in
// a valid weight checkpoint, and asserted against at registration if desired.
// ===========================================================================

// CRITICAL: lanes with e == 0 must never flow through fp32-subnormal
// arithmetic — denormal microcode assists (~100 cycles, triggered when ANY
// lane is denormal) would crater throughput. Normal lanes use the exponent
// trick; subnormal lanes are rebuilt exactly as m * 2^-9 via int->float
// (always fp32-normal or zero); the two disjoint paths are OR-blended.
__attribute__((target("avx2,fma"), always_inline))
static inline __m256 decode8_avx2(const uint8_t* p) {
    const __m128i b    = _mm_loadl_epi64((const __m128i*)p);
    const __m256i w32  = _mm256_cvtepu8_epi32(b);
    const __m256i sgn  = _mm256_slli_epi32(_mm256_and_si256(w32, _mm256_set1_epi32(0x80)), 24);
    const __m256i mag  = _mm256_slli_epi32(_mm256_and_si256(w32, _mm256_set1_epi32(0x7F)), 20);
    const __m256i sub  = _mm256_cmpeq_epi32(_mm256_and_si256(w32, _mm256_set1_epi32(0x78)),
                                            _mm256_setzero_si256());       // e == 0 lanes
    // normal path: zero the sub lanes BEFORE the multiply (0 * 2^120 = 0)
    const __m256i nbits = _mm256_andnot_si256(sub, _mm256_or_si256(sgn, mag));
    const __m256  vnorm = _mm256_mul_ps(_mm256_castsi256_ps(nbits), _mm256_set1_ps(0x1p120f));
    // subnormal path: m * 2^-9, sign OR'd into the bit pattern (exact)
    const __m256  msub  = _mm256_mul_ps(
        _mm256_cvtepi32_ps(_mm256_and_si256(w32, _mm256_set1_epi32(0x7))),
        _mm256_set1_ps(0x1p-9f));
    const __m256i sbits = _mm256_and_si256(sub, _mm256_or_si256(_mm256_castps_si256(msub), sgn));
    return _mm256_castsi256_ps(_mm256_or_si256(_mm256_castps_si256(vnorm), sbits));
}

__attribute__((target("avx2"), always_inline))
static inline float hsum256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

__attribute__((target("avx512f"), always_inline))
static inline __m512 decode16_avx512(const uint8_t* p) {
    const __m128i   b    = _mm_loadu_si128((const __m128i*)p);
    const __m512i   w32  = _mm512_cvtepu8_epi32(b);
    const __m512i   sgn  = _mm512_slli_epi32(_mm512_and_si512(w32, _mm512_set1_epi32(0x80)), 24);
    const __m512i   mag  = _mm512_slli_epi32(_mm512_and_si512(w32, _mm512_set1_epi32(0x7F)), 20);
    const __mmask16 sub  = _mm512_cmpeq_epi32_mask(_mm512_and_si512(w32, _mm512_set1_epi32(0x78)),
                                                   _mm512_setzero_si512());  // e == 0 lanes
    const __m512i nbits = _mm512_maskz_or_epi32((__mmask16)~sub, sgn, mag);  // sub lanes -> 0
    const __m512  vnorm = _mm512_mul_ps(_mm512_castsi512_ps(nbits), _mm512_set1_ps(0x1p120f));
    const __m512  msub  = _mm512_mul_ps(
        _mm512_cvtepi32_ps(_mm512_and_si512(w32, _mm512_set1_epi32(0x7))),
        _mm512_set1_ps(0x1p-9f));
    const __m512i sbits = _mm512_maskz_or_epi32(sub, _mm512_castps_si512(msub), sgn);
    return _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(vnorm), sbits));
}

// ===========================================================================
// Stage 1 kernels: fused gate/up rows.
// For output rows n in [n0, n1) of the intermediate dimension I, and MR
// tokens whose f32 activation rows are xrow[0..MR), compute
//   h[t][n] = swiglu_bf16( dot(w_gate[n], x_t), dot(w_up[n], x_t) )
// with FP8 rows streamed ONCE per n and reused across all MR tokens.
// ===========================================================================

// ---------------------------- scalar ------------------------------------- //
template<int MR>
static void gateup_rows_scalar(const ExpertWeights& w,
                               const float* const* xrow, float* const* hrow,
                               int n0, int n1) {
    const float* lut = fp8_lut();
    const int K = w.H, N = w.I;
    const int sK = w.sg_cols;
    for (int n = n0; n < n1; ++n) {
        const uint8_t* wg = w.w_gate + (size_t)n * K;
        const uint8_t* wu = w.w_up   + (size_t)n * K;
        const int srow = scale_row_index(n, w.sg_rows, N);
        const float* sg = w.s_gate + (size_t)srow * sK;
        const float* su = w.s_up   + (size_t)srow * sK;
        float accg[MR] = {}, accu[MR] = {};
        for (int kg = 0; kg < sK; ++kg) {
            const int kbeg = kg * kGroupK;
            if (kbeg >= K) break;
            const int kend = std::min(kbeg + kGroupK, K);
            float pg[MR] = {}, pu[MR] = {};
            for (int k = kbeg; k < kend; ++k) {
                const float wgv = lut[wg[k]], wuv = lut[wu[k]];
                for (int t = 0; t < MR; ++t) {
                    pg[t] += wgv * xrow[t][k];
                    pu[t] += wuv * xrow[t][k];
                }
            }
            for (int t = 0; t < MR; ++t) {
                accg[t] += pg[t] * sg[kg];
                accu[t] += pu[t] * su[kg];
            }
        }
        for (int t = 0; t < MR; ++t)
            hrow[t][n] = swiglu_bf16(accg[t], accu[t]);
    }
}

// ----------------------------- AVX2 -------------------------------------- //
// UK = independent 8-wide FMA chains per matrix per token (latency hiding).
template<int MR>
__attribute__((target("avx2,fma")))
static void gateup_rows_avx2(const ExpertWeights& w,
                             const float* const* xrow, float* const* hrow,
                             int n0, int n1) {
    constexpr int UK   = (MR == 1) ? 2 : 1;
    constexpr int STEP = 8 * UK;
    const float* lut = fp8_lut();
    const int K = w.H, N = w.I;
    const int sK = w.sg_cols;

    for (int n = n0; n < n1; ++n) {
        const uint8_t* wg = w.w_gate + (size_t)n * K;
        const uint8_t* wu = w.w_up   + (size_t)n * K;
        const int srow = scale_row_index(n, w.sg_rows, N);
        const float* sg = w.s_gate + (size_t)srow * sK;
        const float* su = w.s_up   + (size_t)srow * sK;

        __m256 accg[MR], accu[MR];
        float  sacg[MR] = {}, sacu[MR] = {};             // scalar-tail accum
        for (int t = 0; t < MR; ++t) { accg[t] = _mm256_setzero_ps(); accu[t] = _mm256_setzero_ps(); }

        for (int kg = 0; kg < sK; ++kg) {
            const int kbeg = kg * kGroupK;
            if (kbeg >= K) break;
            const int kend = std::min(kbeg + kGroupK, K);

            __m256 gg[MR][UK], gu[MR][UK];
            for (int t = 0; t < MR; ++t)
                for (int u = 0; u < UK; ++u) { gg[t][u] = _mm256_setzero_ps(); gu[t][u] = _mm256_setzero_ps(); }

            int k = kbeg;
            for (; k + STEP <= kend; k += STEP) {
                __m256 wgf[UK], wuf[UK];
                for (int u = 0; u < UK; ++u) {
                    wgf[u] = decode8_avx2(wg + k + 8 * u);
                    wuf[u] = decode8_avx2(wu + k + 8 * u);
                }
                for (int t = 0; t < MR; ++t)
                    for (int u = 0; u < UK; ++u) {
                        const __m256 xv = _mm256_loadu_ps(xrow[t] + k + 8 * u);
                        gg[t][u] = _mm256_fmadd_ps(wgf[u], xv, gg[t][u]);
                        gu[t][u] = _mm256_fmadd_ps(wuf[u], xv, gu[t][u]);
                    }
            }
            for (; k + 8 <= kend; k += 8) {              // 8-wide tail
                const __m256 wgf = decode8_avx2(wg + k);
                const __m256 wuf = decode8_avx2(wu + k);
                for (int t = 0; t < MR; ++t) {
                    const __m256 xv = _mm256_loadu_ps(xrow[t] + k);
                    gg[t][0] = _mm256_fmadd_ps(wgf, xv, gg[t][0]);
                    gu[t][0] = _mm256_fmadd_ps(wuf, xv, gu[t][0]);
                }
            }
            float tg[MR] = {}, tu[MR] = {};              // scalar tail (<8)
            for (; k < kend; ++k) {
                const float wgv = lut[wg[k]], wuv = lut[wu[k]];
                for (int t = 0; t < MR; ++t) { tg[t] += wgv * xrow[t][k]; tu[t] += wuv * xrow[t][k]; }
            }

            const __m256 vsg = _mm256_set1_ps(sg[kg]);
            const __m256 vsu = _mm256_set1_ps(su[kg]);
            for (int t = 0; t < MR; ++t) {
                __m256 gsum = gg[t][0], usum = gu[t][0];
                for (int u = 1; u < UK; ++u) { gsum = _mm256_add_ps(gsum, gg[t][u]); usum = _mm256_add_ps(usum, gu[t][u]); }
                accg[t] = _mm256_fmadd_ps(gsum, vsg, accg[t]);
                accu[t] = _mm256_fmadd_ps(usum, vsu, accu[t]);
                sacg[t] += tg[t] * sg[kg];
                sacu[t] += tu[t] * su[kg];
            }
        }
        for (int t = 0; t < MR; ++t)
            hrow[t][n] = swiglu_bf16(hsum256(accg[t]) + sacg[t],
                                     hsum256(accu[t]) + sacu[t]);
    }
}

// ---------------------------- AVX-512 ------------------------------------ //
template<int MR>
__attribute__((target("avx512f")))
static void gateup_rows_avx512(const ExpertWeights& w,
                               const float* const* xrow, float* const* hrow,
                               int n0, int n1) {
    constexpr int UK   = (MR == 1) ? 2 : ((MR == 4) ? 2 : 1);
    constexpr int STEP = 16 * UK;
    const float* lut = fp8_lut();
    const int K = w.H, N = w.I;
    const int sK = w.sg_cols;

    for (int n = n0; n < n1; ++n) {
        const uint8_t* wg = w.w_gate + (size_t)n * K;
        const uint8_t* wu = w.w_up   + (size_t)n * K;
        const int srow = scale_row_index(n, w.sg_rows, N);
        const float* sg = w.s_gate + (size_t)srow * sK;
        const float* su = w.s_up   + (size_t)srow * sK;

        __m512 accg[MR], accu[MR];
        float  sacg[MR] = {}, sacu[MR] = {};
        for (int t = 0; t < MR; ++t) { accg[t] = _mm512_setzero_ps(); accu[t] = _mm512_setzero_ps(); }

        for (int kg = 0; kg < sK; ++kg) {
            const int kbeg = kg * kGroupK;
            if (kbeg >= K) break;
            const int kend = std::min(kbeg + kGroupK, K);

            __m512 gg[MR][UK], gu[MR][UK];
            for (int t = 0; t < MR; ++t)
                for (int u = 0; u < UK; ++u) { gg[t][u] = _mm512_setzero_ps(); gu[t][u] = _mm512_setzero_ps(); }

            int k = kbeg;
            for (; k + STEP <= kend; k += STEP) {
                __m512 wgf[UK], wuf[UK];
                for (int u = 0; u < UK; ++u) {
                    wgf[u] = decode16_avx512(wg + k + 16 * u);
                    wuf[u] = decode16_avx512(wu + k + 16 * u);
                }
                for (int t = 0; t < MR; ++t)
                    for (int u = 0; u < UK; ++u) {
                        const __m512 xv = _mm512_loadu_ps(xrow[t] + k + 16 * u);
                        gg[t][u] = _mm512_fmadd_ps(wgf[u], xv, gg[t][u]);
                        gu[t][u] = _mm512_fmadd_ps(wuf[u], xv, gu[t][u]);
                    }
            }
            for (; k + 16 <= kend; k += 16) {
                const __m512 wgf = decode16_avx512(wg + k);
                const __m512 wuf = decode16_avx512(wu + k);
                for (int t = 0; t < MR; ++t) {
                    const __m512 xv = _mm512_loadu_ps(xrow[t] + k);
                    gg[t][0] = _mm512_fmadd_ps(wgf, xv, gg[t][0]);
                    gu[t][0] = _mm512_fmadd_ps(wuf, xv, gu[t][0]);
                }
            }
            float tg[MR] = {}, tu[MR] = {};
            for (; k < kend; ++k) {
                const float wgv = lut[wg[k]], wuv = lut[wu[k]];
                for (int t = 0; t < MR; ++t) { tg[t] += wgv * xrow[t][k]; tu[t] += wuv * xrow[t][k]; }
            }

            const __m512 vsg = _mm512_set1_ps(sg[kg]);
            const __m512 vsu = _mm512_set1_ps(su[kg]);
            for (int t = 0; t < MR; ++t) {
                __m512 gsum = gg[t][0], usum = gu[t][0];
                for (int u = 1; u < UK; ++u) { gsum = _mm512_add_ps(gsum, gg[t][u]); usum = _mm512_add_ps(usum, gu[t][u]); }
                accg[t] = _mm512_fmadd_ps(gsum, vsg, accg[t]);
                accu[t] = _mm512_fmadd_ps(usum, vsu, accu[t]);
                sacg[t] += tg[t] * sg[kg];
                sacu[t] += tu[t] * su[kg];
            }
        }
        for (int t = 0; t < MR; ++t)
            hrow[t][n] = swiglu_bf16(_mm512_reduce_add_ps(accg[t]) + sacg[t],
                                     _mm512_reduce_add_ps(accu[t]) + sacu[t]);
    }
}

// ===========================================================================
// Stage 2 kernels: down projection rows.
//   y[t][n] = bf16( sum_k w_down[n,k] * h[t][k] * scale )    for n in [n0,n1)
// ===========================================================================

template<int MR>
static void down_rows_scalar(const ExpertWeights& w,
                             const float* const* hrow, uint16_t* const* yrow,
                             int n0, int n1) {
    const float* lut = fp8_lut();
    const int K = w.I, N = w.H;
    const int sK = w.sd_cols;
    for (int n = n0; n < n1; ++n) {
        const uint8_t* wd = w.w_down + (size_t)n * K;
        const int srow = scale_row_index(n, w.sd_rows, N);
        const float* sd = w.s_down + (size_t)srow * sK;
        float acc[MR] = {};
        for (int kg = 0; kg < sK; ++kg) {
            const int kbeg = kg * kGroupK;
            if (kbeg >= K) break;
            const int kend = std::min(kbeg + kGroupK, K);
            float p[MR] = {};
            for (int k = kbeg; k < kend; ++k) {
                const float wv = lut[wd[k]];
                for (int t = 0; t < MR; ++t) p[t] += wv * hrow[t][k];
            }
            for (int t = 0; t < MR; ++t) acc[t] += p[t] * sd[kg];
        }
        for (int t = 0; t < MR; ++t) yrow[t][n] = f32_to_bf16(acc[t]);
    }
}

template<int MR>
__attribute__((target("avx2,fma")))
static void down_rows_avx2(const ExpertWeights& w,
                           const float* const* hrow, uint16_t* const* yrow,
                           int n0, int n1) {
    constexpr int UK   = (MR == 1) ? 4 : ((MR == 2) ? 2 : 1);
    constexpr int STEP = 8 * UK;
    const float* lut = fp8_lut();
    const int K = w.I, N = w.H;
    const int sK = w.sd_cols;

    for (int n = n0; n < n1; ++n) {
        const uint8_t* wd = w.w_down + (size_t)n * K;
        const int srow = scale_row_index(n, w.sd_rows, N);
        const float* sd = w.s_down + (size_t)srow * sK;

        __m256 acc[MR];
        float  sac[MR] = {};
        for (int t = 0; t < MR; ++t) acc[t] = _mm256_setzero_ps();

        for (int kg = 0; kg < sK; ++kg) {
            const int kbeg = kg * kGroupK;
            if (kbeg >= K) break;
            const int kend = std::min(kbeg + kGroupK, K);

            __m256 g[MR][UK];
            for (int t = 0; t < MR; ++t)
                for (int u = 0; u < UK; ++u) g[t][u] = _mm256_setzero_ps();

            int k = kbeg;
            for (; k + STEP <= kend; k += STEP) {
                __m256 wf[UK];
                for (int u = 0; u < UK; ++u) wf[u] = decode8_avx2(wd + k + 8 * u);
                for (int t = 0; t < MR; ++t)
                    for (int u = 0; u < UK; ++u) {
                        const __m256 hv = _mm256_loadu_ps(hrow[t] + k + 8 * u);
                        g[t][u] = _mm256_fmadd_ps(wf[u], hv, g[t][u]);
                    }
            }
            for (; k + 8 <= kend; k += 8) {
                const __m256 wf = decode8_avx2(wd + k);
                for (int t = 0; t < MR; ++t) {
                    const __m256 hv = _mm256_loadu_ps(hrow[t] + k);
                    g[t][0] = _mm256_fmadd_ps(wf, hv, g[t][0]);
                }
            }
            float tp[MR] = {};
            for (; k < kend; ++k) {
                const float wv = lut[wd[k]];
                for (int t = 0; t < MR; ++t) tp[t] += wv * hrow[t][k];
            }

            const __m256 vs = _mm256_set1_ps(sd[kg]);
            for (int t = 0; t < MR; ++t) {
                __m256 gs = g[t][0];
                for (int u = 1; u < UK; ++u) gs = _mm256_add_ps(gs, g[t][u]);
                acc[t] = _mm256_fmadd_ps(gs, vs, acc[t]);
                sac[t] += tp[t] * sd[kg];
            }
        }
        for (int t = 0; t < MR; ++t)
            yrow[t][n] = f32_to_bf16(hsum256(acc[t]) + sac[t]);
    }
}

template<int MR>
__attribute__((target("avx512f")))
static void down_rows_avx512(const ExpertWeights& w,
                             const float* const* hrow, uint16_t* const* yrow,
                             int n0, int n1) {
    constexpr int UK   = (MR == 1) ? 4 : 2;
    constexpr int STEP = 16 * UK;
    const float* lut = fp8_lut();
    const int K = w.I, N = w.H;
    const int sK = w.sd_cols;

    for (int n = n0; n < n1; ++n) {
        const uint8_t* wd = w.w_down + (size_t)n * K;
        const int srow = scale_row_index(n, w.sd_rows, N);
        const float* sd = w.s_down + (size_t)srow * sK;

        __m512 acc[MR];
        float  sac[MR] = {};
        for (int t = 0; t < MR; ++t) acc[t] = _mm512_setzero_ps();

        for (int kg = 0; kg < sK; ++kg) {
            const int kbeg = kg * kGroupK;
            if (kbeg >= K) break;
            const int kend = std::min(kbeg + kGroupK, K);

            __m512 g[MR][UK];
            for (int t = 0; t < MR; ++t)
                for (int u = 0; u < UK; ++u) g[t][u] = _mm512_setzero_ps();

            int k = kbeg;
            for (; k + STEP <= kend; k += STEP) {
                __m512 wf[UK];
                for (int u = 0; u < UK; ++u) wf[u] = decode16_avx512(wd + k + 16 * u);
                for (int t = 0; t < MR; ++t)
                    for (int u = 0; u < UK; ++u) {
                        const __m512 hv = _mm512_loadu_ps(hrow[t] + k + 16 * u);
                        g[t][u] = _mm512_fmadd_ps(wf[u], hv, g[t][u]);
                    }
            }
            for (; k + 16 <= kend; k += 16) {
                const __m512 wf = decode16_avx512(wd + k);
                for (int t = 0; t < MR; ++t) {
                    const __m512 hv = _mm512_loadu_ps(hrow[t] + k);
                    g[t][0] = _mm512_fmadd_ps(wf, hv, g[t][0]);
                }
            }
            float tp[MR] = {};
            for (; k < kend; ++k) {
                const float wv = lut[wd[k]];
                for (int t = 0; t < MR; ++t) tp[t] += wv * hrow[t][k];
            }

            const __m512 vs = _mm512_set1_ps(sd[kg]);
            for (int t = 0; t < MR; ++t) {
                __m512 gs = g[t][0];
                for (int u = 1; u < UK; ++u) gs = _mm512_add_ps(gs, g[t][u]);
                acc[t] = _mm512_fmadd_ps(gs, vs, acc[t]);
                sac[t] += tp[t] * sd[kg];
            }
        }
        for (int t = 0; t < MR; ++t)
            yrow[t][n] = f32_to_bf16(_mm512_reduce_add_ps(acc[t]) + sac[t]);
    }
}

// ===========================================================================
// ISA dispatch
// ===========================================================================

using GateUpFn = void (*)(const ExpertWeights&, const float* const*, float* const*, int, int);
using DownFn   = void (*)(const ExpertWeights&, const float* const*, uint16_t* const*, int, int);

struct Kernels {
    GateUpFn gateup1, gateup4;
    DownFn   down1,   down4;
    const char* name;
};

static const Kernels& kernels() {
    static const Kernels k = [] {
        Kernels r{};
        // Optional cap for debugging/benchmarks: GARLIC_CPU_ISA=scalar|avx2|avx512
        const char* cap = std::getenv("GARLIC_CPU_ISA");
        const bool allow512 = !cap || std::strcmp(cap, "avx512") == 0;
        const bool allow2   = !cap || allow512 || std::strcmp(cap, "avx2") == 0;
#if defined(__x86_64__) || defined(_M_X64)
        __builtin_cpu_init();
        if (allow512 && __builtin_cpu_supports("avx512f")) {
            r = { gateup_rows_avx512<1>, gateup_rows_avx512<4>,
                  down_rows_avx512<1>,   down_rows_avx512<4>, "avx512" };
            return r;
        }
        if (allow2 && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
            r = { gateup_rows_avx2<1>, gateup_rows_avx2<4>,
                  down_rows_avx2<1>,   down_rows_avx2<4>, "avx2" };
            return r;
        }
#endif
        r = { gateup_rows_scalar<1>, gateup_rows_scalar<4>,
              down_rows_scalar<1>,   down_rows_scalar<4>, "scalar" };
        return r;
    }();
    return k;
}

const char* active_isa() { return kernels().name; }

// ===========================================================================
// Chunk runners: apply a kernel over a row range for all T tokens, tiling
// tokens by MR=4 (weights streamed once per 4 tokens) with an MR=1 remainder.
// ===========================================================================

static void run_gateup_chunk(const ExpertJob& j, int n0, int n1) {
    const Kernels& k = kernels();
    const int H = j.w->H, I = j.w->I;
    int t = 0;
    for (; t + 4 <= j.T; t += 4) {
        const float* xr[4]; float* hr[4];
        for (int i = 0; i < 4; ++i) { xr[i] = j.xf + (size_t)(t + i) * H; hr[i] = j.h + (size_t)(t + i) * I; }
        k.gateup4(*j.w, xr, hr, n0, n1);
    }
    for (; t < j.T; ++t) {
        const float* xr[1] = { j.xf + (size_t)t * H };
        float*       hr[1] = { j.h  + (size_t)t * I };
        k.gateup1(*j.w, xr, hr, n0, n1);
    }
}

static void run_down_chunk(const ExpertJob& j, int n0, int n1) {
    const Kernels& k = kernels();
    const int H = j.w->H, I = j.w->I;
    int t = 0;
    for (; t + 4 <= j.T; t += 4) {
        const float* hr[4]; uint16_t* yr[4];
        for (int i = 0; i < 4; ++i) { hr[i] = j.h + (size_t)(t + i) * I; yr[i] = j.y_bf16 + (size_t)(t + i) * H; }
        k.down4(*j.w, hr, yr, n0, n1);
    }
    for (; t < j.T; ++t) {
        const float* hr[1] = { j.h + (size_t)t * I };
        uint16_t*    yr[1] = { j.y_bf16 + (size_t)t * H };
        k.down1(*j.w, hr, yr, n0, n1);
    }
}

// ===========================================================================
// Single-shot entry points
// ===========================================================================

void expert_ffn_fp8(const ExpertWeights& w,
                    const uint16_t* x_bf16, uint16_t* y_bf16, int T,
                    float* xf, float* h) {
    assert(w.valid());
    assert((int64_t)w.sg_cols * kGroupK >= w.H && (int64_t)w.sd_cols * kGroupK >= w.I);
    for (size_t i = 0; i < (size_t)T * w.H; ++i) xf[i] = bf16_to_f32(x_bf16[i]);
    ExpertJob j;
    j.w = &w; j.x_bf16 = x_bf16; j.y_bf16 = y_bf16; j.xf = xf; j.h = h; j.T = T;
    run_gateup_chunk(j, 0, w.I);
    run_down_chunk(j, 0, w.H);
}

void expert_ffn_fp8_ref(const ExpertWeights& w,
                        const uint16_t* x_bf16, uint16_t* y_bf16, int T) {
    const int H = w.H, I = w.I;
    std::vector<double> xf((size_t)H), h((size_t)I);
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < H; ++k) xf[k] = (double)bf16_to_f32(x_bf16[(size_t)t * H + k]);
        for (int n = 0; n < I; ++n) {
            const uint8_t* wg = w.w_gate + (size_t)n * H;
            const uint8_t* wu = w.w_up   + (size_t)n * H;
            const int srow = scale_row_index(n, w.sg_rows, I);
            double ag = 0, au = 0;
            for (int kg = 0; kg < w.sg_cols; ++kg) {
                const int kbeg = kg * kGroupK;
                if (kbeg >= H) break;
                const int kend = std::min(kbeg + kGroupK, H);
                double pg = 0, pu = 0;
                for (int k = kbeg; k < kend; ++k) {
                    pg += (double)fp8_e4m3_to_float(wg[k]) * xf[k];
                    pu += (double)fp8_e4m3_to_float(wu[k]) * xf[k];
                }
                ag += pg * (double)w.s_gate[(size_t)srow * w.sg_cols + kg];
                au += pu * (double)w.s_up  [(size_t)srow * w.sg_cols + kg];
            }
            h[n] = (double)swiglu_bf16((float)ag, (float)au);
        }
        for (int n = 0; n < H; ++n) {
            const uint8_t* wd = w.w_down + (size_t)n * I;
            const int srow = scale_row_index(n, w.sd_rows, H);
            double a = 0;
            for (int kg = 0; kg < w.sd_cols; ++kg) {
                const int kbeg = kg * kGroupK;
                if (kbeg >= I) break;
                const int kend = std::min(kbeg + kGroupK, I);
                double p = 0;
                for (int k = kbeg; k < kend; ++k)
                    p += (double)fp8_e4m3_to_float(wd[k]) * h[k];
                a += p * (double)w.s_down[(size_t)srow * w.sd_cols + kg];
            }
            y_bf16[(size_t)t * H + n] = f32_to_bf16((float)a);
        }
    }
}

// ===========================================================================
// ExpertExecutor — two-stage chunked fork-join with cross-job stealing.
// All scheduling state is mutated under one mutex; the compute itself runs
// unlocked. Chunks are >= tens of microseconds, so lock traffic is noise.
// ===========================================================================

struct ExpertExecutor::Impl {
    std::mutex mu;
    std::condition_variable cv;        // workers: "work may be available"
    std::condition_variable cv_idle;   // wait_all(): "everything finished"
    std::deque<ExpertJob*> active;     // jobs that still have unclaimed chunks
    std::vector<std::thread> workers;
    int pending = 0;                   // submitted, not yet finished
    bool stop = false;

    static constexpr int kChunkRows = 64;

    struct Work { ExpertJob* j; int stage; int idx; };

    // Caller holds mu.
    bool find_work(Work& out) {
        for (size_t i = 0; i < active.size(); ) {
            ExpertJob* j = active[i];
            if (j->n1_claim < j->c1) {
                out = { j, 1, j->n1_claim++ };
                return true;
            }
            if (j->n1_done == j->c1 && j->n2_claim < j->c2) {
                out = { j, 2, j->n2_claim++ };
                if (j->n2_claim == j->c2)                 // fully claimed
                    active.erase(active.begin() + i);
                return true;
            }
            ++i;                                          // stage-1 in flight elsewhere
        }
        return false;
    }

    void run(const Work& w) {
        const ExpertJob& j = *w.j;
        if (w.stage == 1) {
            const int n0 = w.idx * j.chunk1;
            run_gateup_chunk(j, n0, std::min(n0 + j.chunk1, j.w->I));
        } else {
            const int n0 = w.idx * j.chunk2;
            run_down_chunk(j, n0, std::min(n0 + j.chunk2, j.w->H));
        }
    }

    // Caller holds mu.
    void mark_done(const Work& w) {
        ExpertJob* j = w.j;
        if (w.stage == 1) {
            if (++j->n1_done == j->c1)
                cv.notify_all();                          // stage 2 unlocked
        } else {
            if (++j->n2_done == j->c2) {
                j->finished.store(1, std::memory_order_release);
                if (--pending == 0) cv_idle.notify_all();
            }
        }
    }

    void worker_loop() {
        std::unique_lock<std::mutex> lk(mu);
        while (true) {
            Work w;
            if (find_work(w)) {
                lk.unlock();
                run(w);
                lk.lock();
                mark_done(w);
                continue;
            }
            if (stop) return;
            cv.wait(lk);
        }
    }
};

ExpertExecutor::ExpertExecutor(int nthreads, int reserve) {
    if (nthreads <= 0) {
        const int hw = (int)std::thread::hardware_concurrency();
        nthreads = std::max(1, hw - std::max(0, reserve));
    }
    nthreads_ = nthreads;
    impl_ = new Impl();
    impl_->workers.reserve(nthreads);
    for (int i = 0; i < nthreads; ++i)
        impl_->workers.emplace_back([this] { impl_->worker_loop(); });
}

ExpertExecutor::~ExpertExecutor() {
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->stop = true;
    }
    impl_->cv.notify_all();
    for (auto& t : impl_->workers) t.join();
    delete impl_;
}

void ExpertExecutor::submit(ExpertJob& job) {
    assert(job.w && job.w->valid() && job.xf && job.h && job.x_bf16 && job.y_bf16);
    job.reset();
    if (job.T <= 0) {
        job.finished.store(1, std::memory_order_release);
        return;
    }
    // bf16 -> f32 on the submitting thread: T*H elements, microseconds.
    const size_t nx = (size_t)job.T * job.w->H;
    for (size_t i = 0; i < nx; ++i) job.xf[i] = bf16_to_f32(job.x_bf16[i]);

    job.chunk1 = Impl::kChunkRows;
    job.chunk2 = Impl::kChunkRows;
    job.c1 = (job.w->I + job.chunk1 - 1) / job.chunk1;
    job.c2 = (job.w->H + job.chunk2 - 1) / job.chunk2;

    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->active.push_back(&job);
        ++impl_->pending;
    }
    impl_->cv.notify_all();
}

void ExpertExecutor::wait_all() {
    std::unique_lock<std::mutex> lk(impl_->mu);
    impl_->cv_idle.wait(lk, [this] { return impl_->pending == 0; });
}

} // namespace garlic_cpu
