#include "ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>

#include "lb_common.h"
#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#include <immintrin.h>
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace lb {

#if defined(__x86_64__) || defined(_M_X64)
bool cpu_has_f16c_fma() {
    static const bool v = [] {
        unsigned a, b, c, d;
        if (!__get_cpuid(1, &a, &b, &c, &d)) return false;
        bool avx = (c & (1u << 28)) && (c & (1u << 27));
        bool fma = c & (1u << 12), f16c = c & (1u << 29);
        if (!avx) return false;
        unsigned xlo, xhi;
        __asm__ volatile("xgetbv" : "=a"(xlo), "=d"(xhi) : "c"(0));
        return fma && f16c && ((xlo & 6) == 6);
    }();
    return v;
}
#else
bool cpu_has_f16c_fma() { return false; }
#endif

void rmsnorm(const float* x, const float* w, int n, float eps, float* out) {
    float ss = 0.f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    const float inv = 1.0f / std::sqrt(ss / (float)n + eps);
    for (int i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

void rope_inplace(float* v, int head_dim, const float* cs, const float* sn) {
    const int h = head_dim / 2;
    for (int i = 0; i < h; i++) {
        float a = v[i], b = v[i + h];
        v[i] = a * cs[i] - b * sn[i];
        v[i + h] = b * cs[i] + a * sn[i];
    }
}

#if defined(__x86_64__) || defined(_M_X64)
// Vectorised expf (Cephes polynomial, ~1 ulp): the scalar libm expf made SiLU cost ~10 ms per decoded token.
__attribute__((target("avx2,fma")))
static inline __m256 exp256(__m256 x) {
    x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949f));
    __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
    __m256 z = _mm256_mul_ps(x, x);
    __m256 y = _mm256_set1_ps(1.9875691500E-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894E-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201E-1f));
    y = _mm256_fmadd_ps(y, z, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.0f));
    __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(e));
}
__attribute__((target("avx2,fma")))
static void silu_mul_avx2(const float* gate, const float* up, int n, float* out) {
    int i = 0;
    const __m256 one = _mm256_set1_ps(1.0f), zero = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 d = _mm256_add_ps(one, exp256(_mm256_sub_ps(zero, g)));
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_div_ps(g, d), _mm256_loadu_ps(up + i)));
    }
    for (; i < n; i++) {
        float g = gate[i];
        out[i] = (g / (1.0f + std::exp(-g))) * up[i];
    }
}
#endif

#if defined(__aarch64__)

static inline float32x4_t exp_neon(float32x4_t x) {

    x = vminq_f32(x, vdupq_n_f32(88.3762626647949f));

    x = vmaxq_f32(x, vdupq_n_f32(-88.3762626647949f));

    float32x4_t fx = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(1.44269504088896341f)));

    x = vfmsq_f32(x, fx, vdupq_n_f32(0.693359375f));

    x = vfmsq_f32(x, fx, vdupq_n_f32(-2.12194440e-4f));

    float32x4_t z = vmulq_f32(x, x);

    float32x4_t y = vdupq_n_f32(1.9875691500E-4f);

    y = vfmaq_f32(vdupq_n_f32(1.3981999507E-3f), y, x);

    y = vfmaq_f32(vdupq_n_f32(8.3334519073E-3f), y, x);

    y = vfmaq_f32(vdupq_n_f32(4.1665795894E-2f), y, x);

    y = vfmaq_f32(vdupq_n_f32(1.6666665459E-1f), y, x);

    y = vfmaq_f32(vdupq_n_f32(5.0000001201E-1f), y, x);

    y = vfmaq_f32(x, y, z);

    y = vaddq_f32(y, vdupq_n_f32(1.0f));

    int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtnq_s32_f32(fx), vdupq_n_s32(127)), 23);

    return vmulq_f32(y, vreinterpretq_f32_s32(e));

}

static void silu_mul_neon(const float* gate, const float* up, int n, float* out) {

    int i = 0;

    const float32x4_t one = vdupq_n_f32(1.0f);

    for (; i + 4 <= n; i += 4) {

        float32x4_t g = vld1q_f32(gate + i);

        float32x4_t d = vaddq_f32(one, exp_neon(vnegq_f32(g)));

        vst1q_f32(out + i, vmulq_f32(vdivq_f32(g, d), vld1q_f32(up + i)));

    }

    for (; i < n; i++) {

        float g = gate[i];

        out[i] = (g / (1.0f + std::exp(-g))) * up[i];

    }

}

#endif



void silu_mul(const float* gate, const float* up, int n, float* out) {
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu_has_f16c_fma()) { silu_mul_avx2(gate, up, n, out); return; }
#endif
#if defined(__aarch64__)
    silu_mul_neon(gate, up, n, out);
    return;
#endif
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        out[i] = (g / (1.0f + std::exp(-g))) * up[i];
    }
}

void softmax_inplace(float* x, int n) {
    float m = x[0];
    for (int i = 1; i < n; i++) m = std::max(m, x[i]);
    float s = 0.f;
    for (int i = 0; i < n; i++) { x[i] = std::exp(x[i] - m); s += x[i]; }
    const float inv = 1.f / s;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

// ------------------------------------------------------------------ dot_f16
#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx2,fma,f16c")))
static float dot_f16_avx2(const float* x, const uint16_t* w, int n) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 w0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w + i)));
        __m256 w1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(w + i + 8)));
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), w0, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8), w1, acc1);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    alignas(32) float t[8];
    _mm256_store_ps(t, acc0);
    float s = ((t[0] + t[4]) + (t[1] + t[5])) + ((t[2] + t[6]) + (t[3] + t[7]));
    for (; i < n; i++) s += x[i] * f16_to_f32(w[i]);
    return s;
}
#endif

float dot_f16(const float* x, const uint16_t* w, int n) {
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu_has_f16c_fma()) return dot_f16_avx2(x, w, n);
#endif
#if defined(__aarch64__)
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        float16x8_t wh = vreinterpretq_f16_u16(vld1q_u16(w + i));
        a0 = vfmaq_f32(a0, vld1q_f32(x + i), vcvt_f32_f16(vget_low_f16(wh)));
        a1 = vfmaq_f32(a1, vld1q_f32(x + i + 4), vcvt_f32_f16(vget_high_f16(wh)));
    }
    float s = vaddvq_f32(vaddq_f32(a0, a1));
    for (; i < n; i++) s += x[i] * f16_to_f32(w[i]);
    return s;
#else
    float s = 0.f;
    for (int i = 0; i < n; i++) s += x[i] * f16_to_f32(w[i]);
    return s;
#endif
}

#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx2")))
static int32_t dot_i8_avx2(const int8_t* x, const int8_t* w, int n) {
    __m256i acc = _mm256_setzero_si256();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i a = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(x + i)));
        __m256i b = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(w + i)));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a, b));
    }
    __m128i lo = _mm256_castsi256_si128(acc), hi = _mm256_extracti128_si256(acc, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0x4E));
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0xB1));
    int32_t s = _mm_cvtsi128_si32(lo);
    for (; i < n; i++) s += (int32_t)x[i] * (int32_t)w[i];
    return s;
}
#endif

int32_t dot_i8(const int8_t* x, const int8_t* w, int n) {
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu_has_f16c_fma()) return dot_i8_avx2(x, w, n);
#endif
    int32_t s = 0;
    for (int i = 0; i < n; i++) s += (int32_t)x[i] * (int32_t)w[i];
    return s;
}



// ------------------------------------------------------------------ elementwise helpers
#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx2"))) static void vec_mul_avx2(const float* a, const float* b, int n, float* out) {
    int i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    for (; i < n; i++) out[i] = a[i] * b[i];
}
__attribute__((target("avx2"))) static void vec_mad_avx2(float* y, const float* u, const float* h, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_add_ps(_mm256_loadu_ps(y + i), _mm256_mul_ps(_mm256_loadu_ps(u + i), _mm256_loadu_ps(h + i))));
    for (; i < n; i++) y[i] += u[i] * h[i];
}
__attribute__((target("avx2"))) static void vec_acc_i32_avx2(float* o, const int32_t* a, float sc, int n) {
    const __m256 vs = _mm256_set1_ps(sc);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(a + i))), vs);
        _mm256_storeu_ps(o + i, _mm256_add_ps(_mm256_loadu_ps(o + i), v));
    }
    for (; i < n; i++) o[i] += (float)a[i] * sc;
}
#endif
void vec_mul(const float* a, const float* b, int n, float* out) {
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu_has_f16c_fma()) { vec_mul_avx2(a, b, n, out); return; }
#endif
    for (int i = 0; i < n; i++) out[i] = a[i] * b[i];
}
void vec_mul_inplace(float* a, const float* b, int n) { vec_mul(a, b, n, a); }
void vec_mad(float* y, const float* u, const float* h, int n) {
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu_has_f16c_fma()) { vec_mad_avx2(y, u, h, n); return; }
#endif
    for (int i = 0; i < n; i++) y[i] += u[i] * h[i];
}
void vec_acc_i32(float* o, const int32_t* a, float sc, int n) {
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu_has_f16c_fma()) { vec_acc_i32_avx2(o, a, sc, n); return; }
#endif
    for (int i = 0; i < n; i++) o[i] += (float)a[i] * sc;
}

// ------------------------------------------------------------------ head_dots_i8
#if defined(__x86_64__) || defined(_M_X64)
static bool cpu_has_avx512_vnni() {
    static const bool v = [] {
        unsigned a, b, c, d;
        if (!__get_cpuid(1, &a, &b, &c, &d)) return false;
        if (!(c & (1u << 27)) || !(c & (1u << 28))) return false;
        unsigned xlo, xhi;
        __asm__ volatile("xgetbv" : "=a"(xlo), "=d"(xhi) : "c"(0));
        if ((xlo & 0xe6) != 0xe6) return false;
        if (__get_cpuid_max(0, nullptr) < 7) return false;
        __cpuid_count(7, 0, a, b, c, d);
        return (b & (1u << 16)) && (b & (1u << 30)) && (c & (1u << 11));  // AVX512F, BW, VNNI
    }();
    return v;
}

template <int NG>
__attribute__((target("avx512f,avx512bw,avx512vnni")))
static void head_dots_vnni_group(const int8_t* w, int rows, int H, const uint8_t* xu, int np, int p0, int32_t* out) {
    int r = 0;
    for (; r + 4 <= rows; r += 4) {
        __m512i acc[4][NG];
        for (int i = 0; i < 4; i++) for (int p = 0; p < NG; p++) acc[i][p] = _mm512_setzero_si512();
        const int8_t* w0 = w + (size_t)r * H;
        for (int k = 0; k < H; k += 64) {
            const __m512i a0 = _mm512_loadu_si512(w0 + k), a1 = _mm512_loadu_si512(w0 + H + k);
            const __m512i a2 = _mm512_loadu_si512(w0 + 2 * (size_t)H + k), a3 = _mm512_loadu_si512(w0 + 3 * (size_t)H + k);
            for (int p = 0; p < NG; p++) {
                const __m512i xv = _mm512_loadu_si512(xu + (size_t)(p0 + p) * H + k);
                acc[0][p] = _mm512_dpbusd_epi32(acc[0][p], xv, a0);
                acc[1][p] = _mm512_dpbusd_epi32(acc[1][p], xv, a1);
                acc[2][p] = _mm512_dpbusd_epi32(acc[2][p], xv, a2);
                acc[3][p] = _mm512_dpbusd_epi32(acc[3][p], xv, a3);
            }
        }
        for (int i = 0; i < 4; i++) for (int p = 0; p < NG; p++) out[(size_t)(r + i) * np + p0 + p] = _mm512_reduce_add_epi32(acc[i][p]);
    }
    for (; r < rows; r++) {
        __m512i acc[NG];
        for (int p = 0; p < NG; p++) acc[p] = _mm512_setzero_si512();
        const int8_t* w0 = w + (size_t)r * H;
        for (int k = 0; k < H; k += 64) {
            const __m512i a0 = _mm512_loadu_si512(w0 + k);
            for (int p = 0; p < NG; p++) acc[p] = _mm512_dpbusd_epi32(acc[p], _mm512_loadu_si512(xu + (size_t)(p0 + p) * H + k), a0);
        }
        for (int p = 0; p < NG; p++) out[(size_t)r * np + p0 + p] = _mm512_reduce_add_epi32(acc[p]);
    }
}

__attribute__((target("avx2")))
static void head_dots_avx2(const int8_t* w, int rows, int H, const uint8_t* xu, int np, int32_t* out) {
    for (int r = 0; r < rows; r++) {
        const int8_t* wr = w + (size_t)r * H;
        for (int p = 0; p < np; p++) {
            const uint8_t* x = xu + (size_t)p * H;
            __m256i acc = _mm256_setzero_si256();
            for (int i = 0; i + 16 <= H; i += 16) {
                __m256i a = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(x + i)));
                __m256i b = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(wr + i)));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a, b));
            }
            __m128i lo = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
            lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0x4E));
            lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0xB1));
            int32_t s = _mm_cvtsi128_si32(lo);
            for (int i = H & ~15; i < H; i++) s += (int32_t)x[i] * (int32_t)wr[i];
            out[(size_t)r * np + p] = s;
        }
    }
}
#endif

void head_dots_i8(const int8_t* w, int rows, int H, const uint8_t* xu, int np, int32_t* out) {
#if defined(__aarch64__)
    if ((H % 16) == 0) {  // sdot on (xu ^ 0x80) as int8, then add back 128 * row sum so the result is sum(xu * w)
        const uint8x16_t flip = vdupq_n_u8(0x80);
        const int8x16_t ones = vdupq_n_s8(1);
        for (int r = 0; r < rows; r++) {
            const int8_t* wr = w + (size_t)r * H;
            int32x4_t sw4 = vdupq_n_s32(0);
            for (int k = 0; k < H; k += 16) sw4 = vdotq_s32(sw4, ones, vld1q_s8(wr + k));
            const int32_t sw = vaddvq_s32(sw4);
            for (int p = 0; p < np; p++) {
                const uint8_t* x = xu + (size_t)p * H;
                int32x4_t a = vdupq_n_s32(0);
                for (int k = 0; k < H; k += 16)
                    a = vdotq_s32(a, vreinterpretq_s8_u8(veorq_u8(vld1q_u8(x + k), flip)), vld1q_s8(wr + k));
                out[(size_t)r * np + p] = vaddvq_s32(a) + 128 * sw;
            }
        }
        return;
    }
#endif
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu_has_avx512_vnni() && (H % 64) == 0) {
        int p0 = 0;
        for (; p0 + 4 <= np; p0 += 4) head_dots_vnni_group<4>(w, rows, H, xu, np, p0, out);
        if (np - p0 == 3) head_dots_vnni_group<3>(w, rows, H, xu, np, p0, out);
        else if (np - p0 == 2) head_dots_vnni_group<2>(w, rows, H, xu, np, p0, out);
        else if (np - p0 == 1) head_dots_vnni_group<1>(w, rows, H, xu, np, p0, out);
        return;
    }
    if (cpu_has_f16c_fma()) { head_dots_avx2(w, rows, H, xu, np, out); return; }
#endif
    for (int r = 0; r < rows; r++)
        for (int p = 0; p < np; p++) {
            int32_t s = 0;
            for (int i = 0; i < H; i++) s += (int32_t)xu[(size_t)p * H + i] * (int32_t)w[(size_t)r * H + i];
            out[(size_t)r * np + p] = s;
        }
}

#if defined(__x86_64__) || defined(_M_X64)
template <int NP>
__attribute__((target("avx2")))
static void head_dots_s8_avx2_np(const int8_t* w, int rows, int H, const int8_t* xs, int np, int32_t* out) {
    const __m256i ones16 = _mm256_set1_epi16(1);
    for (int r = 0; r < rows; r++) {
        const int8_t* wr = w + (size_t)r * H;
        __m256i acc[NP];
        for (int p = 0; p < NP; p++) acc[p] = _mm256_setzero_si256();
        int k = 0;
        for (; k + 32 <= H; k += 32) {
            const __m256i wv = _mm256_loadu_si256((const __m256i*)(wr + k));
            const __m256i aw = _mm256_abs_epi8(wv);
            for (int p = 0; p < NP; p++) {
                const __m256i xv = _mm256_loadu_si256((const __m256i*)(xs + (size_t)p * H + k));
                acc[p] = _mm256_add_epi32(acc[p], _mm256_madd_epi16(_mm256_maddubs_epi16(aw, _mm256_sign_epi8(xv, wv)), ones16));
            }
        }
        for (int p = 0; p < NP; p++) {
            __m128i lo = _mm_add_epi32(_mm256_castsi256_si128(acc[p]), _mm256_extracti128_si256(acc[p], 1));
            lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0x4E));
            lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0xB1));
            int32_t s = _mm_cvtsi128_si32(lo);
            for (int i = k; i < H; i++) s += (int32_t)xs[(size_t)p * H + i] * (int32_t)wr[i];
            out[(size_t)r * np + p] = s;
        }
    }
}
#endif

bool head_dots_s8(const int8_t* w, int rows, int H, const int8_t* xs, int np, int32_t* out) {
#if defined(__x86_64__) || defined(_M_X64)
    if (!cpu_has_f16c_fma() || np < 1 || np > 8) return false;
    switch (np) {
        case 1: head_dots_s8_avx2_np<1>(w, rows, H, xs, np, out); break;
        case 2: head_dots_s8_avx2_np<2>(w, rows, H, xs, np, out); break;
        case 3: head_dots_s8_avx2_np<3>(w, rows, H, xs, np, out); break;
        case 4: head_dots_s8_avx2_np<4>(w, rows, H, xs, np, out); break;
        case 5: head_dots_s8_avx2_np<5>(w, rows, H, xs, np, out); break;
        case 6: head_dots_s8_avx2_np<6>(w, rows, H, xs, np, out); break;
        case 7: head_dots_s8_avx2_np<7>(w, rows, H, xs, np, out); break;
        default: head_dots_s8_avx2_np<8>(w, rows, H, xs, np, out); break;
    }
    return true;
#else
    return false;
#endif
}

#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx2,fma")))
static float dot_f32_i8_avx2(const float* x, const int8_t* q, int n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = a0;
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 q0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(q + i))));
        __m256 q1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(q + i + 8))));
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), q0, a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8), q1, a1);
    }
    a0 = _mm256_add_ps(a0, a1);
    alignas(32) float t[8];
    _mm256_store_ps(t, a0);
    float s = ((t[0] + t[4]) + (t[1] + t[5])) + ((t[2] + t[6]) + (t[3] + t[7]));
    for (; i < n; i++) s += x[i] * (float)q[i];
    return s;
}
#endif

#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx2,fma")))
static float dot_f32_i8_blocks_avx2(const float* x, const int8_t* q, const float* sc, int block, int n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = a0;
    for (int i = 0, b = 0, j = 0; i < n; i += 16) {
        if (j == block) { j = 0; b++; }
        j += 16;
        const __m256 s = _mm256_set1_ps(sc[b]);   // block % 16 == 0: both halves share the scale
        const __m256 q0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(q + i))));
        const __m256 q1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(q + i + 8))));
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), _mm256_mul_ps(q0, s), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + 8), _mm256_mul_ps(q1, s), a1);
    }
    a0 = _mm256_add_ps(a0, a1);
    alignas(32) float t[8];
    _mm256_store_ps(t, a0);
    return ((t[0] + t[4]) + (t[1] + t[5])) + ((t[2] + t[6]) + (t[3] + t[7]));
}
#endif

#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx2,fma")))
static void axpy_i8_blocks_avx2(float* y, float w, const int8_t* q, const float* sc, int block, int n) {
    for (int b = 0; b * block < n; b++) {
        const __m256 ws = _mm256_set1_ps(w * sc[b]);
        for (int i = b * block; i < (b + 1) * block; i += 8) {
            const __m256 qf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(q + i))));
            _mm256_storeu_ps(y + i, _mm256_fmadd_ps(qf, ws, _mm256_loadu_ps(y + i)));
        }
    }
}
#endif

void axpy_i8_blocks(float* y, float w, const int8_t* q, const float* sc, int block, int n) {
#if defined(__x86_64__) || defined(_M_X64)
    if ((block % 8) == 0 && (n % block) == 0 && cpu_has_f16c_fma()) { axpy_i8_blocks_avx2(y, w, q, sc, block, n); return; }
#endif
    for (int b = 0; b < n / block; b++) {
        const float ww = w * sc[b];
        for (int i = b * block; i < (b + 1) * block; i++) y[i] += ww * (float)q[i];
    }
}

float dot_f32_i8_blocks(const float* x, const int8_t* q, const float* sc, int block, int n) {
#if defined(__x86_64__) || defined(_M_X64)
    if ((block % 16) == 0 && (n % block) == 0 && cpu_has_f16c_fma()) return dot_f32_i8_blocks_avx2(x, q, sc, block, n);
#endif
    float s = 0.f;
    for (int b = 0; b < n / block; b++) s += dot_f32_i8(x + b * block, q + b * block, block) * sc[b];
    return s;
}

float dot_f32_i8(const float* x, const int8_t* q, int n) {
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu_has_f16c_fma()) return dot_f32_i8_avx2(x, q, n);
#endif
#if defined(__aarch64__)
    {
        float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
        int i = 0;
        for (; i + 16 <= n; i += 16) {
            int8x16_t v = vld1q_s8(q + i);
            int16x8_t l = vmovl_s8(vget_low_s8(v)), h = vmovl_s8(vget_high_s8(v));
            a0 = vfmaq_f32(a0, vld1q_f32(x + i), vcvtq_f32_s32(vmovl_s16(vget_low_s16(l))));
            a1 = vfmaq_f32(a1, vld1q_f32(x + i + 4), vcvtq_f32_s32(vmovl_s16(vget_high_s16(l))));
            a2 = vfmaq_f32(a2, vld1q_f32(x + i + 8), vcvtq_f32_s32(vmovl_s16(vget_low_s16(h))));
            a3 = vfmaq_f32(a3, vld1q_f32(x + i + 12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(h))));
        }
        float s = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
        for (; i < n; i++) s += x[i] * (float)q[i];
        return s;
    }
#endif
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += x[i] * (float)q[i];
        s1 += x[i + 1] * (float)q[i + 1];
        s2 += x[i + 2] * (float)q[i + 2];
        s3 += x[i + 3] * (float)q[i + 3];
    }
    float s = (s0 + s1) + (s2 + s3);
    for (; i < n; i++) s += x[i] * (float)q[i];
    return s;
}

// ------------------------------------------------------------------ grouped-query attention chunk (head_dim 128, int8 KV, 8 scales of 16 dims)
#if defined(__x86_64__) || defined(_M_X64)
// o[GH heads][16 dims] += sum_j w[j][h] * v[j][16 dims]   (v: dequantised float rows of stride 128, w: [nj][8] weights)
template <int GH>
__attribute__((target("avx2,fma")))
static inline void attn_v_accum(float* accl, int hg, const float* vf, const float* wt, int nj) {
    for (int db = 0; db < 8; db++) {
        __m256 a0[GH], a1[GH];
        for (int h = 0; h < GH; h++) { a0[h] = _mm256_loadu_ps(accl + (size_t)(hg + h) * 128 + db * 16); a1[h] = _mm256_loadu_ps(accl + (size_t)(hg + h) * 128 + db * 16 + 8); }
        for (int j = 0; j < nj; j++) {
            const __m256 v0 = _mm256_loadu_ps(vf + (size_t)j * 128 + db * 16), v1 = _mm256_loadu_ps(vf + (size_t)j * 128 + db * 16 + 8);
            for (int h = 0; h < GH; h++) {
                const __m256 w = _mm256_broadcast_ss(wt + (size_t)j * 8 + hg + h);
                a0[h] = _mm256_fmadd_ps(w, v0, a0[h]);
                a1[h] = _mm256_fmadd_ps(w, v1, a1[h]);
            }
        }
        for (int h = 0; h < GH; h++) { _mm256_storeu_ps(accl + (size_t)(hg + h) * 128 + db * 16, a0[h]); _mm256_storeu_ps(accl + (size_t)(hg + h) * 128 + db * 16 + 8, a1[h]); }
    }
}

template <int G>
__attribute__((target("avx2,fma")))
static void attn_chunk_avx2(const float* q, const int8_t* k8, const float* ks, const int8_t* v8, const float* vs, int n, float* m, float* l, float* acc, float* scratch) {
    float* sc = scratch;                       // [n][8] scores, later exp weights
    float* vf = scratch + (size_t)n * 8;       // [32][128] dequantised V rows
    const __m256 z = _mm256_setzero_ps();
    for (int j = 0; j < n; j++) {
        const int8_t* kr = k8 + (size_t)j * 128;
        const float* ksj = ks + (size_t)j * 8;
        __m256 a[8];
        for (int h = 0; h < 8; h++) a[h] = z;
        for (int b = 0; b < 8; b++) {
            const __m256 s = _mm256_broadcast_ss(ksj + b);
            for (int u = 0; u < 2; u++) {
                const int g = b * 2 + u;
                const __m256 kf = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(kr + g * 8)))), s);
                for (int h = 0; h < G; h++) a[h] = _mm256_fmadd_ps(_mm256_loadu_ps(q + (size_t)h * 128 + g * 8), kf, a[h]);
            }
        }
        const __m256 t0 = _mm256_hadd_ps(a[0], a[1]), t1 = _mm256_hadd_ps(a[2], a[3]), t2 = _mm256_hadd_ps(a[4], a[5]), t3 = _mm256_hadd_ps(a[6], a[7]);
        const __m256 u0 = _mm256_hadd_ps(t0, t1), u1 = _mm256_hadd_ps(t2, t3);
        const __m128 r0 = _mm_add_ps(_mm256_castps256_ps128(u0), _mm256_extractf128_ps(u0, 1));
        const __m128 r1 = _mm_add_ps(_mm256_castps256_ps128(u1), _mm256_extractf128_ps(u1, 1));
        _mm256_storeu_ps(sc + (size_t)j * 8, _mm256_set_m128(r1, r0));
    }
    __m256 mx = _mm256_set1_ps(-INFINITY);
    for (int j = 0; j < n; j++) mx = _mm256_max_ps(mx, _mm256_loadu_ps(sc + (size_t)j * 8));
    __m256 sum = z;
    for (int j = 0; j < n; j++) {
        const __m256 e = exp256(_mm256_sub_ps(_mm256_loadu_ps(sc + (size_t)j * 8), mx));
        _mm256_storeu_ps(sc + (size_t)j * 8, e);
        sum = _mm256_add_ps(sum, e);
    }
    alignas(32) float mt[8], st[8];
    _mm256_store_ps(mt, mx);
    _mm256_store_ps(st, sum);
    for (int h = 0; h < G; h++) { m[h] = mt[h]; l[h] = st[h]; }
    float accl[G * 128];
    for (int i = 0; i < G * 128; i++) accl[i] = 0.f;
    for (int j0 = 0; j0 < n; j0 += 32) {
        const int nj = std::min(32, n - j0);
        for (int jj = 0; jj < nj; jj++) {
            const int8_t* vr = v8 + (size_t)(j0 + jj) * 128;
            const float* vsj = vs + (size_t)(j0 + jj) * 8;
            for (int b = 0; b < 8; b++) {
                const __m256 s = _mm256_broadcast_ss(vsj + b);
                for (int u = 0; u < 2; u++) {
                    const int g = b * 2 + u;
                    _mm256_storeu_ps(vf + (size_t)jj * 128 + g * 8, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(vr + g * 8)))), s));
                }
            }
        }
        const float* wt = sc + (size_t)j0 * 8;
        if (G >= 4) { for (int hg = 0; hg + 4 <= G; hg += 4) attn_v_accum<4>(accl, hg, vf, wt, nj); }
        else if (G == 2) attn_v_accum<2>(accl, 0, vf, wt, nj);
        else attn_v_accum<1>(accl, 0, vf, wt, nj);
    }
    for (int i = 0; i < G * 128; i++) acc[i] = accl[i];
}
#endif

bool attn_gqa_chunk(int G, const float* q, const int8_t* k8, const float* ks, const int8_t* v8, const float* vs, int n, float* m, float* l, float* acc, float* scratch) {
#if defined(__x86_64__) || defined(_M_X64)
    if (!cpu_has_f16c_fma() || n <= 0) return false;
    switch (G) {
        case 1: attn_chunk_avx2<1>(q, k8, ks, v8, vs, n, m, l, acc, scratch); return true;
        case 2: attn_chunk_avx2<2>(q, k8, ks, v8, vs, n, m, l, acc, scratch); return true;
        case 4: attn_chunk_avx2<4>(q, k8, ks, v8, vs, n, m, l, acc, scratch); return true;
        case 8: attn_chunk_avx2<8>(q, k8, ks, v8, vs, n, m, l, acc, scratch); return true;
        default: return false;
    }
#else
    return false;
#endif
}

}  // namespace lb
