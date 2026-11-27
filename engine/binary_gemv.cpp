#include "binary_gemv.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "lb_common.h"
#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

namespace lb {

const char* isa_name(Isa isa) {
    switch (isa) {
        case Isa::Scalar: return "scalar";
        case Isa::Avx2: return "avx2";
        case Isa::Avx512: return "avx512-vbmi";
        case Isa::Neon: return "neon";
    }
    return "?";
}

#if defined(__x86_64__) || defined(_M_X64)
// Own CPUID/XGETBV probing: __builtin_cpu_supports needs compiler-rt init that is not
// reliable in every toolchain (zig/clang on Windows, static Android builds).
struct X86Caps { bool avx2 = false, avx512 = false; };
static X86Caps probe_x86() {
    X86Caps c;
    unsigned a, b, cc, d;
    if (!__get_cpuid(1, &a, &b, &cc, &d)) return c;
    bool osxsave = cc & (1u << 27), avx = cc & (1u << 28);
    if (!osxsave || !avx) return c;
    unsigned xlo, xhi;
    __asm__ volatile("xgetbv" : "=a"(xlo), "=d"(xhi) : "c"(0));
    bool ymm = (xlo & 6) == 6, zmm = (xlo & 0xe6) == 0xe6;
    if (__get_cpuid_max(0, nullptr) < 7) return c;
    __cpuid_count(7, 0, a, b, cc, d);
    c.avx2 = ymm && (b & (1u << 5));
    c.avx512 = zmm && (b & (1u << 16)) && (b & (1u << 30)) && (b & (1u << 31)) && (cc & (1u << 1)) && (cc & (1u << 11));  // F, BW, VL, VBMI, VNNI
    return c;
}
static const X86Caps& x86_caps() { static const X86Caps c = probe_x86(); return c; }
#endif

bool isa_available(Isa isa) {
    switch (isa) {
        case Isa::Scalar: return true;
#if defined(__x86_64__) || defined(_M_X64)
        case Isa::Avx2: return x86_caps().avx2;
        case Isa::Avx512: return x86_caps().avx512;
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
        case Isa::Neon: return true;
#endif
        default: return false;
    }
}

Isa best_isa() {
    if (const char* e = lb::env_get("ISA")) {  // explicit override: scalar | avx2 | avx512
        std::string v(e);
        if (v == "scalar") return Isa::Scalar;
        if (v == "avx2" && isa_available(Isa::Avx2)) return Isa::Avx2;
        if (v == "avx512" && isa_available(Isa::Avx512)) return Isa::Avx512;
    }
    if (isa_available(Isa::Avx512)) return Isa::Avx512;
    if (isa_available(Isa::Avx2)) return Isa::Avx2;
    if (isa_available(Isa::Neon)) return Isa::Neon;
    return Isa::Scalar;
}

static inline unsigned nibble_of(const uint64_t* row_bits, size_t words, int g) {
    size_t k = (size_t)g * 4;
    size_t w = k >> 6;
    if (w >= words) return 0;
    return (unsigned)((row_bits[w] >> (k & 63)) & 0xf);
}

void pack_bits_to_lut(const uint64_t* bits, int rows, int cols, uint8_t* out) {
    const int rp = pad_rows(rows);
    const int kp = pad_k(cols);
    const size_t words = bits_words_per_row(cols);
    const int pairs = kp / 8;
    for (int rb = 0; rb < rp / kRowBlock; rb++) {
        for (int p = 0; p < pairs; p++) {
            uint8_t* dst = out + ((size_t)rb * pairs + p) * 64;
            for (int i = 0; i < 64; i++) {
                int row = rb * kRowBlock + i;
                unsigned lo = 0, hi = 0;
                if (row < rows) {
                    const uint64_t* rbits = bits + (size_t)row * words;
                    lo = nibble_of(rbits, words, 2 * p);
                    hi = nibble_of(rbits, words, 2 * p + 1);
                }
                dst[i] = (uint8_t)(lo | (hi << 4));
            }
        }
    }
}

void unpack_lut_to_bits(const uint8_t* lut, int rows, int cols, uint64_t* bits) {
    const int kp = pad_k(cols);
    const size_t words = bits_words_per_row(cols);
    const int pairs = kp / 8;
    std::memset(bits, 0, (size_t)rows * words * 8);
    for (int row = 0; row < rows; row++) {
        int rb = row / kRowBlock, i = row % kRowBlock;
        for (int p = 0; p < pairs; p++) {
            uint8_t b = lut[((size_t)rb * pairs + p) * 64 + i];
            for (int h = 0; h < 2; h++) {
                int g = 2 * p + h;
                unsigned nib = (b >> (4 * h)) & 0xf;
                for (int j = 0; j < 4; j++) {
                    int k = 4 * g + j;
                    if (k < cols && ((nib >> j) & 1)) bits[(size_t)row * words + (k >> 6)] |= 1ull << (k & 63);
                }
            }
        }
    }
}

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
__attribute__((target("avx2")))
static float amax_avx2(const float* x, int n) {
    const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 m0 = _mm256_setzero_ps(), m1 = m0;
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        m0 = _mm256_max_ps(m0, _mm256_and_ps(_mm256_loadu_ps(x + i), mask));
        m1 = _mm256_max_ps(m1, _mm256_and_ps(_mm256_loadu_ps(x + i + 8), mask));
    }
    m0 = _mm256_max_ps(m0, m1);
    alignas(32) float t[8];
    _mm256_store_ps(t, m0);
    float a = 0.f;
    for (int k = 0; k < 8; k++) a = std::max(a, t[k]);
    for (; i < n; i++) a = std::max(a, std::fabs(x[i]));
    return a;
}
__attribute__((target("avx2")))
static void quant_avx2(const float* x, int n, float inv, int8_t* q) {
    const __m256 vinv = _mm256_set1_ps(inv), hi = _mm256_set1_ps(127.f), lo = _mm256_set1_ps(-127.f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_min_ps(hi, _mm256_max_ps(lo, _mm256_mul_ps(_mm256_loadu_ps(x + i), vinv)));
        __m256i i32 = _mm256_cvtps_epi32(v);  // round-to-nearest-even == std::nearbyint
        __m128i p16 = _mm_packs_epi32(_mm256_castsi256_si128(i32), _mm256_extracti128_si256(i32, 1));
        _mm_storel_epi64((__m128i*)(q + i), _mm_packs_epi16(p16, p16));
    }
    for (; i < n; i++) {
        float v = std::nearbyint(x[i] * inv);
        q[i] = (int8_t)std::min(127.f, std::max(-127.f, v));
    }
}
// 16 int16 table entries per group: T[p] = sum_j (bit_j(p) ? a_j : -a_j)
template <bool L512>
__attribute__((target("avx2")))
static void build_luts_avx2(const int8_t* q, int groups, int8_t* lut) {
    alignas(32) static int16_t S[4][16];
    static bool init = false;
    if (!init) {
        for (int j = 0; j < 4; j++) for (int p = 0; p < 16; p++) S[j][p] = ((p >> j) & 1) ? 1 : -1;
        init = true;
    }
    const __m256i s0 = _mm256_load_si256((const __m256i*)S[0]), s1 = _mm256_load_si256((const __m256i*)S[1]);
    const __m256i s2 = _mm256_load_si256((const __m256i*)S[2]), s3 = _mm256_load_si256((const __m256i*)S[3]);
    // gather even bytes (lo) into the low 8 bytes and odd bytes (hi) into the high 8 bytes of each 128-bit lane
    const __m256i shuf = _mm256_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
                                          0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    for (int g = 0; g < groups; g++) {
        __m256i t = _mm256_mullo_epi16(s0, _mm256_set1_epi16(q[4 * g]));
        t = _mm256_add_epi16(t, _mm256_mullo_epi16(s1, _mm256_set1_epi16(q[4 * g + 1])));
        t = _mm256_add_epi16(t, _mm256_mullo_epi16(s2, _mm256_set1_epi16(q[4 * g + 2])));
        t = _mm256_add_epi16(t, _mm256_mullo_epi16(s3, _mm256_set1_epi16(q[4 * g + 3])));
        t = _mm256_shuffle_epi8(t, shuf);                 // [lo0..7 hi0..7 | lo8..15 hi8..15]
        t = _mm256_permute4x64_epi64(t, 0xD8);            // [lo0..15 | hi0..15]
        if (L512) {
            int8_t* d = lut + (size_t)(g >> 3) * 256 + (size_t)(g & 1) * 128 + (size_t)((g >> 1) & 3) * 16;
            _mm_storeu_si128((__m128i*)d, _mm256_castsi256_si128(t));
            _mm_storeu_si128((__m128i*)(d + 64), _mm256_extracti128_si256(t, 1));
        } else {
            _mm256_storeu_si256((__m256i*)(lut + (size_t)g * 32), t);
        }
    }
}

// AVX-512: table of 16 int32 entries per group from two vpmaddwd (sign matrices x activation pairs), then vpmovdb to the
// lo/hi byte planes -- ~3 uops per group on the shuffle port instead of ~12 for the AVX2 builder.
template <bool L512>
__attribute__((target("avx512f,avx512bw,avx512vl,avx2")))
static void build_luts_avx512(const int8_t* q, int groups, int8_t* lut) {
    alignas(64) static int32_t S01[16], S23[16];
    static bool init = false;
    if (!init) {
        for (int p = 0; p < 16; p++) {
            auto sg = [&](int j) { return (uint32_t)(uint16_t)(int16_t)(((p >> j) & 1) ? 1 : -1); };
            S01[p] = (int32_t)(sg(0) | (sg(1) << 16));
            S23[p] = (int32_t)(sg(2) | (sg(3) << 16));
        }
        init = true;
    }
    const __m512i s01 = _mm512_load_si512((const void*)S01), s23 = _mm512_load_si512((const void*)S23);
    alignas(64) int16_t tmp[32];
    for (int g0 = 0; g0 < groups; g0 += 8) {
        _mm512_store_si512((void*)tmp, _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*)(q + 4 * g0))));
        for (int j = 0; j < 8; j++) {
            const int g = g0 + j;
            const __m512i a01 = _mm512_set1_epi32(*(const int32_t*)(tmp + 4 * j));
            const __m512i a23 = _mm512_set1_epi32(*(const int32_t*)(tmp + 4 * j + 2));
            const __m512i t = _mm512_add_epi32(_mm512_madd_epi16(s01, a01), _mm512_madd_epi16(s23, a23));
            int8_t* d = L512 ? lut + (size_t)(g >> 3) * 256 + (size_t)(g & 1) * 128 + (size_t)((g >> 1) & 3) * 16 : lut + (size_t)g * 32;
            _mm_storeu_si128((__m128i*)d, _mm512_cvtepi32_epi8(t));
            _mm_storeu_si128((__m128i*)(d + (L512 ? 64 : 16)), _mm512_cvtepi32_epi8(_mm512_srai_epi32(t, 8)));
        }
    }
}
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
// 16 int16 table entries per group: T[p] = sum_j (bit_j(p) ? a_j : -a_j); lo/hi byte planes via narrowing moves.
static void build_luts_neon(const int8_t* q, int groups, int8_t* lut) {
    static const int16_t S[4][16] = {
        {-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1},
        {-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1},
        {-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1},
        {-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1}};
    int16x8_t sl[4], sh[4];
    for (int j = 0; j < 4; j++) { sl[j] = vld1q_s16(S[j]); sh[j] = vld1q_s16(S[j] + 8); }
    for (int g = 0; g < groups; g++) {
        const int16_t a0 = q[4 * g], a1 = q[4 * g + 1], a2 = q[4 * g + 2], a3 = q[4 * g + 3];
        int16x8_t tl = vmulq_n_s16(sl[0], a0), th = vmulq_n_s16(sh[0], a0);
        tl = vmlaq_n_s16(tl, sl[1], a1); th = vmlaq_n_s16(th, sh[1], a1);
        tl = vmlaq_n_s16(tl, sl[2], a2); th = vmlaq_n_s16(th, sh[2], a2);
        tl = vmlaq_n_s16(tl, sl[3], a3); th = vmlaq_n_s16(th, sh[3], a3);
        int8_t* d = lut + (size_t)g * 32;
        vst1q_s8(d, vcombine_s8(vmovn_s16(tl), vmovn_s16(th)));
        vst1q_s8(d + 16, vcombine_s8(vshrn_n_s16(tl, 8), vshrn_n_s16(th, 8)));
    }
}
static float amax_neon(const float* x, int n) {
    float32x4_t m = vdupq_n_f32(0.f);
    int i = 0;
    for (; i + 4 <= n; i += 4) m = vmaxq_f32(m, vabsq_f32(vld1q_f32(x + i)));
    float a = vmaxvq_f32(m);
    for (; i < n; i++) a = std::max(a, std::fabs(x[i]));
    return a;
}
static void quant_neon(const float* x, int n, float inv, int8_t* q) {
    const float32x4_t vinv = vdupq_n_f32(inv), hi = vdupq_n_f32(127.f), lo = vdupq_n_f32(-127.f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        int32x4_t a = vcvtnq_s32_f32(vminq_f32(hi, vmaxq_f32(lo, vmulq_f32(vld1q_f32(x + i), vinv))));
        int32x4_t b = vcvtnq_s32_f32(vminq_f32(hi, vmaxq_f32(lo, vmulq_f32(vld1q_f32(x + i + 4), vinv))));
        vst1_s8(q + i, vqmovn_s16(vcombine_s16(vqmovn_s32(a), vqmovn_s32(b))));
    }
    for (; i < n; i++) {
        float v = std::nearbyint(x[i] * inv);
        q[i] = (int8_t)std::min(127.f, std::max(-127.f, v));
    }
}
#endif

float quantize_activations_i8(const float* x, int n, int k_pad, int8_t* q) {
#if defined(__aarch64__) || defined(_M_ARM64)
    {
        float amax = amax_neon(x, n);
        float scale = amax > 0.f ? amax / 127.f : 1.f;
        float inv = amax > 0.f ? 127.f / amax : 0.f;
        quant_neon(x, n, inv, q);
        for (int i = n; i < k_pad; i++) q[i] = 0;
        return scale;
    }
#endif
#if defined(__x86_64__) || defined(_M_X64)
    if (x86_caps().avx2) {
        float amax = amax_avx2(x, n);
        float scale = amax > 0.f ? amax / 127.f : 1.f;
        float inv = amax > 0.f ? 127.f / amax : 0.f;
        quant_avx2(x, n, inv, q);
        for (int i = n; i < k_pad; i++) q[i] = 0;
        return scale;
    }
#endif
    float amax = 0.f;
    for (int i = 0; i < n; i++) amax = std::max(amax, std::fabs(x[i]));
    float scale = amax > 0.f ? amax / 127.f : 1.f;
    float inv = amax > 0.f ? 127.f / amax : 0.f;
    for (int i = 0; i < n; i++) {
        float v = std::nearbyint(x[i] * inv);
        v = std::min(127.f, std::max(-127.f, v));
        q[i] = (int8_t)v;
    }
    for (int i = n; i < k_pad; i++) q[i] = 0;
    return scale;
}

#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx2,fma")))
static void absmax_sumsq_avx2(const float* x, int n, float* amax, float* sumsq) {
    const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 m = _mm256_setzero_ps(), a0 = m, a1 = m;
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256 v0 = _mm256_loadu_ps(x + i), v1 = _mm256_loadu_ps(x + i + 8);
        m = _mm256_max_ps(m, _mm256_max_ps(_mm256_and_ps(v0, mask), _mm256_and_ps(v1, mask)));
        a0 = _mm256_fmadd_ps(v0, v0, a0);
        a1 = _mm256_fmadd_ps(v1, v1, a1);
    }
    a0 = _mm256_add_ps(a0, a1);
    alignas(32) float tm[8], ts[8];
    _mm256_store_ps(tm, m);
    _mm256_store_ps(ts, a0);
    float mx = 0.f, ss = 0.f;
    for (int k = 0; k < 8; k++) { mx = std::max(mx, tm[k]); ss += ts[k]; }
    for (; i < n; i++) { mx = std::max(mx, std::fabs(x[i])); ss += x[i] * x[i]; }
    *amax = mx; *sumsq = ss;
}
#endif

void absmax_sumsq(const float* x, int n, float* amax, float* sumsq) {
#if defined(__aarch64__) || defined(_M_ARM64)
    {
        float32x4_t m = vdupq_n_f32(0.f), a = m;
        int i = 0;
        for (; i + 4 <= n; i += 4) {
            float32x4_t v = vld1q_f32(x + i);
            m = vmaxq_f32(m, vabsq_f32(v));
            a = vfmaq_f32(a, v, v);
        }
        float mx = vmaxvq_f32(m), ss = vaddvq_f32(a);
        for (; i < n; i++) { mx = std::max(mx, std::fabs(x[i])); ss += x[i] * x[i]; }
        *amax = mx; *sumsq = ss;
        return;
    }
#endif
#if defined(__x86_64__) || defined(_M_X64)
    if (x86_caps().avx2) { absmax_sumsq_avx2(x, n, amax, sumsq); return; }
#endif
    float mx = 0.f, ss = 0.f;
    for (int i = 0; i < n; i++) { mx = std::max(mx, std::fabs(x[i])); ss += x[i] * x[i]; }
    *amax = mx; *sumsq = ss;
}

void build_luts(const int8_t* q, int k_pad, int8_t* lut, bool avx512_layout) {
    const int groups = k_pad / 4;
#if defined(__aarch64__) || defined(_M_ARM64)
    (void)avx512_layout;
    build_luts_neon(q, groups, lut);
    return;
#endif
#if defined(__x86_64__) || defined(_M_X64)
    if (x86_caps().avx512 && (groups & 7) == 0) {
        if (avx512_layout) build_luts_avx512<true>(q, groups, lut); else build_luts_avx512<false>(q, groups, lut);
        return;
    }
    if (avx512_layout) { build_luts_avx2<true>(q, groups, lut); return; }
    if (x86_caps().avx2) { build_luts_avx2<false>(q, groups, lut); return; }
#endif
    for (int g = 0; g < groups; g++) {
        const int a0 = q[4 * g], a1 = q[4 * g + 1], a2 = q[4 * g + 2], a3 = q[4 * g + 3];
        // subset sums built with additions only; T[p] = 2*sub[p] - total
        int sub[16];
        sub[0] = 0;
        sub[1] = a0;
        sub[2] = a1;
        sub[3] = a0 + a1;
        sub[4] = a2;
        sub[5] = sub[4] + a0;
        sub[6] = sub[4] + a1;
        sub[7] = sub[4] + sub[3];
        for (int p = 8; p < 16; p++) sub[p] = sub[p - 8] + a3;
        const int total = sub[15];
        int8_t* dst = lut + (size_t)g * 32;
        for (int p = 0; p < 16; p++) {
            int t = sub[p] + sub[p] - total;
            uint16_t u = (uint16_t)(int16_t)t;
            dst[p] = (int8_t)(u & 0xff);
            dst[16 + p] = (int8_t)(u >> 8);
        }
    }
}

void repack_lut_avx512(const uint8_t* lut_w, int rows, int cols, uint8_t* out) {
    const int nrb = pad_rows(rows) / kRowBlock, kp = pad_k(cols), nwv = kp / 32, pairs = kp / 8;
    for (int rb = 0; rb < nrb; rb++) {
        const uint8_t* wb = lut_w + (size_t)rb * pairs * 64;
        uint8_t* ob = out + (size_t)rb * nwv * 256;
        for (int c0 = 0; c0 < nwv; c0 += 16) {
            const int nc = std::min(16, nwv - c0);
            for (int g = 0; g < 4; g++)
                for (int i = 0; i < nc; i++) {
                    const int w = c0 + i;
                    uint8_t* d = ob + (size_t)c0 * 256 + (size_t)g * nc * 64 + (size_t)i * 64;
                    for (int b = 0; b < 4; b++) {
                        const uint8_t* src = wb + (size_t)(4 * w + b) * 64 + 16 * g;
                        for (int j = 0; j < 16; j++) d[4 * j + b] = src[j];
                    }
                }
        }
    }
}

void gemv_scalar(const BinMatView& w, const int8_t* luts, int T, int32_t* out, int rb_begin, int rb_end) {
    if (rb_end < 0) rb_end = pad_rows(w.rows) / kRowBlock;
    const int rp = pad_rows(w.rows), kp = pad_k(w.cols);
    const int pairs = kp / 8;
    const size_t lutTok = lut_bytes_per_token(kp);
    for (int t = 0; t < T; t++) {
        const int8_t* L = luts + t * lutTok;
        int32_t* o = out + (size_t)t * rp;
        for (int rb = rb_begin; rb < rb_end; rb++) {
            const uint8_t* wb = w.lut_w + (size_t)rb * pairs * 64;
            for (int i = 0; i < kRowBlock; i++) {
                int32_t acc = 0;
                for (int p = 0; p < pairs; p++) {
                    uint8_t b = wb[(size_t)p * 64 + i];
                    for (int h = 0; h < 2; h++) {
                        unsigned nib = (b >> (4 * h)) & 0xf;
                        const int8_t* g = L + (size_t)(2 * p + h) * 32;
                        uint16_t u = (uint16_t)((uint8_t)g[nib] | ((uint16_t)(uint8_t)g[16 + nib] << 8));
                        acc += (int16_t)u;
                    }
                }
                o[rb * kRowBlock + i] = acc;
            }
        }
    }
}

// Reference grid kernel (portable; also the NEON fallback). LUTs in the scalar/AVX2 layout, gs in row order.
void grid_scalar(const GridView& g, const int8_t* luts, int E, float* yE, int rb_begin, int rb_end) {
    const int rp = pad_rows(g.rows);
    if (rb_end < 0) rb_end = rp / kRowBlock;
    const size_t tileBytes = (size_t)rp * 16;
    for (int rb = rb_begin; rb < rb_end; rb++)
        for (int e = 0; e < E; e++)
            for (int i = 0; i < kRowBlock; i++) {
                float y = 0.f;
                for (int tile = 0; tile < g.ntile; tile++) {
                    const int8_t* L = luts + ((size_t)tile * E + e) * kTileLutBytes;
                    int32_t a = 0;
                    for (int p = 0; p < g.P; p++) {
                        const uint8_t* wb = g.w[p] + (size_t)tile * tileBytes + (size_t)rb * 1024;
                        int32_t s = 0;
                        for (int pr = 0; pr < 16; pr++) {
                            const uint8_t b = wb[(size_t)pr * 64 + i];
                            for (int h = 0; h < 2; h++) {
                                const unsigned nib = (b >> (4 * h)) & 0xf;
                                const int8_t* gq = L + (size_t)(2 * pr + h) * 32;
                                s += (int16_t)(uint16_t)((uint8_t)gq[nib] | ((uint16_t)(uint8_t)gq[16 + nib] << 8));
                            }
                        }
                        a = a * 2 + s;
                    }
                    y += (float)a * g.gs[(size_t)tile * g.gs_stride + (size_t)rb * kRowBlock + i];
                }
                yE[(size_t)e * rp + (size_t)rb * kRowBlock + i] = y;
            }
}

void grid_gemv(Isa isa, const GridView& g, const int8_t* luts, int E, float* yE, int rb_begin, int rb_end) {
    if (rb_end < 0) rb_end = pad_rows(g.rows) / kRowBlock;
    switch (isa) {
#if defined(__x86_64__) || defined(_M_X64)
        case Isa::Avx2: grid_avx2(g, luts, E, yE, rb_begin, rb_end); return;
        case Isa::Avx512: grid_avx512(g, luts, E, yE, rb_begin, rb_end); return;
#endif
        default: grid_scalar(g, luts, E, yE, rb_begin, rb_end); return;
    }
}

void binary_gemv_direct(const uint64_t* bits, int rows, int cols, const int8_t* a, int32_t* out) {
    const size_t words = bits_words_per_row(cols);
    for (int r = 0; r < rows; r++) {
        int32_t acc = 0;
        for (int k = 0; k < cols; k++) {
            bool plus = (bits[(size_t)r * words + (k >> 6)] >> (k & 63)) & 1;
            acc += plus ? a[k] : -a[k];
        }
        out[r] = acc;
    }
}

void binary_gemv(Isa isa, const BinMatView& w, const int8_t* luts, int T, int32_t* out, int rb_begin, int rb_end, int k_begin, int k_end) {
    switch (isa) {
#if defined(__x86_64__) || defined(_M_X64)
        case Isa::Avx2: gemv_avx2(w, luts, T, out, rb_begin, rb_end); return;
        case Isa::Avx512: gemv_avx512(w, luts, T, out, rb_begin, rb_end, k_begin, k_end); return;
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
        case Isa::Neon: gemv_neon(w, luts, T, out, rb_begin, rb_end); return;
#endif
        default: gemv_scalar(w, luts, T, out, rb_begin, rb_end); return;
    }
}

}  // namespace lb
