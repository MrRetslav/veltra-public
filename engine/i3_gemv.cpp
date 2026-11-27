#include "i3_gemv.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "lb_common.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace lb {

// ----------------------------------------------------------------------------------------------- stream helpers
static inline int i3_q(const uint8_t* row96, int tile_in_pair, int col128) {   // 3-bit level of column col128 (0..127) of one tile
    const int k = col128 >> 5, i = col128 & 31;
    const int lo = (row96[32 * tile_in_pair + i] >> (2 * k)) & 3;
    const int hi = (row96[64 + i] >> (4 * tile_in_pair + k)) & 1;
    return lo | (hi << 2);
}
static inline float i3_h(const uint8_t* p) { uint16_t h; std::memcpy(&h, p, 2); return f16_to_f32(h); }

void i3_dequant_row(const I3View& v, int row, float* out) {
    const int rb = row / 64, r = row % 64, npair = v.ntile / 2;
    for (int p = 0; p < npair; p++) {
        const uint8_t* hdr = v.w + ((size_t)rb * npair + p) * kI3Chunk;
        const uint8_t* rw = hdr + kI3Hdr + (size_t)r * 96;
        for (int t = 0; t < 2; t++) {
            const float sc = i3_h(hdr + (size_t)(t * 2 + 0) * 128 + 2 * r), mn = i3_h(hdr + (size_t)(t * 2 + 1) * 128 + 2 * r);
            for (int c = 0; c < 128; c++) out[(size_t)(2 * p + t) * 128 + c] = (float)i3_q(rw, t, c) * sc + mn;
        }
    }
}

void i3_pack(int rows, int n, const uint8_t* q, const float* scale, const float* mn, uint8_t* out) {
    const int ntile = n / 128, npair = ntile / 2;
    for (int rb = 0; rb < rows / 64; rb++)
        for (int p = 0; p < npair; p++) {
            uint8_t* hdr = out + ((size_t)rb * npair + p) * kI3Chunk;
            std::memset(hdr + kI3Hdr, 0, kI3Data);
            for (int r = 0; r < 64; r++) {
                const size_t row = (size_t)rb * 64 + r;
                uint8_t* rw = hdr + kI3Hdr + (size_t)r * 96;
                for (int t = 0; t < 2; t++) {
                    const int tl = 2 * p + t;
                    const uint16_t s = f32_to_f16(scale[row * ntile + tl]), m = f32_to_f16(mn[row * ntile + tl]);
                    std::memcpy(hdr + (size_t)(t * 2 + 0) * 128 + 2 * r, &s, 2);
                    std::memcpy(hdr + (size_t)(t * 2 + 1) * 128 + 2 * r, &m, 2);
                    for (int c = 0; c < 128; c++) {
                        const int v = q[row * n + (size_t)tl * 128 + c] & 7, k = c >> 5, i = c & 31;
                        rw[32 * t + i] |= (uint8_t)((v & 3) << (2 * k));
                        rw[64 + i] |= (uint8_t)(((v >> 2) & 1) << (4 * t + k));
                    }
                }
            }
        }
}

// ----------------------------------------------------------------------------------------------- scalar reference
void i3_scalar(const I3View& v, const int8_t* q, size_t qs, const int32_t* A, int E, float* yE, size_t yst, int rb0, int rb1, const float* ss) {
    const int nt = v.ntile, npair = nt / 2;
    for (int rb = rb0; rb < rb1; rb++)
        for (int e = 0; e < E; e++) {
            const int8_t* qe = q + (size_t)e * qs;
            for (int r = 0; r < 64; r++) {
                float acc = 0.f;
                for (int tl = 0; tl < nt; tl++) {
                    const int p = tl / 2, t = tl % 2;
                    const uint8_t* hdr = v.w + ((size_t)rb * npair + p) * kI3Chunk;
                    const uint8_t* rw = hdr + kI3Hdr + (size_t)r * 96;
                    int32_t s = 0;
                    for (int c = 0; c < 128; c++) s += (int32_t)i3_q(rw, t, c) * qe[(size_t)tl * 128 + c];
                    const float sc = i3_h(hdr + (size_t)(t * 2 + 0) * 128 + 2 * r), mn = i3_h(hdr + (size_t)(t * 2 + 1) * 128 + 2 * r);
                    const float am = (float)A[(size_t)e * nt + tl] * mn;
                    const float c = std::fma((float)s, sc, am);
                    if (ss) acc = std::fma(c, ss[(size_t)e * nt + tl], acc); else acc = acc + c;
                }
                yE[(size_t)e * yst + (size_t)rb * 64 + r] = acc;
            }
        }
}

#if defined(__x86_64__) || defined(_M_X64)
#define I3_AVX2 __attribute__((target("avx2,fma,f16c")))

// lane i of the result = horizontal sum of the 8 int32 lanes of s[i]
I3_AVX2 static inline __m256i i3_red8(const __m256i* s) {
    const __m256i u01 = _mm256_add_epi32(_mm256_unpacklo_epi32(s[0], s[1]), _mm256_unpackhi_epi32(s[0], s[1]));
    const __m256i u23 = _mm256_add_epi32(_mm256_unpacklo_epi32(s[2], s[3]), _mm256_unpackhi_epi32(s[2], s[3]));
    const __m256i u45 = _mm256_add_epi32(_mm256_unpacklo_epi32(s[4], s[5]), _mm256_unpackhi_epi32(s[4], s[5]));
    const __m256i u67 = _mm256_add_epi32(_mm256_unpacklo_epi32(s[6], s[7]), _mm256_unpackhi_epi32(s[6], s[7]));
    const __m256i a = _mm256_add_epi32(_mm256_unpacklo_epi64(u01, u23), _mm256_unpackhi_epi64(u01, u23));
    const __m256i b = _mm256_add_epi32(_mm256_unpacklo_epi64(u45, u67), _mm256_unpackhi_epi64(u45, u67));
    return _mm256_add_epi32(_mm256_permute2x128_si256(a, b, 0x20), _mm256_permute2x128_si256(a, b, 0x31));
}

// the 4 u8 weight vectors (levels 0..7, columns 32k..32k+31 of tile TT) of one row; wr = the row's 96 bytes
template <int TT>
I3_AVX2 static inline void i3_unpack(const uint8_t* wr, __m256i* w) {
    const __m256i m3 = _mm256_set1_epi8(3), m4 = _mm256_set1_epi8(4);
    const __m256i lo = _mm256_loadu_si256((const __m256i*)(wr + 32 * TT)), hb = _mm256_loadu_si256((const __m256i*)(wr + 64));
    __m256i h0, h1, h2, h3;
    if constexpr (TT == 0) {
        h0 = _mm256_and_si256(_mm256_slli_epi16(hb, 2), m4);
        h1 = _mm256_and_si256(_mm256_slli_epi16(hb, 1), m4);
        h2 = _mm256_and_si256(hb, m4);
        h3 = _mm256_and_si256(_mm256_srli_epi16(hb, 1), m4);
    } else {
        h0 = _mm256_and_si256(_mm256_srli_epi16(hb, 2), m4);
        h1 = _mm256_and_si256(_mm256_srli_epi16(hb, 3), m4);
        h2 = _mm256_and_si256(_mm256_srli_epi16(hb, 4), m4);
        h3 = _mm256_and_si256(_mm256_srli_epi16(hb, 5), m4);
    }
    w[0] = _mm256_or_si256(_mm256_and_si256(lo, m3), h0);
    w[1] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lo, 2), m3), h1);
    w[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lo, 4), m3), h2);
    w[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lo, 6), m3), h3);
}

template <int TT, int TG>
I3_AVX2 static inline void i3_tile(const uint8_t* hdr, const uint8_t* dat, int tl, int nt, const int8_t* q, size_t qs, const int32_t* A, const float* ss, int e0,
                                   float (*acc)[8][8]) {
    const __m256i ones = _mm256_set1_epi16(1);
    const int8_t* ap[TG];
    float af[TG], sk[TG];
    for (int k = 0; k < TG; k++) {
        ap[k] = q + (size_t)(e0 + k) * qs + (size_t)tl * 128;
        af[k] = (float)A[(size_t)(e0 + k) * nt + tl];
        sk[k] = ss ? ss[(size_t)(e0 + k) * nt + tl] : 1.f;
    }
    for (int rg = 0; rg < 8; rg++) {
        __m256i s[TG][8];
        for (int j = 0; j < 8; j++) {
            __m256i w[4];
            i3_unpack<TT>(dat + (size_t)(rg * 8 + j) * 96, w);
            for (int k = 0; k < TG; k++) {
                const __m256i a0 = _mm256_loadu_si256((const __m256i*)(ap[k])), a1 = _mm256_loadu_si256((const __m256i*)(ap[k] + 32));
                const __m256i a2 = _mm256_loadu_si256((const __m256i*)(ap[k] + 64)), a3 = _mm256_loadu_si256((const __m256i*)(ap[k] + 96));
                const __m256i p = _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(w[0], a0), _mm256_maddubs_epi16(w[1], a1)),
                                                   _mm256_add_epi16(_mm256_maddubs_epi16(w[2], a2), _mm256_maddubs_epi16(w[3], a3)));
                s[k][j] = _mm256_madd_epi16(p, ones);
            }
        }
        const __m256 sc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(hdr + (size_t)(TT * 2 + 0) * 128 + rg * 16)));
        const __m256 mn = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(hdr + (size_t)(TT * 2 + 1) * 128 + rg * 16)));
        for (int k = 0; k < TG; k++) {
            const __m256 c = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i3_red8(s[k])), sc, _mm256_mul_ps(_mm256_set1_ps(af[k]), mn));
            const __m256 a = _mm256_load_ps(acc[k][rg]);
            _mm256_store_ps(acc[k][rg], ss ? _mm256_fmadd_ps(c, _mm256_set1_ps(sk[k]), a) : _mm256_add_ps(a, c));
        }
    }
}

template <int TG>
I3_AVX2 static void i3_rb(const I3View& v, int rb, const int8_t* q, size_t qs, const int32_t* A, const float* ss, int e0, float* yE, size_t yst) {
    const int nt = v.ntile, npair = nt / 2;
    alignas(32) float acc[TG][8][8];
    std::memset(acc, 0, sizeof acc);
    const uint8_t* base = v.w + (size_t)rb * npair * kI3Chunk;
    for (int p = 0; p < npair; p++) {
        const uint8_t* hdr = base + (size_t)p * kI3Chunk;
        const uint8_t* dat = hdr + kI3Hdr;
        i3_tile<0, TG>(hdr, dat, 2 * p, nt, q, qs, A, ss, e0, acc);
        i3_tile<1, TG>(hdr, dat, 2 * p + 1, nt, q, qs, A, ss, e0, acc);
    }
    for (int k = 0; k < TG; k++) std::memcpy(yE + (size_t)(e0 + k) * yst + (size_t)rb * 64, acc[k], 64 * sizeof(float));
}

#define I3_CASE(N) case N: i3_rb<N>(v, rb, q, qs, A, ss, e0, yE, yst); break;
I3_AVX2 static void i3_dispatch(int tg, const I3View& v, int rb, const int8_t* q, size_t qs, const int32_t* A, const float* ss, int e0, float* yE, size_t yst) {
    switch (tg) { I3_CASE(1) I3_CASE(2) I3_CASE(3) I3_CASE(4) I3_CASE(5) I3_CASE(6) I3_CASE(7) default: i3_rb<8>(v, rb, q, qs, A, ss, e0, yE, yst); }
}

static const int g_i3_tg = lb::env_get("I3_TG") ? std::max(1, std::min(8, std::atoi(lb::env_get("I3_TG")))) : 4;   // slots per weight pass
I3_AVX2 void i3_avx2(const I3View& v, const int8_t* q, size_t qs, const int32_t* A, int E, float* yE, size_t yst, int rb0, int rb1, const float* ss) {
    for (int rb = rb0; rb < rb1; rb++)
        for (int e0 = 0; e0 < E; e0 += g_i3_tg) i3_dispatch(std::min(g_i3_tg, E - e0), v, rb, q, qs, A, ss, e0, yE, yst);
}
#endif

void i3_gemm(Isa isa, const I3View& v, const int8_t* q, size_t qs, const int32_t* A, int E, float* yE, size_t yst, int rb0, int rb1, const float* ss) {
#if defined(__x86_64__) || defined(_M_X64)
    if (isa == Isa::Avx2 || isa == Isa::Avx512) { i3_avx2(v, q, qs, A, E, yE, yst, rb0, rb1, ss); return; }   // AVX-512 hosts use the AVX2 kernel
#endif
    i3_scalar(v, q, qs, A, E, yE, yst, rb0, rb1, ss);
}

}  // namespace lb
