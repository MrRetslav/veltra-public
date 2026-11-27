#include "nibble_gemv.h"
#include "lb_common.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#ifndef NB_DEFAULT_ZMM
#define NB_DEFAULT_ZMM -1  // AVX-512 path: -1 = zmm for E <= 2 slots (decode) else ymm vpdpbusd (measured faster for batches); 0 / 1 force ymm / zmm (env LB_NIB_ZMM)
#endif

namespace lb {

// Arithmetic contract (identical to the bit-plane grid kernels, so both formats give bit-identical results on the same weights):
//   per tile: Z = 2*S - 15*A  (exact int32, S = sum j*q, A = sum q)   [= 2 * (S - 7.5 A)]
//   acc = fma(float(Z), gh, acc) in tile order, gh = 0.5 * gs (NibView::gs), yE = acc (the optional per-tile activation scale is folded into gh).

// ----------------------------------------------------------------------------------------------- scalar reference
void nibble_scalar(const NibView& v, const int8_t* q, size_t qs, const int32_t* A, int E, float* yE, size_t yst, int rb0, int rb1, const float* ss) {
    const int nt = v.ntile;
    for (int rb = rb0; rb < rb1; rb++) {
        const uint8_t* wb = v.w + (size_t)rb * nt * kNibTileBytes;
        const float* gb = v.gs + (size_t)rb * nt * 64;
        for (int e = 0; e < E; e++) {
            const int8_t* qe = q + (size_t)e * qs;
            for (int r = 0; r < 64; r++) {
                float acc = 0.f;
                for (int t = 0; t < nt; t++) {
                    const uint8_t* wr = wb + (size_t)t * kNibTileBytes + r * 64;
                    int32_t s = 0;
                    for (int i = 0; i < 64; i++) s += (int32_t)(wr[i] & 15) * qe[t * 128 + i] + (int32_t)(wr[i] >> 4) * qe[t * 128 + 64 + i];
                    const int32_t z = 2 * s - 15 * A[(size_t)e * nt + t];
                    float g = gb[(size_t)t * 64 + r];
                    if (ss) g *= ss[(size_t)e * nt + t];
                    acc = std::fma((float)z, g, acc);
                }
                yE[(size_t)e * yst + (size_t)rb * 64 + r] = acc;
            }
        }
    }
}

#if defined(__x86_64__) || defined(_M_X64)
#define NB_AVX2 __attribute__((target("avx2,fma")))
#define NB_VNNI __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,avx2,fma")))

// lane i of the result = horizontal sum of the 8 int32 lanes of s[i]
NB_AVX2 static inline __m256i nb_red8(const __m256i* s) {
    const __m256i u01 = _mm256_add_epi32(_mm256_unpacklo_epi32(s[0], s[1]), _mm256_unpackhi_epi32(s[0], s[1]));
    const __m256i u23 = _mm256_add_epi32(_mm256_unpacklo_epi32(s[2], s[3]), _mm256_unpackhi_epi32(s[2], s[3]));
    const __m256i u45 = _mm256_add_epi32(_mm256_unpacklo_epi32(s[4], s[5]), _mm256_unpackhi_epi32(s[4], s[5]));
    const __m256i u67 = _mm256_add_epi32(_mm256_unpacklo_epi32(s[6], s[7]), _mm256_unpackhi_epi32(s[6], s[7]));
    const __m256i a = _mm256_add_epi32(_mm256_unpacklo_epi64(u01, u23), _mm256_unpackhi_epi64(u01, u23));   // per 128-bit lane: partial sums of s0..s3
    const __m256i b = _mm256_add_epi32(_mm256_unpacklo_epi64(u45, u67), _mm256_unpackhi_epi64(u45, u67));   // s4..s7
    return _mm256_add_epi32(_mm256_permute2x128_si256(a, b, 0x20), _mm256_permute2x128_si256(a, b, 0x31));
}

// MODE 0: AVX2 (vpmaddubsw + vpmaddwd), 1: ymm vpdpbusd (AVX-512 VNNI + VL), 2: zmm vpdpbusd (rows folded 512 -> 256 before the reduction)
#define NB_ATTR NB_AVX2
#define NB_FN nb_rb_avx2
#define NB_MODE 0
#include "nibble_body.h"
#undef NB_ATTR
#undef NB_FN
#undef NB_MODE
#define NB_ATTR NB_VNNI
#define NB_FN nb_rb_vnni
#define NB_MODE 1
#include "nibble_body.h"
#undef NB_ATTR
#undef NB_FN
#undef NB_MODE
#define NB_ATTR NB_VNNI
#define NB_FN nb_rb_zmm
#define NB_MODE 2
#include "nibble_body.h"
#undef NB_ATTR
#undef NB_FN
#undef NB_MODE

#define NB_CASE(FN, N) case N: FN<N>(v, rb, q, qs, A, ss, e0, yE, yst); break;
NB_AVX2 static void nb_avx2_dispatch(int tg, const NibView& v, int rb, const int8_t* q, size_t qs, const int32_t* A, const float* ss, int e0, float* yE, size_t yst) {
    switch (tg) { NB_CASE(nb_rb_avx2, 1) NB_CASE(nb_rb_avx2, 2) NB_CASE(nb_rb_avx2, 3) NB_CASE(nb_rb_avx2, 4) NB_CASE(nb_rb_avx2, 5) NB_CASE(nb_rb_avx2, 6) NB_CASE(nb_rb_avx2, 7) default: nb_rb_avx2<8>(v, rb, q, qs, A, ss, e0, yE, yst); }
}
NB_VNNI static void nb_vnni_dispatch(int tg, const NibView& v, int rb, const int8_t* q, size_t qs, const int32_t* A, const float* ss, int e0, float* yE, size_t yst) {
    switch (tg) { NB_CASE(nb_rb_vnni, 1) NB_CASE(nb_rb_vnni, 2) NB_CASE(nb_rb_vnni, 3) NB_CASE(nb_rb_vnni, 4) NB_CASE(nb_rb_vnni, 5) NB_CASE(nb_rb_vnni, 6) NB_CASE(nb_rb_vnni, 7) default: nb_rb_vnni<8>(v, rb, q, qs, A, ss, e0, yE, yst); }
}
NB_VNNI static void nb_zmm_dispatch(int tg, const NibView& v, int rb, const int8_t* q, size_t qs, const int32_t* A, const float* ss, int e0, float* yE, size_t yst) {
    switch (tg) { NB_CASE(nb_rb_zmm, 1) NB_CASE(nb_rb_zmm, 2) NB_CASE(nb_rb_zmm, 3) default: nb_rb_zmm<4>(v, rb, q, qs, A, ss, e0, yE, yst); }
}

static const int g_nib_tg = lb::env_get("NIB_TG") ? std::max(1, std::min(8, std::atoi(lb::env_get("NIB_TG")))) : 8;   // slots per weight pass (AVX2 / ymm-VNNI)
NB_AVX2 void nibble_avx2(const NibView& v, const int8_t* q, size_t qs, const int32_t* A, int E, float* yE, size_t yst, int rb0, int rb1, const float* ss) {
    for (int rb = rb0; rb < rb1; rb++)
        for (int e0 = 0; e0 < E; e0 += g_nib_tg) nb_avx2_dispatch(std::min(g_nib_tg, E - e0), v, rb, q, qs, A, ss, e0, yE, yst);
}

static const int g_nib_zmm = lb::env_get("NIB_ZMM") ? std::atoi(lb::env_get("NIB_ZMM")) : NB_DEFAULT_ZMM;
NB_VNNI void nibble_avx512(const NibView& v, const int8_t* q, size_t qs, const int32_t* A, int E, float* yE, size_t yst, int rb0, int rb1, const float* ss) {
    if (g_nib_zmm == 0 || (g_nib_zmm < 0 && E > 2)) {
        for (int rb = rb0; rb < rb1; rb++)
            for (int e0 = 0; e0 < E; e0 += g_nib_tg) nb_vnni_dispatch(std::min(g_nib_tg, E - e0), v, rb, q, qs, A, ss, e0, yE, yst);
        return;
    }
    for (int rb = rb0; rb < rb1; rb++)
        for (int e0 = 0; e0 < E; e0 += 4) nb_zmm_dispatch(std::min(4, E - e0), v, rb, q, qs, A, ss, e0, yE, yst);
}
#endif

void nibble_gemm(Isa isa, const NibView& v, const int8_t* q, size_t qs, const int32_t* A, int E, float* yE, size_t yst, int rb0, int rb1, const float* ss) {
#if defined(__x86_64__) || defined(_M_X64)
    if (isa == Isa::Avx512) { nibble_avx512(v, q, qs, A, E, yE, yst, rb0, rb1, ss); return; }
    if (isa == Isa::Avx2) { nibble_avx2(v, q, qs, A, E, yE, yst, rb0, rb1, ss); return; }
#endif
    nibble_scalar(v, q, qs, A, E, yE, yst, rb0, rb1, ss);
}

}  // namespace lb
