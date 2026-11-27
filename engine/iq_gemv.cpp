#include "iq_gemv.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#include "iq_tables.h"
#include "lb_common.h"

namespace lb {

namespace {
inline uint16_t ld16(const uint8_t* p) { uint16_t h; std::memcpy(&h, p, 2); return h; }
inline uint32_t ld32(const uint8_t* p) { uint32_t h; std::memcpy(&h, p, 4); return h; }
inline uint64_t ld64(const uint8_t* p) { uint64_t h; std::memcpy(&h, p, 8); return h; }

// pshufb controls of the IQ2_XS scale vectors: sub-block ib -> lane 0: int16 (lo scale of ib) x8, lane 1: int16 (hi scale of ib) x8, from bytes [lo0 hi0 lo1 hi1 ...]
struct ScSel {
    alignas(32) uint8_t v[8][32];
    ScSel() { for (int ib = 0; ib < 8; ib++) for (int b = 0; b < 32; b++) v[ib][b] = (b & 1) ? 0x80 : (uint8_t)(2 * ib + (b >= 16 ? 1 : 0)); }
};
// pshufb controls expanding one sign byte per group into 8 bytes: sub-block ib of IQ2_XS (source lane (ib>>1)&1, byte base 4*(ib&1) + 8*(ib>>2)) and of
// IQ3_XXS (source lane ib>>2, dword ib&3). Destination lane 0 = groups 0,1 of the sub-block (8 bytes each), lane 1 = groups 2,3.
struct SgnSel {
    alignas(32) uint8_t v2[8][32], v3[8][32];
    alignas(32) uint8_t par[32], kmask[32], ones[32];
    SgnSel() {
        for (int ib = 0; ib < 8; ib++) {
            const int b2 = 4 * (ib & 1) + 8 * (ib >> 2), b3 = 4 * (ib & 3);
            for (int b = 0; b < 32; b++) {
                const int lane = b / 16, g = (b % 16) / 8;   // group within the lane
                v2[ib][b] = (uint8_t)(b2 + 2 * lane + g);
                v3[ib][b] = (uint8_t)(b3 + 2 * lane + g);
            }
        }
        for (int b = 0; b < 32; b++) {
            int i = b % 16, pr = 0;
            for (int k = 0; k < 4; k++) pr ^= (i >> k) & 1;
            par[b] = pr ? 0x80 : 0;
            kmask[b] = (uint8_t)(1u << (b % 8));
            ones[b] = 1;
        }
    }
};
const SgnSel g_sgn;
const ScSel g_scsel_s;
const uint8_t (*const g_scsel)[32] = g_scsel_s.v;

inline float kscale(int type) { return type == QK_IQ2XS ? 0.125f : type == QK_IQ3XXS ? 0.25f : 1.0f; }

// Integer part of one (row, superblock) against one int8 activation slice: sum_blocks ls * sum(w * a) (the ggml integer `bsum`); the caller multiplies by d * kscale.
int32_t sb_dot_scalar(int type, const uint8_t* p, const int8_t* a) {
    int32_t bsum = 0;
    if (type == QK_IQ2XS) {
        const uint8_t* sc = p + 66;
        for (int ib = 0; ib < 8; ib++) {
            int32_t s[2] = {0, 0};
            for (int l = 0; l < 4; l++) {
                const uint16_t e = ld16(p + 2 + 2 * (4 * ib + l));
                const uint8_t* grid = (const uint8_t*)(iq2xs_grid + (e & 511));
                const uint8_t sg = ksigns_iq2xs[e >> 9];
                for (int j = 0; j < 8; j++) s[l / 2] += grid[j] * a[32 * ib + 8 * l + j] * ((sg & kmask_iq2xs[j]) ? -1 : 1);
            }
            bsum += s[0] * (2 * (sc[ib] & 15) + 1) + s[1] * (2 * (sc[ib] >> 4) + 1);
        }
    } else if (type == QK_IQ3XXS) {
        for (int ib = 0; ib < 8; ib++) {
            const uint32_t aux = ld32(p + 66 + 4 * ib);
            int32_t s = 0;
            for (int l = 0; l < 4; l++) {
                const uint8_t* g1 = (const uint8_t*)(iq3xxs_grid + p[2 + 8 * ib + 2 * l]);
                const uint8_t* g2 = (const uint8_t*)(iq3xxs_grid + p[2 + 8 * ib + 2 * l + 1]);
                const uint8_t sg = ksigns_iq2xs[(aux >> (7 * l)) & 127];
                for (int j = 0; j < 4; j++) {
                    s += g1[j] * a[32 * ib + 8 * l + j] * ((sg & kmask_iq2xs[j]) ? -1 : 1);
                    s += g2[j] * a[32 * ib + 8 * l + 4 + j] * ((sg & kmask_iq2xs[j + 4]) ? -1 : 1);
                }
            }
            bsum += s * (int32_t)(2 * (aux >> 28) + 1);
        }
    } else {
        const uint16_t sh = ld16(p + 2);
        const uint8_t* sl = p + 4;
        const uint8_t* qs = p + 8;
        for (int ib = 0; ib < 8; ib++) {
            const int ls = ((sl[ib / 2] >> (4 * (ib & 1))) & 15) | (((sh >> (2 * ib)) & 3) << 4);
            int32_t s = 0;
            for (int j = 0; j < 16; j++) {
                s += a[32 * ib + j] * kvalues_iq4nl[qs[16 * ib + j] & 15];
                s += a[32 * ib + 16 + j] * kvalues_iq4nl[qs[16 * ib + j] >> 4];
            }
            bsum += s * (ls - 32);
        }
    }
    return bsum;
}

void iq_scalar(const QkView& v, const int8_t* q, size_t qs, int E, float* yE, size_t yst, int g0, int g1, const float* ss) {
    const int SB = kQkSbBytes[v.type];
    const float ks = kscale(v.type);
    for (int g = g0; g < g1; g++)
        for (int r = 0; r < 4; r++)
            for (int e = 0; e < E; e++) {
                float acc = 0.f;
                for (int t = 0; t < v.nsb; t++) {
                    const uint8_t* p = v.w + (((size_t)g * v.nsb + t) * 4 + r) * SB;
                    const float d = f16_to_f32(ld16(p)) * ks * (ss ? ss[(size_t)e * v.nsb + t] : 1.f);
                    acc += (float)sb_dot_scalar(v.type, p, q + (size_t)e * qs + (size_t)t * 256) * d;
                }
                yE[(size_t)e * yst + (size_t)g * 4 + r] = acc;
            }
}

#if defined(__x86_64__) || defined(_M_X64)
#define IQ_AVX2 __attribute__((target("avx2,fma,f16c")))

IQ_AVX2 inline float hsum8(__m256 v) {
    __m128 x = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_shuffle_ps(x, x, 1));
    return _mm_cvtss_f32(x);
}
IQ_AVX2 inline float f16s(const uint8_t* p) { return _cvtsh_ss(ld16(p)); }
// products of 32 weights (u8 magnitudes G with sign bytes S, or s8 weights) with the int8 activations a: 16 int16 pair sums
IQ_AVX2 inline __m256i dot_sgn(__m256i G, __m256i S, const int8_t* a) {
    return _mm256_maddubs_epi16(G, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)a), S));
}

// sign bytes (7-bit index | parity << 7) of 32 groups from bytes holding the 7-bit indices
IQ_AVX2 inline __m256i sign_bytes(__m256i idx) {
    const __m256i m15 = _mm256_set1_epi8(15), lut = _mm256_load_si256((const __m256i*)g_sgn.par);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(idx, 4), m15);
    return _mm256_or_si256(idx, _mm256_xor_si256(_mm256_shuffle_epi8(lut, _mm256_and_si256(idx, m15)), _mm256_shuffle_epi8(lut, hi)));
}
// +-1 bytes of the 4 groups of one sub-block from a lane-duplicated vector of sign bytes
IQ_AVX2 inline __m256i sign_mask(__m256i src, const uint8_t* ctrl) {
    const __m256i km = _mm256_load_si256((const __m256i*)g_sgn.kmask);
    const __m256i x = _mm256_shuffle_epi8(src, _mm256_load_si256((const __m256i*)ctrl));
    return _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(x, km), km), _mm256_load_si256((const __m256i*)g_sgn.ones));
}

template <int TYPE, int TG>
IQ_AVX2 static void iq_group(const uint8_t* wg, int nsb, const int8_t* const* ap, const float* const* sp, __m256 (*acc)[4]) {
    constexpr int SB = kQkSbBytes[TYPE];
    constexpr float KS = TYPE == QK_IQ2XS ? 0.125f : TYPE == QK_IQ3XXS ? 0.25f : 1.0f;
    const __m256i m15 = _mm256_set1_epi8(15);
    const __m256i lut = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)kvalues_iq4nl));
    for (int t = 0; t < nsb; t++) {
        const uint8_t* gp = wg + (size_t)t * 4 * SB;
        for (int r = 0; r < 4; r++) {
            const uint8_t* p = gp + r * SB;
            __m256i is[TG];
            for (int k = 0; k < TG; k++) is[k] = _mm256_setzero_si256();
            if constexpr (TYPE == QK_IQ2XS) {
                // per-superblock scale vectors for all 8 sub-blocks: byte pairs (2*lo+1, 2*hi+1) -> pshufb into [lo x8 int16 | hi x8 int16]
                const __m128i s8 = _mm_loadl_epi64((const __m128i*)(p + 66));
                const __m128i lo4 = _mm_and_si128(s8, _mm256_castsi256_si128(m15)), hi4 = _mm_and_si128(_mm_srli_epi16(s8, 4), _mm256_castsi256_si128(m15));
                const __m128i one = _mm_set1_epi8(1);
                const __m128i sx = _mm_unpacklo_epi8(_mm_add_epi8(_mm_add_epi8(lo4, lo4), one), _mm_add_epi8(_mm_add_epi8(hi4, hi4), one));
                const __m256i sxx = _mm256_broadcastsi128_si256(sx);
                const __m256i qa = _mm256_srli_epi16(_mm256_loadu_si256((const __m256i*)(p + 2)), 9), qb = _mm256_srli_epi16(_mm256_loadu_si256((const __m256i*)(p + 34)), 9);
                const __m256i sg = sign_bytes(_mm256_packus_epi16(qa, qb));
                const __m256i sp0 = _mm256_permute2x128_si256(sg, sg, 0x00), sp1 = _mm256_permute2x128_si256(sg, sg, 0x11);
#pragma GCC unroll 8
                for (int ib = 0; ib < 8; ib++) {
                    const uint64_t w = ld64(p + 2 + 8 * ib);
                    __m128i g0 = _mm_loadl_epi64((const __m128i*)&iq2xs_grid[w & 511]);
                    __m128i g1 = _mm_loadl_epi64((const __m128i*)&iq2xs_grid[(w >> 32) & 511]);
                    g0 = _mm_castpd_si128(_mm_loadh_pd(_mm_castsi128_pd(g0), (const double*)&iq2xs_grid[(w >> 16) & 511]));
                    g1 = _mm_castpd_si128(_mm_loadh_pd(_mm_castsi128_pd(g1), (const double*)&iq2xs_grid[w >> 48 & 511]));
                    const __m256i G = _mm256_set_m128i(g1, g0);
                    const __m256i S = sign_mask((ib >> 1) & 1 ? sp1 : sp0, g_sgn.v2[ib]);
                    const __m256i sv = _mm256_shuffle_epi8(sxx, _mm256_load_si256((const __m256i*)g_scsel[ib]));
                    for (int k = 0; k < TG; k++) is[k] = _mm256_add_epi32(is[k], _mm256_madd_epi16(dot_sgn(G, S, ap[k] + t * 256 + 32 * ib), sv));
                }
            } else if constexpr (TYPE == QK_IQ3XXS) {
                // 7-bit sign indices of all 8 sub-blocks: dword ib = 4 index bytes (idx_l = (aux >> 7l) & 127)
                const __m256i D = _mm256_loadu_si256((const __m256i*)(p + 66));
                const __m256i X = _mm256_or_si256(_mm256_or_si256(_mm256_and_si256(D, _mm256_set1_epi32(0x7f)), _mm256_and_si256(_mm256_slli_epi32(D, 1), _mm256_set1_epi32(0x7f00))),
                                                  _mm256_or_si256(_mm256_and_si256(_mm256_slli_epi32(D, 2), _mm256_set1_epi32(0x7f0000)), _mm256_and_si256(_mm256_slli_epi32(D, 3), _mm256_set1_epi32(0x7f000000))));
                const __m256i sg = sign_bytes(X);
                const __m256i sq0 = _mm256_permute2x128_si256(sg, sg, 0x00), sq1 = _mm256_permute2x128_si256(sg, sg, 0x11);
#pragma GCC unroll 8
                for (int ib = 0; ib < 8; ib++) {
                    const uint32_t aux = ld32(p + 66 + 4 * ib);
                    auto g32 = [&](int k) { return (const float*)&iq3xxs_grid[p[2 + 8 * ib + k]]; };   // direct byte loads: fewer ALU uops than shifting one u64
                    __m128 ga = _mm_load_ss(g32(0));
                    ga = _mm_insert_ps(ga, _mm_load_ss(g32(1)), 0x10);
                    ga = _mm_insert_ps(ga, _mm_load_ss(g32(2)), 0x20);
                    ga = _mm_insert_ps(ga, _mm_load_ss(g32(3)), 0x30);
                    __m128 gb = _mm_load_ss(g32(4));
                    gb = _mm_insert_ps(gb, _mm_load_ss(g32(5)), 0x10);
                    gb = _mm_insert_ps(gb, _mm_load_ss(g32(6)), 0x20);
                    gb = _mm_insert_ps(gb, _mm_load_ss(g32(7)), 0x30);
                    const __m256i G = _mm256_castps_si256(_mm256_set_m128(gb, ga));
                    const __m256i S = sign_mask(ib >> 2 ? sq1 : sq0, g_sgn.v3[ib]);
                    const __m256i sv = _mm256_set1_epi16((short)(2 * (aux >> 28) + 1));
                    for (int k = 0; k < TG; k++) is[k] = _mm256_add_epi32(is[k], _mm256_madd_epi16(dot_sgn(G, S, ap[k] + t * 256 + 32 * ib), sv));
                }
            } else {
                const uint16_t sh = ld16(p + 2);
                const uint8_t* sl = p + 4;
                for (int ib = 0; ib < 8; ib++) {
                    const __m128i vq = _mm_loadu_si128((const __m128i*)(p + 8 + 16 * ib));
                    const __m128i lo = _mm_and_si128(vq, _mm256_castsi256_si128(m15)), hi = _mm_and_si128(_mm_srli_epi16(vq, 4), _mm256_castsi256_si128(m15));
                    const __m256i W = _mm256_shuffle_epi8(lut, _mm256_set_m128i(hi, lo));   // s8 weights, in order
                    const __m256i Wa = _mm256_abs_epi8(W);
                    const int ls = ((sl[ib / 2] >> (4 * (ib & 1))) & 15) | (((sh >> (2 * ib)) & 3) << 4);
                    const __m256i sv = _mm256_set1_epi16((short)(ls - 32));
                    for (int k = 0; k < TG; k++) is[k] = _mm256_add_epi32(is[k], _mm256_madd_epi16(dot_sgn(Wa, W, ap[k] + t * 256 + 32 * ib), sv));
                }
            }
            const float d = f16s(p) * KS;
            for (int k = 0; k < TG; k++) {
                const __m256 dv = _mm256_set1_ps(sp ? d * sp[k][t] : d);
                acc[k][r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(is[k]), dv, acc[k][r]);
            }
        }
    }
}

template <int TYPE, int TG>
IQ_AVX2 static void iq_pass(const QkView& v, const int8_t* q, size_t qs, int e0, float* yE, size_t yst, int g0, int g1, const float* ss) {
    const int8_t* ap[TG];
    const float* sp[TG];
    for (int k = 0; k < TG; k++) { ap[k] = q + (size_t)(e0 + k) * qs; sp[k] = ss ? ss + (size_t)(e0 + k) * v.nsb : nullptr; }
    constexpr int SB = kQkSbBytes[TYPE];
    for (int g = g0; g < g1; g++) {
        __m256 acc[TG][4];
        for (int k = 0; k < TG; k++) for (int r = 0; r < 4; r++) acc[k][r] = _mm256_setzero_ps();
        iq_group<TYPE, TG>(v.w + (size_t)g * v.nsb * 4 * SB, v.nsb, ap, ss ? sp : nullptr, acc);
        for (int k = 0; k < TG; k++) for (int r = 0; r < 4; r++) yE[(size_t)(e0 + k) * yst + (size_t)g * 4 + r] = hsum8(acc[k][r]);
    }
}

template <int TYPE>
IQ_AVX2 static void iq_type(const QkView& v, const int8_t* q, size_t qs, int E, float* yE, size_t yst, int g0, int g1, const float* ss) {
    for (int e0 = 0; e0 < E;) {
        const int tg = std::min(4, E - e0);
        switch (tg) {
            case 1: iq_pass<TYPE, 1>(v, q, qs, e0, yE, yst, g0, g1, ss); break;
            case 2: iq_pass<TYPE, 2>(v, q, qs, e0, yE, yst, g0, g1, ss); break;
            case 3: iq_pass<TYPE, 3>(v, q, qs, e0, yE, yst, g0, g1, ss); break;
            default: iq_pass<TYPE, 4>(v, q, qs, e0, yE, yst, g0, g1, ss); break;
        }
        e0 += tg;
    }
}
#endif
}  // namespace

void iq_gemm(Isa isa, const QkView& v, const int8_t* q, size_t qs, int E, float* yE, size_t yst, int g0, int g1, const float* ss) {
#if defined(__x86_64__) || defined(_M_X64)
    if (isa == Isa::Avx2 || isa == Isa::Avx512) {
        switch (v.type) {
            case QK_IQ2XS: iq_type<QK_IQ2XS>(v, q, qs, E, yE, yst, g0, g1, ss); return;
            case QK_IQ3XXS: iq_type<QK_IQ3XXS>(v, q, qs, E, yE, yst, g0, g1, ss); return;
            default: iq_type<QK_IQ4XS>(v, q, qs, E, yE, yst, g0, g1, ss); return;
        }
    }
#endif
    iq_scalar(v, q, qs, E, yE, yst, g0, g1, ss);
}

void iq_dequant_row_sb(int type, const uint8_t* p, float* y) {
    const float d = f16_to_f32(ld16(p));
    if (type == QK_IQ2XS) {
        for (int ib = 0; ib < 8; ib++) {
            const float db[2] = {d * (0.5f + (p[66 + ib] & 15)) * 0.25f, d * (0.5f + (p[66 + ib] >> 4)) * 0.25f};
            for (int l = 0; l < 4; l++) {
                const uint16_t e = ld16(p + 2 + 2 * (4 * ib + l));
                const uint8_t* grid = (const uint8_t*)(iq2xs_grid + (e & 511));
                const uint8_t sg = ksigns_iq2xs[e >> 9];
                for (int j = 0; j < 8; j++) *y++ = db[l / 2] * grid[j] * ((sg & kmask_iq2xs[j]) ? -1.f : 1.f);
            }
        }
    } else if (type == QK_IQ3XXS) {
        for (int ib = 0; ib < 8; ib++) {
            const uint32_t aux = ld32(p + 66 + 4 * ib);
            const float db = d * (0.5f + (aux >> 28)) * 0.5f;
            for (int l = 0; l < 4; l++) {
                const uint8_t sg = ksigns_iq2xs[(aux >> (7 * l)) & 127];
                const uint8_t* g1 = (const uint8_t*)(iq3xxs_grid + p[2 + 8 * ib + 2 * l]);
                const uint8_t* g2 = (const uint8_t*)(iq3xxs_grid + p[2 + 8 * ib + 2 * l + 1]);
                for (int j = 0; j < 4; j++) {
                    y[j] = db * g1[j] * ((sg & kmask_iq2xs[j]) ? -1.f : 1.f);
                    y[j + 4] = db * g2[j] * ((sg & kmask_iq2xs[j + 4]) ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    } else {
        const uint16_t sh = ld16(p + 2);
        for (int ib = 0; ib < 8; ib++) {
            const int ls = ((p[4 + ib / 2] >> (4 * (ib & 1))) & 15) | (((sh >> (2 * ib)) & 3) << 4);
            const float dl = d * (float)(ls - 32);
            for (int j = 0; j < 16; j++) {
                y[j] = dl * kvalues_iq4nl[p[8 + 16 * ib + j] & 15];
                y[j + 16] = dl * kvalues_iq4nl[p[8 + 16 * ib + j] >> 4];
            }
            y += 32;
        }
    }
}

int run_iq_selftest(std::string& report) {
    std::mt19937_64 rng(4242);
    int checks = 0, failures = 0;
    char buf[256];
    const Isa isas[] = {Isa::Scalar, Isa::Avx2};
    const int shapes[][3] = {{16, 3, 7}, {8, 8, 1}, {32, 2, 2}, {12, 4, 9}, {64, 1, 5}};   // rows, superblocks, slots
    for (int type : {QK_IQ2XS, QK_IQ3XXS, QK_IQ4XS}) for (const auto& shp : shapes) {
        const int SB = kQkSbBytes[type], rows = shp[0], nsb = shp[1], n = nsb * 256, ng = rows / 4;
        std::vector<uint8_t> w((size_t)rows * nsb * SB);
        for (auto& b : w) b = (uint8_t)rng();
        for (size_t i = 0; i < w.size(); i += SB) {   // sane scale d (f16 ~ 0.01..0.1)
            const uint16_t h = f32_to_f16(0.01f + 0.05f * (float)(rng() % 1000) / 1000.f);
            std::memcpy(&w[i], &h, 2);
        }
        const int E = shp[2];
        std::vector<int8_t> q((size_t)E * n);
        for (auto& x : q) x = (int8_t)((int)(rng() % 255) - 127);
        std::vector<float> ss((size_t)E * nsb);
        for (auto& x : ss) x = 0.5f + (float)(rng() % 100) / 100.f;
        QkView v; v.type = type; v.rows = rows; v.nsb = nsb; v.w = w.data();
        // double reference from the dequantised weights
        std::vector<double> ref((size_t)E * rows), refs((size_t)E * rows);
        std::vector<float> dq(256);
        for (int g = 0; g < ng; g++) for (int r = 0; r < 4; r++) for (int e = 0; e < E; e++) {
            double a = 0, as = 0;
            for (int t = 0; t < nsb; t++) {
                iq_dequant_row_sb(type, w.data() + (((size_t)g * nsb + t) * 4 + r) * SB, dq.data());
                double s = 0;
                for (int i = 0; i < 256; i++) s += (double)dq[i] * q[(size_t)e * n + t * 256 + i];
                a += s; as += s * ss[(size_t)e * nsb + t];
            }
            ref[(size_t)e * rows + g * 4 + r] = a; refs[(size_t)e * rows + g * 4 + r] = as;
        }
        for (Isa isa : isas) {
            if (!isa_available(isa)) continue;
            for (int use_ss = 0; use_ss < 2; use_ss++) {
                std::vector<float> y((size_t)E * rows, -1.f), y1((size_t)E * rows, -2.f);
                iq_gemm(isa, v, q.data(), (size_t)n, E, y.data(), (size_t)rows, 0, ng, use_ss ? ss.data() : nullptr);
                const auto& rr = use_ss ? refs : ref;
                bool ok = true;
                double nr = 0; for (double x : rr) nr += x * x;
                nr = std::sqrt(nr / rr.size());
                for (size_t i = 0; i < y.size() && ok; i++) ok = std::fabs(y[i] - rr[i]) <= 2e-4 * nr + 1e-3;
                // slot independence: every slot alone == the batched result bit for bit
                bool same = true;
                for (int e = 0; e < E && same; e++) {
                    std::vector<float> ye((size_t)rows);
                    iq_gemm(isa, v, q.data() + (size_t)e * n, (size_t)n, 1, ye.data(), (size_t)rows, 0, ng, use_ss ? ss.data() + (size_t)e * nsb : nullptr);
                    same = std::memcmp(ye.data(), &y[(size_t)e * rows], (size_t)rows * 4) == 0;
                }
                checks += 2;
                if (!ok) { failures++; std::snprintf(buf, sizeof buf, "FAIL iq type %d %s ss=%d vs double reference\n", type, isa_name(isa), use_ss); report += buf; }
                if (!same) { failures++; std::snprintf(buf, sizeof buf, "FAIL iq type %d %s ss=%d slot independence\n", type, isa_name(isa), use_ss); report += buf; }
            }
        }
    }
    std::snprintf(buf, sizeof buf, "iq selftest (IQ2_XS/IQ3_XXS/IQ4_XS, scalar+avx2): %d checks, %d failures\n", checks, failures);
    report += buf;
    return failures;
}

}  // namespace lb
