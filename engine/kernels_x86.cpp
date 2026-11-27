// x86-64 T-MAC LUT kernels. Compiled with function-level target attributes so one binary
// contains scalar + AVX2 + AVX-512(VBMI) paths and selects at runtime.
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

#include <algorithm>
#include <cstring>

#include "binary_gemv.h"

namespace lb {

// ------------------------------------------------------------------ AVX2 (pshufb) ----
// 64 rows per row block = 2 x 32-row halves. _mm256_shuffle_epi8 looks up 32 rows at once
// (the same 16-entry table is broadcast into both 128-bit lanes).
#define LB_AVX2 __attribute__((target("avx2")))

LB_AVX2 static inline void avx2_group(__m256i& accL, __m256i& accH, __m256i tlo, __m256i thi, __m256i idx) {
    __m256i lo = _mm256_shuffle_epi8(tlo, idx);
    __m256i hi = _mm256_shuffle_epi8(thi, idx);
    accL = _mm256_add_epi16(accL, _mm256_unpacklo_epi8(lo, hi));
    accH = _mm256_add_epi16(accH, _mm256_unpackhi_epi8(lo, hi));
}

LB_AVX2 static inline void avx2_widen_add(int32_t* dst, __m256i a) {
    __m256i w0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(a));
    __m256i w1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a, 1));
    _mm256_storeu_si256((__m256i*)dst, _mm256_add_epi32(_mm256_loadu_si256((const __m256i*)dst), w0));
    _mm256_storeu_si256((__m256i*)(dst + 8), _mm256_add_epi32(_mm256_loadu_si256((const __m256i*)(dst + 8)), w1));
}

LB_AVX2 void gemv_avx2(const BinMatView& w, const int8_t* luts, int T, int32_t* out, int rb_begin, int rb_end) {
    if (rb_end < 0) rb_end = pad_rows(w.rows) / kRowBlock;
    const int rp = pad_rows(w.rows), kp = pad_k(w.cols);
    const int pairs = kp / 8, ntile = kp / kTileK;
    const size_t lutTok = lut_bytes_per_token(kp);
    const __m256i m0f = _mm256_set1_epi8(0x0f);
    alignas(32) int32_t acc[kMaxBatch][64];

    for (int rb = rb_begin; rb < rb_end; rb++) {
        const uint8_t* wb = w.lut_w + (size_t)rb * pairs * 64;
        std::memset(acc, 0, sizeof(acc));
        for (int tile = 0; tile < ntile; tile++) {
            const uint8_t* wp0 = wb + (size_t)tile * 16 * 64;
            for (int t = 0; t < T; t++) {
                const int8_t* L = luts + t * lutTok + (size_t)tile * kTileGroups * 32;
                __m256i a0l = _mm256_setzero_si256(), a0h = a0l, a1l = a0l, a1h = a0l;
                const uint8_t* wp = wp0;
                for (int p = 0; p < 16; p++, wp += 64, L += 64) {
                    __m256i w0 = _mm256_loadu_si256((const __m256i*)wp);
                    __m256i w1 = _mm256_loadu_si256((const __m256i*)(wp + 32));
                    __m256i t0l = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)L));
                    __m256i t0h = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(L + 16)));
                    __m256i t1l = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(L + 32)));
                    __m256i t1h = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(L + 48)));
                    // low nibbles -> group 2p, high nibbles -> group 2p+1
                    avx2_group(a0l, a0h, t0l, t0h, _mm256_and_si256(w0, m0f));
                    avx2_group(a1l, a1h, t0l, t0h, _mm256_and_si256(w1, m0f));
                    avx2_group(a0l, a0h, t1l, t1h, _mm256_and_si256(_mm256_srli_epi16(w0, 4), m0f));
                    avx2_group(a1l, a1h, t1l, t1h, _mm256_and_si256(_mm256_srli_epi16(w1, 4), m0f));
                }
                avx2_widen_add(&acc[t][0], a0l);
                avx2_widen_add(&acc[t][16], a0h);
                avx2_widen_add(&acc[t][32], a1l);
                avx2_widen_add(&acc[t][48], a1h);
            }
        }
        // storage index s = R*16 + e ; R: 0=a0l 1=a0h 2=a1l 3=a1h ; row = half*32 + 16*(e/8) + isH*8 + e%8
        for (int t = 0; t < T; t++) {
            int32_t* o = out + (size_t)t * rp + rb * kRowBlock;
            for (int R = 0; R < 4; R++)
                for (int e = 0; e < 16; e++) o[(R >> 1) * 32 + 16 * (e >> 3) + (R & 1) * 8 + (e & 7)] = acc[t][R * 16 + e];
        }
    }
}

// ------------------------------------------------- AVX-512 VBMI (vpermb, 64 rows) ----
#define LB_AVX512 __attribute__((target("avx512f,avx512bw,avx512vbmi,avx512vl,avx2")))

LB_AVX512 static inline void avx512_group(__m512i& accL, __m512i& accH, __m512i tlo, __m512i thi, __m512i idx) {
    __m512i lo = _mm512_permutexvar_epi8(idx, tlo);
    __m512i hi = _mm512_permutexvar_epi8(idx, thi);
    accL = _mm512_add_epi16(accL, _mm512_unpacklo_epi8(lo, hi));
    accH = _mm512_add_epi16(accH, _mm512_unpackhi_epi8(lo, hi));
}

LB_AVX512 static inline void avx512_widen_add(int32_t* dst, __m512i a) {
    __m512i w0 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(a));
    __m512i w1 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(a, 1));
    _mm512_storeu_si512(dst, _mm512_add_epi32(_mm512_loadu_si512(dst), w0));
    _mm512_storeu_si512(dst + 16, _mm512_add_epi32(_mm512_loadu_si512(dst + 16), w1));
}

#define LB_AVX512V __attribute__((target("avx512f,avx512bw,avx512vbmi,avx512vl,avx512vnni,avx2")))

// NT tokens x one 16-row group x nw weight vectors. One vector = 16 rows x 4 pairs (8 groups); a vpermb with a 64-byte
// table (4 groups x 16 entries) looks up 4 groups for 16 rows at once, vpdpbusd then sums the 4 adjacent bytes of a row.
// The int16 table entries come as lo/hi byte planes: row sum = sum(lo) + 256 * sum(hi_signed).
#ifndef LB_PF
#define LB_PF 16
#endif
template <int NT>
LB_AVX512V static void tg_run(const uint8_t* wp, int nw, const int8_t* L, size_t lutTok, int32_t* accbuf) {
    __m512i a[NT][4];
    for (int t = 0; t < NT; t++)
        for (int k = 0; k < 4; k++) a[t][k] = _mm512_load_si512((const void*)(accbuf + (t * 4 + k) * 16));
    const __m512i m0f = _mm512_set1_epi8(0x0f), off = _mm512_set1_epi32(0x30201000), ones = _mm512_set1_epi8(1);
    for (int i = 0; i < nw; i++) {
        _mm_prefetch((const char*)(wp + (size_t)(i + LB_PF) * 64), _MM_HINT_T0);
        const __m512i wv = _mm512_loadu_si512((const void*)(wp + (size_t)i * 64));
        const __m512i A = _mm512_ternarylogic_epi32(wv, m0f, off, 0xEA);                          // (wv & 0x0f) | off
        const __m512i B = _mm512_ternarylogic_epi32(_mm512_srli_epi16(wv, 4), m0f, off, 0xEA);
        const int8_t* base = L + (size_t)i * 256;
        for (int t = 0; t < NT; t++) {
            const int8_t* bt = base + (size_t)t * lutTok;
            const __m512i alo = _mm512_permutexvar_epi8(A, _mm512_loadu_si512((const void*)bt));
            const __m512i ahi = _mm512_permutexvar_epi8(A, _mm512_loadu_si512((const void*)(bt + 64)));
            const __m512i blo = _mm512_permutexvar_epi8(B, _mm512_loadu_si512((const void*)(bt + 128)));
            const __m512i bhi = _mm512_permutexvar_epi8(B, _mm512_loadu_si512((const void*)(bt + 192)));
            a[t][0] = _mm512_dpbusd_epi32(a[t][0], alo, ones);
            a[t][1] = _mm512_dpbusd_epi32(a[t][1], blo, ones);
            a[t][2] = _mm512_dpbusd_epi32(a[t][2], ones, ahi);
            a[t][3] = _mm512_dpbusd_epi32(a[t][3], ones, bhi);
        }
    }
    for (int t = 0; t < NT; t++)
        for (int k = 0; k < 4; k++) _mm512_store_si512((void*)(accbuf + (t * 4 + k) * 16), a[t][k]);
}

LB_AVX512V void gemv_avx512(const BinMatView& w, const int8_t* luts, int T, int32_t* out, int rb_begin, int rb_end, int k_begin, int k_end) {
    if (rb_end < 0) rb_end = pad_rows(w.rows) / kRowBlock;
    const int rp = pad_rows(w.rows), kp = pad_k(w.cols);
    const int nwv = kp / 32;
    const size_t lutTok = lut_bytes_per_token(kp);
    alignas(64) int32_t buf[4][kMaxBatch][4][16];
    constexpr int kChunk = kKChunk;  // vectors per K chunk: T tokens x 16 x 256 B of LUT stay L1/L2 resident while 4 row groups stream past
    for (int rb = rb_begin; rb < rb_end; rb++) {
        std::memset(buf, 0, sizeof(buf));
        const uint8_t* ob = w.w512 + (size_t)rb * nwv * 256;
        const int c_end = k_end < 0 ? nwv : std::min(nwv, k_end * kChunk);
        for (int c0 = k_begin * kChunk; c0 < c_end; c0 += kChunk) {
            const int nc = std::min(kChunk, nwv - c0);
            for (int g = 0; g < 4; g++) {
                const uint8_t* wp = ob + (size_t)c0 * 256 + (size_t)g * nc * 64;
                for (int t0 = 0; t0 < T; t0 += 4) {
                    const int nt = std::min(4, T - t0);
                    const int8_t* L = luts + (size_t)t0 * lutTok + (size_t)c0 * 256;
                    int32_t* ab = &buf[g][t0][0][0];
                    switch (nt) {
                        case 4: tg_run<4>(wp, nc, L, lutTok, ab); break;
                        case 3: tg_run<3>(wp, nc, L, lutTok, ab); break;
                        case 2: tg_run<2>(wp, nc, L, lutTok, ab); break;
                        default: tg_run<1>(wp, nc, L, lutTok, ab); break;
                    }
                }
            }
        }
        for (int g = 0; g < 4; g++)
            for (int t = 0; t < T; t++) {
                const __m512i lo = _mm512_add_epi32(_mm512_load_si512((const void*)buf[g][t][0]), _mm512_load_si512((const void*)buf[g][t][1]));
                const __m512i hi = _mm512_add_epi32(_mm512_load_si512((const void*)buf[g][t][2]), _mm512_load_si512((const void*)buf[g][t][3]));
                _mm512_storeu_si512((void*)(out + (size_t)t * rp + (size_t)rb * kRowBlock + g * 16), _mm512_add_epi32(lo, _mm512_slli_epi32(hi, 8)));
            }
    }
}

// ------------------------------------------------------------------ bit-plane grid kernels ----
// AVX2: per (row block, tile, activation): the P planes are accumulated in int16 exactly like gemv_avx2 (one tile = 32 groups),
// then Horner-combined in int32 (a = 2a + S_p), converted to float, multiplied by the tile group scale and summed over tiles.
LB_AVX2 static inline void avx2_widen_h(int32_t* dst, __m256i a, bool first) {
    __m256i w0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(a));
    __m256i w1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a, 1));
    if (!first) {
        w0 = _mm256_add_epi32(w0, _mm256_slli_epi32(_mm256_loadu_si256((const __m256i*)dst), 1));
        w1 = _mm256_add_epi32(w1, _mm256_slli_epi32(_mm256_loadu_si256((const __m256i*)(dst + 8)), 1));
    }
    _mm256_storeu_si256((__m256i*)dst, w0);
    _mm256_storeu_si256((__m256i*)(dst + 8), w1);
}

__attribute__((target("avx2,fma"))) void grid_avx2(const GridView& g, const int8_t* luts, int E, float* yE, int rb_begin, int rb_end) {
    const int rp = pad_rows(g.rows);
    const size_t tileBytes = (size_t)rp * 16;
    const __m256i m0f = _mm256_set1_epi8(0x0f);
    alignas(32) float yacc[kMaxBatch][64];
    alignas(32) int32_t a32[64];
    for (int rb = rb_begin; rb < rb_end; rb++) {
        std::memset(yacc, 0, sizeof(yacc));
        for (int tile = 0; tile < g.ntile; tile++) {
            const float* gsr = g.gs + (size_t)tile * g.gs_stride + (size_t)rb * kRowBlock;
            if (tile + 1 < g.ntile)
                for (int p = 0; p < g.P; p++) {
                    const char* nx = (const char*)(g.w[p] + (size_t)(tile + 1) * tileBytes + (size_t)rb * 1024);
                    for (int o = 0; o < 1024; o += 64) _mm_prefetch(nx + o, _MM_HINT_T0);
                }
            for (int e = 0; e < E; e++) {
                const int8_t* L0 = luts + ((size_t)tile * E + e) * kTileLutBytes;
                for (int p = 0; p < g.P; p++) {
                    const uint8_t* wp = g.w[p] + (size_t)tile * tileBytes + (size_t)rb * 1024;
                    const int8_t* L = L0;
                    __m256i a0l = _mm256_setzero_si256(), a0h = a0l, a1l = a0l, a1h = a0l;
                    for (int q = 0; q < 16; q++, wp += 64, L += 64) {
                        __m256i w0 = _mm256_loadu_si256((const __m256i*)wp);
                        __m256i w1 = _mm256_loadu_si256((const __m256i*)(wp + 32));
                        __m256i t0l = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)L));
                        __m256i t0h = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(L + 16)));
                        __m256i t1l = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(L + 32)));
                        __m256i t1h = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(L + 48)));
                        avx2_group(a0l, a0h, t0l, t0h, _mm256_and_si256(w0, m0f));
                        avx2_group(a1l, a1h, t0l, t0h, _mm256_and_si256(w1, m0f));
                        avx2_group(a0l, a0h, t1l, t1h, _mm256_and_si256(_mm256_srli_epi16(w0, 4), m0f));
                        avx2_group(a1l, a1h, t1l, t1h, _mm256_and_si256(_mm256_srli_epi16(w1, 4), m0f));
                    }
                    const bool first = p == 0;
                    avx2_widen_h(&a32[0], a0l, first);
                    avx2_widen_h(&a32[16], a0h, first);
                    avx2_widen_h(&a32[32], a1l, first);
                    avx2_widen_h(&a32[48], a1h, first);
                }
                float* ya = yacc[e];
                for (int j = 0; j < 64; j += 8)
                    _mm256_store_ps(ya + j, _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_load_si256((const __m256i*)(a32 + j))),
                                                           _mm256_loadu_ps(gsr + j), _mm256_load_ps(ya + j)));
            }
        }
        for (int e = 0; e < E; e++) {
            float* o = yE + (size_t)e * rp + (size_t)rb * kRowBlock;
            for (int s = 0; s < 64; s++) o[grid_avx2_perm(s)] = yacc[e][s];
        }
    }
}

// AVX-512: the plane weight 2^(P-1-p) rides on the vpdpbusd multiplier operand (instead of `ones`), so all planes of a tile
// accumulate into the same 4 int32 accumulators; lo + 256*hi gives sum_p 2^(P-1-p) S_p directly.
template <int NT>
LB_AVX512V static void grid512_run(const GridView& g, const int8_t* luts, int E, int e0, float* yE, int rb, int rp) {
    const size_t tileBytes = (size_t)rp * 16;
    const __m512i m0f = _mm512_set1_epi8(0x0f), off = _mm512_set1_epi32(0x30201000);
    for (int gr = 0; gr < 4; gr++) {
        __m512 y[NT];
        for (int t = 0; t < NT; t++) y[t] = _mm512_setzero_ps();
        for (int tile = 0; tile < g.ntile; tile++) {
            __m512i a[NT][4];
            for (int t = 0; t < NT; t++)
                for (int k = 0; k < 4; k++) a[t][k] = _mm512_setzero_si512();
            for (int p = 0; p < g.P; p++) {
                const uint8_t* wp = g.w[p] + (size_t)tile * tileBytes + (size_t)rb * 1024 + (size_t)gr * 256;
                const __m512i wm = _mm512_set1_epi8((char)(1 << (g.P - 1 - p)));
                for (int i = 0; i < 4; i++) {
                    const __m512i wv = _mm512_loadu_si512((const void*)(wp + (size_t)i * 64));
                    const __m512i A = _mm512_ternarylogic_epi32(wv, m0f, off, 0xEA);
                    const __m512i B = _mm512_ternarylogic_epi32(_mm512_srli_epi16(wv, 4), m0f, off, 0xEA);
                    for (int t = 0; t < NT; t++) {
                        const int8_t* bt = luts + ((size_t)tile * E + e0 + t) * kTileLutBytes + (size_t)i * 256;
                        const __m512i alo = _mm512_permutexvar_epi8(A, _mm512_loadu_si512((const void*)bt));
                        const __m512i ahi = _mm512_permutexvar_epi8(A, _mm512_loadu_si512((const void*)(bt + 64)));
                        const __m512i blo = _mm512_permutexvar_epi8(B, _mm512_loadu_si512((const void*)(bt + 128)));
                        const __m512i bhi = _mm512_permutexvar_epi8(B, _mm512_loadu_si512((const void*)(bt + 192)));
                        a[t][0] = _mm512_dpbusd_epi32(a[t][0], alo, wm);
                        a[t][1] = _mm512_dpbusd_epi32(a[t][1], blo, wm);
                        a[t][2] = _mm512_dpbusd_epi32(a[t][2], wm, ahi);
                        a[t][3] = _mm512_dpbusd_epi32(a[t][3], wm, bhi);
                    }
                }
            }
            const __m512 gsv = _mm512_loadu_ps(g.gs + (size_t)tile * g.gs_stride + (size_t)rb * kRowBlock + gr * 16);
            for (int t = 0; t < NT; t++) {
                const __m512i lo = _mm512_add_epi32(a[t][0], a[t][1]);
                const __m512i hi = _mm512_add_epi32(a[t][2], a[t][3]);
                y[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(lo, _mm512_slli_epi32(hi, 8))), gsv, y[t]);
            }
        }
        for (int t = 0; t < NT; t++) _mm512_storeu_ps(yE + (size_t)(e0 + t) * rp + (size_t)rb * kRowBlock + gr * 16, y[t]);
    }
}

LB_AVX512V void grid_avx512(const GridView& g, const int8_t* luts, int E, float* yE, int rb_begin, int rb_end) {
    const int rp = pad_rows(g.rows);
    for (int rb = rb_begin; rb < rb_end; rb++)
        for (int e0 = 0; e0 < E; e0 += 4) {
            switch (std::min(4, E - e0)) {
                case 4: grid512_run<4>(g, luts, E, e0, yE, rb, rp); break;
                case 3: grid512_run<3>(g, luts, E, e0, yE, rb, rp); break;
                case 2: grid512_run<2>(g, luts, E, e0, yE, rb, rp); break;
                default: grid512_run<1>(g, luts, E, e0, yE, rb, rp); break;
            }
        }
}

}  // namespace lb
#endif
