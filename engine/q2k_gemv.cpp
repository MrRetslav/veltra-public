#include "q2k_gemv.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "iq_gemv.h"
#include "lb_common.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace lb {

void qk_sums16(const int8_t* q, int n, int16_t* b16) {
    for (int j = 0; j < n / 16; j++) {
        int s = 0;
        for (int i = 0; i < 16; i++) s += q[j * 16 + i];
        b16[j] = (int16_t)s;
    }
}

// ----------------------------------------------------------------------------------------------- decode of one (row, superblock)
namespace {
struct SbDec {
    int u[256];          // unsigned quant (Q3_K: with the hbit, value is u - 4)
    int sc[16], mn[16];  // per 16 weights (Q4_K: per 32, duplicated)
    float d, dmin;
    int off;             // Q3_K: 4 (value = u - off), else 0
};
inline uint16_t ld16(const uint8_t* p) { uint16_t h; std::memcpy(&h, p, 2); return h; }
void sb_decode(int type, const uint8_t* p, SbDec& o) {
    if (type == QK_Q2K) {
        const uint8_t* qs = p + 16;
        for (int j = 0; j < 16; j++) { o.sc[j] = p[j] & 15; o.mn[j] = p[j] >> 4; }
        for (int h = 0; h < 2; h++) for (int j = 0; j < 4; j++) for (int l = 0; l < 32; l++) o.u[128 * h + 32 * j + l] = (qs[32 * h + l] >> (2 * j)) & 3;
        o.d = f16_to_f32(ld16(p + 80)); o.dmin = f16_to_f32(ld16(p + 82)); o.off = 0;
    } else if (type == QK_Q3K) {
        const uint8_t *hm = p, *qs = p + 32;
        for (int j = 0; j < 16; j++) { o.sc[j] = (int8_t)p[96 + j]; o.mn[j] = 0; }
        for (int h = 0; h < 2; h++) for (int j = 0; j < 4; j++) for (int l = 0; l < 32; l++)
            o.u[128 * h + 32 * j + l] = ((qs[32 * h + l] >> (2 * j)) & 3) | (((hm[l] >> (4 * h + j)) & 1) << 2);
        o.d = f16_to_f32(ld16(p + 112)); o.dmin = 0.f; o.off = 4;
    } else {
        const uint8_t* qs = p + 20;
        for (int j = 0; j < 8; j++) { o.sc[2 * j] = o.sc[2 * j + 1] = p[4 + j]; o.mn[2 * j] = o.mn[2 * j + 1] = p[12 + j]; }
        for (int g = 0; g < 4; g++) for (int l = 0; l < 32; l++) { o.u[64 * g + l] = qs[32 * g + l] & 15; o.u[64 * g + 32 + l] = qs[32 * g + l] >> 4; }
        o.d = f16_to_f32(ld16(p)); o.dmin = f16_to_f32(ld16(p + 2)); o.off = 0;
    }
}
}  // namespace

void qk_dequant_row_sb(int type, const uint8_t* sb, float* out) {
    SbDec s;
    sb_decode(type, sb, s);
    for (int i = 0; i < 256; i++) {
        const int j = i / 16;
        const float dl = s.d * (float)s.sc[j], ml = s.dmin * (float)s.mn[j];   // same op order as gguf-py: (d*sc)*q - (dmin*mn)
        out[i] = dl * (float)(s.u[i] - s.off) - ml;
    }
}

void qk_scalar(const QkView& v, const int8_t* q, size_t qs, const int16_t* b16, size_t bs, int E, float* yE, size_t yst, int g0, int g1) {
    const int SB = kQkSbBytes[v.type];
    SbDec s;
    for (int g = g0; g < g1; g++)
        for (int r = 0; r < 4; r++)
            for (int e = 0; e < E; e++) {
                double acc = 0;
                const int8_t* qe = q + (size_t)e * qs;
                const int16_t* be = b16 + (size_t)e * bs;
                for (int t = 0; t < v.nsb; t++) {
                    sb_decode(v.type, v.w + (((size_t)g * v.nsb + t) * 4 + r) * SB, s);
                    long ti = 0, tm = 0;
                    for (int j = 0; j < 16; j++) {
                        long S = 0;
                        for (int i = 0; i < 16; i++) S += (long)(s.u[16 * j + i] - s.off) * qe[t * 256 + 16 * j + i];
                        ti += (long)s.sc[j] * S;
                        tm += (long)s.mn[j] * be[t * 16 + j];
                    }
                    acc += (double)s.d * (double)ti - (double)s.dmin * (double)tm;
                }
                yE[(size_t)e * yst + (size_t)g * 4 + r] = (float)acc;
            }
}

#if defined(__x86_64__) || defined(_M_X64)
#define QK_AVX2 __attribute__((target("avx2,fma,f16c")))

namespace {
alignas(32) uint8_t g_mask[12][32];   // 0..3: Q2_K/Q3_K (word 2j | 2j+1 of a 16-word vector), 4..11: Q4_K (word c replicated)
[[maybe_unused]] bool g_mask_init = [] {
    for (int j = 0; j < 4; j++)
        for (int b = 0; b < 32; b++) {
            const int lane = b / 16, i = b % 16;
            const int word = 2 * j + lane;
            g_mask[j][b] = (uint8_t)(i % 2 ? 2 * word + 1 : 2 * word);
        }
    for (int c = 0; c < 8; c++)
        for (int b = 0; b < 32; b++) g_mask[4 + c][b] = (uint8_t)(2 * c + (b % 2));
    return true;
}();

QK_AVX2 inline float qk_hsum(__m256 v) {
    __m128 x = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_shuffle_ps(x, x, 1));
    return _mm_cvtss_f32(x);
}
QK_AVX2 inline __m256 f16x(const uint8_t* p) { uint16_t h; std::memcpy(&h, p, 2); return _mm256_set1_ps(_cvtsh_ss(h)); }
QK_AVX2 inline __m256i dot32(__m256i w, const int8_t* a, __m256i sv) { return _mm256_madd_epi16(_mm256_maddubs_epi16(w, _mm256_loadu_si256((const __m256i*)a)), sv); }

// One group of 4 rows, TG slots (activation pointers ap[k] = int8 row, bp[k] = int16 sums), accumulate over all superblocks.
template <int TYPE, int TG>
QK_AVX2 static void qk_group(const uint8_t* wg, int nsb, const int8_t* const* ap, const int16_t* const* bp, __m256 (*acc)[4]) {
    constexpr int SB = kQkSbBytes[TYPE];
    const __m256i m3 = _mm256_set1_epi8(3), m4b = _mm256_set1_epi8(4), m15 = _mm256_set1_epi8(15);
    for (int t = 0; t < nsb; t++) {
        const uint8_t* gp = wg + (size_t)t * 4 * SB;
        __m256i bs[TG];
        for (int k = 0; k < TG; k++) bs[k] = _mm256_loadu_si256((const __m256i*)(bp[k] + t * 16));
        for (int r = 0; r < 4; r++) {
            const uint8_t* p = gp + r * SB;
            __m256i is[TG];
            for (int k = 0; k < TG; k++) is[k] = _mm256_setzero_si256();
            if constexpr (TYPE == QK_Q2K || TYPE == QK_Q3K) {
                __m256i sc16;
                const uint8_t* qp;
                if constexpr (TYPE == QK_Q2K) {
                    const __m128i s8 = _mm_loadu_si128((const __m128i*)p);
                    sc16 = _mm256_cvtepu8_epi16(_mm_and_si128(s8, _mm256_castsi256_si128(m15)));
                    qp = p + 16;
                } else {
                    sc16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(p + 96)));
                    qp = p + 32;
                }
                const __m256i Xlo = _mm256_permute2x128_si256(sc16, sc16, 0x00), Xhi = _mm256_permute2x128_si256(sc16, sc16, 0x11);
                const __m256i q0 = _mm256_loadu_si256((const __m256i*)qp), q1 = _mm256_loadu_si256((const __m256i*)(qp + 32));
                [[maybe_unused]] const __m256i hm = _mm256_loadu_si256((const __m256i*)p);
#define QK_CHUNK(C)                                                                                                                  \
    {                                                                                                                                \
        constexpr int c = C, j = c & 3;                                                                                              \
        const __m256i v = c < 4 ? q0 : q1;                                                                                           \
        __m256i w = _mm256_and_si256(j == 0 ? v : _mm256_srli_epi16(v, 2 * j), m3);                                                  \
        if constexpr (TYPE == QK_Q3K) {                                                                                              \
            const __m256i hb = c <= 2 ? _mm256_slli_epi16(hm, 2 - c) : _mm256_srli_epi16(hm, c - 2);               \
            w = _mm256_or_si256(w, _mm256_and_si256(hb, m4b));                                                                       \
        }                                                                                                                            \
        const __m256i sv = _mm256_shuffle_epi8(c < 4 ? Xlo : Xhi, _mm256_load_si256((const __m256i*)g_mask[j]));                    \
        for (int k = 0; k < TG; k++) is[k] = _mm256_add_epi32(is[k], dot32(w, ap[k] + t * 256 + 32 * c, sv));                       \
    }
                QK_CHUNK(0) QK_CHUNK(1) QK_CHUNK(2) QK_CHUNK(3) QK_CHUNK(4) QK_CHUNK(5) QK_CHUNK(6) QK_CHUNK(7)
#undef QK_CHUNK
                if constexpr (TYPE == QK_Q3K) {
                    const __m256 d = f16x(p + 112);
                    for (int k = 0; k < TG; k++) {
                        is[k] = _mm256_sub_epi32(is[k], _mm256_slli_epi32(_mm256_madd_epi16(bs[k], sc16), 2));
                        acc[k][r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(is[k]), d, acc[k][r]);
                    }
                } else {
                    const __m256 d = f16x(p + 80), dm = f16x(p + 82);
                    const __m256i mn16 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(_mm_loadu_si128((const __m128i*)p), 4), _mm256_castsi256_si128(m15)));
                    for (int k = 0; k < TG; k++) {
                        acc[k][r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(is[k]), d, acc[k][r]);
                        acc[k][r] = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(bs[k], mn16)), dm, acc[k][r]);
                    }
                }
            } else {   // Q4_K
                const __m256i X = _mm256_broadcastsi128_si256(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(p + 4))));
                const uint8_t* qp = p + 20;
#define QK_CHUNK4(G)                                                                                                                 \
    {                                                                                                                                \
        const __m256i v = _mm256_loadu_si256((const __m256i*)(qp + 32 * G));                                                         \
        const __m256i wl = _mm256_and_si256(v, m15), wh = _mm256_and_si256(_mm256_srli_epi16(v, 4), m15);                          \
        const __m256i s0 = _mm256_shuffle_epi8(X, _mm256_load_si256((const __m256i*)g_mask[4 + 2 * G]));                             \
        const __m256i s1 = _mm256_shuffle_epi8(X, _mm256_load_si256((const __m256i*)g_mask[5 + 2 * G]));                             \
        for (int k = 0; k < TG; k++) {                                                                                               \
            is[k] = _mm256_add_epi32(is[k], dot32(wl, ap[k] + t * 256 + 64 * G, s0));                                                \
            is[k] = _mm256_add_epi32(is[k], dot32(wh, ap[k] + t * 256 + 64 * G + 32, s1));                                           \
        }                                                                                                                            \
    }
                QK_CHUNK4(0) QK_CHUNK4(1) QK_CHUNK4(2) QK_CHUNK4(3)
#undef QK_CHUNK4
                const __m256 d = f16x(p), dm = f16x(p + 2);
                const __m128i m8 = _mm_loadl_epi64((const __m128i*)(p + 12));
                const __m256i mn16 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(m8, m8));
                for (int k = 0; k < TG; k++) {
                    acc[k][r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(is[k]), d, acc[k][r]);
                    acc[k][r] = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(bs[k], mn16)), dm, acc[k][r]);
                }
            }
        }
    }
}

template <int TYPE, int TG>
QK_AVX2 static void qk_pass(const QkView& v, const int8_t* q, size_t qs, const int16_t* b16, size_t bs, int e0, float* yE, size_t yst, int g0, int g1) {
    const int8_t* ap[TG];
    const int16_t* bp[TG];
    for (int k = 0; k < TG; k++) { ap[k] = q + (size_t)(e0 + k) * qs; bp[k] = b16 + (size_t)(e0 + k) * bs; }
    constexpr int SB = kQkSbBytes[TYPE];
    for (int g = g0; g < g1; g++) {
        __m256 acc[TG][4];
        for (int k = 0; k < TG; k++) for (int r = 0; r < 4; r++) acc[k][r] = _mm256_setzero_ps();
        qk_group<TYPE, TG>(v.w + (size_t)g * v.nsb * 4 * SB, v.nsb, ap, bp, acc);
        for (int k = 0; k < TG; k++) for (int r = 0; r < 4; r++) yE[(size_t)(e0 + k) * yst + (size_t)g * 4 + r] = qk_hsum(acc[k][r]);
    }
}
template <int TYPE>
QK_AVX2 static void qk_slots(const QkView& v, const int8_t* q, size_t qs, const int16_t* b16, size_t bs, int E, float* yE, size_t yst, int g0, int g1) {
    for (int e0 = 0; e0 < E;) {
        const int tg = std::min(4, E - e0);
        switch (tg) {
            case 1: qk_pass<TYPE, 1>(v, q, qs, b16, bs, e0, yE, yst, g0, g1); break;
            case 2: qk_pass<TYPE, 2>(v, q, qs, b16, bs, e0, yE, yst, g0, g1); break;
            case 3: qk_pass<TYPE, 3>(v, q, qs, b16, bs, e0, yE, yst, g0, g1); break;
            default: qk_pass<TYPE, 4>(v, q, qs, b16, bs, e0, yE, yst, g0, g1); break;
        }
        e0 += tg;
    }
}
// More than 4 slots (batched decode): every 4-row weight group (nsb * 4 * SB bytes, about 10-30 KB) serves all slot blocks while it is in L1 / L2,
// so a step streams the weights from memory once instead of once per 4 slots. Same per-slot arithmetic, bit-identical (LB_QK_BLOCK=0: old order).
const bool g_qk_block = !lb::env_get("QK_BLOCK") || std::atoi(lb::env_get("QK_BLOCK")) != 0;
template <int TYPE>
QK_AVX2 static void qk_type(const QkView& v, const int8_t* q, size_t qs, const int16_t* b16, size_t bs, int E, float* yE, size_t yst, int g0, int g1) {
    if (E <= 4 || !g_qk_block) { qk_slots<TYPE>(v, q, qs, b16, bs, E, yE, yst, g0, g1); return; }
    for (int g = g0; g < g1; g++) qk_slots<TYPE>(v, q, qs, b16, bs, E, yE, yst, g, g + 1);
}
}  // namespace

// ---- weight-stationary prefill kernel (E >= g_qk_wst_min slots) ------------------------------------------------------------------------------
// Two 4-row groups (8 rows) x one superblock are unpacked ONCE (8x8 dword transposes) into u8 [64 column quads][8 rows x 4 cols] plus per-sub-block int16
// scale vectors, then every slot block runs vpmaddubsw(w, broadcast 4 activation bytes) with the 8 rows in the 8 int32 lanes: int16 adds over one sub-block
// (4 steps of 16 columns; Q4_K: 8 steps of 32), ONE vpmaddwd with the scale vector per sub-block, no horizontal reduction anywhere.
// The integer part is exact (same S, B sums as qk_avx2); only the float summation order differs (per row: fma(float(total int), d) per superblock instead of
// per-lane partials + hsum), so results agree with qk_avx2 to float rounding, NOT bit-for-bit.
namespace {
QK_AVX2 inline void qk_tr8x8d(const uint8_t* const* rp, int off, __m256i* T) {   // T[m] lane r = dword m of the 32 bytes at rp[r] + off
    __m256i x[8];
    for (int r = 0; r < 8; r++) x[r] = _mm256_loadu_si256((const __m256i*)(rp[r] + off));
    const __m256i t0 = _mm256_unpacklo_epi32(x[0], x[1]), t1 = _mm256_unpackhi_epi32(x[0], x[1]);
    const __m256i t2 = _mm256_unpacklo_epi32(x[2], x[3]), t3 = _mm256_unpackhi_epi32(x[2], x[3]);
    const __m256i t4 = _mm256_unpacklo_epi32(x[4], x[5]), t5 = _mm256_unpackhi_epi32(x[4], x[5]);
    const __m256i t6 = _mm256_unpacklo_epi32(x[6], x[7]), t7 = _mm256_unpackhi_epi32(x[6], x[7]);
    const __m256i u0 = _mm256_unpacklo_epi64(t0, t2), u1 = _mm256_unpackhi_epi64(t0, t2), u2 = _mm256_unpacklo_epi64(t1, t3), u3 = _mm256_unpackhi_epi64(t1, t3);
    const __m256i u4 = _mm256_unpacklo_epi64(t4, t6), u5 = _mm256_unpackhi_epi64(t4, t6), u6 = _mm256_unpacklo_epi64(t5, t7), u7 = _mm256_unpackhi_epi64(t5, t7);
    T[0] = _mm256_permute2x128_si256(u0, u4, 0x20); T[1] = _mm256_permute2x128_si256(u1, u5, 0x20);
    T[2] = _mm256_permute2x128_si256(u2, u6, 0x20); T[3] = _mm256_permute2x128_si256(u3, u7, 0x20);
    T[4] = _mm256_permute2x128_si256(u0, u4, 0x31); T[5] = _mm256_permute2x128_si256(u1, u5, 0x31);
    T[6] = _mm256_permute2x128_si256(u2, u6, 0x31); T[7] = _mm256_permute2x128_si256(u3, u7, 0x31);
}
// 8 bytes at rp[r] + off for 8 rows -> out[i]: qword0 = byte 2i of all 8 rows, qword1 = byte 2i + 1 (row r at byte r), i = 0..3
QK_AVX2 inline void qk_tr8b(const uint8_t* const* rp, int off, __m128i* out) {
    __m128i x[8];
    for (int r = 0; r < 8; r++) x[r] = _mm_loadl_epi64((const __m128i*)(rp[r] + off));
    const __m128i a01 = _mm_unpacklo_epi8(x[0], x[1]), a23 = _mm_unpacklo_epi8(x[2], x[3]), a45 = _mm_unpacklo_epi8(x[4], x[5]), a67 = _mm_unpacklo_epi8(x[6], x[7]);
    const __m128i c0 = _mm_unpacklo_epi16(a01, a23), c1 = _mm_unpackhi_epi16(a01, a23), c4 = _mm_unpacklo_epi16(a45, a67), c5 = _mm_unpackhi_epi16(a45, a67);
    out[0] = _mm_unpacklo_epi32(c0, c4); out[1] = _mm_unpackhi_epi32(c0, c4); out[2] = _mm_unpacklo_epi32(c1, c5); out[3] = _mm_unpackhi_epi32(c1, c5);
}
QK_AVX2 inline __m256i qk_dup16lo(__m128i v, bool sgn) { const __m128i d = _mm_unpacklo_epi8(v, v); return sgn ? _mm256_cvtepi8_epi16(d) : _mm256_cvtepu8_epi16(d); }
QK_AVX2 inline __m256i qk_dup16hi(__m128i v, bool sgn) { const __m128i d = _mm_unpackhi_epi8(v, v); return sgn ? _mm256_cvtepi8_epi16(d) : _mm256_cvtepu8_epi16(d); }
QK_AVX2 inline __m256i qk_pair16(__m128i v, bool sgn) { const __m128i d = _mm_unpacklo_epi8(v, _mm_srli_si128(v, 8)); return sgn ? _mm256_cvtepi8_epi16(d) : _mm256_cvtepu8_epi16(d); }
QK_AVX2 inline __m256 qk_f16x8(const uint8_t* const* rp, int off) {
    __m128i h = _mm_setzero_si128();
    h = _mm_insert_epi16(h, ld16(rp[0] + off), 0); h = _mm_insert_epi16(h, ld16(rp[1] + off), 1); h = _mm_insert_epi16(h, ld16(rp[2] + off), 2); h = _mm_insert_epi16(h, ld16(rp[3] + off), 3);
    h = _mm_insert_epi16(h, ld16(rp[4] + off), 4); h = _mm_insert_epi16(h, ld16(rp[5] + off), 5); h = _mm_insert_epi16(h, ld16(rp[6] + off), 6); h = _mm_insert_epi16(h, ld16(rp[7] + off), 7);
    return _mm256_cvtph_ps(h);
}

struct alignas(32) QkWst {
    uint8_t W[64][32];     // column quad cq: 8 rows x 4 columns (u8 quant values)
    int16_t SV[16][16];    // per sub-block: scale of row r in int16 lanes 2r, 2r+1
    int16_t MP[8][16];     // min pairs: lane pair r = (mn[2i], mn[2i+1]) of row r (Q4_K: mn_i twice; Q3_K: 4 * sc)
    float dv[8], dmv[8];
};

template <int TYPE>
QK_AVX2 static void qk_wst_unpack(const uint8_t* const* rp, QkWst& S) {
    const __m256i m3 = _mm256_set1_epi8(3), m4b = _mm256_set1_epi8(4), m15 = _mm256_set1_epi8(15);
    __m256i T[8];
    if constexpr (TYPE == QK_Q2K || TYPE == QK_Q3K) {
        constexpr bool q3 = TYPE == QK_Q3K;
        __m256i Th[8];
        if constexpr (TYPE == QK_Q3K) qk_tr8x8d(rp, 0, Th);
        for (int h = 0; h < 2; h++) {
            qk_tr8x8d(rp, (q3 ? 32 : 16) + 32 * h, T);
            for (int j = 0; j < 4; j++)
                for (int m = 0; m < 8; m++) {
                    __m256i w = _mm256_and_si256(j == 0 ? T[m] : _mm256_srli_epi16(T[m], 2 * j), m3);
                    if constexpr (TYPE == QK_Q3K) {
                        const int b = 4 * h + j;
                        const __m256i hb = b >= 2 ? _mm256_srli_epi16(Th[m], b - 2) : _mm256_slli_epi16(Th[m], 2 - b);
                        w = _mm256_or_si256(w, _mm256_and_si256(hb, m4b));
                    }
                    _mm256_store_si256((__m256i*)S.W[32 * h + 8 * j + m], w);
                }
        }
        __m128i V[8];
        qk_tr8b(rp, q3 ? 96 : 0, V);
        qk_tr8b(rp, q3 ? 104 : 8, V + 4);
        for (int i = 0; i < 8; i++) {
            if constexpr (TYPE == QK_Q2K) {
                const __m128i lo = _mm_and_si128(V[i], _mm256_castsi256_si128(m15)), hi = _mm_and_si128(_mm_srli_epi16(V[i], 4), _mm256_castsi256_si128(m15));
                _mm256_store_si256((__m256i*)S.SV[2 * i], qk_dup16lo(lo, false));
                _mm256_store_si256((__m256i*)S.SV[2 * i + 1], qk_dup16hi(lo, false));
                _mm256_store_si256((__m256i*)S.MP[i], qk_pair16(hi, false));
            } else {
                _mm256_store_si256((__m256i*)S.SV[2 * i], qk_dup16lo(V[i], true));
                _mm256_store_si256((__m256i*)S.SV[2 * i + 1], qk_dup16hi(V[i], true));
                _mm256_store_si256((__m256i*)S.MP[i], _mm256_slli_epi16(qk_pair16(V[i], true), 2));
            }
        }
        if constexpr (TYPE == QK_Q2K) { _mm256_store_ps(S.dv, qk_f16x8(rp, 80)); _mm256_store_ps(S.dmv, qk_f16x8(rp, 82)); }
        else _mm256_store_ps(S.dv, qk_f16x8(rp, 112));
    } else {
        for (int g = 0; g < 4; g++) {
            qk_tr8x8d(rp, 20 + 32 * g, T);
            for (int m = 0; m < 8; m++) {
                _mm256_store_si256((__m256i*)S.W[16 * g + m], _mm256_and_si256(T[m], m15));
                _mm256_store_si256((__m256i*)S.W[16 * g + 8 + m], _mm256_and_si256(_mm256_srli_epi16(T[m], 4), m15));
            }
        }
        __m128i Vs[4], Vm[4];
        qk_tr8b(rp, 4, Vs);
        qk_tr8b(rp, 12, Vm);
        for (int i = 0; i < 4; i++) {
            _mm256_store_si256((__m256i*)S.SV[2 * i], qk_dup16lo(Vs[i], false));
            _mm256_store_si256((__m256i*)S.SV[2 * i + 1], qk_dup16hi(Vs[i], false));
            _mm256_store_si256((__m256i*)S.MP[2 * i], qk_dup16lo(Vm[i], false));
            _mm256_store_si256((__m256i*)S.MP[2 * i + 1], qk_dup16hi(Vm[i], false));
        }
        _mm256_store_ps(S.dv, qk_f16x8(rp, 0));
        _mm256_store_ps(S.dmv, qk_f16x8(rp, 2));
    }
}

template <int TYPE, int TS>
QK_AVX2 static void qk_wst_block(const QkWst& S, const int8_t* q, size_t qs, const int16_t* b16, size_t bs, int t, int k0, float* accf, const float* ss, int nsb) {
    constexpr int NSUB = TYPE == QK_Q4K ? 8 : 16, STEPS = TYPE == QK_Q4K ? 8 : 4;
    const int8_t* ap[TS];
    const int16_t* bp[TS];
    for (int k = 0; k < TS; k++) { ap[k] = q + (size_t)(k0 + k) * qs + (size_t)t * 256; bp[k] = b16 + (size_t)(k0 + k) * bs + (size_t)t * 16; }
    __m256i s32[TS];
    for (int j = 0; j < NSUB; j++) {
        __m256i p[TS];
        {
            const __m256i w = _mm256_load_si256((const __m256i*)S.W[j * STEPS]);
            for (int k = 0; k < TS; k++) p[k] = _mm256_maddubs_epi16(w, _mm256_set1_epi32(*(const int*)(ap[k] + 4 * (j * STEPS))));
        }
        for (int c = 1; c < STEPS; c++) {
            const int cq = j * STEPS + c;
            const __m256i w = _mm256_load_si256((const __m256i*)S.W[cq]);
            for (int k = 0; k < TS; k++) p[k] = _mm256_add_epi16(p[k], _mm256_maddubs_epi16(w, _mm256_set1_epi32(*(const int*)(ap[k] + 4 * cq))));
        }
        const __m256i sv = _mm256_load_si256((const __m256i*)S.SV[j]);
        if (j == 0) { for (int k = 0; k < TS; k++) s32[k] = _mm256_madd_epi16(p[k], sv); }
        else { for (int k = 0; k < TS; k++) s32[k] = _mm256_add_epi32(s32[k], _mm256_madd_epi16(p[k], sv)); }
    }
    const __m256 dv0 = _mm256_load_ps(S.dv), dm0 = _mm256_load_ps(S.dmv);
    for (int k = 0; k < TS; k++) {
        // optional per-slot per-superblock activation scale folded into d / dmin (per-tile activation quantisation)
        const __m256 sk = ss ? _mm256_set1_ps(ss[(size_t)(k0 + k) * nsb + t]) : _mm256_set1_ps(1.f);
        const __m256 dv = ss ? _mm256_mul_ps(dv0, sk) : dv0;
        __m256i m = _mm256_madd_epi16(_mm256_load_si256((const __m256i*)S.MP[0]), _mm256_set1_epi32(*(const int*)bp[k]));
        for (int i = 1; i < 8; i++) m = _mm256_add_epi32(m, _mm256_madd_epi16(_mm256_load_si256((const __m256i*)S.MP[i]), _mm256_set1_epi32(*(const int*)(bp[k] + 2 * i))));
        float* a = accf + (size_t)(k0 + k) * 8;
        if constexpr (TYPE == QK_Q3K) {
            _mm256_store_ps(a, _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(s32[k], m)), dv, _mm256_load_ps(a)));
        } else {
            const __m256 x = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s32[k]), dv, _mm256_load_ps(a));
            _mm256_store_ps(a, _mm256_fnmadd_ps(_mm256_cvtepi32_ps(m), ss ? _mm256_mul_ps(dm0, sk) : dm0, x));
        }
    }
}

const int g_qk_wst_ts = lb::env_get("QK_WSTTS") ? std::max(1, std::min(6, std::atoi(lb::env_get("QK_WSTTS")))) : 6;

template <int TYPE>
QK_AVX2 static void qk_wst_type(const QkView& v, const int8_t* q, size_t qs, const int16_t* b16, size_t bs, int E, float* yE, size_t yst, int g0, int g1, const float* ss) {
    constexpr int SB = kQkSbBytes[TYPE];
    static thread_local std::vector<float> accv;
    static thread_local QkWst scr_store;
    QkWst* const scr = &scr_store;
    accv.resize((size_t)E * 8 + 8);
    float* accf = (float*)(((uintptr_t)accv.data() + 31) & ~(uintptr_t)31);
    const int nblk = (E + g_qk_wst_ts - 1) / g_qk_wst_ts;
    for (int ga = g0; ga < g1; ga += 2) {   // 8 rows = groups ga, ga + 1
        std::memset(accf, 0, (size_t)E * 8 * sizeof(float));
        for (int t = 0; t < v.nsb; t++) {
            const uint8_t* rp[8];
            for (int r = 0; r < 8; r++) rp[r] = v.w + (((size_t)(ga + (r >> 2)) * v.nsb + t) * 4 + (r & 3)) * SB;
            qk_wst_unpack<TYPE>(rp, *scr);
            int k0 = 0;
            for (int b = 0; b < nblk; b++) {
                const int ts = (E - k0 + (nblk - b) - 1) / (nblk - b);
                switch (ts) {
                    case 1: qk_wst_block<TYPE, 1>(*scr, q, qs, b16, bs, t, k0, accf, ss, v.nsb); break;
                    case 2: qk_wst_block<TYPE, 2>(*scr, q, qs, b16, bs, t, k0, accf, ss, v.nsb); break;
                    case 3: qk_wst_block<TYPE, 3>(*scr, q, qs, b16, bs, t, k0, accf, ss, v.nsb); break;
                    case 4: qk_wst_block<TYPE, 4>(*scr, q, qs, b16, bs, t, k0, accf, ss, v.nsb); break;
                    case 5: qk_wst_block<TYPE, 5>(*scr, q, qs, b16, bs, t, k0, accf, ss, v.nsb); break;
                    default: qk_wst_block<TYPE, 6>(*scr, q, qs, b16, bs, t, k0, accf, ss, v.nsb); break;
                }
                k0 += ts;
            }
        }
        for (int e = 0; e < E; e++) std::memcpy(yE + (size_t)e * yst + (size_t)ga * 4, accf + (size_t)e * 8, 8 * sizeof(float));
    }
}
}  // namespace

QK_AVX2 void qk_avx2_wst(const QkView& v, const int8_t* q, size_t qs, const int16_t* b16, size_t bs, int E, float* yE, size_t yst, int g0, int g1, const float* ss) {
    switch (v.type) {
        case QK_Q2K: qk_wst_type<QK_Q2K>(v, q, qs, b16, bs, E, yE, yst, g0, g1, ss); break;
        case QK_Q3K: qk_wst_type<QK_Q3K>(v, q, qs, b16, bs, E, yE, yst, g0, g1, ss); break;
        default: qk_wst_type<QK_Q4K>(v, q, qs, b16, bs, E, yE, yst, g0, g1, ss); break;
    }
}

QK_AVX2 void qk_avx2(const QkView& v, const int8_t* q, size_t qs, const int16_t* b16, size_t bs, int E, float* yE, size_t yst, int g0, int g1, int wst_min, const float* ss) {
    if ((ss || E >= wst_min) && !(g0 & 1) && !(g1 & 1)) { qk_avx2_wst(v, q, qs, b16, bs, E, yE, yst, g0, g1, ss); return; }
    switch (v.type) {
        case QK_Q2K: qk_type<QK_Q2K>(v, q, qs, b16, bs, E, yE, yst, g0, g1); break;
        case QK_Q3K: qk_type<QK_Q3K>(v, q, qs, b16, bs, E, yE, yst, g0, g1); break;
        default: qk_type<QK_Q4K>(v, q, qs, b16, bs, E, yE, yst, g0, g1); break;
    }
}
#endif

void qk_gemm(Isa isa, const QkView& v, const int8_t* q, size_t qs, const int16_t* b16, size_t bs, int E, float* yE, size_t yst, int g0, int g1, int wst_min, const float* ss) {
    if (v.type >= QK_IQ2XS) { iq_gemm(isa, v, q, qs, E, yE, yst, g0, g1, ss); return; }
#if defined(__x86_64__) || defined(_M_X64)
    if (isa == Isa::Avx2 || isa == Isa::Avx512) { qk_avx2(v, q, qs, b16, bs, E, yE, yst, g0, g1, wst_min, ss); return; }
#endif
    qk_scalar(v, q, qs, b16, bs, E, yE, yst, g0, g1);
}

}  // namespace lb
