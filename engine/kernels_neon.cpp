// ARM64 NEON T-MAC LUT kernel (vqtbl1q_s8 = 16 byte-table lookups per instruction,
// no multiply). Build for Android with -march=armv8.2-a+dotprod+fp16.
#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>

#include <cstring>

#include "binary_gemv.h"

namespace lb {

static inline void neon_group(int16x8_t& accL, int16x8_t& accH, int8x16_t tlo, int8x16_t thi, uint8x16_t idx) {
    int8x16_t lo = vqtbl1q_s8(tlo, idx);
    int8x16_t hi = vqtbl1q_s8(thi, idx);
    // zip(lo,hi) rebuilds little-endian int16 values: rows 0..7 / 8..15 in natural order
    accL = vaddq_s16(accL, vreinterpretq_s16_s8(vzip1q_s8(lo, hi)));
    accH = vaddq_s16(accH, vreinterpretq_s16_s8(vzip2q_s8(lo, hi)));
}

void gemv_neon(const BinMatView& w, const int8_t* luts, int T, int32_t* out, int rb_begin, int rb_end) {
    if (rb_end < 0) rb_end = pad_rows(w.rows) / kRowBlock;
    const int rp = pad_rows(w.rows), kp = pad_k(w.cols);
    const int pairs = kp / 8, ntile = kp / kTileK;
    const size_t lutTok = lut_bytes_per_token(kp);
    const uint8x16_t m0f = vdupq_n_u8(0x0f);
    int32_t acc[kMaxBatch][64];

    for (int rb = rb_begin; rb < rb_end; rb++) {
        const uint8_t* wb = w.lut_w + (size_t)rb * pairs * 64;
        std::memset(acc, 0, sizeof(acc));
        for (int tile = 0; tile < ntile; tile++) {
            const uint8_t* wp0 = wb + (size_t)tile * 16 * 64;
            for (int t = 0; t < T; t++) {
                const int8_t* L = luts + t * lutTok + (size_t)tile * kTileGroups * 32;
                int16x8_t aL[4], aH[4];
                for (int v = 0; v < 4; v++) aL[v] = aH[v] = vdupq_n_s16(0);
                const uint8_t* wp = wp0;
                for (int p = 0; p < 16; p++, wp += 64, L += 64) {
                    int8x16_t t0l = vld1q_s8(L), t0h = vld1q_s8(L + 16);
                    int8x16_t t1l = vld1q_s8(L + 32), t1h = vld1q_s8(L + 48);
                    for (int v = 0; v < 4; v++) {
                        uint8x16_t wv = vld1q_u8(wp + 16 * v);
                        neon_group(aL[v], aH[v], t0l, t0h, vandq_u8(wv, m0f));
                        neon_group(aL[v], aH[v], t1l, t1h, vshrq_n_u8(wv, 4));
                    }
                }
                for (int v = 0; v < 4; v++) {
                    int32_t* d = &acc[t][16 * v];
                    vst1q_s32(d, vaddq_s32(vld1q_s32(d), vmovl_s16(vget_low_s16(aL[v]))));
                    vst1q_s32(d + 4, vaddq_s32(vld1q_s32(d + 4), vmovl_s16(vget_high_s16(aL[v]))));
                    vst1q_s32(d + 8, vaddq_s32(vld1q_s32(d + 8), vmovl_s16(vget_low_s16(aH[v]))));
                    vst1q_s32(d + 12, vaddq_s32(vld1q_s32(d + 12), vmovl_s16(vget_high_s16(aH[v]))));
                }
            }
        }
        for (int t = 0; t < T; t++) std::memcpy(out + (size_t)t * rp + rb * kRowBlock, acc[t], 64 * sizeof(int32_t));
    }
}

}  // namespace lb
#endif
