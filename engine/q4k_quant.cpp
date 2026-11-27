// Q4_K superblock quantiser used by the on-device converter (engine/hf_convert.cpp). An exporter can compile this file into a small shared
// library, so that both write bit-identical weights.
#include "q4k_quant.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "lb_common.h"

namespace veltra {
namespace {

// llama.cpp-style Q4_K. A sub-block of 32 weights is fitted as x ~ scale * q - min (q 0..15, min >= 0): several start scales,
// then a weighted least-squares refit; the fit with the smallest weighted error wins.
void fit_sub_block(const float* x, float* out_scale, float* out_min) {
    float lo = 0.f, hi = 0.f, absum = 0.f;
    for (int i = 0; i < 32; i++) { lo = std::min(lo, x[i]); hi = std::max(hi, x[i]); absum += std::fabs(x[i]); }
    *out_scale = 0.f;
    *out_min = 0.f;
    if (hi - lo <= 0.f) return;
    float wt[32];
    for (int i = 0; i < 32; i++) wt[i] = std::fabs(x[i]) + absum / 32.f;
    float best = -1.f;
    auto consider = [&](float scale, float mn) {
        const float inv = 1.0f / scale;
        float err = 0.f;
        for (int i = 0; i < 32; i++) {
            const float q = std::min(15.0f, std::max(0.0f, std::nearbyintf((x[i] + mn) * inv)));
            const float d = scale * q - mn - x[i];
            err += wt[i] * d * d;
        }
        if (best < 0.f || err < best) { best = err; *out_scale = scale; *out_min = mn; }
    };
    for (int k = 0; k <= 20; k++) {
        const float scale = (hi - lo) / (14.0f + 0.1f * (float)k);
        consider(scale, -lo);
        const float inv = 1.0f / scale;
        float sw = 0.f, sq = 0.f, sqq = 0.f, sx = 0.f, sqx = 0.f;
        for (int i = 0; i < 32; i++) {
            const float q = std::min(15.0f, std::max(0.0f, std::nearbyintf((x[i] - lo) * inv)));
            sw += wt[i]; sq += wt[i] * q; sqq += wt[i] * q * q; sx += wt[i] * x[i]; sqx += wt[i] * q * x[i];
        }
        const float det = sw * sqq - sq * sq;
        if (det <= 0.f) continue;
        float rs = (sw * sqx - sq * sx) / det, rb = (sqq * sx - sq * sqx) / det;   // x ~ rs * q + rb
        if (rb > 0.f) { rb = 0.f; rs = sqq > 0.f ? sqx / sqq : 0.f; }
        if (rs > 0.f) consider(rs, -rb);
    }
}

// 256 weights -> one 148-byte block of the engine's layout: d f16 | dmin f16 | sc[8] | mn[8] | qs[128] (engine/q2k_gemv.h).
}  // namespace

void quant_q4k(const float* x, uint8_t* out) {
    float scale[8], mn[8], max_scale = 0.f, max_min = 0.f;
    for (int j = 0; j < 8; j++) {
        fit_sub_block(x + 32 * j, &scale[j], &mn[j]);
        max_scale = std::max(max_scale, scale[j]);
        max_min = std::max(max_min, mn[j]);
    }
    const uint16_t dh = lb::f32_to_f16(max_scale / 63.0f), mh = lb::f32_to_f16(max_min / 63.0f);
    const float d = lb::f16_to_f32(dh), dmin = lb::f16_to_f32(mh);
    std::memcpy(out, &dh, 2);
    std::memcpy(out + 2, &mh, 2);
    uint8_t q[256];
    for (int j = 0; j < 8; j++) {
        const int sc = d > 0.f ? (int)std::min(63.0f, std::max(0.0f, std::nearbyintf(scale[j] / d))) : 0;
        const int m = dmin > 0.f ? (int)std::min(63.0f, std::max(0.0f, std::nearbyintf(mn[j] / dmin))) : 0;
        out[4 + j] = (uint8_t)sc;
        out[12 + j] = (uint8_t)m;
        const float dl = d * (float)sc, ml = dmin * (float)m;   // quantise against what the kernel will really use
        for (int i = 0; i < 32; i++)
            q[32 * j + i] = dl > 0.f ? (uint8_t)std::min(15.0f, std::max(0.0f, std::nearbyintf((x[32 * j + i] + ml) / dl))) : 0;
    }
    for (int g = 0; g < 4; g++)
        for (int l = 0; l < 32; l++) out[20 + 32 * g + l] = (uint8_t)(q[64 * g + l] | (q[64 * g + 32 + l] << 4));
}

}  // namespace veltra

// nsb consecutive superblocks of 256 weights -> nsb * 148 bytes.
extern "C" void veltra_q4k_quantize(const float* x, long long nsb, unsigned char* out) {
    for (long long i = 0; i < nsb; i++) veltra::quant_q4k(x + i * 256, out + i * 148);
}
