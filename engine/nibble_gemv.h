// Nibble-INT4 (.n4) GEMV/GEMM kernels, contract: docs/bitplane_spec.md section 5.
//
// Weights stay in the mmap'd tensor (no copy): [row_block rb][tile t][row r%64][64 B], byte i = j[r,128t+i] | j[r,128t+64+i] << 4,
// level j in 0..15 means value j - 7.5.  Activations are per-slot int8 (slot = one token or one residual plane of a token).
//   yE[e][r] = sum_t gs[r,t] * (S[e,r,t] - 7.5 * A[e,t]),  S = sum_{c in tile} j[r,c] * q_e[c],  A = sum_{c in tile} q_e[c]
// (unscaled by the activation scale; the caller multiplies by it). Evaluated exactly like the bit-plane kernels: per tile the exact int32
// Z = 2S - 15A, then acc = fma(float(Z), 0.5*gs, acc) in tile order, so results are bit-identical to the plane format of the same weights.
// AVX2: vpmaddubsw (u8 weights x s8 activations, int16 partials of <= 8 products <= 15240) + vpmaddwd; AVX-512: vpdpbusd (ymm, or zmm for
// <= 2 slots). 8 rows' lane vectors are reduced together (transposed add tree), so there is no per-row horizontal sum and no accumulator
// reload. The weight tile is unpacked ONCE per row and reused for up to 8 slots per pass.
#pragma once
#include <cstddef>
#include <cstdint>

#include "binary_gemv.h"

namespace lb {

struct NibView {
    int rows = 0, ntile = 0;
    const uint8_t* w = nullptr;   // n4 tensor data, 64-byte aligned
    const float* gs = nullptr;    // f32 HALF group scales 0.5*gs, layout [row_block][tile][64 rows] (rows >= `rows` are 0)
};

constexpr size_t kNibTileBytes = 64 * 64;   // one 64-row x 128-col tile

// ss (optional): per-slot per-tile activation scales [E][ntile]; when given yE already includes them (gs[r,t] * ss[e][t] replaces gs[r,t]), else yE is unscaled.
// q: slot e at q + e * q_stride (>= ntile*128 int8), A: [E][ntile] int32 tile sums of q, yE: [E][yE_stride] floats, rows of row blocks [rb_begin, rb_end) written
// (all 64 rows of every block, padding rows give 0). E is arbitrary (processed in groups).
void nibble_gemm(Isa isa, const NibView& v, const int8_t* q, size_t q_stride, const int32_t* A, int E, float* yE, size_t yE_stride, int rb_begin, int rb_end, const float* ss = nullptr);
void nibble_scalar(const NibView& v, const int8_t* q, size_t q_stride, const int32_t* A, int E, float* yE, size_t yE_stride, int rb_begin, int rb_end, const float* ss = nullptr);
#if defined(__x86_64__) || defined(_M_X64)
void nibble_avx2(const NibView& v, const int8_t* q, size_t q_stride, const int32_t* A, int E, float* yE, size_t yE_stride, int rb_begin, int rb_end, const float* ss = nullptr);
void nibble_avx512(const NibView& v, const int8_t* q, size_t q_stride, const int32_t* A, int E, float* yE, size_t yE_stride, int rb_begin, int rb_end, const float* ss = nullptr);
#endif

}  // namespace lb
