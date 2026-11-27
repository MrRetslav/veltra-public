// INT3 g128 asymmetric (`.i3`) GEMV/GEMM kernels for the attention linears (docs/i3_format.md).
//
// Tensor `<name>.i3` (U8, shape [m, n/256 * 104], m % 64 == 0, n % 256 == 0, 64-byte aligned) is read in place (no copy, no lazy expansion).
// Stream order: [row block rb (64 rows)][tile pair p (256 columns)] chunks of kI3Chunk = 6656 bytes:
//   header 512 B: f16 [tile 2][field 2: 0 = scale, 1 = min][64 rows]            W_hat = q * scale + min,  q in 0..7, group = one 128-column tile
//   data  6144 B: [row 64][96 B]: bytes 0..31  low 2 bits of tile 0, bytes 32..63 low 2 bits of tile 1 (byte i: weights i, i+32, i+64, i+96 of the tile in bit fields 0-1, 2-3, 4-5, 6-7),
//                                bytes 64..95  high bit plane: bit v of byte i = bit 2 of weight i + 32*(v % 4) of tile v / 4.
// Activations are per-slot int8 exactly as for the nibble path (one slot = one token or one residual plane). Per tile and row:
//   S = sum q*a (exact int32), A = sum a (exact, tile sum of the int8 activations)   ->   contribution fma(float(S), scale, float(A) * min)
// acc += contribution (times the optional per-slot per-tile activation scale ss), in tile order. The scalar reference and the AVX2 kernel evaluate the same operations in the
// same order, so they are bit-identical; the result of a slot does not depend on the number of slots processed in the same pass.
#pragma once
#include <cstddef>
#include <cstdint>

#include "binary_gemv.h"

namespace lb {

constexpr int kI3Hdr = 512, kI3Data = 6144, kI3Chunk = kI3Hdr + kI3Data;   // per (64 rows x 256 columns)

struct I3View {
    int rows = 0, ntile = 0;      // ntile = n / 128 (even)
    const uint8_t* w = nullptr;
};

inline size_t i3_bytes(int rows, int n) { return (size_t)rows / 64 * (size_t)(n / 256) * kI3Chunk; }

// Same calling convention as nibble_gemm (q: slot e at q + e * q_stride, A: [E][ntile] int32 tile sums, yE: [E][yE_stride], rows of row blocks [rb0, rb1) written).
void i3_gemm(Isa isa, const I3View& v, const int8_t* q, size_t q_stride, const int32_t* A, int E, float* yE, size_t yE_stride, int rb0, int rb1, const float* ss = nullptr);
void i3_scalar(const I3View& v, const int8_t* q, size_t q_stride, const int32_t* A, int E, float* yE, size_t yE_stride, int rb0, int rb1, const float* ss = nullptr);
#if defined(__x86_64__) || defined(_M_X64)
void i3_avx2(const I3View& v, const int8_t* q, size_t q_stride, const int32_t* A, int E, float* yE, size_t yE_stride, int rb0, int rb1, const float* ss = nullptr);
#endif

// Dequantise one row (all n columns) of a `.i3` tensor: out[n] floats = q * scale + min (reference / tests / exporter checks).
void i3_dequant_row(const I3View& v, int row, float* out);
// Pack q (uint8 0..7, [rows][n]) + scale/min (f32, already f16-representable, [rows][ntile]) into the stream layout (used by tests and the bench; the exporter has a numpy twin).
void i3_pack(int rows, int n, const uint8_t* q, const float* scale, const float* mn, uint8_t* out);

}  // namespace lb
