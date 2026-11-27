// i-quant GEMV kernels (llama.cpp IQ2_XS / IQ3_XXS / IQ4_XS super-blocks of 256 weights, kept byte for byte, docs/qk_format.md v2).
// Weight stream per row group, as for the K-quants: [group of 4 rows][superblock][row 0..3][ggml block bytes]  (74 / 98 / 136 B).
//   IQ2_XS : d f16 | qs[32] u16 (9-bit index into iq2xs_grid, 7-bit sign index) | scales[8] (two 4-bit scales per byte, 16 weights each)    y = 0.125 d sum (2s+1) grid*sign*a
//   IQ3_XXS: d f16 | qs[64] u8 grid index (iq3xxs_grid, 4 weights each) | 8 x u32 (4 x 7-bit sign index, 4-bit scale at bit 28, per 32 weights)  y = 0.25 d sum (2s+1) ...
//   IQ4_XS : d f16 | scales_h u16 | scales_l[4] | qs[128] (nibble indices into kvalues_iq4nl, low nibbles = weights 0..15 of a 32-block)         y = d sum (ls-32) ...
// Same contract as qk_gemm (engine/q2k_gemv.h): int8 activation slots, result UNSCALED by the slot's activation scale; `ss` (optional, [E][nsb]) is a per-slot per-
// superblock scale folded into d (tile activation quantisation). Every slot runs the same instruction sequence (T=1 == batched, bit for bit).
#pragma once
#include <string>

#include "q2k_gemv.h"

namespace lb {

void iq_gemm(Isa isa, const QkView& v, const int8_t* q, size_t qstride, int E, float* yE, size_t yst, int g0, int g1, const float* ss);
// Reference dequantisation of one (row, superblock) (port of ggml dequantize_row_iq*; used by the selftest).
void iq_dequant_row_sb(int type, const uint8_t* sb, float* out256);
int run_iq_selftest(std::string& report);

}  // namespace lb
