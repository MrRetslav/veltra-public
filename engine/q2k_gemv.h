// `.qk` GEMV/GEMM kernels: llama.cpp K-quant superblocks (Q2_K / Q3_K / Q4_K, 256 weights) transcoded bit-exactly from a GGUF and
// streamed zero-copy from the mmap'd `.lbt` tensor.  Layout contract: docs/qk_format.md.
//
// Weights: [group of 4 rows][superblock t][row r 0..3][SB bytes]   (one contiguous stream per row group), SB per type:
//   Q2_K  84 B : sc[16] (lo nibble scale, hi nibble min, per 16 weights) | qs[64] (2 bit, ggml order) | d f16 | dmin f16
//   Q3_K 114 B : hmask[32] | qs[64] (low 2 bits) | sc[16] int8 (ggml 6 bit scale - 32, already unpacked) | d f16      u = lo2 | hbit << 2, w = d*sc*(u-4)
//   Q4_K 148 B : d f16 | dmin f16 | sc[8] u8 | mn[8] u8 (already unpacked from ggml's 12 B) | qs[128] (lo nibble: weights 0..31 of each 64, hi: 32..63)
// Activations: int8 per slot (one token or one residual plane), plus the per-16-column sums b16 (int16).  For one row and superblock
//   y += d * sum_j sc_j * S_j  -  dmin * sum_j mn_j * B_j      (S_j = sum q*a over the sub-block, B_j = sum a over it; Q3_K: q = u - 4 folded in)
// Result is UNSCALED by the activation scale (the caller multiplies).  Every slot is evaluated independently with the same instruction
// sequence, so results do not depend on how many slots share a pass (T=1 vs batched is bit-identical).
#pragma once
#include <cstddef>
#include <cstdint>

#include "binary_gemv.h"

namespace lb {

// QK_IQ2XS / QK_IQ3XXS / QK_IQ4XS: llama.cpp i-quant blocks kept byte for byte (engine/iq_gemv.h), dispatched by qk_gemm.
enum QkType : int { QK_Q2K = 0, QK_Q3K = 1, QK_Q4K = 2, QK_IQ2XS = 3, QK_IQ3XXS = 4, QK_IQ4XS = 5 };
constexpr int kQkSuper = 256;                                   // weights per superblock
constexpr int kQkSbBytes[6] = {84, 114, 148, 74, 98, 136};                   // bytes per (row, superblock)
constexpr int kQkGroupRows = 4;

struct QkView {
    int type = QK_Q2K, rows = 0, nsb = 0;   // rows % 4 == 0
    const uint8_t* w = nullptr;
};
inline size_t qk_bytes(int type, int rows, int n) { return (size_t)rows * (n / kQkSuper) * kQkSbBytes[type]; }

// b16[j] = sum of q[16 j .. 16 j + 15]  (n % 16 == 0)
void qk_sums16(const int8_t* q, int n, int16_t* b16);

// yE[e][r] for the 4-row groups [g0, g1) (rows 4 g0 .. 4 g1 - 1).  q: slot e at q + e * qstride; b16: slot e at b16 + e * bstride; yE: [E][yst].
// E is arbitrary (processed in passes of up to 4 slots).  Isa::Avx512 uses the AVX2 kernel.
// wst_min: from this many slots on (and even g0, g1) the weight-stationary AVX2 kernel runs (equal to float rounding, not bit-identical to the 4-slot passes); default off.
void qk_gemm(Isa isa, const QkView& v, const int8_t* q, size_t qstride, const int16_t* b16, size_t bstride, int E, float* yE, size_t yst, int g0, int g1, int wst_min = 1 << 30, const float* ss = nullptr);
void qk_scalar(const QkView& v, const int8_t* q, size_t qstride, const int16_t* b16, size_t bstride, int E, float* yE, size_t yst, int g0, int g1);
#if defined(__x86_64__) || defined(_M_X64)
void qk_avx2(const QkView& v, const int8_t* q, size_t qstride, const int16_t* b16, size_t bstride, int E, float* yE, size_t yst, int g0, int g1, int wst_min = 1 << 30, const float* ss = nullptr);
// Weight-stationary prefill kernel (8 rows x 1 superblock unpacked once, lane-wise int accumulation, no horizontal sums): same integers as qk_avx2, float summation
// order differs (equal to float rounding, NOT bit-identical). Needs g0, g1 even; qk_avx2 dispatches to it when E >= wst_min slots.
// ss (optional, [E][nsb] floats): per-slot per-superblock activation scale folded into d / dmin (the result then already includes the activation scale); requires the wst path (g0, g1 even).
void qk_avx2_wst(const QkView& v, const int8_t* q, size_t qstride, const int16_t* b16, size_t bstride, int E, float* yE, size_t yst, int g0, int g1, const float* ss = nullptr);
#endif

// Reference dequantisation of one (row, superblock) to floats (used by tests; same formulas as gguf-py).
void qk_dequant_row_sb(int type, const uint8_t* sb, float* out256);

}  // namespace lb
