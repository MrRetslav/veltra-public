// Small dense ops used by the transformer: RMSNorm, RoPE, SiLU, softmax, f16/int8 matvec for the LM head.
// Each op processes one token with a fixed summation order, so batched (verify) and single-token
// (plain decode) evaluation are bit-identical -> speculative decoding is token-exact under greedy.
#pragma once
#include <cstddef>
#include <cstdint>

namespace lb {

void rmsnorm(const float* x, const float* w, int n, float eps, float* out);
// rotate-half RoPE on one head vector (HF Llama/Qwen convention). cos/sin: [head_dim/2] for this position.
void rope_inplace(float* v, int head_dim, const float* cos, const float* sin);
void silu_mul(const float* gate, const float* up, int n, float* out);
void softmax_inplace(float* x, int n);

// dot(x[n] float, w[n] f16 bits) -> float with a deterministic order (8 lanes + tree reduce when SIMD).
float dot_f16(const float* x, const uint16_t* w, int n);
// dot(xq[n] int8, wq[n] int8) -> int32
int32_t dot_i8(const int8_t* x, const int8_t* w, int n);
// int8 dot against a float vector with per-row scale applied by the caller: sum x[i]*q[i]
float dot_f32_i8(const float* x, const int8_t* q, int n);
// same with one scale per `block` contiguous int8 values: sum_b sc[b] * sum_{i in block b} x[i]*q[i]
float dot_f32_i8_blocks(const float* x, const int8_t* q, const float* sc, int block, int n);
// y[i] += w * sc[i / block] * q[i]
void axpy_i8_blocks(float* y, float w, const int8_t* q, const float* sc, int block, int n);

// Exact-head screening kernel: out[r*np + p] = sum_i xu[p][i] * w[r][i] (xu = uint8 activation planes, w = int8 rows,
// row stride H). Integer arithmetic, exact; AVX-512 VNNI when available, AVX2 / scalar otherwise.
void head_dots_i8(const int8_t* w, int rows, int H, const uint8_t* xu, int np, int32_t* out);

// Signed-int8 dot products for the INT8 output head: out[r*np + p] = sum_i xs[p][i] * w[r][i] (xs: np activation planes of H int8 in [-127,127],
// w: int8 rows, row stride H), exact integer arithmetic (AVX2 |w| x sign(x,w) maddubs: 32 MACs per ~3 uops). np <= 8. Returns false when no SIMD
// implementation exists on this CPU (callers then keep their scalar path).
bool head_dots_s8(const int8_t* w, int rows, int H, const int8_t* xs, int np, int32_t* out);

// Elementwise helpers (separate mul/add, no FMA: bit-identical to the scalar loops, SIMD when available).
void vec_mul(const float* a, const float* b, int n, float* out);              // out = a * b
void vec_mul_inplace(float* a, const float* b, int n);                          // a *= b
void vec_mad(float* y, const float* u, const float* h, int n);                  // y += u * h
void vec_acc_i32(float* o, const int32_t* a, float sc, int n);                  // o += (float)a * sc

// Grouped-query attention over one chunk of n cached positions for G (1,2,4,8) query heads sharing one KV head. head_dim must be 128, int8 KV with
// 8 scales per row (one per 16 dims). q: [G][128] (already scaled by 1/sqrt(d)); k8/v8: [n][128]; ks/vs: [n][8]. Outputs per head: m = max score,
// l = sum exp(s-m), acc[h][128] = sum exp(s-m) * v. scratch: >= n*8 + 32*128 floats. Returns false (nothing written) when no SIMD kernel exists.
bool attn_gqa_chunk(int G, const float* q, const int8_t* k8, const float* ks, const int8_t* v8, const float* vs, int n, float* m, float* l, float* acc, float* scratch);

bool cpu_has_f16c_fma();

}  // namespace lb
