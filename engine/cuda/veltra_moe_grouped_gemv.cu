// engine/cuda/veltra_moe_grouped_gemv.cu - kernels declared in veltra_moe_grouped_gemv.cuh.
// Decode (T = 1) is memory-bound: every kernel streams its weights exactly once with coalesced warp reads; the fused grouped
// MoE launch removes ~5-10 us launch overhead per expert linear (> 1000 launches per token otherwise on an RTX 3060).
#include "veltra_moe_grouped_gemv.cuh"

#include <cuda_fp16.h>
#include <cfloat>

namespace veltra::cuda {
namespace {

constexpr int kWarpsPerBlock = 4;

__device__ __forceinline__ float f16(const uint8_t* p) { return __half2float(*reinterpret_cast<const __half*>(p)); }
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return __shfl_sync(0xffffffffu, v, 0);
}

// ---- K-quant superblock (256 weights) . x[0..255], 32 lanes x 8 weights ---------------------------------------------------
// Q2_K / Q3_K, ggml order: value v = n*128 + j*32 + half*16 + l  (n 0..1, j 0..3 = 2-bit shift, half 0..1, l 0..15);
// qs byte n*32 + half*16 + l, scale index n*8 + j*2 + half. Lane -> (n, j, half) = lane >> 1, l = (lane & 1) * 8 .. +7.
__device__ __forceinline__ float q2k_dot(const uint8_t* sb, const float* x, int lane) {
    const int combo = lane >> 1, l0 = (lane & 1) * 8;
    const int n = combo >> 3, j = (combo >> 1) & 3, hf = combo & 1;
    const uint8_t s = sb[n * 8 + j * 2 + hf];
    const float dl = f16(sb + 80) * (float)(s & 0xF), ml = f16(sb + 82) * (float)(s >> 4);
    const uint8_t* q = sb + 16 + n * 32 + hf * 16 + l0;
    const float* xv = x + n * 128 + j * 32 + hf * 16 + l0;
    float acc = 0.f, xs = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) { acc += (float)((q[k] >> (2 * j)) & 3) * xv[k]; xs += xv[k]; }
    return dl * acc - ml * xs;
}
// Q3_K (114 B): hmask[32] | qs[64] | sc[16] int8 (6-bit scale - 32, unpacked) | d f16 @112.  w = d * sc * (lo2 + 4*hbit - 4),
// hbit = hmask[half*16 + l] bit (n*4 + j).
__device__ __forceinline__ float q3k_dot(const uint8_t* sb, const float* x, int lane) {
    const int combo = lane >> 1, l0 = (lane & 1) * 8;
    const int n = combo >> 3, j = (combo >> 1) & 3, hf = combo & 1;
    const float dl = f16(sb + 112) * (float)(int8_t)sb[96 + n * 8 + j * 2 + hf];
    const uint8_t* q = sb + 32 + n * 32 + hf * 16 + l0;
    const uint8_t* hm = sb + hf * 16 + l0;
    const int hbit = n * 4 + j;
    const float* xv = x + n * 128 + j * 32 + hf * 16 + l0;
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) {
        const int u = ((q[k] >> (2 * j)) & 3) | (((hm[k] >> hbit) & 1) << 2);
        acc += (float)(u - 4) * xv[k];
    }
    return dl * acc;
}
// Q4_K (148 B): d @0 | dmin @2 | sc[8] @4 | mn[8] @12 | qs[128] @20. 64-value group g: weights 0..31 = lo nibbles of
// qs[32g .. 32g+31] with (sc, mn)[2g], weights 32..63 = hi nibbles with (sc, mn)[2g+1]. Lane -> values 8*lane .. 8*lane+7.
__device__ __forceinline__ float q4k_dot(const uint8_t* sb, const float* x, int lane) {
    const int v0 = lane * 8, g = v0 >> 6, within = v0 & 63, hi = within >= 32, l = within & 31;
    const int s = 2 * g + hi;
    const float dl = f16(sb) * (float)sb[4 + s], ml = f16(sb + 2) * (float)sb[12 + s];
    const uint8_t* q = sb + 20 + 32 * g + l;
    const float* xv = x + v0;
    float acc = 0.f, xs = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) { acc += (float)(hi ? (q[k] >> 4) : (q[k] & 0xF)) * xv[k]; xs += xv[k]; }
    return dl * acc - ml * xs;
}

__global__ void grouped_gemv_kernel(const ExpertJob* __restrict__ jobs) {
    const ExpertJob jb = jobs[blockIdx.y];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= jb.rows) return;
    const int nsb = jb.cols / 256, SB = kSbBytes[jb.qtype];
    const uint8_t* g = jb.w + (size_t)(row >> 2) * nsb * 4 * SB + (size_t)(row & 3) * SB;   // [group][sb][row][SB]
    float acc = 0.f;
    for (int t = 0; t < nsb; t++) {
        const uint8_t* sb = g + (size_t)t * 4 * SB;
        const float* xt = jb.x + t * 256;
        acc += jb.qtype == kQ2K ? q2k_dot(sb, xt, lane) : jb.qtype == kQ3K ? q3k_dot(sb, xt, lane) : q4k_dot(sb, xt, lane);
    }
    acc = warp_sum(acc);
    if (lane == 0) jb.y[row] = acc;
}

// ---- GQA decode attention: one block (4 warps) per query head; warps take keys round-robin, online softmax per warp,
// lanes hold D/32 output dims; the 4 partial (m, l, acc) are merged through shared memory.
constexpr int kMaxD = 256;
__global__ void gqa_decode_kernel(const float* __restrict__ q, const __half* __restrict__ K, const __half* __restrict__ V, float* __restrict__ out,
                                  int H, int KVH, int D, int cap, int len, float scale) {
    const int h = blockIdx.x, kvh = h / (H / KVH);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, per = D / 32;   // D % 32 == 0
    float qv[kMaxD / 32], acc[kMaxD / 32];
    for (int i = 0; i < per; i++) { qv[i] = q[h * D + lane * per + i]; acc[i] = 0.f; }
    float m = -FLT_MAX, l = 0.f;
    const __half* Kh = K + (size_t)kvh * cap * D;
    const __half* Vh = V + (size_t)kvh * cap * D;
    for (int j = warp; j < len; j += kWarpsPerBlock) {
        const __half* kj = Kh + (size_t)j * D + lane * per;
        float s = 0.f;
        for (int i = 0; i < per; i++) s += qv[i] * __half2float(kj[i]);
        s = warp_sum(s) * scale;
        const float mn = fmaxf(m, s), c = __expf(m - mn), p = __expf(s - mn);
        const __half* vj = Vh + (size_t)j * D + lane * per;
        for (int i = 0; i < per; i++) acc[i] = acc[i] * c + p * __half2float(vj[i]);
        l = l * c + p;
        m = mn;
    }
    __shared__ float sm[kWarpsPerBlock], sl[kWarpsPerBlock], sacc[kWarpsPerBlock][kMaxD];
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
    for (int i = 0; i < per; i++) sacc[warp][lane * per + i] = acc[i];
    __syncthreads();
    if (warp == 0) {
        float M = -FLT_MAX;
        for (int w = 0; w < kWarpsPerBlock; w++) M = fmaxf(M, sm[w]);
        float L = 0.f;
        for (int w = 0; w < kWarpsPerBlock; w++) L += sl[w] * __expf(sm[w] - M);
        for (int i = 0; i < per; i++) {
            float o = 0.f;
            for (int w = 0; w < kWarpsPerBlock; w++) o += sacc[w][lane * per + i] * __expf(sm[w] - M);
            out[h * D + lane * per + i] = L > 0.f ? o / L : 0.f;
        }
    }
}

// ---- heads: one warp per vocabulary row
__global__ void head_i8_kernel(const int8_t* __restrict__ W, const float* __restrict__ rs, const float* __restrict__ x, float* __restrict__ y,
                               const int* __restrict__ rows, int V, int Hd) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int i = blockIdx.x * kWarpsPerBlock + warp;
    if (i >= V) return;
    const int r = rows ? rows[i] : i;
    const int8_t* w = W + (size_t)r * Hd;
    float acc = 0.f;
    for (int h = lane * 4; h < Hd; h += 128) {
        const char4 c = *reinterpret_cast<const char4*>(w + h);
        acc += c.x * x[h] + c.y * x[h + 1] + c.z * x[h + 2] + c.w * x[h + 3];
    }
    acc = warp_sum(acc);
    if (lane == 0) y[i] = acc * rs[r];
}
__global__ void head_screen4_kernel(const uint8_t* __restrict__ W4, const float* __restrict__ rs, const float* __restrict__ x, float* __restrict__ y, int V, int Hd) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * kWarpsPerBlock + warp;
    if (r >= V) return;
    const uint8_t* w = W4 + (size_t)r * (Hd / 2);
    float acc = 0.f;
    for (int b = lane * 4; b < Hd / 2; b += 128) {
        const uchar4 c = *reinterpret_cast<const uchar4*>(w + b);
        const uint8_t v[4] = {c.x, c.y, c.z, c.w};
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int h = 2 * (b + k);
            acc += (float)((int)(v[k] & 0xF) - 8) * x[h] + (float)((int)(v[k] >> 4) - 8) * x[h + 1];
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) y[r] = acc * rs[r];
}

}  // namespace

cudaError_t moe_grouped_gemv(const ExpertJob* d_jobs, int n_jobs, int max_rows, cudaStream_t s) {
    if (n_jobs <= 0 || max_rows <= 0) return cudaSuccess;
    const dim3 grid((unsigned)((max_rows + kWarpsPerBlock - 1) / kWarpsPerBlock), (unsigned)n_jobs);
    grouped_gemv_kernel<<<grid, kWarpsPerBlock * 32, 0, s>>>(d_jobs);
    return cudaGetLastError();
}

cudaError_t gqa_attention_decode(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, float* out, int H, int KVH, int D, int cap,
                                 int len, float scale, cudaStream_t s) {
    if (D % 32 != 0 || D > kMaxD || KVH <= 0 || H % KVH != 0) return cudaErrorInvalidValue;
    gqa_decode_kernel<<<H, kWarpsPerBlock * 32, 0, s>>>(q, reinterpret_cast<const __half*>(k_cache), reinterpret_cast<const __half*>(v_cache), out, H,
                                                        KVH, D, cap, len, scale);
    return cudaGetLastError();
}

cudaError_t head_i8_gemv(const int8_t* W, const float* row_scale, const float* x, float* y, int V, int Hd, cudaStream_t s) {
    if (Hd % 4 != 0) return cudaErrorInvalidValue;
    head_i8_kernel<<<(V + kWarpsPerBlock - 1) / kWarpsPerBlock, kWarpsPerBlock * 32, 0, s>>>(W, row_scale, x, y, nullptr, V, Hd);
    return cudaGetLastError();
}
cudaError_t head_i8_rows(const int8_t* W, const float* row_scale, const float* x, const int* rows, int n, float* y, int Hd, cudaStream_t s) {
    if (Hd % 4 != 0) return cudaErrorInvalidValue;
    if (n <= 0) return cudaSuccess;
    head_i8_kernel<<<(n + kWarpsPerBlock - 1) / kWarpsPerBlock, kWarpsPerBlock * 32, 0, s>>>(W, row_scale, x, y, rows, n, Hd);
    return cudaGetLastError();
}
cudaError_t head_screen4_gemv(const uint8_t* W4, const float* row_scale4, const float* x, float* y4, int V, int Hd, cudaStream_t s) {
    if (Hd % 8 != 0) return cudaErrorInvalidValue;
    head_screen4_kernel<<<(V + kWarpsPerBlock - 1) / kWarpsPerBlock, kWarpsPerBlock * 32, 0, s>>>(W4, row_scale4, x, y4, V, Hd);
    return cudaGetLastError();
}

}  // namespace veltra::cuda
