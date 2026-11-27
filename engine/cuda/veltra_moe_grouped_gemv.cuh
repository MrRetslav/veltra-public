// engine/cuda/veltra_moe_grouped_gemv.cuh - CUDA kernels of the Veltra GPU backend (-DVELTRA_CUDA=ON, default sm_86 = RTX 3060).
// STATUS: written against the CPU reference formulas; not yet compiled or run on a GPU (no GPU on the development machines). `--cuda-selftest`
// (engine/cuda/cuda_backend.cu) checks every kernel against a CPU reference on the target machine.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace veltra::cuda {

// Same ids as lb::QkType (engine/q2k_gemv.h).
enum QType : int { kQ2K = 0, kQ3K = 1, kQ4K = 2 };
constexpr int kSbBytes[3] = {84, 114, 148};   // bytes per (row, 256-weight superblock), docs/qk_format.md

// One GEMV y[rows] = W[rows x cols] . x[cols] of one expert linear for one token.
// W is the .q2k / .q3k / .q4k stream exactly as in the .vlt file (groups of 4 rows: [group][superblock][row 0..3][SB bytes]),
// copied unchanged into VRAM by the weight uploader.
struct ExpertJob {
    const uint8_t* w;   // device
    const float* x;     // device, [cols]
    float* y;           // device, [rows]
    int rows, cols, qtype;
};

// Fused grouped MoE GEMV: ALL jobs of one layer (8 experts x {gate, up} or 8 x {down}, times T tokens) in ONE launch.
// grid = (ceil(max_rows / 4), n_jobs), 128 threads = 4 warps, one warp per output row, lanes split each superblock 8 weights each.
// 48 launches per token (one per layer and projection pair) instead of 384+.
cudaError_t moe_grouped_gemv(const ExpertJob* d_jobs, int n_jobs, int max_rows, cudaStream_t s);

// GQA attention for decode (T = 1): q [H][D] f32 (RoPE applied), K/V cache [KVH][cap][D] f16 (keys after RoPE), first `len`
// positions valid; out [H][D] f32. Head h reads kv head h / (H / KVH). Chunked online softmax, D <= 256.
cudaError_t gqa_attention_decode(const float* q, const uint16_t* k_cache, const uint16_t* v_cache, float* out, int H, int KVH, int D,
                                 int cap, int len, float scale, cudaStream_t s);

// INT8 LM head: y[v] = row_scale[v] * sum_h W[v][h] * x[h]   (W int8 [V][Hd] row-major, row_scale f32 [V]).
cudaError_t head_i8_gemv(const int8_t* W, const float* row_scale, const float* x, float* y, int V, int Hd, cudaStream_t s);

// 4-bit screened head: y4[v] = row_scale4[v] * sum_h (W4[v][h] - 8) * x[h]   (W4: two nibbles per byte, low nibble = even h).
// The host keeps every row within the 6-sigma margin of the best screened score and rescores those with head_i8_rows.
cudaError_t head_screen4_gemv(const uint8_t* W4, const float* row_scale4, const float* x, float* y4, int V, int Hd, cudaStream_t s);
// Exact INT8 re-scoring of a candidate list rows[0..n) -> y[i].
cudaError_t head_i8_rows(const int8_t* W, const float* row_scale, const float* x, const int* rows, int n, float* y, int Hd, cudaStream_t s);

}  // namespace veltra::cuda
