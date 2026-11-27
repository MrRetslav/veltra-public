// engine/backend.h - ComputeBackend: the seam between the transformer and the weight kernels.
//
//   CpuAvx2Backend  the existing CPU kernels (Q2_K / Q3_K / Q4_K, INT3, IQ2_XS / IQ3_XXS / IQ4_XS, nibble INT4, grid),
//                   runtime-dispatched AVX2 / AVX-512 / NEON. Bit-identical to calling linear_forward* directly.
//   CudaBackend     engine/cuda/ (only with -DVELTRA_CUDA=ON, default arch sm_86 = RTX 3060): VRAM weight uploader with a
//                   per-layer CPU fallback, cuBLAS for attention / head, one grouped-GEMV launch for all active experts.
//                   STATUS: kernels written (grouped MoE GEMV for Q2_K/Q3_K/Q4_K, GQA decode attention, int8 + 4-bit screened
//                   head) with a CPU-reference --cuda-selftest, but never compiled or run (no GPU on the development machines). moe_experts() runs
//                   resident K-quant experts on the GPU; attention/head are not yet wired into linear() (CPU fallback).
//
// The decode path (transformer.cpp) still calls linear_forward* directly; routing it through ComputeBackend is a
// per-call virtual hop and is switched over together with the first CUDA kernel that passes the eval gates.
#pragma once
#include <cstdint>
#include <memory>
#include <string>

#include "lbt_model.h"

namespace lb {

class ThreadPool;

class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;
    virtual const char* name() const = 0;       // "cpu-avx2", "cpu-avx512", "cpu-neon", "cuda-sm86", ...
    virtual std::string device() const = 0;     // human label for the UI: "CPU AVX2", "CUDA GPU (NVIDIA GeForce RTX 3060)"
    virtual bool is_gpu() const { return false; }

    // y[T][m] = W x for one linear (attention q/k/v/o, LM head, dense FFN).
    virtual void linear(const LbLinear& w, const float* x, int T, float* y, LinearScratch& s) = 0;
    // Several linears reading the same x (q/k/v, gate/up).
    virtual void linear_multi(const LbLinear* const* ws, int nw, const float* x, int T, float* const* ys, LinearScratch& s) = 0;
    // MoE: every active expert of one layer (gate+up and down jobs) in one call - one parallel region on the CPU,
    // one kernel launch on CUDA (engine/cuda/veltra_moe_grouped_gemv.cu).
    virtual void moe_experts(const LinJob* jobs, int nj, LinearScratch& s) = 0;

    // Makes weights resident on the device within budget_bytes (CUDA: VRAM upload, layers that do not fit stay on the
    // CPU). Returns the bytes now resident on the device; the CPU backend reads the mmap'd file in place and returns 0.
    virtual uint64_t upload_weights(const LbtFile& f, uint64_t budget_bytes) { (void)f; (void)budget_bytes; return 0; }
};

class CpuAvx2Backend final : public ComputeBackend {
public:
    CpuAvx2Backend(Isa isa, ThreadPool* pool) : isa_(isa), pool_(pool) {}
    const char* name() const override;
    std::string device() const override;
    void linear(const LbLinear& w, const float* x, int T, float* y, LinearScratch& s) override;
    void linear_multi(const LbLinear* const* ws, int nw, const float* x, int T, float* const* ys, LinearScratch& s) override;
    void moe_experts(const LinJob* jobs, int nj, LinearScratch& s) override;
    Isa isa() const { return isa_; }
private:
    Isa isa_;
    ThreadPool* pool_;
};

std::unique_ptr<ComputeBackend> make_cpu_backend(Isa isa, ThreadPool* pool);
// nullptr when the build has no CUDA backend (VELTRA_CUDA off) or no usable device; *err says why.
// cpu_fallback receives every linear whose weights are not resident on the GPU.
std::unique_ptr<ComputeBackend> make_cuda_backend(ComputeBackend* cpu_fallback, std::string* err);
bool cuda_backend_built();
// --cuda-selftest: every CUDA kernel (grouped MoE GEMV Q2_K/Q3_K/Q4_K, GQA decode attention, int8 / 4-bit heads) against a CPU
// reference on synthetic data, plus one real expert linear per K-quant type of model_path (optional). Multi-line PASS/FAIL report.
std::string cuda_selftest(const std::string& model_path);

}  // namespace lb
