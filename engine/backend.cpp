// engine/backend.cpp - CpuAvx2Backend (thin, bit-identical adapter over the existing kernels) + CUDA factory hook.
#include "backend.h"

namespace lb {

const char* CpuAvx2Backend::name() const {
    switch (isa_) {
        case Isa::Avx512: return "cpu-avx512";
        case Isa::Avx2: return "cpu-avx2";
        case Isa::Neon: return "cpu-neon";
        default: return "cpu-scalar";
    }
}
std::string CpuAvx2Backend::device() const {
    switch (isa_) {
        case Isa::Avx512: return "CPU AVX-512";
        case Isa::Avx2: return "CPU AVX2";
        case Isa::Neon: return "CPU NEON";
        default: return "CPU";
    }
}
void CpuAvx2Backend::linear(const LbLinear& w, const float* x, int T, float* y, LinearScratch& s) {
    linear_forward(w, isa_, x, T, y, /*use_res=*/true, s, pool_);
}
void CpuAvx2Backend::linear_multi(const LbLinear* const* ws, int nw, const float* x, int T, float* const* ys, LinearScratch& s) {
    linear_forward_multi(ws, nw, isa_, x, T, ys, /*use_res=*/true, s, pool_);
}
void CpuAvx2Backend::moe_experts(const LinJob* jobs, int nj, LinearScratch& s) {
    linear_forward_jobs(jobs, nj, isa_, s, pool_);
}

std::unique_ptr<ComputeBackend> make_cpu_backend(Isa isa, ThreadPool* pool) {
    return std::make_unique<CpuAvx2Backend>(isa, pool);
}

#if defined(VELTRA_CUDA)
std::unique_ptr<ComputeBackend> make_cuda_backend_impl(ComputeBackend* cpu_fallback, std::string* err);   // engine/cuda/cuda_backend.cu
std::unique_ptr<ComputeBackend> make_cuda_backend(ComputeBackend* cpu_fallback, std::string* err) {
    return make_cuda_backend_impl(cpu_fallback, err);
}
bool cuda_backend_built() { return true; }
#else
std::unique_ptr<ComputeBackend> make_cuda_backend(ComputeBackend*, std::string* err) {
    if (err) *err = "this build has no CUDA backend (configure with -DVELTRA_CUDA=ON)";
    return nullptr;
}
bool cuda_backend_built() { return false; }
std::string cuda_selftest(const std::string&) { return "cuda selftest: FAIL - this build has no CUDA backend (configure with -DVELTRA_CUDA=ON)\n"; }
#endif

}  // namespace lb
