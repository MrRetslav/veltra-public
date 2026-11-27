// engine/cuda/cuda_backend.cu - CudaBackend (engine/backend.h), built only with -DVELTRA_CUDA=ON.
// STATUS: written, never compiled or run (no GPU on the development machines); docs/GPU_RTX3060_GUIDE.md has the first-run checklist (--cuda-selftest).
//
//   * VRAM weight uploader: copies the K-quant expert streams byte-for-byte from the mmap'd file into a VRAM arena (no repacking,
//     no intermediate host buffer), lowest layers first. What does not fit in the budget (free VRAM - 1.2 GiB reserve) stays on the
//     CPU: a 12 GB RTX 3060 holds most of the 30B's ~9.4 GB of experts, an 8 GB card the first ~60 %.
//   * Transformer::run_moe_jobs() hands every MoE projection to moe_experts(): resident experts of the layer run in ONE grouped
//     launch (veltra_moe_grouped_gemv.cu), the others on the CPU kernels; only x and y cross PCIe (a few KB per layer).
//   * Attention / head kernels (GQA decode, int8 + 4-bit screened head) exist and are selftested, but linear() still uses the CPU.
#include "backend.h"
#include "veltra_moe_grouped_gemv.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace lb {
namespace {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
constexpr size_t kArenaChunk = (size_t)256 << 20;
constexpr size_t kReserve = (size_t)(1.2 * kGiB);   // KV cache, activations, cuBLAS workspace

#define VCUDA_OK(x) ((x) == cudaSuccess)

class CudaBackend final : public ComputeBackend {
public:
    CudaBackend(ComputeBackend* cpu, int dev) : cpu_(cpu), dev_(dev) {
        cudaSetDevice(dev_);
        cudaDeviceProp p{};
        cudaGetDeviceProperties(&p, dev_);
        dev_name_ = p.name;
        sm_ = p.major * 10 + p.minor;
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
        cublasCreate(&blas_);
        cublasSetStream(blas_, stream_);
    }
    ~CudaBackend() override {
        cudaStreamSynchronize(stream_);
        for (void* p : arena_) cudaFree(p);
        if (d_jobs_) cudaFree(d_jobs_);
        if (d_io_) cudaFree(d_io_);
        cublasDestroy(blas_);
        cudaStreamDestroy(stream_);
    }
    const char* name() const override { return sm_ == 86 ? "cuda-sm86" : "cuda"; }
    std::string device() const override { return "CUDA GPU (" + dev_name_ + ")"; }
    bool is_gpu() const override { return true; }

    uint64_t upload_weights(const LbtFile& f, uint64_t budget) override {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const uint64_t cap = free_b > kReserve ? free_b - kReserve : 0;
        if (budget == 0 || budget > cap) budget = cap;
        // Only what a GPU kernel reads today: the K-quant expert streams (.q2k / .q3k / .q4k), lowest layers first.
        // Attention / head stay on the CPU until their kernels are wired into linear() (no VRAM spent on them).
        std::vector<std::pair<std::string, const TensorRef*>> order;
        auto is_qk = [](const std::string& n) {
            return n.size() > 4 && (n.compare(n.size() - 4, 4, ".q2k") == 0 || n.compare(n.size() - 4, 4, ".q3k") == 0 || n.compare(n.size() - 4, 4, ".q4k") == 0);
        };
        for (const auto& kv : f.tensors())
            if (kv.first.rfind("blk.", 0) == 0 && is_qk(kv.first)) order.emplace_back(kv.first, &kv.second);
        auto rank = [](const std::string& n) -> long { return std::atol(n.c_str() + 4); };   // layer index
        std::stable_sort(order.begin(), order.end(), [&](const auto& a, const auto& b) { return rank(a.first) < rank(b.first); });
        uint64_t used = 0;
        for (const auto& [name, t] : order) {
            if (used + t->nbytes > budget) continue;   // stays on the CPU (fallback)
            void* d = alloc(t->nbytes);
            if (!d) break;
            for (uint64_t off = 0; off < t->nbytes; off += (uint64_t)64 << 20) {
                const size_t n = (size_t)std::min<uint64_t>(t->nbytes - off, (uint64_t)64 << 20);
                cudaMemcpyAsync((uint8_t*)d + off, t->data + off, n, cudaMemcpyHostToDevice, stream_);
            }
            resident_[t->data] = (uint8_t*)d;
            used += t->nbytes;
        }
        cudaStreamSynchronize(stream_);
        resident_bytes_ = used;
        std::fprintf(stderr, "[veltra-cuda] %s: %.2f GiB of weights resident in VRAM (budget %.2f GiB), rest on CPU\n",
                     dev_name_.c_str(), used / kGiB, budget / kGiB);
        return used;
    }

    void linear(const LbLinear& w, const float* x, int T, float* y, LinearScratch& s) override {
        // TODO: dense attention / head on the GPU (dequantise i3 / n4 to f16 on upload, then cublasGemmEx).
        cpu_->linear(w, x, T, y, s);
    }
    void linear_multi(const LbLinear* const* ws, int nw, const float* x, int T, float* const* ys, LinearScratch& s) override {
        cpu_->linear_multi(ws, nw, x, T, ys, s);
    }

    void moe_experts(const LinJob* jobs, int nj, LinearScratch& s) override {
        std::vector<LinJob> cpu_jobs;
        std::vector<const LinJob*> gpu_jobs;
        for (int i = 0; i < nj; i++) (gpu_ok(jobs[i]) ? (void)gpu_jobs.push_back(&jobs[i]) : cpu_jobs.push_back(jobs[i]));
        if (!gpu_jobs.empty() && !run_gpu(gpu_jobs)) {
            for (const LinJob* j : gpu_jobs) cpu_jobs.push_back(*j);   // launch failed: whole layer on the CPU
        }
        if (!cpu_jobs.empty()) cpu_->moe_experts(cpu_jobs.data(), (int)cpu_jobs.size(), s);
    }

    // CPU kernel (int8-quantised activations) vs GPU (f32 activations) on one resident K-quant linear:
    // max |diff| / max |y_cpu|, or a negative value on failure. Expected ~1e-2 (activation quantisation of the CPU path).
    double check_qk(const LbLinear& w, LinearScratch& s) {
        std::vector<float> x(w.n), yc(w.m), yg(w.m);
        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (float& v : x) v = nd(rng);
        float* yp = yc.data();
        LinJob j{};
        j.w[0] = &w; j.nw = 1; j.x = x.data(); j.T = 1; j.y[0] = yp;
        cpu_->moe_experts(&j, 1, s);
        j.y[0] = yg.data();
        std::vector<const LinJob*> one{&j};
        if (!gpu_ok(j) || !run_gpu(one)) return -1.0;
        double mx = 0, md = 0;
        for (int r = 0; r < w.m; r++) { mx = std::max(mx, (double)std::fabs(yc[r])); md = std::max(md, (double)std::fabs(yc[r] - yg[r])); }
        return mx > 0 ? md / mx : md;
    }

private:
    bool gpu_ok(const LinJob& j) const {
        for (int k = 0; k < j.nw; k++) {
            const LbLinear* w = j.w[k];
            if (!w->is_qk() || w->qk.type > veltra::cuda::kQ4K || !resident_.count(w->qk.w) || w->n % 256 != 0) return false;
        }
        return true;
    }
    void* alloc(size_t n) {
        n = (n + 255) & ~(size_t)255;
        if (n > kArenaChunk) {
            void* p = nullptr;
            if (!VCUDA_OK(cudaMalloc(&p, n))) return nullptr;
            arena_.push_back(p);
            return p;
        }
        if (arena_left_ < n) {
            void* p = nullptr;
            if (!VCUDA_OK(cudaMalloc(&p, kArenaChunk))) return nullptr;
            arena_.push_back(p);
            arena_cur_ = (uint8_t*)p;
            arena_left_ = kArenaChunk;
        }
        void* r = arena_cur_;
        arena_cur_ += n;
        arena_left_ -= n;
        return r;
    }
    bool ensure_io(size_t floats, size_t njobs) {
        if (floats > io_cap_) {
            if (d_io_) cudaFree(d_io_);
            d_io_ = nullptr;
            if (!VCUDA_OK(cudaMalloc(&d_io_, floats * sizeof(float)))) { io_cap_ = 0; return false; }
            io_cap_ = floats;
        }
        if (njobs > jobs_cap_) {
            if (d_jobs_) cudaFree(d_jobs_);
            d_jobs_ = nullptr;
            if (!VCUDA_OK(cudaMalloc(&d_jobs_, njobs * sizeof(veltra::cuda::ExpertJob)))) { jobs_cap_ = 0; return false; }
            jobs_cap_ = njobs;
        }
        return true;
    }
    // x of every (job, token) up, one launch for all (linear, token) GEMVs, y back down.
    bool run_gpu(const std::vector<const LinJob*>& js) {
        size_t floats = 0, n = 0;
        int max_rows = 0;
        for (const LinJob* j : js) {
            floats += (size_t)j->T * j->w[0]->n;
            for (int k = 0; k < j->nw; k++) { floats += (size_t)j->T * j->w[k]->m; n += j->T; max_rows = std::max(max_rows, j->w[k]->m); }
        }
        if (!ensure_io(floats, n)) return false;
        std::vector<veltra::cuda::ExpertJob> hj;
        hj.reserve(n);
        std::vector<std::pair<float*, const float*>> outs;   // (host y, device y)
        float* cur = d_io_;
        for (const LinJob* j : js) {
            const int cols = j->w[0]->n;
            cudaMemcpyAsync(cur, j->x, (size_t)j->T * cols * sizeof(float), cudaMemcpyHostToDevice, stream_);
            float* dx = cur;
            cur += (size_t)j->T * cols;
            for (int k = 0; k < j->nw; k++) {
                const LbLinear* w = j->w[k];
                for (int t = 0; t < j->T; t++) {
                    hj.push_back({resident_.at(w->qk.w), dx + (size_t)t * cols, cur + (size_t)t * w->m, w->m, cols, w->qk.type});
                }
                outs.emplace_back(j->y[k], cur);
                cur += (size_t)j->T * w->m;
                (void)outs;
            }
        }
        cudaMemcpyAsync(d_jobs_, hj.data(), hj.size() * sizeof(hj[0]), cudaMemcpyHostToDevice, stream_);
        if (veltra::cuda::moe_grouped_gemv(d_jobs_, (int)hj.size(), max_rows, stream_) != cudaSuccess) return false;
        size_t oi = 0;
        for (const LinJob* j : js)
            for (int k = 0; k < j->nw; k++, oi++)
                cudaMemcpyAsync(j->y[k], outs[oi].second, (size_t)j->T * j->w[k]->m * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        return VCUDA_OK(cudaStreamSynchronize(stream_));
    }

    ComputeBackend* cpu_;
    int dev_ = 0, sm_ = 0;
    std::string dev_name_;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t blas_ = nullptr;
    std::vector<void*> arena_;
    uint8_t* arena_cur_ = nullptr;
    size_t arena_left_ = 0;
    std::unordered_map<const uint8_t*, const uint8_t*> resident_;   // host weight pointer (mmap) -> device copy
    uint64_t resident_bytes_ = 0;
    float* d_io_ = nullptr;
    size_t io_cap_ = 0;
    veltra::cuda::ExpertJob* d_jobs_ = nullptr;
    size_t jobs_cap_ = 0;
};

}  // namespace

std::unique_ptr<ComputeBackend> make_cuda_backend_impl(ComputeBackend* cpu_fallback, std::string* err) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        if (err) *err = "no CUDA device (driver missing or no NVIDIA GPU)";
        return nullptr;
    }
    if (!cpu_fallback) {
        if (err) *err = "the CUDA backend needs a CPU fallback backend";
        return nullptr;
    }
    return std::make_unique<CudaBackend>(cpu_fallback, 0);
}

// ---------------------------------------------------------------------------------------------------------------- --cuda-selftest
// Every CUDA kernel against a CPU reference on synthetic data (fp32 references, tight tolerance), then - when a model is given -
// one real expert linear per K-quant type against the CPU kernel (loose tolerance: the CPU path quantises activations to int8).
namespace {
struct Report {
    std::string text;
    int fails = 0;
    void add(const char* what, double err, double tol) {
        const bool ok = err >= 0 && err <= tol;
        if (!ok) fails++;
        char b[200];
        std::snprintf(b, sizeof b, "  %-40s %s  (max rel err %.2e, tol %.0e)\n", what, ok ? "PASS" : "FAIL", err, tol);
        text += b;
    }
};
template <class T> T* dev_copy(const std::vector<T>& h) {
    T* d = nullptr;
    if (cudaMalloc(&d, h.size() * sizeof(T)) != cudaSuccess) return nullptr;
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
double rel_err(const std::vector<float>& ref, const std::vector<float>& got) {
    double mx = 0, md = 0;
    for (size_t i = 0; i < ref.size(); i++) { mx = std::max(mx, (double)std::fabs(ref[i])); md = std::max(md, (double)std::fabs(ref[i] - got[i])); }
    return mx > 0 ? md / mx : md;
}
uint16_t h16(float f) { const __half h = __float2half(f); uint16_t u; std::memcpy(&u, &h, 2); return u; }
float f32h(uint16_t u) { __half h; std::memcpy(&h, &u, 2); return __half2float(h); }

// rows x cols K-quant stream with random content and sane d / dmin: GPU grouped GEMV vs qk_dequant_row_sb . x (fp32 reference)
double test_qk(int type, std::mt19937& rng) {
    const int rows = 8, cols = 512, nsb = cols / 256, SB = veltra::cuda::kSbBytes[type];
    std::vector<uint8_t> w((size_t)rows * nsb * SB);
    std::uniform_int_distribution<int> ub(0, 255);
    for (auto& b : w) b = (uint8_t)ub(rng);
    auto sbp = [&](int r, int t) { return w.data() + (((size_t)(r >> 2) * nsb + t) * 4 + (r & 3)) * SB; };
    for (int r = 0; r < rows; r++)
        for (int t = 0; t < nsb; t++) {
            uint8_t* p = sbp(r, t);
            auto put = [&](int off, float v) { const uint16_t u = h16(v); std::memcpy(p + off, &u, 2); };
            if (type == veltra::cuda::kQ2K) { put(80, 0.012f); put(82, 0.004f); }
            else if (type == veltra::cuda::kQ3K) { for (int i = 0; i < 16; i++) p[96 + i] = (uint8_t)(int8_t)((int)(p[96 + i] % 64) - 32); put(112, 0.009f); }
            else { put(0, 0.011f); put(2, 0.006f); for (int i = 0; i < 16; i++) p[4 + i] &= 63; }
        }
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(cols), ref(rows, 0.f), got(rows), deq(256);
    for (float& v : x) v = nd(rng);
    for (int r = 0; r < rows; r++)
        for (int t = 0; t < nsb; t++) {
            lb::qk_dequant_row_sb(type, sbp(r, t), deq.data());
            for (int i = 0; i < 256; i++) ref[r] += deq[i] * x[(size_t)t * 256 + i];
        }
    uint8_t* dw = dev_copy(w);
    float* dx = dev_copy(x);
    float* dy = nullptr;
    cudaMalloc(&dy, rows * sizeof(float));
    std::vector<veltra::cuda::ExpertJob> hj{{dw, dx, dy, rows, cols, type}};
    veltra::cuda::ExpertJob* dj = dev_copy(hj);
    double e = -1;
    if (dw && dx && dy && dj && veltra::cuda::moe_grouped_gemv(dj, 1, rows, nullptr) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess) {
        cudaMemcpy(got.data(), dy, rows * sizeof(float), cudaMemcpyDeviceToHost);
        e = rel_err(ref, got);
    }
    cudaFree(dw); cudaFree(dx); cudaFree(dy); cudaFree(dj);
    return e;
}

double test_attention(std::mt19937& rng, int H, int KVH, int D, int len) {
    const int cap = len + 3;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> q((size_t)H * D), ref((size_t)H * D, 0.f), got((size_t)H * D);
    std::vector<uint16_t> K((size_t)KVH * cap * D), V((size_t)KVH * cap * D);
    for (float& v : q) v = nd(rng);
    for (auto& v : K) v = h16(nd(rng));
    for (auto& v : V) v = h16(nd(rng));
    const float scale = 1.f / std::sqrt((float)D);
    for (int h = 0; h < H; h++) {
        const int kv = h / (H / KVH);
        std::vector<double> s(len);
        double mx = -1e30;
        for (int j = 0; j < len; j++) {
            double a = 0;
            for (int d = 0; d < D; d++) a += q[(size_t)h * D + d] * f32h(K[((size_t)kv * cap + j) * D + d]);
            s[j] = a * scale;
            mx = std::max(mx, s[j]);
        }
        double l = 0;
        for (double& v : s) { v = std::exp(v - mx); l += v; }
        for (int j = 0; j < len; j++)
            for (int d = 0; d < D; d++) ref[(size_t)h * D + d] += (float)(s[j] / l) * f32h(V[((size_t)kv * cap + j) * D + d]);
    }
    float* dq = dev_copy(q);
    uint16_t* dk = dev_copy(K);
    uint16_t* dv = dev_copy(V);
    float* dout = nullptr;
    cudaMalloc(&dout, got.size() * sizeof(float));
    double e = -1;
    if (dq && dk && dv && dout && veltra::cuda::gqa_attention_decode(dq, dk, dv, dout, H, KVH, D, cap, len, scale, nullptr) == cudaSuccess &&
        cudaDeviceSynchronize() == cudaSuccess) {
        cudaMemcpy(got.data(), dout, got.size() * sizeof(float), cudaMemcpyDeviceToHost);
        e = rel_err(ref, got);
    }
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout);
    return e;
}

void test_heads(std::mt19937& rng, Report& rep) {
    const int V = 1000, Hd = 512;
    std::uniform_int_distribution<int> i8d(-127, 127), u8d(0, 255);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<int8_t> W((size_t)V * Hd);
    std::vector<uint8_t> W4((size_t)V * Hd / 2);
    std::vector<float> rs(V), rs4(V), x(Hd), ref(V), ref4(V), got(V), got4(V);
    for (auto& v : W) v = (int8_t)i8d(rng);
    for (auto& v : W4) v = (uint8_t)u8d(rng);
    for (int v = 0; v < V; v++) { rs[v] = 0.001f * (1 + v % 7); rs4[v] = 0.01f * (1 + v % 5); }
    for (float& v : x) v = nd(rng);
    for (int v = 0; v < V; v++) {
        double a = 0, b = 0;
        for (int h = 0; h < Hd; h++) {
            a += W[(size_t)v * Hd + h] * x[h];
            const uint8_t byte = W4[(size_t)v * (Hd / 2) + h / 2];
            b += ((h & 1 ? byte >> 4 : byte & 0xF) - 8) * x[h];
        }
        ref[v] = (float)(a * rs[v]);
        ref4[v] = (float)(b * rs4[v]);
    }
    std::vector<int> rows = {999, 0, 512, 7};
    std::vector<float> refr, gotr(rows.size());
    for (int r : rows) refr.push_back(ref[r]);
    int8_t* dW = dev_copy(W);
    uint8_t* dW4 = dev_copy(W4);
    float *drs = dev_copy(rs), *drs4 = dev_copy(rs4), *dx = dev_copy(x), *dy = nullptr, *dy4 = nullptr, *dyr = nullptr;
    int* drows = dev_copy(rows);
    cudaMalloc(&dy, V * sizeof(float)); cudaMalloc(&dy4, V * sizeof(float)); cudaMalloc(&dyr, rows.size() * sizeof(float));
    double e1 = -1, e2 = -1, e3 = -1;
    if (veltra::cuda::head_i8_gemv(dW, drs, dx, dy, V, Hd, nullptr) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess) {
        cudaMemcpy(got.data(), dy, V * sizeof(float), cudaMemcpyDeviceToHost); e1 = rel_err(ref, got);
    }
    if (veltra::cuda::head_screen4_gemv(dW4, drs4, dx, dy4, V, Hd, nullptr) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess) {
        cudaMemcpy(got4.data(), dy4, V * sizeof(float), cudaMemcpyDeviceToHost); e2 = rel_err(ref4, got4);
    }
    if (veltra::cuda::head_i8_rows(dW, drs, dx, drows, (int)rows.size(), dyr, Hd, nullptr) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess) {
        cudaMemcpy(gotr.data(), dyr, rows.size() * sizeof(float), cudaMemcpyDeviceToHost); e3 = rel_err(refr, gotr);
    }
    cudaFree(dW); cudaFree(dW4); cudaFree(drs); cudaFree(drs4); cudaFree(dx); cudaFree(dy); cudaFree(dy4); cudaFree(dyr); cudaFree(drows);
    rep.add("head int8 GEMV (V=1000, H=512)", e1, 1e-5);
    rep.add("head 4-bit screen GEMV", e2, 1e-5);
    rep.add("head int8 candidate rescoring", e3, 1e-5);
}
}  // namespace

std::string cuda_selftest(const std::string& model_path) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return "cuda selftest: FAIL - no CUDA device\n";
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, 0);
    Report rep;
    rep.text = "cuda selftest on " + std::string(p.name) + " (sm_" + std::to_string(p.major * 10 + p.minor) + ")\n";
    std::mt19937 rng(1234);
    rep.add("grouped MoE GEMV Q2_K", test_qk(veltra::cuda::kQ2K, rng), 1e-4);
    rep.add("grouped MoE GEMV Q3_K", test_qk(veltra::cuda::kQ3K, rng), 1e-4);
    rep.add("grouped MoE GEMV Q4_K", test_qk(veltra::cuda::kQ4K, rng), 1e-4);
    rep.add("GQA decode attention D=128 (32/4 heads)", test_attention(rng, 32, 4, 128, 517), 1e-4);
    rep.add("GQA decode attention D=256 (16/2 heads)", test_attention(rng, 16, 2, 256, 70), 1e-4);
    test_heads(rng, rep);
    if (!model_path.empty()) {
        LbtFile f;
        std::string err;
        if (!f.open(model_path, &err)) {
            rep.add(("model: " + err).c_str(), -1, 0);
        } else {
            auto cpu = make_cpu_backend(best_isa(), nullptr);
            auto gpu = make_cuda_backend_impl(cpu.get(), &err);
            auto* cb = static_cast<CudaBackend*>(gpu.get());
            if (cb) cb->upload_weights(f, 0);
            for (const char* suf : {".q2k", ".q3k", ".q4k"}) {
                std::string name;   // lowest layer first: the uploader makes early layers resident first
                long best = 1L << 30;
                for (const auto& kv : f.tensors())
                    if (kv.first.size() > 4 && kv.first.compare(kv.first.size() - 4, 4, suf) == 0 && kv.first.rfind("blk.", 0) == 0) {
                        const long layer = std::atol(kv.first.c_str() + 4);
                        if (layer < best) { best = layer; name = kv.first.substr(0, kv.first.size() - 4); }
                    }
                if (name.empty()) continue;
                LbLinear w;
                LinearScratch s;
                const double e = (cb && load_linear(f, name, &w, &err)) ? cb->check_qk(w, s) : -1.0;
                rep.add(("model " + name + suf + " vs CPU kernel").c_str(), e, 3e-2);
            }
        }
    }
    rep.text += rep.fails ? "cuda selftest: " + std::to_string(rep.fails) + " FAILED\n" : std::string("cuda selftest: all PASS\n");
    return rep.text;
}

}  // namespace lb
