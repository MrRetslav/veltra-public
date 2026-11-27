// .lbt container: zero-copy mmap loader + LittleBitLinear layer view.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "binary_gemv.h"
#include "engine.h"
#include "i3_gemv.h"
#include "q2k_gemv.h"
#include "lb_common.h"
#include "platform.h"

namespace lb {
class ThreadPool;

enum DType : uint32_t { DT_F32 = 0, DT_F16 = 1, DT_BF16 = 2, DT_U8 = 3, DT_U64 = 4, DT_LUT8 = 5, DT_I8 = 6, DT_I32 = 7 };

#pragma pack(push, 1)
struct LbtHeader {
    char magic[8];
    uint32_t version, n_tensors;
    uint64_t meta_off, meta_len, table_off, data_off, draft_layer_mask;
    uint32_t flags, reserved;
};
struct LbtEntry {
    char name[96];
    uint32_t dtype, ndim;
    uint64_t shape[4];
    uint64_t offset, nbytes;
    uint8_t pad[8];
};
#pragma pack(pop)
static_assert(sizeof(LbtHeader) == 64, "header must be 64 bytes");
static_assert(sizeof(LbtEntry) == 160, "table entry must be 160 bytes");

struct TensorRef {
    const uint8_t* data = nullptr;
    uint32_t dtype = 0, ndim = 0;
    uint64_t shape[4] = {0, 0, 0, 0};
    uint64_t nbytes = 0;
    uint64_t numel() const { uint64_t n = 1; for (uint32_t i = 0; i < ndim; i++) n *= shape[i]; return ndim ? n : 0; }
};

// Read-only mapping of a file (mmap / MapViewOfFile). Pages are only touched when used.
class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile();
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    bool open(const std::string& path, std::string* err);
    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }
    bool resident() const { return heap_; }  // true: whole file lives in ordinary heap memory (not a file mapping)
private:
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
    veltra::plat::FileMap fmap_;  // zero-copy mapping (engine/platform.h: mmap / MapViewOfFile)
    bool heap_ = false;
    void* anon_base_ = nullptr;   // LB_ANON_HUGE: anonymous THP copy of the file (POSIX only)
    size_t anon_len_ = 0;
};

class LbtFile {
public:
    bool open(const std::string& path, std::string* err);
    const LbtHeader& header() const { return *hdr_; }
    const std::string& meta_json() const { return meta_json_; }
    bool resident() const { return map_.resident(); }
    bool has(const std::string& name) const { return tensors_.count(name) != 0; }
    const TensorRef* find(const std::string& name) const;
    const TensorRef& get(const std::string& name) const;     // throws if missing
    const std::unordered_map<std::string, TensorRef>& tensors() const { return tensors_; }
    // Dequantise a small f32/f16/bf16 vector tensor into floats.
    std::vector<float> vec_f32(const std::string& name) const;
private:
    MappedFile map_;
    const LbtHeader* hdr_ = nullptr;
    std::string meta_json_;
    std::unordered_map<std::string, TensorRef> tensors_;
};

// One factorised path: diag(h) U diag(l) Vt diag(g).
struct LbPath {
    int m = 0, n = 0, r = 0;
    BinMatView vt;               // [r x n]  (V_b^T)
    BinMatView u;                // [m x r]
    std::vector<float> h, l, g;  // fp32 copies (small)
    AlignedVec<uint8_t> w512_u, w512_vt;  // AVX-512 repacked weights (built on demand by ensure_isa)
    bool valid() const { return r > 0; }
};

// Bit-plane grid linear (docs/bitplane_spec.md): P in {2,3} tile-major +-1 planes, group scales gs (G = 128 per tile, or G = 0 per row).
struct GridLinear {
    int P = 0, G = 0, ntile = 0;
    const uint8_t* plane[4] = {nullptr, nullptr, nullptr, nullptr};   // tile-major LUT8 (points into the file)
    AlignedVec<uint8_t> w512[4];                              // AVX-512 repack of every tile (built on demand)
    AlignedVec<float> gs_row;                                 // [ntile][rp] (G = 128) or [rp] of ones (G = 0), x 0.5, row order
    AlignedVec<float> gs_avx2;                                // same in gemv_avx2 storage order (built on demand)
    AlignedVec<float> rscale;                                 // G = 0: 0.5 * gs[r] [rp]; empty for G = 128
    bool valid() const { return P > 0; }
};

// Nibble-INT4 linear (docs/bitplane_spec.md section 5): the mmap'd `.n4` tensor is read directly by the kernels (no weight copy);
// only the f32 (half) group scales are expanded ([row_block][tile][64 rows], 4 B per 128 weights).
struct NibLinear {
    int ntile = 0;
    const uint8_t* w = nullptr;
    AlignedVec<float> gs;             // expanded lazily on first use (ensure_nib): loading a 30B MoE must not touch every expert's scales
    const uint8_t* gs_src = nullptr;  // the file's `.gs` tensor ([rows][ntile], f16 / f32 / bf16)
    uint32_t gs_dtype = 0;
    int rows = 0;
    bool valid() const { return w != nullptr; }
};

// K-quant linear (`<name>.q2k` / `.q3k` / `.q4k`, transcoded from a GGUF, engine/q2k_gemv.h): the mmap'd tensor is streamed directly, nothing is copied or expanded.
struct QkLinear {
    int type = -1;                 // QkType, -1 = not a qk linear
    int nsb = 0;                   // superblocks per row
    const uint8_t* w = nullptr;
    bool valid() const { return w != nullptr && type >= 0; }
};

// INT3 g128 asymmetric linear (`<name>.i3`, engine/i3_gemv.h, docs/i3_format.md): streamed in place, scales/mins live inside the stream.
struct I3Linear {
    int ntile = 0;
    const uint8_t* w = nullptr;
    bool valid() const { return w != nullptr; }
};

struct LbLinear {
    int m = 0, n = 0;
    LbPath prim, res;
    GridLinear grid;
    NibLinear nib;
    QkLinear qk;
    I3Linear i3;
    bool is_grid() const { return grid.valid(); }
    bool is_nibble() const { return nib.valid(); }
    bool is_qk() const { return qk.valid(); }
    bool is_i3() const { return i3.valid(); }
    bool has_res() const { return res.valid(); }
    // Optional bias (Qwen2.5 qkv)
    std::vector<float> bias;
};

// One group of linears applied to T consecutive input rows (an MoE expert: gate+up share the input, down has its own).
struct LinJob {
    const LbLinear* w[2] = {nullptr, nullptr};
    int nw = 1;
    const float* x = nullptr;   // [T][n]
    int T = 0;
    float* y[2] = {nullptr, nullptr};   // [T][m_i]
};
// Tokens per weight pass of the nibble / K-quant paths (default 32, env LB_SUBTOK, <= kMaxSub): bigger chunks stream every expert's / linear's
// weights fewer times during prefill (results do not depend on it: every slot is evaluated independently).
constexpr int kMaxSub = 64;
int sub_tokens();
// Internal per-chunk state of linear_forward_jobs (<= kMaxSub tokens of one job).
struct NibSub {
    const LbLinear* w[2];
    int nw, T, n, ntile, E;
    const float* x;
    float* y[2];
    int slot1[kMaxSub], slot2[kMaxSub];
    float sscale[2 * kMaxSub];
    size_t qoff, aoff, ssoff, yoff[2];
};
struct NibTask { int sub, li, b0, b1; };

// Per-call scratch (re-used across calls to avoid allocation).
struct LinearScratch {
    bool tile_act = false;       // prefill: per-tile (128 / 256 column) activation scales instead of per-token scale + second plane (dense nibble + K-quant jobs)
    bool fast_prefill = false;   // set by the transformer for forwards of >= 16 tokens: K-quant linears may use the weight-stationary kernel (float-order differs from T=1)
    std::vector<NibSub> nsub;
    std::vector<float> ss;       // qk tile mode: per slot per superblock activation scales
    std::vector<uint8_t> tpf;    // jobs paths: per routed token, needs a second activation plane
    std::vector<NibTask> ntask;
    std::vector<float> xs;       // scaled activations
    std::vector<int8_t> q;       // quantised activations [T][k_pad]
    AlignedVec<int8_t> luts;    // [T][lut_bytes]
    std::vector<int32_t> acc;    // [T][rows_pad]
    std::vector<int16_t> b16;    // qk linears: per-16-column activation sums [E][n/16]
    std::vector<float> z;        // [T][r]
    std::vector<float> y;        // path output
    AlignedVec<float> yE;        // grid linears: [E][rows_pad] per-activation-plane results
};

// Two-plane int8 activation quantisation threshold (crest factor amax/rms); 0 disables.
void set_two_plane_crest(float c);
// Nibble linears only: per-128-column int8 activation scales instead of per-token + two-plane (env LB_NIB_ACT=tile). Default off.
void set_nibble_act_tile(bool on);
bool get_nibble_act_tile();
float get_two_plane_crest();

// ISA for a model: grid (INT2/3/4 bit-plane) models run faster on AVX2 than on the AVX-512 repack path on the machines measured
// (i7-1185G7: gemv 4.9 s vs 6.4 s per 48 tokens; the AVX-512 path also keeps a second ~1 GB copy of the weights in RAM).
// LB_ISA=avx512 (or --isa avx512 in lb_gen) forces AVX-512.
inline Isa default_isa_for(const LbtFile& f) {
    Isa isa = best_isa();
    // Nibble-INT4 models (blk.0.q.n4) keep no second weight copy on AVX-512 and measured faster there (laptop: decode +10 %, prefill +17 %).
    if (isa == Isa::Avx512 && !lb::env_get("ISA") && f.has("blk.0.q.gp0_lut") && isa_available(Isa::Avx2)) return Isa::Avx2;
    return isa;
}

bool load_linear(const LbtFile& f, const std::string& name, LbLinear* out, std::string* err);
// Expands the group scales of a nibble linear on first use (idempotent, NOT thread-safe: call from the serial part of a forward).
void ensure_nib(const LbLinear& w);
// Builds ISA-specific weight copies (AVX-512 only) -- idempotent; linear_forward calls it lazily.
void ensure_isa(const LbLinear& w, Isa isa);

// y[T][m] = (prim [+ res]) (x[T][n]); T arbitrary (internally chunked by kMaxBatch).
void linear_forward(const LbLinear& w, Isa isa, const float* x, int T, float* y, bool use_res, LinearScratch& s,
                    ThreadPool* pool = nullptr);

// Several linears that read the SAME input x (q/k/v, gate/up): grid linears share one activation quantisation/LUT build and run as one
// parallel region; anything else falls back to sequential linear_forward calls. ys[i] gets y_i[T][m_i].
void linear_forward_multi(const LbLinear* const* ws, int nw, Isa isa, const float* x, int T, float* const* ys, bool use_res, LinearScratch& s,
                          ThreadPool* pool = nullptr);

// Many independent (gate+up / down) linear groups in ONE parallel region (MoE experts): activations of every job are quantised once,
// then all row blocks of all jobs/linears are spread over the pool. Results are bit-identical to linear_forward_multi per job.
// Nibble linears only get the fused path; anything else falls back to sequential linear_forward_multi.
void linear_forward_jobs(const LinJob* jobs, int nj, Isa isa, LinearScratch& s, ThreadPool* pool = nullptr);

}  // namespace lb
