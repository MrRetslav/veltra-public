#include "lbt_model.h"
#include "ops.h"
#include "nibble_gemv.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "lb_common.h"
#include "threadpool.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace lb {
double g_prof2_ms[4] = {0};
static const bool g_prof2_on = lb::env_get("PROFILE") != nullptr;
#define PROF2_BEGIN(v) auto v = std::chrono::steady_clock::now()
#define PROF2_END(v, k) do { if (g_prof2_on) g_prof2_ms[k] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - v).count(); } while (0)

// ------------------------------------------------------------------------ MappedFile
#ifdef _WIN32
MappedFile::~MappedFile() {
    if (data_ && heap_) VirtualFree((void*)data_, 0, MEM_RELEASE);
    veltra::plat::unmap_file(&fmap_);
}
bool MappedFile::open(const std::string& path, std::string* err) {
    HANDLE f = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) { if (err) *err = "cannot open " + path; return false; }
    LARGE_INTEGER sz;
    GetFileSizeEx(f, &sz);
    // Default: read the whole file into ordinary committed memory. Streaming weights out of a file mapping
    // is ~2.4x slower for token decode on Windows (decode is memory-bound); LB_MMAP=1 keeps the mapping.
    if (!lb::env_get("MMAP")) {
        void* buf = VirtualAlloc(nullptr, (size_t)sz.QuadPart, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        if (buf) {
            size_t done = 0;
            bool ok = true;
            while (done < (size_t)sz.QuadPart) {
                DWORD chunk = (DWORD)std::min<size_t>((size_t)sz.QuadPart - done, 64u << 20), got = 0;
                if (!ReadFile(f, (char*)buf + done, chunk, &got, nullptr) || got == 0) { ok = false; break; }
                done += got;
            }
            if (ok) {
                CloseHandle(f);
                heap_ = true;
                data_ = (const uint8_t*)buf; size_ = (size_t)sz.QuadPart;
                return true;
            }
            VirtualFree(buf, 0, MEM_RELEASE);
        }
    }
    CloseHandle(f);
    if (!veltra::plat::map_file_ro(path, &fmap_, err)) return false;   // CreateFileMappingA + MapViewOfFile
    data_ = fmap_.data; size_ = fmap_.size;
    return true;
}
#else
MappedFile::~MappedFile() {
    if (anon_base_) { munmap(anon_base_, anon_len_); return; }
    veltra::plat::unmap_file(&fmap_);
}
bool MappedFile::open(const std::string& path, std::string* err) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { if (err) *err = "cannot open " + path; return false; }
    struct stat st;
    if (fstat(fd, &st) != 0) { ::close(fd); if (err) *err = "fstat failed"; return false; }
    if (const char* ah = lb::env_get("ANON_HUGE")) {
        // opt-in: copy the file into an anonymous 2 MB-aligned THP region (madvise HUGEPAGE; LB_ANON_HUGE=2 also mlock's it). Uses file-size RAM, removes 4 KB dTLB pressure.
        const size_t sz = (size_t)st.st_size, al = (size_t)2 << 20, len = (sz + al - 1) / al * al;
        void* raw = mmap(nullptr, len + al, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw != MAP_FAILED) {
            uint8_t* p = (uint8_t*)(((uintptr_t)raw + al - 1) & ~(uintptr_t)(al - 1));
#ifdef MADV_HUGEPAGE
            madvise(p, len, MADV_HUGEPAGE);
#endif
            size_t done = 0;
            bool ok = true;
            while (done < sz) {
                ssize_t r = pread(fd, p + done, std::min<size_t>(sz - done, (size_t)1 << 26), (off_t)done);
                if (r <= 0) { ok = false; break; }
                done += (size_t)r;
            }
            if (ok) {
                if (ah[0] == '2') mlock(p, sz);
                mprotect(p, len, PROT_READ);
                ::close(fd);
                data_ = p; size_ = sz; anon_base_ = raw; anon_len_ = len + al;
                return true;
            }
            munmap(raw, len + al);
        }
    }
    ::close(fd);
    veltra::plat::MapHints hints;
    hints.populate = lb::env_get("PREFAULT") != nullptr;   // benchmarking: map every page up front (no page faults inside the timed decode)
    // Prefetch small models only: for multi-GB MoE files (only ~top-k experts per layer are touched per token) WILLNEED would pull the
    // whole file into the page cache / working set up front. Big files fault in lazily (zero-copy mmap, RSS = touched pages).
    hints.willneed_below = (size_t)3 << 30;
    if (!veltra::plat::map_file_ro(path, &fmap_, err, hints)) return false;
    data_ = fmap_.data; size_ = fmap_.size;
    return true;
}
#endif

// ------------------------------------------------------------------------ LbtFile
bool LbtFile::open(const std::string& path, std::string* err) {
    if (!map_.open(path, err)) return false;
    if (map_.size() < sizeof(LbtHeader)) { if (err) *err = "file too small"; return false; }
    hdr_ = (const LbtHeader*)map_.data();
    // Magic: "VELTRA\0\0" (current writers) or "LITTLEBT" (legacy .lbt files); identical layout, version 1.
    const bool magic_ok = std::memcmp(hdr_->magic, "VELTRA\0\0", 8) == 0 || std::memcmp(hdr_->magic, "LITTLEBT", 8) == 0;
    if (!magic_ok || hdr_->version != 1) { if (err) *err = "not a Veltra v1 model file (.vlt / legacy .lbt)"; return false; }
    const uint8_t* base = map_.data();
    if (hdr_->meta_off + hdr_->meta_len > map_.size() || hdr_->table_off + (uint64_t)hdr_->n_tensors * sizeof(LbtEntry) > map_.size()) {
        if (err) *err = "truncated file";
        return false;
    }
    meta_json_.assign((const char*)base + hdr_->meta_off, (size_t)hdr_->meta_len);
    const LbtEntry* e = (const LbtEntry*)(base + hdr_->table_off);
    tensors_.reserve(hdr_->n_tensors * 2);
    for (uint32_t i = 0; i < hdr_->n_tensors; i++) {
        if (e[i].offset + e[i].nbytes > map_.size()) { if (err) *err = "tensor out of range"; return false; }
        TensorRef t;
        t.data = base + e[i].offset;
        t.dtype = e[i].dtype; t.ndim = e[i].ndim; t.nbytes = e[i].nbytes;
        for (int k = 0; k < 4; k++) t.shape[k] = e[i].shape[k];
        tensors_.emplace(std::string(e[i].name, strnlen(e[i].name, 96)), t);
    }
    return true;
}

const TensorRef* LbtFile::find(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}
const TensorRef& LbtFile::get(const std::string& name) const {
    auto* t = find(name);
    if (!t) fail("missing tensor: " + name);
    return *t;
}

std::vector<float> LbtFile::vec_f32(const std::string& name) const {
    const TensorRef& t = get(name);
    uint64_t n = t.numel();
    std::vector<float> v(n);
    if (t.dtype == DT_F32) std::memcpy(v.data(), t.data, n * 4);
    else if (t.dtype == DT_F16) for (uint64_t i = 0; i < n; i++) { uint16_t h; std::memcpy(&h, t.data + 2 * i, 2); v[i] = f16_to_f32(h); }
    else if (t.dtype == DT_BF16) for (uint64_t i = 0; i < n; i++) { uint16_t h; std::memcpy(&h, t.data + 2 * i, 2); v[i] = bf16_to_f32(h); }
    else fail("vec_f32: unsupported dtype for " + name);
    return v;
}

// ------------------------------------------------------------------------ linear layers
static bool load_path(const LbtFile& f, const std::string& base, LbPath* p, std::string* err) {
    const TensorRef* ul = f.find(base + ".u_lut");
    const TensorRef* vl = f.find(base + ".vt_lut");
    if (!ul || !vl) return false;
    p->m = (int)ul->shape[0];
    p->r = (int)ul->shape[1];
    p->n = (int)vl->shape[1];
    if ((int)vl->shape[0] != p->r) { if (err) *err = "rank mismatch in " + base; return false; }
    if (ul->nbytes < packed_weight_bytes(p->m, p->r) || vl->nbytes < packed_weight_bytes(p->r, p->n)) {
        if (err) *err = "LUT tensor too small in " + base;
        return false;
    }
    p->u.rows = p->m; p->u.cols = p->r; p->u.lut_w = ul->data;
    p->vt.rows = p->r; p->vt.cols = p->n; p->vt.lut_w = vl->data;
    if (auto* ub = f.find(base + ".u_bits")) p->u.bits = (const uint64_t*)ub->data;
    if (auto* vb = f.find(base + ".vt_bits")) p->vt.bits = (const uint64_t*)vb->data;
    p->h = f.vec_f32(base + ".h");
    p->l = f.vec_f32(base + ".l");
    p->g = f.vec_f32(base + ".g");
    if ((int)p->h.size() != p->m || (int)p->l.size() != p->r || (int)p->g.size() != p->n) {
        if (err) *err = "scale vector size mismatch in " + base;
        return false;
    }
    return true;
}

// Bit-plane grid linear: <name>.gp{p}_lut (tile-major LUT8, p = 0..P-1) + <name>.gs (f16 [m, n/128] or [m, 1]).
static bool load_grid(const LbtFile& f, const std::string& name, LbLinear* out, std::string* err) {
    GridLinear& g = out->grid;
    int P = 0;
    while (P < 5 && f.has(name + ".gp" + std::to_string(P) + "_lut")) P++;
    if (P < 2 || P > 4) { if (err) *err = "grid linear " + name + ": need 2..4 planes, found " + std::to_string(P); return false; }
    const TensorRef& t0 = f.get(name + ".gp0_lut");
    if (t0.ndim != 2) { if (err) *err = "grid plane shape of " + name; return false; }
    const int m = (int)t0.shape[0], n = (int)t0.shape[1];
    if (n % kTileK != 0) { if (err) *err = "grid linear " + name + ": in_features must be a multiple of 128"; return false; }
    const int ntile = n / kTileK, rp = pad_rows(m);
    const size_t need = (size_t)ntile * rp * 16;
    for (int p = 0; p < P; p++) {
        const TensorRef& t = f.get(name + ".gp" + std::to_string(p) + "_lut");
        if (t.ndim != 2 || (int)t.shape[0] != m || (int)t.shape[1] != n || t.nbytes < need) {
            if (err) *err = "grid plane " + std::to_string(p) + " of " + name + " has the wrong shape/size";
            return false;
        }
        g.plane[p] = t.data;
    }
    const TensorRef& gst = f.get(name + ".gs");
    if (m <= 0 || gst.numel() % (uint64_t)m != 0) { if (err) *err = "grid gs shape of " + name; return false; }
    const int ng = (int)(gst.numel() / (uint64_t)m);
    if (ng == ntile) g.G = kTileK;
    else if (ng == 1) g.G = 0;
    else { if (err) *err = "grid gs of " + name + " must have n/128 or 1 columns"; return false; }
    const std::vector<float> gs = f.vec_f32(name + ".gs");
    if (g.G == kTileK) {
        g.gs_row.assign((size_t)ntile * rp, 0.f);
        for (int r = 0; r < m; r++)
            for (int t = 0; t < ntile; t++) g.gs_row[(size_t)t * rp + r] = 0.5f * gs[(size_t)r * ntile + t];
    } else {  // row scale: tiles accumulate unscaled (ones), the scale is applied once per row afterwards
        g.gs_row.assign((size_t)rp, 1.f);
        g.rscale.assign((size_t)rp, 0.f);
        for (int r = 0; r < m; r++) g.rscale[r] = 0.5f * gs[r];
    }
    g.P = P; g.ntile = ntile;
    out->m = m; out->n = n;
    if (f.has(name + ".bias")) out->bias = f.vec_f32(name + ".bias");
    return true;
}

// Nibble-INT4 linear: <name>.n4 (U8 [m, n], row-block-major 4096 B tiles) + <name>.gs (f16 [m, n/128]).
static bool load_nibble(const LbtFile& f, const std::string& name, LbLinear* out, std::string* err) {
    const TensorRef& t = f.get(name + ".n4");
    if (t.ndim != 2) { if (err) *err = "n4 shape of " + name; return false; }
    const int m = (int)t.shape[0], n = (int)t.shape[1];
    if (m <= 0 || n <= 0 || n % kTileK != 0) { if (err) *err = "nibble linear " + name + ": in_features must be a multiple of 128"; return false; }
    const int ntile = n / kTileK, rp = pad_rows(m);
    if (t.nbytes < (size_t)(rp / kRowBlock) * ntile * 4096) { if (err) *err = "n4 tensor of " + name + " is too small"; return false; }
    const TensorRef& gst = f.get(name + ".gs");
    if (gst.numel() != (uint64_t)m * ntile) { if (err) *err = "nibble gs of " + name + " must be [m, n/128]"; return false; }
    NibLinear& nb = out->nib;
    nb.gs_src = gst.data; nb.gs_dtype = gst.dtype; nb.rows = m;
    if (gst.dtype != DT_F16 && gst.dtype != DT_F32 && gst.dtype != DT_BF16) { if (err) *err = "nibble gs dtype of " + name; return false; }
    nb.w = t.data; nb.ntile = ntile;
    out->m = m; out->n = n;
    if (f.has(name + ".bias")) out->bias = f.vec_f32(name + ".bias");
    return true;
}

// K-quant linear: <name>.q2k|q3k|q4k (U8 [m, n/256*SB], [group of 4 rows][superblock][row][SB bytes]).
static bool load_qk(const LbtFile& f, const std::string& name, int type, const char* suffix, LbLinear* out, std::string* err) {
    const TensorRef& t = f.get(name + suffix);
    if (t.ndim != 2) { if (err) *err = std::string("qk shape of ") + name; return false; }
    const int m = (int)t.shape[0], nsb = (int)(t.shape[1] / (uint64_t)kQkSbBytes[type]);
    if (m <= 0 || m % kQkGroupRows != 0 || nsb <= 0 || (uint64_t)nsb * kQkSbBytes[type] != t.shape[1]) { if (err) *err = "qk tensor " + name + ": bad shape"; return false; }
    if (t.nbytes < (size_t)m * t.shape[1]) { if (err) *err = "qk tensor " + name + " is too small"; return false; }
    out->qk.type = type; out->qk.nsb = nsb; out->qk.w = t.data;
    out->m = m; out->n = nsb * kQkSuper;
    return true;
}

// INT3 g128 linear: <name>.i3 (U8 [m, n/256*kI3Chunk/64], stream of [row block][tile pair] chunks, see engine/i3_gemv.h).
static bool load_i3(const LbtFile& f, const std::string& name, LbLinear* out, std::string* err) {
    const TensorRef& t = f.get(name + ".i3");
    if (t.ndim != 2) { if (err) *err = "i3 shape of " + name; return false; }
    const int m = (int)t.shape[0];
    const uint64_t per_row = t.shape[1];
    if (m <= 0 || m % 64 != 0 || per_row == 0 || per_row % (kI3Chunk / 64) != 0) { if (err) *err = "i3 tensor " + name + ": bad shape"; return false; }
    const int n = (int)(per_row / (kI3Chunk / 64) * 256);
    if (t.nbytes < i3_bytes(m, n)) { if (err) *err = "i3 tensor " + name + " is too small"; return false; }
    out->i3.w = t.data; out->i3.ntile = n / kTileK;
    out->m = m; out->n = n;
    if (f.has(name + ".bias")) out->bias = f.vec_f32(name + ".bias");
    return true;
}

bool load_linear(const LbtFile& f, const std::string& name, LbLinear* out, std::string* err) {
    if (f.has(name + ".i3")) return load_i3(f, name, out, err);
    if (f.has(name + ".n4")) return load_nibble(f, name, out, err);
    if (f.has(name + ".q2k")) return load_qk(f, name, QK_Q2K, ".q2k", out, err);
    if (f.has(name + ".q3k")) return load_qk(f, name, QK_Q3K, ".q3k", out, err);
    if (f.has(name + ".q4k")) return load_qk(f, name, QK_Q4K, ".q4k", out, err);
    if (f.has(name + ".iq2xs")) return load_qk(f, name, QK_IQ2XS, ".iq2xs", out, err);
    if (f.has(name + ".iq3xxs")) return load_qk(f, name, QK_IQ3XXS, ".iq3xxs", out, err);
    if (f.has(name + ".iq4xs")) return load_qk(f, name, QK_IQ4XS, ".iq4xs", out, err);
    if (f.has(name + ".gp0_lut")) return load_grid(f, name, out, err);
    if (!load_path(f, name + ".prim", &out->prim, err)) {
        if (err && err->empty()) *err = "no primary path for " + name;
        return false;
    }
    out->m = out->prim.m; out->n = out->prim.n;
    std::string e2;
    if (f.has(name + ".res.u_lut")) {
        if (!load_path(f, name + ".res", &out->res, err)) return false;
        if (out->res.m != out->m || out->res.n != out->n) { if (err) *err = "res/prim shape mismatch " + name; return false; }
    }
    if (f.has(name + ".bias")) out->bias = f.vec_f32(name + ".bias");
    return true;
}

void ensure_nib(const LbLinear& wl) {
    NibLinear& nb = const_cast<NibLinear&>(wl.nib);
    if (!nb.valid() || !nb.gs.empty()) return;
    const int m = nb.rows, ntile = nb.ntile, rp = pad_rows(m);
    nb.gs.assign((size_t)ntile * rp, 0.f);
    for (int r = 0; r < m; r++) {
        const size_t rb = (size_t)(r / kRowBlock), rr = (size_t)(r % kRowBlock);
        for (int tl = 0; tl < ntile; tl++) {
            const size_t i = (size_t)r * ntile + tl;
            float v;
            if (nb.gs_dtype == DT_F32) std::memcpy(&v, nb.gs_src + 4 * i, 4);
            else { uint16_t h; std::memcpy(&h, nb.gs_src + 2 * i, 2); v = nb.gs_dtype == DT_F16 ? f16_to_f32(h) : bf16_to_f32(h); }
            nb.gs[(rb * ntile + tl) * 64 + rr] = 0.5f * v;   // half scale: the kernels work on Z = 2S - 15A
        }
    }
}

static void ensure_path(const LbPath& cp) {
    LbPath& p = const_cast<LbPath&>(cp);
    if (!p.valid() || p.u.w512) return;
    p.w512_u.resize(packed_weight_bytes(p.m, p.r));
    repack_lut_avx512(p.u.lut_w, p.m, p.r, p.w512_u.data());
    p.w512_vt.resize(packed_weight_bytes(p.r, p.n));
    repack_lut_avx512(p.vt.lut_w, p.r, p.n, p.w512_vt.data());
    p.u.w512 = p.w512_u.data();
    p.vt.w512 = p.w512_vt.data();
}
static void ensure_grid(const LbLinear& wl, Isa isa) {
    GridLinear& g = const_cast<GridLinear&>(wl.grid);
    const int rp = pad_rows(wl.m);
    if (isa == Isa::Avx512 && g.w512[0].empty()) {
        for (int p = 0; p < g.P; p++) {
            g.w512[p].resize((size_t)g.ntile * rp * 16);
            for (int t = 0; t < g.ntile; t++) repack_lut_avx512(g.plane[p] + (size_t)t * rp * 16, wl.m, kTileK, g.w512[p].data() + (size_t)t * rp * 16);
        }
    } else if (isa == Isa::Avx2 && g.gs_avx2.empty()) {
        g.gs_avx2 = g.gs_row;
        if (g.G == kTileK)
            for (int t = 0; t < g.ntile; t++)
                for (int rb = 0; rb < rp / kRowBlock; rb++)
                    for (int s2 = 0; s2 < kRowBlock; s2++) g.gs_avx2[(size_t)t * rp + rb * kRowBlock + s2] = g.gs_row[(size_t)t * rp + rb * kRowBlock + grid_avx2_perm(s2)];
    }
}
void ensure_isa(const LbLinear& w, Isa isa) {
    if (w.is_qk() || w.is_i3()) return;
    if (w.is_nibble()) { ensure_nib(w); return; }
    if (w.is_grid()) { ensure_grid(w, isa); return; }
    if (isa != Isa::Avx512) return;
    ensure_path(w.prim);
    ensure_path(w.res);
}

// Crest-factor threshold (amax/rms) above which a token's activations are quantised with TWO int8
// planes (x ~= s1*q1 + (s1/254)*q2, ~15 bit effective). LLM activations carry massive outlier channels
// that otherwise dominate the per-token int8 step; the second plane costs one extra multiply-free LUT pass.
static float g_two_plane_crest = lb::env_get("CREST") ? (float)std::atof(lb::env_get("CREST")) : 6.0f;
void set_two_plane_crest(float c) { g_two_plane_crest = c; }
float get_two_plane_crest() { return g_two_plane_crest; }

// out[t][row] (float) = sum_k sign(B[row,k]) * x[t][k], computed with int8 LUT kernels, rows <= view.rows.
static void binary_matvec_f32(const BinMatView& B, Isa isa, const float* x, int T, float* out, LinearScratch& s, ThreadPool* pool) {
    const int kp = pad_k(B.cols), rp = pad_rows(B.rows);
    const size_t lutTok = lut_bytes_per_token(kp);
    struct Plane { int t; float scale; };
    Plane planes[2 * kMaxBatch];
    int E = 0;
    PROF2_BEGIN(tp0);
    s.q.assign((size_t)2 * T * kp, 0);
    s.luts.resize((size_t)2 * T * lutTok);
    s.xs.resize((size_t)T * B.cols);
    // pass A (cheap, serial): decide per token whether it needs the second int8 plane and assign plane slots
    int slot1[2 * kMaxBatch], slot2[2 * kMaxBatch];
    for (int t = 0; t < T; t++) {
        const float* xt = x + (size_t)t * B.cols;
        float amax, ss;
        absmax_sumsq(xt, B.cols, &amax, &ss);
        const float rms = std::sqrt(ss / (float)B.cols);
        slot1[t] = E++;
        slot2[t] = -1;
        if (g_two_plane_crest > 0.f && amax > g_two_plane_crest * rms && amax > 0.f) slot2[t] = E++;
    }
    // pass B: quantize + build LUTs per token (independent -> spread over the pool when T > 1)
    auto prep_token = [&](int t) {
        const float* xt = x + (size_t)t * B.cols;
        int8_t* q1 = &s.q[(size_t)slot1[t] * kp];
        float s1 = quantize_activations_i8(xt, B.cols, kp, q1);
        planes[slot1[t]] = {t, s1};
        build_luts(q1, kp, &s.luts[(size_t)slot1[t] * lutTok], isa == Isa::Avx512);
        if (slot2[t] >= 0) {
            float* xs = &s.xs[(size_t)t * B.cols];
            for (int i = 0; i < B.cols; i++) xs[i] = xt[i] - (float)q1[i] * s1;
            int8_t* q2 = &s.q[(size_t)slot2[t] * kp];
            float s2 = quantize_activations_i8(xs, B.cols, kp, q2);
            planes[slot2[t]] = {t, s2};
            build_luts(q2, kp, &s.luts[(size_t)slot2[t] * lutTok], isa == Isa::Avx512);
        }
    };
    if (pool && pool->size() > 1 && T > 1) pool->run(T, prep_token);
    else for (int t = 0; t < T; t++) prep_token(t);
    PROF2_END(tp0, 0);
    PROF2_BEGIN(tp1);
    // Few row blocks (the V^T stage: rank rows x many columns) cannot keep all threads busy: split K as well and add partials.
    const int nrb = rp / kRowBlock, nthr = pool ? pool->size() : 1;
    static const int par_min_rb = lb::env_get("PAR_MIN_RB") ? std::atoi(lb::env_get("PAR_MIN_RB")) : 8;
    int ksplit = 1;
    if (isa == Isa::Avx512 && nthr > 1 && nrb < par_min_rb) ksplit = std::min(num_k_chunks(B.cols), (2 * nthr + nrb - 1) / nrb);
    const int nkc = num_k_chunks(B.cols);
    s.acc.resize((size_t)E * rp * ksplit);
    for (int e0 = 0; e0 < E; e0 += kMaxBatch) {
        int c = std::min(kMaxBatch, E - e0);
        const int8_t* L = &s.luts[(size_t)e0 * lutTok];
        if (ksplit > 1) {
            pool->run(nrb * ksplit, [&](int ti) {
                const int rb = ti / ksplit, sp = ti % ksplit;
                binary_gemv(isa, B, L, c, &s.acc[((size_t)sp * E + e0) * rp], rb, rb + 1, nkc * sp / ksplit, nkc * (sp + 1) / ksplit);
            });
            continue;
        }
        int32_t* A = &s.acc[(size_t)e0 * rp];
        if (pool && nthr > 1 && nrb >= par_min_rb) {
            const int tasks = std::min(nrb, nthr * 2);
            pool->run(tasks, [&](int ti) {
                int b0 = (int)((long long)nrb * ti / tasks), b1 = (int)((long long)nrb * (ti + 1) / tasks);
                binary_gemv(isa, B, L, c, A, b0, b1);
            });
        } else {
            binary_gemv(isa, B, L, c, A);
        }
    }
    PROF2_END(tp1, 1);
    PROF2_BEGIN(tp2);
    std::fill(out, out + (size_t)T * B.rows, 0.f);
    for (int e = 0; e < E; e++) {
        const int32_t* a = &s.acc[(size_t)e * rp];
        float* o = out + (size_t)planes[e].t * B.rows;
        const float sc = planes[e].scale;
        if (ksplit == 1) {
            vec_acc_i32(o, a, sc, B.rows);
        } else {
            for (int i = 0; i < B.rows; i++) {
                int32_t v = a[i];
                for (int sp = 1; sp < ksplit; sp++) v += s.acc[((size_t)sp * E + e) * rp + i];
                o[i] += (float)v * sc;
            }
        }
    }
    PROF2_END(tp2, 2);
}

static void path_forward_chunk(const LbPath& p, Isa isa, const float* x, int T, float* y, LinearScratch& s, ThreadPool* pool) {
    // stage 1: z = V_b^T (g * x)
    s.y.resize((size_t)T * p.n);
    for (int t = 0; t < T; t++) vec_mul(x + (size_t)t * p.n, p.g.data(), p.n, &s.y[(size_t)t * p.n]);
    std::vector<float> z((size_t)T * p.r);
    binary_matvec_f32(p.vt, isa, s.y.data(), T, z.data(), s, pool);
    for (int t = 0; t < T; t++) vec_mul_inplace(&z[(size_t)t * p.r], p.l.data(), p.r);
    // stage 2: y += h * U_b z
    std::vector<float> u((size_t)T * p.m);
    binary_matvec_f32(p.u, isa, z.data(), T, u.data(), s, pool);
    for (int t = 0; t < T; t++) vec_mad(y + (size_t)t * p.m, &u[(size_t)t * p.m], p.h.data(), p.m);
}

// Grid linears sharing one input (q/k/v, gate/up) for up to kMaxBatch tokens: ONE int8 quantisation (+ optional second residual
// plane) and ONE LUT build per token serve all P weight planes, all tiles and all `nw` linears, and the whole group is a single
// parallel region. Activation planes ("slots", E <= 2*kMaxBatch) go through the fused kernel in chunks of <= kMaxBatch; each
// result row is later combined per token as sum_e scale_e * yE[e][r].
// Debug oracle (LB_GRID_FLOATACT=1): float activations, no int8 quantisation: y = W_hat x straight from the planes.
static void grid_forward_float_debug(const LbLinear& w, const float* x, int T, float* y, ThreadPool* pool) {
    const GridLinear& g = w.grid;
    const int m = w.m, n = w.n, rp = pad_rows(m);
    auto run = [&](int rb) {
        for (int i = 0; i < kRowBlock; i++) {
            const int r = rb * kRowBlock + i;
            if (r >= m) break;
            for (int t = 0; t < T; t++) {
                const float* xt = x + (size_t)t * n;
                double acc = 0.0;
                for (int tile = 0; tile < g.ntile; tile++) {
                    double ts = 0.0;
                    for (int c = 0; c < kTileK; c++) {
                        const int grp = c >> 2, pr = grp >> 1;
                        double q = 0.0;
                        for (int p = 0; p < g.P; p++) {
                            const uint8_t b = g.plane[p][(size_t)tile * rp * 16 + (size_t)rb * 1024 + (size_t)pr * 64 + i];
                            const int bit = (b >> (4 * (grp & 1) + (c & 3))) & 1;
                            q += std::ldexp((double)(2 * bit - 1), g.P - 2 - p);
                        }
                        ts += q * xt[tile * kTileK + c];
                    }
                    acc += ts * (double)(g.gs_row[(g.G == kTileK ? (size_t)tile * rp : 0) + r]) * (g.G == kTileK ? 2.0 : 1.0);
                }
                if (g.G == 0) acc *= (double)g.rscale[r] * 2.0;
                y[(size_t)t * m + r] = (float)acc;
            }
        }
    };
    const int nrb = rp / kRowBlock;
    if (pool && pool->size() > 1) pool->run(nrb, run); else for (int rb = 0; rb < nrb; rb++) run(rb);
}

static void grid_forward_multi(const LbLinear* const* ws, int nw, Isa isa, const float* x, int T, float* const* ys, LinearScratch& s, ThreadPool* pool) {
    static const bool dbg_float = lb::env_get("GRID_FLOATACT") != nullptr;
    if (dbg_float) { for (int i = 0; i < nw; i++) grid_forward_float_debug(*ws[i], x, T, ys[i], pool); return; }
    const int n = ws[0]->n, ntile = ws[0]->grid.ntile;
    const bool l512 = isa == Isa::Avx512;
    float slot_scale[2 * kMaxBatch];
    int slot1[kMaxBatch], slot2[kMaxBatch], E = 0;
    PROF2_BEGIN(tp0);
    s.q.resize((size_t)2 * T * n);
    s.xs.resize((size_t)T * n);
    for (int t = 0; t < T; t++) {
        float amax, ss;
        absmax_sumsq(x + (size_t)t * n, n, &amax, &ss);
        const float rms = std::sqrt(ss / (float)n);
        slot1[t] = E++;
        slot2[t] = -1;
        if (g_two_plane_crest > 0.f && amax > g_two_plane_crest * rms && amax > 0.f) slot2[t] = E++;
    }
    size_t yoff[8];
    size_t ytot = 0;
    for (int i = 0; i < nw; i++) { yoff[i] = ytot; ytot += (size_t)E * pad_rows(ws[i]->m); }
    s.luts.resize((size_t)E * ntile * kTileLutBytes);
    s.yE.resize(ytot);
    // LUT slot layout: chunk c0 (first activation plane e0, size cs): [tile][e - e0][1024 B]
    auto build_slot = [&](int e, const int8_t* q) {
        const int e0 = e - e % kMaxBatch, cs = std::min(kMaxBatch, E - e0);
        int8_t* base = &s.luts[(size_t)e0 * ntile * kTileLutBytes];
        for (int t = 0; t < ntile; t++) build_luts(q + (size_t)t * kTileK, kTileK, base + ((size_t)t * cs + (e - e0)) * kTileLutBytes, l512);
    };
    auto prep_token = [&](int t) {
        const float* xt = x + (size_t)t * n;
        int8_t* q1 = &s.q[(size_t)slot1[t] * n];
        const float s1 = quantize_activations_i8(xt, n, n, q1);
        slot_scale[slot1[t]] = s1;
        build_slot(slot1[t], q1);
        if (slot2[t] >= 0) {
            float* xs = &s.xs[(size_t)t * n];
            for (int i = 0; i < n; i++) xs[i] = xt[i] - (float)q1[i] * s1;
            int8_t* q2 = &s.q[(size_t)slot2[t] * n];
            slot_scale[slot2[t]] = quantize_activations_i8(xs, n, n, q2);
            build_slot(slot2[t], q2);
        }
    };
    if (pool && pool->size() > 1 && T > 1) pool->run(T, prep_token);
    else for (int t = 0; t < T; t++) prep_token(t);
    PROF2_END(tp0, 0);
    PROF2_BEGIN(tp1);
    GridView gv[8];
    int totrb = 0;
    for (int i = 0; i < nw; i++) {
        const GridLinear& g = ws[i]->grid;
        const int rp = pad_rows(ws[i]->m);
        gv[i].rows = ws[i]->m; gv[i].ntile = ntile; gv[i].P = g.P;
        for (int p = 0; p < g.P; p++) gv[i].w[p] = l512 ? g.w512[p].data() : g.plane[p];
        gv[i].gs = isa == Isa::Avx2 ? g.gs_avx2.data() : g.gs_row.data();
        gv[i].gs_stride = g.G == kTileK ? (size_t)rp : 0;
        totrb += rp / kRowBlock;
    }
    const int nthr = pool ? pool->size() : 1;
    // tasks: runs of `gran` row blocks of one linear (about 2 tasks per thread over the whole group)
    const int gran = nthr > 1 ? std::max(1, totrb / (2 * nthr)) : 1 << 20;
    int task_lin[256], task_b0[256], task_b1[256], nt = 0;
    for (int i = 0; i < nw; i++) {
        const int nrb = pad_rows(ws[i]->m) / kRowBlock;
        for (int b0 = 0; b0 < nrb && nt < 256; b0 += gran) { task_lin[nt] = i; task_b0[nt] = b0; task_b1[nt] = std::min(nrb, b0 + gran); nt++; }
    }
    auto task = [&](int ti) {
        const int li = task_lin[ti], b0 = task_b0[ti], b1 = task_b1[ti];
        const LbLinear& w = *ws[li];
        const GridLinear& g = w.grid;
        const int m = w.m, rp = pad_rows(m);
        float* yE = &s.yE[yoff[li]];
        for (int e0 = 0; e0 < E; e0 += kMaxBatch) {
            const int cs = std::min(kMaxBatch, E - e0);
            grid_gemv(isa, gv[li], &s.luts[(size_t)e0 * ntile * kTileLutBytes], cs, yE + (size_t)e0 * rp, b0, b1);
        }
        const float* rsc = g.rscale.empty() ? nullptr : g.rscale.data();
        const int r0 = b0 * kRowBlock, r1 = std::min(m, b1 * kRowBlock);
        for (int t = 0; t < T; t++) {
            float* o = ys[li] + (size_t)t * m;
            const float* a = yE + (size_t)slot1[t] * rp;
            const float s1 = slot_scale[slot1[t]];
            if (slot2[t] < 0) {
                for (int r = r0; r < r1; r++) o[r] = s1 * a[r];
            } else {
                const float* b = yE + (size_t)slot2[t] * rp;
                const float s2 = slot_scale[slot2[t]];
                for (int r = r0; r < r1; r++) o[r] = s1 * a[r] + s2 * b[r];
            }
            if (rsc) for (int r = r0; r < r1; r++) o[r] *= rsc[r];
        }
    };
    if (nt > 1 && pool) pool->run(nt, task); else for (int i = 0; i < nt; i++) task(i);
    PROF2_END(tp1, 1);
}

// ---- nibble-INT4 linears: same activation handling as grid_forward_multi (int8 per token, optional second residual plane), but ONE
// pass over each row block serves all slots and there are no LUTs. yE[e][r] = sum_t gs*(S - 7.5*A) (nibble_gemv.h), combined per token below.
// Per-128-column activation scales (LB_NIB_ACT=tile, like llama.cpp's block-wise q8): ONE int8 slot per token, scale per tile folded into the
// group scale inside the kernel; no second residual plane needed (the outlier only coarsens its own tile).
static bool g_nib_act_tile = lb::env_get("NIB_ACT") && std::string(lb::env_get("NIB_ACT")) == "tile";
void set_nibble_act_tile(bool on) { g_nib_act_tile = on; }
bool get_nibble_act_tile() { return g_nib_act_tile; }

int sub_tokens() {
    static const int v = lb::env_get("SUBTOK") ? std::max(1, std::min(kMaxSub, std::atoi(lb::env_get("SUBTOK")))) : 32;
    return v;
}

static void nibble_forward_multi(const LbLinear* const* ws, int nw, Isa isa, const float* x, int T, float* const* ys, LinearScratch& s, ThreadPool* pool) {
    const int n = ws[0]->n, ntile = ws[0]->n / kTileK;
    const bool tile_act = g_nib_act_tile || s.tile_act;
    float slot_scale[2 * kMaxSub];
    int slot1[kMaxSub], slot2[kMaxSub], E = 0;
    PROF2_BEGIN(tp0);
    s.q.resize((size_t)2 * T * n);
    s.xs.resize((size_t)T * n);
    for (int t = 0; t < T; t++) {
        float amax, ss;
        absmax_sumsq(x + (size_t)t * n, n, &amax, &ss);
        const float rms = std::sqrt(ss / (float)n);
        slot1[t] = E++;
        slot2[t] = -1;
        if (!tile_act && g_two_plane_crest > 0.f && amax > g_two_plane_crest * rms && amax > 0.f) slot2[t] = E++;
    }
    size_t yoff[8];
    size_t ytot = 0;
    for (int i = 0; i < nw; i++) { yoff[i] = ytot; ytot += (size_t)E * pad_rows(ws[i]->m); }
    s.yE.resize(ytot);
    s.acc.resize((size_t)E * ntile);
    if (tile_act) s.z.resize((size_t)E * ntile);
    int32_t* A = s.acc.data();
    auto tile_sums = [&](int e) {
        const int8_t* qe = &s.q[(size_t)e * n];
        for (int tl = 0; tl < ntile; tl++) {
            int32_t a = 0;
            for (int c = 0; c < kTileK; c++) a += qe[tl * kTileK + c];
            A[(size_t)e * ntile + tl] = a;
        }
    };
    auto prep_token = [&](int t) {
        const float* xt = x + (size_t)t * n;
        int8_t* q1 = &s.q[(size_t)slot1[t] * n];
        if (tile_act) {
            float* sc = &s.z[(size_t)slot1[t] * ntile];
            for (int tl = 0; tl < ntile; tl++) {
                const float* xp = xt + (size_t)tl * kTileK;
                float am = 0.f;
                for (int c = 0; c < kTileK; c++) am = std::max(am, std::fabs(xp[c]));
                const float inv = am > 0.f ? 127.f / am : 0.f;
                sc[tl] = am > 0.f ? am / 127.f : 0.f;
                int32_t a = 0;
                for (int c = 0; c < kTileK; c++) { const int v = (int)std::nearbyint(xp[c] * inv); q1[(size_t)tl * kTileK + c] = (int8_t)v; a += v; }
                A[(size_t)slot1[t] * ntile + tl] = a;
            }
            slot_scale[slot1[t]] = 1.f;
            return;
        }
        slot_scale[slot1[t]] = quantize_activations_i8(xt, n, n, q1);
        tile_sums(slot1[t]);
        if (slot2[t] >= 0) {
            float* xs = &s.xs[(size_t)t * n];
            for (int i = 0; i < n; i++) xs[i] = xt[i] - (float)q1[i] * slot_scale[slot1[t]];
            int8_t* q2 = &s.q[(size_t)slot2[t] * n];
            slot_scale[slot2[t]] = quantize_activations_i8(xs, n, n, q2);
            tile_sums(slot2[t]);
        }
    };
    if (pool && pool->size() > 1 && T > 1) pool->run(T, prep_token);
    else for (int t = 0; t < T; t++) prep_token(t);
    PROF2_END(tp0, 0);
    PROF2_BEGIN(tp1);
    NibView nv[8];
    I3View iv[8];
    int totrb = 0;
    for (int i = 0; i < nw; i++) {
        if (ws[i]->is_i3()) { iv[i].rows = ws[i]->m; iv[i].ntile = ntile; iv[i].w = ws[i]->i3.w; }
        else { nv[i].rows = ws[i]->m; nv[i].ntile = ntile; nv[i].w = ws[i]->nib.w; nv[i].gs = ws[i]->nib.gs.data(); }
        totrb += pad_rows(ws[i]->m) / kRowBlock;
    }
    const int nthr = pool ? pool->size() : 1;
    const int gran = nthr > 1 ? std::max(1, totrb / (2 * nthr)) : 1 << 20;
    int task_lin[256], task_b0[256], task_b1[256], nt = 0;
    for (int i = 0; i < nw; i++) {
        const int nrb = pad_rows(ws[i]->m) / kRowBlock;
        for (int b0 = 0; b0 < nrb && nt < 256; b0 += gran) { task_lin[nt] = i; task_b0[nt] = b0; task_b1[nt] = std::min(nrb, b0 + gran); nt++; }
    }
    auto task = [&](int ti) {
        const int li = task_lin[ti], b0 = task_b0[ti], b1 = task_b1[ti];
        const int m = ws[li]->m, rp = pad_rows(m);
        float* yE = &s.yE[yoff[li]];
        if (ws[li]->is_i3()) i3_gemm(isa, iv[li], s.q.data(), (size_t)n, A, E, yE, (size_t)rp, b0, b1, tile_act ? s.z.data() : nullptr);
        else nibble_gemm(isa, nv[li], s.q.data(), (size_t)n, A, E, yE, (size_t)rp, b0, b1, tile_act ? s.z.data() : nullptr);
        const int r0 = b0 * kRowBlock, r1 = std::min(m, b1 * kRowBlock);
        for (int t = 0; t < T; t++) {
            float* o = ys[li] + (size_t)t * m;
            const float* a = yE + (size_t)slot1[t] * rp;
            const float s1 = slot_scale[slot1[t]];
            if (slot2[t] < 0) {
                for (int r = r0; r < r1; r++) o[r] = s1 * a[r];
            } else {
                const float* b = yE + (size_t)slot2[t] * rp;
                const float s2 = slot_scale[slot2[t]];
                for (int r = r0; r < r1; r++) o[r] = s1 * a[r] + s2 * b[r];
            }
        }
    };
    if (nt > 1 && pool) pool->run(nt, task); else for (int i = 0; i < nt; i++) task(i);
    PROF2_END(tp1, 1);
}

// Per token of all jobs: does it need a second int8 plane (crest factor amax/rms above the threshold)?  Independent per token -> parallel over the pool
// (the serial scan of every routed token's activations was ~9 % of a prefill).  Flags are laid out job after job.
static void two_plane_flags(const LinJob* jobs, int nj, std::vector<uint8_t>& fl, ThreadPool* pool) {
    size_t tot = 0;
    for (int j = 0; j < nj; j++) tot += (size_t)jobs[j].T;
    fl.assign(tot, 0);
    if (g_two_plane_crest <= 0.f) return;
    std::vector<size_t> off(nj + 1, 0);
    for (int j = 0; j < nj; j++) off[j + 1] = off[j] + (size_t)jobs[j].T;
    constexpr int kBlk = 8;
    std::vector<std::pair<int, int>> blocks;
    for (int j = 0; j < nj; j++) for (int t0 = 0; t0 < jobs[j].T; t0 += kBlk) blocks.push_back({j, t0});
    auto work = [&](int bi) {
        const int j = blocks[bi].first, t0 = blocks[bi].second, n = jobs[j].w[0]->n;
        for (int t = t0; t < std::min(jobs[j].T, t0 + kBlk); t++) {
            float amax, ss;
            absmax_sumsq(jobs[j].x + (size_t)t * n, n, &amax, &ss);
            const float rms = std::sqrt(ss / (float)n);
            fl[off[j] + t] = (amax > g_two_plane_crest * rms && amax > 0.f) ? 1 : 0;
        }
    };
    if (pool && pool->size() > 1 && tot > 32) pool->run((int)blocks.size(), work);
    else for (int i = 0; i < (int)blocks.size(); i++) work(i);
}

// ---- K-quant (`.qk`) linears: same activation handling as the nibble jobs path (int8 per token, optional second residual plane), plus the
// per-16-column activation sums the K-quant min terms need (q2k_gemv.h). One pass over each 64-row block serves all slots of a chunk.
static void qk_forward_jobs(const LinJob* jobs, int nj, Isa isa, LinearScratch& s, ThreadPool* pool) {
    PROF2_BEGIN(tp0);
    s.nsub.clear();
    size_t qtot = 0, btot = 0, ytot = 0;
    int ntok = 0;
    // per-superblock activation scales (prefill only, LinearScratch::tile_act): no second plane, scale folded into d / dmin by the weight-stationary kernel
    bool tile = s.tile_act && isa != Isa::Scalar;
    for (int j = 0; tile && j < nj; j++) for (int i = 0; tile && i < jobs[j].nw; i++) tile = jobs[j].w[i]->m % 8 == 0 && jobs[j].w[i]->n % kQkSuper == 0;
    if (tile) s.tpf.assign(1, 0); else two_plane_flags(jobs, nj, s.tpf, pool);
    const uint8_t* tpf = s.tpf.data();
    size_t sstot = 0;
    size_t fbase = 0;
    for (int j = 0; j < nj; j++) {
        const LinJob& jb = jobs[j];
        const int n = jb.w[0]->n;
        for (int t0 = 0; t0 < jb.T; t0 += sub_tokens()) {
            s.nsub.emplace_back();
            NibSub& sb = s.nsub.back();
            sb.nw = jb.nw; sb.T = std::min(sub_tokens(), jb.T - t0); sb.n = n; sb.ntile = n / kQkSuper;
            sb.x = jb.x + (size_t)t0 * n;
            int E = 0;
            for (int t = 0; t < sb.T; t++) {
                sb.slot1[t] = E++;
                sb.slot2[t] = -1;
                if (!tile && tpf[fbase + t0 + t]) sb.slot2[t] = E++;
            }
            sb.E = E;
            sb.qoff = qtot; qtot += (size_t)E * n;
            sb.aoff = btot; btot += (size_t)E * (n / 16);
            sb.ssoff = sstot; sstot += (size_t)E * sb.ntile;
            for (int i = 0; i < jb.nw; i++) {
                sb.w[i] = jb.w[i];
                sb.y[i] = jb.y[i] + (size_t)t0 * jb.w[i]->m;
                sb.yoff[i] = ytot; ytot += (size_t)E * jb.w[i]->m;
            }
            ntok += sb.T;
        }
        fbase += (size_t)jb.T;
    }
    s.q.resize(qtot);
    s.b16.resize(btot);
    if (tile) s.ss.resize(sstot);
    s.yE.resize(ytot);
    int8_t* Q = s.q.data();
    int16_t* B = s.b16.data();
    const int nsub = (int)s.nsub.size();
    auto prep = [&](int k) {
        NibSub& sb = s.nsub[k];
        const int n = sb.n;
        static thread_local std::vector<float> xs;
        for (int t = 0; t < sb.T; t++) {
            const float* xt = sb.x + (size_t)t * n;
            int8_t* q1 = Q + sb.qoff + (size_t)sb.slot1[t] * n;
            float s1;
            if (tile) {
                float* sst = s.ss.data() + sb.ssoff + (size_t)sb.slot1[t] * sb.ntile;
                for (int b = 0; b < sb.ntile; b++) sst[b] = quantize_activations_i8(xt + (size_t)b * kQkSuper, kQkSuper, kQkSuper, q1 + (size_t)b * kQkSuper);
                s1 = 1.f;
            } else s1 = quantize_activations_i8(xt, n, n, q1);
            sb.sscale[sb.slot1[t]] = s1;
            qk_sums16(q1, n, B + sb.aoff + (size_t)sb.slot1[t] * (n / 16));
            if (sb.slot2[t] >= 0) {
                xs.resize(n);
                for (int i = 0; i < n; i++) xs[i] = xt[i] - (float)q1[i] * s1;
                int8_t* q2 = Q + sb.qoff + (size_t)sb.slot2[t] * n;
                sb.sscale[sb.slot2[t]] = quantize_activations_i8(xs.data(), n, n, q2);
                qk_sums16(q2, n, B + sb.aoff + (size_t)sb.slot2[t] * (n / 16));
            }
        }
    };
    if (pool && pool->size() > 1 && ntok > 16 && nsub > 1) pool->run(nsub, prep);
    else for (int k = 0; k < nsub; k++) prep(k);
    PROF2_END(tp0, 0);
    PROF2_BEGIN(tp1);
    static const int wst_env = lb::env_get("QK_WST") ? std::atoi(lb::env_get("QK_WST")) : 6;   // min slots per pass for the weight-stationary kernel (prefill only)
    const int wst_min = s.fast_prefill ? wst_env : 1 << 30;
    const int nthr = pool ? pool->size() : 1;
    int totrb = 0;
    for (const NibSub& sb : s.nsub) for (int i = 0; i < sb.nw; i++) totrb += (sb.w[i]->m + kRowBlock - 1) / kRowBlock;
    const int gran = nthr > 1 ? std::max(1, totrb / (4 * nthr)) : 1 << 20;
    s.ntask.clear();
    for (int k = 0; k < nsub; k++)
        for (int li = 0; li < s.nsub[k].nw; li++) {
            const int nrb = (s.nsub[k].w[li]->m + kRowBlock - 1) / kRowBlock;
            for (int b0 = 0; b0 < nrb; b0 += gran) s.ntask.push_back({k, li, b0, std::min(nrb, b0 + gran)});
        }
    if (s.ntask.size() > 2) std::stable_sort(s.ntask.begin(), s.ntask.end(), [&](const NibTask& a, const NibTask& b) {   // longest tasks first (pool hands tasks out in order)
        return (long)s.nsub[a.sub].E * (a.b1 - a.b0) > (long)s.nsub[b.sub].E * (b.b1 - b.b0);
    });
    auto task = [&](int ti) {
        const NibTask& tk = s.ntask[ti];
        const NibSub& sb = s.nsub[tk.sub];
        const LbLinear& w = *sb.w[tk.li];
        const int m = w.m;
        QkView v;
        v.type = w.qk.type; v.rows = m; v.nsb = w.qk.nsb; v.w = w.qk.w;
        float* yE = &s.yE[sb.yoff[tk.li]];
        const int g0 = tk.b0 * (kRowBlock / kQkGroupRows), g1 = std::min(m / kQkGroupRows, tk.b1 * (kRowBlock / kQkGroupRows));
        qk_gemm(isa, v, Q + sb.qoff, (size_t)sb.n, B + sb.aoff, (size_t)(sb.n / 16), sb.E, yE, (size_t)m, g0, g1, tile ? 1 : wst_min, tile ? s.ss.data() + sb.ssoff : nullptr);
        const int r0 = g0 * kQkGroupRows, r1 = g1 * kQkGroupRows;
        for (int t = 0; t < sb.T; t++) {
            float* o = sb.y[tk.li] + (size_t)t * m;
            const float* a = yE + (size_t)sb.slot1[t] * m;
            const float s1 = sb.sscale[sb.slot1[t]];
            if (sb.slot2[t] < 0) {
                for (int r = r0; r < r1; r++) o[r] = s1 * a[r];
            } else {
                const float* b = yE + (size_t)sb.slot2[t] * m;
                const float s2 = sb.sscale[sb.slot2[t]];
                for (int r = r0; r < r1; r++) o[r] = s1 * a[r] + s2 * b[r];
            }
        }
    };
    const int nt = (int)s.ntask.size();
    if (nt > 1 && pool) pool->run(nt, task); else for (int i = 0; i < nt; i++) task(i);
    PROF2_END(tp1, 1);
}

void linear_forward_jobs(const LinJob* jobs, int nj, Isa isa, LinearScratch& s, ThreadPool* pool) {
    {
        bool allqk = nj > 0;
        for (int j = 0; allqk && j < nj; j++)
            for (int i = 0; allqk && i < jobs[j].nw; i++) allqk = jobs[j].w[i]->is_qk() && jobs[j].w[i]->n == jobs[j].w[0]->n;
        if (allqk) { qk_forward_jobs(jobs, nj, isa, s, pool); return; }
    }
    bool ok = !g_nib_act_tile;
    for (int j = 0; ok && j < nj; j++) {
        const LinJob& jb = jobs[j];
        for (int i = 0; ok && i < jb.nw; i++)
            ok = jb.w[i]->is_nibble() && jb.w[i]->bias.empty() && jb.w[i]->n == jb.w[0]->n && jb.w[i]->nib.ntile == jb.w[0]->nib.ntile;
    }
    if (!ok) {
        for (int j = 0; j < nj; j++) {
            const LbLinear* ws[2] = {jobs[j].w[0], jobs[j].w[1]};
            float* ys[2] = {jobs[j].y[0], jobs[j].y[1]};
            linear_forward_multi(ws, jobs[j].nw, isa, jobs[j].x, jobs[j].T, ys, true, s, pool);
        }
        return;
    }
    for (int j = 0; j < nj; j++) for (int i = 0; i < jobs[j].nw; i++) ensure_nib(*jobs[j].w[i]);
    PROF2_BEGIN(tp0);
    // pass A (serial, cheap): split into chunks of <= kMaxBatch tokens, decide the number of activation planes per token, lay out the buffers
    s.nsub.clear();
    size_t qtot = 0, atot = 0, ytot = 0;
    int ntok = 0;
    two_plane_flags(jobs, nj, s.tpf, pool);
    const uint8_t* tpf = s.tpf.data();
    size_t fbase = 0;
    for (int j = 0; j < nj; j++) {
        const LinJob& jb = jobs[j];
        const int n = jb.w[0]->n, ntile = jb.w[0]->nib.ntile;
        for (int t0 = 0; t0 < jb.T; t0 += sub_tokens()) {
            s.nsub.emplace_back();
            NibSub& sb = s.nsub.back();
            sb.nw = jb.nw; sb.T = std::min(sub_tokens(), jb.T - t0); sb.n = n; sb.ntile = ntile;
            sb.x = jb.x + (size_t)t0 * n;
            int E = 0;
            for (int t = 0; t < sb.T; t++) {
                sb.slot1[t] = E++;
                sb.slot2[t] = -1;
                if (tpf[fbase + t0 + t]) sb.slot2[t] = E++;
            }
            sb.E = E;
            sb.qoff = qtot; qtot += (size_t)E * n;
            sb.aoff = atot; atot += (size_t)E * ntile;
            for (int i = 0; i < jb.nw; i++) {
                sb.w[i] = jb.w[i];
                sb.y[i] = jb.y[i] + (size_t)t0 * jb.w[i]->m;
                sb.yoff[i] = ytot; ytot += (size_t)E * pad_rows(jb.w[i]->m);
            }
            ntok += sb.T;
        }
        fbase += (size_t)jb.T;
    }
    s.q.resize(qtot);
    s.acc.resize(atot);
    s.yE.resize(ytot);
    int8_t* Q = s.q.data();
    int32_t* A = s.acc.data();
    const int nsub = (int)s.nsub.size();
    auto prep = [&](int k) {
        NibSub& sb = s.nsub[k];
        const int n = sb.n, ntile = sb.ntile;
        static thread_local std::vector<float> xs;
        for (int t = 0; t < sb.T; t++) {
            const float* xt = sb.x + (size_t)t * n;
            int8_t* q1 = Q + sb.qoff + (size_t)sb.slot1[t] * n;
            const float s1 = quantize_activations_i8(xt, n, n, q1);
            sb.sscale[sb.slot1[t]] = s1;
            auto sums = [&](int e) {
                const int8_t* qe = Q + sb.qoff + (size_t)e * n;
                for (int tl = 0; tl < ntile; tl++) {
                    int32_t a = 0;
                    for (int c = 0; c < kTileK; c++) a += qe[tl * kTileK + c];
                    A[sb.aoff + (size_t)e * ntile + tl] = a;
                }
            };
            sums(sb.slot1[t]);
            if (sb.slot2[t] >= 0) {
                xs.resize(n);
                for (int i = 0; i < n; i++) xs[i] = xt[i] - (float)q1[i] * s1;
                int8_t* q2 = Q + sb.qoff + (size_t)sb.slot2[t] * n;
                sb.sscale[sb.slot2[t]] = quantize_activations_i8(xs.data(), n, n, q2);
                sums(sb.slot2[t]);
            }
        }
    };
    if (pool && pool->size() > 1 && ntok > 16 && nsub > 1) pool->run(nsub, prep);
    else for (int k = 0; k < nsub; k++) prep(k);
    PROF2_END(tp0, 0);
    PROF2_BEGIN(tp1);
    // tasks: runs of row blocks of one linear of one chunk (about 4 tasks per thread over everything)
    const int nthr = pool ? pool->size() : 1;
    int totrb = 0;
    for (const NibSub& sb : s.nsub) for (int i = 0; i < sb.nw; i++) totrb += pad_rows(sb.w[i]->m) / kRowBlock;
    const int gran = nthr > 1 ? std::max(1, totrb / (4 * nthr)) : 1 << 20;
    s.ntask.clear();
    for (int k = 0; k < nsub; k++)
        for (int li = 0; li < s.nsub[k].nw; li++) {
            const int nrb = pad_rows(s.nsub[k].w[li]->m) / kRowBlock;
            for (int b0 = 0; b0 < nrb; b0 += gran) s.ntask.push_back({k, li, b0, std::min(nrb, b0 + gran)});
        }
    if (s.ntask.size() > 2) std::stable_sort(s.ntask.begin(), s.ntask.end(), [&](const NibTask& a, const NibTask& b) {   // longest tasks first (pool hands tasks out in order)
        return (long)s.nsub[a.sub].E * (a.b1 - a.b0) > (long)s.nsub[b.sub].E * (b.b1 - b.b0);
    });
    auto task = [&](int ti) {
        const NibTask& tk = s.ntask[ti];
        const NibSub& sb = s.nsub[tk.sub];
        const LbLinear& w = *sb.w[tk.li];
        const int m = w.m, rp = pad_rows(m);
        NibView nv;
        nv.rows = m; nv.ntile = sb.ntile; nv.w = w.nib.w; nv.gs = w.nib.gs.data();
        float* yE = &s.yE[sb.yoff[tk.li]];
        nibble_gemm(isa, nv, Q + sb.qoff, (size_t)sb.n, A + sb.aoff, sb.E, yE, (size_t)rp, tk.b0, tk.b1, nullptr);
        const int r0 = tk.b0 * kRowBlock, r1 = std::min(m, tk.b1 * kRowBlock);
        for (int t = 0; t < sb.T; t++) {
            float* o = sb.y[tk.li] + (size_t)t * m;
            const float* a = yE + (size_t)sb.slot1[t] * rp;
            const float s1 = sb.sscale[sb.slot1[t]];
            if (sb.slot2[t] < 0) {
                for (int r = r0; r < r1; r++) o[r] = s1 * a[r];
            } else {
                const float* b = yE + (size_t)sb.slot2[t] * rp;
                const float s2 = sb.sscale[sb.slot2[t]];
                for (int r = r0; r < r1; r++) o[r] = s1 * a[r] + s2 * b[r];
            }
        }
    };
    const int nt = (int)s.ntask.size();
    if (nt > 1 && pool) pool->run(nt, task); else for (int i = 0; i < nt; i++) task(i);
    PROF2_END(tp1, 1);
}

void linear_forward_multi(const LbLinear* const* ws, int nw, Isa isa, const float* x, int T, float* const* ys, bool use_res, LinearScratch& s, ThreadPool* pool) {
    if (nw >= 1 && ws[0]->is_qk()) {   // qk linears sharing the input run as fused jobs of <= 2 linears
        for (int i = 0; i < nw; i += 2) {
            LinJob jb;
            jb.nw = std::min(2, nw - i); jb.x = x; jb.T = T;
            bool same = true;
            for (int k = 0; k < jb.nw; k++) { jb.w[k] = ws[i + k]; jb.y[k] = ys[i + k]; same = same && ws[i + k]->is_qk() && ws[i + k]->n == ws[i]->n; }
            if (!same) fail("linear_forward_multi: mixed qk / non-qk linears");
            qk_forward_jobs(&jb, 1, isa, s, pool);
        }
        return;
    }
    bool fused = nw >= 1 && nw <= 8;
    const bool nib = nw >= 1 && (ws[0]->is_nibble() || ws[0]->is_i3());   // n4 and i3 linears share the activation preparation (mixed groups are fine)
    for (int i = 0; fused && i < nw; i++)
        fused = (nib ? (ws[i]->is_nibble() || ws[i]->is_i3()) : ws[i]->is_grid()) && ws[i]->n == ws[0]->n && ws[i]->bias.empty() &&
                (nib ? true : ws[i]->grid.ntile == ws[0]->grid.ntile);
    if (!fused) {
        for (int i = 0; i < nw; i++) linear_forward(*ws[i], isa, x, T, ys[i], use_res, s, pool);
        return;
    }
    for (int i = 0; i < nw; i++) ensure_isa(*ws[i], isa);
    const int chunk = nib ? sub_tokens() : kMaxBatch;
    for (int t0 = 0; t0 < T; t0 += chunk) {
        const int c = std::min(chunk, T - t0);
        float* yc[8];
        for (int i = 0; i < nw; i++) yc[i] = ys[i] + (size_t)t0 * ws[i]->m;
        if (nib) nibble_forward_multi(ws, nw, isa, x + (size_t)t0 * ws[0]->n, c, yc, s, pool);
        else grid_forward_multi(ws, nw, isa, x + (size_t)t0 * ws[0]->n, c, yc, s, pool);
    }
}

void linear_forward(const LbLinear& w, Isa isa, const float* x, int T, float* y, bool use_res, LinearScratch& s, ThreadPool* pool) {
    if (w.is_qk()) {
        LinJob jb;
        jb.nw = 1; jb.w[0] = &w; jb.x = x; jb.T = T; jb.y[0] = y;
        qk_forward_jobs(&jb, 1, isa, s, pool);
        return;
    }
    ensure_isa(w, isa);
    std::fill(y, y + (size_t)T * w.m, 0.f);
    const int chunk = (w.is_nibble() || w.is_i3()) ? sub_tokens() : kMaxBatch;
    for (int t0 = 0; t0 < T; t0 += chunk) {
        int c = std::min(chunk, T - t0);
        const float* xc = x + (size_t)t0 * w.n;
        float* yc = y + (size_t)t0 * w.m;
        if (w.is_nibble() || w.is_i3()) { const LbLinear* wp = &w; nibble_forward_multi(&wp, 1, isa, xc, c, &yc, s, pool); continue; }
        if (w.is_grid()) { const LbLinear* wp = &w; grid_forward_multi(&wp, 1, isa, xc, c, &yc, s, pool); continue; }
        path_forward_chunk(w.prim, isa, xc, c, yc, s, pool);
        if (use_res && w.has_res()) path_forward_chunk(w.res, isa, xc, c, yc, s, pool);
    }
    if (!w.bias.empty())
        for (int t = 0; t < T; t++)
            for (int i = 0; i < w.m; i++) y[(size_t)t * w.m + i] += w.bias[i];
}

}  // namespace lb
