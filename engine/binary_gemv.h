// T-MAC style multiplication-free binary GEMV:  out[row] = sum_k sign(B[row,k]) * a[k]
//
// Weights  : +-1 matrix, bit=1 -> +1, bit=0 -> -1, packed 4 bits per "group".
// Acts     : INT8 (per-token dynamic quantisation), [-127,127].
// LUT      : for each group of 4 activations, the 16 possible signed partial sums
//            T[p] = sum_i (bit_i(p) ? +a_i : -a_i)  (|T| <= 508, fits INT16). Stored as
//            two byte planes (lo/hi) so that a byte-shuffle (pshufb / vpermb / tbl)
//            performs 16/32/64 table lookups at once. No multiply instruction anywhere.
//
// Packed weight layout ("LUT8"): [row_block(64 rows)][pair p = groups 2p,2p+1][64 bytes],
// byte i = nibble(row_block*64+i, group 2p) | nibble(.., group 2p+1) << 4.
// Rows are padded to a multiple of 64, columns (K) to a multiple of 128.
#pragma once
#include <cstddef>
#include <cstdint>
#include <new>
#include <vector>

namespace lb {

// 64-byte aligned storage (zmm loads that straddle cache lines are ~1.5x slower)
template <class T>
struct AlignedAlloc {
    using value_type = T;
    AlignedAlloc() = default;
    template <class U> AlignedAlloc(const AlignedAlloc<U>&) {}
    T* allocate(size_t n) { return (T*)::operator new(n * sizeof(T), std::align_val_t(64)); }
    void deallocate(T* p, size_t) { ::operator delete(p, std::align_val_t(64)); }
    template <class U> bool operator==(const AlignedAlloc<U>&) const { return true; }
    template <class U> bool operator!=(const AlignedAlloc<U>&) const { return false; }
};
template <class T> using AlignedVec = std::vector<T, AlignedAlloc<T>>;

enum class Isa : int { Scalar = 0, Avx2 = 1, Avx512 = 2, Neon = 3 };

constexpr int kRowBlock = 64;
constexpr int kTileGroups = 32;               // 32 groups * |508| < 2^15 -> safe INT16 accumulation
constexpr int kTileK = kTileGroups * 4;       // 128
constexpr int kMaxBatch = 16;                 // tokens per kernel call (LUTs for more tokens are chunked)

inline int pad_rows(int r) { return (r + kRowBlock - 1) & ~(kRowBlock - 1); }
inline int pad_k(int k) { return (k + kTileK - 1) & ~(kTileK - 1); }
inline size_t lut_bytes_per_token(int k_pad) { return (size_t)k_pad / 4 * 32; }
inline size_t packed_weight_bytes(int rows, int cols) {
    return (size_t)pad_rows(rows) / kRowBlock * ((size_t)pad_k(cols) / 8) * 64;
}
inline size_t bits_words_per_row(int cols) { return ((size_t)cols + 63) / 64; }

const char* isa_name(Isa isa);
bool isa_available(Isa isa);
Isa best_isa();

struct BinMatView {
    int rows = 0, cols = 0;
    const uint8_t* lut_w = nullptr;   // LUT8 layout
    const uint8_t* w512 = nullptr;    // AVX-512 repacked weights (repack_lut_avx512); required by gemv_avx512
    const uint64_t* bits = nullptr;   // optional canonical row-major uint64 bit-packing
};

// Canonical uint64 bits (row-major, bit k of row j at word k/64, bit k%64) -> LUT8 layout.
void pack_bits_to_lut(const uint64_t* bits, int rows, int cols, uint8_t* out);
// Inverse of pack_bits_to_lut for the first `cols` columns (used for cross-checks).
void unpack_lut_to_bits(const uint8_t* lut, int rows, int cols, uint64_t* bits);

// Per-token symmetric INT8 quantisation. q is written with k_pad entries (zero padded).
// Returns the scale (x ~= q * scale).
float quantize_activations_i8(const float* x, int n, int k_pad, int8_t* q);

// Build LUTs for one token (k_pad/4 groups x 32 bytes: 16 lo bytes, 16 hi bytes).
// avx512_layout=true: per 8 groups one 256-byte block [even-group lo x4 | even hi x4 | odd lo x4 | odd hi x4] (16 B per
// group plane) -- the layout gemv_avx512 reads with one vpermb per 4 groups.
void build_luts(const int8_t* q, int k_pad, int8_t* lut, bool avx512_layout = false);
// LUT8 weights -> AVX-512 layout: [row_block][k chunk of 16 vectors][row group of 16 rows][vector: 64 B]; byte 4*j+b of
// a vector = LUT8 byte of row 16g+j for pair 4w+b (4 adjacent bytes = one row, so vpdpbusd sums 4 lookups per row).
void repack_lut_avx512(const uint8_t* lut_w, int rows, int cols, uint8_t* out);
// max |x| and sum x^2 in one pass (SIMD where available)
void absmax_sumsq(const float* x, int n, float* amax, float* sumsq);

// out[t][pad_rows(rows)] (int32) for t in [0,T), T <= kMaxBatch. luts: T consecutive per-token LUTs.
// AVX-512 only: k_begin/k_end select a range of K chunks (kKChunk weight vectors = 512 columns each); the output is then
// the partial sum over that range (callers add the partials). Other ISAs ignore the range and must be called with the full one.
constexpr int kKChunk = 16;
inline int num_k_chunks(int cols) { return (pad_k(cols) / 32 + kKChunk - 1) / kKChunk; }
void binary_gemv(Isa isa, const BinMatView& w, const int8_t* luts, int T, int32_t* out, int rb_begin = 0, int rb_end = -1,
                 int k_begin = 0, int k_end = -1);

// ---- bit-plane grid linears (docs/bitplane_spec.md): P tile-major planes, group = one 128-column tile ----
// out(e, r) = sum_tile gs[tile][r] * sum_p 2^(P-1-p) * S_p(e, tile, r),  S_p = sum_{c in tile} sign_p[r,c] * a_e[c]  (int8 LUT kernel)
// w[p]    : plane p, tile-major; scalar/AVX2 read the LUT8 bytes, AVX-512 reads the repack_lut_avx512 copy of every tile
//           (tile t of a plane starts at t * pad_rows(rows) * 16 bytes in both layouts)
// gs      : float [tile * gs_stride + row] (AVX2: row index = storage order of gemv_avx2, see grid_avx2_perm); gs_stride may be 0
// luts    : per call E <= kMaxBatch activations, layout [tile][e][1024 bytes] (each 1024 B = build_luts of 128 columns, ISA layout)
// yE      : float [E][pad_rows(rows)] row order, written for row blocks [rb_begin, rb_end)
struct GridView {
    int rows = 0, ntile = 0, P = 0;
    const uint8_t* w[4] = {nullptr, nullptr, nullptr, nullptr};
    const float* gs = nullptr;
    size_t gs_stride = 0;
};
constexpr size_t kTileLutBytes = (size_t)kTileGroups * 32;   // 1024
inline int grid_avx2_perm(int s) { return ((s >> 4) >> 1) * 32 + 16 * ((s & 15) >> 3) + ((s >> 4) & 1) * 8 + (s & 7); }
void grid_gemv(Isa isa, const GridView& g, const int8_t* luts, int E, float* yE, int rb_begin = 0, int rb_end = -1);
void grid_scalar(const GridView& g, const int8_t* luts, int E, float* yE, int rb_begin, int rb_end);
#if defined(__x86_64__) || defined(_M_X64)
void grid_avx2(const GridView& g, const int8_t* luts, int E, float* yE, int rb_begin, int rb_end);
void grid_avx512(const GridView& g, const int8_t* luts, int E, float* yE, int rb_begin, int rb_end);
#endif

// Independent oracle straight from the +-1 bit matrix (does not touch the LUT layout).
void binary_gemv_direct(const uint64_t* bits, int rows, int cols, const int8_t* a, int32_t* out);

// Kernel entry points (implemented in separate translation units).
void gemv_scalar(const BinMatView& w, const int8_t* luts, int T, int32_t* out, int rb_begin = 0, int rb_end = -1);
#if defined(__x86_64__) || defined(_M_X64)
void gemv_avx2(const BinMatView& w, const int8_t* luts, int T, int32_t* out, int rb_begin = 0, int rb_end = -1);
void gemv_avx512(const BinMatView& w, const int8_t* luts, int T, int32_t* out, int rb_begin = 0, int rb_end = -1, int k_begin = 0, int k_end = -1);
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
void gemv_neon(const BinMatView& w, const int8_t* luts, int T, int32_t* out, int rb_begin = 0, int rb_end = -1);
#endif

}  // namespace lb
