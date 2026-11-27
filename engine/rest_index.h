// REST: static retrieval draft index for lossless greedy speculative decoding.
//
// A flat open-addressing hash table, memory-mapped from `rest.idx` (built offline by scripts/rest_build.py), maps the hash of the last
// 2..4 token ids to a pre-computed continuation chain (<= 8 tokens, the greedy path of the corpus n-gram model) plus the per-step
// probability that each chain token matches.  One lookup touches one 64-byte bucket (2 entries) + at most 3 adjacent buckets; no
// pointer chasing, no allocation.  The drafts are verified by the exact target model (SpecDecoder), so output stays bit-identical.
//
// File layout (little endian):
//   [0,64)      header  : "RESTIDX1" u32 version(1) u32 0 u32 probe(4) u32 max_chain(8) u32 n_entries u32 orders_mask u32 0 u32 0 | @40 u64 n_buckets | @48 u64 corpus_tokens
//   [4096, ...) buckets : n_buckets * 64 B, each = 2 entries of 32 B (slot 0 is filled before slot 1; a bucket with a free slot ends a probe sequence)
//   entry (32 B): u32 tag (0 = empty) | u32 pq (8 nibbles, nibble j = round(p_j * 15), 0 ends the chain) | 8 x 24-bit token ids
//   hash(ctx[0..n)) = see rest_hash(); bucket = ((hash & 0xffffffff) * n_buckets) >> 32, tag = hash >> 32 (0 -> 1)
#pragma once
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <string>

#include "platform.h"

namespace lb {

constexpr int kRestMaxChain = 8;
constexpr int kRestMaxCost = 10;   // cost table indexed by verify length T = 1..kRestMaxCost-1

inline uint64_t rest_hash(const int32_t* ctx, int n) {
    uint64_t h = 0x243F6A8885A308D3ull ^ ((uint64_t)n * 0x100000001B3ull);
    for (int i = 0; i < n; i++) {
        h = (h ^ ((uint64_t)(uint32_t)ctx[i] + 1ull)) * 0x9E3779B97F4A7C15ull;
        h ^= h >> 29;
    }
    h ^= h >> 32;
    h *= 0xD6E8FEB86659FD93ull;
    h ^= h >> 32;
    return h;
}

struct RestDraft {
    int n = 0;                          // chain length chosen for verification (0 = no draft)
    int tok[kRestMaxChain] = {0};
    float p[kRestMaxChain] = {0};       // raw (uncalibrated) match probability of each chain step
    int order = 0;                      // n-gram order that produced it
    float gain = 0.f;                   // calibrated expected speedup of this verify pass vs plain decode (>1 = profitable)
    float eacc = 0.f;                   // calibrated expected accepted tokens
};

// Online calibration of the table probabilities against what the target model really accepts (per chain step), including
// "shadow" evaluation of skipped candidates, so the gate keeps learning while drafting is switched off.
struct RestGate {
    double pred[kRestMaxChain], hit[kRestMaxChain];
    // shadow candidate (not verified, only compared with the tokens plain decode emits)
    int sh_n = 0, sh_j = 0, sh_tok[kRestMaxChain] = {0};
    float sh_p[kRestMaxChain] = {0};
    RestGate() { for (int j = 0; j < kRestMaxChain; j++) { pred[j] = 4.0; hit[j] = 4.0 * 0.7; } }
    float calib(int j) const { double r = hit[j] / pred[j]; return (float)(r < 0.05 ? 0.05 : r > 1.5 ? 1.5 : r); }
    void update(int j, float p, bool ok) { pred[j] = pred[j] * 0.97 + p; hit[j] = hit[j] * 0.97 + (ok ? 1.0 : 0.0); }
    void shadow_set(const RestDraft& d) {
        if (sh_n > 0 || d.n <= 0) return;
        sh_n = d.n; sh_j = 0;
        for (int j = 0; j < d.n; j++) { sh_tok[j] = d.tok[j]; sh_p[j] = d.p[j]; }
    }
    void shadow_feed(int token) {
        if (sh_n <= 0) return;
        const bool ok = token == sh_tok[sh_j];
        update(sh_j, sh_p[sh_j], ok);
        if (!ok || ++sh_j >= sh_n) sh_n = 0;
    }
};

// Byte accountant: relative cost of a verify pass of T tokens versus one plain T=1 pass.
//   cost(T) = bytes(T) / bytes(1) + g * (T - 1)
//   bytes(T) = F + U(T) * e   (F = bytes that do not depend on T: attention, head scan, norms; e = bytes of one expert; U(T) = unique experts summed over the layers = popcount of
//                              the per-layer uint64 expert masks of the T tokens, Transformer::unique_experts_last())
// bytes(T)/bytes(1) = (f + U(T)/U(1)) / (f + 1) with f = F / (U(1) e) (prior 0.9: measured ~0.6 GB fixed vs ~0.6 GB of experts per token on the q30b qk model).
// g = extra compute per additional token (n4 attention is compute bound, one head dot per token), prior 0.13 = fitted to bench-batch (T=1 47 ms, T=3 88, T=5 127).
// The routing of the drafted positions is only known inside the layers, so U(T) is a running estimate: prior = top_k + 0.5 x (independent-routing union - top_k), then an EMA of the
// observed unique-expert counts of real passes; g is an EMA of (measured time ratio - byte ratio) / (T - 1). Both calibrate while REST is on, also from the T=1 passes.
struct RestCost {
    float tab[kRestMaxCost];          // cost(T), tab[1] = 1
    double U[kRestMaxCost];           // unique experts summed over layers (running estimate)
    int nU[kRestMaxCost] = {0};
    double ms[kRestMaxCost] = {0};    // EMA of measured pass time
    int nms[kRestMaxCost] = {0};
    double f = 0.9, g = 0.13;
    explicit RestCost(int n_layers = 48, int n_experts = 128, int top_k = 8) {
        for (int t = 1; t < kRestMaxCost; t++) {
            const double indep = n_experts * (1.0 - std::pow(1.0 - (double)top_k / n_experts, t));
            U[t] = n_layers * (t == 1 ? (double)top_k : top_k + 0.5 * (indep - top_k));   // consecutive positions overlap: measured 12.0 / 15.4 / 18.4 unique experts per layer for T = 2 / 3 / 4
        }
        refresh();
    }
    double bytes_ratio(int T) const { return (f + U[T] / U[1]) / (f + 1.0); }
    void refresh() { tab[0] = 1.f; for (int t = 1; t < kRestMaxCost; t++) tab[t] = (float)(bytes_ratio(t) + g * (t - 1)); }
    // T tokens were verified in `t_ms` and the pass opened `uniq` unique experts (summed over layers)
    void observe(int T, double t_ms, double uniq) {
        if (T < 1 || T >= kRestMaxCost) return;
        if (uniq > 0) { U[T] = nU[T] == 0 ? uniq : 0.9 * U[T] + 0.1 * uniq; nU[T]++; }
        ms[T] = nms[T] == 0 ? t_ms : 0.8 * ms[T] + 0.2 * t_ms;
        nms[T]++;
        if (nms[1] >= 3) for (int t = 2; t < kRestMaxCost; t++) if (nms[t] >= 3) {
            const double gg = (ms[t] / ms[1] - bytes_ratio(t)) / (t - 1);
            g = 0.9 * g + 0.1 * std::min(0.4, std::max(0.0, gg));
        }
        refresh();
    }
};

class RestIndex {
public:
    RestIndex() = default;
    ~RestIndex();
    RestIndex(const RestIndex&) = delete;
    RestIndex& operator=(const RestIndex&) = delete;
    bool open(const std::string& path, std::string* err);
    bool loaded() const { return buckets_ != nullptr; }
    // Looks up orders 4,3,2 of the context suffix and returns the best draft (highest calibrated gain). cost[T] = time of a verify pass of
    // T tokens relative to one plain decode pass (cost[1] = 1). Returns false when no candidate reaches `min_gain`; `cand` still receives
    // the best candidate (n > 0) when one exists, so the caller can shadow-evaluate it.
    bool propose(const int32_t* ctx, size_t S, int max_len, const float* cost, const RestGate& gate, float min_gain, RestDraft& cand) const;
    uint64_t n_entries() const { return n_entries_; }
    size_t mapped_bytes() const { return map_bytes_; }
    uint32_t orders_mask() const { return orders_; }
    // exact chain lookup of one context (tests / leak scan): returns chain length, tokens in tok[], probabilities in p[]
    int lookup(const int32_t* ctx, int n, int* tok, float* p) const;
    int lookup_hash(uint64_t h, int* tok, float* p) const;   // same by precomputed rest_hash (builder cross-check)
private:
    struct Entry { uint32_t tag, pq; uint8_t tok[24]; };
    const Entry* find(const int32_t* ctx, int n) const;
    const Entry* find_hash(uint64_t h) const;
    uint64_t bucket_of(uint64_t h) const { return ((h & 0xFFFFFFFFull) * nb_) >> 32; }
    const uint8_t* base_ = nullptr;
    const Entry* buckets_ = nullptr;
    size_t map_bytes_ = 0;
    uint64_t nb_ = 0, n_entries_ = 0;
    uint32_t probe_ = 4, orders_ = 0;
    void* map_ = nullptr;
    veltra::plat::FileMap fmap_;   // zero-copy mapping (POSIX mmap / Win32 MapViewOfFile)
    void* raw_ = nullptr;
    bool owned_ = false;
};

// Process-wide index (loaded once by lb_server / lb_gen); nullptr when REST is not enabled.
RestIndex* rest_index();
bool rest_load(const std::string& path, std::string* err);
// default verify-pass cost table (relative to one plain decode pass), overridable with LB_REST_COST="1,1.2,1.4,..."
const float* rest_cost_table();

}  // namespace lb
