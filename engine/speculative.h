// Multi-stage self-speculative decoding.
//   Stage A  zero-cost 3/4-gram prompt+context lookahead (no neural compute)
//   Stage B  primary-path-only + layer-skipped draft model for the remaining draft tokens
//   Stage C  one batched verify pass through the FULL model (prim+res, all layers) with exact
//            speculative rejection sampling and O(1) KV-cache rollback (set len)
//   Stage D  adaptive draft window K in [k_min, k_max] driven by the recent acceptance rate
// Greedy decoding is token-for-token identical to plain decoding (kernels are batch-invariant).
#pragma once
#include <cstdint>
#include <functional>
#include <random>
#include <unordered_map>
#include <vector>

#include "engine.h"
#include "transformer.h"

namespace lb {

// Applies temperature / top-k / top-p to logits. temperature <= 0 => greedy one-hot.
void sampling_probs(const float* logits, int V, const SamplingParams& sp, std::vector<float>& out);
int argmax(const float* v, int n);
// Tokens per prefill forward (--prefill-chunk / LB_PREFILL_CHUNK, default 64): larger chunks stream every weight fewer times.
int prefill_chunk();
void set_prefill_chunk(int n);

class NgramIndex {
public:
    // Index the whole context (call once per generation with the prompt).
    void reset() { map3_.clear(); map4_.clear(); indexed_ = 0; }
    // Propose up to max_len continuation tokens for the suffix of ctx; empty if no match.
    std::vector<int> propose(const std::vector<int>& ctx, int max_len, bool allow3);
    // Make all grams that END at positions <= limit available for later lookups.
    void index_upto(const std::vector<int>& ctx, size_t limit);
private:
    static uint64_t key(const int* p, int n) {
        uint64_t h = 1469598103934665603ull;
        for (int i = 0; i < n; i++) { h ^= (uint64_t)(uint32_t)p[i]; h *= 1099511628211ull; }
        return h;
    }
    std::unordered_map<uint64_t, int> map3_, map4_;   // gram hash -> end index (next token position)
    size_t indexed_ = 0;                              // grams ending at e <= indexed_ are inserted
};

struct SpecResult {
    std::vector<int> tokens;   // generated tokens (excluding prompt)
    GenStats stats;
};

// token callback: (token) -> keep going?
using TokenSink = std::function<bool(int)>;

class SpecDecoder {
public:
    explicit SpecDecoder(Transformer& model) : m_(model) {}
    // Generates until eos/stop/max_new/cancel. The KV cache is reset at the start.
    // reuse > 0: the KV cache already holds valid K/V for positions [0, reuse) of exactly this prompt (prefix cache); only the
    // remaining prompt tokens are prefilled. The result is bit-identical to reuse = 0 (kernels are batch-invariant).
    GenStats run(const std::vector<int>& prompt, const GenOptions& opt, const std::vector<int>& eos_ids, const TokenSink& sink, int reuse = 0);
    // Tokens whose K/V are valid in the KV cache after the last run() (prompt + generated, minus the not-yet-fed last token).
    const std::vector<int>& committed() const { return committed_; }
private:
    std::vector<int> committed_;
    Transformer& m_;
    NgramIndex ngram_;
};

}  // namespace lb
