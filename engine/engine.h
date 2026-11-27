// High-level engine API: load a .lbt, tokenize, generate (plain or self-speculative), stream tokens.
#pragma once
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "search_verifier.h"

namespace lb {

struct ModelConfig {
    std::string arch;          // "qwen3" | "qwen2" | "llama" | "granite"
    std::string model_name;
    int hidden_size = 0, num_layers = 0, num_heads = 0, num_kv_heads = 0, head_dim = 0;
    int intermediate_size = 0, vocab_size = 0, max_position_embeddings = 0;
    // mixture-of-experts FFN (Qwen3-MoE): num_experts > 0 enables it for layers that have a router tensor
    int num_experts = 0, num_experts_per_tok = 0, moe_intermediate_size = 0;
    bool norm_topk_prob = true;
    float rope_theta = 10000.f, rms_norm_eps = 1e-6f;
    float attn_scale = 0.f;    // factor on q.k before the softmax; 0 = 1/sqrt(head_dim). Granite sets its own (typically 1/head_dim).
    float attention_scale() const;
    bool tie_word_embeddings = false, qk_norm = false, attn_bias = false;
    int bos_token_id = -1;
    std::vector<int> eos_token_ids;
    std::string chat_template;           // raw jinja (informational); engine uses built-in ChatML/Llama3 by arch
    std::vector<int> draft_layer_skip;   // layers skipped during draft
    // Hybrid layout (Qwen3.5 / Qwen3.6, docs/qwen36_lbt_contract.md); empty `layer_full` = every layer is a classic full-attention layer (all older models).
    std::vector<uint8_t> layer_full;     // per layer: 1 = full attention, 0 = Gated DeltaNet (meta `layer_types`)
    int rotary_dim = 0;                  // RoPE dims per head (0 = head_dim)
    bool attn_output_gate = false;       // q_proj has 2x rows: [q | gate] per head, attention output * sigmoid(gate)
    int lin_k_heads = 0, lin_v_heads = 0, lin_k_dim = 0, lin_v_dim = 0, lin_conv = 4;   // DeltaNet geometry
    int shared_expert_inter = 0;         // > 0: every MoE layer adds sigmoid(x . sh_gate) * shared_expert(x)
    bool hybrid() const { return !layer_full.empty(); }
    int num_full_layers() const { if (layer_full.empty()) return num_layers; int n = 0; for (auto f : layer_full) n += f; return n; }
};

struct ChatMessage { std::string role, content; };

struct SamplingParams {
    float temperature = 0.f;   // 0 => greedy (argmax)
    float top_p = 1.f;
    int top_k = 0;             // 0 => disabled
    uint64_t seed = 0;
};

struct SpecOptions {
    bool enabled = true;       // self-speculative decoding
    bool ngram = true;         // stage A: 3/4-gram prompt/context lookahead
    bool neural = false;       // stage B: neural primary-draft (off by default: weak on 0.55 bpp checkpoints)
    bool layer_skip = true;    // stage B: skip draft_layer_skip layers during draft
    int k_min = 2, k_max = 8;  // stage D: adaptive window bounds
    int k_init = 5;
    bool adaptive = true;
    bool rest = false;         // REST: static retrieval index + calibrated cost gate (greedy only; needs rest_load() for the static part)
    int rest_k = 2;            // maximum drafted tokens per REST verify pass
    float rest_min_gain = 1.30f;   // draft only when the calibrated expected speedup of the verify pass is >= this
    bool mtp = false;          // native MTP drafts (Qwen3.6 MTP block; greedy requests only, lossless): replaces the n-gram / neural / REST stages
    int mtp_k = 2;             // maximum MTP drafts per verify pass (verify T <= 3); the byte accountant picks 0..mtp_k per pass
};

inline float ModelConfig::attention_scale() const { return attn_scale > 0.f ? attn_scale : 1.0f / std::sqrt((float)head_dim); }

// Batched test-time search (engine/search.cpp): n independent samples of the same prompt, decoded in one batch over a shared prompt KV cache,
// then ranked by the blind verifier (engine/search_verifier.h). n <= 1 = off (the normal decoder, untouched).
struct SearchOptions {
    int n = 1;
    float temperature = 0.65f, top_p = 0.95f;
    uint64_t base_seed = 1;                    // branch i samples with seed base_seed + i
    VerifySpec verify;                         // only what the request itself asked for
    std::vector<std::string> stop_strings;     // a branch ends (text cut) at the first of these
    std::function<bool()> keep_going;          // optional: false = the client is gone, stop all branches
};

struct GenOptions {
    int max_new_tokens = 256;
    SamplingParams sampling;
    SpecOptions spec;
    std::vector<int> stop_tokens;   // additional stop ids (eos ids from the model are always stops)
    int n_threads = 0;              // 0 => hardware concurrency
    SearchOptions search;
};

struct GenStats {
    int prompt_tokens = 0, generated_tokens = 0;
    double prefill_s = 0, decode_s = 0, tokens_per_s = 0;   // tokens_per_s = generated / decode_s
    int drafted = 0, accepted = 0, verify_passes = 0;       // speculative telemetry
    int ngram_drafted = 0, ngram_accepted = 0;
    int rest_lookups = 0, rest_hits = 0, rest_passes = 0, rest_drafted = 0, rest_accepted = 0;   // static REST index telemetry (passes = verify passes drafted from the static index)
    double rest_draft_s = 0;                                 // time spent drafting (session + static lookups) while REST is on
    int head_screened = 0, head_full = 0;                   // single-token heads served by the 4-bit screened head (--screen-head) / by the full head while screening was enabled
    int cached_tokens = 0;                                  // prompt tokens served from the prefix KV cache (prefill skipped)
    int mtp_drafted[2] = {0, 0}, mtp_accepted[2] = {0, 0};  // MTP: drafts verified / accepted per chain position (position 2 counts only after position 1 was accepted)
    double mtp_s = 0;                                       // time in MTP steps (prompt + decode)
    std::string finish_reason;                              // "stop" | "length" | "cancelled"
    // test-time search (search_branches == 0: off). generated_tokens / tokens_per_s describe the returned (winning) answer.
    int search_branches = 0, search_valid = 0, search_pruned = 0, search_winner = 0;
    double search_consensus = 0;
    int search_total_tokens = 0, search_steps = 0;          // tokens of all branches; batched decode steps
    std::string search_detail;                              // JSON array: per branch tokens / score / mean logprob / finish / why
};

// Return false to cancel generation. `piece` is the UTF-8 text delta for this token (may be empty
// while a multi-byte character is incomplete).
using TokenCallback = std::function<bool(int token, const std::string& piece)>;

// Load-time knobs (lb_server / lb_gen flags). Defaults reproduce the historical behaviour.
struct EngineOptions {
    int threads = 0;                    // compute pool size (0 => hardware concurrency; LB_THREADS overrides when 0)
    int screen_head = 0;                // --screen-head 1: 4-bit screened greedy head for int8 heads (probabilistic margin, ~half the head bytes); 2: 2-bit screener (~quarter)
    float screen_sigma2 = 0.f;          // --screen-sigma2: margin of the 2-bit screener in sigmas (0 = Transformer default)
    bool int8_head = false;             // --int8-head: INT8 output head with int8 activations (f16 files: runtime copy, i8 files: in place)
    int exact_head = -1;                // --exact-head [0|1]: -1 = platform default (on for desktop f16 heads), 0 off, 1 on (no-op for int8 heads)
    int head_vocab = 0;                 // --head-vocab N: only ids < N (+ eos/special ids + --head-vocab-file ids) get logits, others -inf
    std::string head_vocab_file;        // --head-vocab-file PATH: token ids (one per line or comma/space separated)
    bool turbo = false;                 // --mode turbo: MoE expert pruning in decode (default strict = all experts)
    float moe_alpha = -1.f, moe_mass = -1.f;   // --moe-alpha / --moe-mass (-1 = unset)
    int moe_min_k = -1, moe_kmax = -1, moe_edge = -1, moe_edge_k = -1, moe_prefill_prune = -1;   // --moe-min-k / --moe-kmax / --moe-edge / --moe-edge-k / --moe-prefill-prune
    // CPU accelerators (all off by default; --pareto-max sets the locked preset, see apply_pareto_max)
    int moe_k = 0;                      // --moe-k N: top-N routed experts for every token (0 = all)
    std::string moe_k_layers;           // --moe-k-layers "8,6,..": per-layer budget (0 or missing entries fall back to moe_k)
    float teal_sparsity = 0.f;          // --teal-sparsity P: magnitude sparsity on the MoE down_proj input, calibrated at load on built-in public text
    int shadow_full = 0;                // --shadow-full R with --shadow-masks FILE: experimental shadow-expert quality emulation (Transformer::shadow_full)
    std::string shadow_masks;
    bool teal_report = false;
    int enable_thinking = 1;            // --thinking [0|1] / LB_THINKING: chat-template enable_thinking (0 = empty think block in the generation prompt; use for evals)
    int prefix_cache = 1;               // --prefix-cache [0|1]: keep the KV state of the last 2 prompts and reuse the longest common token prefix
    int max_context = 4096;             // --ctx N: positions to allocate (KV cache + RoPE table), never more than the model supports
};
// --pareto-max: the locked CPU preset, chosen from the measured sweep in benchmarks/results/pareto_sweep.json. A flag given explicitly wins.
// It trades a little quality for speed; without it nothing changes.
struct ParetoMaxPreset {
    static constexpr int moe_k = 6;            // top-6 of the routed experts
    static constexpr int screen_head = 2;      // 2-bit screened greedy head (probabilistic margin); same 160 outputs as the 4-bit one, +5% decode
    static constexpr int prefill_chunk = 512;  // larger prefill forwards
    static constexpr int lookup_n = 8;         // prompt-lookup drafts per verify pass (exact greedy verification)
};

// Options picked up by Engine::load (set by lb_server before loading; header-only so alternative Engine implementations need no change).
inline EngineOptions& engine_default_options() { static EngineOptions o; return o; }

struct BatchPoint { int batch = 0, steps = 0; double step_ms_median = 0, agg_tok_s = 0; };   // one batch size of Engine::batch_probe

class Engine {
public:
    virtual ~Engine() = default;
    // mmap's the file. Returns nullptr and fills *err on failure.
    static std::unique_ptr<Engine> load(const std::string& lbt_path, std::string* err);

    virtual const ModelConfig& config() const = 0;
    virtual std::string isa_name() const = 0;

    virtual std::vector<int> encode(const std::string& text, bool add_special = false) const = 0;
    virtual std::string decode(const std::vector<int>& ids) const = 0;
    virtual std::vector<int> apply_chat_template(const std::vector<ChatMessage>& msgs, bool add_generation_prompt = true) const = 0;

    // Single sequence at a time (the engine owns one KV-cache); not thread-safe.
    virtual GenStats generate(const std::vector<int>& prompt, const GenOptions& opt, const TokenCallback& cb) = 0;
    // Reset KV-cache / context state.
    virtual void reset() = 0;

    // Positions generate() can hold (KV capacity); prompt + new tokens must stay below it.
    virtual int context_limit() const { return config().max_position_embeddings; }
    virtual size_t kv_bytes() const { return 0; }   // allocated KV cache
    // True when this instance hands work to a GPU backend (VELTRA_CUDA build with a usable device).
    virtual bool gpu_active() const { return false; }
    // Measures weight-stationary batched greedy decode (Transformer::forward_batch): prefills one KV slice per prompt, then
    // times `steps` decode steps for every batch size in `batches` (<= prompts.size()). keep_going(batch; 0 = prefill) is
    // asked before every forward; returning false stops early with the finished points. False + *why when unsupported.
    virtual bool batch_probe(const std::vector<std::vector<int>>& prompts, const std::vector<int>& batches, int steps,
                             const std::function<bool(int batch)>& keep_going, std::vector<BatchPoint>* out, std::string* why) {
        (void)prompts; (void)batches; (void)steps; (void)keep_going; (void)out;
        if (why) *why = "batched decode is niet beschikbaar in deze engine";
        return false;
    }
};

}  // namespace lb
