// Engine implementation: .lbt model + embedded (or sidecar) tokenizer.json + built-in chat template,
// generation through SpecDecoder (plain decode when spec is disabled).
#include "engine.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include <fstream>
#include <sstream>

#include "binary_gemv.h"
#include "chat_template.h"
#include "lbt_model.h"
#include "model_config.h"
#include "search.h"
#include "speculative.h"
#include "tokenizer.h"
#include "teal_text.h"
#include "transformer.h"
#include "lb_common.h"

namespace lb {
namespace {

bool read_file(const std::string& path, std::string* out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::ostringstream ss;
    ss << in.rdbuf();
    *out = ss.str();
    return true;
}

std::string blob(const LbtFile& f, const char* name) {
    const TensorRef* t = f.find(name);
    if (!t || !t->data) return {};
    return std::string((const char*)t->data, (size_t)t->nbytes);
}

class EngineImpl final : public Engine {
public:
    EngineImpl(std::unique_ptr<LbtFile> file, ModelConfig cfg, std::unique_ptr<Tokenizer> tok, const EngineOptions& opt)
        : file_(std::move(file)), cfg_(std::move(cfg)), tok_(std::move(tok)) {
        isa_ = default_isa_for(*file_);
        const int want_ctx = std::max(512, std::min(opt.max_context, 32768));
        const int max_ctx = cfg_.max_position_embeddings > 0 ? std::min(cfg_.max_position_embeddings, want_ctx) : std::min(want_ctx, 2048);
        int threads = opt.threads;
        if (threads <= 0) if (const char* e = lb::env_get("THREADS")) threads = std::atoi(e);
        model_ = std::make_unique<Transformer>(*file_, cfg_, max_ctx, KvMode::Int8, threads, isa_);
        {  // lossless greedy head (+1 byte/weight of RAM): default on desktop, opt-in on Android (LB_EXACT_HEAD=1/0 overrides)
            const char* e = lb::env_get("EXACT_HEAD");
#if defined(__ANDROID__)
            bool on = e && e[0] == '1';
#else
            bool on = !(e && e[0] == '0');
#endif
            if (opt.exact_head >= 0) on = opt.exact_head == 1;
            if (opt.head_vocab > 0 || !opt.head_vocab_file.empty()) on = false;   // a restricted head is not "all ids"
            if (on) model_->enable_exact_head();
        }
        thinking_ = opt.enable_thinking != 0;
        if (const char* e = lb::env_get("THINKING")) thinking_ = e[0] != '0';
        prefix_cache_ = opt.prefix_cache != 0;
        if (const char* e = lb::env_get("PREFIX_CACHE")) prefix_cache_ = e[0] != '0';
        if (cfg_.hybrid()) prefix_cache_ = false;   // the DeltaNet state has no per-position snapshot: every request prefills from position 0
        model_->moe_turbo.apply(opt.turbo, opt.moe_alpha, opt.moe_min_k, opt.moe_kmax, opt.moe_mass, opt.moe_edge, opt.moe_edge_k, opt.moe_prefill_prune);
        model_->moe_k = std::max(0, opt.moe_k);
        for (size_t p = 0; p < opt.moe_k_layers.size();) {
            model_->moe_k_layer.push_back(std::max(0, std::atoi(opt.moe_k_layers.c_str() + p)));
            const size_t c = opt.moe_k_layers.find(',', p);
            p = c == std::string::npos ? opt.moe_k_layers.size() : c + 1;
        }
        if (opt.shadow_full > 0) {   // an unusable mask file must not silently run the full model: the numbers would be attributed to the wrong setup
            if (!model_->load_shadow_masks(opt.shadow_masks)) throw std::runtime_error("--shadow-masks: cannot load a mask file that matches this model");
            model_->shadow_full = opt.shadow_full;
        }
        if (opt.teal_sparsity > 0.f && cfg_.num_experts > 0) teal_calibrate(*model_, *tok_, std::min(opt.teal_sparsity, 0.9f), max_ctx);
        if (opt.int8_head) model_->enable_int8_head();
        if (opt.screen_sigma2 > 0.f) model_->screen_sigma2 = opt.screen_sigma2;
        if (opt.screen_head) model_->enable_screen_head(opt.screen_head == 2 ? 2 : 4);
        if (opt.head_vocab > 0 || !opt.head_vocab_file.empty()) {
            std::vector<int> extra;
            std::string e;
            if (!opt.head_vocab_file.empty() && !load_id_list_file(opt.head_vocab_file, &extra, &e)) throw std::runtime_error(e);
            model_->set_head_vocab(head_vocab_ids(cfg_, opt.head_vocab, extra));
        }
    }

    const ModelConfig& config() const override { return cfg_; }
    std::string isa_name() const override { return lb::isa_name(isa_); }

    std::vector<int> encode(const std::string& text, bool add_special) const override {
        return tok_->encode(text, add_special);
    }
    std::string decode(const std::vector<int>& ids) const override { return tok_->decode(ids, false); }

    std::vector<int> apply_chat_template(const std::vector<ChatMessage>& msgs, bool add_gen) const override {
        // Qwen3 thinking templates mention enable_thinking; the Instruct-2507 template (no think block) does not -> plain ChatML
        const bool plain = cfg_.arch == "qwen3" && !cfg_.chat_template.empty() && cfg_.chat_template.find("enable_thinking") == std::string::npos;
        return lb::apply_chat_template(*tok_, plain ? "chatml" : (cfg_.hybrid() && cfg_.arch == "qwen3" ? "qwen35" : cfg_.arch), msgs, add_gen, thinking_);
    }

    GenStats generate(const std::vector<int>& prompt, const GenOptions& opt, const TokenCallback& cb) override {
        struct TealStats { Transformer& m; ~TealStats() { if (m.teal_active() && lb::env_get("TEAL_STATS")) std::fprintf(stderr, "[veltra] %s\n", m.teal_report().c_str()); } } teal_stats{*model_};
        Detok detok(*tok_, true);
        bool cancelled = false;
        const int reuse = prefix_cache_ ? select_slot(prompt) : 0;
        if (!prefix_cache_) { slots_[0].tokens.clear(); slots_[1].tokens.clear(); }
        if (opt.search.n > 1) return generate_search(prompt, opt, cb, reuse);
        SpecDecoder dec(*model_);
        GenStats st;
        try {
            st = dec.run(prompt, opt, cfg_.eos_token_ids, [&](int id) {
                if (!cb) return true;
                if (!cb(id, detok.push(id))) { cancelled = true; return false; }
                return true;
            }, reuse);
        } catch (...) {
            slots_[active_].tokens.clear();
            throw;
        }
        slots_[active_].tokens = dec.committed();
        slots_[active_].stamp = ++clock_;
        st.cached_tokens = reuse;
        if (cancelled) st.finish_reason = "cancelled";
        if (model_->moe_turbo.active() && model_->moe_stat_tokens() > 0)
            std::fprintf(stderr, "[moe-turbo] avg experts/token-layer %.3f over %ld decode tokens*layers\n", model_->moe_avg_experts(), model_->moe_stat_tokens());
        return st;
    }

    // generate() always starts from the prompt, so a reset only has to drop the cached prefixes when the cache is off
    // (the prefix cache stays valid: it is keyed by the exact token ids of the previous prompts).
    void reset() override {
        model_->kv().len = 0;
        if (!prefix_cache_) { slots_[0].tokens.clear(); slots_[1].tokens.clear(); }
    }

    int context_limit() const override { return model_->kv().max_seq; }
    size_t kv_bytes() const override { return model_->kv().bytes(); }
    bool gpu_active() const override { return model_->gpu_active(); }

    bool batch_probe(const std::vector<std::vector<int>>& prompts, const std::vector<int>& batches, int steps,
                     const std::function<bool(int)>& keep_going, std::vector<BatchPoint>* out, std::string* why) override {
        auto unsupported = [&](const std::string& m) { if (why) *why = m; return false; };
        if (cfg_.hybrid()) return unsupported("batched decode ondersteunt geen hybride (DeltaNet) modellen");
        const int n = (int)prompts.size();
        int longest = 0;
        for (const auto& p : prompts) { if (p.size() < 2) return unsupported("prompt te kort voor de batchtest"); longest = std::max<int>(longest, (int)p.size()); }
        const int need = longest + steps + 8;
        if (n <= 0 || steps <= 0 || need > model_->kv().max_seq) return unsupported("batchtest past niet in de contextlimiet");
        struct Slices { Transformer& m; ~Slices() { m.clear_kv_slices(); } } guard{*model_};
        std::vector<int> slice(n);
        for (int i = 0; i < n; i++) slice[i] = model_->add_kv_slice(need);
        std::vector<BatchItem> it;
        for (int i = 0; i < n; i++) {   // every stream alone, chunks of 64 (as lb_gen --batch)
            const auto& p = prompts[i];
            const int P = (int)p.size();
            for (int c0 = 0; c0 < P - 1; c0 += 64) {
                if (!keep_going(0)) return true;
                const int c = std::min(64, P - 1 - c0);
                it.assign(c, BatchItem());
                for (int j = 0; j < c; j++) it[j] = {p[c0 + j], c0 + j, slice[i]};
                BatchOpts bo; bo.logits_from = c;
                model_->forward_batch(it.data(), c, bo, nullptr);
            }
        }
        const int V = cfg_.vocab_size;
        std::vector<float> logits;
        for (int b : batches) {
            if (b < 1 || b > n) continue;
            logits.resize((size_t)b * V);
            std::vector<int> cur(b), pos(b);
            for (int j = 0; j < b; j++) { cur[j] = prompts[j].back(); pos[j] = (int)prompts[j].size() - 1; }
            std::vector<double> ms;
            it.assign(b, BatchItem());
            BatchOpts bo; bo.greedy = true;
            for (int s = -1; s < steps; s++) {   // step -1 is an untimed warm-up
                if (!keep_going(b)) return true;
                for (int j = 0; j < b; j++) it[j] = {cur[j], pos[j], slice[j]};
                const auto t0 = std::chrono::steady_clock::now();
                model_->forward_batch(it.data(), b, bo, logits.data());
                const double dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (s >= 0) ms.push_back(dt);
                for (int j = 0; j < b; j++) {
                    const float* row = logits.data() + (size_t)j * V;
                    cur[j] = (int)(std::max_element(row, row + V) - row);
                    pos[j]++;
                }
            }
            std::sort(ms.begin(), ms.end());
            const double med = ms[ms.size() / 2];
            out->push_back(BatchPoint{b, steps, med, med > 0 ? b * 1000.0 / med : 0.0});
        }
        return true;
    }

private:
    // Test-time search: all branches finish first, then the winning answer is delivered through the same token callback.
    GenStats generate_search(const std::vector<int>& prompt, const GenOptions& opt, const TokenCallback& cb, int reuse) {
        SearchRun run;
        try {
            run = run_search(*model_, *tok_, prompt, opt, cfg_.eos_token_ids, reuse);
        } catch (...) {
            slots_[active_].tokens.clear();
            throw;
        }
        slots_[active_].tokens = run.committed;
        slots_[active_].stamp = ++clock_;
        if (!run.cancelled && cb) {
            Detok detok(*tok_, true);
            for (int id : run.branches[run.verdict.winner].ids)
                if (!cb(id, detok.push(id))) break;
        }
        return run.stats;
    }

    // Prefix KV cache: two KV caches (the model's own + `spare_`); slots_[i] says which tokens the cache of slot i holds valid K/V for.
    // The model always works on slot `active_`; choosing the other slot swaps the two caches (O(1) pointer swaps).
    struct Slot { std::vector<int> tokens; uint64_t stamp = 0; };
    // Returns the number of prompt tokens (<= P-1) whose K/V is already in the (now active) cache.
    int select_slot(const std::vector<int>& prompt) {
        int L[2];
        for (int i = 0; i < 2; i++) {
            const auto& t = slots_[i].tokens;
            const size_t n = std::min(t.size(), prompt.size() > 0 ? prompt.size() - 1 : 0);
            size_t k = 0;
            while (k < n && t[k] == prompt[k]) k++;
            L[i] = (int)k;
        }
        int pick = active_;
        if (L[1 - active_] > L[active_]) pick = 1 - active_;
        else if (L[active_] == 0 && slots_[1 - active_].stamp < slots_[active_].stamp) pick = 1 - active_;   // nothing to reuse: overwrite the LRU slot
        if (pick != active_) {
            if (spare_.k8.empty() && spare_.kf.empty()) {
                const KvCache& k = model_->kv();
                spare_.nb = k.nb;
                spare_.init(k.mode, k.layers, k.kv_heads, k.head_dim, k.max_seq);
            }
            std::swap(model_->kv(), spare_);
            active_ = pick;
        }
        if (lb::env_get("DEBUG_PC")) std::fprintf(stderr, "[pc] L=%d,%d pick=%d active=%d tok0=%zu tok1=%zu P=%zu\n", L[0], L[1], pick, active_, slots_[0].tokens.size(), slots_[1].tokens.size(), prompt.size());
        return L[pick];
    }

    KvCache spare_;
    Slot slots_[2];
    int active_ = 0;
    uint64_t clock_ = 0;
    bool prefix_cache_ = true;
    bool thinking_ = true;
    std::unique_ptr<LbtFile> file_;   // must outlive model_ (Transformer keeps a reference)
    ModelConfig cfg_;
    std::unique_ptr<Tokenizer> tok_;
    Isa isa_ = Isa::Scalar;
    std::unique_ptr<Transformer> model_;
};

}  // namespace

std::unique_ptr<Engine> Engine::load(const std::string& lbt_path, std::string* err) {
    const EngineOptions opts = engine_default_options();
    auto file = std::make_unique<LbtFile>();
    std::string e;
    if (!file->open(lbt_path, &e)) { if (err) *err = "open: " + e; return nullptr; }
    ModelConfig cfg;
    if (!parse_model_config(*file, &cfg, &e)) { if (err) *err = "config: " + e; return nullptr; }

    // tokenizer: embedded blob first, else sidecar "<model>.tokenizer.json" (+ "<model>.tokenizer_config.json")
    std::string tj = blob(*file, "tokenizer.json"), tc = blob(*file, "tokenizer_config.json");
    if (tj.empty()) {
        read_file(lbt_path + ".tokenizer.json", &tj);
        if (tc.empty()) read_file(lbt_path + ".tokenizer_config.json", &tc);
    }
    if (tj.empty()) { if (err) *err = "no tokenizer.json embedded or as sidecar"; return nullptr; }
    std::unique_ptr<Tokenizer> tok = Tokenizer::from_json(tj, tc, &e);
    if (!tok) { if (err) *err = "tokenizer: " + e; return nullptr; }
    if (cfg.eos_token_ids.empty() && tok->eos_token_id() >= 0) cfg.eos_token_ids.push_back(tok->eos_token_id());
    try {
        return std::make_unique<EngineImpl>(std::move(file), std::move(cfg), std::move(tok), opts);
    } catch (const std::exception& ex) {
        if (err) *err = ex.what();
        return nullptr;
    }
}

}  // namespace lb
