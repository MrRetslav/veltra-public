#include "capi.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "binary_gemv.h"
#include "engine.h"
#include "lbt_model.h"
#include "selftest.h"
#include "threadpool.h"

namespace {

struct Handle {
    lb::LbtFile file;
    std::unique_ptr<lb::Engine> engine;  // null when the file holds only bare layers (tests)
    std::map<std::string, lb::LbLinear> layers;
    lb::LinearScratch scratch;
    std::map<int, std::unique_ptr<lb::ThreadPool>> pools;
    std::string config_json;
    std::string isa;
};

void set_err(char* err, int n, const std::string& m) {
    if (err && n > 0) std::snprintf(err, (size_t)n, "%s", m.c_str());
}

int copy_out(const std::string& s, char* out, int cap) {
    if (out && cap > 0) std::snprintf(out, (size_t)cap, "%s", s.c_str());
    return (int)s.size();
}

}  // namespace

#ifdef LB_HAVE_ENGINE
#include "json.h"
#endif

extern "C" {

LB_API lb_handle lb_open(const char* path, char* err, int err_len) {
    try {
        auto h = std::make_unique<Handle>();
        std::string e;
        if (!h->file.open(path, &e)) { set_err(err, err_len, e); return nullptr; }
        h->config_json = h->file.meta_json();
        h->isa = lb::isa_name(lb::best_isa());
#ifdef LB_HAVE_ENGINE
        // Full model files carry hidden_size in the meta; bare-layer files do not.
        if (h->config_json.find("\"hidden_size\"") != std::string::npos) {
            h->engine = lb::Engine::load(path, &e);
            if (!h->engine) { set_err(err, err_len, e); return nullptr; }
        }
#endif
        return h.release();
    } catch (const std::exception& ex) {
        set_err(err, err_len, ex.what());
        return nullptr;
    }
}

LB_API void lb_close(lb_handle h) { delete (Handle*)h; }
LB_API const char* lb_isa_name(lb_handle h) { return ((Handle*)h)->isa.c_str(); }
LB_API const char* lb_config_json(lb_handle h) { return ((Handle*)h)->config_json.c_str(); }

LB_API int lb_linear_dims(lb_handle hh, const char* name, int* m, int* n, int* rp, int* rr) {
    Handle* h = (Handle*)hh;
    try {
        auto it = h->layers.find(name);
        if (it == h->layers.end()) {
            lb::LbLinear l;
            std::string e;
            if (!lb::load_linear(h->file, name, &l, &e)) return -1;
            it = h->layers.emplace(name, std::move(l)).first;
        }
        const lb::LbLinear& l = it->second;
        if (m) *m = l.m;
        if (n) *n = l.n;
        if (rp) *rp = l.prim.r;
        if (rr) *rr = l.has_res() ? l.res.r : 0;
        return 0;
    } catch (...) { return -1; }
}

LB_API int lb_linear_forward(lb_handle hh, const char* name, const float* x, int T, float* y, int use_res, int isa) {
    Handle* h = (Handle*)hh;
    if (lb_linear_dims(hh, name, nullptr, nullptr, nullptr, nullptr) != 0) return -1;
    lb::Isa which = isa < 0 ? lb::best_isa() : (lb::Isa)isa;
    if (!lb::isa_available(which)) return -2;
    lb::linear_forward(h->layers.at(name), which, x, T, y, use_res != 0, h->scratch);
    return 0;
}

LB_API int lb_linear_forward_mt(lb_handle hh, const char* name, const float* x, int T, float* y, int use_res, int isa, int threads) {
    Handle* h = (Handle*)hh;
    if (lb_linear_dims(hh, name, nullptr, nullptr, nullptr, nullptr) != 0) return -1;
    lb::Isa which = isa < 0 ? lb::best_isa() : (lb::Isa)isa;
    if (!lb::isa_available(which)) return -2;
    lb::ThreadPool* pool = nullptr;
    if (threads > 1) {
        auto& pp = h->pools[threads];
        if (!pp) pp = std::make_unique<lb::ThreadPool>(threads);
        pool = pp.get();
    }
    lb::linear_forward(h->layers.at(name), which, x, T, y, use_res != 0, h->scratch, pool);
    return 0;
}

LB_API int lb_grid_info(lb_handle hh, const char* name, int* planes, int* group) {
    Handle* h = (Handle*)hh;
    if (lb_linear_dims(hh, name, nullptr, nullptr, nullptr, nullptr) != 0) return -1;
    const lb::LbLinear& l = h->layers.at(name);
    if (l.is_nibble()) { if (planes) *planes = 4; if (group) *group = 128; return 0; }   // nibble-INT4 (.n4) == P=4, G=128
    if (!l.is_grid()) return -3;
    if (planes) *planes = l.grid.P;
    if (group) *group = l.grid.G;
    return 0;
}

LB_API int lb_kernel_selftest(char* out, int cap) {
    std::string rep;
    int f = lb::run_kernel_selftest(rep);
    copy_out(rep, out, cap);
    return f;
}

LB_API int lb_isa_available(int isa) { return lb::isa_available((lb::Isa)isa) ? 1 : 0; }

#ifdef LB_HAVE_ENGINE
LB_API int lb_encode(lb_handle hh, const char* text, int add_special, int32_t* out, int cap) {
    Handle* h = (Handle*)hh;
    if (!h->engine) return -1;
    auto ids = h->engine->encode(text, add_special != 0);
    if (!out || cap < (int)ids.size()) return -(int)ids.size();
    for (size_t i = 0; i < ids.size(); i++) out[i] = ids[i];
    return (int)ids.size();
}

LB_API int lb_decode(lb_handle hh, const int32_t* ids, int n, char* out, int cap) {
    Handle* h = (Handle*)hh;
    if (!h->engine) return -1;
    return copy_out(h->engine->decode(std::vector<int>(ids, ids + n)), out, cap);
}

LB_API int lb_chat_tokens(lb_handle hh, const char* messages_json, int32_t* out, int cap) {
    Handle* h = (Handle*)hh;
    if (!h->engine) return -1;
    try {
        lb::json::Value v;
        if (!lb::json::parse(messages_json, &v)) return -1;
        std::vector<lb::ChatMessage> msgs;
        for (auto& m : v.as_array()) msgs.push_back({m["role"].as_string(), m["content"].as_string()});
        auto ids = h->engine->apply_chat_template(msgs, true);
        if (!out || cap < (int)ids.size()) return -(int)ids.size();
        for (size_t i = 0; i < ids.size(); i++) out[i] = ids[i];
        return (int)ids.size();
    } catch (...) { return -1; }
}

LB_API int lb_generate(lb_handle hh, const int32_t* prompt, int n_prompt, const char* opts_json, lb_token_cb cb,
                       void* user, char* stats_json, int stats_cap) {
    Handle* h = (Handle*)hh;
    if (!h->engine) return -1;
    try {
        lb::GenOptions o;
        if (opts_json && *opts_json) {
            lb::json::Value v;
            if (!lb::json::parse(opts_json, &v)) return -1;
            auto num = [&](const char* k, double d) { return v.find(k) ? (*v.find(k)).as_double(d) : d; };
            auto flag = [&](const char* k, bool d) { return v.find(k) ? (*v.find(k)).as_bool(d) : d; };
            o.max_new_tokens = (int)num("max_new_tokens", 256);
            o.sampling.temperature = (float)num("temperature", 0);
            o.sampling.top_p = (float)num("top_p", 1);
            o.sampling.top_k = (int)num("top_k", 0);
            o.sampling.seed = (uint64_t)num("seed", 0);
            o.spec.enabled = flag("speculative", true);
            o.spec.ngram = flag("ngram", true);
            o.spec.layer_skip = flag("layer_skip", true);
            o.spec.k_min = (int)num("k_min", 2);
            o.spec.k_max = (int)num("k_max", 8);
            o.n_threads = (int)num("n_threads", 0);
        }
        std::vector<int> p(prompt, prompt + n_prompt);
        auto st = h->engine->generate(p, o, [&](int tok, const std::string& piece) {
            return cb ? cb(tok, piece.c_str(), user) == 0 : true;
        });
        if (stats_json && stats_cap > 0) {
            char b[512];
            std::snprintf(b, sizeof b,
                          "{\"prompt_tokens\":%d,\"generated_tokens\":%d,\"prefill_s\":%.4f,\"decode_s\":%.4f,\"tokens_per_s\":%.3f,"
                          "\"drafted\":%d,\"accepted\":%d,\"verify_passes\":%d,\"ngram_drafted\":%d,\"ngram_accepted\":%d,\"finish_reason\":\"%s\"}",
                          st.prompt_tokens, st.generated_tokens, st.prefill_s, st.decode_s, st.tokens_per_s, st.drafted, st.accepted,
                          st.verify_passes, st.ngram_drafted, st.ngram_accepted, st.finish_reason.c_str());
            std::snprintf(stats_json, (size_t)stats_cap, "%s", b);
        }
        return st.generated_tokens;
    } catch (...) { return -2; }
}

LB_API void lb_reset(lb_handle hh) {
    Handle* h = (Handle*)hh;
    if (h->engine) h->engine->reset();
}
#endif

}  // extern "C"
