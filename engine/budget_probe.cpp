#include "budget_probe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>

#include "lb_common.h"

namespace lb {
namespace {

std::vector<int> read_ids(const std::string& path) {
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string s = ss.str();
    std::vector<int> ids;
    for (size_t i = 0; i < s.size();) {
        if (s[i] < '0' || s[i] > '9') { i++; continue; }
        long v = 0;
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') v = std::min(v * 10 + (s[i++] - '0'), 100000000L);
        ids.push_back((int)v);
    }
    return ids;
}

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string x;
    while (std::getline(ss, x, ','))
        if (!x.empty()) out.push_back(x);
    return out;
}

struct Set {
    std::vector<std::vector<int>> segments;
    std::vector<std::vector<float>> ref;   // per segment: log-softmax rows [T][V] of the full model
    long positions = 0;
};

struct Score {
    double kl = 0, top1 = 0, nll = 0;   // mean over positions: KL(full || variant), top-1 agreement with the full model, NLL of the real next token
};

void log_softmax(float* row, int V) {
    float mx = row[0];
    for (int i = 1; i < V; i++) mx = std::max(mx, row[i]);
    double z = 0;
    for (int i = 0; i < V; i++) z += std::exp((double)(row[i] - mx));
    const float lz = (float)std::log(z) + mx;
    for (int i = 0; i < V; i++) row[i] -= lz;
}

// Forward one segment from position 0 in chunks of 64 (as the decoder's prefill does) and hand every row of logits to fn(position, row).
template <class Fn>
void forward_rows(Transformer& m, const ModelConfig& cfg, const std::vector<int>& ids, std::vector<float>& buf, Fn&& fn) {
    const int V = cfg.vocab_size, T = (int)ids.size();
    m.kv().len = 0;
    for (int i = 0; i < T; i += 64) {
        const int c = std::min(64, T - i);
        buf.resize((size_t)c * V);
        ForwardOpts fo;
        fo.logits_from = 0;
        m.forward(ids.data() + i, c, i, fo, buf.data());
        for (int t = 0; t < c; t++) fn(i + t, buf.data() + (size_t)t * V);
    }
}

Score measure(Transformer& m, const ModelConfig& cfg, const Set& set, std::vector<float>& buf) {
    const int V = cfg.vocab_size;
    Score sc;
    long n = 0, nn = 0;
    for (size_t s = 0; s < set.segments.size(); s++) {
        const std::vector<int>& ids = set.segments[s];
        forward_rows(m, cfg, ids, buf, [&](int t, float* row) {
            log_softmax(row, V);
            const float* ref = set.ref[s].data() + (size_t)t * V;
            double kl = 0;
            int a = 0, b = 0;
            for (int i = 0; i < V; i++) {
                if (ref[i] > -30.f) kl += std::exp((double)ref[i]) * (double)(ref[i] - row[i]);
                if (ref[i] > ref[a]) a = i;
                if (row[i] > row[b]) b = i;
            }
            sc.kl += kl;
            sc.top1 += a == b;
            n++;
            if (t + 1 < (int)ids.size()) { sc.nll -= row[ids[t + 1]]; nn++; }
        });
    }
    sc.kl /= std::max(1L, n);
    sc.top1 /= std::max(1L, n);
    sc.nll /= std::max(1L, nn);
    return sc;
}

// Loads every id file of the comma separated list and stores the full model's log-probabilities for it.
bool load_set(Transformer& m, const ModelConfig& cfg, const std::string& files, std::vector<float>& buf, Set& set) {
    const int V = cfg.vocab_size;
    for (const std::string& f : split(files)) {
        std::vector<int> ids = read_ids(f);
        if (ids.size() < 8 || (int)ids.size() > m.kv().max_seq - 2) { std::fprintf(stderr, "bad segment %s (%zu tokens)\n", f.c_str(), ids.size()); return false; }
        set.segments.push_back(ids);
        set.positions += (long)ids.size();
    }
    if (set.segments.empty()) { std::fprintf(stderr, "need --budget-fit / --budget-held id files\n"); return false; }
    for (const auto& ids : set.segments) {
        set.ref.emplace_back((size_t)ids.size() * V);
        std::vector<float>& ref = set.ref.back();
        forward_rows(m, cfg, ids, buf, [&](int t, float* row) { log_softmax(row, V); std::copy(row, row + V, ref.begin() + (size_t)t * V); });
    }
    return true;
}

}  // namespace

int run_shadow_probe(Transformer& model, const ModelConfig& cfg, const std::map<std::string, std::string>& args) {
    auto get = [&](const char* k, const char* d) { auto it = args.find(k); return it == args.end() ? std::string(d) : it->second; };
    if (cfg.num_experts <= 0) { std::fprintf(stderr, "shadow probe needs a MoE model\n"); return 2; }
    Set held;
    std::vector<float> buf;
    model.shadow_full = 0;
    model.moe_k = 0;
    model.moe_k_layer.clear();
    if (!load_set(model, cfg, get("--budget-held", ""), buf, held)) return 2;
    std::string out = "{\"held_positions\":" + std::to_string(held.positions) + ",\"variants\":[";
    bool first = true;
    for (const std::string& v : split(get("--shadow-variants", ""))) {
        model.shadow_full = 0;
        model.moe_k = 0;
        if (v.rfind("k:", 0) == 0) model.moe_k = std::atoi(v.c_str() + 2);
        else if (v.rfind("s:", 0) == 0 && v.find(':', 2) != std::string::npos) {
            const size_t c = v.find(':', 2);
            if (!model.load_shadow_masks(v.substr(c + 1))) { std::fprintf(stderr, "bad mask file in variant %s\n", v.c_str()); return 2; }
            model.shadow_full = std::atoi(v.c_str() + 2);
        } else if (v != "full") { std::fprintf(stderr, "bad variant %s\n", v.c_str()); return 2; }
        const Score s = measure(model, cfg, held, buf);
        char num[160];
        std::snprintf(num, sizeof num, "\",\"kl\":%.5f,\"top1\":%.4f,\"nll\":%.4f}", s.kl, s.top1, s.nll);
        out += std::string(first ? "" : ",") + "{\"variant\":\"" + v.substr(0, v.find('/') == std::string::npos ? v.size() : v.find('/')) +
               v.substr(v.find_last_of('/') == std::string::npos ? v.size() : v.find_last_of('/') + 1) + num;
        first = false;
        std::printf("%-60s held-out KL %.4f top-1 %.2f%% NLL %.4f\n", v.c_str(), s.kl, 100 * s.top1, s.nll);
        std::fflush(stdout);
    }
    model.shadow_full = 0;
    model.moe_k = 0;
    out += "]}";
    const std::string path = get("--budget-out", "shadow_probe.json");
    std::ofstream(path) << out;
    std::printf("wrote %s\n", path.c_str());
    return 0;
}

int run_budget_probe(Transformer& model, const ModelConfig& cfg, const std::map<std::string, std::string>& args) {
    auto get = [&](const char* k, const char* d) { auto it = args.find(k); return it == args.end() ? std::string(d) : it->second; };
    if (cfg.num_experts <= 0) { std::fprintf(stderr, "budget probe needs a MoE model\n"); return 2; }
    const int L = cfg.num_layers, K = cfg.num_experts_per_tok, V = cfg.vocab_size;
    Set fit, held;
    std::vector<float> buf;
    for (Set* set : {&fit, &held}) {
        for (const std::string& f : split(get(set == &fit ? "--budget-fit" : "--budget-held", ""))) {
            std::vector<int> ids = read_ids(f);
            if (ids.size() < 8 || (int)ids.size() > model.kv().max_seq - 2) { std::fprintf(stderr, "bad segment %s (%zu tokens)\n", f.c_str(), ids.size()); return 2; }
            set->segments.push_back(ids);
            set->positions += (long)ids.size();
        }
        if (set->segments.empty()) { std::fprintf(stderr, "need --budget-fit and --budget-held id files\n"); return 2; }
        model.moe_k_layer.assign(L, 0);
        for (const auto& ids : set->segments) {
            set->ref.emplace_back((size_t)ids.size() * V);
            std::vector<float>& ref = set->ref.back();
            forward_rows(model, cfg, ids, buf, [&](int t, float* row) { log_softmax(row, V); std::copy(row, row + V, ref.begin() + (size_t)t * V); });
        }
    }
    std::fprintf(stderr, "budget probe: %ld fit and %ld held-out positions, %d layers, top-%d\n", fit.positions, held.positions, L, K);
    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };

    // sanity: the unchanged configuration must reproduce the reference exactly
    const Score zero = measure(model, cfg, fit, buf);
    if (zero.kl > 1e-9 || zero.top1 < 1.0) { std::fprintf(stderr, "reference is not reproducible (kl %g)\n", zero.kl); return 1; }

    std::vector<int> ks;
    for (const std::string& s : split(get("--budget-ks", "6,4"))) ks.push_back(std::atoi(s.c_str()));
    std::sort(ks.rbegin(), ks.rend());
    std::vector<std::vector<double>> sens(ks.size(), std::vector<double>(L, 0.0));
    for (int l = 0; l < L; l++) {
        for (size_t j = 0; j < ks.size(); j++) {
            model.moe_k_layer.assign(L, 0);
            model.moe_k_layer[l] = ks[j];
            sens[j][l] = measure(model, cfg, fit, buf).kl;
        }
        if ((l + 1) % 8 == 0) std::fprintf(stderr, "sensitivity: %d/%d layers (%.0f s)\n", l + 1, L, elapsed());
    }

    // Greedy allocation: every step lowers one layer to its next smaller k, choosing the best saved expert reads per added KL (sensitivities added
    // up, an assumption that the combined measurement below checks).
    auto allocate = [&](int saved_target) {
        std::vector<int> level(L, -1), kk(L, K);
        int saved = 0;
        while (saved < saved_target) {
            int best = -1;
            double best_ratio = -1;
            for (int l = 0; l < L; l++) {
                const int j = level[l] + 1;
                if (j >= (int)ks.size()) continue;
                const double add = std::max(1e-7, sens[j][l] - (level[l] >= 0 ? sens[level[l]][l] : 0.0));
                const double ratio = (kk[l] - ks[j]) / add;
                if (ratio > best_ratio) { best_ratio = ratio; best = l; }
            }
            if (best < 0) break;
            const int j = ++level[best];
            saved += kk[best] - ks[j];
            kk[best] = ks[j];
        }
        return kk;
    };

    std::string out = "{\"layers\":" + std::to_string(L) + ",\"top_k\":" + std::to_string(K) + ",\"fit_positions\":" + std::to_string(fit.positions) +
                      ",\"held_positions\":" + std::to_string(held.positions) + ",\"sensitivity_kl\":{";
    char num[96];
    for (size_t j = 0; j < ks.size(); j++) {
        out += std::string(j ? "," : "") + "\"" + std::to_string(ks[j]) + "\":[";
        for (int l = 0; l < L; l++) { std::snprintf(num, sizeof num, "%s%.6g", l ? "," : "", sens[j][l]); out += num; }
        out += "]";
    }
    out += "},\"comparisons\":[";
    bool first = true;
    for (int uniform = K - 1; uniform >= std::max(1, K - 3); uniform--) {   // same number of expert reads: uniform k = 7, 6, 5 vs the allocation
        const int target = (K - uniform) * L;
        const std::vector<int> alloc = allocate(target);
        for (int variant = 0; variant < 2; variant++) {
            if (variant == 0) model.moe_k_layer.assign(L, uniform);
            else model.moe_k_layer = alloc;
            int reads = 0;
            for (int l = 0; l < L; l++) reads += model.moe_k_layer[l] > 0 ? model.moe_k_layer[l] : K;
            const Score a = measure(model, cfg, fit, buf), b = measure(model, cfg, held, buf);
            std::string list;
            for (int l = 0; l < L; l++) list += (l ? "," : "") + std::to_string(model.moe_k_layer[l]);
            std::snprintf(num, sizeof num, "%s{\"mode\":\"%s\",\"uniform_equivalent\":%d,\"expert_reads\":%d,", first ? "" : ",", variant ? "budget" : "uniform", uniform, reads);
            first = false;
            out += num;
            std::snprintf(num, sizeof num, "\"fit\":{\"kl\":%.5f,\"top1\":%.4f,\"nll\":%.4f},", a.kl, a.top1, a.nll);
            out += num;
            std::snprintf(num, sizeof num, "\"held\":{\"kl\":%.5f,\"top1\":%.4f,\"nll\":%.4f},", b.kl, b.top1, b.nll);
            out += num;
            out += "\"k_per_layer\":\"" + list + "\"}";
            std::printf("%-8s reads %3d (uniform-%d equivalent): held-out KL %.4f top-1 %.2f%% NLL %.4f | fit KL %.4f\n", variant ? "budget" : "uniform", reads,
                        uniform, b.kl, 100 * b.top1, b.nll, a.kl);
            std::fflush(stdout);
        }
    }
    model.moe_k_layer.assign(L, 0);
    const Score ref_held = measure(model, cfg, held, buf);
    std::snprintf(num, sizeof num, "],\"full_model_held_nll\":%.4f,\"seconds\":%.0f}", ref_held.nll, elapsed());
    out += num;
    const std::string path = get("--budget-out", "budget_probe.json");
    std::ofstream(path) << out;
    std::printf("full model held-out NLL %.4f; wrote %s (%.0f s)\n", ref_held.nll, path.c_str(), elapsed());
    return 0;
}

}  // namespace lb
