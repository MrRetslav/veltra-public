#include "batch_gen.h"
#include "lb_common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>

namespace lb {

namespace {
using Clk = std::chrono::steady_clock;

std::vector<int> read_ids(const std::string& path) {
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    std::string s = ss.str();
    std::vector<int> ids;
    size_t i = 0;
    while (i < s.size()) {
        if (s[i] >= '0' && s[i] <= '9') { long v = 0; while (i < s.size() && s[i] >= '0' && s[i] <= '9') v = std::min(v * 10 + (s[i++] - '0'), 100000000L); ids.push_back((int)v); }
        else i++;
    }
    return ids;
}

int argmax_row(const float* v, int n) {
    int b = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[b]) b = i;
    return b;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

std::string proc_status(const char* key) {
    std::ifstream f("/proc/self/status");
    std::string ln;
    const std::string k = std::string(key) + ":";
    while (std::getline(f, ln)) if (ln.rfind(k, 0) == 0) return ln.substr(k.size());
    return "0";
}

std::string load1() {
    std::ifstream f("/proc/loadavg");
    std::string a;
    f >> a;
    return a.empty() ? "?" : a;
}
}  // namespace

int run_batch_gen(Transformer& model, const ModelConfig& cfg, const std::map<std::string, std::string>& a) {
    auto get = [&](const char* k, const char* d) { auto it = a.find(k); return it == a.end() ? std::string(d) : it->second; };
    std::vector<int> sweep;   // --sweep 1,2,4,6,16 [--reps R]: prefill max(B) streams once, then interleave decode runs over the first b streams (A/B under a noisy box)
    {
        const std::string s = get("--sweep", "");
        size_t p = 0;
        while (p < s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); if (q > p) sweep.push_back(std::atoi(s.substr(p, q - p).c_str())); p = q + 1; }
    }
    const int reps = std::max(1, std::atoi(get("--reps", "3").c_str()));
    const int B = sweep.empty() ? std::max(1, std::atoi(get("--batch", "1").c_str())) : *std::max_element(sweep.begin(), sweep.end());
    const int N = std::max(1, std::atoi(get("--n", "64").c_str()));
    const int Lp = std::max(2, std::atoi(get("--prompt-len", "480").c_str()));
    const int fastv = std::atoi(get("--batch-fast", "-1").c_str());
    const bool verify = get("--verify", "0") == "1", print_tokens = get("--print-tokens", "0") == "1";
    const bool screen = get("--screen-head", "0") != "0";
    if (a.count("--head-mode")) model.head_mode = std::atoi(get("--head-mode", "-1").c_str());   // 1 = two-plane s8 head (integer-exact per token, ~2x faster at B>=6; same mode must be used for the B=1 reference)
    const int V = cfg.vocab_size;
    std::vector<std::vector<int>> files;
    {
        std::string s = get("--prompt-files", "");
        size_t p = 0;
        while (p < s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); if (q > p) files.push_back(read_ids(s.substr(p, q - p))); p = q + 1; }
    }
    if (files.empty()) { std::fprintf(stderr, "veltra-gen --batch needs --prompt-files a.ids,b.ids,...\n"); return 2; }
    const int F = (int)files.size();
    std::vector<std::vector<int>> prompts(B);
    for (int i = 0; i < B; i++) {   // stream i: file i % F, variant i / F starts 40 tokens later (distinct contexts, fixed regardless of B)
        const auto& f = files[i % F];
        const size_t off = std::min<size_t>((size_t)(i / F) * 40, f.size() > 8 ? f.size() - 8 : 0);
        prompts[i].assign(f.begin() + off, f.begin() + std::min(f.size(), off + (size_t)Lp));
        if (prompts[i].size() < 2) { std::fprintf(stderr, "prompt %d too short\n", i); return 2; }
    }
    const int maxP = [&] { int m = 0; for (auto& p : prompts) m = std::max<int>(m, (int)p.size()); return m; }();
    std::vector<int> slice(B);
    for (int i = 0; i < B; i++) slice[i] = model.add_kv_slice(maxP + N + 8);

    // ---- prefill (each stream alone, chunks of 64 like SpecDecoder): identical for every B
    const auto tp0 = Clk::now();
    for (int i = 0; i < B; i++) {
        const auto& p = prompts[i];
        const int P = (int)p.size();
        std::vector<BatchItem> it;
        for (int c0 = 0; c0 < P - 1; c0 += 64) {
            const int c = std::min(64, P - 1 - c0);
            it.assign(c, BatchItem());
            for (int j = 0; j < c; j++) it[j] = {p[c0 + j], c0 + j, slice[i]};
            BatchOpts bo; bo.logits_from = c;
            model.forward_batch(it.data(), c, bo, nullptr);
        }
    }
    const double prefill_s = std::chrono::duration<double>(Clk::now() - tp0).count();

    std::vector<float> logits((size_t)std::max(B, 6) * V);
    auto decode = [&](const std::vector<int>& which, bool stats, std::vector<std::vector<int>>& out, std::vector<double>& step_ms) {
        const int nb = (int)which.size();
        std::vector<int> cur(nb), pos(nb);
        for (int j = 0; j < nb; j++) { cur[j] = prompts[which[j]].back(); pos[j] = (int)prompts[which[j]].size() - 1; }
        out.assign(nb, {});
        step_ms.clear();
        std::vector<BatchItem> it(nb);
        BatchOpts bo; bo.greedy = true; bo.fast = fastv; bo.screen_ok = screen && nb == 1; bo.count_experts = stats;
        for (int s = 0; s < N; s++) {
            for (int j = 0; j < nb; j++) it[j] = {cur[j], pos[j], slice[which[j]]};
            const auto t0 = Clk::now();
            model.forward_batch(it.data(), nb, bo, logits.data());
            step_ms.push_back(std::chrono::duration<double, std::milli>(Clk::now() - t0).count());
            for (int j = 0; j < nb; j++) { cur[j] = argmax_row(&logits[(size_t)j * V], V); out[j].push_back(cur[j]); pos[j]++; }
        }
    };
    std::vector<int> all(B);
    for (int i = 0; i < B; i++) all[i] = i;
    if (!sweep.empty()) {
        std::vector<std::vector<double>> sm(sweep.size()), su(sweep.size());
        for (int r = 0; r < reps; r++)
            for (size_t k = 0; k < sweep.size(); k++) {
                std::vector<int> w(sweep[k]);
                for (int i = 0; i < sweep[k]; i++) w[i] = i;
                std::vector<std::vector<int>> o2;
                std::vector<double> st2;
                model.batch_stats_reset();
                const std::string la = load1();
                decode(w, true, o2, st2);
                sm[k].push_back(median(st2));
                su[k].push_back(model.batch_unique_experts());
                std::fprintf(stderr, "sweep rep %d B=%d step_ms_median %.2f load %s->%s\n", r, sweep[k], sm[k].back(), la.c_str(), load1().c_str());
            }
        const long peak = std::atol(proc_status("VmHWM").c_str()) / 1024;
        for (size_t k = 0; k < sweep.size(); k++) {
            const double ms = median(sm[k]);
            std::printf("{\"sweep_B\":%d,\"n\":%d,\"prompt_len\":%d,\"fast\":%d,\"reps\":%d,\"step_ms_median\":%.2f,\"step_ms_min\":%.2f,\"step_ms_max\":%.2f,\"agg_tok_s\":%.2f,\"per_stream_tok_s\":%.2f,"
                        "\"unique_experts_per_layer\":%.2f,\"peak_rss_mb\":%ld}\n",
                        sweep[k], N, Lp, fastv, reps, ms, *std::min_element(sm[k].begin(), sm[k].end()), *std::max_element(sm[k].begin(), sm[k].end()), sweep[k] * 1000.0 / ms, 1000.0 / ms, median(su[k]), peak);
        }
        return 0;
    }
    std::vector<std::vector<int>> out;
    std::vector<double> steps;
    const std::string load_a = load1();
    model.batch_stats_reset();
    decode(all, true, out, steps);
    const std::string load_b = load1();
    const double uniq = model.batch_unique_experts();
    const long peak_mb = std::atol(proc_status("VmHWM").c_str()) / 1024;
    double tot = 0;
    for (double v : steps) tot += v;
    const double med = median(steps);
    const double agg_med = B * 1000.0 / med, agg_mean = B * N * 1000.0 / tot;
    double bprof[8];
    for (int k = 0; k < 8; k++) bprof[k] = model.batch_prof_ms[k] / N;

    std::vector<int> mism(B, -1);   // per stream: first differing index vs the same stream alone (-1 = identical), -2 = not checked
    std::vector<double> solo_med;
    int nbad = 0, tok_bad = 0, chain_bad = 0, chain_tot = 0;
    if (verify) {
        for (int i = 0; i < B; i++) {
            std::vector<std::vector<int>> o1;
            std::vector<double> st1;
            decode({i}, false, o1, st1);
            solo_med.push_back(median(st1));
            int first = -1, nd = 0;
            for (int j = 0; j < N; j++) if (o1[0][j] != out[i][j]) { if (first < 0) first = j; nd++; }
            mism[i] = first;
            if (first >= 0) nbad++;
            tok_bad += nd;
        }
        {   // verify chain: K consecutive positions of ONE slice in one forward_batch (speculative-verify shape) must reproduce the T=1 greedy tokens
            std::vector<std::vector<int>> o0;
            std::vector<double> s0;
            decode({0}, false, o0, s0);
            std::vector<int> seq(1, prompts[0].back());
            seq.insert(seq.end(), o0[0].begin(), o0[0].end());
            const int P0 = (int)prompts[0].size();
            for (int K : {2, 4, 6})
                for (int st = 0; st + K <= N; st += K) {
                    std::vector<BatchItem> it(K);
                    for (int j = 0; j < K; j++) it[j] = {seq[st + j], P0 - 1 + st + j, slice[0]};
                    BatchOpts bo; bo.greedy = true; bo.fast = fastv;
                    model.forward_batch(it.data(), K, bo, logits.data());
                    for (int j = 0; j < K; j++) { chain_tot++; if (argmax_row(&logits[(size_t)j * V], V) != o0[0][st + j]) chain_bad++; }
                }
        }
    } else std::fill(mism.begin(), mism.end(), -2);

    std::printf("{\"batch\":%d,\"n\":%d,\"prompt_len\":%d,\"fast\":%d,\"ctx_mid\":%d,\"prefill_s\":%.2f,\"step_ms_median\":%.2f,\"step_ms_mean\":%.2f,"
                "\"agg_tok_s\":%.2f,\"agg_tok_s_mean\":%.2f,\"per_stream_tok_s\":%.2f,\"unique_experts_per_layer\":%.2f,\"load\":\"%s/%s\",\"peak_rss_mb\":%ld",
                B, N, Lp, fastv, maxP + N / 2, prefill_s, med, tot / N, agg_med, agg_mean, agg_med / B, uniq, load_a.c_str(), load_b.c_str(), peak_mb);
    if (verify) {
        std::printf(",\"streams_mismatch\":%d,\"tokens_mismatch\":%d,\"tokens_total\":%d,\"solo_step_ms_median\":%.2f,\"mismatch_first_idx\":[", nbad, tok_bad, B * N, median(solo_med));
        for (int i = 0; i < B; i++) std::printf("%s%d", i ? "," : "", mism[i]);
        std::printf("],\"chain_mismatch\":%d,\"chain_total\":%d", chain_bad, chain_tot);
    }
    if (print_tokens) {
        std::printf(",\"tokens\":[");
        for (int i = 0; i < B; i++) { std::printf("%s[", i ? "," : ""); for (int j = 0; j < N; j++) std::printf("%s%d", j ? "," : "", out[i][j]); std::printf("]"); }
        std::printf("]");
    }
    std::printf("}\n");
    if (lb::env_get("BPROF"))
        std::fprintf(stderr, "bprof ms/step (batch decode): qkv %.2f attn %.2f o %.2f moe %.2f head %.2f other %.2f\n", bprof[0], bprof[1], bprof[2], bprof[3], bprof[4], bprof[5]);
    return nbad ? 3 : 0;
}

}  // namespace lb
