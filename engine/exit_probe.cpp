#include "exit_probe.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>

namespace lb {

namespace {

std::vector<int> split_ints(const std::string& q) {
    std::vector<int> v;
    size_t p0 = 0;
    while (p0 < q.size()) { size_t p1 = q.find(',', p0); if (p1 == std::string::npos) p1 = q.size(); v.push_back(std::atoi(q.substr(p0, p1 - p0).c_str())); p0 = p1 + 1; }
    return v;
}

std::vector<int> read_ids(const std::string& path) {
    std::ifstream in(path);
    std::stringstream ss;
    ss << in.rdbuf();
    std::string s = ss.str();
    std::vector<int> ids;
    for (size_t i = 0; i < s.size();) {
        if (s[i] >= '0' && s[i] <= '9') { long v = 0; while (i < s.size() && s[i] >= '0' && s[i] <= '9') v = std::min(v * 10 + (s[i++] - '0'), 100000000L); ids.push_back((int)v); }
        else i++;
    }
    return ids;
}

// argmax + top-1 softmax probability (double accumulation)
void top1(const float* l, int n, int* arg, double* conf) {
    int b = 0;
    for (int i = 1; i < n; i++) if (l[i] > l[b]) b = i;
    double s = 0.0;
    for (int i = 0; i < n; i++) if (l[i] > -1e30f) s += std::exp((double)l[i] - (double)l[b]);
    *arg = b;
    *conf = 1.0 / s;
}

double rms(const float* x, int n) { double s = 0; for (int i = 0; i < n; i++) s += (double)x[i] * x[i]; return std::sqrt(s / n); }

// full head, 4-bit screen head, rescaled full head (only tokens after the calibration window), plain full head on those same tokens
struct Rec { bool has_r = false, a0 = false, a1 = false, aR = false; double c0 = 0, c1 = 0, cR = 0; };

const double kThr[4] = {0.5, 0.8, 0.9, 0.95};

void print_stats(const char* name, const std::vector<std::pair<bool, double>>& v) {
    int agree = 0;
    for (auto& r : v) agree += r.first;
    std::printf("\"%s\":{\"n\":%zu,\"agree\":%.4f,\"thr\":[", name, v.size(), v.empty() ? 0.0 : (double)agree / v.size());
    for (int t = 0; t < 4; t++) {
        int n = 0, ag = 0;
        for (auto& r : v) if (r.second >= kThr[t]) { n++; ag += r.first; }
        std::printf("%s{\"t\":%.2f,\"frac\":%.4f,\"agree\":%.4f}", t ? "," : "", kThr[t], v.empty() ? 0.0 : (double)n / v.size(), n ? (double)ag / n : 0.0);
    }
    std::printf("]}");
}

}  // namespace

int run_exit_probe(Transformer& model, const ModelConfig& cfg, const std::map<std::string, std::string>& a) {
    auto get = [&](const char* k, const char* d) { auto it = a.find(k); return it == a.end() ? std::string(d) : it->second; };
    std::vector<int> Ls = split_ints(get("--exit-probe", "14,16,20,24,32"));
    const int NL = cfg.num_layers, H = cfg.hidden_size, V = cfg.vocab_size;
    for (int L : Ls) if (L < 1 || L >= NL) { std::fprintf(stderr, "exit-probe: L must be in 1..%d\n", NL - 1); return 2; }
    std::vector<int> all = Ls;
    all.push_back(NL);   // sanity: layer NL through the probe head == the forward's own logits (must agree 100%); also the 4-bit head at full depth
    model.probe_layers = all;
    model.enable_screen_head();
    const bool have_screen = model.head_is_int8();
    std::vector<std::string> files;
    { std::string q = get("--prompt-files", ""); size_t p0 = 0; while (p0 < q.size()) { size_t p1 = q.find(',', p0); if (p1 == std::string::npos) p1 = q.size(); files.push_back(q.substr(p0, p1 - p0)); p0 = p1 + 1; } }
    if (files.empty()) { std::fprintf(stderr, "exit-probe: --prompt-files required\n"); return 2; }
    const int n_new = std::atoi(get("--n", "100").c_str()), calib = std::atoi(get("--calib", "100").c_str());
    std::vector<std::vector<Rec>> recs(all.size());
    std::vector<double> ratio_sum(all.size(), 0.0);   // sum of rms(x_NL)/rms(x_L) over the first `calib` tokens
    std::vector<float> lg(V), pl(V), ps(V), px(H);
    int ntok = 0;
    long mism48 = 0;
    for (const std::string& pf : files) {
        std::vector<int> ids = read_ids(pf);
        if (ids.empty()) continue;
        model.kv().len = 0;
        ForwardOpts fo; fo.logits_from = (int)ids.size() - 1; fo.greedy = false; fo.screen_ok = false;
        model.forward(ids.data(), (int)ids.size(), 0, fo, lg.data());
        int pos = (int)ids.size();
        for (int step = 0; step < n_new; step++) {
            int ref; double rc; top1(lg.data(), V, &ref, &rc);
            const float* x48 = &model.probe_x[(size_t)Ls.size() * H];
            const bool calibrating = ntok < calib;
            for (size_t i = 0; i < all.size(); i++) {
                const float* x = &model.probe_x[i * H];
                Rec r;
                int am; double cf;
                model.probe_logits(x, 0, pl.data()); top1(pl.data(), V, &am, &cf); r.a0 = am == ref; r.c0 = cf;
                if (all[i] == NL && !r.a0) mism48++;
                if (have_screen) { model.probe_logits(x, 1, ps.data()); top1(ps.data(), V, &am, &cf); r.a1 = am == ref; r.c1 = cf; }
                if (calibrating) ratio_sum[i] += rms(x48, H) / rms(x, H);
                else {   // rescale variant: x * (calibrated rms ratio) before output_norm
                    const double sc = ratio_sum[i] / calib;
                    for (int j = 0; j < H; j++) px[j] = (float)(x[j] * sc);
                    model.probe_logits(px.data(), 0, pl.data()); top1(pl.data(), V, &am, &cf); r.aR = am == ref; r.cR = cf; r.has_r = true;
                }
                recs[i].push_back(r);
            }
            ntok++;
            if (std::find(cfg.eos_token_ids.begin(), cfg.eos_token_ids.end(), ref) != cfg.eos_token_ids.end()) break;
            if (pos + 1 >= model.kv().max_seq) break;
            fo.logits_from = 0;
            model.forward(&ref, 1, pos, fo, lg.data());
            pos++;
        }
        std::fprintf(stderr, "[exit-probe] %s: %d tokens so far\n", pf.c_str(), ntok);
    }
    std::printf("{\"label\":\"%s\",\"n\":%d,\"sanity_mismatch_at_full_depth\":%ld,\"screen\":%d,\"layers\":[", get("--label", "?").c_str(), ntok, mism48, have_screen ? 1 : 0);
    for (size_t i = 0; i < all.size(); i++) {
        std::vector<std::pair<bool, double>> f, s, rr, bb;
        for (const Rec& r : recs[i]) {
            f.push_back({r.a0, r.c0});
            if (have_screen) s.push_back({r.a1, r.c1});
            if (r.has_r) { rr.push_back({r.aR, r.cR}); bb.push_back({r.a0, r.c0}); }
        }
        std::printf("%s{\"L\":%d,", i ? "," : "", all[i]);
        print_stats("full", f); std::printf(",");
        print_stats("screen", s); std::printf(",");
        print_stats("resc", rr); std::printf(",");
        print_stats("base_same_tokens", bb);
        std::printf("}");
    }
    std::printf("]}\n");
    model.probe_layers.clear();
    return 0;
}

}  // namespace lb
