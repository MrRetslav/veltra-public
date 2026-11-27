#include "aux_dump.h"

#include <algorithm>
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

void write_npy_f32(const std::string& path, const std::vector<float>& d, size_t n0, size_t n1, size_t n2) {
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(n0) + ", " + std::to_string(n1) + ", " + std::to_string(n2) + "), }";
    while ((10 + hdr.size() + 1) % 64) hdr += ' ';
    hdr += '\n';
    std::ofstream o(path, std::ios::binary);
    const unsigned char magic[10] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, (unsigned char)(hdr.size() & 255), (unsigned char)(hdr.size() >> 8)};
    o.write((const char*)magic, 10);
    o.write(hdr.data(), (std::streamsize)hdr.size());
    o.write((const char*)d.data(), (std::streamsize)(d.size() * 4));
}

}  // namespace

int run_aux_dump(Transformer& model, const ModelConfig& cfg, const std::map<std::string, std::string>& a) {
    auto get = [&](const char* k, const char* d) { auto it = a.find(k); return it == a.end() ? std::string(d) : it->second; };
    const std::vector<int> Ls = split_ints(get("--dump-aux", "2,24,45"));
    const int NL = (int)Ls.size(), H = cfg.hidden_size, V = cfg.vocab_size;
    for (int L : Ls) if (L < 1 || L > cfg.num_layers) { std::fprintf(stderr, "dump-aux: L must be in 1..%d\n", cfg.num_layers); return 2; }
    const std::string prefix = get("--dump-aux-out", "");
    if (prefix.empty()) { std::fprintf(stderr, "dump-aux: --dump-aux-out PREFIX required\n"); return 2; }
    std::vector<std::string> files;
    { std::string q = get("--prompt-files", ""); size_t p0 = 0; while (p0 < q.size()) { size_t p1 = q.find(',', p0); if (p1 == std::string::npos) p1 = q.size(); files.push_back(q.substr(p0, p1 - p0)); p0 = p1 + 1; } }
    if (files.empty()) { std::fprintf(stderr, "dump-aux: --prompt-files required\n"); return 2; }
    const int n_new = std::atoi(get("--n", "150").c_str());
    model.probe_layers = Ls;   // forward() copies the residual stream of the (only) token after each listed layer into probe_x
    std::vector<float> lg(V);
    for (const std::string& pf : files) {
        std::vector<int> prompt = read_ids(pf);
        if (prompt.empty()) continue;
        const int P = (int)prompt.size();
        std::vector<int> ids = prompt, y(P, -1);
        std::vector<float> aux;
        aux.reserve((size_t)(P + n_new + 1) * NL * H);
        model.kv().len = 0;
        ForwardOpts fo; fo.greedy = false; fo.screen_ok = false;
        for (int j = 0; j < P + n_new + 1 && j < model.kv().max_seq; j++) {
            fo.logits_from = j >= P - 1 ? 0 : 1;   // T=1: logits_from 1 = none (prompt positions before the last)
            model.forward(&ids[j], 1, j, fo, lg.data());
            aux.insert(aux.end(), model.probe_x.begin(), model.probe_x.begin() + (size_t)NL * H);
            if (j >= P - 1) {
                int b = 0;
                for (int i = 1; i < V; i++) if (lg[i] > lg[b]) b = i;
                if (j == (int)y.size()) y.push_back(b); else y[j] = b;
                if (j + 1 >= (int)ids.size()) {
                    if ((int)ids.size() - P >= n_new || std::find(cfg.eos_token_ids.begin(), cfg.eos_token_ids.end(), b) != cfg.eos_token_ids.end()) break;
                    ids.push_back(b);
                }
            }
        }
        const int N = (int)ids.size();
        y.resize(N, -1);
        std::string base = prefix + "_" + pf.substr(pf.find_last_of("/\\") == std::string::npos ? 0 : pf.find_last_of("/\\") + 1);
        if (base.size() > 4 && base.compare(base.size() - 4, 4, ".ids") == 0) base.resize(base.size() - 4);
        write_npy_f32(base + ".aux.npy", aux, (size_t)N, (size_t)NL, (size_t)H);
        std::ofstream o(base + ".json");
        o << "{\"n_prompt\":" << P << ",\"layers\":[";
        for (int i = 0; i < NL; i++) o << (i ? "," : "") << Ls[i];
        o << "],\"ids\":[";
        for (int i = 0; i < N; i++) o << (i ? "," : "") << ids[i];
        o << "],\"argmax\":[";
        for (int i = 0; i < N; i++) o << (i ? "," : "") << y[i];
        o << "]}\n";
        std::fprintf(stderr, "[dump-aux] %s: prompt %d + generated %d -> %s.aux.npy\n", pf.c_str(), P, N - P, base.c_str());
    }
    model.probe_layers.clear();
    std::printf("{\"dump_aux\":\"ok\"}\n");
    return 0;
}

}  // namespace lb
