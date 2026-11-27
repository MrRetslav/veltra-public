// Tokenizer-free CLI over the engine core (used by tests): token ids in, token ids out.
//   veltra-gen --model m.vlt --ids 1,2,3 --n 32 [--spec 0|1] [--ngram 0|1] [--layer-skip 0|1] [--kmin 2 --kmax 8 --k 5]
//          [--kv f32|int8] [--threads N] [--temp T --top-k K --top-p P --seed S] [--isa auto|scalar|avx2|avx512]
//          [--dump-logits out.bin]   (full-model logits for every prompt position, float32 [T][vocab])
//          [--bench-draft]           (time draft-forward vs full-forward per token)
//          [--int8-head] [--exact-head] [--head-vocab N] [--head-vocab-file PATH]   (output-head options, same as veltra-server)
//          [--bench-linear model.lbt blk.3.down T threads [isa]]   (ms per linear_forward of one linear)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "binary_gemv.h"
#include "budget_probe.h"
#include "chat_template.h"
#include "json.h"
#include "search.h"
#include "teal_text.h"
#include "tokenizer.h"
#include "lbt_model.h"
#include "model_config.h"
#include "ops.h"
#include "rest_index.h"
#include "speculative.h"
#include "threadpool.h"
#include "transformer.h"
#include "batch_gen.h"
#include "aux_dump.h"
#include "exit_probe.h"
#include "lb_common.h"

namespace lb { extern double g_prof_ms[16]; extern double g_prof2_ms[4]; }
using namespace lb;

int main(int argc, char** argv) {
    if (argc >= 4 && std::string(argv[1]) == "--rest-check") {   // lb_gen --rest-check rest.idx rest.idx.check : hash + chain cross-check against scripts/rest_build.py
        std::string err;
        if (!rest_load(argv[2], &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        std::ifstream in(argv[3]);
        std::string line;
        int ok = 0, bad = 0;
        while (std::getline(in, line)) {
            if (line.rfind("H ", 0) == 0) {   // "H n t1 .. tn hash": hash function parity
                std::istringstream ss(line.substr(2));
                int n = 0; ss >> n;
                std::vector<int32_t> c(n);
                for (auto& x : c) ss >> x;
                unsigned long long h = 0; ss >> h;
                if (rest_hash(c.data(), n) == h) ok++; else { bad++; std::printf("HASH MISMATCH n=%d\n", n); }
                continue;
            }
            std::istringstream ss(line);
            int order = 0, len = 0; unsigned long long h = 0;
            ss >> order >> h >> len;
            int tok[8]; float p[8];
            const int got = rest_index()->lookup_hash(h, tok, p);
            bool same = got == len;
            for (int j = 0; j < len && same; j++) { char colon; int t, q; ss >> t >> colon >> q; if (tok[j] != t || (int)(p[j] * 15.f + 0.5f) != q) same = false; }
            if (same) ok++; else { bad++; if (bad < 5) std::printf("ENTRY MISMATCH hash %llu\n", h); }
        }
        std::printf("rest-check: %d ok, %d bad, entries %llu, %.1f MB\n", ok, bad, (unsigned long long)rest_index()->n_entries(), rest_index()->mapped_bytes() / 1e6);
        return bad ? 1 : 0;
    }

    if (std::string(argv[1]) == "--bench-pool") {  // fork-join latency: tasks of ~0 work
        const int th = std::atoi(argv[2]), ntask = std::atoi(argv[3]), work = argc > 4 ? std::atoi(argv[4]) : 0;
        ThreadPool pool(th);
        std::vector<float> sink(64, 1.f);
        auto fn = [&](int t) { float a = 1.f; for (int i = 0; i < work; i++) a = a * 1.0000001f + 1e-9f; sink[t & 63] = a; };
        for (int i = 0; i < 2000; i++) pool.run(ntask, fn);
        const int reps = 20000;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; i++) pool.run(ntask, fn);
        std::printf("threads=%d tasks=%d work=%d: %.2f us per run()\n", th, ntask, work, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps);
        return 0;
    }
    if (std::string(argv[1]) == "--bench-gemv-mt") {  // DRAM-resident: nmat matrices cycled, pool-parallel over row blocks: rows cols T nmat threads
        using namespace lb;
        const int rows = std::atoi(argv[2]), cols = std::atoi(argv[3]), T = std::atoi(argv[4]), nmat = std::atoi(argv[5]), th = std::atoi(argv[6]);
        const int kp = pad_k(cols), rp = pad_rows(rows), nrb = rp / 64;
        std::vector<AlignedVec<uint8_t>> w5(nmat);
        std::vector<uint8_t> w(packed_weight_bytes(rows, cols));
        for (int m = 0; m < nmat; m++) {
            for (size_t i = 0; i < w.size(); i++) w[i] = (uint8_t)((i + m) * 2654435761u >> 13);
            w5[m].resize(w.size());
            repack_lut_avx512(w.data(), rows, cols, w5[m].data());
        }
        AlignedVec<int8_t> luts((size_t)T * lut_bytes_per_token(kp));
        std::vector<int8_t> q(kp);
        for (int t = 0; t < T; t++) { for (int k = 0; k < kp; k++) q[k] = (int8_t)((k * 7 + t) % 200 - 100); build_luts(q.data(), kp, &luts[(size_t)t * lut_bytes_per_token(kp)], true); }
        std::vector<int32_t> out((size_t)T * rp);
        ThreadPool pool(th);
        using C = std::chrono::steady_clock;
        auto run_one = [&](int m) {
            BinMatView v; v.rows = rows; v.cols = cols; v.w512 = w5[m].data();
            const int tasks = std::min(nrb, th * 2);
            pool.run(tasks, [&](int ti) { binary_gemv(Isa::Avx512, v, luts.data(), T, out.data(), nrb * ti / tasks, nrb * (ti + 1) / tasks); });
        };
        for (int m = 0; m < nmat; m++) run_one(m);
        const int passes = 5;
        auto t0 = C::now();
        for (int p = 0; p < passes; p++) for (int m = 0; m < nmat; m++) run_one(m);
        const double us = std::chrono::duration<double, std::micro>(C::now() - t0).count() / (passes * nmat);
        std::printf("rows=%d cols=%d T=%d nmat=%d threads=%d: %.1f us per matrix, %.1f GB/s weights\n", rows, cols, T, nmat, th, us, (double)w.size() / us / 1e3);
        return 0;
    }
    if (std::string(argv[1]) == "--bench-gemv") {  // single-thread kernel micro-benchmark: rows cols T isa(2=avx512,1=avx2)
        using namespace lb;
        const int rows = std::atoi(argv[2]), cols = std::atoi(argv[3]), T = std::atoi(argv[4]);
        const Isa isa = (Isa)std::atoi(argv[5]);
        const int kp = pad_k(cols), rp = pad_rows(rows);
        std::vector<uint8_t> w(packed_weight_bytes(rows, cols));
        for (size_t i = 0; i < w.size(); i++) w[i] = (uint8_t)(i * 2654435761u >> 13);
        std::vector<uint8_t> w5(w.size());
        repack_lut_avx512(w.data(), rows, cols, w5.data());
        std::vector<int8_t> q((size_t)T * kp, 3), lutbuf((size_t)T * lut_bytes_per_token(kp) + 64);
        int8_t* lutp = (int8_t*)(((uintptr_t)lutbuf.data() + 63) & ~(uintptr_t)63); if (lb::env_get("MISALIGN")) lutp += 16;
        for (int t = 0; t < T; t++) { for (int k = 0; k < kp; k++) q[(size_t)t * kp + k] = (int8_t)((k * 7 + t) % 200 - 100); build_luts(&q[(size_t)t * kp], kp, lutp + (size_t)t * lut_bytes_per_token(kp), isa == Isa::Avx512); }
        BinMatView v; v.rows = rows; v.cols = cols; v.lut_w = w.data(); v.w512 = w5.data();
        std::vector<int32_t> out((size_t)T * rp);
        using C = std::chrono::steady_clock;
        for (int i = 0; i < 3; i++) binary_gemv(isa, v, lutp, T, out.data());
        const int reps = 30;
        auto t0 = C::now();
        for (int i = 0; i < reps; i++) binary_gemv(isa, v, lutp, T, out.data());
        const double us = std::chrono::duration<double, std::micro>(C::now() - t0).count() / reps;
        const double nwv = (double)(rp / 64) * (kp / 32) * 4;  // 64-byte weight vectors
        std::printf("rows=%d cols=%d T=%d isa=%d: %.1f us, %.2f ns per 64B weight vector, %.1f GB/s weights\n", rows, cols, T, (int)isa, us, us * 1000 / nwv, (double)w.size() / us / 1e3);
        return 0;
    }
    if (std::string(argv[1]) == "--bench-linear") {  // one linear of a model: file name T threads [isa]  (ms per linear_forward, min/median of 30)
        using namespace lb;
        const std::string path = argv[2], name = argv[3];
        const int T = std::atoi(argv[4]), th = std::atoi(argv[5]);
        Isa isa = best_isa();
        if (argc > 6) { std::string s6 = argv[6]; isa = s6 == "avx2" ? Isa::Avx2 : s6 == "avx512" ? Isa::Avx512 : s6 == "scalar" ? Isa::Scalar : isa; }
        LbtFile f; std::string err;
        if (!f.open(path, &err)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
        LbLinear w;
        if (!load_linear(f, name, &w, &err)) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
        ThreadPool pool(th);
        std::vector<float> x((size_t)T * w.n), y((size_t)T * w.m);
        for (size_t i = 0; i < x.size(); i++) x[i] = std::sin((float)i * 0.37f) * 1.5f;
        LinearScratch sc;
        using C = std::chrono::steady_clock;
        for (int i = 0; i < 5; i++) linear_forward(w, isa, x.data(), T, y.data(), true, sc, &pool);
        std::vector<double> ms;
        for (int i = 0; i < 40; i++) { auto t0 = C::now(); linear_forward(w, isa, x.data(), T, y.data(), true, sc, &pool); ms.push_back(std::chrono::duration<double, std::milli>(C::now() - t0).count()); }
        std::sort(ms.begin(), ms.end());
        const size_t bytes = w.is_nibble() ? (size_t)w.nib.ntile * pad_rows(w.m) * 64 : w.is_grid() ? (size_t)w.grid.P * w.grid.ntile * pad_rows(w.m) * 16 : 0;
        std::printf("%s m=%d n=%d %s T=%d threads=%d isa=%s: min %.3f ms, median %.3f ms%s\n", name.c_str(), w.m, w.n, w.is_nibble() ? "nibble" : w.is_grid() ? "grid" : "littlebit", T, th, isa_name(isa), ms[0], ms[ms.size() / 2],
                    bytes ? (" (" + std::to_string((double)bytes / ms[ms.size() / 2] / 1e6) + " GB/s weight stream)").c_str() : "");
        return 0;
    }
    if (std::string(argv[1]) == "--test-attn") {  // GQA attention chunk kernel vs a double-precision scalar reference on random data
        std::srand(7);
        auto rnd = [] { return (float)std::rand() / RAND_MAX * 2.f - 1.f; };
        double worst = 0;
        for (int G : {1, 2, 4, 8}) for (int n : {1, 5, 33, 128}) {
            std::vector<float> q((size_t)G * 128), ks((size_t)n * 8), vs((size_t)n * 8), scr((size_t)n * 8 + 32 * 128 + 64), m(8), l(8), acc((size_t)G * 128);
            std::vector<int8_t> k8((size_t)n * 128), v8((size_t)n * 128);
            for (auto& x : q) x = rnd() * 0.3f;
            for (auto& x : ks) x = (rnd() + 1.1f) * 0.02f;
            for (auto& x : vs) x = (rnd() + 1.1f) * 0.02f;
            for (auto& x : k8) x = (int8_t)(rnd() * 127);
            for (auto& x : v8) x = (int8_t)(rnd() * 127);
            if (!attn_gqa_chunk(G, q.data(), k8.data(), ks.data(), v8.data(), vs.data(), n, m.data(), l.data(), acc.data(), scr.data())) { std::printf("no kernel\n"); return 1; }
            for (int h = 0; h < G; h++) {
                std::vector<double> s(n);
                double mx = -1e300;
                for (int j = 0; j < n; j++) { double d = 0; for (int i = 0; i < 128; i++) d += (double)q[h * 128 + i] * k8[j * 128 + i] * ks[j * 8 + i / 16]; s[j] = d; mx = std::max(mx, d); }
                double ls = 0; std::vector<double> o(128, 0.0);
                for (int j = 0; j < n; j++) { double e = std::exp(s[j] - mx); ls += e; for (int i = 0; i < 128; i++) o[i] += e * v8[j * 128 + i] * vs[j * 8 + i / 16]; }
                double err = std::fabs(m[h] - mx) / (1 + std::fabs(mx)) + std::fabs(l[h] - ls) / ls;
                for (int i = 0; i < 128; i++) err = std::max(err, std::fabs(acc[h * 128 + i] - o[i]) / (std::fabs(o[i]) + 0.05 * ls));
                worst = std::max(worst, err);
            }
        }
        for (int G : {8, 4}) {   // speed: ns per (position x kv-head) for a 128-position chunk
            const int n = 128;
            std::vector<float> q((size_t)G * 128, 0.1f), ks((size_t)n * 8, 0.02f), vs((size_t)n * 8, 0.02f), scr((size_t)n * 8 + 32 * 128 + 64), m(8), l(8), acc((size_t)G * 128);
            std::vector<int8_t> k8((size_t)n * 128, 3), v8((size_t)n * 128, 5);
            using C = std::chrono::steady_clock;
            for (int r = 0; r < 200; r++) attn_gqa_chunk(G, q.data(), k8.data(), ks.data(), v8.data(), vs.data(), n, m.data(), l.data(), acc.data(), scr.data());
            auto t0 = C::now();
            for (int r = 0; r < 5000; r++) attn_gqa_chunk(G, q.data(), k8.data(), ks.data(), v8.data(), vs.data(), n, m.data(), l.data(), acc.data(), scr.data());
            const double ns = std::chrono::duration<double, std::nano>(C::now() - t0).count() / 5000 / n;
            std::printf("G=%d: %.1f ns per position (all %d heads) = %.1f ns/head-position\n", G, ns, G, ns / G);
        }
        std::printf("attn_gqa_chunk worst relative error %.3g (%s)\n", worst, worst < 1e-4 ? "OK" : "FAIL");
        return worst < 1e-4 ? 0 : 1;
    }
    std::map<std::string, std::string> a;
    for (int i = 1; i + 1 < argc + 1; i++) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0) continue;
        if (k == "--bench-draft" || k == "--bench-batch") { a[k] = "1"; continue; }
        if (k == "--int8-head" || k == "--exact-head" || k == "--screen-head") {  // bare flag = on; optional 0/1 value
            if (i + 1 < argc && (std::string(argv[i + 1]) == "0" || std::string(argv[i + 1]) == "1" || (k == "--screen-head" && std::string(argv[i + 1]) == "2"))) a[k] = argv[++i]; else a[k] = "1";
            continue;
        }
        if (i + 1 < argc) a[k] = argv[++i];
    }
    auto get = [&](const char* k, const char* d) { auto it = a.find(k); return it == a.end() ? std::string(d) : it->second; };
    if (!a.count("--model")) { std::fprintf(stderr, "usage: veltra-gen --model m.vlt --ids 1,2,3 ...  (legacy .lbt files load too; env VELTRA_<NAME> or LB_<NAME>)\n"); return 2; }
    try {
        LbtFile f;
        std::string err;
        if (!f.open(a["--model"], &err)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
        ModelConfig cfg;
        if (!parse_model_config(f, &cfg, &err)) { std::fprintf(stderr, "config: %s\n", err.c_str()); return 1; }
        if (a.count("--encode-file")) {   // tokenizer / chat-template parity: JSON lines {"text": s} or {"messages": [...], "thinking": 0|1} -> one id list per line
            const TensorRef* tj = f.find("tokenizer.json");
            const TensorRef* tc = f.find("tokenizer_config.json");
            if (!tj) { std::fprintf(stderr, "the model has no embedded tokenizer.json\n"); return 1; }
            std::unique_ptr<Tokenizer> tok = Tokenizer::from_json(std::string((const char*)tj->data, (size_t)tj->nbytes),
                                                                  tc ? std::string((const char*)tc->data, (size_t)tc->nbytes) : std::string(), &err);
            if (!tok) { std::fprintf(stderr, "tokenizer: %s\n", err.c_str()); return 1; }
            std::ifstream in(a["--encode-file"]);
            std::string line;
            while (std::getline(in, line)) {
                json::Value v;
                if (line.empty() || !json::parse(line, &v, &err)) continue;
                std::vector<int> out;
                if (v.find("messages")) {
                    std::vector<ChatMessage> msgs;
                    for (const auto& m : v["messages"].as_array()) msgs.push_back({m["role"].as_string(""), m["content"].as_string("")});
                    out = apply_chat_template(*tok, cfg.arch, msgs, true, v["thinking"].as_int(0) != 0);
                } else out = tok->encode(v["text"].as_string(""), false);
                std::printf("[");
                for (size_t i = 0; i < out.size(); i++) std::printf("%s%d", i ? "," : "", out[i]);
                std::printf("]\n");
            }
            return 0;
        }
        std::string isas = get("--isa", "auto");
        Isa isa = default_isa_for(f);
        if (isas == "scalar") isa = Isa::Scalar; else if (isas == "avx2") isa = Isa::Avx2; else if (isas == "avx512") isa = Isa::Avx512;
        if (!isa_available(isa)) { std::fprintf(stderr, "isa not available\n"); return 1; }
        KvMode kvm = get("--kv", "int8") == "f32" ? KvMode::F32 : KvMode::Int8;
        std::vector<int> ids;
        {
            std::string s = get("--ids", "1");
            size_t p = 0;
            while (p < s.size()) {
                size_t q = s.find(',', p);
                if (q == std::string::npos) q = s.size();
                ids.push_back(std::atoi(s.substr(p, q - p).c_str()));
                p = q + 1;
            }
        }
        int max_ctx = std::atoi(get("--ctx", "2048").c_str());
        if (a.count("--draft-skip")) {  // override draft skip set: N middle layers (0 = none)
            cfg.draft_layer_skip.clear();
            int nskip = std::atoi(a["--draft-skip"].c_str()), c0 = cfg.num_layers / 2 - nskip / 2;
            for (int i = 0; i < nskip; i++) cfg.draft_layer_skip.push_back(c0 + i);
        }
        Transformer model(f, cfg, max_ctx, kvm, std::atoi(get("--threads", "0").c_str()), isa);

        model.moe_turbo.apply(get("--mode", "strict") == "turbo", a.count("--moe-alpha") ? (float)std::atof(a["--moe-alpha"].c_str()) : -1.f,
                              std::atoi(get("--moe-min-k", "-1").c_str()), std::atoi(get("--moe-kmax", "-1").c_str()),
                              a.count("--moe-mass") ? (float)std::atof(a["--moe-mass"].c_str()) : -1.f,
                              std::atoi(get("--moe-edge", "-1").c_str()), std::atoi(get("--moe-edge-k", "-1").c_str()), std::atoi(get("--moe-prefill-prune", "-1").c_str()));
        const bool pareto = get("--pareto-max", "0") != "0";   // --pareto-max 1: the locked preset; the single flags below win
        model.moe_k = std::atoi(get("--moe-k", pareto ? std::to_string(ParetoMaxPreset::moe_k).c_str() : "0").c_str());
        set_prefill_chunk(std::atoi(get("--prefill-chunk", pareto ? std::to_string(ParetoMaxPreset::prefill_chunk).c_str() : "0").c_str()));
        if (a.count("--moe-k-layers")) {
            const std::string list = a["--moe-k-layers"];
            for (size_t p = 0; p < list.size();) {
                model.moe_k_layer.push_back(std::atoi(list.c_str() + p));
                const size_t c = list.find(',', p);
                p = c == std::string::npos ? list.size() : c + 1;
            }
        }
        if (a.count("--shadow-full")) {   // experimental shadow-expert quality emulation (Transformer::shadow_full)
            if (!model.load_shadow_masks(a["--shadow-masks"])) { std::fprintf(stderr, "--shadow-masks: cannot load a mask file that matches this model\n"); return 2; }
            model.shadow_full = std::atoi(a["--shadow-full"].c_str());
        }
        if (a.count("--shadow-variants")) return run_shadow_probe(model, cfg, a);
        if (a.count("--budget-fit")) return run_budget_probe(model, cfg, a);   // LBT-50: per-layer output-sensitive expert budget (engine/budget_probe.cpp)
        if (a.count("--teal-sparsity")) {
            const TensorRef* tj = f.find("tokenizer.json");
            const TensorRef* tc = f.find("tokenizer_config.json");
            if (!tj) { std::fprintf(stderr, "teal: the model has no embedded tokenizer.json\n"); return 1; }
            std::unique_ptr<Tokenizer> tok = Tokenizer::from_json(std::string((const char*)tj->data, (size_t)tj->nbytes),
                                                                  tc ? std::string((const char*)tc->data, (size_t)tc->nbytes) : std::string(), &err);
            if (!tok) { std::fprintf(stderr, "tokenizer: %s\n", err.c_str()); return 1; }
            teal_calibrate(model, *tok, (float)std::atof(a["--teal-sparsity"].c_str()), max_ctx);
        }
        if (get("--int8-head", "0") == "1") model.enable_int8_head();
        if (get("--exact-head", "0") == "1") model.enable_exact_head();
        if (a.count("--screen-sigma2")) model.screen_sigma2 = (float)std::atof(a["--screen-sigma2"].c_str());
        const std::string pm_head = std::to_string(ParetoMaxPreset::screen_head);
        if (get("--screen-head", pareto ? pm_head.c_str() : "0") != "0" || a.count("--bench-head") || a.count("--screen-calib"))
            model.enable_screen_head(get("--screen-head", pareto ? pm_head.c_str() : "1") == "2" ? 2 : 4);
        if (a.count("--screen-sigma")) model.screen_sigma = (float)std::atof(a["--screen-sigma"].c_str());
        if (a.count("--screen-calib")) {   // --screen-calib hidden.bin [--screen-head 2]: z = (exact - est) / predicted sigma over all rows, and rerank survivors per sigma
            std::vector<float> hid;
            { std::ifstream in(a["--screen-calib"], std::ios::binary | std::ios::ate); size_t n = (size_t)in.tellg() / 4; hid.resize(n); in.seekg(0); in.read((char*)hid.data(), n * 4); }
            const int H = cfg.hidden_size, V = cfg.vocab_size, NV = std::min((int)(hid.size() / H), std::atoi(get("--calib-n", "100000").c_str()));
            std::vector<float> ex((size_t)V), est((size_t)V), sig((size_t)V);
            const double zk[5] = {3, 4, 5, 6, 8};
            long nz[5] = {0}, ntot = 0, ztop_over[5] = {0};
            double zmax = 0, ztop_max = 0, zsum2 = 0, ztop_sum = 0;
            long ntop = 0;
            const float sgs[6] = {3, 4, 5, 6, 7, 8};
            std::vector<long> cs[6];
            long miss[6] = {0};
            model.head_mode = 0;
            for (int v = 0; v < NV; v++) {
                const float* x = &hid[(size_t)v * H];
                model.head_only(x, 1, ex.data());   // full exact int8 head (float x i8)
                model.screen_estimate(x, est.data(), sig.data());
                int am = 0;
                for (int r = 1; r < V; r++) if (ex[r] > ex[am]) am = r;
                for (int r = 0; r < V; r++) {
                    const double z = std::fabs((double)(ex[r] - est[r]) / sig[r]);
                    zsum2 += z * z; ntot++;
                    zmax = std::max(zmax, z);
                    for (int k = 0; k < 5; k++) if (z > zk[k]) nz[k]++;
                    if (ex[r] >= ex[am] - 2.f) { ztop_sum += (ex[r] - est[r]) / sig[r]; ntop++; ztop_max = std::max(ztop_max, z); for (int k = 0; k < 5; k++) if (z > zk[k]) ztop_over[k]++; }   // rows near the top (within 2 logits)
                }
                for (int k = 0; k < 6; k++) {
                    float lb = -INFINITY;
                    for (int r = 0; r < V; r++) lb = std::max(lb, est[r] - sgs[k] * sig[r]);
                    long c = 0;
                    for (int r = 0; r < V; r++) if (est[r] + sgs[k] * sig[r] >= lb) c++;
                    if (est[am] + sgs[k] * sig[am] < lb) miss[k]++;
                    cs[k].push_back(c);
                }
            }
            std::printf("screen-calib bits=%d vectors=%d rows=%ld rms_z=%.3f max_z=%.2f |z|>3/4/5/6/8: %ld/%ld/%ld/%ld/%ld near-top (%ld rows) mean signed z=%.2f max_z=%.2f over 3/4/5/6/8: %ld/%ld/%ld/%ld/%ld\n",
                        model.screen_bits(), NV, ntot, std::sqrt(zsum2 / std::max(1L, ntot)), zmax, nz[0], nz[1], nz[2], nz[3], nz[4], ntop, ztop_sum / std::max(1L, ntop), ztop_max, ztop_over[0], ztop_over[1], ztop_over[2], ztop_over[3], ztop_over[4]);
            for (int k = 0; k < 6; k++) {
                std::sort(cs[k].begin(), cs[k].end());
                double mean = 0; for (long c : cs[k]) mean += c; mean /= std::max<size_t>(1, cs[k].size());
                std::printf("  sigma %.0f: survivors mean %.0f median %ld p99 %ld max %ld, argmax pruned %ld / %d\n", sgs[k], mean, cs[k][cs[k].size() / 2], cs[k][cs[k].size() * 99 / 100], cs[k].back(), miss[k], NV);
            }
            return 0;
        }
        if (a.count("--head-vocab") || a.count("--head-vocab-file")) {  // ids [0,N) + --head-vocab-file ids + eos / special ids (+ prompt ids)
            std::vector<int> extra = ids;
            if (a.count("--head-vocab-file") && !load_id_list_file(a["--head-vocab-file"], &extra, &err)) { std::fprintf(stderr, "head-vocab-file: %s\n", err.c_str()); return 1; }
            model.set_head_vocab(head_vocab_ids(cfg, std::atoi(get("--head-vocab", "0").c_str()), extra));
        }
        if (a.count("--exit-probe")) return run_exit_probe(model, cfg, a);   // early-exit diagnostic (engine/exit_probe.cpp)
        if (a.count("--dump-aux")) return run_aux_dump(model, cfg, a);   // EAGLE-3 probe: residual stream after layers L1,L2,.. for every position (engine/aux_dump.cpp)
        if (a.count("--batch") || a.count("--sweep")) return run_batch_gen(model, cfg, a);   // weight-stationary batched decode of B streams (engine/batch_gen.cpp)
        if (a.count("--search-n")) {   // batched test-time search: --search-n N [--n 128] [--seed 1] [--temp 0.65] [--top-p 0.95] [--json-mode 1] (engine/search.cpp)
            const TensorRef* tj = f.find("tokenizer.json");
            const TensorRef* tc = f.find("tokenizer_config.json");
            if (!tj) { std::fprintf(stderr, "search: the model has no embedded tokenizer.json\n"); return 1; }
            std::unique_ptr<Tokenizer> tok = Tokenizer::from_json(std::string((const char*)tj->data, (size_t)tj->nbytes),
                                                                  tc ? std::string((const char*)tc->data, (size_t)tc->nbytes) : std::string(), &err);
            if (!tok) { std::fprintf(stderr, "tokenizer: %s\n", err.c_str()); return 1; }
            GenOptions go;
            go.max_new_tokens = std::atoi(get("--n", "128").c_str());
            go.search.n = std::atoi(a["--search-n"].c_str());
            go.search.base_seed = (uint64_t)std::atoll(get("--seed", "1").c_str());
            go.search.temperature = (float)std::atof(get("--temp", "0.65").c_str());
            go.search.top_p = (float)std::atof(get("--top-p", "0.95").c_str());
            if (get("--json-mode", "0") == "1") go.search.verify.format = OutputFormat::JsonObject;
            const SearchRun r = run_search(model, *tok, ids, go, cfg.eos_token_ids);
            const BranchResult& w = r.branches[r.verdict.winner];
            std::string text;
            json::Value::escape_to(text, w.text);
            std::printf("{\"winner\":%d,\"branches_evaluated\":%d,\"valid_branches\":%d,\"pruned_branches\":%d,\"consensus_ratio\":%.3f,\"tokens\":[",
                        r.verdict.winner, r.verdict.branches_evaluated, r.verdict.valid_branches, r.verdict.pruned_branches, r.verdict.consensus_ratio);
            for (size_t i = 0; i < w.ids.size(); i++) std::printf("%s%d", i ? "," : "", w.ids[i]);
            std::printf("],\"text\":%s,\"prefill_s\":%.3f,\"decode_s\":%.3f,\"total_tokens\":%d,\"steps\":%d,\"branches\":%s}\n",
                        text.c_str(), r.stats.prefill_s, r.stats.decode_s, r.stats.search_total_tokens, r.stats.search_steps, r.stats.search_detail.c_str());
            return 0;
        }
        if (a.count("--dump-hidden")) {   // post-norm hidden vectors (what the head sees) for every prompt position: float32 [T][H]
            std::vector<float> lg((size_t)ids.size() * cfg.vocab_size);
            ForwardOpts fo; fo.logits_from = 0;
            model.forward(ids.data(), (int)ids.size(), 0, fo, lg.data());
            std::ofstream o(a["--dump-hidden"], std::ios::binary | std::ios::app);
            o.write((const char*)model.hidden_buf(), (size_t)ids.size() * cfg.hidden_size * 4);
            std::printf("{\"dumped_hidden\":%zu,\"H\":%d}\n", ids.size(), cfg.hidden_size);
            return 0;
        }
        if (a.count("--bench-head")) {    // --bench-head hidden.bin [--modes 0,1,2] [--reps 40]: head-only timing (interleaved modes) + argmax parity vs the first mode
            std::vector<float> hid;
            { std::ifstream in(a["--bench-head"], std::ios::binary | std::ios::ate); size_t n = (size_t)in.tellg() / 4; hid.resize(n); in.seekg(0); in.read((char*)hid.data(), n * 4); }
            const int H = cfg.hidden_size, V = cfg.vocab_size, NV = (int)(hid.size() / H);
            std::vector<int> modes;
            { std::string q = get("--modes", "0,1"); size_t p0 = 0; while (p0 < q.size()) { size_t p1 = q.find(',', p0); if (p1 == std::string::npos) p1 = q.size(); modes.push_back(std::atoi(q.substr(p0, p1 - p0).c_str())); p0 = p1 + 1; } }
            const int reps = std::atoi(get("--reps", "40").c_str());
            std::vector<float> lg((size_t)V);
            using C = std::chrono::steady_clock;
            auto argmax = [&](const float* l) { int b = 0; for (int i = 1; i < V; i++) if (l[i] > l[b]) b = i; return b; };
            std::vector<std::vector<double>> ms(modes.size());
            for (int r = 0; r < reps + 3; r++)
                for (size_t m = 0; m < modes.size(); m++) {
                    model.head_mode = modes[m];
                    const float* x = &hid[(size_t)(r % NV) * H];
                    auto t0 = C::now();
                    model.head_only(x, 1, lg.data());
                    const double d = std::chrono::duration<double, std::milli>(C::now() - t0).count();
                    if (r >= 3) ms[m].push_back(d);
                }
            const double mb = (double)V * H / 1e6;
            for (size_t m = 0; m < modes.size(); m++) {
                std::sort(ms[m].begin(), ms[m].end());
                std::printf("head mode %d: min %.2f ms median %.2f ms (%.1f GB/s at median, %.1f at min) load=%s\n", modes[m], ms[m][0], ms[m][ms[m].size() / 2], mb / ms[m][ms[m].size() / 2], mb / ms[m][0], get("--load", "?").c_str());
            }
            if (modes.size() > 1 && a.count("--parity")) {
                std::vector<int> ref(NV);
                model.head_mode = modes[0];
                for (int i = 0; i < NV; i++) { model.head_only(&hid[(size_t)i * H], 1, lg.data()); ref[i] = argmax(lg.data()); }
                for (size_t m = 1; m < modes.size(); m++) {
                    model.head_mode = modes[m];
                    int bad = 0;
                    for (int i = 0; i < NV; i++) { model.head_only(&hid[(size_t)i * H], 1, lg.data()); if (argmax(lg.data()) != ref[i]) bad++; }
                    std::printf("parity mode %d vs %d: %d / %d argmax mismatches\n", modes[m], modes[0], bad, NV);
                    if (modes[m] == 3 && model.screen_cand_calls) std::printf("screen bits %d: reranked rows mean %.1f max %ld over %ld calls\n", model.screen_bits(), (double)model.screen_cand_sum / model.screen_cand_calls, model.screen_cand_max, model.screen_cand_calls);
                }
            }
            return 0;
        }
        if (a.count("--dump-logits")) {
            std::vector<float> lg((size_t)ids.size() * cfg.vocab_size);
            ForwardOpts fo;
            fo.logits_from = 0;
            if (a.count("--seq-logits")) {   // one token per forward (T=1 decode path): must be bit-identical to the batched dump
                for (size_t i = 0; i < ids.size(); i++) model.forward(&ids[i], 1, (int)i, fo, lg.data() + i * cfg.vocab_size);
            } else if (a.count("--rollback-at")) {   // hybrid models: chain of 12 tokens, then restart INSIDE the chain at J (partial commit = DeltaNet replay), rows >= J from the restart
                const int J = std::atoi(a["--rollback-at"].c_str()), n0 = (int)std::min<size_t>(ids.size(), 12);
                model.forward(ids.data(), n0, 0, fo, lg.data());
                model.forward(&ids[J], (int)ids.size() - J, J, fo, lg.data() + (size_t)J * cfg.vocab_size);
            } else if (a.count("--chunk-logits")) {   // chunks of N tokens (hybrid models: the DeltaNet state is carried across the forwards)
                const int ch = std::max(1, std::atoi(a["--chunk-logits"].c_str()));
                for (size_t i = 0; i < ids.size(); i += ch) model.forward(&ids[i], (int)std::min<size_t>(ch, ids.size() - i), (int)i, fo, lg.data() + i * cfg.vocab_size);
            } else {
                model.forward(ids.data(), (int)ids.size(), 0, fo, lg.data());
            }
            std::ofstream o(a["--dump-logits"], std::ios::binary);
            o.write((const char*)lg.data(), lg.size() * 4);
            std::printf("{\"dumped\":%zu,\"vocab\":%d}\n", ids.size(), cfg.vocab_size);
            if (model.moe_turbo.active()) std::fprintf(stderr, "[moe-turbo] avg experts/token-layer %.3f over %ld decode tokens*layers\n", model.moe_avg_experts(), model.moe_stat_tokens());
            return 0;
        }
        if (a.count("--union-test")) {   // REST union head vs the full head on real windows: lb_gen --ids ... --screen-head 1 --union-test 1 [--ut-from 16 --ut-to 120 --ut-step 2 --ut-T 2,3,4]
            using std::vector;
            const int V = cfg.vocab_size;
            const int s0 = std::atoi(get("--ut-from", "16").c_str()), s1 = std::min((int)ids.size() - 8, std::atoi(get("--ut-to", "120").c_str())), st = std::atoi(get("--ut-step", "2").c_str());
            vector<int> Ts;
            { std::string q = get("--ut-T", "2,3,4"); size_t p0 = 0; while (p0 < q.size()) { size_t p1 = q.find(',', p0); if (p1 == std::string::npos) p1 = q.size(); Ts.push_back(std::atoi(q.substr(p0, p1 - p0).c_str())); p0 = p1 + 1; } }
            vector<float> pre((size_t)ids.size() * V);
            ForwardOpts po; po.logits_from = (int)ids.size();   // prefill only (no logits)
            model.kv().len = 0;
            for (int i = 0; i < (int)ids.size(); i += 64) { const int c = std::min(64, (int)ids.size() - i); ForwardOpts fo; fo.logits_from = c; model.forward(&ids[i], c, i, fo, nullptr); }
            long rows = 0, argmax_same = 0, vals_same = 0, vals_cmp = 0, cand_hit = 0, cand_total = 0, cand_ok = 0, passes_ok = 0, passes = 0, unions = 0;
            double uniq_sum[16] = {0}; int uniq_n[16] = {0};
            for (int T : Ts) for (int s = s0; s <= s1; s += st) {
                vector<float> lf((size_t)T * V), lu((size_t)T * V);
                ForwardOpts f; f.logits_from = 0; f.greedy = true; f.screen_ok = true;
                model.forward(&ids[s], T, s, f, lf.data());
                uniq_sum[T] += (double)model.unique_experts_last(); uniq_n[T]++;
                int cand[16];
                for (int t = 0; t < T; t++) {
                    const float* L = &lf[(size_t)t * V];
                    int b1 = argmax(L, V);
                    int b2 = -1;
                    for (int r = 0; r < V; r++) if (r != b1 && (b2 < 0 || L[r] > L[b2])) b2 = r;
                    cand[t] = (((s / st) + t) % 3 == 2) ? b2 : (t == T - 1 ? -1 : b1);   // right draft, wrong draft (runner-up), or none
                }
                ForwardOpts u = f; u.union_head = true; u.head_cand = cand;
                model.forward(&ids[s], T, s, u, lu.data());
                bool pass_ok = true;
                for (int t = 0; t < T; t++) {
                    const float* A = &lf[(size_t)t * V]; const float* B = &lu[(size_t)t * V];
                    const int a1 = argmax(A, V), b1 = argmax(B, V);
                    rows++; if (a1 == b1) argmax_same++; else pass_ok = false;
                    if (cand[t] >= 0) { cand_total++; if (cand[t] == a1) cand_hit++; if ((cand[t] == a1) == (cand[t] == b1)) cand_ok++; }
                    for (int r = 0; r < V; r++) if (B[r] > -INFINITY) { vals_cmp++; if (B[r] == A[r]) vals_same++; }
                }
                passes++; if (pass_ok) passes_ok++;
                unions++;
            }
            std::printf("{\"union_test\":{\"windows\":%ld,\"rows\":%ld,\"argmax_identical\":%ld,\"finite_values_bit_identical\":\"%ld/%ld\",\"draft_accept_decisions_identical\":\"%ld/%ld\",\"drafts_that_were_right\":%ld,\"windows_all_identical\":%ld,\"unique_experts_per_pass\":{", passes, rows, argmax_same, vals_same, vals_cmp, cand_ok, cand_total, cand_hit, passes_ok);
            bool fst = true; for (int T : Ts) { std::printf("%s\"T%d\":%.1f", fst ? "" : ",", T, uniq_n[T] ? uniq_sum[T] / uniq_n[T] : 0.0); fst = false; }
            std::printf("}}}\n");
            return rows == argmax_same ? 0 : 3;
        }
        if (a.count("--bench-batch")) {  // ms per forward for a batch of T tokens (speculative verify cost)
            using C = std::chrono::steady_clock;
            std::vector<float> lg((size_t)16 * cfg.vocab_size);
            std::vector<float> pre((size_t)ids.size() * cfg.vocab_size);
            model.kv().len = 0;
            model.forward(ids.data(), (int)ids.size(), 0, ForwardOpts{}, pre.data());
            model.kv().len = (int)ids.size();
            const int pos = (int)ids.size();
            if (lb::env_get("PROFILE")) { for (double& v : g_prof_ms) v = 0; for (double& v : g_prof2_ms) v = 0; }   // profile only the timed forwards
            std::printf("{\"batch_ms\":{");
            bool first = true;
            std::vector<int> tl = {1, 2, 3, 4, 5, 6, 8};
            if (a.count("--batch-list")) { tl.clear(); std::string q = a["--batch-list"]; size_t p0 = 0; while (p0 < q.size()) { size_t p1 = q.find(',', p0); if (p1 == std::string::npos) p1 = q.size(); tl.push_back(std::atoi(q.substr(p0, p1 - p0).c_str())); p0 = p1 + 1; } }
            for (int T : tl) {
                std::vector<int> tk(T, ids[0]);
                if (a.count("--bench-real")) for (int i = 0; i < T; i++) tk[i] = ids[(size_t)(i * 7 + 3) % ids.size()];   // distinct real tokens (realistic expert union) instead of one repeated id
                ForwardOpts fo; fo.logits_from = 0; fo.greedy = true;
                for (int w = 0; w < 3; w++) model.forward(tk.data(), T, pos, fo, lg.data());
                std::vector<double> ms;
                for (int r = 0; r < 7; r++) {
                    auto t0 = C::now();
                    for (int i = 0; i < 6; i++) model.forward(tk.data(), T, pos, fo, lg.data());
                    ms.push_back(std::chrono::duration<double>(C::now() - t0).count() / 6 * 1000.0);
                }
                std::sort(ms.begin(), ms.end());
                std::printf("%s\"%d\":%.2f,\"%d_min\":%.2f", first ? "" : ",", T, ms[3], T, ms[0]);
                first = false;
            }
            std::printf("}}\n");
            if (lb::env_get("PROFILE")) std::fprintf(stderr, "prof ms total: moe[router %.0f route %.0f gateup %.0f silu %.0f down %.0f combine %.0f] qkv %.0f attn %.0f o %.0f gate+up %.0f down %.0f head %.0f norm+rope+kv %.0f silu+resid %.0f | linear: prep %.0f gemv %.0f post %.0f\n", g_prof_ms[8], g_prof_ms[9], g_prof_ms[10], g_prof_ms[11], g_prof_ms[12], g_prof_ms[13], g_prof_ms[0], g_prof_ms[1], g_prof_ms[2], g_prof_ms[3], g_prof_ms[4], g_prof_ms[5], g_prof_ms[6], g_prof_ms[7], g_prof2_ms[0], g_prof2_ms[1], g_prof2_ms[2]);
            return 0;
        }
        if (a.count("--bench-draft")) {
            using C = std::chrono::steady_clock;
            std::vector<float> lg(cfg.vocab_size);
            auto run = [&](bool draft, bool skip) {
                ForwardOpts fo; fo.draft = draft; fo.skip_layers = skip; fo.logits_from = 0;
                const int n = 24, reps = 3;
                std::vector<float> pre((size_t)ids.size() * cfg.vocab_size);
                model.kv().len = 0;
                model.forward(ids.data(), (int)ids.size(), 0, ForwardOpts{}, pre.data());
                model.kv().len = (int)ids.size();  // commit prompt (forward does not touch len)
                const int pos = (int)ids.size();
                model.forward(&ids[0], 1, pos, fo, lg.data());  // warmup
                std::vector<double> ms;
                for (int r = 0; r < reps; r++) {
                    auto t0 = C::now();
                    for (int i = 0; i < n; i++) {
                        model.kv().len = pos + i;  // each step reads the same context
                        model.forward(&ids[0], 1, pos + i, fo, lg.data());
                    }
                    ms.push_back(std::chrono::duration<double>(C::now() - t0).count() / n * 1000.0);
                }
                std::sort(ms.begin(), ms.end());
                return ms;  // min, median, max
            };
            auto full = run(false, false), prim = run(true, false), skip = run(true, true);
            std::vector<int> sl;
            const int SL = std::atoi(get("--shortlist", "8192").c_str());
            for (int i = 0; i < SL && i < cfg.vocab_size; i++) sl.push_back(i);
            sl.insert(sl.end(), ids.begin(), ids.end());
            model.set_draft_vocab(sl);
            auto shortl = run(true, true);
            model.set_draft_vocab({});
            auto pr = [](const char* k, const std::vector<double>& v) {
                std::printf("\"%s\":{\"min\":%.2f,\"median\":%.2f,\"max\":%.2f},", k, v[0], v[1], v[2]);
            };
            std::printf("{\"ms_per_token\":{");
            pr("full", full); pr("primary_only", prim); pr("primary_layerskip", skip); pr("layerskip_shortlist", shortl);
            std::printf("\"draft_skip_layers\":%zu}}\n", model.draft_skip_mask().size() ? (size_t)std::count(model.draft_skip_mask().begin(), model.draft_skip_mask().end(), 1) : 0);
            return 0;
        }

        GenOptions o;
        o.max_new_tokens = std::atoi(get("--n", "32").c_str());
        o.sampling.temperature = (float)std::atof(get("--temp", "0").c_str());
        o.sampling.top_k = std::atoi(get("--top-k", "0").c_str());
        o.sampling.top_p = (float)std::atof(get("--top-p", "1").c_str());
        o.sampling.seed = (uint64_t)std::atoll(get("--seed", "0").c_str());
        o.spec.enabled = get("--spec", "1") == "1";
        o.spec.ngram = get("--ngram", "1") == "1";
        o.spec.layer_skip = get("--layer-skip", "1") == "1";
        o.spec.neural = get("--draft-mode", "auto") == "neural";  // auto = n-gram lookahead only
        o.spec.k_min = std::atoi(get("--kmin", "2").c_str());
        o.spec.k_max = std::atoi(get("--kmax", "8").c_str());
        o.spec.k_init = std::atoi(get("--k", "5").c_str());
        if (const int ln = std::atoi(get("--lookup-n", pareto ? std::to_string(ParetoMaxPreset::lookup_n).c_str() : "0").c_str()); ln > 0) {
            o.spec.enabled = o.spec.ngram = true;
            o.spec.k_max = ln;
            o.spec.k_min = std::min(o.spec.k_min, ln);
            o.spec.k_init = std::min(o.spec.k_init, ln);
        }
        o.spec.adaptive = get("--adaptive", "1") == "1";
        o.spec.mtp = get("--mtp", "0") != "0";   // --mtp 1: native MTP drafts (greedy, models with an MTP block); works with --spec 0 too
        o.spec.mtp_k = std::atoi(get("--mtp-k", "2").c_str());
        if (o.spec.mtp && !model.mtp_available()) std::fprintf(stderr, "mtp: model has no usable MTP block, plain decoding\n");
        if (get("--rest", "0") != "0") {   // --rest IDX_PATH (or "session" = session lookup + cost gate only) [--rest-k N] [--rest-min-gain X]
            o.spec.rest = true;
            o.spec.rest_k = std::atoi(get("--rest-k", "2").c_str());
            o.spec.rest_min_gain = (float)std::atof(get("--rest-min-gain", "1.30").c_str());
            if (get("--rest", "0") != "session" && !rest_load(get("--rest", "0"), &err)) { std::fprintf(stderr, "rest: %s\n", err.c_str()); return 1; }
        }
        SpecDecoder dec(model);
        struct TealPrint { Transformer& m; ~TealPrint() { if (m.teal_active()) std::fprintf(stderr, "%s\n", m.teal_report().c_str()); } } teal_print{model};
        std::vector<int> out;
        GenStats st = dec.run(ids, o, cfg.eos_token_ids, [&](int t) { out.push_back(t); return true; });
        std::printf("{\"tokens\":[");
        for (size_t i = 0; i < out.size(); i++) std::printf("%s%d", i ? "," : "", out[i]);
        std::printf("],\"isa\":\"%s\",\"threads\":%d,\"kv_bytes\":%zu,\"stats\":{\"prompt_tokens\":%d,\"generated_tokens\":%d,\"prefill_s\":%.5f,"
                    "\"decode_s\":%.5f,\"tokens_per_s\":%.3f,\"drafted\":%d,\"accepted\":%d,\"verify_passes\":%d,\"ngram_drafted\":%d,"
                    "\"ngram_accepted\":%d,\"finish_reason\":\"%s\"}}\n",
                    isa_name(isa), model.n_threads(), model.kv().bytes(), st.prompt_tokens, st.generated_tokens, st.prefill_s, st.decode_s,
                    st.tokens_per_s, st.drafted, st.accepted, st.verify_passes, st.ngram_drafted, st.ngram_accepted, st.finish_reason.c_str());
        if (o.spec.mtp) std::fprintf(stderr, "MTP_STATS {\"d1\":%d,\"a1\":%d,\"d2\":%d,\"a2\":%d,\"passes\":%d,\"generated\":%d,\"mtp_s\":%.4f,\"prefill_s\":%.4f,\"tok_s\":%.3f}\n",
                                     st.mtp_drafted[0], st.mtp_accepted[0], st.mtp_drafted[1], st.mtp_accepted[1], st.verify_passes, st.generated_tokens, st.mtp_s, st.prefill_s, st.tokens_per_s);
        if (o.spec.rest) std::fprintf(stderr, "REST_STATS {\"lookups\":%d,\"hits\":%d,\"static_passes\":%d,\"static_drafted\":%d,\"static_accepted\":%d,\"draft_s\":%.6f,\"passes\":%d,\"generated\":%d}\n",
                                      st.rest_lookups, st.rest_hits, st.rest_passes, st.rest_drafted, st.rest_accepted, st.rest_draft_s, st.verify_passes, st.generated_tokens);
        if (model.moe_turbo.active()) std::fprintf(stderr, "[moe-turbo] avg experts/token-layer %.3f over %ld decode tokens*layers\n", model.moe_avg_experts(), model.moe_stat_tokens());
#if defined(__linux__)
        {   // resident set (RSS) / peak, MB: weights are mmap'd zero-copy, so this is the touched working set
            std::ifstream st_("/proc/self/status");
            std::string ln_;
            long rss = 0, hwm = 0;
            while (std::getline(st_, ln_)) { if (ln_.rfind("VmRSS:", 0) == 0) rss = std::atol(ln_.c_str() + 6); else if (ln_.rfind("VmHWM:", 0) == 0) hwm = std::atol(ln_.c_str() + 6); }
            std::fprintf(stderr, "rss_mb=%ld peak_rss_mb=%ld\n", rss / 1024, hwm / 1024);
        }
#endif
        if (lb::env_get("PROFILE")) std::fprintf(stderr, "prof ms total: moe[router %.0f route %.0f gateup %.0f silu %.0f down %.0f combine %.0f] qkv %.0f attn %.0f o %.0f gate+up %.0f down %.0f head %.0f norm+rope+kv %.0f silu+resid %.0f | linear internals: prep(quant+LUT) %.0f gemv %.0f post %.0f\n", g_prof_ms[8], g_prof_ms[9], g_prof_ms[10], g_prof_ms[11], g_prof_ms[12], g_prof_ms[13], g_prof_ms[0], g_prof_ms[1], g_prof_ms[2], g_prof_ms[3], g_prof_ms[4], g_prof_ms[5], g_prof_ms[6], g_prof_ms[7], g_prof2_ms[0], g_prof2_ms[1], g_prof2_ms[2]);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
