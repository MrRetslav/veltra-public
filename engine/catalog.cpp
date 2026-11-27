// engine/catalog.cpp - Marketplace data for GET /api/catalog: published model scores and facts, plus a speed figure per
// model for THIS machine. Scores are copied from the official Hugging Face model cards (fetched
// 2026-11-26); -1 means the card prints none, and nothing is filled in or derived. The makers measure differently
// (thinking on/off, benchmark versions), which every entry states in `mode`.
// Speed: a measurement from <app data>/local_benchmarks.json when this machine has one for the model ("measured"),
// else an estimate from this machine's own memory read bandwidth and the bytes a token touches ("estimate").
#include "catalog.h"

#include "json.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace veltra::market {
namespace {

const ModelMeta kMeta[] = {
    // id, name, source, total_b, active_b, bits/weight, head_gb, moe, mtp, license, commercial, languages, tags,
    // code, code_bench, reasoning, reasoning_bench, knowledge, knowledge_bench, tools, tools_bench, mode
    {"qwen3-0.6b", "Qwen3-0.6B", "Qwen/Qwen3-0.6B", 0.75, 0.6, 4.6, 0.16, false, false, "Apache 2.0", true, "NL + EN", "Allround",
     -1, "", -1, "", -1, "", -1, "", "de modelkaart noemt geen scores"},
    {"qwen3-1.7b", "Qwen3-1.7B", "Qwen/Qwen3-1.7B", 2.0, 1.75, 4.6, 0.31, false, false, "Apache 2.0", true, "NL + EN", "Allround",
     -1, "", -1, "", -1, "", -1, "", "de modelkaart noemt geen scores"},
    {"qwen3-4b", "Qwen3-4B", "Qwen/Qwen3-4B", 4.0, 3.6, 4.6, 0.39, false, false, "Apache 2.0", true, "NL + EN", "Allround",
     -1, "", -1, "", -1, "", -1, "", "de modelkaart noemt geen scores"},
    {"qwen3-8b", "Qwen3-8B", "Qwen/Qwen3-8B", 8.2, 7.6, 4.6, 0.62, false, false, "Apache 2.0", true, "NL + EN", "Allround",
     -1, "", -1, "", -1, "", -1, "", "de modelkaart noemt geen scores"},
    {"qwen3-30b-a3b", "Qwen3-30B-A3B-Instruct-2507", "Qwen/Qwen3-30B-A3B-Instruct-2507", 30.5, 3.3, 3.5, 0.31, true, false, "Apache 2.0", true,
     "NL + EN", "Allround", 43.2, "LiveCodeBench v6", 61.3, "AIME25", 70.4, "GPQA", 65.1, "BFCL-v3", "gemeten zonder thinking"},
    {"qwen3.6-35b-a3b", "Qwen3.6-35B-A3B", "Qwen/Qwen3.6-35B-A3B", 35.0, 3.0, 3.9, 0.39, true, true, "Apache 2.0", true, "NL + EN",
     "Allround,Code", 80.4, "LiveCodeBench v6", 92.7, "AIME 2026", 86.0, "GPQA", 67.2, "TAU3-Bench", "meetwijze per rij niet vermeld (standaard met thinking)"},
    {"qwen3.5-9b", "Qwen3.5-9B", "Qwen/Qwen3.5-9B", 9.0, 9.0, 4.6, 0.62, false, false, "Apache 2.0", true, "NL + EN", "Allround",
     65.6, "LiveCodeBench v6", -1, "", 81.7, "GPQA Diamond", 64.5, "IFBench", "meetwijze per rij niet vermeld"},
    {"qwen2.5-coder-7b", "Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct", 7.6, 7.6, 4.6, 0.55, false, false, "Apache 2.0", true,
     "Alleen EN", "Code", -1, "", -1, "", -1, "", -1, "", "de modelkaart noemt geen scores"},
    {"gemma-4-26b-a4b", "Gemma 4 26B-A4B", "google/gemma-4-26B-A4B-it", 25.2, 3.8, 4.6, 0.6, true, false, "Apache 2.0", true, "NL + EN", "Allround",
     77.1, "LiveCodeBench v6", 88.3, "AIME 2026 zonder tools", 82.3, "GPQA Diamond", 68.2, "Tau2", "thinking-instelling niet vermeld"},
    {"gpt-oss-20b", "GPT-OSS-20B", "openai/gpt-oss-20b", 21.0, 3.6, 4.6, 0.6, true, false, "Apache 2.0", true, "niet vermeld", "Allround",
     -1, "", -1, "", 58.6, "GPQA Diamond", -1, "", "meetwijze niet vermeld"},
    {"qwen3-235b-a22b", "Qwen3-235B-A22B-Instruct-2507", "Qwen/Qwen3-235B-A22B-Instruct-2507", 235.0, 22.0, 4.6, 0.62, true, false, "Apache 2.0",
     true, "NL + EN", "Allround", 51.8, "LiveCodeBench v6", 70.3, "AIME25", 77.5, "GPQA", 70.9, "BFCL-v3", "gemeten zonder thinking"},
};

std::string jq(const std::string& s) {
    std::string o;
    lb::json::Value::escape_to(o, s);
    return o;
}
std::string num(double v, int prec = 1) {
    if (v < 0) return "null";
    char b[32];
    std::snprintf(b, sizeof b, "%.*f", prec, v);
    return b;
}
std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// Sequential read speed of this machine's memory with all threads (GB/s). Measured once, ~0.3 s, on first use.
double memory_bandwidth_gbs(bool engine_busy) {
    static std::mutex mu;
    static double cached = 0.0;
    std::lock_guard<std::mutex> lk(mu);
    if (cached > 0 || engine_busy) return cached;
    const size_t words = (192u << 20) / 8;
    std::unique_ptr<uint64_t[]> buf(new uint64_t[words]);
    for (size_t i = 0; i < words; i++) buf[i] = i;
    const int threads = std::max(1, std::min(16, static_cast<int>(std::thread::hardware_concurrency()) / 2));
    std::atomic<uint64_t> sink{0};
    double best = 0;
    for (int pass = 0; pass < 4; pass++) {
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; t++)
            pool.emplace_back([&, t] {
                uint64_t a = 0, b = 0, c = 0, d = 0;
                const size_t lo = words * t / threads, hi = words * (t + 1) / threads;
                for (size_t i = lo; i + 4 <= hi; i += 4) { a += buf[i]; b += buf[i + 1]; c += buf[i + 2]; d += buf[i + 3]; }
                sink += a + b + c + d;
            });
        for (auto& th : pool) th.join();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (s > 0) best = std::max(best, words * 8 / 1e9 / s);
    }
    cached = best > 0 && sink.load() != 1 ? best : 10.0;
    return cached;
}

}  // namespace

const ModelMeta* find(const std::string& id) {
    for (const ModelMeta& m : kMeta)
        if (id == m.id) return &m;
    return nullptr;
}
const ModelMeta* all(size_t* n) {
    *n = sizeof kMeta / sizeof kMeta[0];
    return kMeta;
}

double file_estimate_gib(const ModelMeta& m) { return m.total_b * 1e9 * m.bits / 8.0 / 1073741824.0 + m.head_gb; }

Speed speeds_for(const std::vector<const ModelMeta*>& models, const std::string& bench_history_json, bool engine_busy) {
    Speed out;
    out.bandwidth_gbs = memory_bandwidth_gbs(engine_busy);
    // bytes a decoded token touches: the active weights plus the output head. 1.25: fitted on the reference box (Ryzen 9
    // 3900, 26-27.5 GB/s measured), where the uncalibrated estimate is 5-10% below the measured Qwen3-30B-A3B (20.6) and
    // Qwen3-1.7B (26.1) and about 25% above the hybrid Qwen3.6-35B-A3B (14.1). CPUs that are compute-bound (thin laptops)
    // land below it until one own benchmark rescales the estimates.
    auto raw = [&](const ModelMeta& m) { return out.bandwidth_gbs * 1.25 / (m.active_b * m.bits / 8.0 + m.head_gb); };
    out.measured.assign(models.size(), -1.0);
    out.measured_ttft.assign(models.size(), -1.0);
    lb::json::Value h;
    if (lb::json::parse(bench_history_json, &h) && h.is_array()) {
        for (const lb::json::Value& run : h.as_array()) {   // oldest first: a later run of a model wins
            const std::string name = lower(run["model"].as_string(""));
            const double tps = run["decode_tok_s"].as_double(-1);
            if (name.empty() || tps <= 0) continue;
            for (size_t i = 0; i < models.size(); i++) {
                const std::string id = models[i]->id, src = lower(models[i]->source);
                const size_t slash = src.find('/');
                if (name == id || name == src || name == src.substr(slash + 1)) {
                    out.measured[i] = tps;
                    out.measured_ttft[i] = run["ttft_s"].as_double(-1);
                }
            }
        }
    }
    // when this machine has measurements, the estimates for the other models are scaled by how far off those were
    double ratio = 0;
    int n = 0;
    for (size_t i = 0; i < models.size(); i++)
        if (out.measured[i] > 0 && out.bandwidth_gbs > 0) { ratio += out.measured[i] / raw(*models[i]); n++; }
    out.calibration = n ? std::max(0.2, std::min(2.0, ratio / n)) : 1.0;
    out.calibrated_from = n;
    out.estimate.resize(models.size());
    for (size_t i = 0; i < models.size(); i++) out.estimate[i] = out.bandwidth_gbs > 0 ? raw(*models[i]) * out.calibration : -1.0;
    return out;
}

// The metadata members of one catalog object (starting with a comma).
std::string meta_fields(const ModelMeta& m, const Speed& s, size_t i) {
    const double v[4] = {m.code, m.reasoning, m.knowledge, m.tools};
    const char* names[4] = {"code", "reasoning", "knowledge", "tools_json"};
    const char* bench[4] = {m.code_bench, m.reasoning_bench, m.knowledge_bench, m.tools_bench};
    double sum = 0;
    int have = 0;
    for (double x : v)
        if (x >= 0) { sum += x; have++; }
    std::string o = ",\"scores\":{\"overall_iq\":" + num(have >= 3 ? sum / have : -1);
    o += ",\"iq_columns\":" + std::to_string(have);
    for (int k = 0; k < 4; k++) o += std::string(",\"") + names[k] + "\":" + num(v[k]) + ",\"" + names[k] + "_bench\":" + jq(bench[k]);
    o += ",\"mode\":" + jq(m.mode) + "}";
    o += ",\"total_params_b\":" + num(m.total_b, 2) + ",\"active_params_b\":" + num(m.active_b, 2);
    o += std::string(",\"is_moe\":") + (m.moe ? "true" : "false") + ",\"supports_mtp\":" + (m.mtp ? "true" : "false");
    o += ",\"license\":" + jq(m.license) + ",\"commercial\":" + (m.commercial ? "true" : "false") + ",\"languages\":" + jq(m.languages);
    o += ",\"best_for\":[";
    std::string tags = m.tags;
    for (size_t a = 0, first = 1; a < tags.size(); first = 0) {
        size_t b = tags.find(',', a);
        if (b == std::string::npos) b = tags.size();
        o += (first ? "" : ",") + jq(tags.substr(a, b - a));
        a = b + 1;
    }
    o += "],\"card_url\":" + jq(std::string("https://huggingface.co/") + m.source);
    const double measured = i < s.measured.size() ? s.measured[i] : -1, est = i < s.estimate.size() ? s.estimate[i] : -1;
    // time to first token for a 500-token prompt: prefill runs about 4.5x as fast as decoding on the reference box
    o += ",\"est_tok_s\":" + num(est) + ",\"est_ttft_s\":" + num(est > 0 ? 500.0 / (4.5 * est) : -1, 2);
    o += ",\"measured_tok_s\":" + num(measured) + ",\"measured_ttft_s\":" + num(i < s.measured_ttft.size() ? s.measured_ttft[i] : -1, 2);
    return o;
}

}  // namespace veltra::market
