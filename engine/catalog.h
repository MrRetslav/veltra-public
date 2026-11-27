// engine/catalog.h - Marketplace data (see catalog.cpp).
#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace veltra::market {

struct ModelMeta {
    const char* id;
    const char* name;
    const char* source;      // Hugging Face repository
    double total_b, active_b, bits, head_gb;
    bool moe, mtp;
    const char* license;
    bool commercial;
    const char* languages;
    const char* tags;        // comma separated
    double code;
    const char* code_bench;
    double reasoning;
    const char* reasoning_bench;
    double knowledge;
    const char* knowledge_bench;
    double tools;
    const char* tools_bench;
    const char* mode;        // how the makers measured
};

struct Speed {
    double bandwidth_gbs = 0, calibration = 1.0;
    int calibrated_from = 0;                          // number of this machine's own measurements used
    std::vector<double> estimate, measured, measured_ttft;   // per model, -1 = none
};

const ModelMeta* find(const std::string& id);
const ModelMeta* all(size_t* n);
double file_estimate_gib(const ModelMeta& m);         // size of a Q4_K file of a model Veltra cannot run (yet)
// engine_busy: do not measure the memory bandwidth now (it would slow the answer and measure too low); until it has been
// measured once there are no estimates.
Speed speeds_for(const std::vector<const ModelMeta*>& models, const std::string& bench_history_json, bool engine_busy);
std::string meta_fields(const ModelMeta& m, const Speed& s, size_t index);

}  // namespace veltra::market
