// engine/test_runner.h - the fixed GPU / engine test suites of the opt-in remote test mode.
//
// The suite table below is the allowlist: a remote job is only one of these ids. Titles, steps and estimates come from
// this compiled-in table, never from the relay. Every number in a report is measured in this process; what cannot run
// in this build is reported as "skipped" with the reason.
#pragma once
#include <atomic>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "engine.h"

namespace veltra::tests {

struct StepInfo {
    const char* id;
    const char* name;   // Dutch, shown in the consent popup
};

struct SuiteInfo {
    const char* id;
    const char* title;
    int est_minutes;
    std::vector<StepInfo> steps;
    std::vector<int> decode_ctx;   // decode_sweep context sizes
    bool quality_subset;           // quality_gate runs only the quick subset
};

const std::vector<SuiteInfo>& suites();
const SuiteInfo* find_suite(std::string_view id);   // nullptr = not on the allowlist

struct Progress {
    int step = 0, total = 0;   // 1-based step index
    std::string name;          // Dutch step name
    std::string message;
    double tok_s = -1;         // last measured speed, < 0 = none
};
using ProgressFn = std::function<void(const Progress&)>;

// Everything the runner may use. The caller holds the server's generation lock for the whole suite.
struct EngineContext {
    lb::Engine* engine = nullptr;   // nullptr = no model loaded
    std::string model_path;         // used to run the CUDA selftest and to scrub it from messages; never reported
    std::string model_file;         // file name only
    std::string model_id;
    std::string device;             // the device this build really runs the model on, e.g. "CPU AVX2"
    int threads = 0;
    int max_context = 0;            // smallest of the engine's KV capacity and the model's own limit
    bool spec_default = true, rest = false, mtp = false;
    std::function<lb::GenOptions(int max_new_tokens)> chat_options;   // the options of a temperature-0 chat request
};

struct StepSummary {
    std::string id, name, status, summary;   // status: pass | fail | skipped | error
};

struct Report {
    std::string status = "error";   // done | cancelled | error
    std::string headline;
    std::vector<StepSummary> steps;
    std::string json;               // the full report sent to the relay
};

Report run_suite(const SuiteInfo& suite, EngineContext& ctx, const std::atomic<bool>& cancel, const ProgressFn& progress);

// The quality_gate cases and their checks (exposed so they can be compared with scripts/run_sanity10.py and
// scripts/eval_raptor.py on identical outputs).
std::vector<std::string> quality_case_ids();
bool quality_check(const std::string& id, const std::string& output, std::string* reason);

}  // namespace veltra::tests
