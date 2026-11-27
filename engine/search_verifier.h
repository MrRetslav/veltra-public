// Blind verifier for batched test-time search (--search-n). It only ever sees the generated branches and the format the REQUEST asked for
// (response_format / tools). It has no access to eval prompts' expected answers, test keys or benchmark patterns: every check here is generic
// (JSON syntax, JSON schema, tool-call shape, repetition, model log-probability, agreement between branches).
#pragma once
#include <string>
#include <vector>

namespace lb {

enum class OutputFormat { Free = 0, JsonObject = 1, JsonSchema = 2, ToolCall = 3 };

struct VerifySpec {
    OutputFormat format = OutputFormat::Free;
    std::string schema_json;                // JsonSchema: the schema itself
    std::vector<std::string> tool_names;    // ToolCall: names the request declared (empty = any name)
};

// State of a text that is meant to become one JSON value: Invalid can never be repaired by appending text.
enum class JsonState { Invalid, Incomplete, Complete };
// Scans one JSON value starting at s[pos]; on Complete *end is the position after it.
JsonState scan_json(const std::string& s, size_t pos, size_t* end);

// True when the tail of the token stream repeats one period (>= 4 times and >= 48 tokens).
bool token_loop(const std::vector<int>& ids);
// Mid-stream check: a reason to stop this branch now, or "" to let it continue.
std::string prune_reason(const std::vector<int>& ids, const std::string& text, const VerifySpec& spec);

struct BranchResult {
    std::vector<int> ids;          // generated tokens (without the stop token)
    std::string text;
    double cum_logprob = 0;        // sum of the model's log-probabilities (temperature 1) of the sampled tokens, stop token included
    int token_count = 0;           // tokens that contributed to cum_logprob
    bool finished = false;
    bool pruned = false;
    std::string finish_reason;     // "stop" | "length" | "pruned"
    // filled by judge_branches
    bool valid = false;
    std::string outcome;           // normalised final result used for self-consistency ("" = none)
    std::string why;               // why invalid
    double score = 0;
};

struct SearchVerdict {
    int winner = 0;
    int branches_evaluated = 0, valid_branches = 0, pruned_branches = 0;
    double consensus_ratio = 0;    // share of the valid branches that agree with the winner's outcome (winner included)
};

constexpr double kInvalidScore = -1e9;
constexpr double kConsensusBonus = 0.15;

std::string normalized_outcome(const std::string& text);
bool check_format(const std::string& text, const VerifySpec& spec, std::string* why);
// score = cum_logprob / max(1, token_count) + 0.15 per OTHER valid branch with the same outcome; format violations and pruned branches get -1e9.
SearchVerdict judge_branches(std::vector<BranchResult>& branches, const VerifySpec& spec);
// Deterministic self-check of the generic rules (used by veltra-selftest). Returns the number of failures.
int search_verifier_selftest(std::string* report);

}  // namespace lb
