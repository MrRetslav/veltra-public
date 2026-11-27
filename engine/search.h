// Batched test-time search (--search-n): the prompt is prefilled ONCE into the model's own KV cache; N branches share it (Transformer::fork_kv_slice),
// sample independently (reproducible: seed base_seed + branch) and decode together in one Transformer::forward_batch per step, so the weights are
// streamed once per step. The blind verifier prunes broken branches mid-stream and picks the answer (engine/search_verifier.h).
#pragma once
#include <functional>
#include <vector>

#include "engine.h"
#include "tokenizer.h"
#include "transformer.h"

namespace lb {

struct SearchRun {
    std::vector<BranchResult> branches;
    SearchVerdict verdict;
    GenStats stats;                        // generated_tokens / finish_reason describe the winner
    std::vector<int> committed;            // tokens whose K/V the main kv() holds afterwards (the prompt)
    bool cancelled = false;
};

// `reuse` prompt tokens are already in the main kv() (prefix cache). eos ids and opt.stop_tokens end a branch.
SearchRun run_search(Transformer& model, const Tokenizer& tok, const std::vector<int>& prompt, const GenOptions& opt, const std::vector<int>& eos_ids,
                     int reuse = 0);

// veltra-selftest --model M: shared-prefix branch slices decoded with forward_batch (B = 1, 4, 8) must give logits bit-identical to every branch
// running alone with one token per forward. Returns the number of failing batch sizes.
int run_shared_prefix_selftest(const std::string& model_path, int prompt_len, int steps, std::string& report);

}  // namespace lb
