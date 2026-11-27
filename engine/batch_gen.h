// lb_gen --batch B: B independent greedy streams through Transformer::forward_batch (weight-stationary batched decode) + stats + exactness check vs B=1.
#pragma once
#include <map>
#include <string>

#include "model_config.h"
#include "transformer.h"

namespace lb {

// args (all optional except --prompt-files): --batch B --prompt-files a.ids,b.ids,... --prompt-len L (default 480) --n N (default 64) --batch-fast -1|0|1
// --verify 1 (re-run every stream alone at T=1 and compare the token sequences) --print-tokens 1 --screen-head 1 (only used when a step has one stream) --head-mode N (Transformer::head_mode)
int run_batch_gen(Transformer& model, const ModelConfig& cfg, const std::map<std::string, std::string>& args);

}  // namespace lb
