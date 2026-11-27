// lb_gen --dump-aux L1,L2,L3 --dump-aux-out PREFIX: EAGLE-3 probe dump (diagnostic, default off, engine/aux_dump.cpp). Greedy decode (T=1 forwards also for the prompt, so every position
// has its own capture) of real prompts; writes the residual stream after L layers (L in 1..num_layers) for every position, the token ids and the full-head argmax per position.
#pragma once
#include <map>
#include <string>

#include "model_config.h"
#include "transformer.h"

namespace lb {

// args: --dump-aux 2,24,45 --dump-aux-out PREFIX --prompt-files a.ids,b.ids [--n 150 new tokens] (or --ids). Per prompt <PREFIX>_<name>.aux.npy = float32 [N][NL][H] (N = prompt + generated tokens)
// and <PREFIX>_<name>.json = {"ids":[N tokens],"n_prompt":P,"layers":[..],"argmax":[N; target argmax after position j (full int8 head, no screening), -1 for j < P-1]}.
int run_aux_dump(Transformer& model, const ModelConfig& cfg, const std::map<std::string, std::string>& args);

}  // namespace lb
