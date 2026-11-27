// lb_gen --exit-probe L1,L2,..: early-exit / layer-skip diagnostic. During a normal greedy decode (full-model tokens drive the sequence, teacher forced) the hidden
// state after layer L is passed through output_norm + head for every listed L and compared with the full model's argmax. No speculative decoder is involved.
#pragma once
#include <map>
#include <string>

#include "model_config.h"
#include "transformer.h"

namespace lb {

// args: --exit-probe 14,16,20,24,32  --prompt-files a.ids,b.ids,..  [--n 100 tokens per prompt] [--label name] [--calib 100 tokens for the rescale variant].
// The 4-bit screening head is enabled automatically (needs an int8 output head). Prints one JSON line
// {"label":..,"n":..,"layers":[{"L":..,"full":{..},"screen":{..},"resc":{..},"base_same_tokens":{..}},..]}; the last entry is L = num_layers (sanity).
int run_exit_probe(Transformer& model, const ModelConfig& cfg, const std::map<std::string, std::string>& args);

}  // namespace lb
