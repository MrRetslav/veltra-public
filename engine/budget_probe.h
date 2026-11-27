// LBT-50 output-sensitive error budget over layers with the existing kernels (veltra-gen --budget-probe): measures, on teacher-forced text, how much
// the model's output distribution changes when ONE layer reads fewer routed experts, builds a per-layer allocation from those sensitivities and then
// measures the combined allocation end to end against the uniform choice with exactly the same number of expert reads.
#pragma once
#include <map>
#include <string>

#include "engine.h"
#include "transformer.h"

namespace lb {

// args: --budget-fit a.ids,b.ids  --budget-held c.ids,d.ids  [--budget-ks 6,4]  [--budget-out file.json]
int run_budget_probe(Transformer& model, const ModelConfig& cfg, const std::map<std::string, std::string>& args);

// Shadow-expert variants against the full model on the same held-out positions (veltra-gen --shadow-variants).
// args: --budget-held c.ids,d.ids  --shadow-variants k:6,s:4:/path/masks.bin,..  [--budget-out file.json]
int run_shadow_probe(Transformer& model, const ModelConfig& cfg, const std::map<std::string, std::string>& args);

}  // namespace lb
