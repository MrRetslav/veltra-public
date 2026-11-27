#pragma once
#include <string>

#include "engine.h"
#include "lbt_model.h"

namespace lb {
// Fills ModelConfig from the .lbt meta JSON (+ header draft mask when the meta has no list).
bool parse_model_config(const LbtFile& f, ModelConfig* cfg, std::string* err);
}  // namespace lb
