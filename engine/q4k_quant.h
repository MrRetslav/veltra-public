#pragma once
#include <cstdint>

namespace veltra {
// 256 weights -> one 148-byte Q4_K block of the engine's layout: d f16 | dmin f16 | sc[8] | mn[8] | qs[128] (engine/q2k_gemv.h).
void quant_q4k(const float* x, uint8_t* out);
}  // namespace veltra
