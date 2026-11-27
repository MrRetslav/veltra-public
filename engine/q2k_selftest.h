#pragma once
#include <string>

namespace lb {
// `.qk` K-quant kernels: scalar vs AVX2 vs float dequant, T=1 vs batched bit-identity, linear_forward(_jobs) wiring. Returns the number of failures.
int run_qk_selftest(std::string& report);
}
