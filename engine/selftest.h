#pragma once
#include <string>

namespace lb {
// Bit-exact check of every available ISA kernel against an independent +-1 oracle,
// including worst-case activations (all +-127) that stress the INT16 accumulators.
// Returns number of failures (0 = all exact). Appends a human-readable report.
int run_kernel_selftest(std::string& report);
}
