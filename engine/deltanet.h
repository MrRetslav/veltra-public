// Gated DeltaNet kernels (Qwen3.5 / Qwen3.6 hybrid models, docs/qwen36_lbt_contract.md section 0.7).
// One value head, one token:   S [dk][dv] f32 (row-major, dv contiguous)
//   kvm[j]   = sum_i Sin[i][j] * k[i]
//   delta[j] = (v[j] - a * kvm[j]) * beta                      (a = exp(g), the decay of this token)
//   Sout[i][j] = fma(k[i], delta[j], a * Sin[i][j])             (== HF: S = S*a + k (x) delta, with the decay applied first)
//   o[j]     = sum_i Sout[i][j] * q[i]
// Sin may equal Sout (in place). Every summation runs in a fixed order, so the result does not depend on how many tokens are processed per call
// (decode T=1 == verify chain == prefill) and a replay of logged (k, delta, a) reproduces the state bit for bit.
#pragma once
#include <string>

namespace lb {

void dn_head_step(int dk, int dv, const float* Sin, float* Sout, const float* q, const float* k, const float* v, float a, float beta,
                  float* delta, float* o);
// S = fma(k, delta, a * S), in place: exactly the state update of dn_head_step (used to roll the state forward from a log after a rejected draft).
void dn_head_replay(int dk, int dv, float* S, const float* k, const float* delta, float a);
// Independent double-precision reference of one step (tests).
void dn_head_step_ref(int dk, int dv, const double* Sin, double* Sout, const double* q, const double* k, const double* v, double a, double beta, double* o);

// Numerical checks of the kernels (scalar/AVX2 vs double reference, in-place == out-of-place, replay == step state). Returns failures.
int run_deltanet_selftest(std::string& report);

}  // namespace lb
