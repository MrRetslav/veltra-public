#include "deltanet.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#include "ops.h"

namespace lb {

#if defined(__x86_64__) || defined(_M_X64)
// dv % 32 == 0. Columns in chunks of 32 (4 ymm accumulators); rows sequential, so the order of every sum is fixed.
__attribute__((target("avx2,fma")))
static void step_avx2(int dk, int dv, const float* Sin, float* Sout, const float* q, const float* k, const float* v, float a, float beta, float* delta, float* o) {
    const __m256 va = _mm256_set1_ps(a), vb = _mm256_set1_ps(beta);
    for (int jc = 0; jc < dv; jc += 32) {
        __m256 c0 = _mm256_setzero_ps(), c1 = c0, c2 = c0, c3 = c0;
        const float* s = Sin + jc;
        for (int i = 0; i < dk; i++, s += dv) {
            const __m256 kb = _mm256_set1_ps(k[i]);
            c0 = _mm256_fmadd_ps(_mm256_loadu_ps(s), kb, c0);
            c1 = _mm256_fmadd_ps(_mm256_loadu_ps(s + 8), kb, c1);
            c2 = _mm256_fmadd_ps(_mm256_loadu_ps(s + 16), kb, c2);
            c3 = _mm256_fmadd_ps(_mm256_loadu_ps(s + 24), kb, c3);
        }
        const __m256 d0 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(v + jc), _mm256_mul_ps(va, c0)), vb);
        const __m256 d1 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(v + jc + 8), _mm256_mul_ps(va, c1)), vb);
        const __m256 d2 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(v + jc + 16), _mm256_mul_ps(va, c2)), vb);
        const __m256 d3 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(v + jc + 24), _mm256_mul_ps(va, c3)), vb);
        _mm256_storeu_ps(delta + jc, d0); _mm256_storeu_ps(delta + jc + 8, d1); _mm256_storeu_ps(delta + jc + 16, d2); _mm256_storeu_ps(delta + jc + 24, d3);
        __m256 o0 = _mm256_setzero_ps(), o1 = o0, o2 = o0, o3 = o0;
        const float* si = Sin + jc;
        float* so = Sout + jc;
        for (int i = 0; i < dk; i++, si += dv, so += dv) {
            const __m256 kb = _mm256_set1_ps(k[i]), qb = _mm256_set1_ps(q[i]);
            const __m256 n0 = _mm256_fmadd_ps(kb, d0, _mm256_mul_ps(va, _mm256_loadu_ps(si)));
            const __m256 n1 = _mm256_fmadd_ps(kb, d1, _mm256_mul_ps(va, _mm256_loadu_ps(si + 8)));
            const __m256 n2 = _mm256_fmadd_ps(kb, d2, _mm256_mul_ps(va, _mm256_loadu_ps(si + 16)));
            const __m256 n3 = _mm256_fmadd_ps(kb, d3, _mm256_mul_ps(va, _mm256_loadu_ps(si + 24)));
            _mm256_storeu_ps(so, n0); _mm256_storeu_ps(so + 8, n1); _mm256_storeu_ps(so + 16, n2); _mm256_storeu_ps(so + 24, n3);
            o0 = _mm256_fmadd_ps(n0, qb, o0); o1 = _mm256_fmadd_ps(n1, qb, o1); o2 = _mm256_fmadd_ps(n2, qb, o2); o3 = _mm256_fmadd_ps(n3, qb, o3);
        }
        _mm256_storeu_ps(o + jc, o0); _mm256_storeu_ps(o + jc + 8, o1); _mm256_storeu_ps(o + jc + 16, o2); _mm256_storeu_ps(o + jc + 24, o3);
    }
}

__attribute__((target("avx2,fma")))
static void replay_avx2(int dk, int dv, float* S, const float* k, const float* delta, float a) {
    const __m256 va = _mm256_set1_ps(a);
    for (int i = 0; i < dk; i++) {
        const __m256 kb = _mm256_set1_ps(k[i]);
        float* s = S + (size_t)i * dv;
        for (int j = 0; j < dv; j += 8)
            _mm256_storeu_ps(s + j, _mm256_fmadd_ps(kb, _mm256_loadu_ps(delta + j), _mm256_mul_ps(va, _mm256_loadu_ps(s + j))));
    }
}
#endif

static bool simd_ok(int dv) {
#if defined(__x86_64__) || defined(_M_X64)
    return dv % 32 == 0 && cpu_has_f16c_fma();
#else
    (void)dv;
    return false;
#endif
}

void dn_head_step(int dk, int dv, const float* Sin, float* Sout, const float* q, const float* k, const float* v, float a, float beta, float* delta, float* o) {
#if defined(__x86_64__) || defined(_M_X64)
    if (simd_ok(dv)) { step_avx2(dk, dv, Sin, Sout, q, k, v, a, beta, delta, o); return; }
#endif
    for (int j = 0; j < dv; j++) {
        float acc = 0.f;
        for (int i = 0; i < dk; i++) acc += Sin[(size_t)i * dv + j] * k[i];
        delta[j] = (v[j] - a * acc) * beta;
        o[j] = 0.f;
    }
    for (int i = 0; i < dk; i++) {
        const float* si = Sin + (size_t)i * dv;
        float* so = Sout + (size_t)i * dv;
        for (int j = 0; j < dv; j++) {
            const float nv = std::fmaf(k[i], delta[j], a * si[j]);
            so[j] = nv;
            o[j] += nv * q[i];
        }
    }
}

void dn_head_replay(int dk, int dv, float* S, const float* k, const float* delta, float a) {
#if defined(__x86_64__) || defined(_M_X64)
    if (simd_ok(dv)) { replay_avx2(dk, dv, S, k, delta, a); return; }
#endif
    for (int i = 0; i < dk; i++) {
        float* s = S + (size_t)i * dv;
        for (int j = 0; j < dv; j++) s[j] = std::fmaf(k[i], delta[j], a * s[j]);
    }
}

void dn_head_step_ref(int dk, int dv, const double* Sin, double* Sout, const double* q, const double* k, const double* v, double a, double beta, double* o) {
    std::vector<double> S((size_t)dk * dv), kv(dv, 0.0), delta(dv);
    for (size_t i = 0; i < S.size(); i++) S[i] = Sin[i] * a;                       // HF order: decay first
    for (int i = 0; i < dk; i++) for (int j = 0; j < dv; j++) kv[j] += S[(size_t)i * dv + j] * k[i];
    for (int j = 0; j < dv; j++) delta[j] = (v[j] - kv[j]) * beta;
    for (int i = 0; i < dk; i++) for (int j = 0; j < dv; j++) S[(size_t)i * dv + j] += k[i] * delta[j];
    for (int j = 0; j < dv; j++) { o[j] = 0; for (int i = 0; i < dk; i++) o[j] += S[(size_t)i * dv + j] * q[i]; }
    for (size_t i = 0; i < S.size(); i++) Sout[i] = S[i];
}

int run_deltanet_selftest(std::string& report) {
    std::mt19937_64 rng(777);
    std::normal_distribution<float> nd(0.f, 1.f);
    int checks = 0, failures = 0;
    char buf[256];
    const int shapes[][2] = {{128, 128}, {64, 128}, {128, 64}, {16, 32}, {40, 48}, {128, 256}};
    for (auto& sh : shapes) {
        const int dk = sh[0], dv = sh[1];
        const int T = 12;
        std::vector<float> S((size_t)dk * dv), S2, S3;
        std::vector<double> Sd((size_t)dk * dv);
        for (size_t i = 0; i < S.size(); i++) { S[i] = 0.3f * nd(rng); Sd[i] = S[i]; }
        S2 = S; S3 = S;
        std::vector<float> q(dk), k(dk), v(dv), delta(dv), o(dv), o2(dv), delta2(dv);
        std::vector<double> qd(dk), kd(dk), vd(dv), od(dv);
        std::vector<float> lk((size_t)T * dk), ld((size_t)T * dv), la(T);   // log of the chain for the replay check
        double worst = 0, worst_s = 0;
        bool inplace_ok = true;
        std::vector<float> Sout = S;   // out-of-place for the first token, in place afterwards
        const float* sin = S.data();
        for (int t = 0; t < T; t++) {
            for (int i = 0; i < dk; i++) { k[i] = nd(rng) / std::sqrt((float)dk); q[i] = nd(rng) / std::sqrt((float)dk); }
            for (int j = 0; j < dv; j++) v[j] = nd(rng);
            const float a = std::exp(-0.7f * (float)std::fabs(nd(rng))), beta = 1.f / (1.f + std::exp(-nd(rng)));
            for (int i = 0; i < dk; i++) { kd[i] = k[i]; qd[i] = q[i]; }
            for (int j = 0; j < dv; j++) vd[j] = v[j];
            dn_head_step(dk, dv, sin, Sout.data(), q.data(), k.data(), v.data(), a, beta, delta.data(), o.data());
            // the same step in place on S2 must be bit-identical to the out-of-place result
            dn_head_step(dk, dv, S2.data(), S2.data(), q.data(), k.data(), v.data(), a, beta, delta2.data(), o2.data());
            if (std::memcmp(o.data(), o2.data(), (size_t)dv * 4) != 0 || std::memcmp(Sout.data(), S2.data(), Sout.size() * 4) != 0) inplace_ok = false;
            sin = Sout.data();
            std::vector<double> Sdn((size_t)dk * dv);
            dn_head_step_ref(dk, dv, Sd.data(), Sdn.data(), qd.data(), kd.data(), vd.data(), a, beta, od.data());
            Sd = Sdn;
            double en = 0, nr = 0;
            for (int j = 0; j < dv; j++) { en += (o[j] - od[j]) * (o[j] - od[j]); nr += od[j] * od[j]; }
            worst = std::max(worst, std::sqrt(en / (nr + 1e-30)));
            double es = 0, ns = 0;
            for (size_t i = 0; i < Sd.size(); i++) { es += (Sout[i] - Sd[i]) * (Sout[i] - Sd[i]); ns += Sd[i] * Sd[i]; }
            worst_s = std::max(worst_s, std::sqrt(es / (ns + 1e-30)));
            std::memcpy(&lk[(size_t)t * dk], k.data(), (size_t)dk * 4); std::memcpy(&ld[(size_t)t * dv], delta.data(), (size_t)dv * 4); la[t] = a;
        }
        // replay from the initial state reproduces the state bit for bit
        for (int t = 0; t < T; t++) dn_head_replay(dk, dv, S3.data(), &lk[(size_t)t * dk], &ld[(size_t)t * dv], la[t]);
        const bool replay_ok = std::memcmp(S3.data(), Sout.data(), Sout.size() * 4) == 0;
        const bool num_ok = worst < 2e-5 && worst_s < 2e-5;
        checks += 3;
        if (!inplace_ok) { failures++; std::snprintf(buf, sizeof buf, "FAIL deltanet in-place != out-of-place dk=%d dv=%d\n", dk, dv); report += buf; }
        if (!replay_ok) { failures++; std::snprintf(buf, sizeof buf, "FAIL deltanet replay != step state dk=%d dv=%d\n", dk, dv); report += buf; }
        if (!num_ok) { failures++; std::snprintf(buf, sizeof buf, "FAIL deltanet vs double dk=%d dv=%d out %.3g state %.3g\n", dk, dv, worst, worst_s); report += buf; }
    }
    std::snprintf(buf, sizeof buf, "deltanet selftest: %d checks, %d failures\n", checks, failures);
    report += buf;
    return failures;
}

}  // namespace lb
