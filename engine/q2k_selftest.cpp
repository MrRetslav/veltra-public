// Self-test of the `.qk` K-quant kernels (q2k_gemv.*): AVX2 vs scalar reference vs an independent float dequantisation, T=1 vs batched bit-identity,
// group ranges, and the linear_forward / linear_forward_jobs wiring (lbt_model.cpp).
#include "q2k_selftest.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "lb_common.h"
#include "lbt_model.h"
#include "q2k_gemv.h"

namespace lb {

static void rand_matrix(int type, int rows, int n, std::mt19937& rng, std::vector<uint8_t>& w) {
    const int SB = kQkSbBytes[type], nsb = n / kQkSuper;
    w.resize(qk_bytes(type, rows, n));
    for (auto& x : w) x = (uint8_t)rng();
    for (size_t i = 0; i < (size_t)rows * nsb; i++) {
        uint8_t* p = w.data() + i * SB;
        const uint16_t d = f32_to_f16(0.002f + (rng() % 1000) * 1e-5f), dm = f32_to_f16(0.002f + (rng() % 1000) * 1e-5f);
        if (type == QK_Q2K) { std::memcpy(p + 80, &d, 2); std::memcpy(p + 82, &dm, 2); }
        else if (type == QK_Q3K) std::memcpy(p + 112, &d, 2);
        else { std::memcpy(p, &d, 2); std::memcpy(p + 2, &dm, 2); for (int j = 0; j < 16; j++) p[4 + j] &= 63; }
    }
}

int run_qk_selftest(std::string& report) {
    int fails = 0;
    char buf[256];
    std::mt19937 rng(42);
    const char* tn[3] = {"Q2_K", "Q3_K", "Q4_K"};
    const bool avx2 = isa_available(Isa::Avx2);
    for (int type = 0; type < 3; type++) {
        for (int n : {256, 768, 2048}) {
            const int rows = 64, nsb = n / kQkSuper, SB = kQkSbBytes[type];
            std::vector<uint8_t> w;
            rand_matrix(type, rows, n, rng, w);
            const int E = 7;
            std::vector<int8_t> q((size_t)E * n);
            std::vector<int16_t> b((size_t)E * n / 16);
            for (int e = 0; e < E; e++) {
                // slot 0: worst case (+-127 patterns), others random
                for (int i = 0; i < n; i++) q[(size_t)e * n + i] = e == 0 ? (int8_t)((i & 1) ? 127 : -127) : (int8_t)((int)(rng() % 255) - 127);
                qk_sums16(&q[(size_t)e * n], n, &b[(size_t)e * n / 16]);
            }
            QkView v;
            v.type = type; v.rows = rows; v.nsb = nsb; v.w = w.data();
            std::vector<float> ys((size_t)E * rows, 0.f), ya((size_t)E * rows, 0.f), y1((size_t)E * rows, 0.f), yr((size_t)E * rows, 0.f);
            qk_scalar(v, q.data(), n, b.data(), n / 16, E, ys.data(), rows, 0, rows / 4);
            for (int e = 0; e < E; e++)
                for (int g = 0; g < rows / 4; g++)
                    for (int r = 0; r < 4; r++) {
                        double acc = 0;
                        float dq[256];
                        for (int t = 0; t < nsb; t++) {
                            qk_dequant_row_sb(type, w.data() + (((size_t)g * nsb + t) * 4 + r) * SB, dq);
                            for (int i = 0; i < 256; i++) acc += (double)dq[i] * q[(size_t)e * n + t * 256 + i];
                        }
                        yr[(size_t)e * rows + g * 4 + r] = (float)acc;
                    }
            double nrm = 0, es = 0;
            for (size_t i = 0; i < ys.size(); i++) { nrm = std::fmax(nrm, std::fabs(yr[i])); es = std::fmax(es, std::fabs(ys[i] - yr[i])); }
            if (!(es <= 2e-4 * nrm)) { fails++; std::snprintf(buf, sizeof buf, "FAIL qk %s n=%d scalar vs dequant err %g (max %g)\n", tn[type], n, es, nrm); report += buf; }
            if (avx2) {
                // split group ranges must not matter, nor must the number of slots in a pass
                qk_avx2(v, q.data(), n, b.data(), n / 16, E, ya.data(), rows, 0, 5);
                qk_avx2(v, q.data(), n, b.data(), n / 16, E, ya.data(), rows, 5, rows / 4);
                for (int e = 0; e < E; e++) qk_avx2(v, q.data() + (size_t)e * n, n, b.data() + (size_t)e * n / 16, n / 16, 1, y1.data() + (size_t)e * rows, rows, 0, rows / 4);
                double ea = 0;
                for (size_t i = 0; i < ya.size(); i++) ea = std::fmax(ea, std::fabs(ya[i] - ys[i]));
                if (!(ea <= 2e-4 * nrm)) { fails++; std::snprintf(buf, sizeof buf, "FAIL qk %s n=%d avx2 vs scalar err %g (max %g)\n", tn[type], n, ea, nrm); report += buf; }
                if (std::memcmp(ya.data(), y1.data(), ya.size() * 4) != 0) { fails++; std::snprintf(buf, sizeof buf, "FAIL qk %s n=%d batched != single-slot bits\n", tn[type], n); report += buf; }
                // weight-stationary prefill kernel: same integers, float order differs -> equal to the scalar reference within float rounding, for slot counts 1..E and split row ranges
                for (int Ew : {1, 3, E}) {
                    std::vector<float> yw((size_t)Ew * rows, 0.f);
                    qk_avx2_wst(v, q.data(), n, b.data(), n / 16, Ew, yw.data(), rows, 0, 6);
                    qk_avx2_wst(v, q.data(), n, b.data(), n / 16, Ew, yw.data(), rows, 6, rows / 4);
                    double ew = 0;
                    for (size_t i = 0; i < yw.size(); i++) ew = std::fmax(ew, std::fabs(yw[i] - ys[i]));
                    if (!(ew <= 2e-4 * nrm)) { fails++; std::snprintf(buf, sizeof buf, "FAIL qk %s n=%d wst E=%d vs scalar err %g (max %g)\n", tn[type], n, Ew, ew, nrm); report += buf; }
                }
                {   // per-superblock activation scales: slot e scaled by 0.5 * (e + 1) must scale the unscaled result by exactly that factor (float rounding only)
                    std::vector<float> ss((size_t)E * nsb), yss((size_t)E * rows, 0.f), y0((size_t)E * rows, 0.f);
                    for (int e = 0; e < E; e++) for (int t = 0; t < nsb; t++) ss[(size_t)e * nsb + t] = 0.5f * (float)(e + 1);
                    qk_avx2_wst(v, q.data(), n, b.data(), n / 16, E, y0.data(), rows, 0, rows / 4);
                    qk_avx2_wst(v, q.data(), n, b.data(), n / 16, E, yss.data(), rows, 0, rows / 4, ss.data());
                    double es2 = 0;
                    for (int e = 0; e < E; e++) for (int r = 0; r < rows; r++) es2 = std::fmax(es2, std::fabs(yss[(size_t)e * rows + r] - 0.5 * (e + 1) * y0[(size_t)e * rows + r]));
                    if (!(es2 <= 1e-5 * nrm * E)) { fails++; std::snprintf(buf, sizeof buf, "FAIL qk %s n=%d wst activation tile scales err %g (max %g)\n", tn[type], n, es2, nrm); report += buf; }
                }
            }
        }
    }
    // linear_forward / linear_forward_jobs: T=1 vs T=5 bit-identical, close to float dequant x x
    for (int type = 0; type < 3; type++) {
        const int rows = 128, n = 1024, nsb = n / kQkSuper, SB = kQkSbBytes[type];
        std::vector<uint8_t> w;
        rand_matrix(type, rows, n, rng, w);
        LbLinear lin;
        lin.m = rows; lin.n = n; lin.qk.type = type; lin.qk.nsb = nsb; lin.qk.w = w.data();
        const int T = 5;
        std::vector<float> x((size_t)T * n);
        for (auto& f : x) f = ((int)(rng() % 2001) - 1000) * 1e-3f;
        for (int i = 0; i < n; i += 97) x[(size_t)2 * n + i] *= 40.f;      // outlier token -> second activation plane
        std::vector<float> yb((size_t)T * rows), y1((size_t)T * rows), yj((size_t)T * rows * 2);
        LinearScratch s;
        const Isa isa = best_isa();
        linear_forward(lin, isa, x.data(), T, yb.data(), true, s, nullptr);
        for (int t = 0; t < T; t++) linear_forward(lin, isa, x.data() + (size_t)t * n, 1, y1.data() + (size_t)t * rows, true, s, nullptr);
        LinJob jb;
        jb.nw = 2; jb.w[0] = &lin; jb.w[1] = &lin; jb.x = x.data(); jb.T = T; jb.y[0] = yj.data(); jb.y[1] = yj.data() + (size_t)T * rows;
        linear_forward_jobs(&jb, 1, isa, s, nullptr);
        if (std::memcmp(yb.data(), y1.data(), yb.size() * 4) != 0) { fails++; std::snprintf(buf, sizeof buf, "FAIL qk %s linear_forward T=%d vs T=1 not bit-identical\n", tn[type], T); report += buf; }
        if (std::memcmp(yb.data(), yj.data(), yb.size() * 4) != 0 || std::memcmp(yb.data(), yj.data() + (size_t)T * rows, yb.size() * 4) != 0) { fails++; std::snprintf(buf, sizeof buf, "FAIL qk %s linear_forward_jobs != linear_forward\n", tn[type]); report += buf; }
        double nrm = 0, e = 0;
        for (int t = 0; t < T; t++)
            for (int r = 0; r < rows; r++) {
                double acc = 0;
                const int g = r / 4, rr = r % 4;
                float dq[256];
                for (int k = 0; k < nsb; k++) {
                    qk_dequant_row_sb(type, w.data() + (((size_t)g * nsb + k) * 4 + rr) * SB, dq);
                    for (int i = 0; i < 256; i++) acc += (double)dq[i] * x[(size_t)t * n + k * 256 + i];
                }
                nrm = std::fmax(nrm, std::fabs(acc));
                e = std::fmax(e, std::fabs(acc - yb[(size_t)t * rows + r]));
            }
        if (!(e <= 0.02 * nrm)) { fails++; std::snprintf(buf, sizeof buf, "FAIL qk %s linear_forward vs float err %g (max %g)\n", tn[type], e, nrm); report += buf; }
    }
    // Prefill GEMM path (LinearScratch::fast_prefill + tile_act, as the transformer sets for T >= 16): linear_forward_jobs over several
    // expert-like jobs (M = 40 / 17 / 5 routed tokens, so >= 16, the sub_tokens() split and M < 8 are all covered) against an fp32/fp64
    // reference matmul (qk_dequant_row_sb weights x the FLOAT activations). Rigorous tolerance from the per-256-tile int8 activation
    // quantisation: |err(t, r)| <= sum_i |w_ri| * s_tile(i) / 2 (round to nearest, s = amax / 127), plus float rounding.
    for (int type = 0; type < 3; type++) {
        const int rows = 64, n = type == QK_Q2K ? 1024 : 768, nsb = n / kQkSuper, SB = kQkSbBytes[type];
        std::vector<uint8_t> w;
        rand_matrix(type, rows, n, rng, w);
        LbLinear lin;
        lin.m = rows; lin.n = n; lin.qk.type = type; lin.qk.nsb = nsb; lin.qk.w = w.data();
        std::vector<float> dq((size_t)rows * n);
        for (int r = 0; r < rows; r++)
            for (int t = 0; t < nsb; t++) qk_dequant_row_sb(type, w.data() + (((size_t)(r / 4) * nsb + t) * 4 + r % 4) * SB, &dq[(size_t)r * n + (size_t)t * 256]);
        const int Ts[3] = {40, 17, 5};
        int Ttot = 0;
        for (int T : Ts) Ttot += T;
        std::vector<float> x((size_t)Ttot * n);
        for (auto& f : x) f = ((int)(rng() % 2001) - 1000) * 1e-3f;
        for (int t = 0; t < Ttot; t += 3) x[(size_t)t * n + (size_t)(rng() % n)] *= 30.f;   // outliers coarsen only their own tile
        std::vector<float> y((size_t)Ttot * rows * 2, 0.f);
        LinJob jb[3];
        for (int j = 0, off = 0; j < 3; off += Ts[j], j++) {
            jb[j].nw = 2; jb[j].w[0] = &lin; jb[j].w[1] = &lin; jb[j].x = x.data() + (size_t)off * n; jb[j].T = Ts[j];
            jb[j].y[0] = y.data() + (size_t)off * rows; jb[j].y[1] = y.data() + (size_t)(Ttot + off) * rows;
        }
        LinearScratch s;
        s.fast_prefill = true; s.tile_act = true;
        linear_forward_jobs(jb, 3, best_isa(), s, nullptr);
        double worst = 0, nrm = 0;
        for (int t = 0; t < Ttot; t++) {
            float st[16];
            for (int b = 0; b < nsb; b++) {
                float am = 0;
                for (int i = 0; i < 256; i++) am = std::fmax(am, std::fabs(x[(size_t)t * n + b * 256 + i]));
                st[b] = am / 127.f;
            }
            for (int r = 0; r < rows; r++) {
                double ref = 0, bound = 0, mag = 0;
                for (int i = 0; i < n; i++) {
                    const double wv = dq[(size_t)r * n + i], xv = x[(size_t)t * n + i];
                    ref += wv * xv; bound += std::fabs(wv) * 0.5 * st[i / 256]; mag += std::fabs(wv * xv);
                }
                const double tol = 1.001 * bound + 1e-5 * mag + 1e-6;
                nrm = std::fmax(nrm, std::fabs(ref));
                for (int c = 0; c < 2; c++) worst = std::fmax(worst, std::fabs(y[(size_t)(c * Ttot + t) * rows + r] - ref) / tol);
            }
        }
        if (!(worst <= 1.0) || nrm == 0) { fails++; std::snprintf(buf, sizeof buf, "FAIL qk %s prefill GEMM (tile, M=40/17/5) vs fp32 reference: err/bound %.3f\n", tn[type], worst); report += buf; }
        if (std::memcmp(y.data(), y.data() + (size_t)Ttot * rows, (size_t)Ttot * rows * 4) != 0) { fails++; std::snprintf(buf, sizeof buf, "FAIL qk %s prefill GEMM: identical weights in one job differ\n", tn[type]); report += buf; }
    }
    std::snprintf(buf, sizeof buf, "qk selftest (Q2_K/Q3_K/Q4_K, %s): %s\n", avx2 ? "scalar+avx2" : "scalar", fails ? "FAILED" : "OK");
    report += buf;
    return fails;
}

}  // namespace lb
