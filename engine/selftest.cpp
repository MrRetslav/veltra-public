#include "selftest.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "binary_gemv.h"
#include "i3_gemv.h"
#include "lb_common.h"
#include "nibble_gemv.h"
#include "ops.h"

namespace lb {

int run_kernel_selftest(std::string& report) {
    std::mt19937_64 rng(12345);
    int failures = 0, checks = 0;
    const Isa isas[] = {Isa::Scalar, Isa::Avx2, Isa::Avx512, Isa::Neon};
    const int shapes[][2] = {{64, 128}, {1, 1}, {5, 37}, {70, 129}, {128, 256}, {200, 1000}, {512, 300}, {257, 4096}};
    char buf[256];

    for (auto& sh : shapes) {
        const int rows = sh[0], cols = sh[1];
        const size_t words = bits_words_per_row(cols);
        for (int mode = 0; mode < 3; mode++) {  // 0 random, 1 worst-case +127 all-plus, 2 worst-case alternating
            std::vector<uint64_t> bits((size_t)rows * words, 0);
            for (int r = 0; r < rows; r++)
                for (int k = 0; k < cols; k++) {
                    bool plus = mode == 1 ? true : (rng() & 1);
                    if (plus) bits[(size_t)r * words + (k >> 6)] |= 1ull << (k & 63);
                }
            std::vector<uint8_t> packed(packed_weight_bytes(rows, cols));
            pack_bits_to_lut(bits.data(), rows, cols, packed.data());

            // pack/unpack round trip
            std::vector<uint64_t> back((size_t)rows * words);
            unpack_lut_to_bits(packed.data(), rows, cols, back.data());
            checks++;
            if (back != bits) {
                failures++;
                std::snprintf(buf, sizeof buf, "FAIL pack roundtrip rows=%d cols=%d mode=%d\n", rows, cols, mode);
                report += buf;
            }

            const int T = 1 + (int)(rng() % kMaxBatch);
            const int kp = pad_k(cols), rp = pad_rows(rows);
            std::vector<int8_t> q((size_t)T * kp, 0);
            std::vector<int8_t> luts((size_t)T * lut_bytes_per_token(kp));
            for (int t = 0; t < T; t++) {
                for (int k = 0; k < cols; k++) {
                    int v = mode == 0 ? (int)(rng() % 255) - 127 : (mode == 1 ? 127 : ((k & 1) ? 127 : -127));
                    q[(size_t)t * kp + k] = (int8_t)v;
                }
                build_luts(&q[(size_t)t * kp], kp, &luts[(size_t)t * lut_bytes_per_token(kp)]);
            }
            std::vector<int32_t> expect((size_t)T * rp, 0);
            for (int t = 0; t < T; t++) binary_gemv_direct(bits.data(), rows, cols, &q[(size_t)t * kp], &expect[(size_t)t * rp]);

            BinMatView view;
            view.rows = rows; view.cols = cols; view.lut_w = packed.data(); view.bits = bits.data();
            std::vector<uint8_t> w512(packed.size());
            repack_lut_avx512(packed.data(), rows, cols, w512.data());
            view.w512 = w512.data();
            std::vector<int8_t> luts512((size_t)T * lut_bytes_per_token(kp));
            for (int t = 0; t < T; t++) build_luts(&q[(size_t)t * kp], kp, &luts512[(size_t)t * lut_bytes_per_token(kp)], true);
            for (Isa isa : isas) {
                if (!isa_available(isa)) continue;
                std::vector<int32_t> got((size_t)T * rp, 0x7eadbeef);
                binary_gemv(isa, view, isa == Isa::Avx512 ? luts512.data() : luts.data(), T, got.data());
                bool ok = true;
                for (int t = 0; t < T && ok; t++)
                    for (int r = 0; r < rows; r++)
                        if (got[(size_t)t * rp + r] != expect[(size_t)t * rp + r]) {
                            ok = false;
                            std::snprintf(buf, sizeof buf, "FAIL %s rows=%d cols=%d T=%d mode=%d t=%d row=%d got=%d want=%d\n",
                                          isa_name(isa), rows, cols, T, mode, t, r, got[(size_t)t * rp + r], expect[(size_t)t * rp + r]);
                            report += buf;
                            break;
                        }
                checks++;
                if (!ok) failures++;
            }
        }
    }

    // ---- bit-plane grid kernels (fused P planes x tiles x group scales) vs a direct +-1 oracle; integer gs -> exact in float
    for (int rows : {64, 70, 200})
        for (int ntile : {1, 3, 5})
            for (int P : {2, 3, 4}) {
                const int E = 1 + (int)(rng() % kMaxBatch), rp = pad_rows(rows), cols = ntile * kTileK;
                const size_t words = bits_words_per_row(kTileK), tb = (size_t)rp * 16;
                std::vector<std::vector<uint64_t>> bits((size_t)P * ntile);  // [p][tile] canonical bits of [rows, 128]
                std::vector<std::vector<uint8_t>> lutw(P), w512(P);
                for (int p = 0; p < P; p++) {
                    lutw[p].resize((size_t)ntile * tb);
                    w512[p].resize((size_t)ntile * tb);
                    for (int t = 0; t < ntile; t++) {
                        auto& b = bits[(size_t)p * ntile + t];
                        b.assign((size_t)rows * words, 0);
                        for (auto& v : b) v = (rows == 70 && ntile == 3) ? ~0ull : rng();   // worst case: all +1 (int16 tile accumulators at their max)
                        pack_bits_to_lut(b.data(), rows, kTileK, &lutw[p][(size_t)t * tb]);
                        repack_lut_avx512(&lutw[p][(size_t)t * tb], rows, kTileK, &w512[p][(size_t)t * tb]);
                    }
                }
                std::vector<float> gs((size_t)ntile * rp), gs2(gs.size());
                for (auto& v : gs) v = (float)(1 + rng() % 4);
                for (int t = 0; t < ntile; t++)
                    for (int rb = 0; rb < rp / kRowBlock; rb++)
                        for (int s = 0; s < kRowBlock; s++) gs2[(size_t)t * rp + rb * kRowBlock + s] = gs[(size_t)t * rp + rb * kRowBlock + grid_avx2_perm(s)];
                std::vector<int8_t> q((size_t)E * cols);
                for (auto& v : q) v = (rows == 70 && ntile == 3) ? (int8_t)127 : (int8_t)((int)(rng() % 255) - 127);
                std::vector<float> want((size_t)E * rp, 0.f);
                for (int e = 0; e < E; e++)
                    for (int t = 0; t < ntile; t++) {
                        std::vector<int32_t> comb(rows, 0), sp(rows);
                        for (int p = 0; p < P; p++) {
                            binary_gemv_direct(bits[(size_t)p * ntile + t].data(), rows, kTileK, &q[(size_t)e * cols + (size_t)t * kTileK], sp.data());
                            for (int r = 0; r < rows; r++) comb[r] = comb[r] * 2 + sp[r];
                        }
                        for (int r = 0; r < rows; r++) want[(size_t)e * rp + r] += (float)comb[r] * gs[(size_t)t * rp + r];
                    }
                for (Isa isa : isas) {
                    if (!isa_available(isa) || isa == Isa::Neon) continue;
                    const bool l512 = isa == Isa::Avx512;
                    std::vector<int8_t> luts((size_t)E * ntile * kTileLutBytes);
                    for (int e = 0; e < E; e++)
                        for (int t = 0; t < ntile; t++)
                            build_luts(&q[(size_t)e * cols + (size_t)t * kTileK], kTileK, &luts[((size_t)t * E + e) * kTileLutBytes], l512);
                    GridView gv;
                    gv.rows = rows; gv.ntile = ntile; gv.P = P; gv.gs_stride = rp;
                    for (int p = 0; p < P; p++) gv.w[p] = l512 ? w512[p].data() : lutw[p].data();
                    gv.gs = isa == Isa::Avx2 ? gs2.data() : gs.data();
                    std::vector<float> got((size_t)E * rp, -1.f);
                    grid_gemv(isa, gv, luts.data(), E, got.data());
                    bool ok = true;
                    for (int e = 0; e < E && ok; e++)
                        for (int r = 0; r < rows; r++)
                            if (got[(size_t)e * rp + r] != want[(size_t)e * rp + r]) {
                                ok = false;
                                std::snprintf(buf, sizeof buf, "FAIL grid %s rows=%d ntile=%d P=%d E=%d e=%d row=%d got=%g want=%g\n", isa_name(isa), rows, ntile, P, E, e, r,
                                              got[(size_t)e * rp + r], want[(size_t)e * rp + r]);
                                report += buf;
                                break;
                            }
                    checks++;
                    if (!ok) failures++;
                }
            }

    // ---- LUT builder (all layouts) against a straightforward reference
    for (int layout = 0; layout < 2; layout++) {
        const int kp = 256, groups = kp / 4;
        std::vector<int8_t> q(kp), lut(lut_bytes_per_token(kp), 0x55);
        for (auto& v : q) v = (int8_t)((int)(rng() % 255) - 127);
        q[0] = q[1] = q[2] = q[3] = 127; q[4] = q[5] = q[6] = q[7] = -127;
        if (layout == 1) {
#if defined(__aarch64__) || defined(_M_ARM64)
            continue;  // NEON uses the plain layout only
#endif
        }
        build_luts(q.data(), kp, lut.data(), layout == 1);
        bool ok = true;
        for (int g = 0; g < groups && ok; g++)
            for (int p = 0; p < 16 && ok; p++) {
                int t = 0;
                for (int j = 0; j < 4; j++) t += ((p >> j) & 1) ? q[4 * g + j] : -q[4 * g + j];
                const uint16_t u = (uint16_t)(int16_t)t;
                size_t lo_off = (size_t)g * 32 + p, hi_off = (size_t)g * 32 + 16 + p;
                if (layout == 1) {
                    const size_t base = (size_t)(g >> 3) * 256 + (size_t)(g & 1) * 128 + (size_t)((g >> 1) & 3) * 16;
                    lo_off = base + p; hi_off = base + 64 + p;
                }
                if ((uint8_t)lut[lo_off] != (u & 0xff) || (uint8_t)lut[hi_off] != (u >> 8)) ok = false;
            }
        checks++;
        if (!ok) { failures++; std::snprintf(buf, sizeof buf, "FAIL build_luts layout=%d\n", layout); report += buf; }
    }
    // ---- exact-head screening kernel, SiLU, activation quantiser
    for (int H : {64, 2048, 2560}) {
        const int rows = 37, np = 1 + (int)(rng() % 9);
        std::vector<int8_t> w((size_t)rows * H);
        std::vector<uint8_t> xu((size_t)np * H);
        for (auto& v : w) v = (int8_t)((int)(rng() % 255) - 127);
        for (auto& v : xu) v = (uint8_t)(1 + rng() % 255);
        std::vector<int32_t> got((size_t)rows * np), want((size_t)rows * np);
        head_dots_i8(w.data(), rows, H, xu.data(), np, got.data());
        for (int r = 0; r < rows; r++)
            for (int p = 0; p < np; p++) {
                int32_t a = 0;
                for (int i = 0; i < H; i++) a += (int32_t)xu[(size_t)p * H + i] * (int32_t)w[(size_t)r * H + i];
                want[(size_t)r * np + p] = a;
            }
        checks++;
        if (got != want) { failures++; std::snprintf(buf, sizeof buf, "FAIL head_dots_i8 H=%d np=%d\n", H, np); report += buf; }
    }
    {
        const int n = 1003;
        std::vector<float> g(n), u(n), o(n);
        for (int i = 0; i < n; i++) { g[i] = ((int)(rng() % 4001) - 2000) / 100.f; u[i] = ((int)(rng() % 2001) - 1000) / 500.f; }
        silu_mul(g.data(), u.data(), n, o.data());
        double worst = 0;
        for (int i = 0; i < n; i++) {
            const double ref = (double)g[i] / (1.0 + std::exp(-(double)g[i])) * u[i];
            worst = std::max(worst, std::fabs(o[i] - ref) / (std::fabs(ref) + 1e-6));
        }
        checks++;
        if (worst > 2e-5) { failures++; std::snprintf(buf, sizeof buf, "FAIL silu_mul rel err %.3g\n", worst); report += buf; }
        std::vector<float> x(n);
        for (auto& v : x) v = ((int)(rng() % 20001) - 10000) / 77.f;
        std::vector<int8_t> q(1024 + 8, 99);
        const float sc = quantize_activations_i8(x.data(), n, 1024, q.data());
        float amax = 0; for (float v : x) amax = std::max(amax, std::fabs(v));
        bool ok = std::fabs(sc - amax / 127.f) < 1e-6f;
        for (int i = 0; i < n && ok; i++) ok = (int)q[i] == (int)std::nearbyint(x[i] * (127.f / amax)) || std::fabs(x[i] * (127.f / amax) - std::floor(x[i] * (127.f / amax)) - 0.5f) < 1e-3f;
        for (int i = n; i < 1024 && ok; i++) ok = q[i] == 0;
        checks++;
        if (!ok) { failures++; report += "FAIL quantize_activations_i8\n"; }
    }
    // ---- nibble-INT4 kernels (.n4 layout) vs an independent scalar oracle built straight from the level matrix j[r][c] (no n4 layout in the oracle)
    {
        char nb[320];
        const int shapes_n[][2] = {{64, 1}, {1, 1}, {70, 3}, {200, 5}, {130, 16}, {192, 48}};
        for (const auto& sh : shapes_n)
            for (int mode = 0; mode < 4; mode++) {  // 0 random, 1 all j=15 & +127, 2 all j=15 & alternating +-127, 3 random j & -127/+127 random
                const int rows = sh[0], nt = sh[1], n = nt * 128, rp = pad_rows(rows), nrb = rp / 64;
                const int E = mode == 0 ? 1 + (int)(rng() % 12) : 1 + (int)(rng() % 3);
                std::vector<uint8_t> j((size_t)rows * n), w((size_t)nrb * nt * 4096, 0);
                std::vector<float> gs((size_t)rows * nt), gsk((size_t)nt * rp, 0.f);
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < n; c++) j[(size_t)r * n + c] = (mode == 1 || mode == 2) ? 15 : (uint8_t)(rng() % 16);
                for (auto& g : gs) g = ((int)(rng() % 2001) - 1000) / 256.f;   // exact binary fractions
                for (int r = 0; r < rows; r++)
                    for (int t = 0; t < nt; t++) {
                        gsk[((size_t)(r / 64) * nt + t) * 64 + r % 64] = 0.5f * gs[(size_t)r * nt + t];
                        for (int i = 0; i < 64; i++)
                            w[((size_t)(r / 64) * nt + t) * 4096 + (size_t)(r % 64) * 64 + i] = (uint8_t)(j[(size_t)r * n + t * 128 + i] | (j[(size_t)r * n + t * 128 + 64 + i] << 4));
                    }
                std::vector<int8_t> q((size_t)E * n);
                std::vector<int32_t> A((size_t)E * nt, 0);
                for (int e = 0; e < E; e++)
                    for (int c = 0; c < n; c++) {
                        int v = mode == 0 ? (int)(rng() % 255) - 127 : mode == 1 ? 127 : mode == 2 ? ((c & 1) ? 127 : -127) : ((rng() & 1) ? 127 : -127);
                        q[(size_t)e * n + c] = (int8_t)v;
                        A[(size_t)e * nt + c / 128] += v;
                    }
                std::vector<float> ssv((size_t)E * nt, 1.f);
                const bool use_ss = mode == 3 || (mode == 0 && (E & 1));   // per-tile activation scales
                if (use_ss) for (auto& x : ssv) x = (1 + (int)(rng() % 63)) / 64.f;
                std::vector<double> want((size_t)E * rows);
                for (int e = 0; e < E; e++)
                    for (int r = 0; r < rows; r++) {
                        double acc = 0;
                        for (int t = 0; t < nt; t++) {
                            long long ssum = 0;
                            for (int c = 0; c < 128; c++) ssum += (long long)j[(size_t)r * n + t * 128 + c] * q[(size_t)e * n + t * 128 + c];
                            acc += (double)gs[(size_t)r * nt + t] * (double)ssv[(size_t)e * nt + t] * ((double)ssum - 7.5 * A[(size_t)e * nt + t]);
                        }
                        want[(size_t)e * rows + r] = acc;
                    }
                NibView nv; nv.rows = rows; nv.ntile = nt; nv.w = w.data(); nv.gs = gsk.data();
                for (Isa isa : isas) {
                    if (!isa_available(isa) || isa == Isa::Neon) continue;
                    std::vector<float> got((size_t)E * rp, -777.f);
                    nibble_gemm(isa, nv, q.data(), (size_t)n, A.data(), E, got.data(), (size_t)rp, 0, nrb, use_ss ? ssv.data() : nullptr);
                    bool ok = true;
                    for (int e = 0; e < E && ok; e++)
                        for (int r = 0; r < rows; r++) {
                            double tol = 2e-5 * (std::fabs(want[(size_t)e * rows + r]) + 15.0 * 127.0 * 128.0 * nt * 4.0) + 1e-3;
                            if (!(std::fabs(got[(size_t)e * rp + r] - want[(size_t)e * rows + r]) <= tol)) {
                                ok = false;
                                std::snprintf(nb, sizeof nb, "FAIL nibble %s rows=%d nt=%d E=%d mode=%d e=%d row=%d got=%g want=%g\n", isa_name(isa), rows, nt, E, mode, e, r, got[(size_t)e * rp + r], want[(size_t)e * rows + r]);
                                report += nb;
                                break;
                            }
                        }
                    for (int e = 0; e < E && ok; e++)
                        for (int r = rows; r < rp; r++)
                            if (got[(size_t)e * rp + r] != 0.f) { ok = false; report += "FAIL nibble padding rows not zero\n"; break; }
                    checks++;
                    if (!ok) failures++;
                }
            }
    }

    // ---- INT3 g128 asymmetric (.i3 layout) vs an independent double oracle built from the level matrix q[r][c] and the f16 scale / min (no i3 layout in the oracle)
    {
        char nb[320];
        const int shapes_i[][2] = {{64, 2}, {128, 4}, {64, 16}, {192, 8}, {64, 32}};
        for (const auto& sh : shapes_i)
            for (int mode = 0; mode < 4; mode++) {   // 0 random, 1 all q=7 & +127, 2 all q=7 & alternating +-127, 3 random q & -127/+127 random
                const int rows = sh[0], nt = sh[1], n = nt * 128;
                const int E = mode == 0 ? 1 + (int)(rng() % 12) : 1 + (int)(rng() % 3);
                std::vector<uint8_t> qv((size_t)rows * n), w(i3_bytes(rows, n));
                std::vector<float> sc((size_t)rows * nt), mn((size_t)rows * nt);
                for (auto& x : qv) x = (mode == 1 || mode == 2) ? 7 : (uint8_t)(rng() % 8);
                for (auto& x : sc) x = f16_to_f32(f32_to_f16(((int)(rng() % 1000) + 1) / 4096.f));
                for (auto& x : mn) x = f16_to_f32(f32_to_f16(((int)(rng() % 2001) - 1000) / 1024.f));
                i3_pack(rows, n, qv.data(), sc.data(), mn.data(), w.data());
                std::vector<int8_t> q((size_t)E * n);
                std::vector<int32_t> A((size_t)E * nt, 0);
                for (int e = 0; e < E; e++)
                    for (int c = 0; c < n; c++) {
                        int v = mode == 0 ? (int)(rng() % 255) - 127 : mode == 1 ? 127 : mode == 2 ? ((c & 1) ? 127 : -127) : ((rng() & 1) ? 127 : -127);
                        q[(size_t)e * n + c] = (int8_t)v;
                        A[(size_t)e * nt + c / 128] += v;
                    }
                std::vector<float> ssv((size_t)E * nt, 1.f);
                const bool use_ss = mode == 3 || (mode == 0 && (E & 1));
                if (use_ss) for (auto& x : ssv) x = (1 + (int)(rng() % 63)) / 64.f;
                std::vector<double> want((size_t)E * rows);
                for (int e = 0; e < E; e++)
                    for (int r = 0; r < rows; r++) {
                        double acc = 0;
                        for (int t = 0; t < nt; t++) {
                            long long ssum = 0;
                            for (int c = 0; c < 128; c++) ssum += (long long)qv[(size_t)r * n + t * 128 + c] * q[(size_t)e * n + t * 128 + c];
                            acc += (double)ssv[(size_t)e * nt + t] * ((double)sc[(size_t)r * nt + t] * (double)ssum + (double)mn[(size_t)r * nt + t] * A[(size_t)e * nt + t]);
                        }
                        want[(size_t)e * rows + r] = acc;
                    }
                I3View iv; iv.rows = rows; iv.ntile = nt; iv.w = w.data();
                std::vector<float> first;
                for (Isa isa : isas) {
                    if (!isa_available(isa) || isa == Isa::Neon) continue;
                    std::vector<float> got((size_t)E * rows, -777.f);
                    i3_gemm(isa, iv, q.data(), (size_t)n, A.data(), E, got.data(), (size_t)rows, 0, rows / 64, use_ss ? ssv.data() : nullptr);
                    bool ok = true;
                    for (int e = 0; e < E && ok; e++)
                        for (int r = 0; r < rows; r++) {
                            double tol = 2e-5 * (std::fabs(want[(size_t)e * rows + r]) + 7.0 * 127.0 * 128.0 * nt * 0.25 + 127.0 * 128.0 * nt) + 1e-3;
                            if (!(std::fabs(got[(size_t)e * rows + r] - want[(size_t)e * rows + r]) <= tol)) {
                                ok = false;
                                std::snprintf(nb, sizeof nb, "FAIL i3 %s rows=%d nt=%d E=%d mode=%d e=%d row=%d got=%g want=%g\n", isa_name(isa), rows, nt, E, mode, e, r, got[(size_t)e * rows + r], want[(size_t)e * rows + r]);
                                report += nb;
                                break;
                            }
                        }
                    if (ok && first.empty()) first = got;
                    else if (ok && std::memcmp(first.data(), got.data(), got.size() * 4) != 0) { ok = false; report += "FAIL i3 ISAs not bit-identical\n"; }
                    // slot-count independence: slot 0 alone must equal slot 0 of the batched pass
                    if (ok && E > 1) {
                        std::vector<float> one(rows);
                        i3_gemm(isa, iv, q.data(), (size_t)n, A.data(), 1, one.data(), (size_t)rows, 0, rows / 64, use_ss ? ssv.data() : nullptr);
                        if (std::memcmp(one.data(), got.data(), (size_t)rows * 4) != 0) { ok = false; report += "FAIL i3 T=1 != batched\n"; }
                    }
                    checks++;
                    if (!ok) failures++;
                }
            }
    }

    std::snprintf(buf, sizeof buf, "kernel selftest: %d checks, %d failures; best ISA = %s\n", checks, failures, isa_name(best_isa()));
    report += buf;
    return failures;
}

}  // namespace lb
