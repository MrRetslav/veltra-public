// Apex experiment: three exact AVX2 kernels for 2-bit weights x int8 activations on Zen 2, same arithmetic result, different layouts.
//   maddubs : 4 weights per byte, unpacked with one variable shift to 32 u8 values, then vpmaddubsw (the style of the engine's K-quant kernels)
//   bitplane: two 32-bit planes per 32 weights, expanded to byte masks; y = sum_{b0} x + 2 sum_{b1} x
//   lut32   : "VLT-Q2-LUT32", T-MAC style: 2 weights form a 4-bit index into a per-activation-pair table of partial sums (int16 split into two
//             byte tables), looked up with lane-local vpshufb for 32 rows at once
// No scales (identical per-group scale handling would be added to all three). Every kernel is checked bit-exactly against a scalar reference.
// Counters come from perf_event_open (no perf tool needed): cycles, instructions, L1D read misses, last-level cache misses.
//   g++ -O3 -mavx2 -std=c++20 -pthread q2kernels.cpp -o q2kernels && ./q2kernels <dram GB> <threads> <reps>
#include <immintrin.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

constexpr int N = 2048;        // columns (activations)
constexpr int TILE = 32;       // rows per tile

struct Packed {
    int rows = 0;
    std::vector<uint8_t> mad;    // [rows][N/32][8]
    std::vector<uint32_t> bp;    // [rows][N/32][2]
    std::vector<uint8_t> lut;    // [rows/32][N/4][32]: block q holds column pairs 2q, 2q+1; byte r: low nibble pair 2q, high nibble pair 2q+1
};

static void pack(const std::vector<uint8_t>& w, int rows, Packed& p, int which) {
    p.rows = rows;
    if (which == 0) {
        p.mad.assign((size_t)rows * (N / 32) * 8, 0);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < N; c++) {
                const int block = c / 32, i = c % 32;   // byte i%8 holds weights i%8, +8, +16, +24 in bit pairs 0, 2, 4, 6
                p.mad[((size_t)r * (N / 32) + block) * 8 + i % 8] |= (uint8_t)(w[(size_t)r * N + c] << (2 * (i / 8)));
            }
    } else if (which == 1) {
        p.bp.assign((size_t)rows * (N / 32) * 2, 0);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < N; c++) {
                const uint8_t v = w[(size_t)r * N + c];
                uint32_t* q = &p.bp[((size_t)r * (N / 32) + c / 32) * 2];
                q[0] |= (uint32_t)(v & 1) << (c % 32);
                q[1] |= (uint32_t)(v >> 1) << (c % 32);
            }
    } else {
        p.lut.assign((size_t)(rows / TILE) * (N / 4) * 32, 0);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < N; c++) {
                const int pair = c / 2, inpair = c % 2;
                uint8_t* byte = &p.lut[((size_t)(r / TILE) * (N / 4) + pair / 2) * 32 + r % TILE];
                *byte |= (uint8_t)(w[(size_t)r * N + c] << (2 * inpair + 4 * (pair % 2)));
            }
    }
}

static void kernel_maddubs(const Packed& p, const int8_t* x, int32_t* y, int r0, int r1) {
    const __m256i shifts = _mm256_setr_epi64x(0, 2, 4, 6), mask = _mm256_set1_epi8(3), ones = _mm256_set1_epi16(1);
    for (int r = r0; r < r1; r++) {
        __m256i acc = _mm256_setzero_si256();
        const uint8_t* w = &p.mad[(size_t)r * (N / 32) * 8];
        for (int b = 0; b < N / 32; b++) {
            uint64_t q;
            std::memcpy(&q, w + 8 * b, 8);
            const __m256i u = _mm256_and_si256(_mm256_srlv_epi64(_mm256_set1_epi64x((long long)q), shifts), mask);   // 32 values 0..3 in column order
            const __m256i a = _mm256_loadu_si256((const __m256i*)(x + 32 * b));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(u, a), ones));
        }
        __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
        y[r] = _mm_cvtsi128_si32(s);
    }
}

static void kernel_bitplane(const Packed& p, const int8_t* x, int32_t* y, int r0, int r1) {
    const __m256i spread = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i bit = _mm256_set1_epi64x((long long)0x8040201008040201ull), ones8 = _mm256_set1_epi8(1), ones16 = _mm256_set1_epi16(1);
    auto masked = [&](uint32_t m, __m256i a) {
        __m256i v = _mm256_shuffle_epi8(_mm256_set1_epi32((int)m), spread);
        v = _mm256_cmpeq_epi8(_mm256_and_si256(v, bit), bit);
        return _mm256_and_si256(v, a);
    };
    for (int r = r0; r < r1; r++) {
        __m256i acc = _mm256_setzero_si256();
        const uint32_t* w = &p.bp[(size_t)r * (N / 32) * 2];
        for (int b = 0; b < N / 32; b++) {
            const __m256i a = _mm256_loadu_si256((const __m256i*)(x + 32 * b));
            const __m256i lo = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, masked(w[2 * b], a)), ones16);
            const __m256i hi = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, masked(w[2 * b + 1], a)), ones16);
            acc = _mm256_add_epi32(acc, _mm256_add_epi32(lo, _mm256_add_epi32(hi, hi)));
        }
        __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
        y[r] = _mm_cvtsi128_si32(s);
    }
}

// Tables for one activation vector: for every column pair 16 int16 sums a0*(idx&3) + a1*(idx>>2), split into low and high bytes.
static void build_tables(const int8_t* x, uint8_t* t) {
    for (int pair = 0; pair < N / 2; pair++) {
        uint8_t* lo = t + (size_t)pair * 32;
        uint8_t* hi = lo + 16;
        for (int i = 0; i < 16; i++) {
            const int16_t v = (int16_t)(x[2 * pair] * (i & 3) + x[2 * pair + 1] * (i >> 2));
            lo[i] = (uint8_t)(v & 0xFF);
            hi[i] = (uint8_t)((v >> 8) & 0xFF);
        }
    }
}

static void kernel_lut32(const Packed& p, const uint8_t* tables, int32_t* y, int r0, int r1) {
    const __m256i nib = _mm256_set1_epi8(0x0F), ones = _mm256_set1_epi16(1);
    constexpr int kFlush = 20;   // |entry| <= 762, 40 entries per int16 lane stay below 32767; one block adds 2 entries per lane
    for (int t = r0 / TILE; t < r1 / TILE; t++) {
        __m256i acc32[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256()};
        const uint8_t* w = &p.lut[(size_t)t * (N / 4) * 32];
        for (int q0 = 0; q0 < N / 4; q0 += kFlush) {
            __m256i a16 = _mm256_setzero_si256(), b16 = _mm256_setzero_si256();
            const int q1 = std::min(N / 4, q0 + kFlush);
            for (int q = q0; q < q1; q++) {
                const __m256i idx = _mm256_loadu_si256((const __m256i*)(w + 32 * q));
                const __m256i i0 = _mm256_and_si256(idx, nib), i1 = _mm256_and_si256(_mm256_srli_epi16(idx, 4), nib);
                const uint8_t* tp = tables + (size_t)q * 64;
                const __m256i lo0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)tp));
                const __m256i hi0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tp + 16)));
                const __m256i lo1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tp + 32)));
                const __m256i hi1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(tp + 48)));
                const __m256i l0 = _mm256_shuffle_epi8(lo0, i0), h0 = _mm256_shuffle_epi8(hi0, i0);
                const __m256i l1 = _mm256_shuffle_epi8(lo1, i1), h1 = _mm256_shuffle_epi8(hi1, i1);
                a16 = _mm256_add_epi16(a16, _mm256_add_epi16(_mm256_unpacklo_epi8(l0, h0), _mm256_unpacklo_epi8(l1, h1)));   // rows 0-7, 16-23
                b16 = _mm256_add_epi16(b16, _mm256_add_epi16(_mm256_unpackhi_epi8(l0, h0), _mm256_unpackhi_epi8(l1, h1)));   // rows 8-15, 24-31
            }
            acc32[0] = _mm256_add_epi32(acc32[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(a16)));        // rows 0-7
            acc32[2] = _mm256_add_epi32(acc32[2], _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a16, 1)));  // rows 16-23
            acc32[1] = _mm256_add_epi32(acc32[1], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b16)));        // rows 8-15
            acc32[3] = _mm256_add_epi32(acc32[3], _mm256_cvtepi16_epi32(_mm256_extracti128_si256(b16, 1)));  // rows 24-31
        }
        (void)ones;
        for (int k = 0; k < 4; k++) _mm256_storeu_si256((__m256i*)(y + t * TILE + 8 * k), acc32[k]);
    }
}

struct Counters {
    int fd[4] = {-1, -1, -1, -1};
    Counters() {
        const uint64_t cfg[4][2] = {{PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES}, {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
                                    {PERF_TYPE_HW_CACHE, PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)},
                                    {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES}};
        for (int i = 0; i < 4; i++) {
            perf_event_attr a{};
            a.size = sizeof a;
            a.type = (uint32_t)cfg[i][0];
            a.config = cfg[i][1];
            a.disabled = 1;
            a.exclude_kernel = 1;
            a.inherit = 1;
            a.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
            fd[i] = (int)syscall(SYS_perf_event_open, &a, 0, -1, -1, 0);
        }
    }
    void start() { for (int f : fd) if (f >= 0) { ioctl(f, PERF_EVENT_IOC_RESET, 0); ioctl(f, PERF_EVENT_IOC_ENABLE, 0); } }
    void stop(double* v) {
        for (int i = 0; i < 4; i++) {
            v[i] = -1;
            if (fd[i] < 0) continue;
            ioctl(fd[i], PERF_EVENT_IOC_DISABLE, 0);
            uint64_t r[3] = {0, 0, 0};
            if (read(fd[i], r, sizeof r) == sizeof r && r[2]) v[i] = (double)r[0] * (double)r[1] / (double)r[2];
        }
    }
};

int main(int argc, char** argv) {
    const double dram_gb = argc > 1 ? std::atof(argv[1]) : 1.0;
    const int threads = argc > 2 ? std::atoi(argv[2]) : 8, reps = argc > 3 ? std::atoi(argv[3]) : 5;
    std::mt19937 rng(11);
    std::vector<int8_t> x(N);
    for (auto& v : x) v = (int8_t)((int)(rng() % 255) - 127);
    std::vector<uint8_t> tables((size_t)(N / 2) * 32);
    build_tables(x.data(), tables.data());
    const char* names[3] = {"maddubs", "bitplane", "lut32"};
    auto run = [&](int k, const Packed& p, int32_t* y, int r0, int r1) {
        if (k == 0) kernel_maddubs(p, x.data(), y, r0, r1);
        else if (k == 1) kernel_bitplane(p, x.data(), y, r0, r1);
        else kernel_lut32(p, tables.data(), y, r0, r1);
    };
    // exactness
    {
        const int rows = 256;
        std::vector<uint8_t> w((size_t)rows * N);
        for (auto& v : w) v = (uint8_t)(rng() & 3);
        std::vector<int32_t> ref(rows, 0), y(rows);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < N; c++) ref[r] += x[c] * w[(size_t)r * N + c];
        for (int k = 0; k < 3; k++) {
            Packed p;
            pack(w, rows, p, k);
            run(k, p, y.data(), 0, rows);
            int bad = 0;
            for (int r = 0; r < rows; r++) bad += y[r] != ref[r];
            std::printf("{\"exact\":\"%s\",\"rows\":%d,\"wrong\":%d}\n", names[k], rows, bad);
            if (bad) return 1;
        }
    }
    // cost of building the LUTs once per activation vector (shared by every row of every matrix that reads x)
    {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 2000; i++) build_tables(x.data(), tables.data());
        std::printf("{\"lut_build_us_per_vector\":%.2f}\n", std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 2000);
    }
    struct Case { const char* where; int rows; int nthr; int passes; };
    const int dram_rows = (int)(dram_gb * 1e9 / (N / 4)) / TILE * TILE;
    const Case cases[3] = {{"in-L2 1 thread", 512, 1, 400}, {"dram 1 thread", dram_rows, 1, 1}, {"dram all threads", dram_rows, threads, 1}};
    for (const Case& c : cases) {
        std::vector<uint8_t> w((size_t)c.rows * N);
        for (size_t i = 0; i < w.size(); i += 8) {
            uint64_t v = ((uint64_t)rng() << 32 | rng()) & 0x0303030303030303ull;
            std::memcpy(&w[i], &v, std::min<size_t>(8, w.size() - i));
        }
        Packed p[3];
        for (int k = 0; k < 3; k++) pack(w, c.rows, p[k], k);
        w.clear();
        w.shrink_to_fit();
        std::vector<int32_t> y(c.rows);
        std::vector<std::vector<double>> sec(3), cyc(3), ins(3), l1(3), llc(3);
        for (int r = 0; r < reps; r++)
            for (int k = 0; k < 3; k++) {
                Counters ctr;
                double v[4];
                ctr.start();
                const auto t0 = std::chrono::steady_clock::now();
                if (c.nthr == 1) {
                    for (int s = 0; s < c.passes; s++) run(k, p[k], y.data(), 0, c.rows);
                } else {
                    std::vector<std::thread> th;
                    std::atomic<int> next{0};
                    for (int t = 0; t < c.nthr; t++)
                        th.emplace_back([&] {
                            for (int s; (s = next.fetch_add(TILE * 4)) < c.rows;) run(k, p[k], y.data(), s, std::min(c.rows, s + TILE * 4));
                        });
                    for (auto& t : th) t.join();
                }
                sec[k].push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                ctr.stop(v);
                const double units = (double)c.rows * N * c.passes / 32.0;
                cyc[k].push_back(v[0] / units);
                ins[k].push_back(v[1] / units);
                l1[k].push_back(v[2] / units);
                llc[k].push_back(v[3] / units);
            }
        for (int k = 0; k < 3; k++) {
            auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
            const double s = med(sec[k]), weights = (double)c.rows * N * c.passes;
            std::printf("{\"case\":\"%s\",\"kernel\":\"%s\",\"threads\":%d,\"gweights_s\":%.1f,\"gb_s\":%.2f,\"cycles_per_32w\":%.3f,\"instr_per_32w\":%.3f,"
                        "\"l1d_miss_per_32w\":%.4f,\"llc_miss_per_32w\":%.4f}\n",
                        c.where, names[k], c.nthr, weights / s / 1e9, weights / 4 / s / 1e9, med(cyc[k]), med(ins[k]), med(l1[k]), med(llc[k]));
            std::fflush(stdout);
        }
    }
    return 0;
}
