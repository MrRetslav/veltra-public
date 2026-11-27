// Memory-bandwidth probe that mimics how the engine streams weights: threads read disjoint blocks of a large anonymous buffer with AVX2 loads,
// either sequentially or as "expert-sized" blocks (1.7 MB) in random order, with an optional software prefetch distance and thread pinning.
//   g++ -O3 -mavx2 -std=c++20 -pthread membw.cpp -o membw
//   ./membw <GB buffer> <reps>   -> one JSON line per configuration (median GB/s of <reps> passes, configurations interleaved)
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

struct Config {
    const char* name;
    std::vector<int> cpus;   // empty = unpinned
    int threads;
    bool expert_blocks;      // random 1.7 MB blocks instead of one sequential range per thread
    int prefetch_lines;      // 0 = hardware prefetcher only
};

static constexpr size_t kBlock = 1747 * 1024 / 64 * 64;   // about one expert of one layer

__attribute__((target("avx2"))) static uint64_t read_range(const uint8_t* p, size_t n, int pf) {
    __m256i a = _mm256_setzero_si256(), b = a, c = a, d = a;
    for (size_t i = 0; i + 128 <= n; i += 128) {
        if (pf) {
            _mm_prefetch((const char*)p + i + pf * 64, _MM_HINT_T0);
            _mm_prefetch((const char*)p + i + pf * 64 + 64, _MM_HINT_T0);
        }
        a = _mm256_add_epi32(a, _mm256_load_si256((const __m256i*)(p + i)));
        b = _mm256_add_epi32(b, _mm256_load_si256((const __m256i*)(p + i + 32)));
        c = _mm256_add_epi32(c, _mm256_load_si256((const __m256i*)(p + i + 64)));
        d = _mm256_add_epi32(d, _mm256_load_si256((const __m256i*)(p + i + 96)));
    }
    a = _mm256_add_epi32(_mm256_add_epi32(a, b), _mm256_add_epi32(c, d));
    alignas(32) uint32_t v[8];
    _mm256_store_si256((__m256i*)v, a);
    uint64_t s = 0;
    for (uint32_t x : v) s += x;
    return s;
}

int main(int argc, char** argv) {
    const double gb = argc > 1 ? std::atof(argv[1]) : 2.0;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 5;
    const size_t bytes = (size_t)(gb * 1e9) / kBlock * kBlock;
    uint8_t* buf = (uint8_t*)mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (buf == MAP_FAILED) { std::perror("mmap"); return 1; }
    std::memset(buf, 1, bytes);
    const size_t nblocks = bytes / kBlock;
    std::vector<uint32_t> order(nblocks);
    for (size_t i = 0; i < nblocks; i++) order[i] = (uint32_t)i;
    std::shuffle(order.begin(), order.end(), std::mt19937(7));
    const std::vector<int> two_ccd = {0, 1, 2, 3, 6, 7, 8, 9}, four_ccx = {0, 1, 3, 4, 6, 7, 9, 10}, one_ccd = {0, 1, 2, 3, 4, 5, 12, 13};
    std::vector<Config> cfg = {
        {"unpinned-1t-seq", {}, 1, false, 0}, {"unpinned-2t-seq", {}, 2, false, 0}, {"unpinned-4t-seq", {}, 4, false, 0},
        {"unpinned-6t-seq", {}, 6, false, 0}, {"unpinned-8t-seq", {}, 8, false, 0},
        {"unpinned-8t-expert", {}, 8, true, 0},
        {"pin-4+4-ccd-expert", two_ccd, 8, true, 0}, {"pin-2x4-ccx-expert", four_ccx, 8, true, 0}, {"pin-one-ccd-smt-expert", one_ccd, 8, true, 0},
        {"pin-2x4-ccx-seq", four_ccx, 8, false, 0},
        {"unpinned-8t-expert-pf2", {}, 8, true, 2}, {"unpinned-8t-expert-pf4", {}, 8, true, 4}, {"unpinned-8t-expert-pf8", {}, 8, true, 8},
        {"unpinned-8t-expert-pf16", {}, 8, true, 16},
    };
    std::vector<std::vector<double>> res(cfg.size());
    std::atomic<uint64_t> sink{0};
    for (int r = 0; r < reps; r++)
        for (size_t c = 0; c < cfg.size(); c++) {
            const Config& k = cfg[c];
            std::atomic<size_t> next{0};
            std::atomic<int> ready{0};
            std::atomic<bool> go{false};
            std::vector<std::thread> th;
            for (int t = 0; t < k.threads; t++)
                th.emplace_back([&, t] {
                    if (!k.cpus.empty()) {
                        cpu_set_t set;
                        CPU_ZERO(&set);
                        CPU_SET(k.cpus[t % k.cpus.size()], &set);
                        pthread_setaffinity_np(pthread_self(), sizeof set, &set);
                    }
                    ready++;
                    while (!go.load()) std::this_thread::yield();
                    uint64_t s = 0;
                    if (k.expert_blocks) {
                        for (size_t i; (i = next.fetch_add(1)) < nblocks;) s += read_range(buf + (size_t)order[i] * kBlock, kBlock, k.prefetch_lines);
                    } else {
                        const size_t part = bytes / k.threads / 128 * 128;
                        s += read_range(buf + (size_t)t * part, part, k.prefetch_lines);
                    }
                    sink += s;
                });
            while (ready.load() < k.threads) std::this_thread::yield();
            const auto t0 = std::chrono::steady_clock::now();
            go = true;
            for (auto& x : th) x.join();
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            res[c].push_back((double)bytes / s / 1e9);
        }
    for (size_t c = 0; c < cfg.size(); c++) {
        std::sort(res[c].begin(), res[c].end());
        std::printf("{\"config\":\"%s\",\"threads\":%d,\"gb_s_median\":%.2f,\"gb_s_min\":%.2f,\"gb_s_max\":%.2f,\"buffer_gb\":%.2f,\"reps\":%d}\n", cfg[c].name,
                    cfg[c].threads, res[c][res[c].size() / 2], res[c].front(), res[c].back(), bytes / 1e9, reps);
    }
    std::fprintf(stderr, "checksum %llu\n", (unsigned long long)sink.load());
    return 0;
}
