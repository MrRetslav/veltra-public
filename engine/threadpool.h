// Minimal fork-join thread pool with lock-free hand-off and hybrid spin/sleep workers (decode issues
// ~800 tiny parallel regions per token, so mutex/condition-variable cost per region would dominate).
//
// One shared job word packs {run id:32 | total tasks:16 | next task:16}. Workers claim tasks with a CAS on that
// word, so a worker that wakes late can never execute a task of a different run: its snapshot carries the old id.
#pragma once
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>
#include <string>

#include "lb_common.h"
#include "platform.h"
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace lb {

inline void cpu_relax() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

class ThreadPool {
public:
    explicit ThreadPool(int n_threads) {
        if (n_threads <= 0) {  // default: physical cores (x86 with SMT: half the logical CPUs), LB_THREADS overrides
            unsigned hw = std::max(1u, std::thread::hardware_concurrency());
#if defined(__x86_64__) || defined(_M_X64)
            if (hw >= 8 && hw % 2 == 0) hw /= 2;
#endif
            n_threads = (int)hw;
            if (const char* e = lb::env_get("THREADS")) if (std::atoi(e) > 0) n_threads = std::atoi(e);
        }
        n_ = n_threads;
        parse_pin();
        pin_self(0);
        for (int i = 1; i < n_; i++) workers_.emplace_back([this, i] { pin_self(i); worker(); });
    }
    ~ThreadPool() {
        stop_.store(true);
        {
            std::lock_guard<std::mutex> lk(mu_);
            epoch_.fetch_add(1);
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }
    int size() const { return n_; }

    // Runs fn(task) for task in [0, n_tasks) across all threads; returns when all are done.
    // (template + function_ref: no std::function construction / heap allocation per parallel region)
    template <class F>
    void run(int n_tasks, F&& fn) {
        if (n_tasks <= 0) return;
        if (n_ == 1 || n_tasks == 1 || n_tasks > 0xffff) { for (int i = 0; i < n_tasks; i++) fn(i); return; }
        using Fn = std::remove_reference_t<F>;
        FnRef ref{(const void*)&fn, [](const void* o, int i) { (*(const Fn*)o)(i); }};
        run_ref(n_tasks, ref);
    }

private:
    // LB_PIN=c0,c1,c2,... pins pool thread i to CPU c[i % len] (Linux only; default: no pinning, the scheduler decides)
    std::vector<int> pin_;
    void parse_pin() {
        const char* e = lb::env_get("PIN");
        if (!e) return;
        std::string t(e);
        size_t p = 0;
        while (p < t.size()) {
            size_t q = t.find(',', p);
            if (q == std::string::npos) q = t.size();
            if (q > p) pin_.push_back(std::atoi(t.substr(p, q - p).c_str()));
            p = q + 1;
        }
    }
    void pin_self(int i) {
        if (pin_.empty()) return;
        veltra::plat::pin_current_thread(pin_[i % pin_.size()]);   // pthread_setaffinity_np / SetThreadAffinityMask
    }
    struct FnRef { const void* obj; void (*call)(const void*, int); };
    void run_ref(int n_tasks, const FnRef& fn) {
        const uint32_t id = ++run_id_;
        fn_.store(&fn, std::memory_order_relaxed);
        done_.store(0, std::memory_order_relaxed);
        word_.store(((uint64_t)id << 32) | ((uint64_t)n_tasks << 16), std::memory_order_seq_cst);
        {
            // publish the new epoch; sleeping workers are only notified when somebody actually sleeps
            std::lock_guard<std::mutex> lk(mu_);
            epoch_.store(id, std::memory_order_seq_cst);
        }
        if (sleepers_.load(std::memory_order_seq_cst) > 0) cv_.notify_all();
        drain(id);
        int spins = 0;
        while (done_.load(std::memory_order_acquire) < n_tasks) {
            if (++spins < 50000) cpu_relax(); else std::this_thread::yield();
        }
    }

    // Claim and run tasks of run `id` until none are left (or the word belongs to a different run).
    void drain(uint32_t id) {
        for (;;) {
            uint64_t v = word_.load(std::memory_order_seq_cst);
            if ((uint32_t)(v >> 32) != id) return;
            const uint32_t total = (uint32_t)((v >> 16) & 0xffff), cnt = (uint32_t)(v & 0xffff);
            if (cnt >= total) return;
            if (!word_.compare_exchange_weak(v, v + 1, std::memory_order_seq_cst)) continue;
            { const FnRef* r = fn_.load(std::memory_order_relaxed); r->call(r->obj, (int)cnt); }
            done_.fetch_add(1, std::memory_order_release);
        }
    }
    void worker() {
        uint32_t seen = 0;
        while (true) {
            int spins = 0;
            while (epoch_.load(std::memory_order_acquire) == seen && !stop_.load(std::memory_order_relaxed)) {
                ++spins;
                if (spins < 100000) { cpu_relax(); continue; }  // hot spin for a few tens of microseconds
                if (spins < 102000) { std::this_thread::yield(); continue; }
                std::unique_lock<std::mutex> lk(mu_);
                sleepers_.fetch_add(1, std::memory_order_seq_cst);
                cv_.wait(lk, [&] { return epoch_.load() != seen || stop_.load(); });
                sleepers_.fetch_sub(1, std::memory_order_seq_cst);
            }
            if (stop_.load()) return;
            seen = epoch_.load(std::memory_order_acquire);
            drain(seen);
        }
    }

    int n_ = 1;
    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::atomic<uint32_t> epoch_{0};
    std::atomic<bool> stop_{false};
    std::atomic<int> sleepers_{0};
    uint32_t run_id_ = 0;  // only touched by the thread calling run()
    std::atomic<uint64_t> word_{0};
    std::atomic<const FnRef*> fn_{nullptr};
    std::atomic<int> done_{0};
};

}  // namespace lb
