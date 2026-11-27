#include "rest_index.h"
#include "lb_common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>


namespace lb {

static const size_t kHeader = 4096;   // header page; buckets start page-aligned (and therefore 64-byte aligned)

RestIndex::~RestIndex() {
    veltra::plat::unmap_file(&fmap_);
    if (owned_) std::free(raw_);
}

bool RestIndex::open(const std::string& path, std::string* err) {
    auto fail = [&](const char* m) { if (err) *err = std::string("rest index ") + path + ": " + m; return false; };
    size_t sz = 0;
    veltra::plat::MapHints hints;
    hints.populate = true;   // the index is probed at random: map every page up front
    if (veltra::plat::map_file_ro(path, &fmap_, nullptr, hints)) {
        sz = fmap_.size;
        map_ = (void*)fmap_.data; owned_ = false;
    } else {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return fail("cannot open");
    std::fseek(f, 0, SEEK_END); sz = (size_t)std::ftell(f); std::fseek(f, 0, SEEK_SET);
    void* raw = std::malloc(sz + 64);   // portable fallback (no mmap): 64-byte aligned copy so a bucket never straddles two cache lines
    void* m = raw ? (void*)(((uintptr_t)raw + 63) & ~(uintptr_t)63) : nullptr;
    if (!m || std::fread(m, 1, sz, f) != sz) { std::fclose(f); std::free(raw); return fail("read failed"); }
    std::fclose(f);
    raw_ = raw; map_ = m; owned_ = true;
    }
    map_bytes_ = sz;
    base_ = (const uint8_t*)map_;
    if (sz < kHeader || std::memcmp(base_, "RESTIDX1", 8) != 0) return fail("bad magic");
    uint32_t h[8];
    std::memcpy(h, base_ + 8, sizeof h);   // version, (reserved), probe, max_chain, n_entries, orders_mask, 2 x reserved
    uint64_t nb = 0;
    std::memcpy(&nb, base_ + 40, 8);       // number of 64-byte buckets (any value; bucket = fastrange(hash low 32 bits))
    if (h[0] != 1 || h[3] != (uint32_t)kRestMaxChain || nb == 0) return fail("unsupported version / chain length");
    if (sz < kHeader + nb * 64) return fail("truncated");
    nb_ = nb; probe_ = h[2]; n_entries_ = h[4]; orders_ = h[5];
    buckets_ = (const Entry*)(base_ + kHeader);
    return true;
}

const RestIndex::Entry* RestIndex::find(const int32_t* ctx, int n) const { return find_hash(rest_hash(ctx, n)); }

const RestIndex::Entry* RestIndex::find_hash(uint64_t h) const {
    uint32_t tag = (uint32_t)(h >> 32);
    if (!tag) tag = 1;
    uint64_t b = bucket_of(h);
    for (uint32_t p = 0; p < probe_; p++, b = (b + 1 == nb_ ? 0 : b + 1)) {
        const Entry* e = buckets_ + b * 2;
        if (e[0].tag == tag) return &e[0];
        if (e[1].tag == tag) return &e[1];
        if (e[0].tag == 0 || e[1].tag == 0) return nullptr;   // a bucket with a free slot ends the probe sequence (builder fills slots in order)
    }
    return nullptr;
}

static inline void unpack(const uint32_t pq, const uint8_t* tk, int* tok, float* p, int* len) {
    int n = 0;
    for (int j = 0; j < kRestMaxChain; j++) {
        const uint32_t q = (pq >> (4 * j)) & 15u;
        if (!q) break;
        tok[n] = (int)tk[3 * j] | ((int)tk[3 * j + 1] << 8) | ((int)tk[3 * j + 2] << 16);
        p[n] = (float)q * (1.0f / 15.0f);
        n++;
    }
    *len = n;
}

int RestIndex::lookup_hash(uint64_t h, int* tok, float* p) const {
    if (!buckets_) return 0;
    const Entry* e = find_hash(h);
    if (!e) return 0;
    int len = 0;
    unpack(e->pq, e->tok, tok, p, &len);
    return len;
}

int RestIndex::lookup(const int32_t* ctx, int n, int* tok, float* p) const {
    if (!buckets_) return 0;
    const Entry* e = find(ctx, n);
    if (!e) return 0;
    int len = 0;
    unpack(e->pq, e->tok, tok, p, &len);
    return len;
}

bool RestIndex::propose(const int32_t* ctx, size_t S, int max_len, const float* cost, const RestGate& gate, float min_gain, RestDraft& cand) const {
    cand = RestDraft();
    if (!buckets_ || max_len <= 0) return false;
    if (max_len > kRestMaxChain) max_len = kRestMaxChain;
    static const int kOrders[3] = {4, 3, 2};
    const Entry* hit[3] = {nullptr, nullptr, nullptr};
    uint64_t bk[3];
    int nord = 0;
    for (int i = 0; i < 3; i++) {   // prefetch all candidate buckets first (3 independent cache misses overlap)
        const int n = kOrders[i];
        if ((int)S < n || !(orders_ & (1u << n))) { bk[i] = ~0ull; continue; }
        bk[i] = bucket_of(rest_hash(ctx + S - n, n));
        __builtin_prefetch(buckets_ + bk[i] * 2);
        nord++;
    }
    if (!nord) return false;
    for (int i = 0; i < 3; i++) if (bk[i] != ~0ull) hit[i] = find(ctx + S - kOrders[i], kOrders[i]);
    float best_gain = 0.f;
    for (int i = 0; i < 3; i++) {
        if (!hit[i]) continue;
        int tok[kRestMaxChain]; float p[kRestMaxChain]; int len = 0;
        unpack(hit[i]->pq, hit[i]->tok, tok, p, &len);
        if (len > max_len) len = max_len;
        double prod = 1.0, eacc = 0.0;
        int bestk = 0; double bestg = 0.0, beste = 0.0;
        for (int j = 0; j < len; j++) {
            double pc = (double)p[j] * gate.calib(j);
            if (pc > 1.0) pc = 1.0;
            prod *= pc;
            eacc += prod;
            const double g = (1.0 + eacc) / cost[j + 2 < kRestMaxCost ? j + 2 : kRestMaxCost - 1];
            if (g > bestg) { bestg = g; bestk = j + 1; beste = eacc; }
        }
        if (bestk > 0 && (float)bestg > best_gain) {
            best_gain = (float)bestg;
            cand.n = bestk; cand.order = kOrders[i]; cand.gain = best_gain; cand.eacc = (float)beste;
            for (int j = 0; j < bestk; j++) { cand.tok[j] = tok[j]; cand.p[j] = p[j]; }
        }
    }
    return cand.n > 0 && cand.gain >= min_gain;
}

const float* rest_cost_table() {
    // verify-pass cost relative to one plain decode pass, index = number of tokens T in the pass (measured on the reference server, qk model, ctx ~500)
    static float t[kRestMaxCost] = {1.f, 1.f, 1.4f, 1.8f, 2.15f, 2.5f, 2.9f, 3.3f, 3.7f, 4.1f};
    static bool init = false;
    if (!init) {
        init = true;
        if (const char* e = lb::env_get("REST_COST")) {
            int i = 1;
            for (const char* q = e; *q && i < kRestMaxCost;) { t[i++] = (float)std::atof(q); while (*q && *q != ',') q++; if (*q == ',') q++; }
        }
    }
    return t;
}

static std::unique_ptr<RestIndex> g_rest;
RestIndex* rest_index() { return g_rest ? g_rest.get() : nullptr; }
bool rest_load(const std::string& path, std::string* err) {
    auto r = std::make_unique<RestIndex>();
    if (!r->open(path, err)) return false;
    g_rest = std::move(r);
    return true;
}

}  // namespace lb
