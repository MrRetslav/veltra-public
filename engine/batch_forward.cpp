// Weight-stationary batched decode: Transformer::forward_batch (see transformer.h BatchItem / BatchOpts).
//
// One pass over the weights serves T tokens that belong to independent streams (each with its own position and KV slice) or to consecutive positions of
// one stream (speculative verify chain). Everything that touches weights already works on T activation vectors at once:
//   * dense attention linears q/k/v/o: linear_forward_multi / linear_forward stream every weight once per <= 32-token chunk,
//   * MoE: per-token routing, tokens bucketed per UNIQUE expert (moe_ffn: moe_list_ = CSR token lists, the expert union), every selected expert's gate/up
//     and down streamed once for all of its tokens, jobs x row blocks spread over the pool in one parallel region (linear_forward_jobs),
//   * head: B hidden vectors per streamed head row (lm_head loops rows outermost).
// Only attention is per stream: attention_gqa_batch below is the single-stream attention_gqa with per-token (position, KV slice).
// Per-token arithmetic is the same as forward() with T=1 (same ops, same summation order), so for T < 64 (BatchOpts::fast < 0) or fast == 0 every token's logits are
// bit-identical to running that token alone; for T >= 64 (prefill chunks) the prefill kernels (weight-stationary K-quant, tile scales) are used.
#include "transformer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "lb_common.h"
#include "ops.h"

namespace lb {

int Transformer::add_kv_slice(int max_seq) {
    if (max_seq <= 0 || max_seq > (1 << 22)) fail("add_kv_slice: bad max_seq");
    auto k = std::make_unique<KvCache>();
    k->nb = kv_.nb;
    k->init(kv_.mode, cfg_.num_layers, cfg_.num_kv_heads, cfg_.head_dim, max_seq);
    slices_.push_back(std::move(k));
    return (int)slices_.size() - 1;
}

// A branch slice stores only positions >= base (a multiple of the 128-position attention chunk): the whole chunks below it are read from the shared
// prompt cache, the partial chunk at the end of the prompt is copied. Chunk boundaries stay where they are, so the result is bit-identical to a
// private full cache.
int Transformer::fork_kv_slice(int prefix_len, int capacity) {
    if (kv_.mode != KvMode::Int8) fail("shared-prefix slices need the int8 KV cache");
    if (prefix_len < 0 || prefix_len > kv_.max_seq || capacity <= 0) fail("fork_kv_slice: bad arguments");
    const int base = prefix_len & ~127, copy = prefix_len - base;
    const int id = add_kv_slice(copy + capacity);
    KvCache& d = *slices_[id];
    d.base = base;
    d.shared = &kv_;
    if (d.layers != kv_.layers || d.nb != kv_.nb) fail("fork_kv_slice: layout mismatch");
    const size_t hd = (size_t)kv_.head_dim, nb = (size_t)kv_.nb;
    for (size_t lh = 0; lh < (size_t)kv_.layers * kv_.kv_heads; lh++) {
        const size_t s = lh * kv_.max_seq + base, t = lh * d.max_seq;
        std::memcpy(&d.k8[t * hd], &kv_.k8[s * hd], (size_t)copy * hd);
        std::memcpy(&d.v8[t * hd], &kv_.v8[s * hd], (size_t)copy * hd);
        std::memcpy(&d.ks[t * nb], &kv_.ks[s * nb], (size_t)copy * nb * sizeof(float));
        std::memcpy(&d.vs[t * nb], &kv_.vs[s * nb], (size_t)copy * nb * sizeof(float));
    }
    return id;
}

// Grouped-query attention with per-token position and KV slice. Identical chunking / merge order as attention_gqa (fixed 128-position chunks merged in
// chunk order), so a token's output does not depend on which other tokens are in the batch.
void Transformer::attention_gqa_batch(int layer, const float* q, int T, const int* pos, KvCache* const* kv, float* out) {
    const int H = cfg_.num_heads, KVH = cfg_.num_kv_heads, G = H / KVH;
    constexpr int kCS = 128, kPart = 16 + 8 * 128;
    const float scale = cfg_.attention_scale();   // head_dim == 128 here
    int maxn = 1;
    for (int t = 0; t < T; t++) maxn = std::max(maxn, pos[t] / kCS + 1);
    const bool split = (long)T * KVH < 4L * pool_.size() && maxn > 1;
    if (split) { attn_part_.resize((size_t)T * KVH * maxn * kPart); attn_cnt_.assign((size_t)T * KVH, 0); }
    auto chunk = [&](int t, int g, int c, float* part) {
        const int n = std::min(kCS, pos[t] + 1 - c * kCS);
        static thread_local std::vector<float> scr, qs;
        scr.resize((size_t)kCS * 8 + 32 * 128 + 64);
        qs.resize((size_t)G * 128);
        const KvCache& own = *kv[t];
        const KvCache& K = c * kCS < own.base ? *own.shared : own;   // base is a multiple of kCS: a whole chunk is either shared or own
        const float* qh = q + ((size_t)t * H + (size_t)g * G) * 128;
        for (int i = 0; i < G * 128; i++) qs[i] = qh[i] * scale;
        const size_t base = ((size_t)layer * KVH + g) * K.max_seq + (size_t)(c * kCS - K.base);
        attn_gqa_chunk(G, qs.data(), &K.k8[base * 128], &K.ks[base * 8], &K.v8[base * 128], &K.vs[base * 8], n, part, part + 8, part + 16, scr.data());
    };
    auto merge = [&](float* run, const float* p) {
        for (int h = 0; h < G; h++) {
            const float mn = std::max(run[h], p[h]);
            const float a1 = std::exp(run[h] - mn), a2 = std::exp(p[h] - mn);
            run[8 + h] = run[8 + h] * a1 + p[8 + h] * a2;
            run[h] = mn;
            float* ra = run + 16 + h * 128;
            const float* pa = p + 16 + h * 128;
            for (int d = 0; d < 128; d++) ra[d] = ra[d] * a1 + pa[d] * a2;
        }
    };
    auto finish = [&](int t, int g, const float* run) {
        for (int h = 0; h < G; h++) {
            const float inv = 1.f / run[8 + h];
            float* o = out + ((size_t)t * H + (size_t)g * G + h) * 128;
            for (int d = 0; d < 128; d++) o[d] = run[16 + h * 128 + d] * inv;
        }
    };
    if (split) {
        pool_.run(T * KVH * maxn, [&](int ti) {
            const int c = ti % maxn, tg = ti / maxn, g = tg % KVH, t = tg / KVH;
            const int nc = pos[t] / kCS + 1;
            if (c >= nc) return;
            chunk(t, g, c, &attn_part_[((size_t)tg * maxn + c) * kPart]);
            if (__atomic_add_fetch(&attn_cnt_[tg], 1, __ATOMIC_ACQ_REL) != nc) return;
            static thread_local std::vector<float> run;
            run.resize(kPart);
            std::memcpy(run.data(), &attn_part_[(size_t)tg * maxn * kPart], kPart * 4);
            for (int cc = 1; cc < nc; cc++) merge(run.data(), &attn_part_[((size_t)tg * maxn + cc) * kPart]);
            finish(t, g, run.data());
        });
    } else {
        pool_.run(T * KVH, [&](int ti) {
            const int g = ti % KVH, t = bat_order_[ti / KVH];   // longest contexts first
            const int nc = pos[t] / kCS + 1;
            static thread_local std::vector<float> run, part;
            run.resize(kPart); part.resize(kPart);
            chunk(t, g, 0, run.data());
            for (int c = 1; c < nc; c++) { chunk(t, g, c, part.data()); merge(run.data(), part.data()); }
            finish(t, g, run.data());
        });
    }
}

void Transformer::forward_batch(const BatchItem* items, int T, const BatchOpts& o, float* logits) {
    using Clk = std::chrono::steady_clock;
    static const bool bprof = lb::env_get("BPROF") != nullptr;
    static const int tile_env = lb::env_get("PREFILL_TILE") ? std::atoi(lb::env_get("PREFILL_TILE")) : 1;
    Clk::time_point tp0;
    auto tick = [&](int k) { if (bprof) { const auto t1 = Clk::now(); batch_prof_ms[k] += std::chrono::duration<double, std::milli>(t1 - tp0).count(); tp0 = t1; } };
    if (T <= 0) return;
    const int H = cfg_.hidden_size, NH = cfg_.num_heads, KVH = cfg_.num_kv_heads, hd = cfg_.head_dim;
    const int qd = NH * hd, kvd = KVH * hd, half = hd / 2;
    if (hd != 128 || kv_.mode != KvMode::Int8 || kv_.nb != 8 || !cpu_has_f16c_fma() || (NH / KVH != 1 && NH / KVH != 2 && NH / KVH != 4 && NH / KVH != 8))
        fail("forward_batch needs int8 KV with head_dim 128, 8 scale blocks and f16c/fma (GQA attention kernel)");
    bat_pos_.resize(T); bat_kv_.resize(T); bat_order_.resize(T);
    for (int t = 0; t < T; t++) {
        const BatchItem& it = items[t];
        if (it.slice >= (int)slices_.size()) fail("forward_batch: bad KV slice id");
        KvCache* K = it.slice < 0 ? &kv_ : slices_[it.slice].get();
        if (it.pos < K->base || it.pos - K->base >= K->max_seq || it.pos >= (int)(rope_cos_.size() / (cfg_.head_dim / 2)))
            fail("forward_batch: position " + std::to_string(it.pos) + " outside KV slice (" + std::to_string(K->base) + " + " + std::to_string(K->max_seq) + ")");
        if (it.token < 0 || it.token >= cfg_.vocab_size) fail("token id out of range");
        bat_pos_[t] = it.pos; bat_kv_[t] = K; bat_order_[t] = t;
    }
    std::stable_sort(bat_order_.begin(), bat_order_.end(), [&](int a, int b) { return bat_pos_[a] > bat_pos_[b]; });
    x_.resize((size_t)T * H); h_.resize((size_t)T * H);
    q_.resize((size_t)T * qd); k_.resize((size_t)T * kvd); v_.resize((size_t)T * kvd);
    att_.resize((size_t)T * qd); o_.resize((size_t)T * H);
    const int Id = cfg_.intermediate_size;
    if (Id > 0 && !all_moe_) { g_.resize((size_t)T * Id); u_.resize((size_t)T * Id); tmp_.resize((size_t)T * Id); }
    const bool use_res = true;
    const bool fast = o.fast < 0 ? T >= 64 : o.fast == 1;   // decode-sized batches stay on the exact 4-slot kernels (measured: wst/tile at T=16 is slower AND not bit-identical)
    lscratch_.fast_prefill = fast;
    lscratch_.tile_act = fast && tile_env;
    if (bprof) tp0 = Clk::now();

    for (int t = 0; t < T; t++) {
        const int id = items[t].token;
        float* dst = &x_[(size_t)t * H];
        if (tok_emb_->dtype == DT_I8) {
            const int8_t* r = (const int8_t*)tok_emb_->data + (size_t)id * H;
            float s = 1.f;
            if (tok_scale_) { uint16_t hs; std::memcpy(&hs, tok_scale_->data + 2 * (size_t)id, 2); s = f16_to_f32(hs); }
            for (int i = 0; i < H; i++) dst[i] = (float)r[i] * s;
        } else {
            const uint16_t* r = (const uint16_t*)tok_emb_->data + (size_t)id * H;
            for (int i = 0; i < H; i++) dst[i] = f16_to_f32(r[i]);
        }
    }
    tick(5);
    for (int l = 0; l < cfg_.num_layers; l++) {
        LayerWeights& w = layers_[l];
        pool_.run(T, [&](int t) { rmsnorm(&x_[(size_t)t * H], w.attn_norm.data(), H, cfg_.rms_norm_eps, &h_[(size_t)t * H]); });
        tick(5);
        { const LbLinear* ws[3] = {&w.q, &w.k, &w.v};
          float* ys[3] = {q_.data(), k_.data(), v_.data()};
          linear_forward_multi(ws, 3, isa_, h_.data(), T, ys, use_res, lscratch_, &pool_); }
        tick(0);
        pool_.run(T, [&](int t) {   // per token: qk-norm, RoPE, KV write into the token's own slice
            const int pos = bat_pos_[t];
            KvCache& K = *bat_kv_[t];
            const float* cs = &rope_cos_[(size_t)pos * half];
            const float* sn = &rope_sin_[(size_t)pos * half];
            float tmp_buf[512];
            float* tmp_hd = tmp_buf;
            for (int h = 0; h < NH; h++) {
                float* qh = &q_[((size_t)t * NH + h) * hd];
                if (cfg_.qk_norm) { rmsnorm(qh, w.q_norm.data(), hd, cfg_.rms_norm_eps, tmp_hd); std::memcpy(qh, tmp_hd, hd * 4); }
                rope_inplace(qh, hd, cs, sn);
            }
            for (int h = 0; h < KVH; h++) {
                float* kh = &k_[((size_t)t * KVH + h) * hd];
                float* vh = &v_[((size_t)t * KVH + h) * hd];
                if (cfg_.qk_norm) { rmsnorm(kh, w.k_norm.data(), hd, cfg_.rms_norm_eps, tmp_hd); std::memcpy(kh, tmp_hd, hd * 4); }
                rope_inplace(kh, hd, cs, sn);
                const size_t slot = (((size_t)l * KVH + h) * K.max_seq + (pos - K.base));
                const int nb = K.nb, bs = hd / nb;
                for (int b = 0; b < nb; b++) {
                    K.ks[slot * nb + b] = quantize_activations_i8(kh + b * bs, bs, bs, &K.k8[slot * hd + b * bs]);
                    K.vs[slot * nb + b] = quantize_activations_i8(vh + b * bs, bs, bs, &K.v8[slot * hd + b * bs]);
                }
            }
        });
        tick(5);
        attention_gqa_batch(l, q_.data(), T, bat_pos_.data(), bat_kv_.data(), att_.data());
        tick(1);
        linear(w.o, att_.data(), T, o_.data(), use_res);
        tick(2);
        for (size_t i = 0; i < (size_t)T * H; i++) x_[i] += o_[i];
        pool_.run(T, [&](int t) { rmsnorm(&x_[(size_t)t * H], w.ffn_norm.data(), H, cfg_.rms_norm_eps, &h_[(size_t)t * H]); });
        tick(5);
        if (w.moe) {
            moe_layer_ = l;
            moe_ffn(w, T, use_res);
            if (o.count_experts && T <= 256) {
                int n = 0;
                for (const auto& li : moe_list_) n += li.empty() ? 0 : 1;
                batch_uniq_sum_ += n; batch_uniq_cnt_++;
            }
        } else {
            const LbLinear* ws[2] = {&w.gate, &w.up};
            float* ys[2] = {g_.data(), u_.data()};
            linear_forward_multi(ws, 2, isa_, h_.data(), T, ys, use_res, lscratch_, &pool_);
            silu_mul(g_.data(), u_.data(), T * cfg_.intermediate_size, tmp_.data());
            linear(w.down, tmp_.data(), T, o_.data(), use_res);
        }
        tick(3);
        for (size_t i = 0; i < (size_t)T * H; i++) x_[i] += o_[i];
        tick(5);
    }
    const int Tl = T - o.logits_from;
    if (Tl > 0 && logits) {
        for (int t = 0; t < Tl; t++)
            rmsnorm(&x_[(size_t)(o.logits_from + t) * H], out_norm_.data(), H, cfg_.rms_norm_eps, &h_[(size_t)t * H]);
        lm_head_draft_ = false;
        lm_head_exact_ = o.greedy && head_vocab_.empty();
        lm_head_screen_ok_ = o.screen_ok;
        lm_head(h_.data(), Tl, logits);
        lm_head_exact_ = false;
        lm_head_screen_ok_ = true;
        tick(4);
    }
}

}  // namespace lb
