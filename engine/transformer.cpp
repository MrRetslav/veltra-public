#include "transformer.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#include "lb_common.h"
#include "ops.h"

namespace lb {

// Optional per-section timing (LB_PROFILE=1); printed by lb_gen after a run.
double g_prof_ms[16] = {0};
static const bool g_prof_on = lb::env_get("PROFILE") != nullptr;
struct ProfScope {
    int k; std::chrono::steady_clock::time_point t0;
    bool live;
    explicit ProfScope(int kk) : k(kk), live(g_prof_on) { if (g_prof_on) t0 = std::chrono::steady_clock::now(); }
    void stop() { if (live) { g_prof_ms[k] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); live = false; } }
    ~ProfScope() { stop(); }
};

// Calibration hook (LB_HESS=dir, opt-in): appends, per layer, the f16 inputs of the attention linears for every forwarded token:
// <dir>/L<ll>_h.f16 = post attn_norm activations [T][H] (shared input of q/k/v), <dir>/L<ll>_a.f16 = attention output [T][NH*hd] (input of o).
// python/quant_i3.py turns them into Hessians (sum x x^T / N) and a held-out split; the hook never touches the numerics.
static const char* g_hess_dir = lb::env_get("HESS");
static void hess_dump(int l, char which, const float* x, size_t n) {
    static FILE* fs[128][2];
    FILE*& f = fs[l & 127][which == 'h' ? 0 : 1];
    if (!f) { char nm[512]; std::snprintf(nm, sizeof nm, "%s/L%02d_%c.f16", g_hess_dir, l, which); f = std::fopen(nm, "ab"); if (!f) { std::fprintf(stderr, "LB_HESS: cannot open %s\n", nm); std::exit(1); } }
    static thread_local std::vector<uint16_t> buf;
    buf.resize(n);
    for (size_t i = 0; i < n; i++) buf[i] = f32_to_f16(x[i]);
    std::fwrite(buf.data(), 2, n, f);
}

// Debug dump (LB_DUMP_LAYERS=prefix, one forward per process): <prefix>.x = hidden state after every layer [layer][T][H] f32,
// <prefix>.xa = after the attention residual of every layer, <prefix>.e = per layer, per token: K expert ids (int32, ascending) + router logit gap
// between the K-th and (K+1)-th expert (f32). Used to tell float-order noise from router near-tie flips.
static FILE* g_dump_f[3] = {nullptr, nullptr, nullptr};
static const char* g_dump_prefix = lb::env_get("DUMP_LAYERS");
static FILE* dump_file(int k) {
    if (!g_dump_prefix) return nullptr;
    if (!g_dump_f[k]) { static const char* sfx[3] = {".x", ".xa", ".e"}; g_dump_f[k] = std::fopen((std::string(g_dump_prefix) + sfx[k]).c_str(), "wb"); }
    return g_dump_f[k];
}

// ------------------------------------------------------------------------------ KvCache
void KvCache::init(KvMode m, int L, int kvh, int hd, int ms) {
    mode = m; layers = L; kv_heads = kvh; head_dim = hd; max_seq = ms; len = 0;
    size_t n = (size_t)L * kvh * ms;
    if (m == KvMode::Int8) {
        k8.assign(n * hd, 0); v8.assign(n * hd, 0);
        if (nb < 1 || hd % nb != 0) nb = 1;
        ks.assign(n * nb, 0.f); vs.assign(n * nb, 0.f);
        kf.clear(); vf.clear();
    } else {
        kf.assign(n * hd, 0.f); vf.assign(n * hd, 0.f);
        k8.clear(); v8.clear(); ks.clear(); vs.clear();
    }
}

size_t KvCache::bytes() const {
    return k8.size() + v8.size() + (ks.size() + vs.size() + kf.size() + vf.size()) * sizeof(float);
}

// ------------------------------------------------------------------------------ Transformer
static bool must_load(const LbtFile& f, const std::string& name, LbLinear* l) {
    std::string e;
    if (!load_linear(f, name, l, &e)) fail("cannot load linear " + name + ": " + e);
    return true;
}

void Transformer::run_moe_jobs() {
    if (gpu_be_) gpu_be_->moe_experts(moe_jobs_.data(), (int)moe_jobs_.size(), lscratch_);
    else linear_forward_jobs(moe_jobs_.data(), (int)moe_jobs_.size(), isa_, lscratch_, &pool_);
}

Transformer::Transformer(const LbtFile& f, const ModelConfig& cfg, int max_ctx, KvMode kvmode, int n_threads, Isa isa)
    : f_(f), cfg_(cfg), isa_(isa), pool_(n_threads) {
    const int L = cfg.num_layers;
    layers_.resize(L);
    if (cuda_backend_built()) {
        const char* cu = lb::env_get("CUDA");
        if (!cu || std::strcmp(cu, "0") != 0) {
            std::string cerr;
            cpu_be_ = make_cpu_backend(isa_, &pool_);
            gpu_be_ = make_cuda_backend(cpu_be_.get(), &cerr);
            if (gpu_be_) gpu_be_->upload_weights(f, 0);   // budget: free VRAM - 1.2 GiB; the rest of the experts stays on the CPU
            else std::fprintf(stderr, "[veltra] CUDA backend not used: %s\n", cerr.c_str());
        }
    }
    hybrid_ = cfg.hybrid();
    if (hybrid_) dn_.resize(L);
    for (int i = 0; i < L; i++) {
        std::string b = "blk." + std::to_string(i) + ".";
        LayerWeights& w = layers_[i];
        w.attn_norm = f.vec_f32(b + "attn_norm");
        w.ffn_norm = f.vec_f32(b + "ffn_norm");
        if (hybrid_) load_hybrid_layer(i, w);   // full-attention layers (gated q) or Gated DeltaNet layers + shared expert (engine/hybrid_forward.cpp)
        else {
        if (cfg.qk_norm) { w.q_norm = f.vec_f32(b + "q_norm"); w.k_norm = f.vec_f32(b + "k_norm"); }
        must_load(f, b + "q", &w.q); must_load(f, b + "k", &w.k); must_load(f, b + "v", &w.v); must_load(f, b + "o", &w.o);
        }
        if (cfg.num_experts > 0 && f.has(b + "moe_gate")) {
            const TensorRef& r = f.get(b + "moe_gate");
            if (r.dtype != DT_F16 || r.numel() != (uint64_t)cfg.num_experts * cfg.hidden_size) fail("bad router tensor in layer " + std::to_string(i));
            w.moe = true;
            w.router = (const uint16_t*)r.data;
            w.egate.resize(cfg.num_experts); w.eup.resize(cfg.num_experts); w.edown.resize(cfg.num_experts);
            for (int e = 0; e < cfg.num_experts; e++) {
                const std::string eb = b + "e" + std::to_string(e) + ".";
                must_load(f, eb + "gate", &w.egate[e]); must_load(f, eb + "up", &w.eup[e]); must_load(f, eb + "down", &w.edown[e]);
            }
        } else {
            must_load(f, b + "gate", &w.gate); must_load(f, b + "up", &w.up); must_load(f, b + "down", &w.down);
        }
    }
    if (hybrid_ && f.has("mtp.norm") && f.has("mtp.blk.0.moe_gate") && !lb::env_get("MTP_OFF")) load_mtp();   // optional MTP block: one extra layer + KV layer (plain decoding never touches it)
    all_moe_ = true;
    for (const auto& lw : layers_) all_moe_ = all_moe_ && lw.moe;
    out_norm_ = f.vec_f32("output_norm");
    tok_emb_ = &f.get("tok_embeddings");
    tok_scale_ = f.find("tok_embeddings.scale");
    if (f.has("output")) { out_w_ = &f.get("output"); out_scale_ = f.find("output.scale"); }
    else { out_w_ = tok_emb_; out_scale_ = tok_scale_; }
    head_ptr_ = (const uint16_t*)out_w_->data;
    if (out_w_->dtype == DT_I8 && out_scale_) out_scale_f_ = f.vec_f32(out_scale_ == tok_scale_ ? "tok_embeddings.scale" : "output.scale");
    if (out_w_->dtype != DT_I8 && !f.resident() && !lb::env_get("NO_HEAP_HEAD")) {
        const size_t n = (size_t)cfg.vocab_size * cfg.hidden_size;
        head_f16_.assign(head_ptr_, head_ptr_ + n);
        head_ptr_ = head_f16_.data();
    }

    draft_skip_.assign(L, 0);
    for (int i : cfg.draft_layer_skip) if (i >= 0 && i < L) draft_skip_[i] = 1;
    if (cfg.draft_layer_skip.empty()) {  // fallback: skip the middle half of the layers in draft mode
        for (int i = L / 4; i < L / 4 + L / 2; i++) draft_skip_[i] = 1;
    }

    const int hd = cfg.head_dim, rd = cfg.rotary_dim > 0 ? cfg.rotary_dim : hd, half = rd / 2;   // partial rotary (hybrid models): RoPE on the first rd dims only
    rope_half_ = half;
    rope_cos_.resize((size_t)max_ctx * half);
    rope_sin_.resize((size_t)max_ctx * half);
    for (int p = 0; p < max_ctx; p++)
        for (int i = 0; i < half; i++) {
            double inv = std::pow((double)cfg.rope_theta, -2.0 * i / rd);
            double a = p * inv;
            rope_cos_[(size_t)p * half + i] = (float)std::cos(a);
            rope_sin_[(size_t)p * half + i] = (float)std::sin(a);
        }
    // int8 KV: one scale per 16 dims (a per-head scale lets massive-activation sink tokens swamp the other dims: INT3 parity 0.994 -> 0.9998)
    kv_.nb = lb::env_get("KV_BLOCKS") ? std::atoi(lb::env_get("KV_BLOCKS")) : (hd % 16 == 0 ? hd / 16 : 1);
    kv_.init(kvmode, cfg.num_full_layers() + (mtp_l_ >= 0 ? 1 : 0), cfg.num_kv_heads, hd, max_ctx);   // hybrid models: KV only for the full-attention layers (+ the MTP layer)
}

void Transformer::linear(const LbLinear& w, const float* in, int T, float* out, bool use_res) {
    linear_forward(w, isa_, in, T, out, use_res, lscratch_, &pool_);
}

// Mixture-of-experts FFN: softmax router -> top-k -> (optionally renormalised) weights; tokens are grouped per expert
// (each expert runs once on its tokens) and accumulated per token in ascending expert order, so the result of a token
// never depends on what else is in the batch (batched verify == single-token decode, bit for bit).
// All selected experts of a chunk run through ONE parallel region for gate+up and ONE for down (linear_forward_jobs), nothing is
// allocated after warm-up (per-expert lists / scratch are members), and tokens are processed in chunks of kMoeChunk.
void Transformer::moe_ffn(const LayerWeights& w, int T, bool use_res) {
    const int H = cfg_.hidden_size, E = cfg_.num_experts, I = cfg_.moe_intermediate_size;
    const int layer_k = moe_layer_ >= 0 && moe_layer_ < (int)moe_k_layer.size() ? moe_k_layer[moe_layer_] : 0;
    const int want_k = layer_k > 0 ? layer_k : moe_k;
    const int Kfull = cfg_.num_experts_per_tok, K = want_k > 0 ? std::min(want_k, Kfull) : Kfull;
    constexpr int kMoeChunk = 256;
    if ((int)moe_list_.size() != E) { moe_list_.assign(E, {}); moe_off_.assign(E + 1, 0); }
    for (int c0 = 0; c0 < T; c0 += kMoeChunk) {
        const int Tc = std::min(kMoeChunk, T - c0);
        const float* hc = &h_[(size_t)c0 * H];
        float* oc = &o_[(size_t)c0 * H];
        // router logits for every (token, expert chunk of 16) in one region
        ProfScope psr(8);
        moe_lg_.resize((size_t)Tc * E);
        const int ech = (E + 15) / 16;
        pool_.run(Tc * ech, [&](int ti) {
            const int t = ti / ech, cc = ti % ech;
            const float* x = hc + (size_t)t * H;
            for (int e = cc * 16; e < std::min(E, cc * 16 + 16); e++) moe_lg_[(size_t)t * E + e] = dot_f16(x, w.router + (size_t)e * H, H);
        });
        psr.stop();
        ProfScope psq(9);
        for (auto& l : moe_list_) l.clear();
        moe_tk_e_.resize((size_t)Tc * K); moe_tk_w_.resize((size_t)Tc * K); moe_tk_i_.resize((size_t)Tc * K); moe_tk_n_.resize(Tc);
        // turbo routing (decode T=1 only unless prefill_prune): keep expert j >= min_k only while p[j] >= alpha * p[0] (and cumulative mass < cap)
        const MoeTurbo& mt = moe_turbo;
        const bool prune = mt.active() && (T == 1 || mt.prefill_prune);
        const bool edge = mt.edge_layers > 0 && (moe_layer_ < mt.edge_layers || moe_layer_ >= cfg_.num_layers - mt.edge_layers);
        int kmin_l = std::min(K, std::max(1, edge ? std::max(mt.min_k, mt.edge_min_k) : mt.min_k));
        int kmax_l = mt.kmax > 0 ? std::min(K, std::max(kmin_l, mt.kmax)) : K;
        if (!edge && moe_layer_ >= 0 && moe_layer_ < (int)mt.kmax_layers.size() && mt.kmax_layers[moe_layer_] > 0) {
            kmax_l = std::min(kmax_l, mt.kmax_layers[moe_layer_]); kmin_l = std::min(kmin_l, kmax_l);
        }
        std::vector<float> dbg_gap; if (g_dump_prefix) dbg_gap.assign(Tc, 0.f);
        pool_.run(Tc, [&](int t) {   // per-token softmax + top-k (independent per token -> parallel)
            static thread_local std::vector<float> pr_v;
            static thread_local std::vector<int> ord_v;
            pr_v.resize(E); ord_v.resize(E);
            float* pr = pr_v.data();
            int* ord = ord_v.data();
            const float* lg = &moe_lg_[(size_t)t * E];
            float mx = lg[0];
            for (int e = 1; e < E; e++) mx = std::max(mx, lg[e]);
            float sum = 0.f;
            for (int e = 0; e < E; e++) { pr[e] = std::exp(lg[e] - mx); sum += pr[e]; }
            for (int e = 0; e < E; e++) pr[e] /= sum;
            for (int e = 0; e < E; e++) ord[e] = e;
            std::partial_sort(ord, ord + Kfull, ord + E, [pr](int a, int b) { return pr[a] > pr[b] || (pr[a] == pr[b] && a < b); });
            if (g_dump_prefix) { float best = -INFINITY; for (int j = K; j < E; j++) best = std::max(best, lg[ord[j]]); dbg_gap[t] = lg[ord[K - 1]] - best; }
            int nk = K;
            if (prune) {
                nk = kmin_l;
                float cum = 0.f;
                for (int j = 0; j < nk; j++) cum += pr[ord[j]];
                const float thr = mt.alpha * pr[ord[0]];
                float mcap = mt.mass;
                if (mt.mass_topk) { float tk = 0.f; for (int j = 0; j < K; j++) tk += pr[ord[j]]; mcap *= tk; }
                while (nk < kmax_l && pr[ord[nk]] >= thr && cum < mcap) { cum += pr[ord[nk]]; nk++; }
            }
            moe_tk_n_[t] = nk;
            float norm = 1.f;
            if (cfg_.norm_topk_prob) { norm = 0.f; for (int j = 0; j < nk; j++) norm += pr[ord[j]]; }
            else if (K < Kfull) {   // --moe-k on an unnormalised router: keep the total weight of the full top-k
                float kept = 0.f, full = 0.f;
                for (int j = 0; j < Kfull; j++) { full += pr[ord[j]]; if (j < nk) kept += pr[ord[j]]; }
                norm = kept / full;
            }
            for (int j = 0; j < nk; j++) { moe_tk_e_[(size_t)t * K + j] = ord[j]; moe_tk_w_[(size_t)t * K + j] = pr[ord[j]] / norm; }
        });
        if (g_dump_prefix) if (FILE* fe = dump_file(2)) {
            for (int t = 0; t < Tc; t++) {
                int ex[64]; const int n = moe_tk_n_[t];
                for (int j = 0; j < K; j++) ex[j] = j < n ? moe_tk_e_[(size_t)t * K + j] : -1;
                std::sort(ex, ex + K);
                std::fwrite(ex, 4, K, fe); std::fwrite(&dbg_gap[t], 4, 1, fe);
            }
        }
        if (T == 1) { moe_stat_sel_ += moe_tk_n_[0]; moe_stat_tok_++; }
        for (int t = 0; t < Tc; t++)
            for (int j = 0; j < moe_tk_n_[t]; j++) {
                auto& l = moe_list_[moe_tk_e_[(size_t)t * K + j]];
                moe_tk_i_[(size_t)t * K + j] = (int)l.size();
                l.push_back({t, moe_tk_w_[(size_t)t * K + j]});
            }
        // gather rows per expert (contiguous), offsets in routed-row units
        int tot = 0;
        { uint64_t m0 = 0, m1 = 0; long uq = 0;
          for (int e = 0; e < E; e++) { moe_off_[e] = tot; const int ne = (int)moe_list_[e].size(); tot += ne; if (ne) { uq++; if (e < 64) m0 |= 1ull << e; else m1 |= 1ull << (e - 64); } }
          if ((int)moe_mask_last_.size() < 2 * cfg_.num_layers) moe_mask_last_.assign(2 * (size_t)cfg_.num_layers, 0);
          if (moe_layer_ >= 0 && moe_layer_ < cfg_.num_layers) { moe_mask_last_[2 * moe_layer_] = m0; moe_mask_last_[2 * moe_layer_ + 1] = m1; }
          moe_uniq_last_ += uq; }   // (E <= 128; chunked prefill forwards accumulate over chunks)
        moe_off_[E] = tot;
        eh_.resize((size_t)tot * H); ey_.resize((size_t)tot * H);
        moe_g_.resize((size_t)tot * I); moe_u_.resize((size_t)tot * I); moe_t_.resize((size_t)tot * I);
        moe_jobs_.clear();
        for (int e = 0; e < E; e++) {
            const auto& list = moe_list_[e];
            const int n = (int)list.size();
            if (n == 0) continue;
            LinJob jb;
            jb.nw = 2; jb.w[0] = &w.egate[e]; jb.w[1] = &w.eup[e];
            jb.x = &eh_[(size_t)moe_off_[e] * H]; jb.T = n;
            jb.y[0] = &moe_g_[(size_t)moe_off_[e] * I]; jb.y[1] = &moe_u_[(size_t)moe_off_[e] * I];
            moe_jobs_.push_back(jb);
        }
        if (tot > 1) pool_.run(E, [&](int e) {
            const auto& list = moe_list_[e];
            for (int i = 0; i < (int)list.size(); i++) std::memcpy(&eh_[(size_t)(moe_off_[e] + i) * H], hc + (size_t)list[i].first * H, (size_t)H * 4);
        });
        else for (int e = 0; e < E; e++) for (int i = 0; i < (int)moe_list_[e].size(); i++) std::memcpy(&eh_[(size_t)(moe_off_[e] + i) * H], hc + (size_t)moe_list_[e][i].first * H, (size_t)H * 4);
        psq.stop();
        { ProfScope p10(10); run_moe_jobs(); }
        { ProfScope p11(11);
          const size_t nel = (size_t)tot * I, sl = 4096;
          if (nel <= 2 * sl) silu_mul(moe_g_.data(), moe_u_.data(), (int)nel, moe_t_.data());
          else pool_.run((int)((nel + sl - 1) / sl), [&](int ci) { const size_t a = ci * sl, b = std::min(nel, a + sl); silu_mul(&moe_g_[a], &moe_u_[a], (int)(b - a), &moe_t_[a]); });
          if (teal_mode_) teal_apply(moe_t_.data(), (size_t)tot, I);
          if (shadow_full > 0 && !shadow_mask.empty() && moe_layer_ >= 0 && moe_layer_ < cfg_.num_layers)
              pool_.run(Tc, [&](int t) {   // every (token, expert) pair owns its own row of moe_t_
                  for (int j = shadow_full; j < moe_tk_n_[t]; j++) {
                      const int e = moe_tk_e_[(size_t)t * K + j];
                      const uint8_t* m = &shadow_mask[((size_t)moe_layer_ * E + e) * I];
                      float* v = &moe_t_[(size_t)(moe_off_[e] + moe_tk_i_[(size_t)t * K + j]) * I];
                      for (int i = 0; i < I; i++) v[i] = m[i] ? v[i] : 0.f;
                  }
              });
        }
        moe_jobs_.clear();
        for (int e = 0; e < E; e++) {
            const int n = (int)moe_list_[e].size();
            if (n == 0) continue;
            LinJob jb;
            jb.nw = 1; jb.w[0] = &w.edown[e];
            jb.x = &moe_t_[(size_t)moe_off_[e] * I]; jb.T = n;
            jb.y[0] = &ey_[(size_t)moe_off_[e] * H];
            moe_jobs_.push_back(jb);
        }
        { ProfScope p12(12); run_moe_jobs(); }
        ProfScope p13(13);
        pool_.run(Tc, [&](int t) {   // per token: experts in ascending order (same summation order as the per-expert loop)
            int ex[64], ix[64], n = moe_tk_n_[t];
            float wt[64];
            for (int j = 0; j < n; j++) { ex[j] = moe_tk_e_[(size_t)t * K + j]; ix[j] = moe_tk_i_[(size_t)t * K + j]; wt[j] = moe_tk_w_[(size_t)t * K + j]; }
            for (int a = 1; a < n; a++) for (int b = a; b > 0 && ex[b] < ex[b - 1]; b--) { std::swap(ex[b], ex[b - 1]); std::swap(ix[b], ix[b - 1]); std::swap(wt[b], wt[b - 1]); }
            float* o = oc + (size_t)t * H;
            std::fill(o, o + H, 0.f);
            for (int j = 0; j < n; j++) {
                const float* y = &ey_[(size_t)(moe_off_[ex[j]] + ix[j]) * H];
                for (int c = 0; c < H; c++) o[c] += wt[j] * y[c];
            }
        });
    }
}

bool Transformer::load_shadow_masks(const std::string& path) {
    const size_t need = (size_t)cfg_.num_layers * cfg_.num_experts * cfg_.moe_intermediate_size;
    shadow_mask.clear();
    FILE* f = need ? std::fopen(path.c_str(), "rb") : nullptr;
    if (!f) return false;
    std::vector<uint8_t> m(need);
    const bool ok = std::fread(m.data(), 1, need, f) == need && std::fgetc(f) == EOF;
    std::fclose(f);
    if (ok) shadow_mask = std::move(m);
    return ok;
}

void Transformer::teal_begin_calibration() {
    teal_hist_.assign((size_t)(cfg_.num_layers + 1) * kTealBins, 0);
    teal_mode_ = 1;
}

void Transformer::teal_finish_calibration(float P) {
    teal_mode_ = 0;
    teal_p_ = P;
    teal_thr_.assign((size_t)cfg_.num_layers + 1, 0.f);
    if (P <= 0.f || teal_hist_.empty()) return;
    for (int l = 0; l <= cfg_.num_layers; l++) {
        const uint64_t* h = &teal_hist_[(size_t)l * kTealBins];
        uint64_t total = 0, cum = 0;
        for (int b = 0; b < kTealBins; b++) total += h[b];
        for (int b = 0; b < kTealBins && total; b++) {
            cum += h[b];
            if ((double)cum >= (double)P * (double)total) { teal_thr_[l] = std::exp2((float)(b + 1) / 4.f - 30.f); break; }
        }
    }
    teal_mode_ = 2;
    teal_zeroed_ = teal_seen_ = 0;
    for (int w = 0; w < kTealWidths; w++) teal_blocks_[w] = teal_zero_blocks_[w] = 0;
}

void Transformer::teal_apply(float* a, size_t rows, int cols) {
    const int l = std::min(std::max(moe_layer_, 0), cfg_.num_layers);
    if (teal_mode_ == 1) {
        uint64_t* h = &teal_hist_[(size_t)l * kTealBins];
        for (size_t i = 0; i < rows * (size_t)cols; i++) {
            const float v = std::fabs(a[i]);
            const int b = v > 0.f ? (int)((std::log2(v) + 30.f) * 4.f) : 0;
            h[std::min(std::max(b, 0), kTealBins - 1)]++;
        }
        return;
    }
    static const bool stats = lb::env_get("TEAL_STATS") != nullptr;
    const float thr = teal_thr_[l];
    pool_.run((int)rows, [&](int r) {
        float* x = a + (size_t)r * cols;
        for (int i = 0; i < cols; i++) x[i] = std::fabs(x[i]) < thr ? 0.f : x[i];
    });
    if (!stats) return;
    static const int widths[kTealWidths] = {4, 16, 32, 256};
    for (size_t r = 0; r < rows; r++) {
        const float* x = a + r * cols;
        for (int i = 0; i < cols; i++) teal_zeroed_ += x[i] == 0.f;
        teal_seen_ += (uint64_t)cols;
        for (int w = 0; w < kTealWidths; w++)
            for (int i = 0; i + widths[w] <= cols; i += widths[w]) {
                bool zero = true;
                for (int j = 0; zero && j < widths[w]; j++) zero = x[i + j] == 0.f;
                teal_blocks_[w]++;
                teal_zero_blocks_[w] += zero;
            }
    }
}

std::string Transformer::teal_report() const {
    char buf[320];
    if (!teal_seen_) { std::snprintf(buf, sizeof buf, "teal P=%.2f (no statistics: set LB_TEAL_STATS=1)", teal_p_); return buf; }
    auto pc = [](uint64_t a, uint64_t b) { return b ? 100.0 * (double)a / (double)b : 0.0; };
    std::snprintf(buf, sizeof buf, "teal P=%.2f: zeroed %.2f%% of down_proj inputs; entirely zero aligned blocks: width 4 %.4f%%, 16 %.6f%%, 32 %.6f%%, 256 %.6f%% (of %llu width-16 blocks: %llu zero)",
                  teal_p_, pc(teal_zeroed_, teal_seen_), pc(teal_zero_blocks_[0], teal_blocks_[0]), pc(teal_zero_blocks_[1], teal_blocks_[1]),
                  pc(teal_zero_blocks_[2], teal_blocks_[2]), pc(teal_zero_blocks_[3], teal_blocks_[3]), (unsigned long long)teal_blocks_[1],
                  (unsigned long long)teal_zero_blocks_[1]);
    return buf;
}

void Transformer::attention(int layer, const float* q, int T, int start, float* out) {
    const int H = cfg_.num_heads, KVH = cfg_.num_kv_heads, hd = cfg_.head_dim;
    const int group = H / KVH;
    const float scale = cfg_.attention_scale();
    const bool i8 = kv_.mode == KvMode::Int8;
    static const bool old_attn = lb::env_get("ATTN_OLD") != nullptr;   // A/B switch: the per-head float x int8 reference kernel
    static const bool attn_check = lb::env_get("ATTN_CHECK") != nullptr;   // debug: new kernel vs old kernel vs fp64 reference on the real q/KV (stderr)
    static const double attn_jitter = lb::env_get("ATTN_JITTER") ? std::atof(lb::env_get("ATTN_JITTER")) : 0.0;   // debug: old kernel + relative noise of this size (noise control)
    const bool gqa_ok = i8 && hd == 128 && kv_.nb == 8 && (group == 1 || group == 2 || group == 4 || group == 8) && cpu_has_f16c_fma();
    if (gqa_ok && !old_attn && !attn_force_old_) {
        attention_gqa(layer, q, T, start, out);
        if (attn_check) {
            std::vector<float> oo((size_t)T * H * hd);
            attn_force_old_ = true; attention(layer, q, T, start, oo.data()); attn_force_old_ = false;
            double wn = 0, wo = 0, wno = 0, sn = 0, so = 0; long cnt = 0;
            const int stride = T > 8 ? 7 : 1;
            for (int t = 0; t < T; t += stride) for (int h = 0; h < H; h++) {
                const int kvh = h / group, pos = start + t;
                const size_t base = ((size_t)layer * KVH + kvh) * kv_.max_seq;
                std::vector<double> scd(pos + 1), ref(hd, 0.0);
                double mx = -1e300;
                for (int j = 0; j <= pos; j++) {
                    double d = 0;
                    for (int b = 0; b < kv_.nb; b++) for (int i = 0; i < hd / kv_.nb; i++) { const int c = b * (hd / kv_.nb) + i; d += (double)q[((size_t)t * H + h) * hd + c] * ((double)kv_.k8[(base + j) * hd + c] * (double)kv_.ks[(base + j) * kv_.nb + b]); }
                    scd[j] = d * (double)scale; mx = std::max(mx, scd[j]);
                }
                double sum = 0;
                for (int j = 0; j <= pos; j++) { scd[j] = std::exp(scd[j] - mx); sum += scd[j]; }
                for (int j = 0; j <= pos; j++) for (int b = 0; b < kv_.nb; b++) for (int i = 0; i < hd / kv_.nb; i++) { const int c = b * (hd / kv_.nb) + i; ref[c] += scd[j] / sum * ((double)kv_.v8[(base + j) * hd + c] * (double)kv_.vs[(base + j) * kv_.nb + b]); }
                double en = 0, eo = 0, eno = 0, nr = 0;
                const float* xn = out + ((size_t)t * H + h) * hd; const float* xo = oo.data() + ((size_t)t * H + h) * hd;
                for (int c = 0; c < hd; c++) { en += (xn[c] - ref[c]) * (xn[c] - ref[c]); eo += (xo[c] - ref[c]) * (xo[c] - ref[c]); eno += (double)(xn[c] - xo[c]) * (xn[c] - xo[c]); nr += ref[c] * ref[c]; }
                nr = std::sqrt(nr) + 1e-30;
                wn = std::max(wn, std::sqrt(en) / nr); wo = std::max(wo, std::sqrt(eo) / nr); wno = std::max(wno, std::sqrt(eno) / nr);
                sn += std::sqrt(en) / nr; so += std::sqrt(eo) / nr; cnt++;
            }
            std::fprintf(stderr, "[attn-check] layer %d T %d start %d: rel L2 err vs fp64 new worst %.3g mean %.3g | old worst %.3g mean %.3g | new-vs-old worst %.3g\n", layer, T, start, wn, sn / cnt, wo, so / cnt, wno);
        }
        return;
    }
    pool_.run(T * H, [&](int task) {
        const int t = task / H, h = task % H, kvh = h / group;
        const int pos = start + t;
        const float* qh = q + ((size_t)t * H + h) * hd;
        const size_t base = ((size_t)layer * KVH + kvh) * kv_.max_seq;
        static thread_local std::vector<float> sc;
        sc.resize((size_t)pos + 1);
        for (int j = 0; j <= pos; j++) {
            if (i8) {
                const int nb = kv_.nb;
                sc[j] = (nb == 1 ? dot_f32_i8(qh, &kv_.k8[(base + j) * hd], hd) * kv_.ks[base + j]
                                 : dot_f32_i8_blocks(qh, &kv_.k8[(base + j) * hd], &kv_.ks[(base + j) * nb], hd / nb, hd)) * scale;
            }
            else {
                const float* kr = &kv_.kf[(base + j) * hd];
                float s = 0.f;
                for (int d = 0; d < hd; d++) s += qh[d] * kr[d];
                sc[j] = s * scale;
            }
        }
        softmax_inplace(sc.data(), pos + 1);
        float* oh = out + ((size_t)t * H + h) * hd;
        for (int d = 0; d < hd; d++) oh[d] = 0.f;
        for (int j = 0; j <= pos; j++) {
            if (i8) {
                const int nb = kv_.nb, bs = hd / nb;
                const int8_t* vr = &kv_.v8[(base + j) * hd];
                if (nb == 1) { const float w = sc[j] * kv_.vs[base + j]; for (int d = 0; d < hd; d++) oh[d] += w * (float)vr[d]; }
                else axpy_i8_blocks(oh, sc[j], vr, &kv_.vs[(base + j) * nb], bs, hd);
            } else {
                const float w = sc[j];
                const float* vr = &kv_.vf[(base + j) * hd];
                for (int d = 0; d < hd; d++) oh[d] += w * vr[d];
            }
        }
    });
    if (attn_jitter != 0.0 && !attn_force_old_) {   // noise control: deterministic pseudo-random relative perturbation of the old kernel output
        uint32_t hsh = 2166136261u ^ (uint32_t)layer;
        for (size_t i = 0; i < (size_t)T * H * hd; i++) { hsh = hsh * 1664525u + 1013904223u; out[i] *= 1.f + (float)(attn_jitter * ((hsh >> 8) * (1.0 / 8388608.0) - 1.0)); }
    }
}

// Grouped-query attention: one task handles the G query heads that share a KV head, so every K/V row is read and dequantised once for all of them.
// Positions are cut into fixed chunks of kCS (absolute boundaries, independent of T and the thread count -> batched == single-token, bit for bit),
// each chunk gives an online-softmax partial (max, sum, weighted V); partials are merged in chunk order. With few (token, kv-head) pairs the chunks run
// as separate tasks (flash-decoding style; the last finisher of a (token, kv-head) merges), otherwise one task walks all chunks of its (token, kv-head).
void Transformer::attention_gqa(int layer, const float* q, int T, int start, float* out) {
    const int H = cfg_.num_heads, KVH = cfg_.num_kv_heads, G = H / KVH;
    constexpr int kCS = 128, kPart = 16 + 8 * 128;
    const float scale = cfg_.attention_scale();   // head_dim == 128 here
    const int maxn = (start + T - 1) / kCS + 1;
    const bool split = (long)T * KVH < 4L * pool_.size() && maxn > 1;
    if (split) { attn_part_.resize((size_t)T * KVH * maxn * kPart); attn_cnt_.assign((size_t)T * KVH, 0); }
    auto chunk = [&](int t, int g, int c, float* part) {
        const int n = std::min(kCS, start + t + 1 - c * kCS);
        static thread_local std::vector<float> scr, qs;
        scr.resize((size_t)kCS * 8 + 32 * 128 + 64);
        qs.resize((size_t)G * 128);
        const float* qh = q + ((size_t)t * H + (size_t)g * G) * 128;
        for (int i = 0; i < G * 128; i++) qs[i] = qh[i] * scale;
        const size_t base = ((size_t)layer * KVH + g) * kv_.max_seq + (size_t)c * kCS;
        attn_gqa_chunk(G, qs.data(), &kv_.k8[base * 128], &kv_.ks[base * 8], &kv_.v8[base * 128], &kv_.vs[base * 8], n, part, part + 8, part + 16, scr.data());
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
            const int nc = (start + t) / kCS + 1;
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
            const int g = ti % KVH, t = T - 1 - ti / KVH;   // longest contexts first
            const int nc = (start + t) / kCS + 1;
            static thread_local std::vector<float> run, part;
            run.resize(kPart); part.resize(kPart);
            chunk(t, g, 0, run.data());
            for (int c = 1; c < nc; c++) { chunk(t, g, c, part.data()); merge(run.data(), part.data()); }
            finish(t, g, run.data());
        });
    }
}

void Transformer::enable_int8_head() {
    if (head8p_) return;
    if (out_w_->dtype == DT_I8) {  // already int8 rows: use them directly with int8 activations (no copy)
        if (out_scale_f_.empty()) out_scale_f_.assign(cfg_.vocab_size, 1.f);
        head8p_ = (const int8_t*)out_w_->data;
        head8s_ = out_scale_f_.data();
        return;
    }
    if (!head8_.empty()) return;
    const int V = cfg_.vocab_size, H = cfg_.hidden_size;
    head8_.resize((size_t)V * H);
    head8_scale_.resize(V);
    const uint16_t* src = (const uint16_t*)out_w_->data;
    pool_.run((V + 255) / 256, [&](int ti) {
        const int r0 = ti * 256, r1 = std::min(V, r0 + 256);
        std::vector<float> row(H);
        for (int r = r0; r < r1; r++) {
            float mx = 0.f;
            for (int i = 0; i < H; i++) { row[i] = f16_to_f32(src[(size_t)r * H + i]); mx = std::max(mx, std::fabs(row[i])); }
            float sc = mx > 0.f ? mx / 127.f : 1.f;
            head8_scale_[r] = sc;
            for (int i = 0; i < H; i++) head8_[(size_t)r * H + i] = (int8_t)std::lrintf(row[i] / sc);
        }
    });
    head8p_ = head8_.data();
    head8s_ = head8_scale_.data();
}

void Transformer::set_draft_vocab(const std::vector<int>& ids) {
    draft_vocab_.clear();
    for (int id : ids) if (id >= 0 && id < cfg_.vocab_size) draft_vocab_.push_back(id);
    std::sort(draft_vocab_.begin(), draft_vocab_.end());
    draft_vocab_.erase(std::unique(draft_vocab_.begin(), draft_vocab_.end()), draft_vocab_.end());
}

void Transformer::set_head_vocab(const std::vector<int>& ids) {
    head_vocab_.clear();
    for (int id : ids) if (id >= 0 && id < cfg_.vocab_size) head_vocab_.push_back(id);
    std::sort(head_vocab_.begin(), head_vocab_.end());
    head_vocab_.erase(std::unique(head_vocab_.begin(), head_vocab_.end()), head_vocab_.end());
}

#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx2")))
static void head4_dots_avx2(const uint8_t* w4, int rows, int hb, const int8_t* x, int32_t* out) {   // out[r] = sum_i n[r][i] * x[i] with n in 0..15
    const __m256i m0f = _mm256_set1_epi8(15), ones = _mm256_set1_epi16(1);
    int r = 0;
    for (; r + 4 <= rows; r += 4) {
        __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
        const uint8_t* p0 = w4 + (size_t)r * hb;
        for (int k = 0; k < hb; k += 32) {
            const __m256i xl = _mm256_loadu_si256((const __m256i*)(x + k)), xh = _mm256_loadu_si256((const __m256i*)(x + hb + k));
#define H4ROW(A, P) { const __m256i w = _mm256_loadu_si256((const __m256i*)((P) + k)); \
            const __m256i pp = _mm256_add_epi16(_mm256_maddubs_epi16(_mm256_and_si256(w, m0f), xl), _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(w, 4), m0f), xh)); \
            A = _mm256_add_epi32(A, _mm256_madd_epi16(pp, ones)); }
            H4ROW(a0, p0) H4ROW(a1, p0 + hb) H4ROW(a2, p0 + 2 * (size_t)hb) H4ROW(a3, p0 + 3 * (size_t)hb)
#undef H4ROW
        }
        __m256i aa[4] = {a0, a1, a2, a3};
        for (int j = 0; j < 4; j++) {
            __m128i lo = _mm_add_epi32(_mm256_castsi256_si128(aa[j]), _mm256_extracti128_si256(aa[j], 1));
            lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0x4E));
            lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0xB1));
            out[r + j] = _mm_cvtsi128_si32(lo);
        }
    }
    for (; r < rows; r++) {
        __m256i a = _mm256_setzero_si256();
        const uint8_t* p0 = w4 + (size_t)r * hb;
        for (int k = 0; k < hb; k += 32) {
            const __m256i w = _mm256_loadu_si256((const __m256i*)(p0 + k));
            const __m256i pp = _mm256_add_epi16(_mm256_maddubs_epi16(_mm256_and_si256(w, m0f), _mm256_loadu_si256((const __m256i*)(x + k))),
                                                _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(w, 4), m0f), _mm256_loadu_si256((const __m256i*)(x + hb + k))));
            a = _mm256_add_epi32(a, _mm256_madd_epi16(pp, ones));
        }
        __m128i lo = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
        lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0x4E));
        lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0xB1));
        out[r] = _mm_cvtsi128_si32(lo);
    }
}
// 2-bit asymmetric screening rows (see h2_ in transformer.h): est[r] = xs * (sum_g scale_g * (q_g . x_g) + min_g * xg[g]) + xos * rs[r] * (wo[r] . xo)
// xg[g] = sum of the int8 activations of group g; wo = exact int8 weights of the K outlier channels, xo = their int16 activations (K multiple of 16, may be 0).
__attribute__((target("avx2,fma,f16c")))
static void head2_dots_avx2(const uint8_t* w2, const uint16_t* sm, int rows, int H, const int8_t* x, const float* xg, float xs,
                            const int8_t* wo, const int16_t* xo, int K, float xos, const float* rsc, float* out) {
    const int NG = H / 128, hq = H / 4;
    const __m256i m3 = _mm256_set1_epi8(3), ones = _mm256_set1_epi16(1);
    for (int r = 0; r < rows; r++) {
        const uint8_t* p = w2 + (size_t)r * hq;
        const uint16_t* s = sm + (size_t)r * 2 * NG;
        __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0;
        for (int g0 = 0; g0 < NG; g0 += 8) {
            const __m256 sv = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(s + g0)));
            acc1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(s + NG + g0))), _mm256_loadu_ps(xg + g0), acc1);
            for (int j = 0; j < 8; j += 2) {
#define H2GRP(J, A) { const int g = g0 + (J); const __m256i w = _mm256_loadu_si256((const __m256i*)(p + (size_t)g * 32)); const int8_t* xx = x + (size_t)g * 128; \
                __m256i t = _mm256_maddubs_epi16(_mm256_and_si256(w, m3), _mm256_loadu_si256((const __m256i*)xx)); \
                t = _mm256_add_epi16(t, _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(w, 2), m3), _mm256_loadu_si256((const __m256i*)(xx + 32)))); \
                t = _mm256_add_epi16(t, _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(w, 4), m3), _mm256_loadu_si256((const __m256i*)(xx + 64)))); \
                t = _mm256_add_epi16(t, _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(w, 6), m3), _mm256_loadu_si256((const __m256i*)(xx + 96)))); \
                A = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(t, ones)), _mm256_permutevar8x32_ps(sv, _mm256_set1_epi32(J)), A); }
                H2GRP(j, acc0) H2GRP(j + 1, acc1)
#undef H2GRP
            }
        }
        __m256 a = _mm256_mul_ps(_mm256_add_ps(acc0, acc1), _mm256_set1_ps(xs));
        if (K > 0) {
            __m256i io = _mm256_setzero_si256();
            const int8_t* pw = wo + (size_t)r * K;
            for (int k = 0; k < K; k += 16)
                io = _mm256_add_epi32(io, _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(pw + k))), _mm256_loadu_si256((const __m256i*)(xo + k))));
            a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(io), _mm256_set1_ps(xos * (rsc ? rsc[r] : 1.f)), a);
        }
        __m128 lo = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
        lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
        out[r] = _mm_cvtss_f32(lo);
    }
}
#endif

void Transformer::enable_screen_head(int bits) {
    if (screen_head_) return;
#if defined(__x86_64__) || defined(_M_X64)
    if (out_w_->dtype != DT_I8 || head8p_ || !cpu_has_f16c_fma()) return;
    const int V = cfg_.vocab_size, H = cfg_.hidden_size, hb = H / 2;
    if (H % 64 != 0 || H > 4096) return;
    if (bits == 2 && H % 1024 != 0) bits = 4;
    screen_bits_ = bits == 2 ? 2 : 4;
    h4_eps_.resize(V); h4_wn_.resize(V);
    const int8_t* w8 = (const int8_t*)out_w_->data;
    if (screen_bits_ == 2) {
        const int NG = H / 128, hq = H / 4;
        int K = 32;
        if (const char* e = lb::env_get("SCREEN2_K")) K = std::max(0, std::min(256, std::atoi(e))) / 16 * 16;
        K = std::min(K, H / 2);
        screen2_k_ = K;
        bool bias_fix = true;
        if (const char* e = lb::env_get("SCREEN2_BIAS")) bias_fix = e[0] != '0';
        std::vector<uint8_t> isout(H, 0);
        {
            std::vector<int> ord(H);
            for (int i = 0; i < H; i++) ord[i] = i;
            std::stable_sort(ord.begin(), ord.end(), [&](int x, int y) { return std::fabs(out_norm_[x]) > std::fabs(out_norm_[y]); });
            h2_ch_.assign(ord.begin(), ord.begin() + K);
            std::sort(h2_ch_.begin(), h2_ch_.end());
            for (int c : h2_ch_) isout[c] = 1;
        }
        h2_.assign((size_t)V * hq, 0);
        h2_sm_.assign((size_t)V * 2 * NG, 0);
        h2o_.assign((size_t)V * std::max(K, 1), 0);
        pool_.run((V + 255) / 256, [&](int ti) {
            const int r0 = ti * 256, r1 = std::min(V, r0 + 256);
            for (int r = r0; r < r1; r++) {
                const int8_t* w = w8 + (size_t)r * H;
                const float rs = out_scale_f_.empty() ? 1.f : out_scale_f_[r];
                double ww = 0.0, res = 0.0;
                uint8_t* o = &h2_[(size_t)r * hq];
                uint16_t* sm = &h2_sm_[(size_t)r * 2 * NG];
                for (int k = 0; k < K; k++) h2o_[(size_t)r * K + k] = w[h2_ch_[k]];
                for (int i = 0; i < H; i++) ww += (double)w[i] * w[i];
                for (int g = 0; g < NG; g++) {
                    const int8_t* wg = w + g * 128;
                    const uint8_t* og = &isout[g * 128];
                    float lo = 127.f, hi = -128.f, mean = 0.f, var = 0.f;
                    int n = 0;
                    for (int i = 0; i < 128; i++) if (!og[i]) { const float v = wg[i]; lo = std::min(lo, v); hi = std::max(hi, v); mean += v; n++; }
                    uint8_t bq[128] = {0};
                    if (n == 0) { sm[g] = 0; sm[NG + g] = 0; for (int jj = 0; jj < 32; jj++) o[g * 32 + jj] = 0; continue; }
                    mean /= (float)n;
                    for (int i = 0; i < 128; i++) if (!og[i]) var += (wg[i] - mean) * (wg[i] - mean);
                    const float sd = std::sqrt(var / (float)n);
                    // asym 4-level grid over the non-outlier channels: two starts (min/max, mean +- 1.5 sd), alternating assign / least-squares (scale, min) refits;
                    // the f16-rounded result with the smallest squared error wins (codes re-assigned on the rounded grid = the residual the kernel really sees)
                    double best = 1e300; uint16_t bs = 0, bm = 0;
                    for (int start = 0; start < 2; start++) {
                        float m = start == 0 ? lo : std::max(lo, mean - 1.5f * sd), sc = start == 0 ? (hi - lo) / 3.f : (std::min(hi, mean + 1.5f * sd) - m) / 3.f;
                        uint8_t q[128] = {0};
                        for (int it = 0; it < 6 && sc > 0.f; it++) {
                            double sq = 0, sqq = 0, sw = 0, sqw = 0;
                            for (int i = 0; i < 128; i++) {
                                if (og[i]) continue;
                                const int qq = std::max(0, std::min(3, (int)std::lrintf((wg[i] - m) / sc)));
                                sq += qq; sqq += qq * qq; sw += wg[i]; sqw += (double)qq * wg[i];
                            }
                            const double det = n * sqq - sq * sq;
                            if (det <= 0) break;
                            const double s2 = (n * sqw - sq * sw) / det;
                            if (s2 <= 0) break;
                            sc = (float)s2; m = (float)((sw - s2 * sq) / n);
                        }
                        if (!(sc > 0.f)) { sc = 0.f; m = mean; }
                        const uint16_t hs = f32_to_f16(sc * rs), hm = f32_to_f16(m * rs);
                        const float scr = f16_to_f32(hs) / rs, mr = f16_to_f32(hm) / rs;
                        double e = 0.0;
                        for (int i = 0; i < 128; i++) {
                            if (og[i]) { q[i] = 0; continue; }
                            const int qq = scr > 0.f ? std::max(0, std::min(3, (int)std::lrintf((wg[i] - mr) / scr))) : 0;
                            q[i] = (uint8_t)qq;
                            const double d = wg[i] - (mr + scr * qq);
                            e += d * d;
                        }
                        if (e < best) { best = e; bs = hs; bm = hm; std::memcpy(bq, q, 128); }
                    }
                    res += best;
                    sm[g] = bs; sm[NG + g] = bm;
                    for (int jj = 0; jj < 32; jj++) o[g * 32 + jj] = (uint8_t)(bq[jj] | (bq[jj + 32] << 2) | (bq[jj + 64] << 4) | (bq[jj + 96] << 6));
                }
                if (bias_fix) {
                    // Least-squares residuals are orthogonal to the fit, so w.w_hat = |w_hat|^2 < |w|^2: for a hidden state ALIGNED with the row (exactly the rows near
                    // the top) x.w_hat underestimates x.w by |d|^2 / |w_hat|^2 (~10 % for 2 bits). Scale the row's grid by c = |w|^2 / (w.w_hat): unbiased along w,
                    // the residual for random directions grows by sqrt(c) (accounted for in eps below).
                    auto code = [&](int g, int i) { return (o[g * 32 + (i & 31)] >> (2 * (i >> 5))) & 3; };
                    double dww = 0.0, dwh = 0.0;
                    for (int g = 0; g < NG; g++) {
                        const float sc = f16_to_f32(sm[g]) / rs, mn = f16_to_f32(sm[NG + g]) / rs;
                        for (int i = 0; i < 128; i++) if (!isout[g * 128 + i]) { const double v = w[g * 128 + i], wh = mn + sc * code(g, i); dww += v * v; dwh += v * wh; }
                    }
                    const double c = dwh > 0.0 ? dww / dwh : 1.0;
                    res = 0.0;
                    for (int g = 0; g < NG; g++) {
                        sm[g] = f32_to_f16((float)(f16_to_f32(sm[g]) * c)); sm[NG + g] = f32_to_f16((float)(f16_to_f32(sm[NG + g]) * c));
                        const float sc = f16_to_f32(sm[g]) / rs, mn = f16_to_f32(sm[NG + g]) / rs;
                        for (int i = 0; i < 128; i++) if (!isout[g * 128 + i]) { const double d = w[g * 128 + i] - (mn + sc * code(g, i)); res += d * d; }
                    }
                }
                h4_eps_[r] = rs * (float)std::sqrt(res / (H - K));
                h4_wn_[r] = rs * (float)std::sqrt(ww / H);
            }
        });
        screen_head_ = true;
        return;
    }
    h4_.assign((size_t)V * hb, 0);
    h4_step_.resize(V);
    pool_.run((V + 255) / 256, [&](int ti) {
        const int r0 = ti * 256, r1 = std::min(V, r0 + 256);
        for (int r = r0; r < r1; r++) {
            const int8_t* w = w8 + (size_t)r * H;
            int amax = 0;
            double ww = 0.0;
            for (int i = 0; i < H; i++) { amax = std::max(amax, std::abs((int)w[i])); ww += (double)w[i] * w[i]; }
            const float step = amax > 0 ? (float)amax / 7.f : 1.f;
            double res = 0.0;
            uint8_t* o = &h4_[(size_t)r * hb];
            int q[4096];
            for (int i = 0; i < H; i++) {
                const int qq = std::max(-7, std::min(7, (int)std::lrintf((float)w[i] / step)));
                q[i] = qq + 8;
                const double d = (double)w[i] - (double)qq * step;
                res += d * d;
            }
            for (int jj = 0; jj < hb; jj++) o[jj] = (uint8_t)(q[jj] | (q[jj + hb] << 4));
            const float rs = out_scale_f_.empty() ? 1.f : out_scale_f_[r];
            h4_step_[r] = rs * step;
            h4_eps_[r] = rs * (float)std::sqrt(res / H);
            h4_wn_[r] = rs * (float)std::sqrt(ww / H);
        }
    });
    screen_head_ = true;
#else
    (void)bits;
#endif
}

void Transformer::screen_prep(const float* x, ScreenX* sx) const {
    const int H = cfg_.hidden_size;
    sx->xq.resize(H);
    if (screen_bits_ == 2) {   // outlier channels: int16 activations, scored exactly; the rest: int8 with its own (much finer) scale
        const int K = screen2_k_, NG = H / 128;
        sx->xo.assign(std::max(K, 16), 0);
        float mo = 0.f, mx = 0.f;
        double ss = 0.0, dd = 0.0;
        std::vector<uint8_t> isout(H, 0);
        for (int k = 0; k < K; k++) { isout[h2_ch_[k]] = 1; mo = std::max(mo, std::fabs(x[h2_ch_[k]])); }
        for (int i = 0; i < H; i++) if (!isout[i]) { mx = std::max(mx, std::fabs(x[i])); ss += (double)x[i] * x[i]; }
        sx->xos = mo > 0.f ? mo / 32767.f : 1.f;
        for (int k = 0; k < K; k++) {
            const float v = x[h2_ch_[k]];
            const int q = std::max(-32767, std::min(32767, (int)std::lrintf(v / sx->xos)));
            sx->xo[k] = (int16_t)q;
            const double d = v - (double)q * sx->xos;
            dd += d * d;
        }
        sx->xs = mx > 0.f ? mx / 127.f : 1.f;
        for (int i = 0; i < H; i++) {
            if (isout[i]) { sx->xq[i] = 0; continue; }
            const int q = (int)std::lrintf(x[i] / sx->xs);
            sx->xq[i] = (int8_t)q;
            const double d = x[i] - (double)q * sx->xs;
            dd += d * d;
        }
        sx->sumx = 0; sx->xn = (float)std::sqrt(ss); sx->dxn = (float)std::sqrt(dd);
        sx->xg.assign(NG, 0.f);
        for (int g = 0; g < NG; g++) { int32_t t = 0; for (int i = 0; i < 128; i++) t += sx->xq[g * 128 + i]; sx->xg[g] = (float)t; }
        return;
    }
    float mx = 0.f;
    double ss = 0.0;
    for (int i = 0; i < H; i++) { mx = std::max(mx, std::fabs(x[i])); ss += (double)x[i] * x[i]; }
    sx->xs = mx > 0.f ? mx / 127.f : 1.f;
    int32_t sumx = 0;
    double dd = 0.0;
    for (int i = 0; i < H; i++) { const int q = (int)std::lrintf(x[i] / sx->xs); sx->xq[i] = (int8_t)q; sumx += q; const double d = x[i] - (double)q * sx->xs; dd += d * d; }
    sx->sumx = sumx; sx->xn = (float)std::sqrt(ss); sx->dxn = (float)std::sqrt(dd);
}

void Transformer::screen_chunk(int r0, int r1, const ScreenX& sx, float* est) const {
#if defined(__x86_64__) || defined(_M_X64)
    const int H = cfg_.hidden_size;
    if (screen_bits_ == 2) {
        const int NG = H / 128;
        head2_dots_avx2(&h2_[(size_t)r0 * (H / 4)], &h2_sm_[(size_t)r0 * 2 * NG], r1 - r0, H, sx.xq.data(), sx.xg.data(), sx.xs,
                        h2o_.data() + (size_t)r0 * screen2_k_, sx.xo.data(), screen2_k_, sx.xos, out_scale_f_.empty() ? nullptr : out_scale_f_.data() + r0, est);
        return;
    }
    int32_t buf[256];
    head4_dots_avx2(&h4_[(size_t)r0 * (H / 2)], r1 - r0, H / 2, sx.xq.data(), buf);
    for (int r = r0; r < r1; r++) est[r - r0] = (float)(buf[r - r0] - 8 * sx.sumx) * h4_step_[r] * sx.xs;
#else
    (void)r0; (void)r1; (void)sx; (void)est;
#endif
}

void Transformer::screen_estimate(const float* hidden, float* est, float* sigma) {
    if (!screen_head_) fail("screen_estimate needs --screen-head");
    const int V = cfg_.vocab_size;
    ScreenX sx;
    screen_prep(hidden, &sx);
    pool_.run((V + 255) / 256, [&](int ti) {
        const int r0 = ti * 256, r1 = std::min(V, r0 + 256);
        screen_chunk(r0, r1, sx, est + r0);
        for (int r = r0; r < r1; r++) { const float a = sx.xn * h4_eps_[r], b = sx.dxn * h4_wn_[r]; sigma[r] = std::sqrt(a * a + b * b); }
    });
}

// Single-token greedy head with 4-bit screening (see enable_screen_head). Writes -inf to every row that is statistically excluded.
void Transformer::lm_head_screen(const float* hidden, float* logits) {
#if defined(__x86_64__) || defined(_M_X64)
    const int V = cfg_.vocab_size, H = cfg_.hidden_size;
    ScreenX sx;
    screen_prep(hidden, &sx);
    const float xn = sx.xn, dxn = sx.dxn;
    const int8_t* w8 = (const int8_t*)out_w_->data;
    const int chunk = 256, ntask = (V + chunk - 1) / chunk;
    std::vector<float> est((size_t)V), pm(ntask, -INFINITY);
    const float sg = screen_sg();
    auto margin = [&](int r) { const float a = xn * h4_eps_[r], b = dxn * h4_wn_[r]; return sg * std::sqrt(a * a + b * b); };
    pool_.run(ntask, [&](int ti) {
        const int r0 = ti * chunk, r1 = std::min(V, r0 + chunk);
        screen_chunk(r0, r1, sx, &est[r0]);
        float best = -INFINITY;
        for (int r = r0; r < r1; r++) best = std::max(best, est[r] - margin(r));
        pm[ti] = best;
    });
    if (screen_est_out_) { std::memcpy(screen_est_out_, est.data(), (size_t)V * 4); return; }   // early-exit probe: raw screening scores, no rerank
    float lb = -INFINITY;
    for (float v : pm) lb = std::max(lb, v);
    std::atomic<long> ncand{0};
    pool_.run(ntask, [&](int ti) {
        const int r0 = ti * chunk, r1 = std::min(V, r0 + chunk);
        long nc = 0;
        for (int r = r0; r < r1; r++) {
            if (est[r] + margin(r) >= lb) { logits[r] = dot_f32_i8(hidden, w8 + (size_t)r * H, H) * (out_scale_f_.empty() ? 1.f : out_scale_f_[r]); nc++; }
            else logits[r] = -INFINITY;
        }
        ncand += nc;
    });
    screen_cand_sum += ncand.load(); screen_cand_max = std::max(screen_cand_max, ncand.load()); screen_cand_calls++;
    if (g_prof_on && lb::env_get("HEAD_STATS")) std::fprintf(stderr, "screen head: %ld candidates\n", ncand.load());
#else
    (void)hidden; (void)logits;
#endif
}

// REST union head: the same screening rule as lm_head_screen, for Tl tokens that share one pass over the 4-bit head. Each 256-row chunk (256 KB) is streamed from DRAM once
// and re-used from L2 for every token. cand[t] >= 0 is the drafted token of position t: its exact int8 score is computed first and raises the pruning bound of that
// position (lb = max(best 4-bit lower bound, exact candidate score): both are lower bounds of the row maximum), and the candidate row is always scored exactly. Non-survivors get
// -inf; survivors get the same exact dot_f32_i8 value as the full head, so the argmax equals lm_head_screen's argmax at T=1 (same margins) and the full head's whenever the
// screening margin holds.
void Transformer::lm_head_screen_multi(const float* hidden, int Tl, const int* cand, float* logits) {
#if defined(__x86_64__) || defined(_M_X64)
    const int V = cfg_.vocab_size, H = cfg_.hidden_size;
    std::vector<ScreenX> sx(Tl);
    std::vector<float> lb(Tl, -INFINITY);
    for (int t = 0; t < Tl; t++) screen_prep(hidden + (size_t)t * H, &sx[t]);
    const int8_t* w8 = (const int8_t*)out_w_->data;
    auto rscale = [&](int r) { return out_scale_f_.empty() ? 1.f : out_scale_f_[r]; };
    std::vector<float> cs(Tl, -INFINITY);   // exact candidate scores
    for (int t = 0; t < Tl; t++) if (cand && cand[t] >= 0 && cand[t] < V) cs[t] = dot_f32_i8(hidden + (size_t)t * H, w8 + (size_t)cand[t] * H, H) * rscale(cand[t]);
    const int chunk = 256, ntask = (V + chunk - 1) / chunk;
    scr_est_.resize((size_t)Tl * V);
    float* est = scr_est_.data();
    std::vector<float> pm((size_t)ntask * Tl, -INFINITY);
    const float sg = screen_sg();
    pool_.run(ntask, [&](int ti) {
        const int r0 = ti * chunk, r1 = std::min(V, r0 + chunk);
        for (int t = 0; t < Tl; t++) {
            float* e = est + (size_t)t * V;
            screen_chunk(r0, r1, sx[t], e + r0);
            float best = -INFINITY;
            for (int r = r0; r < r1; r++) {
                const float a = sx[t].xn * h4_eps_[r], b = sx[t].dxn * h4_wn_[r];
                best = std::max(best, e[r] - sg * std::sqrt(a * a + b * b));
            }
            pm[(size_t)ti * Tl + t] = best;
        }
    });
    for (int t = 0; t < Tl; t++) {
        float m = cs[t];
        for (int ti = 0; ti < ntask; ti++) m = std::max(m, pm[(size_t)ti * Tl + t]);
        lb[t] = m;
    }
    pool_.run(ntask, [&](int ti) {
        const int r0 = ti * chunk, r1 = std::min(V, r0 + chunk);
        for (int t = 0; t < Tl; t++) {
            const float* e = est + (size_t)t * V;
            float* lo = logits + (size_t)t * V;
            const int cr = cand ? cand[t] : -1;
            for (int r = r0; r < r1; r++) {
                const float a = sx[t].xn * h4_eps_[r], b = sx[t].dxn * h4_wn_[r];
                if (e[r] + sg * std::sqrt(a * a + b * b) >= lb[t]) lo[r] = r == cr ? cs[t] : dot_f32_i8(hidden + (size_t)t * H, w8 + (size_t)r * H, H) * rscale(r);
                else lo[r] = -INFINITY;
            }
        }
    });
#else
    (void)hidden; (void)Tl; (void)cand; (void)logits;
#endif
}

void Transformer::enable_exact_head() {
    if (!xq8_.empty() || out_w_->dtype == DT_I8) return;
    const int V = cfg_.vocab_size, H = cfg_.hidden_size;
    xq8_.resize((size_t)V * H);
    xsc_.resize(V);
    xeps_.resize(V);
    xnw_.resize(V);
    xsw_.resize(V);
    pool_.run((V + 255) / 256, [&](int ti) {
        const int r0 = ti * 256, r1 = std::min(V, r0 + 256);
        std::vector<float> row(H);
        for (int r = r0; r < r1; r++) {
            const uint16_t* w = head_ptr_ + (size_t)r * H;
            float amax = 0.f, nn = 0.f;
            for (int i = 0; i < H; i++) { row[i] = f16_to_f32(w[i]); amax = std::max(amax, std::fabs(row[i])); nn += row[i] * row[i]; }
            const float sc = amax > 0.f ? amax / 127.f : 1.f;
            double res = 0.0, qq = 0.0;
            int32_t sw = 0;
            for (int i = 0; i < H; i++) {
                int q = (int)std::lrintf(row[i] / sc);
                q = std::max(-127, std::min(127, q));
                xq8_[(size_t)r * H + i] = (int8_t)q;
                sw += q;
                qq += (double)q * q;
                const double d = (double)row[i] - (double)q * (double)sc;
                res += d * d;
            }
            xsc_[r] = sc;
            xsw_[r] = sw;
            xnw_[r] = (float)(std::sqrt(qq) * sc * 1.001);
            // |x.(w - sc*q)| <= ||w - sc*q|| * ||x|| (Cauchy-Schwarz); + margin for float accumulation error
            xeps_[r] = (float)std::sqrt(res) * 1.001f + 2e-4f * std::sqrt(nn);
        }
    });
}

// Lossless greedy head. Stage 1: every row is scored with the int8 copy using int8-quantised activations
// (one or two planes per token; rows are streamed once for all tokens). Because
//   |x.w - xhat.what| <= ||x|| * ||w - what|| + ||x - xhat|| * ||what||
// each row has a rigorous interval [y - err, y + err]. Stage 2: only rows whose upper bound reaches the best
// lower bound are re-evaluated with the exact f16 dot product; all others cannot be the argmax and get -inf.
void Transformer::lm_head_exact(const float* hidden, int Tl, float* logits) {
    const int V = cfg_.vocab_size, H = cfg_.hidden_size;
    std::vector<float> xn(Tl), exn(Tl);
    std::vector<uint8_t> xu((size_t)2 * Tl * H);
    std::vector<float> pscale(2 * Tl);
    std::vector<int> pfirst(Tl), pcount(Tl);
    int np = 0;
    {
        std::vector<float> r1(H);
        for (int t = 0; t < Tl; t++) {
            const float* x = hidden + (size_t)t * H;
            double ss = 0.0;
            float amax = 0.f;
            for (int i = 0; i < H; i++) { ss += (double)x[i] * x[i]; amax = std::max(amax, std::fabs(x[i])); }
            const double xnorm = std::sqrt(ss);
            xn[t] = (float)(xnorm * 1.0001 + 1e-30);
            pfirst[t] = np;
            const float s1 = amax > 0.f ? amax / 127.f : 1.f;
            double rs = 0.0;
            float rmax = 0.f;
            uint8_t* u1 = &xu[(size_t)np * H];
            for (int i = 0; i < H; i++) {
                int q = std::max(-127, std::min(127, (int)std::lrintf(x[i] / s1)));
                u1[i] = (uint8_t)(q + 128);
                r1[i] = (float)((double)x[i] - (double)q * (double)s1);
                rs += (double)r1[i] * r1[i];
                rmax = std::max(rmax, std::fabs(r1[i]));
            }
            pscale[np++] = s1;
            double rn = std::sqrt(rs);
            if (rn > 0.01 * xnorm && rmax > 0.f) {  // outlier-heavy token: second plane quantises the residual
                const float s2 = rmax / 127.f;
                uint8_t* u2 = &xu[(size_t)np * H];
                double r2s = 0.0;
                for (int i = 0; i < H; i++) {
                    int q = std::max(-127, std::min(127, (int)std::lrintf(r1[i] / s2)));
                    u2[i] = (uint8_t)(q + 128);
                    const double d = (double)r1[i] - (double)q * (double)s2;
                    r2s += d * d;
                }
                pscale[np++] = s2;
                rn = std::sqrt(r2s);
            }
            pcount[t] = np - pfirst[t];
            exn[t] = (float)(rn * 1.001 + 1e-6 * xnorm);  // + float rounding of the residual itself
        }
    }
    const int chunk = 256, ntask = (V + chunk - 1) / chunk;
    std::vector<float> pm((size_t)ntask * Tl, -INFINITY);
    pool_.run(ntask, [&](int ti) {
        const int r0 = ti * chunk, r1 = std::min(V, r0 + chunk), nr = r1 - r0;
        int32_t buf[256 * 32];
        head_dots_i8(xq8_.data() + (size_t)r0 * H, nr, H, xu.data(), np, buf);
        float best[16];
        for (int t = 0; t < Tl; t++) best[t] = -INFINITY;
        for (int i = 0; i < nr; i++) {
            const int r = r0 + i;
            const float sc = xsc_[r];
            const int32_t sw = xsw_[r];
            for (int t = 0; t < Tl; t++) {
                float acc = 0.f;
                for (int p = 0; p < pcount[t]; p++) acc += pscale[pfirst[t] + p] * (float)(buf[(size_t)i * np + pfirst[t] + p] - 128 * sw);
                const float y = sc * acc;
                logits[(size_t)t * V + r] = y;
                const float lb = y - (xeps_[r] * xn[t] + exn[t] * xnw_[r]);
                if (lb > best[t]) best[t] = lb;
            }
        }
        for (int t = 0; t < Tl; t++) pm[(size_t)ti * Tl + t] = best[t];
    });
    std::vector<float> m0(Tl, -INFINITY);
    for (int ti = 0; ti < ntask; ti++)
        for (int t = 0; t < Tl; t++) m0[t] = std::max(m0[t], pm[(size_t)ti * Tl + t]);
    std::atomic<long> ncand{0};
    pool_.run(ntask, [&](int ti) {
        const int r0 = ti * chunk, r1 = std::min(V, r0 + chunk);
        long nc = 0;
        for (int r = r0; r < r1; r++)
            for (int t = 0; t < Tl; t++) {
                float& y = logits[(size_t)t * V + r];
                if (y + (xeps_[r] * xn[t] + exn[t] * xnw_[r]) >= m0[t]) { y = dot_f16(hidden + (size_t)t * H, head_ptr_ + (size_t)r * H, H); nc++; }
                else y = -INFINITY;
            }
        ncand += nc;
    });
    if (g_prof_on && lb::env_get("HEAD_STATS")) std::fprintf(stderr, "exact head: %ld candidates, %d planes / %d tokens\n", ncand.load(), np, Tl);
}

void Transformer::lm_head(const float* hidden, int Tl, float* logits) {
    const int V = cfg_.vocab_size, H = cfg_.hidden_size;
    if (lm_head_exact_ && !xq8_.empty() && !lm_head_draft_) {
        for (int t0 = 0; t0 < Tl; t0 += 16) lm_head_exact(hidden + (size_t)t0 * H, std::min(16, Tl - t0), logits + (size_t)t0 * V);
        return;
    }
    if (screen_head_ && Tl == 1) {   // screening is probabilistic and only valid for argmax decoding: ForwardOpts::screen_ok / greedy gate it per request
        if (lm_head_screen_ok_ && (head_mode == 3 || (head_mode < 0 && lm_head_exact_ && !lm_head_draft_))) { head_screened_calls++; lm_head_screen(hidden, logits); return; }
        head_full_calls++;
    }
    if (screen_head_ && union_head_ && Tl > 1 && Tl <= 16 && lm_head_screen_ok_ && !screen_est_out_ && (head_mode == 3 || (head_mode < 0 && lm_head_exact_ && !lm_head_draft_))) {
        head_screened_calls++;
        lm_head_screen_multi(hidden, Tl, head_cand_, logits);
        return;
    }
    const bool sel_draft = lm_head_draft_ && !draft_vocab_.empty();
    const bool sel_head = !sel_draft && !head_vocab_.empty();
    const bool i8 = out_w_->dtype == DT_I8;
    std::vector<int8_t> xq;
    std::vector<float> xs;
    if (head8p_) {  // quantize the activations once per token (per-token scale), then INT8 x INT8 integer dot
        xq.resize((size_t)Tl * H);
        xs.resize(Tl);
        for (int t = 0; t < Tl; t++) {
            const float* x = hidden + (size_t)t * H;
            float mx = 0.f;
            for (int i = 0; i < H; i++) mx = std::max(mx, std::fabs(x[i]));
            xs[t] = mx > 0.f ? mx / 127.f : 1.f;
            for (int i = 0; i < H; i++) xq[(size_t)t * H + i] = (int8_t)std::lrintf(x[i] / xs[t]);
        }
    }
    // INT8 head, float hidden state (LB_HEAD_S8=1): the hidden vector is quantised to TWO int8 planes (x ~= s1*q1 + s2*q2, ~14 bit effective,
    // i.e. error ~6e-5 relative, far below the int8 weight error) and dotted with the int8 rows by the AVX2 |w| x sign(x, w) kernel: 32 MACs per
    // ~3 uops instead of the float-convert loop. Per-token independent and integer-exact -> batched == single-token. Opt-in via LB_HEAD_S8=1 (default: float path).
    static const bool s8_ok = [] { int8_t a[32] = {1}, b[32] = {1}; int32_t o[2]; int8_t x[64] = {1}; (void)a; return head_dots_s8(b, 1, 32, x, 2, o); }();
    static const bool want_s8 = lb::env_get("HEAD_S8") != nullptr;   // opt-in: no gain measurable on the (bandwidth-contended) reference server yet
    const int hmode = head_mode >= 0 ? head_mode : (want_s8 ? 1 : 0);
    const bool two_plane = i8 && !head8p_ && s8_ok && hmode == 1;
    std::vector<int8_t> xq2;
    std::vector<float> xs2;
    if (two_plane) {
        xq2.assign((size_t)Tl * 2 * H, 0);
        xs2.assign((size_t)Tl * 2, 0.f);
        std::vector<float> res(H);
        for (int t = 0; t < Tl; t++) {
            const float* x = hidden + (size_t)t * H;
            float mx = 0.f;
            for (int i = 0; i < H; i++) mx = std::max(mx, std::fabs(x[i]));
            if (mx <= 0.f) continue;
            const float s1 = mx / 127.f;
            int8_t* q1 = &xq2[(size_t)t * 2 * H];
            int8_t* q2 = q1 + H;
            float rmax = 0.f;
            for (int i = 0; i < H; i++) {
                const int q = std::max(-127, std::min(127, (int)std::lrintf(x[i] / s1)));
                q1[i] = (int8_t)q;
                res[i] = x[i] - (float)q * s1;
                rmax = std::max(rmax, std::fabs(res[i]));
            }
            xs2[(size_t)t * 2] = s1;
            if (rmax > 0.f) {
                const float s2 = rmax / 127.f;
                for (int i = 0; i < H; i++) q2[i] = (int8_t)std::max(-127, std::min(127, (int)std::lrintf(res[i] / s2)));
                xs2[(size_t)t * 2 + 1] = s2;
            }
        }
    }
    const int8_t* w8 = (const int8_t*)out_w_->data;
    auto rscale = [&](int r) { return out_scale_f_.empty() ? 1.f : out_scale_f_[r]; };
    auto score = [&](int t, int r) -> float {
        if (two_plane) {
            int32_t o[2];
            head_dots_s8(w8 + (size_t)r * H, 1, H, &xq2[(size_t)t * 2 * H], 2, o);
            return ((float)o[0] * xs2[(size_t)t * 2] + (float)o[1] * xs2[(size_t)t * 2 + 1]) * rscale(r);
        }
        if (head8p_) return (float)dot_i8(xq.data() + (size_t)t * H, head8p_ + (size_t)r * H, H) * head8s_[r] * xs[t];
        const float* x = hidden + (size_t)t * H;
        if (i8) return dot_f32_i8(x, (const int8_t*)out_w_->data + (size_t)r * H, H) * (out_scale_f_.empty() ? 1.f : out_scale_f_[r]);
        return dot_f16(x, head_ptr_ + (size_t)r * H, H);
    };
    if (sel_draft || sel_head) {  // only listed rows are scored, all other ids get -inf
        const std::vector<int>& sel = sel_draft ? draft_vocab_ : head_vocab_;
        const size_t n = sel.size();
        for (int t = 0; t < Tl; t++) {
            float* lo = logits + (size_t)t * V;
            std::fill(lo, lo + V, -INFINITY);
        }
        pool_.run(((int)n + 255) / 256, [&](int ti) {
            const size_t i0 = (size_t)ti * 256, i1 = std::min(n, i0 + 256);
            for (size_t i = i0; i < i1; i++) {
                const int r = sel[i];
                for (int t = 0; t < Tl; t++) logits[(size_t)t * V + r] = score(t, r);
            }
        });
        return;
    }
    const int chunk = 256, ntask = (V + chunk - 1) / chunk;
    if (two_plane) {
        pool_.run(ntask, [&](int ti) {
            const int r0 = ti * chunk, r1 = std::min(V, r0 + chunk), nr = r1 - r0;
            int32_t buf[256 * 8];
            for (int t0 = 0; t0 < Tl; t0 += 4) {
                const int g = std::min(4, Tl - t0), np = 2 * g;
                head_dots_s8(w8 + (size_t)r0 * H, nr, H, &xq2[(size_t)t0 * 2 * H], np, buf);
                for (int i = 0; i < nr; i++) {
                    const float sc = rscale(r0 + i);
                    for (int j = 0; j < g; j++)
                        logits[(size_t)(t0 + j) * V + r0 + i] = ((float)buf[(size_t)i * np + 2 * j] * xs2[(size_t)(t0 + j) * 2] + (float)buf[(size_t)i * np + 2 * j + 1] * xs2[(size_t)(t0 + j) * 2 + 1]) * sc;
                }
            }
        });
        return;
    }
    pool_.run(ntask, [&](int ti) {
        const int r0 = ti * chunk, r1 = std::min(V, r0 + chunk);
        for (int r = r0; r < r1; r++)
            for (int t = 0; t < Tl; t++) logits[(size_t)t * V + r] = score(t, r);
    });
}

void Transformer::forward(const int* tokens, int T, int start, const ForwardOpts& o, float* logits) {
    const int H = cfg_.hidden_size, NH = cfg_.num_heads, KVH = cfg_.num_kv_heads, hd = cfg_.head_dim;
    const int qd = NH * hd, kvd = KVH * hd, half = hd / 2;
    if (hd > 512) fail("head_dim > 512 unsupported");
    moe_uniq_last_ = 0;
    if (start + T > kv_.max_seq) fail("context length exceeded (" + std::to_string(start + T) + " > " + std::to_string(kv_.max_seq) + ")");
    if (hybrid_) hyb_begin(start, T, o);   // DeltaNet state: commit / roll back the pending chain so that it sits at `start`
    x_.resize((size_t)T * H); h_.resize((size_t)T * H);
    q_.resize((size_t)T * qd); k_.resize((size_t)T * kvd); v_.resize((size_t)T * kvd);
    att_.resize((size_t)T * qd); o_.resize((size_t)T * H);
    const int Id = cfg_.intermediate_size;
    if (Id > 0 && !all_moe_) { g_.resize((size_t)T * Id); u_.resize((size_t)T * Id); tmp_.resize((size_t)T * Id); }
    const bool use_res = !o.draft && o.use_res;
    lscratch_.fast_prefill = T >= 16;   // prefill-sized forwards may use the weight-stationary K-quant kernel; verify batches (T <= 9) stay bit-identical to T=1
    { static const int tile_env = lb::env_get("PREFILL_TILE") ? std::atoi(lb::env_get("PREFILL_TILE")) : 1; lscratch_.tile_act = T >= 16 && tile_env; }   // prefill: per-tile activation scales, no second plane

    // embeddings
    for (int t = 0; t < T; t++) {
        int id = tokens[t];
        if (id < 0 || id >= cfg_.vocab_size) fail("token id out of range");
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

    const bool i8 = kv_.mode == KvMode::Int8;
    for (int l = 0; l < cfg_.num_layers; l++) {
        if (o.draft && o.skip_layers && draft_skip_[l]) continue;
        LayerWeights& w = layers_[l];
        { ProfScope ps(6); pool_.run(T, [&](int t) { rmsnorm(&x_[(size_t)t * H], w.attn_norm.data(), H, cfg_.rms_norm_eps, &h_[(size_t)t * H]); }); }
        if (g_hess_dir && !o.draft) hess_dump(l, 'h', h_.data(), (size_t)T * H);
        if (hybrid_) { ProfScope ps(1); hyb_mixer(l, T, start, use_res); } else {
        { ProfScope ps(0);
        { const LbLinear* ws[3] = {&w.q, &w.k, &w.v};
          float* ys[3] = {q_.data(), k_.data(), v_.data()};
          linear_forward_multi(ws, 3, isa_, h_.data(), T, ys, use_res, lscratch_, &pool_); } }
        { ProfScope ps(6);
        pool_.run(T, [&](int t) {   // per token: qk-norm, RoPE, KV write (independent per token)
            const int pos = start + t;
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
                const size_t slot = (((size_t)l * KVH + h) * kv_.max_seq + pos);
                if (i8) {
                    const int nb = kv_.nb, bs = hd / nb;
                    for (int b = 0; b < nb; b++) {
                        kv_.ks[slot * nb + b] = quantize_activations_i8(kh + b * bs, bs, bs, &kv_.k8[slot * hd + b * bs]);
                        kv_.vs[slot * nb + b] = quantize_activations_i8(vh + b * bs, bs, bs, &kv_.v8[slot * hd + b * bs]);
                    }
                } else {
                    std::memcpy(&kv_.kf[slot * hd], kh, hd * 4);
                    std::memcpy(&kv_.vf[slot * hd], vh, hd * 4);
                }
            }
        }); }
        { ProfScope ps(1); attention(l, q_.data(), T, start, att_.data()); }
        if (g_hess_dir && !o.draft) hess_dump(l, 'a', att_.data(), (size_t)T * NH * hd);
        { ProfScope ps(2); linear(w.o, att_.data(), T, o_.data(), use_res); }
        { ProfScope ps(7); for (size_t i = 0; i < (size_t)T * H; i++) x_[i] += o_[i]; }
        }
        if (g_dump_prefix) if (FILE* f = dump_file(1)) std::fwrite(x_.data(), 4, (size_t)T * H, f);

        { ProfScope ps(6); pool_.run(T, [&](int t) { rmsnorm(&x_[(size_t)t * H], w.ffn_norm.data(), H, cfg_.rms_norm_eps, &h_[(size_t)t * H]); }); }
        if (w.moe) {
            ProfScope ps(3);
            moe_layer_ = l;
            moe_ffn(w, T, use_res);
            if (w.shared) hyb_shared_expert(w, T, use_res);
        } else {
            { ProfScope ps(3);
              const LbLinear* ws[2] = {&w.gate, &w.up};
              float* ys[2] = {g_.data(), u_.data()};
              linear_forward_multi(ws, 2, isa_, h_.data(), T, ys, use_res, lscratch_, &pool_); }
            { ProfScope ps(7); silu_mul(g_.data(), u_.data(), T * cfg_.intermediate_size, tmp_.data()); }
            { ProfScope ps(4); linear(w.down, tmp_.data(), T, o_.data(), use_res); }
        }
        for (size_t i = 0; i < (size_t)T * H; i++) x_[i] += o_[i];
        if (g_dump_prefix) if (FILE* f = dump_file(0)) std::fwrite(x_.data(), 4, (size_t)T * H, f);
        if (!probe_layers.empty()) probe_capture(l, T);
    }
    if (g_dump_prefix) for (FILE* f : g_dump_f) if (f) std::fflush(f);

    const int Tl = T - o.logits_from;
    if (Tl > 0 && logits) {
        for (int t = 0; t < Tl; t++)
            rmsnorm(&x_[(size_t)(o.logits_from + t) * H], out_norm_.data(), H, cfg_.rms_norm_eps, &h_[(size_t)t * H]);
        lm_head_draft_ = o.draft && !draft_vocab_.empty() && out_w_->dtype != DT_I8;
        lm_head_exact_ = o.greedy && !o.draft && head_vocab_.empty();
        lm_head_screen_ok_ = o.screen_ok;
        union_head_ = o.union_head; head_cand_ = o.head_cand;
        { ProfScope ps(5); lm_head(h_.data(), Tl, logits); }
        union_head_ = false; head_cand_ = nullptr;
        lm_head_draft_ = false;
        lm_head_exact_ = false;
        lm_head_screen_ok_ = true;
    }
}

void Transformer::probe_capture(int l, int T) {
    const int H = cfg_.hidden_size;
    probe_x.resize(probe_layers.size() * (size_t)H);
    for (size_t i = 0; i < probe_layers.size(); i++)
        if (probe_layers[i] == l + 1) std::memcpy(&probe_x[i * (size_t)H], &x_[(size_t)(T - 1) * H], (size_t)H * 4);
}

void Transformer::probe_logits(const float* x, int mode, float* logits) {
    const int H = cfg_.hidden_size;
    std::vector<float> hn(H);
    rmsnorm(x, out_norm_.data(), H, cfg_.rms_norm_eps, hn.data());
    lm_head_draft_ = false; lm_head_exact_ = false;
    if (mode == 1) {
        if (!screen_head_) fail("probe_logits mode 1 needs --screen-head");
        const int hm = head_mode;
        head_mode = 3; lm_head_screen_ok_ = true; screen_est_out_ = logits;
        lm_head(hn.data(), 1, logits);
        screen_est_out_ = nullptr; head_mode = hm;
    } else {
        lm_head_screen_ok_ = false;
        lm_head(hn.data(), 1, logits);
        lm_head_screen_ok_ = true;
    }
}

bool load_id_list_file(const std::string& path, std::vector<int>* ids, std::string* err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { if (err) *err = "cannot open " + path; return false; }
    std::string txt;
    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) txt.append(buf, n);
    std::fclose(f);
    size_t i = 0;
    while (i < txt.size()) {
        const char c = txt[i];
        if (c == '#') { while (i < txt.size() && txt[i] != '\n') i++; continue; }
        if (c >= '0' && c <= '9') {
            long v = 0;
            while (i < txt.size() && txt[i] >= '0' && txt[i] <= '9') v = std::min(v * 10 + (txt[i++] - '0'), 100000000L);
            ids->push_back((int)v);
        } else {
            i++;  // separators: newline, comma, space, brackets ...
        }
    }
    return true;
}

std::vector<int> head_vocab_ids(const ModelConfig& cfg, int n_first, const std::vector<int>& extra) {
    std::vector<int> hv;
    for (int i = 0; i < n_first && i < cfg.vocab_size; i++) hv.push_back(i);
    for (int i = 151643; i < std::min(cfg.vocab_size, 151700); i++) hv.push_back(i);
    for (int e : cfg.eos_token_ids) hv.push_back(e);
    if (cfg.bos_token_id >= 0) hv.push_back(cfg.bos_token_id);
    hv.insert(hv.end(), extra.begin(), extra.end());
    return hv;
}

// ------------------------------------------------------------------------------ MTP (Qwen3.6 multi-token prediction block)
// Loaded into layers_[cfg.num_layers] (outside every main-model layer loop) with KV layer index num_full_layers(). Missing or malformed tensors
// leave MTP unavailable (warning) instead of failing the model load.
void Transformer::load_mtp() {
    const ModelConfig& c = cfg_;
    const int L = c.num_layers, H = c.hidden_size;
    const std::string b = "mtp.blk.0.";
    try {
        LayerWeights w;
        w.attn_norm = f_.vec_f32(b + "attn_norm");
        w.ffn_norm = f_.vec_f32(b + "ffn_norm");
        w.full = true;
        w.kvl = c.num_full_layers();
        if (c.qk_norm) { w.q_norm = f_.vec_f32(b + "q_norm"); w.k_norm = f_.vec_f32(b + "k_norm"); }
        must_load(f_, b + "q", &w.q); must_load(f_, b + "k", &w.k); must_load(f_, b + "v", &w.v); must_load(f_, b + "o", &w.o);
        const int qrows = c.num_heads * c.head_dim * (c.attn_output_gate ? 2 : 1);
        if (w.q.m != qrows || w.k.m != c.num_kv_heads * c.head_dim || w.o.n != c.num_heads * c.head_dim) fail("MTP attention shape mismatch");
        const TensorRef& r = f_.get(b + "moe_gate");
        if (c.num_experts <= 0 || r.dtype != DT_F16 || r.numel() != (uint64_t)c.num_experts * H) fail("bad MTP router");
        w.moe = true;
        w.router = (const uint16_t*)r.data;
        w.egate.resize(c.num_experts); w.eup.resize(c.num_experts); w.edown.resize(c.num_experts);
        for (int e = 0; e < c.num_experts; e++) {
            const std::string eb = b + "e" + std::to_string(e) + ".";
            must_load(f_, eb + "gate", &w.egate[e]); must_load(f_, eb + "up", &w.eup[e]); must_load(f_, eb + "down", &w.edown[e]);
        }
        if (c.shared_expert_inter > 0 && f_.has(b + "sh_gate")) {
            must_load(f_, b + "sh.gate", &w.sh_gate_w); must_load(f_, b + "sh.up", &w.sh_up); must_load(f_, b + "sh.down", &w.sh_down);
            const TensorRef& sr = f_.get(b + "sh_gate");
            if (sr.dtype != DT_F16 || sr.numel() != (uint64_t)H) fail("bad MTP sh_gate");
            w.sh_router = (const uint16_t*)sr.data;
            w.shared = true;
        }
        LbLinear fc;
        must_load(f_, "mtp.fc", &fc);
        if (fc.m != H || fc.n != 2 * H) fail("MTP fc shape mismatch");
        std::vector<float> pe = f_.vec_f32("mtp.pre_fc_norm_embedding"), ph = f_.vec_f32("mtp.pre_fc_norm_hidden"), nm = f_.vec_f32("mtp.norm");
        if ((int)pe.size() != H || (int)ph.size() != H || (int)nm.size() != H || (int)w.attn_norm.size() != H || (int)w.ffn_norm.size() != H) fail("MTP norm shape mismatch");
        mtp_fc_ = std::move(fc);
        mtp_pre_emb_ = std::move(pe); mtp_pre_hid_ = std::move(ph); mtp_norm_ = std::move(nm);
        layers_.push_back(std::move(w));
        mtp_l_ = L;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[mtp] block not usable (%s): MTP disabled\n", e.what());
        layers_.resize(L);
        mtp_l_ = -1;
    }
}

int Transformer::mtp_draft_argmax(const float* x) {
    const int V = cfg_.vocab_size, H = cfg_.hidden_size, N = std::min(V, std::max(1, mtp_vocab));
    const bool i8 = out_w_->dtype == DT_I8;
    const int chunk = 512, nch = (N + chunk - 1) / chunk, ntask = nch + (mtp_extra_.empty() ? 0 : 1);
    mtp_best_.assign(ntask, {-INFINITY, -1, 0.f});
    auto score = [&](int r) -> float {
        if (i8) return dot_f32_i8(x, (const int8_t*)out_w_->data + (size_t)r * H, H) * (out_scale_f_.empty() ? 1.f : out_scale_f_[r]);
        return dot_f16(x, head_ptr_ + (size_t)r * H, H);
    };
    pool_.run(ntask, [&](int ti) {
        float bv = -INFINITY, se = 0.f; int bi = -1;
        auto take = [&](int r) {
            const float s = score(r);
            if (s > bv) { se = (bi < 0 ? 0.f : se * std::exp(bv - s)) + 1.f; bv = s; bi = r; }
            else { se += std::exp(s - bv); if (s == bv && r < bi) bi = r; }
        };
        if (ti < nch) for (int r = ti * chunk, r1 = std::min(N, r + chunk); r < r1; r++) take(r);
        else for (int r : mtp_extra_) if (r >= N && r < V) take(r);
        mtp_best_[ti] = {bv, bi, se};
    });
    float bv = -INFINITY; int bi = 0;
    for (const auto& p : mtp_best_) if (p.id >= 0 && (p.v > bv || (p.v == bv && p.id < bi))) { bv = p.v; bi = p.id; }
    double se = 0.0;
    for (const auto& p : mtp_best_) if (p.id >= 0) se += (double)p.s * std::exp((double)p.v - bv);
    mtp_p_ = se > 0.0 ? (float)(1.0 / se) : 0.f;
    return bi;
}

int Transformer::mtp_step(const float* hid, const int* next, int T, int start, int hnorm, bool draft, float* out_hidden) {
    if (mtp_l_ < 0) fail("MTP block not loaded");
    if (T <= 0) return -1;
    const int H = cfg_.hidden_size, qd = cfg_.num_heads * cfg_.head_dim, kvd = cfg_.num_kv_heads * cfg_.head_dim;
    const float eps = cfg_.rms_norm_eps;
    if (start < 0 || start + T > kv_.max_seq) fail("MTP: position out of range");
    mtp_cat_.resize((size_t)T * 2 * H);
    mtp_tmp_.resize((size_t)2 * H);
    const float* nw = hnorm == 1 ? out_norm_.data() : hnorm == 2 ? mtp_norm_.data() : nullptr;
    for (int t = 0; t < T; t++) {   // [pre_fc_norm_embedding(embed(next)) | pre_fc_norm_hidden(hid)] (hid may alias x_: built before x_ is written)
        const int id = next[t];
        if (id < 0 || id >= cfg_.vocab_size) fail("MTP: token id out of range");
        float* e = mtp_tmp_.data();
        if (tok_emb_->dtype == DT_I8) {
            const int8_t* r = (const int8_t*)tok_emb_->data + (size_t)id * H;
            float s = 1.f;
            if (tok_scale_) { uint16_t hs; std::memcpy(&hs, tok_scale_->data + 2 * (size_t)id, 2); s = f16_to_f32(hs); }
            for (int i = 0; i < H; i++) e[i] = (float)r[i] * s;
        } else {
            const uint16_t* r = (const uint16_t*)tok_emb_->data + (size_t)id * H;
            for (int i = 0; i < H; i++) e[i] = f16_to_f32(r[i]);
        }
        float* cat = &mtp_cat_[(size_t)t * 2 * H];
        rmsnorm(e, mtp_pre_emb_.data(), H, eps, cat);
        const float* hr = hid + (size_t)t * H;
        if (nw) { rmsnorm(hr, nw, H, eps, mtp_tmp_.data() + H); hr = mtp_tmp_.data() + H; }
        rmsnorm(hr, mtp_pre_hid_.data(), H, eps, cat + H);
    }
    x_.resize((size_t)T * H); h_.resize((size_t)T * H);
    q_.resize((size_t)T * qd); k_.resize((size_t)T * kvd); v_.resize((size_t)T * kvd);
    att_.resize((size_t)T * qd); o_.resize((size_t)T * H);
    lscratch_.fast_prefill = T >= 16;
    { static const int tile_env = lb::env_get("PREFILL_TILE") ? std::atoi(lb::env_get("PREFILL_TILE")) : 1; lscratch_.tile_act = T >= 16 && tile_env; }
    linear(mtp_fc_, mtp_cat_.data(), T, x_.data(), true);
    LayerWeights& w = layers_[mtp_l_];
    pool_.run(T, [&](int t) { rmsnorm(&x_[(size_t)t * H], w.attn_norm.data(), H, eps, &h_[(size_t)t * H]); });
    hyb_mixer(mtp_l_, T, start, true);   // gated attention over the MTP KV layer, x_ += out
    pool_.run(T, [&](int t) { rmsnorm(&x_[(size_t)t * H], w.ffn_norm.data(), H, eps, &h_[(size_t)t * H]); });
    moe_layer_ = mtp_l_;
    moe_ffn(w, T, true);
    if (w.shared) hyb_shared_expert(w, T, true);
    for (size_t i = 0; i < (size_t)T * H; i++) x_[i] += o_[i];
    const float* xl = &x_[(size_t)(T - 1) * H];
    if (out_hidden) std::memcpy(out_hidden, xl, (size_t)H * 4);
    if (!draft) return -1;
    rmsnorm(xl, mtp_norm_.data(), H, eps, mtp_tmp_.data());
    return mtp_draft_argmax(mtp_tmp_.data());
}

}  // namespace lb
