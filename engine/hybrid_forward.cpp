// Hybrid layer support of the Transformer (Qwen3.5 / Qwen3.6: Gated DeltaNet + gated full attention + shared expert), docs/qwen36_lbt_contract.md.
// Everything here is only reached when the model meta has `layer_types` (ModelConfig::hybrid()); the classic path in transformer.cpp is untouched.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "binary_gemv.h"
#include "deltanet.h"
#include "lb_common.h"
#include "ops.h"
#include "transformer.h"

namespace lb {

namespace {
constexpr int kDnLogMax = 16;      // chains up to this many tokens keep a replay log (speculative verify chains are 2..9 tokens)
constexpr int kConvBlock = 256;    // channels per conv task

inline float sigmoidf(float x) { return 1.f / (1.f + std::exp(-x)); }
inline float softplusf(float x) { return x > 20.f ? x : std::log1p(std::exp(x)); }
}  // namespace

// ------------------------------------------------------------------------------------------------- loading
void Transformer::load_hybrid_layer(int i, LayerWeights& w) {
    const ModelConfig& c = cfg_;
    const std::string b = "blk." + std::to_string(i) + ".";
    auto ld = [&](const std::string& name, LbLinear* l) {
        std::string e;
        if (!load_linear(f_, name, l, &e)) fail("cannot load linear " + name + ": " + e);
    };
    w.full = c.layer_full[i] != 0;
    if (w.full) {
        int n = 0;
        for (int j = 0; j < i; j++) n += c.layer_full[j];
        w.kvl = n;
        if (c.qk_norm) { w.q_norm = f_.vec_f32(b + "q_norm"); w.k_norm = f_.vec_f32(b + "k_norm"); }
        ld(b + "q", &w.q); ld(b + "k", &w.k); ld(b + "v", &w.v); ld(b + "o", &w.o);
        const int qrows = c.num_heads * c.head_dim * (c.attn_output_gate ? 2 : 1);
        if (w.q.m != qrows || w.k.m != c.num_kv_heads * c.head_dim || w.o.n != c.num_heads * c.head_dim) fail("attention shape mismatch in layer " + std::to_string(i));
    } else {
        const int KH = c.lin_k_heads, VH = c.lin_v_heads, dk = c.lin_k_dim, dv = c.lin_v_dim, H = c.hidden_size, Cc = 2 * KH * dk + VH * dv;
        ld(b + "dn.qkv", &w.dn_qkv); ld(b + "dn.z", &w.dn_z); ld(b + "dn.out", &w.dn_out);
        if (w.dn_qkv.m != Cc || w.dn_z.m != VH * dv || w.dn_out.n != VH * dv || w.dn_out.m != H) fail("DeltaNet shape mismatch in layer " + std::to_string(i));
        const TensorRef& ta = f_.get(b + "dn.a");
        const TensorRef& tb = f_.get(b + "dn.b");
        if (ta.dtype != DT_F16 || tb.dtype != DT_F16 || ta.numel() != (uint64_t)VH * H || tb.numel() != (uint64_t)VH * H) fail("bad dn.a / dn.b in layer " + std::to_string(i));
        w.dn_a = (const uint16_t*)ta.data;
        w.dn_b = (const uint16_t*)tb.data;
        const std::vector<float> conv = f_.vec_f32(b + "dn.conv");   // [C][kernel]
        if ((int)conv.size() != Cc * c.lin_conv) fail("bad dn.conv in layer " + std::to_string(i));
        w.dn_conv_t.resize(conv.size());
        for (int ch = 0; ch < Cc; ch++) for (int j = 0; j < c.lin_conv; j++) w.dn_conv_t[(size_t)j * Cc + ch] = conv[(size_t)ch * c.lin_conv + j];
        w.dn_A = f_.vec_f32(b + "dn.A_log");
        w.dn_dt = f_.vec_f32(b + "dn.dt_bias");
        w.dn_norm = f_.vec_f32(b + "dn.norm");
        if ((int)w.dn_A.size() != VH || (int)w.dn_dt.size() != VH || (int)w.dn_norm.size() != dv) fail("bad dn.A_log / dn.dt_bias / dn.norm in layer " + std::to_string(i));
        for (float& a : w.dn_A) a = -std::exp(a);   // store -exp(A_log): g = A * softplus(a + dt_bias)
        DnLayerState& st = dn_[i];
        const size_t ns = (size_t)VH * dk * dv, nc = (size_t)(c.lin_conv - 1) * Cc;
        for (int k = 0; k < 2; k++) { st.S[k].assign(ns, 0.f); st.conv[k].assign(nc, 0.f); }
        st.lx.assign((size_t)kDnLogMax * Cc, 0.f);
        st.lk.assign((size_t)kDnLogMax * KH * dk, 0.f);
        st.ld.assign((size_t)kDnLogMax * VH * dv, 0.f);
        st.la.assign((size_t)kDnLogMax * VH, 0.f);
    }
    if (c.shared_expert_inter > 0 && f_.has(b + "sh_gate")) {
        ld(b + "sh.gate", &w.sh_gate_w); ld(b + "sh.up", &w.sh_up); ld(b + "sh.down", &w.sh_down);
        const TensorRef& r = f_.get(b + "sh_gate");
        if (r.dtype != DT_F16 || r.numel() != (uint64_t)c.hidden_size) fail("bad sh_gate in layer " + std::to_string(i));
        w.sh_router = (const uint16_t*)r.data;
        if (w.sh_gate_w.m != c.shared_expert_inter || w.sh_down.m != c.hidden_size) fail("shared expert shape mismatch in layer " + std::to_string(i));
        w.shared = true;
    }
}

// ------------------------------------------------------------------------------------------------- state management
// The DeltaNet state cannot be rewound by lowering KvCache::len. Every forward writes the NEW state into the second buffer of each layer (the committed
// one stays intact) and leaves it pending; the next forward(start) decides what survives: start == end of the chain keeps all of it (buffer swap),
// start inside the chain (a speculative verify that rejected a draft) replays the first n tokens from the log, start == the chain's start discards it,
// start == 0 resets everything. Any other start is an error (the state has no snapshot there).
void Transformer::hyb_commit(int n) {
    const ModelConfig& c = cfg_;
    const int KH = c.lin_k_heads, VH = c.lin_v_heads, dk = c.lin_k_dim, dv = c.lin_v_dim, Cc = 2 * KH * dk + VH * dv, Km1 = c.lin_conv - 1;
    for (int l = 0; l < c.num_layers; l++) {
        if (layers_[l].full) continue;
        DnLayerState& st = dn_[l];
        const int a = st.cur, bnew = st.cur ^ 1;
        if (n < dn_pend_T_) {
            if (!dn_logged_) fail("hybrid state: partial rollback of a chain longer than the replay log");
            const size_t ns = (size_t)VH * dk * dv;
            std::memcpy(st.S[bnew].data(), st.S[a].data(), ns * 4);
            pool_.run(VH, [&](int h) {
                float* S = st.S[bnew].data() + (size_t)h * dk * dv;
                const int kh = h / (VH / KH);
                for (int t = 0; t < n; t++) dn_head_replay(dk, dv, S, &st.lk[(size_t)t * KH * dk + (size_t)kh * dk], &st.ld[(size_t)t * VH * dv + (size_t)h * dv], st.la[(size_t)t * VH + h]);
            });
            // conv state after n tokens: the last Km1 inputs of [old state (Km1), x_0 .. x_{n-1}]
            for (int j = 0; j < Km1; j++) {
                const int src = n + j;   // index into the concatenation
                for (int ch = 0; ch < Cc; ch++)
                    st.conv[bnew][(size_t)j * Cc + ch] = src < Km1 ? st.conv[a][(size_t)src * Cc + ch] : st.lx[(size_t)(src - Km1) * Cc + ch];
            }
        }
        st.cur = bnew;   // the second buffer now holds the state after n tokens (full chain or replayed prefix)
    }
    dn_pos_ = dn_pend_start_ + n;
    dn_pending_ = false;
}

void Transformer::hyb_begin(int start, int T, const ForwardOpts& o) {
    if (o.draft) fail("draft (layer-skip) passes are not supported on hybrid DeltaNet models");
    const ModelConfig& c = cfg_;
    if (start == 0) {   // new sequence: zero state
        for (int l = 0; l < c.num_layers; l++) {
            if (layers_[l].full) continue;
            DnLayerState& st = dn_[l];
            std::fill(st.S[st.cur].begin(), st.S[st.cur].end(), 0.f);
            std::fill(st.conv[st.cur].begin(), st.conv[st.cur].end(), 0.f);
        }
        dn_pos_ = 0; dn_pending_ = false;
    } else if (dn_pending_) {
        const int ps = dn_pend_start_, pt = dn_pend_T_;
        if (start == ps + pt) hyb_commit(pt);
        else if (start > ps && start < ps + pt) hyb_commit(start - ps);
        else if (start == ps) dn_pending_ = false;
        else fail("hybrid state: forward at position " + std::to_string(start) + " but the DeltaNet state is at " + std::to_string(ps) + " (+" + std::to_string(pt) + " pending)");
    }
    if (start != dn_pos_) fail("hybrid state: forward at position " + std::to_string(start) + " but the committed DeltaNet state is at " + std::to_string(dn_pos_) +
                               " (no prefix reuse / snapshot for DeltaNet models)");
    dn_pending_ = true; dn_pend_start_ = start; dn_pend_T_ = T;
    dn_logged_ = T <= kDnLogMax;
}

// ------------------------------------------------------------------------------------------------- mixers
void Transformer::hyb_mixer(int l, int T, int start, bool use_res) {
    const LayerWeights& w = layers_[l];
    const int H = cfg_.hidden_size;
    if (w.full) hyb_attention(l, T, start, use_res); else hyb_deltanet(l, T, use_res);
    for (size_t i = 0; i < (size_t)T * H; i++) x_[i] += o_[i];
}

void Transformer::hyb_deltanet(int l, int T, bool use_res) {
    const ModelConfig& c = cfg_;
    LayerWeights& w = layers_[l];
    DnLayerState& st = dn_[l];
    const int H = c.hidden_size, KH = c.lin_k_heads, VH = c.lin_v_heads, dk = c.lin_k_dim, dv = c.lin_v_dim, Kc = c.lin_conv, Km1 = Kc - 1;
    const int kd = KH * dk, vd = VH * dv, Cc = 2 * kd + vd;
    const bool logged = dn_logged_;
    hy_qkv_.resize((size_t)T * Cc); hy_z_.resize((size_t)T * vd); hy_ab_.resize((size_t)T * VH); hy_a_.resize((size_t)T * VH);
    hy_qn_.resize((size_t)T * kd); hy_y_.resize((size_t)T * vd);
    if (!logged) { hy_kn_.resize((size_t)T * kd); hy_delta_.resize((size_t)T * vd); }
    {
        const LbLinear* ws[2] = {&w.dn_qkv, &w.dn_z};
        float* ys[2] = {hy_qkv_.data(), hy_z_.data()};
        linear_forward_multi(ws, 2, isa_, h_.data(), T, ys, use_res, lscratch_, &pool_);
    }
    // gates: decay a = exp(g), g = -exp(A_log) * softplus(a_in + dt_bias); beta = sigmoid(b_in)
    pool_.run(T * VH, [&](int ti) {
        const int t = ti / VH, h = ti % VH;
        const float* x = &h_[(size_t)t * H];
        const float ain = dot_f16(x, w.dn_a + (size_t)h * H, H), bin = dot_f16(x, w.dn_b + (size_t)h * H, H);
        hy_a_[(size_t)t * VH + h] = std::exp(w.dn_A[h] * softplusf(ain + w.dn_dt[h]));
        hy_ab_[(size_t)t * VH + h] = sigmoidf(bin);
    });
    if (logged) {
        std::memcpy(st.lx.data(), hy_qkv_.data(), (size_t)T * Cc * 4);
        std::memcpy(st.la.data(), hy_a_.data(), (size_t)T * VH * 4);
    }
    // causal depthwise conv (state: the last Km1 inputs) + SiLU, channels in blocks, tokens sequential inside a block
    {
        static const float kOnes[kConvBlock] = {
#define O8 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
#define O64 O8 O8 O8 O8 O8 O8 O8 O8
            O64 O64 O64 O64
#undef O64
#undef O8
        };
        const float* cold = st.conv[st.cur].data();
        float* cnew = st.conv[st.cur ^ 1].data();
        const float* cw = w.dn_conv_t.data();
        const int nblk = (Cc + kConvBlock - 1) / kConvBlock;
        pool_.run(nblk, [&](int bi) {
            const int c0 = bi * kConvBlock, nc = std::min(kConvBlock, Cc - c0);
            float win[8][kConvBlock];
            for (int j = 0; j < Km1; j++) std::memcpy(win[j], cold + (size_t)j * Cc + c0, (size_t)nc * 4);
            for (int t = 0; t < T; t++) {
                float* q = &hy_qkv_[(size_t)t * Cc + c0];
                for (int ch = 0; ch < nc; ch++) {
                    const float xin = q[ch];
                    float acc = 0.f;
                    for (int j = 0; j < Km1; j++) acc += cw[(size_t)j * Cc + c0 + ch] * win[j][ch];
                    acc += cw[(size_t)Km1 * Cc + c0 + ch] * xin;
                    for (int j = 0; j + 1 < Km1; j++) win[j][ch] = win[j + 1][ch];
                    win[Km1 - 1][ch] = xin;
                    q[ch] = acc;
                }
                silu_mul(q, kOnes, nc, q);
            }
            for (int j = 0; j < Km1; j++) std::memcpy(cnew + (size_t)j * Cc + c0, win[j], (size_t)nc * 4);
        });
    }
    // q, k: L2 norm per head (eps 1e-6), q additionally * 1/sqrt(dk)
    float* kn = logged ? st.lk.data() : hy_kn_.data();
    float* dl = logged ? st.ld.data() : hy_delta_.data();
    {
        const float qscale = 1.f / std::sqrt((float)dk);
        pool_.run(T * KH, [&](int ti) {
            const int t = ti / KH, h = ti % KH;
            const float* q = &hy_qkv_[(size_t)t * Cc + (size_t)h * dk];
            const float* k = q + kd;
            float sq = 0.f, sk = 0.f;
            for (int i = 0; i < dk; i++) { sq += q[i] * q[i]; sk += k[i] * k[i]; }
            const float iq = qscale / std::sqrt(sq + 1e-6f), ik = 1.f / std::sqrt(sk + 1e-6f);
            float* qo = &hy_qn_[(size_t)t * kd + (size_t)h * dk];
            float* ko = kn + (size_t)t * kd + (size_t)h * dk;
            for (int i = 0; i < dk; i++) { qo[i] = q[i] * iq; ko[i] = k[i] * ik; }
        });
    }
    // gated delta rule per value head (tokens sequential: fixed arithmetic order for any T), then the gated RMSNorm with z
    {
        const int grp = VH / KH;
        pool_.run(VH, [&](int h) {
            static thread_local std::vector<float> ob, nb;
            ob.resize(dv); nb.resize(dv);
            const size_t sz = (size_t)dk * dv;
            const float* Sc = st.S[st.cur].data() + (size_t)h * sz;
            float* Sn = st.S[st.cur ^ 1].data() + (size_t)h * sz;
            const int kh = h / grp;
            for (int t = 0; t < T; t++) {
                const float* vptr = &hy_qkv_[(size_t)t * Cc + 2 * kd + (size_t)h * dv];
                dn_head_step(dk, dv, t == 0 ? Sc : Sn, Sn, &hy_qn_[(size_t)t * kd + (size_t)kh * dk], kn + (size_t)t * kd + (size_t)kh * dk, vptr,
                             hy_a_[(size_t)t * VH + h], hy_ab_[(size_t)t * VH + h], dl + (size_t)t * vd + (size_t)h * dv, ob.data());
                float ss = 0.f;
                for (int j = 0; j < dv; j++) ss += ob[j] * ob[j];
                const float inv = 1.f / std::sqrt(ss / (float)dv + c.rms_norm_eps);
                for (int j = 0; j < dv; j++) nb[j] = ob[j] * inv * w.dn_norm[j];
                silu_mul(&hy_z_[(size_t)t * vd + (size_t)h * dv], nb.data(), dv, &hy_y_[(size_t)t * vd + (size_t)h * dv]);   // silu(z) * normed
            }
        });
    }
    linear(w.dn_out, hy_y_.data(), T, o_.data(), use_res);
}

void Transformer::hyb_attention(int l, int T, int start, bool use_res) {
    const ModelConfig& c = cfg_;
    LayerWeights& w = layers_[l];
    const int NH = c.num_heads, KVH = c.num_kv_heads, hd = c.head_dim, qd = NH * hd, kvd = KVH * hd;
    const int rd = c.rotary_dim > 0 ? c.rotary_dim : hd, half = rope_half_;
    const bool gate = c.attn_output_gate, i8 = kv_.mode == KvMode::Int8;
    if (gate) hy_qg_.resize((size_t)T * 2 * qd);
    hy_gate_.resize((size_t)T * qd);
    {
        const LbLinear* ws[3] = {&w.q, &w.k, &w.v};
        float* ys[3] = {gate ? hy_qg_.data() : q_.data(), k_.data(), v_.data()};
        linear_forward_multi(ws, 3, isa_, h_.data(), T, ys, use_res, lscratch_, &pool_);
    }
    pool_.run(T, [&](int t) {   // per token: split [q | gate], qk-norm, partial RoPE, KV write
        const int pos = start + t;
        const float* cs = &rope_cos_[(size_t)pos * half];
        const float* sn = &rope_sin_[(size_t)pos * half];
        float tmp[512];
        for (int h = 0; h < NH; h++) {
            float* qh = &q_[((size_t)t * NH + h) * hd];
            if (gate) {
                const float* src = &hy_qg_[((size_t)t * NH + h) * 2 * hd];
                std::memcpy(qh, src, (size_t)hd * 4);
                std::memcpy(&hy_gate_[((size_t)t * NH + h) * hd], src + hd, (size_t)hd * 4);
            }
            if (c.qk_norm) { rmsnorm(qh, w.q_norm.data(), hd, c.rms_norm_eps, tmp); std::memcpy(qh, tmp, (size_t)hd * 4); }
            rope_inplace(qh, rd, cs, sn);
        }
        for (int h = 0; h < KVH; h++) {
            float* kh = &k_[((size_t)t * KVH + h) * hd];
            float* vh = &v_[((size_t)t * KVH + h) * hd];
            if (c.qk_norm) { rmsnorm(kh, w.k_norm.data(), hd, c.rms_norm_eps, tmp); std::memcpy(kh, tmp, (size_t)hd * 4); }
            rope_inplace(kh, rd, cs, sn);
            const size_t slot = ((size_t)w.kvl * KVH + h) * kv_.max_seq + pos;
            if (i8) {
                const int nb = kv_.nb, bs = hd / nb;
                for (int b = 0; b < nb; b++) {
                    kv_.ks[slot * nb + b] = quantize_activations_i8(kh + b * bs, bs, bs, &kv_.k8[slot * hd + b * bs]);
                    kv_.vs[slot * nb + b] = quantize_activations_i8(vh + b * bs, bs, bs, &kv_.v8[slot * hd + b * bs]);
                }
            } else {
                std::memcpy(&kv_.kf[slot * hd], kh, (size_t)hd * 4);
                std::memcpy(&kv_.vf[slot * hd], vh, (size_t)hd * 4);
            }
        }
    });
    attention(w.kvl, q_.data(), T, start, att_.data());
    if (gate) pool_.run(T, [&](int t) {
        float* a = &att_[(size_t)t * qd];
        const float* g = &hy_gate_[(size_t)t * qd];
        for (int i = 0; i < qd; i++) a[i] *= sigmoidf(g[i]);
    });
    linear(w.o, att_.data(), T, o_.data(), use_res);
    (void)kvd;
}

// o_ (routed experts, from moe_ffn) += sigmoid(h_ . sh_router) * shared_expert(h_)
void Transformer::hyb_shared_expert(const LayerWeights& w, int T, bool use_res) {
    const int H = cfg_.hidden_size, I = cfg_.shared_expert_inter;
    hy_sg_.resize((size_t)T * I); hy_su_.resize((size_t)T * I); hy_st_.resize((size_t)T * I); hy_sy_.resize((size_t)T * H);
    {
        const LbLinear* ws[2] = {&w.sh_gate_w, &w.sh_up};
        float* ys[2] = {hy_sg_.data(), hy_su_.data()};
        linear_forward_multi(ws, 2, isa_, h_.data(), T, ys, use_res, lscratch_, &pool_);
    }
    silu_mul(hy_sg_.data(), hy_su_.data(), T * I, hy_st_.data());
    linear(w.sh_down, hy_st_.data(), T, hy_sy_.data(), use_res);
    pool_.run(T, [&](int t) {
        const float g = sigmoidf(dot_f16(&h_[(size_t)t * H], w.sh_router, H));
        float* o = &o_[(size_t)t * H];
        const float* y = &hy_sy_[(size_t)t * H];
        for (int i = 0; i < H; i++) o[i] += g * y[i];
    });
}

}  // namespace lb
