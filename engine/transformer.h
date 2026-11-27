// Llama/Qwen2/Qwen3 transformer built from LittleBitLinear layers + compressed INT8 KV cache.
#pragma once
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <string>
#include <vector>

#include "backend.h"
#include "engine.h"
#include "lbt_model.h"
#include "threadpool.h"
#include "lb_common.h"

namespace lb {

enum class KvMode { F32, Int8 };

struct LayerWeights {
    std::vector<float> attn_norm, ffn_norm, q_norm, k_norm;
    LbLinear q, k, v, o, gate, up, down;
    // MoE FFN (replaces gate/up/down when `moe`): router [E][H] (f16, points into the file) + per-expert linears
    bool moe = false;
    const uint16_t* router = nullptr;
    std::vector<LbLinear> egate, eup, edown;
    // Hybrid models (engine/hybrid_forward.cpp): token mixer is a Gated DeltaNet (!full) or a gated full-attention layer; MoE layers may carry a shared expert.
    bool full = true;
    int kvl = -1;                          // index into KvCache layers (full-attention layers only)
    LbLinear dn_qkv, dn_z, dn_out;         // DeltaNet projections (n4)
    const uint16_t* dn_a = nullptr;        // [VH][H] f16: decay input (in_proj_a)
    const uint16_t* dn_b = nullptr;        // [VH][H] f16: beta input (in_proj_b)
    std::vector<float> dn_conv_t;          // [kernel][C] (transposed conv weights: tap j of channel c at j * C + c)
    std::vector<float> dn_A, dn_dt, dn_norm;   // A_log [VH], dt_bias [VH], gated-norm weight [dv]
    bool shared = false;                   // shared expert + sigmoid gate
    LbLinear sh_gate_w, sh_up, sh_down;
    const uint16_t* sh_router = nullptr;   // [H] f16
};

// Per-sequence recurrent state of the DeltaNet layers (hybrid models). Double-buffered per layer: a forward reads buffer `cur` and writes the other one,
// so the state of the committed position is never destroyed; the next forward decides how much of the pending chain to keep (hyb_commit).
struct DnLayerState {
    AlignedVec<float> S[2];        // [VH][dk][dv] f32
    AlignedVec<float> conv[2];     // [(kernel-1)][C]: the last kernel-1 conv inputs, oldest first
    int cur = 0;
    // log of the pending chain (T <= kDnLogMax): replay of the first n tokens after a partial accept
    AlignedVec<float> lx, lk, ld, la;   // pre-conv qkv [T][C], normalised k [T][KH*dk], delta [T][VH*dv], decay a [T][VH]
};

// KV cache for one sequence. Keys are stored AFTER RoPE (so cached keys never need re-rotation);
// Int8 mode keeps per-(layer, kv_head, position) fp32 scales. `len` is the number of valid positions;
// speculative rollback = lowering len (no data movement).
// Turbo MoE routing knobs (default = off = strict top-k). See Transformer::moe_turbo.
struct MoeTurbo {
    float alpha = 0.f;        // keep expert j >= min_k only while p[j] >= alpha * p[0]; 0 = off
    int min_k = 4;            // experts always kept
    int kmax = 0;             // hard cap on experts (0 = num_experts_per_tok)
    float mass = 2.f;         // stop adding experts once the kept softmax mass >= mass (>= 1 = off)
    int edge_layers = 0;      // first/last N layers use max(min_k, edge_min_k) as floor (0 = off)
    int edge_min_k = 6;
    bool prefill_prune = false;   // also prune when T > 1 (prefill / verify batches)
    bool mass_topk = false;       // mass is measured relative to the top-K (num_experts_per_tok) softmax mass instead of the full router softmax (top-p over the top-K)
    std::vector<int> kmax_layers; // per-layer cap on experts (index = layer; empty = none); a cap below the floor lowers the floor
    static constexpr float kTurboAlpha = 0.50f;   // --mode turbo = alpha 0.50 + min_k 6 (avg ~6.3 experts, KL ~0.03)
    static constexpr int kTurboMinK = 6;
    // probe knobs (env only, default off = strict unchanged): LB_MOE_MASS_TOPK=1 measures `mass` against the top-K mass (top-p over the top-8),
    // LB_MOE_KMAX_LAYERS=8,8,7,.. caps experts per layer. The --mode turbo2 preset built on them failed its eval gate and was removed.
    static std::vector<int> parse_layers(const char* s) {
        std::vector<int> v;
        for (const char* p = s; p && *p;) { v.push_back(std::atoi(p)); const char* c = std::strchr(p, ','); p = c ? c + 1 : nullptr; }
        return v;
    }
    // -1 = unset sentinel for every argument; explicit values win over --mode turbo, env is read first.
    void apply(bool turbo, float a, int mk, int kx, float ms, int edge, int edge_k, int pp) {
        from_env();
        if (turbo && alpha <= 0.f) { alpha = kTurboAlpha; min_k = kTurboMinK; }
        if (a >= 0.f) alpha = a;
        if (mk >= 0) min_k = mk;
        if (kx >= 0) kmax = kx;
        if (ms >= 0.f) mass = ms;
        if (edge >= 0) edge_layers = edge;
        if (edge_k >= 0) edge_min_k = edge_k;
        if (pp >= 0) prefill_prune = pp == 1;
    }
    bool active() const { return alpha > 0.f || kmax > 0 || mass < 1.f || !kmax_layers.empty(); }
    void from_env() {
        if (const char* e = lb::env_get("MOE_ALPHA")) alpha = (float)std::atof(e);
        if (const char* e = lb::env_get("MOE_MIN_K")) min_k = std::atoi(e);
        if (const char* e = lb::env_get("MOE_KMAX")) kmax = std::atoi(e);
        if (const char* e = lb::env_get("MOE_MASS")) mass = (float)std::atof(e);
        if (const char* e = lb::env_get("MOE_EDGE")) edge_layers = std::atoi(e);
        if (const char* e = lb::env_get("MOE_EDGE_K")) edge_min_k = std::atoi(e);
        if (const char* e = lb::env_get("MOE_PREFILL_PRUNE")) prefill_prune = e[0] == '1';
        if (const char* e = lb::env_get("MOE_MASS_TOPK")) mass_topk = e[0] == '1';
        if (const char* e = lb::env_get("MOE_KMAX_LAYERS")) kmax_layers = parse_layers(e);
    }
};

struct KvCache {
    KvMode mode = KvMode::Int8;
    int layers = 0, kv_heads = 0, head_dim = 0, max_seq = 0;
    std::vector<int8_t> k8, v8;       // [layer][kv_head][pos][head_dim]
    std::vector<float> ks, vs;        // [layer][kv_head][pos][nb] (nb scales per head vector: head_dim/nb contiguous dims share one)
    int nb = 1;
    std::vector<float> kf, vf;        // F32 mode
    int len = 0;
    // Branch slices of a shared prompt (Transformer::fork_kv_slice): positions < base live in `shared`, slot 0 here is position `base`.
    int base = 0;
    const KvCache* shared = nullptr;
    void init(KvMode m, int layers, int kv_heads, int head_dim, int max_seq);
    size_t bytes() const;
};

struct ForwardOpts {
    bool draft = false;               // draft model: primary path only (use_res ignored)
    bool skip_layers = true;          // draft only: also skip the draft_layer_skip layers
    bool use_res = true;              // ignored when draft
    int logits_from = 0;              // compute logits for token indices >= logits_from (T => none)
    bool greedy = false;              // only the argmax of the logits is used: allows the exact 2-stage head
    bool screen_ok = true;            // false: never use the probabilistic 4-bit screened head (--screen-head) for this forward (any sampling request)
    // REST union head (greedy verify passes only, needs --screen-head and screen_ok): the T <= 16 logits rows share ONE 4-bit scan of the head (the 256-row chunk stays in
    // L2 across the tokens) instead of streaming the full int8 head; head_cand[t] (or -1) is the drafted token whose exact score lifts the pruning threshold of position t.
    bool union_head = false;
    const int* head_cand = nullptr;
};

// Weight-stationary batched decode (engine/batch_forward.cpp): one pass over the weights for T tokens of independent streams (or of several consecutive
// positions of one stream, e.g. a speculative verify chain). Every token carries its own position and KV slice; it attends only to its own slice.
struct BatchItem {
    int token = 0;
    int pos = 0;      // absolute position in its slice
    int slice = -1;   // KV slice id (Transformer::add_kv_slice); -1 = the main kv()
};
struct BatchOpts {
    int logits_from = 0;      // logits for items >= logits_from (T => none); layout [(T - logits_from)][vocab]
    bool greedy = false;      // only argmax is used (allows the exact 2-stage head on non-i8 models)
    bool screen_ok = false;   // allow the probabilistic 4-bit screened head when exactly one logits row is requested
    int fast = -1;            // -1 = prefill-sized forwards (T >= 64) use the weight-stationary K-quant kernel / tile scales (float order != T=1), 0 = never (bit-identical to B=1), 1 = always
    bool count_experts = false;   // accumulate unique-expert statistics (T <= 256)
};

class Transformer {
public:
    Transformer(const LbtFile& f, const ModelConfig& cfg, int max_ctx, KvMode kvmode, int n_threads, Isa isa);
    const ModelConfig& cfg() const { return cfg_; }
    KvCache& kv() { return kv_; }
    int n_threads() const { return pool_.size(); }
    Isa isa() const { return isa_; }
    const std::vector<char>& draft_skip_mask() const { return draft_skip_; }
    // Draft-only vocabulary shortlist (ids). Empty = full vocab. Verify passes always use the full head.
    void set_draft_vocab(const std::vector<int>& ids);
    // Permanently restrict the output head to these token ids (others get -inf): fewer bytes streamed per token.
    void set_head_vocab(const std::vector<int>& ids);
    // Exact greedy head over ALL ids: INT8 screening with a per-row error bound, then f16 re-evaluation of every row
    // that could still be the maximum (non-candidates get -inf). The argmax equals the plain f16 head's argmax.
    void enable_exact_head();
    void lm_head_exact(const float* hidden, int Tl, float* logits);
    // Quantizes the output head to per-row INT8 (runtime, one-time). Halves head bandwidth. For files whose head is already
    // int8 (embed_dtype "i8", tied) nothing is copied: the head switches to int8 activations x int8 weights (dot_i8).
    void enable_int8_head();
    // Greedy-only screened head for int8 output heads: a 4-bit copy of the head (half the bytes) scores all rows, a per-row statistical error margin
    // (sigma from the 4-bit residual and the activation quantisation, `screen_sigma` sigmas) selects the rows that can still be the maximum, and only those
    // are re-scored with the exact int8 path (others get -inf). Used for single-token greedy steps; opt-in (probabilistic, not rigorous).
    // bits = 4 (default, per-row symmetric 4-bit copy, ~155 MB) or 2 (2-bit asymmetric copy, groups of 128 with f16 scale+min, ~88 MB; needs H % 1024 == 0,
    // else falls back to 4). The 2-bit copy has ~2-3x the residual, so more rows survive (default screen_sigma2 sigmas); same rerank and REST union head.
    void enable_screen_head(int bits = 4);
    int screen_bits() const { return screen_head_ ? screen_bits_ : 0; }
    // Calibration probe (lb_gen --screen-calib): raw screening estimate (logit units, before the rerank) and the per-row predicted 1-sigma error of every row.
    void screen_estimate(const float* hidden, float* est, float* sigma);
    // Unique experts used by the most recent forward() (sum over layers of the popcount of the union of the T tokens' routed experts; two uint64 words per layer = up to
    // 128 experts) = the number of expert weight blocks that pass streamed. Feeds the REST byte accountant (rest_index.h).
    long unique_experts_last() const { return moe_uniq_last_; }
    const std::vector<uint64_t>& expert_masks_last() const { return moe_mask_last_; }   // [layer][2]
    long screen_cand_sum = 0, screen_cand_max = 0, screen_cand_calls = 0;   // telemetry: rows re-scored by lm_head_screen (sum / max / calls)
    long head_screened_calls = 0, head_full_calls = 0;   // telemetry: single-token heads that used / could have used (screen enabled) the screened head
    float screen_sigma = 6.0f;    // 4-bit screener margin (sigmas)
    float screen_sigma2 = 7.0f;   // 2-bit screener margin (sigmas): calibrated near-top max |z| 4.53 on 4096 real states (4-bit: 4.08 at 6); mean rerank ~300 rows
    bool head_is_int8() const { return head8p_ != nullptr || (out_w_ && out_w_->dtype == DT_I8); }

    // Run T tokens at positions [start, start+T). Writes K/V for those positions (all non-skipped layers).
    // logits: [(T - logits_from)][vocab]. Does not modify kv().len (caller manages commits).
    void forward(const int* tokens, int T, int start, const ForwardOpts& o, float* logits);

    // Benchmark / parity hooks (lb_gen --bench-head / --head-parity): run only the output head on given post-norm hidden vectors.
    void head_only(const float* hidden, int Tl, float* logits) { lm_head(hidden, Tl, logits); }
    const float* hidden_buf() const { return h_.data(); }   // post output-norm hidden vectors of the last forward() with logits
    // Early-exit probe (lb_gen --exit-probe, engine/exit_probe.cpp; diagnostic, default off): if probe_layers is non-empty forward() copies the residual stream of the
    // LAST token after layer L (= after L layers, L in 1..num_layers) into probe_x[i*H..]. probe_logits = output_norm + head on such a vector: mode 0 = the normal head
    // (full logits), mode 1 = the raw 4-bit screening scores of enable_screen_head() (no int8 rerank).
    std::vector<int> probe_layers;
    std::vector<float> probe_x;
    void probe_logits(const float* x, int mode, float* logits);
    // Turbo MoE routing (decode T=1; opt-in, default off = strict top-k, bit-identical): keep expert j (j >= min_k) only while p[j] >= alpha * p[0]
    // (probabilities after softmax, sorted descending) and the kept mass is below `mass`; kept weights are renormalised (norm_topk_prob). Env: LB_MOE_*.
    MoeTurbo moe_turbo;
    // --moe-k N: every token of every forward (prefill, verify, decode, batched, MTP) uses only its N most probable routed experts, their weights
    // renormalised. Applied the same way everywhere, so speculative verification stays exact for the pruned model. 0 = off (all routed experts).
    int moe_k = 0;
    // Per-layer expert budget (--moe-k-layers): entry l > 0 replaces moe_k for layer l; applied to every forward like moe_k. Empty = none.
    std::vector<int> moe_k_layer;
    // --shadow-full R --shadow-masks FILE (experimental quality emulation, off by default): the first R routed experts of a token run in full, every
    // further one keeps only the neurons set in its mask ([layers][experts][intermediate] bytes, 0/1). It reads the same bytes as before, so it
    // measures quality only, never speed. 0 / empty = off, the decoder is then unchanged.
    int shadow_full = 0;
    std::vector<uint8_t> shadow_mask;
    bool load_shadow_masks(const std::string& path);   // false when the file does not have exactly layers x experts x intermediate bytes
    // --teal-sparsity P: TEAL-style magnitude sparsity on the input of the MoE down projection. Per-layer thresholds come from a calibration pass:
    // teal_begin_calibration(), run public text through forward(), teal_finish_calibration(P). P <= 0 switches it off again.
    void teal_begin_calibration();
    void teal_finish_calibration(float P);
    bool teal_active() const { return teal_mode_ == 2; }
    std::string teal_report() const;   // zeroed share and (LB_TEAL_STATS=1) share of aligned column blocks that are entirely zero
    double moe_avg_experts() const { return moe_stat_tok_ ? (double)moe_stat_sel_ / (double)moe_stat_tok_ : 0.0; }   // avg experts per (decode token, layer)
    long moe_stat_tokens() const { return moe_stat_tok_; }
    int head_mode = -1;                                      // -1 = env/default, 0 = float x i8, 1 = two-plane s8, 2 = fast i8 x i8 head
    ThreadPool& pool() { return pool_; }
    // ---- batched decode (batch_forward.cpp)
    int add_kv_slice(int max_seq);                       // new empty KV slice (same layout as kv()); returns its id (>= 0)
    // New slice that shares the first prefix_len positions of the main kv() (which must hold them and stay unchanged while the slice lives) and has
    // room for `capacity` more. Only positions >= prefix_len may be written to it.
    int fork_kv_slice(int prefix_len, int capacity);
    KvCache& kv_slice(int id) { return id < 0 ? kv_ : *slices_[id]; }
    int num_kv_slices() const { return (int)slices_.size(); }
    void clear_kv_slices() { slices_.clear(); }          // drops every slice (their ids become invalid)
    bool gpu_active() const { return gpu_be_ != nullptr; }   // CUDA builds: MoE experts resident in VRAM run on the GPU
    // Runs T tokens (items[i].token at items[i].pos in KV slice items[i].slice). K/V of all items are written before attention of that layer, so items of one
    // stream at consecutive positions see each other (causal), items of different slices never do. For T < 64 (or fast == 0) every item's result is
    // bit-identical to running it alone at T=1. Does not touch KvCache::len.
    void forward_batch(const BatchItem* items, int T, const BatchOpts& o, float* logits);
    double batch_unique_experts() const { return batch_uniq_cnt_ ? (double)batch_uniq_sum_ / (double)batch_uniq_cnt_ : 0.0; }   // mean unique experts per (layer, step)
    void batch_stats_reset() { batch_uniq_sum_ = batch_uniq_cnt_ = 0; for (double& v : batch_prof_ms) v = 0; }
    double batch_prof_ms[8] = {0};   // LB_BPROF=1: ms in qkv / attention / o / moe / head / other

    // ---- MTP: Qwen3.6 native multi-token-prediction block (docs/qwen36_lbt_contract.md 2.5), loaded when the file has it (LB_MTP_OFF=1: never).
    // One gated full-attention MoE layer (index cfg.num_layers in layers_) with its own KV layer, the shared embedding and LM head.
    // Entry at position i: x = fc([pre_fc_norm_embedding(embed(next_i)), pre_fc_norm_hidden(norm?(hid_i))]) -> layer (RoPE position i) -> mtp.norm -> head.
    // hid: [T][H] hidden rows of positions start..start+T-1, next: the tokens at start+1..start+T. hnorm: 0 = hid as is, 1 = output_norm(hid), 2 = mtp.norm(hid).
    // Writes the MTP K/V of those positions. out_hidden (optional): [H] layer output of the last row (pre mtp.norm).
    // draft: returns the argmax of the last row over the draft head (ids [0, mtp_vocab) + set_mtp_extra ids), else -1.
    bool mtp_available() const { return mtp_l_ >= 0; }
    int mtp_step(const float* hid, const int* next, int T, int start, int hnorm, bool draft, float* out_hidden);
    void set_mtp_extra(const std::vector<int>& ids) { mtp_extra_ = ids; }
    int mtp_vocab = 32768;
    float mtp_last_p() const { return mtp_p_; }             // softmax probability of the last draft within the draft vocabulary (confidence)
    const float* resid_buf() const { return x_.data(); }    // pre output-norm hidden rows (residual stream) of every token of the last forward()

private:
    void lm_head(const float* hidden, int Tl, float* logits);
    void lm_head_screen(const float* hidden, float* logits);
    void lm_head_screen_multi(const float* hidden, int Tl, const int* cand, float* logits);   // REST union head: Tl <= 16 tokens, one 4-bit scan
    bool union_head_ = false;
    const int* head_cand_ = nullptr;
    std::vector<uint8_t> h4_;              // [V][H/2] biased nibbles: byte j = (q[j]+8) | (q[j+H/2]+8) << 4
    std::vector<float> h4_step_, h4_eps_, h4_wn_;   // per row: logit units per code step, 4-bit residual sigma per unit |x|, |w|/sqrt(H)  (h4_eps_/h4_wn_ also for 2-bit)
    int screen_bits_ = 4;
    std::vector<uint8_t> h2_;              // [V][H/4] 2-bit codes 0..3; group g (128 values) = 32 bytes, byte j = q[j] | q[j+32] << 2 | q[j+64] << 4 | q[j+96] << 6
    std::vector<uint16_t> h2_sm_;          // [V][2*NG] f16, NG = H/128: NG scales then NG mins (logit units incl. the int8 row scale): w ~= min + scale * q
    // 2-bit only: the K hidden channels with the largest |output_norm weight| (the "massive activation" channels, ~70 % of |x|^2) are scored exactly: int8 head
    // columns [V][K] x int16 activations, and are left out of the 2-bit groups (their code is 0 and their activation is 0 there). K = 32 (LB_SCREEN2_K, multiple of 16).
    int screen2_k_ = 0;
    std::vector<int> h2_ch_;               // [K] channel ids (ascending)
    std::vector<int8_t> h2o_;              // [V][K] exact int8 head weights of those channels
    struct ScreenX { std::vector<int8_t> xq; std::vector<float> xg; std::vector<int16_t> xo; float xs = 1.f, xos = 0.f, xn = 0.f, dxn = 0.f; int32_t sumx = 0; };
    void screen_prep(const float* x, ScreenX* sx) const;
    void screen_chunk(int r0, int r1, const ScreenX& sx, float* est) const;   // est[r - r0] for rows [r0, r1), r1 - r0 <= 256
    float screen_sg() const { return screen_bits_ == 2 ? screen_sigma2 : screen_sigma; }
    bool screen_head_ = false;
    bool lm_head_screen_ok_ = true;   // from ForwardOpts::screen_ok of the running forward
    float* screen_est_out_ = nullptr; // probe only: lm_head_screen() copies the raw 4-bit scores here and returns
    void probe_capture(int l, int T);
    // ---- hybrid (Gated DeltaNet) layers, engine/hybrid_forward.cpp
    void load_hybrid_layer(int i, LayerWeights& w);
    void load_mtp();
    int mtp_draft_argmax(const float* x);
    int mtp_l_ = -1;                       // MTP layer index in layers_ (-1 = no MTP block)
    LbLinear mtp_fc_;
    std::vector<float> mtp_pre_emb_, mtp_pre_hid_, mtp_norm_, mtp_cat_, mtp_tmp_;
    std::vector<int> mtp_extra_;
    struct MtpBest { float v; int id; float s; };   // per chunk: best score, its id, sum of exp(score - v)
    std::vector<MtpBest> mtp_best_;
    float mtp_p_ = 0.f;
    void hyb_begin(int start, int T, const ForwardOpts& o);          // commit / roll back the pending DeltaNet chain so that the state is at `start`
    void hyb_commit(int n);                                         // keep the first n tokens of the pending chain (n == pend_T: just swap the buffers)
    void hyb_mixer(int layer, int T, int start, bool use_res);      // h_ (normed) -> x_ += mixer output, for a DeltaNet or a gated full-attention layer
    void hyb_deltanet(int layer, int T, bool use_res);              // h_ -> o_
    void hyb_attention(int layer, int T, int start, bool use_res);  // h_ -> o_
    void hyb_shared_expert(const LayerWeights& w, int T, bool use_res);   // o_ += sigmoid(h_ . sh_router) * shared_expert(h_)
    bool hybrid_ = false;
    std::vector<DnLayerState> dn_;                                  // indexed by layer (empty for full-attention layers)
    int dn_pos_ = 0;                                                // tokens consumed by the committed state
    bool dn_pending_ = false;
    int dn_pend_start_ = 0, dn_pend_T_ = 0;
    bool dn_logged_ = false;                                        // the pending chain has a replay log
    int rope_half_ = 0;
    std::vector<float> hy_qkv_, hy_z_, hy_ab_, hy_qn_, hy_kn_, hy_y_, hy_delta_, hy_a_, hy_qg_, hy_gate_, hy_sg_, hy_su_, hy_st_, hy_sy_;
    void attention(int layer, const float* q, int T, int start, float* out);
    void attention_gqa(int layer, const float* q, int T, int start, float* out);
    bool attn_force_old_ = false;      // debug (LB_ATTN_CHECK): run the old kernel inside attention()
    std::vector<float> attn_part_;     // split-chunk attention partials
    std::vector<int> attn_cnt_;
    void moe_ffn(const LayerWeights& w, int T, bool use_res);  // h_ -> o_
    void teal_apply(float* a, size_t rows, int cols);          // a: [rows][cols] down_proj input of layer moe_layer_
    static constexpr int kTealBins = 160, kTealWidths = 4;     // 4 bins per octave from 2^-30; block widths 4, 16, 32, 256
    int teal_mode_ = 0;                                         // 0 off, 1 collecting, 2 applying
    float teal_p_ = 0.f;
    std::vector<uint64_t> teal_hist_;                           // [layers + 1][kTealBins]
    std::vector<float> teal_thr_;
    uint64_t teal_zeroed_ = 0, teal_seen_ = 0, teal_blocks_[kTealWidths] = {0}, teal_zero_blocks_[kTealWidths] = {0};
    int moe_layer_ = 0;
    long moe_uniq_last_ = 0;
    std::vector<uint64_t> moe_mask_last_;
    std::vector<float> scr_est_;
    long moe_stat_sel_ = 0, moe_stat_tok_ = 0;
    std::vector<int> moe_tk_n_;   // per token in the chunk: number of selected experts (<= K)

    const LbtFile& f_;
    ModelConfig cfg_;
    Isa isa_;
    ThreadPool pool_;
    KvCache kv_;
    std::vector<LayerWeights> layers_;
    std::vector<char> draft_skip_;
    std::vector<int> draft_vocab_;
    std::vector<int> head_vocab_;
    std::vector<int8_t> xq8_;
    std::vector<float> xsc_, xeps_, xnw_;  // per-row scale, error bound ||w-sc*q||, ||sc*q||
    std::vector<int32_t> xsw_;             // per-row sum of the int8 codes
    bool lm_head_exact_ = false;
    bool lm_head_draft_ = false;
    std::vector<uint16_t> head_f16_;   // heap copy of the f16 lm_head (mapped file pages are slower to stream)
    const uint16_t* head_ptr_ = nullptr;
    std::vector<int8_t> head8_;
    std::vector<float> head8_scale_;
    const int8_t* head8p_ = nullptr;        // active int8-activation head: head8_ copy, or the file's own int8 rows
    const float* head8s_ = nullptr;         // per-row scales for head8p_
    std::vector<float> out_scale_f_;        // f32 copy of the file's int8 head row scales (embed_dtype i8)
    std::vector<float> out_norm_;
    // embeddings / output head
    const TensorRef* tok_emb_ = nullptr;
    const TensorRef* tok_scale_ = nullptr;   // when embeddings are int8
    const TensorRef* out_w_ = nullptr;
    const TensorRef* out_scale_ = nullptr;
    std::vector<float> rope_cos_, rope_sin_;  // [max_ctx][head_dim/2]
    // scratch (grown on demand)
    std::vector<float> x_, h_, q_, k_, v_, att_, o_, g_, u_, tmp_, eh_, ey_, qk_tmp_;
    // MoE scratch (kept across calls: no allocation on the decode path)
    std::vector<std::unique_ptr<KvCache>> slices_;               // batched-decode KV slices (stable addresses)
    long batch_uniq_sum_ = 0, batch_uniq_cnt_ = 0;
    void attention_gqa_batch(int layer, const float* q, int T, const int* pos, KvCache* const* kv, float* out);
    std::vector<int> bat_pos_, bat_order_;
    std::vector<KvCache*> bat_kv_;
    std::vector<std::vector<std::pair<int, float>>> moe_list_;   // per expert: (token in chunk, routing weight)
    std::vector<int> moe_off_, moe_order_;
    std::vector<float> moe_lg_, moe_pr_, moe_g_, moe_u_, moe_t_, moe_tk_w_;
    std::vector<int> moe_tk_e_, moe_tk_i_;                       // per (token, k): expert, row within the expert's list; weight in moe_tk_w_
    std::vector<LinJob> moe_jobs_;
    bool all_moe_ = false;                                       // every layer is MoE: dense MLP scratch is never needed
    LinearScratch lscratch_;
    // CUDA builds only (engine/backend.h): MoE experts whose weights are resident in VRAM run on the GPU, one grouped launch per
    // projection; everything else stays on the CPU kernels. Off unless built with -DVELTRA_CUDA=ON and a GPU is present (env VELTRA_CUDA=0 disables).
    std::unique_ptr<ComputeBackend> cpu_be_, gpu_be_;
    void run_moe_jobs();

    void linear(const LbLinear& w, const float* in, int T, float* out, bool use_res);
};

// Head-vocabulary helpers shared by lb_gen / the engine: ids from a text file (one per line or comma/space separated; '#' starts a comment),
// and the always-kept ids (all ids < n_first, the model's eos ids and the special ids 151643..151699 so generation can terminate).
bool load_id_list_file(const std::string& path, std::vector<int>* ids, std::string* err);
std::vector<int> head_vocab_ids(const ModelConfig& cfg, int n_first, const std::vector<int>& extra);

}  // namespace lb
