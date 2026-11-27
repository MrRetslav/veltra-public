#include "speculative.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <numeric>

#include "lb_common.h"
#include "rest_index.h"

namespace lb {

using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

namespace {
int g_prefill_chunk = lb::env_get("PREFILL_CHUNK") ? std::max(1, std::atoi(lb::env_get("PREFILL_CHUNK"))) : 64;
}
int prefill_chunk() { return g_prefill_chunk; }
void set_prefill_chunk(int n) { if (n > 0) g_prefill_chunk = n; }

int argmax(const float* v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

void sampling_probs(const float* logits, int V, const SamplingParams& sp, std::vector<float>& out) {
    out.assign(V, 0.f);
    if (sp.temperature <= 0.f) { out[argmax(logits, V)] = 1.f; return; }
    const float inv_t = 1.f / sp.temperature;
    float m = logits[0];
    for (int i = 1; i < V; i++) m = std::max(m, logits[i]);
    // top-k threshold
    float thresh = -INFINITY;
    if (sp.top_k > 0 && sp.top_k < V) {
        std::vector<float> tmp(logits, logits + V);
        std::nth_element(tmp.begin(), tmp.begin() + (sp.top_k - 1), tmp.end(), std::greater<float>());
        thresh = tmp[sp.top_k - 1];
    }
    double sum = 0;
    for (int i = 0; i < V; i++) {
        if (logits[i] < thresh) continue;
        out[i] = std::exp((logits[i] - m) * inv_t);
        sum += out[i];
    }
    for (auto& p : out) p = (float)(p / sum);
    if (sp.top_p < 1.f) {
        std::vector<int> idx;
        idx.reserve(V);
        for (int i = 0; i < V; i++) if (out[i] > 0) idx.push_back(i);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return out[a] > out[b] || (out[a] == out[b] && a < b); });
        double c = 0;
        size_t keep = idx.size();
        for (size_t j = 0; j < idx.size(); j++) { c += out[idx[j]]; if (c >= sp.top_p) { keep = j + 1; break; } }
        double s2 = 0;
        for (size_t j = 0; j < keep; j++) s2 += out[idx[j]];
        for (size_t j = keep; j < idx.size(); j++) out[idx[j]] = 0.f;
        for (size_t j = 0; j < keep; j++) out[idx[j]] = (float)(out[idx[j]] / s2);
    }
}

static int sample_from(const std::vector<float>& p, std::mt19937_64& rng) {
    double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    double c = 0;
    int last = 0;
    for (size_t i = 0; i < p.size(); i++) {
        if (p[i] <= 0.f) continue;
        c += p[i]; last = (int)i;
        if (u < c) return (int)i;
    }
    return last;
}

// ------------------------------------------------------------------------------------ n-gram
void NgramIndex::index_upto(const std::vector<int>& ctx, size_t limit) {
    limit = std::min(limit, ctx.size());
    for (size_t e = indexed_ + 1; e <= limit; e++) {
        if (e >= 3) map3_[key(&ctx[e - 3], 3)] = (int)e;
        if (e >= 4) map4_[key(&ctx[e - 4], 4)] = (int)e;
    }
    indexed_ = std::max(indexed_, limit);
}

std::vector<int> NgramIndex::propose(const std::vector<int>& ctx, int max_len, bool allow3) {
    const size_t S = ctx.size();
    auto take = [&](int e) {
        std::vector<int> r;
        // continuation of the earlier occurrence; if it runs into the end of the context the match overlaps itself
        // (a repeating pattern of period S-e), so keep extending it periodically
        const size_t period = S - (size_t)e;
        for (int k = 0; k < max_len && period > 0; k++) r.push_back(ctx[(size_t)e + (size_t)k % period]);
        return r;
    };
    if (S >= 4) {
        auto it = map4_.find(key(&ctx[S - 4], 4));
        if (it != map4_.end() && (size_t)it->second < S) { auto r = take(it->second); if (!r.empty()) return r; }
    }
    if (allow3 && S >= 3) {
        auto it = map3_.find(key(&ctx[S - 3], 3));
        if (it != map3_.end() && (size_t)it->second < S) { auto r = take(it->second); if (!r.empty()) return r; }
    }
    return {};
}

// ------------------------------------------------------------------------------------ decoder
GenStats SpecDecoder::run(const std::vector<int>& prompt, const GenOptions& opt, const std::vector<int>& eos_ids,
                          const TokenSink& sink, int reuse) {
    committed_.clear();
    GenStats st;
    const ModelConfig& cfg = m_.cfg();
    const int V = cfg.vocab_size;
    if (prompt.empty()) fail("empty prompt");
    KvCache& kv = m_.kv();
    kv.len = 0;
    if (reuse < 0 || reuse > (int)prompt.size() - 1) reuse = 0;
    {  // draft shortlist: ids 0..8191 (frequent BPE tokens) + unique prompt tokens
        std::vector<int> sl;
        for (int i = 0; i < 8192 && i < V; i++) sl.push_back(i);
        sl.insert(sl.end(), prompt.begin(), prompt.end());
        m_.set_draft_vocab(sl);
    }
    std::mt19937_64 rng(opt.sampling.seed ? opt.sampling.seed : 0x9e3779b97f4a7c15ull);
    const bool greedy = opt.sampling.temperature <= 0.f;
    // The 4-bit screened head (--screen-head) is probabilistic and writes -inf to excluded ids, so it is only allowed for plain argmax decoding:
    // temperature 0 (the server default when "temperature" is absent) and no top_p / top_k restriction. Anything else scores every id exactly.
    const bool screen_ok = greedy && opt.sampling.top_p >= 1.f && opt.sampling.top_k <= 0;
    const long scr0 = m_.head_screened_calls, full0 = m_.head_full_calls;
    const SpecOptions& so = opt.spec;
    auto is_stop = [&](int t) {
        return std::find(eos_ids.begin(), eos_ids.end(), t) != eos_ids.end() ||
               std::find(opt.stop_tokens.begin(), opt.stop_tokens.end(), t) != opt.stop_tokens.end();
    };

    // ---- MTP (native multi-token prediction block, greedy argmax requests only): drafts come from the model's own MTP layer, verification is the
    // usual exact batched forward, so the output is token-identical to plain greedy decoding. MTP entry i = (hidden of position i, token i+1) at
    // RoPE position i; its K/V must exist for every position, so the prompt is run through the MTP layer too (one extra layer of prefill).
    const bool mtp_on = so.mtp && greedy && opt.sampling.top_p >= 1.f && opt.sampling.top_k <= 0 && m_.mtp_available() && reuse == 0;
    static const int mtp_hn = lb::env_get("MTP_H") ? std::atoi(lb::env_get("MTP_H")) : 1;        // main hidden into MTP: 0 = pre output_norm, 1 = post output_norm
    static const int mtp_rn = lb::env_get("MTP_R") ? std::atoi(lb::env_get("MTP_R")) : 1;        // recursive (2nd draft) hidden: 0 = pre mtp.norm, 1 = post mtp.norm (measured: 1 is better, 93% vs 79% on code)
    static const int mtp_fixed = lb::env_get("MTP_FIXED") ? std::atoi(lb::env_get("MTP_FIXED")) : -1;   // experiments: force k drafts per pass (-1 = byte accountant)
    static const int mtp_vocab = lb::env_get("MTP_VOCAB") ? std::atoi(lb::env_get("MTP_VOCAB")) : 32768;   // draft head: ids [0, N) + prompt ids
    static const bool mtp_trace = lb::env_get("MTP_TRACE") != nullptr;
    const int H = cfg.hidden_size;
    std::vector<float> mh, mhout(H);   // pending MTP entries: hidden rows + next tokens of positions [mstart, mstart + mt.size()) (always ends at kv.len)
    std::vector<int> mt;
    int mstart = 0;
    if (mtp_on) {
        m_.mtp_vocab = mtp_vocab;
        std::vector<int> ex;
        for (int t : prompt) if (t >= mtp_vocab) ex.push_back(t);
        for (int t : eos_ids) if (t >= mtp_vocab) ex.push_back(t);
        std::sort(ex.begin(), ex.end());
        ex.erase(std::unique(ex.begin(), ex.end()), ex.end());
        m_.set_mtp_extra(ex);
    }

    // ---- prefill all but the last prompt token
    const auto t_pre0 = Clock::now();
    const int P = (int)prompt.size();
    if (P > kv.max_seq - 2) fail("prompt longer than context window");
    {
        ForwardOpts fo; fo.logits_from = 1 << 30;
        kv.len = reuse;
        const int chunk = prefill_chunk();
        for (int i = reuse; i < P - 1; i += chunk) {
            int c = std::min(chunk, P - 1 - i);
            fo.logits_from = c;
            m_.forward(&prompt[i], c, i, fo, nullptr);
            kv.len += c;
            if (mtp_on) {   // MTP K/V of positions i .. i+c-1 (next tokens prompt[i+1 .. i+c])
                const auto t_m0 = Clock::now();
                mh.assign(m_.resid_buf(), m_.resid_buf() + (size_t)c * H);
                m_.mtp_step(mh.data(), &prompt[i + 1], c, i, mtp_hn, false, nullptr);
                st.mtp_s += secs(t_m0, Clock::now());
            }
        }
        mh.clear();
        mstart = P - 1;
    }
    st.prompt_tokens = P;
    st.prefill_s = secs(t_pre0, Clock::now());

    std::vector<int> ctx = prompt;
    int last = prompt[P - 1];
    ngram_.reset();
    int K = std::clamp(so.k_init, so.k_min, so.k_max);
    double ema = 0.7, ngram_ema = 0.6;
    // REST (--rest): static retrieval index + calibrated cost/benefit gate. Greedy only: it never touches sampling requests.
    const bool rest_on = so.rest && so.enabled && greedy && opt.sampling.top_p >= 1.f && opt.sampling.top_k <= 0 && !mtp_on;
    // MTP byte accountant: k drafts cost a verify pass of T = k + 1 (relative cost kMtpCost[k], measured on Qwen3.6: 64.1 / 86.4 / 100.0 ms).
    // The first draft is always computed (its MTP layer work is needed for the K/V anyway; the draft head is ~3% of a pass). Its softmax
    // confidence p (within the draft vocabulary) picks a bucket whose online acceptance c(p) is calibrated on EVERY step, verified or not
    // (shadow: the true next token is known after the pass). Then k = argmax over {0, 1, 2} of expected tokens / verify cost, with
    // expected tokens = 1 + c1 + c1*c2; the second draft is only computed when it could win.
    static constexpr double kMtpCost[3] = {1.0, 86.4 / 64.1, 100.0 / 64.1};
    static constexpr int kMtpB = 6;
    auto mtp_bucket = [](float p) { return p < 0.3f ? 0 : p < 0.5f ? 1 : p < 0.7f ? 2 : p < 0.85f ? 3 : p < 0.95f ? 4 : 5; };
    double mcal[2][kMtpB], mcaln[2][kMtpB];
    for (int j = 0; j < 2; j++) for (int b = 0; b < kMtpB; b++) { static constexpr double pr[kMtpB] = {0.2, 0.4, 0.6, 0.77, 0.9, 0.97}; mcal[j][b] = pr[b] * (j ? 0.9 : 1.0); mcaln[j][b] = 4; }
    auto mtp_cal = [&](int j, int b, bool hit) { mcaln[j][b] = std::min(mcaln[j][b] + 1, 64.0); mcal[j][b] += ((hit ? 1.0 : 0.0) - mcal[j][b]) / mcaln[j][b]; };
    double mdc = 0.05, mbase = 64.0;
    int md[2] = {-1, -1}, mb[2] = {0, 0};   // drafts computed this step (verified or not) and their confidence buckets
    float mp[2] = {0.f, 0.f};
    RestIndex* ridx = rest_on ? rest_index() : nullptr;
    static RestCost cost_model(cfg.num_layers, std::max(1, cfg.num_experts), std::max(1, cfg.num_experts_per_tok));   // one model per process: the calibration persists across requests
    const float* rcost = cost_model.tab;
    RestGate gate_static, gate_sess;
    // Byte accountant cap: a verify pass of T = k + 1 tokens opens ~U(T) unique experts per layer; T = 5 is ~26 experts (~3.5 GB, break-even ~2.7 accepted tokens), so by
    // default T <= 3 (k <= 2). LB_REST_MAXK=N lifts the cap (experiments only).
    static const int rest_maxk = lb::env_get("REST_MAXK") ? std::clamp(std::atoi(lb::env_get("REST_MAXK")), 1, kRestMaxChain) : 2;
    const int rest_k = std::clamp(so.rest_k, 1, rest_maxk);
    static const bool rest_trace = lb::env_get("REST_TRACE") != nullptr;
    const int kmax_total = std::max(so.k_max, 1);
    std::vector<float> logits((size_t)(kmax_total + 1) * V);
    std::vector<float> p, q;
    st.finish_reason = "length";

    const auto t_dec0 = Clock::now();
    bool done = false;
    while (!done && st.generated_tokens < opt.max_new_tokens) {
        const int n = kv.len;
        const int room = kv.max_seq - n - 1;
        if (room <= 0) { st.finish_reason = "length"; break; }
        int Kc = so.enabled ? std::min(K, room - 1) : 0;
        Kc = std::max(0, std::min(Kc, opt.max_new_tokens - st.generated_tokens - 1));

        std::vector<int> drafts;
        std::vector<std::vector<float>> qd;  // draft distributions (sampling mode, neural drafts only)
        int m_ng = 0;
        int src = 0;                // REST: 1 = session match, 2 = static index
        RestDraft rd;
        const int krest = std::max(0, std::min(rest_k, std::min(room - 1, opt.max_new_tokens - st.generated_tokens - 1)));   // REST has its own cost gate: the adaptive K does not cap it
        if (rest_on && krest > 0) {
            const auto t_d0 = Clock::now();
            const int kcap = krest;
            // session match first (prompt + generated tokens), gated by its own calibrated acceptance
            if (so.ngram) {
                ngram_.index_upto(ctx, ctx.size() - 1);
                std::vector<int> sd = ngram_.propose(ctx, kcap, ngram_ema > 0.25);
                if (sd.size() >= 2) {
                    RestDraft sc;
                    double prod = 1.0, eacc = 0.0, bestg = 0.0; int bestk = 0;
                    for (int j = 0; j < (int)sd.size(); j++) {
                        prod *= gate_sess.calib(j); eacc += prod;
                        const double g = (1.0 + eacc) / rcost[j + 2 < kRestMaxCost ? j + 2 : kRestMaxCost - 1];
                        if (g > bestg) { bestg = g; bestk = j + 1; sc.eacc = (float)eacc; }
                    }
                    sc.n = bestk; sc.gain = (float)bestg;
                    for (int j = 0; j < bestk; j++) { sc.tok[j] = sd[j]; sc.p[j] = 1.f; }
                    if (sc.n >= 2 && sc.gain >= so.rest_min_gain) { drafts.assign(sd.begin(), sd.begin() + sc.n); src = 1; m_ng = sc.n; }
                    else gate_sess.shadow_set(sc);
                }
            }
            if (src == 0 && ridx) {
                st.rest_lookups++;
                const bool ok = ridx->propose(ctx.data(), ctx.size(), kcap, rcost, gate_static, so.rest_min_gain, rd);
                if (rd.n > 0) st.rest_hits++;
                if (ok) { drafts.assign(rd.tok, rd.tok + rd.n); src = 2; }
                else if (rd.n > 0) gate_static.shadow_set(rd);
            }
            st.rest_draft_s += secs(t_d0, Clock::now());
        }
        if (mtp_on) {
            const int kcap = std::max(0, std::min({so.mtp_k, 2, kmax_total, room - 1, opt.max_new_tokens - st.generated_tokens - 1}));
            md[0] = md[1] = -1;
            const auto t_m0 = Clock::now();
            if (!mt.empty() && (kcap > 0 || (int)mt.size() >= 16)) {
                const bool want = kcap > 0;
                const int d1 = m_.mtp_step(mh.data(), mt.data(), (int)mt.size(), mstart, mtp_hn, want, want && kcap > 1 ? mhout.data() : nullptr);
                mh.clear(); mt.clear(); mstart = n;
                int k = 0;
                if (want) {
                    md[0] = d1; mp[0] = m_.mtp_last_p(); mb[0] = mtp_bucket(mp[0]);
                    const double c1 = mcal[0][mb[0]];
                    double best = 1.0;
                    if ((1.0 + c1) / kMtpCost[1] > best) { best = (1.0 + c1) / kMtpCost[1]; k = 1; }
                    if (kcap > 1 && (mtp_fixed >= 2 || (1.0 + 2.0 * c1) / (kMtpCost[2] + mdc) > best)) {   // the 2nd draft could win even if it is always accepted
                        const auto t_d2 = Clock::now();
                        md[1] = m_.mtp_step(mhout.data(), &d1, 1, n, mtp_rn ? 2 : 0, true, nullptr);   // recursive step at position n (overwritten by the next real entry)
                        mp[1] = m_.mtp_last_p(); mb[1] = mtp_bucket(mp[1]);
                        mdc = 0.9 * mdc + 0.1 * (secs(t_d2, Clock::now()) * 1000.0 / mbase);
                        const double g2 = (1.0 + c1 + c1 * mcal[1][mb[1]]) / kMtpCost[2];
                        if (g2 > best) { best = g2; k = 2; }
                    }
                    if (mtp_fixed >= 0) k = std::min(mtp_fixed, md[1] >= 0 ? 2 : 1);
                    for (int j = 0; j < k; j++) drafts.push_back(md[j]);
                }
            }
            st.mtp_s += secs(t_m0, Clock::now());
        }
        if (!rest_on && !mtp_on && Kc > 0) {
            // ---- Stage A: n-gram lookahead
            if (so.ngram) {
                ngram_.index_upto(ctx, ctx.size() - 1);
                drafts = ngram_.propose(ctx, Kc, ngram_ema > 0.25);
                if (drafts.size() < 2) drafts.clear();  // a single-token match is not worth a verify batch
                m_ng = (int)drafts.size();
            }
            // ---- Stage B: neural draft (primary path only, layer skipping)
            if (so.neural && (int)drafts.size() < Kc) {
                ForwardOpts fo;
                fo.draft = true;                 // draft always uses the primary path only
                fo.skip_layers = so.layer_skip;  // ... and optionally skips the insensitive layers
                std::vector<int> in;
                in.push_back(last);
                in.insert(in.end(), drafts.begin(), drafts.end());
                const int need = Kc - (int)drafts.size();
                fo.logits_from = (int)in.size() - 1;
                std::vector<float> dl(V);
                m_.forward(in.data(), (int)in.size(), n, fo, dl.data());
                int pos = n + (int)in.size();
                for (int i = 0; i < need; i++) {
                    int d;
                    if (greedy) d = argmax(dl.data(), V);
                    else { sampling_probs(dl.data(), V, opt.sampling, q); d = sample_from(q, rng); }
                    if (!greedy) {
                        qd.resize(drafts.size() + 1);  // entries before neural ones stay empty (one-hot ngram)
                        qd[drafts.size()] = q;
                    }
                    drafts.push_back(d);
                    if (i + 1 < need) {
                        ForwardOpts f2; f2.draft = true; f2.skip_layers = so.layer_skip; f2.logits_from = 0;
                        m_.forward(&d, 1, pos, f2, dl.data());
                        pos++;
                    }
                }
            }
        }

        // ---- Stage C: batched verify through the full model
        const int Kd = (int)drafts.size();
        std::vector<int> in;
        in.push_back(last);
        in.insert(in.end(), drafts.begin(), drafts.end());
        int cand_ids[kRestMaxChain + 1];
        for (int i = 0; i < kRestMaxChain + 1; i++) cand_ids[i] = i < Kd ? drafts[i] : -1;   // cand[t] = the draft that position t must reproduce (last position: none)
        ForwardOpts vo; vo.draft = false; vo.use_res = true; vo.logits_from = 0; vo.greedy = greedy; vo.screen_ok = screen_ok;
        const auto t_f0 = Clock::now();
        if (rest_on || (mtp_on && Kd > 0)) { vo.union_head = true; vo.head_cand = drafts.empty() ? nullptr : cand_ids; }   // union head: one shared 4-bit scan for all T positions (needs --screen-head)
        m_.forward(in.data(), Kd + 1, n, vo, logits.data());
        const double verify_ms = secs(t_f0, Clock::now()) * 1000.0;
        if (rest_on) cost_model.observe(Kd + 1, secs(t_f0, Clock::now()) * 1000.0, (double)m_.unique_experts_last());
        st.verify_passes++;
        st.drafted += Kd;
        st.ngram_drafted += m_ng;
        if (src == 2) { st.rest_passes++; st.rest_drafted += Kd; }

        std::vector<int> emit;
        int accepted = 0;
        bool rejected = false;
        for (int i = 0; i < Kd; i++) {
            const float* L = &logits[(size_t)i * V];
            const int d = drafts[i];
            if (greedy) {
                int tgt = argmax(L, V);
                if (tgt == d) { accepted++; emit.push_back(d); continue; }
                emit.push_back(tgt);
                rejected = true;
                break;
            }
            sampling_probs(L, V, opt.sampling, p);
            const bool onehot = (size_t)i >= qd.size() || qd[i].empty();
            const float pd = p[d];
            const float qdv = onehot ? 1.f : qd[i][d];
            const double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
            if (qdv > 0.f && u < std::min(1.0, (double)pd / qdv)) { accepted++; emit.push_back(d); continue; }
            // residual distribution max(p - q, 0), renormalised
            std::vector<float> r(V);
            double s = 0;
            for (int j = 0; j < V; j++) {
                float qj = onehot ? (j == d ? 1.f : 0.f) : qd[i][j];
                r[j] = std::max(0.f, p[j] - qj);
                s += r[j];
            }
            if (s <= 0) r = p; else for (auto& x : r) x = (float)(x / s);
            emit.push_back(sample_from(r, rng));
            rejected = true;
            break;
        }
        if (!rejected) {
            const float* L = &logits[(size_t)Kd * V];
            int bonus;
            if (greedy) bonus = argmax(L, V);
            else { sampling_probs(L, V, opt.sampling, p); bonus = sample_from(p, rng); }
            emit.push_back(bonus);
        }
        st.accepted += accepted;
        st.ngram_accepted += std::min(accepted, m_ng);
        if (mtp_on) {   // rows 0..accepted of the verify pass are real positions n..n+accepted: queue them as MTP entries (next token = emit[j])
            if (mstart + (int)mt.size() != n) fail("MTP: pending entries out of sync");
            const float* xr = m_.resid_buf();
            mh.insert(mh.end(), xr, xr + (size_t)(accepted + 1) * H);
            for (int j = 0; j <= accepted; j++) mt.push_back(emit[j]);
            if (Kd >= 1) { const int a = accepted >= 1; st.mtp_drafted[0]++; st.mtp_accepted[0] += a; }
            if (Kd >= 2 && accepted >= 1) { const int a = accepted >= 2; st.mtp_drafted[1]++; st.mtp_accepted[1] += a; }
            // shadow calibration: emit[0] is the true token after `last`; emit[1] (when the first draft was accepted) the one after it
            if (md[0] >= 0) {
                const bool h1 = emit[0] == md[0];
                mtp_cal(0, mb[0], h1);
                if (md[1] >= 0 && h1 && emit.size() >= 2) mtp_cal(1, mb[1], emit[1] == md[1]);
            }
            mbase = 0.9 * mbase + 0.1 * (verify_ms / kMtpCost[std::min(Kd, 2)]);
            if (mtp_trace) std::fprintf(stderr, "MTP_TRACE n=%d k=%d acc=%d d=%d,%d p=%.2f,%.2f emit0=%d ms=%.1f c1=%.2f c2=%.2f dc=%.3f\n", n, Kd, accepted, md[0], md[1], mp[0], md[1] >= 0 ? mp[1] : 0.f, emit[0], verify_ms, md[0] >= 0 ? mcal[0][mb[0]] : 0.0, md[1] >= 0 ? mcal[1][mb[1]] : 0.0, mdc);
        }
        if (src == 2) st.rest_accepted += accepted;
        if (src && rest_trace) {   // LB_REST_TRACE=1: one line per executed REST pass (offline calibration of the table probabilities)
            std::fprintf(stderr, "REST_TRACE src=%d order=%d n=%d gain=%.2f eacc=%.2f acc=%d p=", src, rd.order, Kd, src == 2 ? rd.gain : 0.f, src == 2 ? rd.eacc : 0.f, accepted);
            for (int j = 0; j < Kd; j++) std::fprintf(stderr, "%s%.2f", j ? "," : "", src == 2 ? rd.p[j] : 1.f);
            std::fprintf(stderr, "\n");
        }
        if (src) {   // calibrate the gate with what the target model really accepted (step j counts when all earlier steps matched)
            RestGate& g = src == 1 ? gate_sess : gate_static;
            for (int j = 0; j < Kd && j <= accepted; j++) g.update(j, src == 1 ? 1.f : rd.p[j], j < accepted);
        }

        // ---- commit + rollback: positions [n, n+accepted] hold valid full-model K/V
        kv.len = n + accepted + 1;
        for (int t : emit) {
            if (st.generated_tokens >= opt.max_new_tokens) { done = true; break; }
            ctx.push_back(t);
            if (rest_on) { gate_static.shadow_feed(t); gate_sess.shadow_feed(t); }
            st.generated_tokens++;
            const bool stop = is_stop(t);
            if (!sink(t)) { st.finish_reason = "cancelled"; done = true; break; }
            if (stop) { st.finish_reason = "stop"; done = true; break; }
        }
        if (st.generated_tokens >= opt.max_new_tokens && !done) done = true;
        last = ctx.back();

        // ---- Stage D: adaptive window
        if (Kd > 0) {
            double rate = (double)accepted / Kd;
            ema = 0.6 * ema + 0.4 * rate;
            if (so.adaptive) {
                if (ema > 0.80 && K < so.k_max) K++;
                else if (ema < 0.45 && K > so.k_min) K--;
            }
            if (m_ng > 0) ngram_ema = 0.7 * ngram_ema + 0.3 * ((double)std::min(accepted, m_ng) / m_ng);
        }
    }
    if (rest_on && rest_trace) {
        std::fprintf(stderr, "REST_COST f=%.2f g=%.3f ms1=%.1f(%d) U(T)/layer=", cost_model.f, cost_model.g, cost_model.ms[1], cost_model.nms[1]);
        for (int t = 1; t < 6; t++) std::fprintf(stderr, "%s%.1f(%d)", t > 1 ? "," : "", cost_model.U[t] / cfg.num_layers, cost_model.nU[t]);
        std::fprintf(stderr, " cost=");
        for (int t = 1; t < 6; t++) std::fprintf(stderr, "%s%.2f", t > 1 ? "," : "", cost_model.tab[t]);
        std::fprintf(stderr, "\n");
    }
    committed_.assign(ctx.begin(), ctx.begin() + std::min<size_t>(ctx.size(), (size_t)kv.len));
    st.decode_s = secs(t_dec0, Clock::now());
    st.tokens_per_s = st.decode_s > 0 ? st.generated_tokens / st.decode_s : 0;
    st.head_screened = (int)(m_.head_screened_calls - scr0); st.head_full = (int)(m_.head_full_calls - full0);
    return st;
}

}  // namespace lb
