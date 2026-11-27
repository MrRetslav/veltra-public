#include "search.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include "json.h"
#include "lb_common.h"
#include "lbt_model.h"
#include "model_config.h"
#include "speculative.h"

namespace lb {
namespace {

using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

// Temperature + top-p sample of one logits row. *logp receives the model's own log-probability (temperature 1) of the chosen token. Only the head of
// the distribution is ordered (grown until it covers top_p), so the cost stays far below a full sort of the vocabulary.
int sample_token(const float* l, int V, float temperature, float top_p, std::mt19937_64& rng, double* logp) {
    int best = 0;
    for (int i = 1; i < V; i++)
        if (l[i] > l[best]) best = i;
    const float m = l[best];
    double z1 = 0, zt = 0;
    const float inv_t = temperature > 0.f ? 1.f / temperature : 0.f;
    for (int i = 0; i < V; i++) {
        const float d = l[i] - m;
        if (d < -60.f) continue;   // below 1e-26 of the maximum
        z1 += std::exp((double)d);
        if (temperature > 0.f) zt += std::exp((double)(d * inv_t));
    }
    int pick = best;
    if (temperature > 0.f) {
        static thread_local std::vector<int> ord;
        auto before = [l](int a, int b) { return l[a] > l[b] || (l[a] == l[b] && a < b); };
        const double want = (double)std::min(1.f, top_p) * zt;
        double kept = 0;
        size_t n = 0;
        for (int K = 256;; K *= 8) {
            ord.resize(V);
            for (int i = 0; i < V; i++) ord[i] = i;
            const int k = std::min(K, V);
            if (k < V) std::nth_element(ord.begin(), ord.begin() + k, ord.end(), before);
            std::sort(ord.begin(), ord.begin() + k, before);
            kept = 0;
            n = 0;
            while (n < (size_t)k) {
                kept += std::exp((double)((l[ord[n]] - m) * inv_t));
                n++;
                if (kept >= want) break;
            }
            if (kept >= want || k == V) break;
        }
        const double u = std::uniform_real_distribution<double>(0.0, 1.0)(rng) * kept;
        double c = 0;
        pick = ord[n - 1];
        for (size_t j = 0; j < n; j++) {
            c += std::exp((double)((l[ord[j]] - m) * inv_t));
            if (u < c) { pick = ord[j]; break; }
        }
    }
    *logp = (double)(l[pick] - m) - std::log(z1);
    return pick;
}

}  // namespace

SearchRun run_search(Transformer& model, const Tokenizer& tok, const std::vector<int>& prompt, const GenOptions& opt, const std::vector<int>& eos_ids,
                     int reuse) {
    const ModelConfig& cfg = model.cfg();
    const SearchOptions& so = opt.search;
    const int V = cfg.vocab_size, N = std::max(1, std::min(16, so.n)), P = (int)prompt.size();
    if (cfg.hybrid()) fail("search_n is not supported for hybrid (DeltaNet) models");
    if (P <= 0) fail("empty prompt");
    KvCache& kv = model.kv();
    const int max_new = std::min(opt.max_new_tokens, kv.max_seq - P);
    if (max_new <= 0) fail("context length exceeded (" + std::to_string(P) + " prompt tokens, limit " + std::to_string(kv.max_seq) + ")");
    if (reuse < 0 || reuse > P - 1) reuse = 0;

    SearchRun run;
    GenStats& st = run.stats;
    st.prompt_tokens = P;
    st.cached_tokens = reuse;

    // ---- one shared prefill, in the same forwards as the normal decoder (chunks, then the last prompt token alone)
    const auto t0 = Clock::now();
    std::vector<float> first((size_t)V);
    {
        const int chunk = prefill_chunk();
        kv.len = 0;
        for (int i = reuse; i < P - 1; i += chunk) {
            const int c = std::min(chunk, P - 1 - i);
            ForwardOpts fo;
            fo.logits_from = c;
            model.forward(prompt.data() + i, c, i, fo, nullptr);
            if (so.keep_going && !so.keep_going()) { run.cancelled = true; st.finish_reason = "cancelled"; return run; }
        }
        ForwardOpts fo;
        fo.logits_from = 0;
        model.forward(&prompt[P - 1], 1, P - 1, fo, first.data());
        kv.len = P;
        run.committed = prompt;
    }
    const auto t1 = Clock::now();
    st.prefill_s = seconds(t0, t1);

    struct Slices {
        Transformer& m;
        ~Slices() { m.clear_kv_slices(); }
    } guard{model};
    model.clear_kv_slices();
    run.branches.assign(N, {});
    std::vector<std::mt19937_64> rng;
    std::vector<Detok> detok;
    std::vector<int> slice(N);
    for (int i = 0; i < N; i++) {
        rng.emplace_back(so.base_seed + (uint64_t)i);
        detok.emplace_back(tok, true);
        slice[i] = model.fork_kv_slice(P, max_new);
    }
    auto is_stop = [&](int id) {
        return std::find(eos_ids.begin(), eos_ids.end(), id) != eos_ids.end() ||
               std::find(opt.stop_tokens.begin(), opt.stop_tokens.end(), id) != opt.stop_tokens.end();
    };
    auto finish = [&](BranchResult& b, int i, const char* reason) {
        b.finished = true;
        b.finish_reason = reason;
        b.text += detok[i].flush();
    };

    std::vector<int> active(N), next_tok(N);
    std::vector<double> next_logp(N);
    std::vector<const float*> row(N, first.data());
    for (int i = 0; i < N; i++) active[i] = i;
    std::vector<float> logits((size_t)N * V);
    std::vector<BatchItem> items;
    while (!active.empty()) {
        const int na = (int)active.size();
        model.pool().run(na, [&](int a) {
            const int i = active[a];
            next_tok[i] = sample_token(row[i], V, so.temperature, so.top_p, rng[i], &next_logp[i]);
        });
        items.clear();
        std::vector<int> still;
        for (int a = 0; a < na; a++) {
            const int i = active[a], id = next_tok[i];
            BranchResult& b = run.branches[i];
            b.cum_logprob += next_logp[i];
            b.token_count++;
            if (is_stop(id)) { finish(b, i, "stop"); continue; }
            b.ids.push_back(id);
            const size_t before = b.text.size();
            b.text += detok[i].push(id);
            bool cut = false;
            for (const std::string& s : so.stop_strings) {
                if (s.empty()) continue;
                const size_t at = b.text.find(s, before > s.size() ? before - s.size() : 0);
                if (at != std::string::npos) { b.text.resize(at); cut = true; }
            }
            if (cut) { b.finished = true; b.finish_reason = "stop"; continue; }
            if (b.ids.size() % 8 == 0) {   // early pruning: a broken branch leaves the batch, which speeds up the others
                const std::string why = prune_reason(b.ids, b.text, so.verify);
                if (!why.empty()) {
                    b.pruned = true;
                    b.why = why;
                    finish(b, i, "pruned");
                    continue;
                }
            }
            if ((int)b.ids.size() >= max_new) { finish(b, i, "length"); continue; }
            items.push_back({id, P + (int)b.ids.size() - 1, slice[i]});
            still.push_back(i);
        }
        active.swap(still);
        if (active.empty()) break;
        if (so.keep_going && !so.keep_going()) { run.cancelled = true; st.finish_reason = "cancelled"; return run; }
        BatchOpts bo;
        model.forward_batch(items.data(), (int)items.size(), bo, logits.data());
        st.search_steps++;
        for (size_t a = 0; a < active.size(); a++) row[active[a]] = logits.data() + a * V;
    }

    run.verdict = judge_branches(run.branches, so.verify);
    const auto t2 = Clock::now();
    const BranchResult& w = run.branches[run.verdict.winner];
    st.decode_s = seconds(t1, t2);
    st.generated_tokens = (int)w.ids.size();
    st.tokens_per_s = st.decode_s > 0 ? st.generated_tokens / st.decode_s : 0;
    st.finish_reason = w.finish_reason == "length" ? "length" : "stop";
    st.search_branches = run.verdict.branches_evaluated;
    st.search_valid = run.verdict.valid_branches;
    st.search_pruned = run.verdict.pruned_branches;
    st.search_winner = run.verdict.winner;
    st.search_consensus = run.verdict.consensus_ratio;
    std::string d = "[";
    for (int i = 0; i < N; i++) {
        const BranchResult& b = run.branches[i];
        st.search_total_tokens += (int)b.ids.size();
        int group = -1;   // branches with the same outcome share the index of the first of them
        for (int j = 0; j <= i && group < 0; j++)
            if (!b.outcome.empty() && run.branches[j].outcome == b.outcome) group = j;
        char buf[256];
        std::snprintf(buf, sizeof buf, "%s{\"branch\":%d,\"tokens\":%d,\"mean_logprob\":%.5f,\"score\":%.5f,\"valid\":%s,\"group\":%d,\"finish\":\"%s\",\"why\":",
                      i ? "," : "", i, (int)b.ids.size(), b.cum_logprob / std::max(1, b.token_count), b.score, b.valid ? "true" : "false", group,
                      b.finish_reason.c_str());
        d += buf;
        json::Value::escape_to(d, b.why);
        d += '}';
    }
    st.search_detail = d + "]";
    return run;
}

int run_shared_prefix_selftest(const std::string& model_path, int prompt_len, int steps, std::string& report) {
    LbtFile f;
    std::string err;
    ModelConfig cfg;
    if (!f.open(model_path, &err) || !parse_model_config(f, &cfg, &err)) { report += "shared-prefix selftest: " + err + "\n"; return 1; }
    if (cfg.hybrid()) { report += "shared-prefix selftest: skipped (hybrid model)\n"; return 0; }
    const int V = cfg.vocab_size, maxB = 8;
    Transformer model(f, cfg, prompt_len + steps + 8, KvMode::Int8, 8, default_isa_for(f));
    std::vector<int> prompt(prompt_len);
    uint64_t x = 0x2545F4914F6CDD1Dull;   // deterministic pseudo-random ids from the frequent end of the vocabulary
    auto rnd = [&](int n) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return (int)(x % (uint64_t)n); };
    for (int& id : prompt) id = 256 + rnd(std::min(V, 20000) - 256);
    for (int i = 0; i < prompt_len; i += 64) {
        const int c = std::min(64, prompt_len - i);
        ForwardOpts fo;
        fo.logits_from = c;
        model.forward(prompt.data() + i, c, i, fo, nullptr);
    }
    // reference: every branch alone, one token per forward, in the main cache (positions >= prompt_len never touch the shared prefix)
    std::vector<std::vector<int>> tok(maxB, std::vector<int>(steps));
    for (auto& b : tok)
        for (int& id : b) id = 256 + rnd(std::min(V, 20000) - 256);
    std::vector<float> ref((size_t)maxB * steps * V), row((size_t)V), batch((size_t)maxB * V);
    for (int b = 0; b < maxB; b++)
        for (int s = 0; s < steps; s++) {
            ForwardOpts fo;
            fo.logits_from = 0;
            model.forward(&tok[b][s], 1, prompt_len + s, fo, row.data());
            std::copy(row.begin(), row.end(), ref.begin() + ((size_t)b * steps + s) * V);
        }
    int fails = 0;
    for (int B : {1, 4, 8}) {
        model.clear_kv_slices();
        std::vector<int> slice(B);
        for (int b = 0; b < B; b++) slice[b] = model.fork_kv_slice(prompt_len, steps);
        long bad = 0;
        std::vector<BatchItem> it(B);
        for (int s = 0; s < steps; s++) {
            for (int b = 0; b < B; b++) it[b] = {tok[b][s], prompt_len + s, slice[b]};
            BatchOpts bo;
            model.forward_batch(it.data(), B, bo, batch.data());
            for (int b = 0; b < B; b++)
                if (std::memcmp(&batch[(size_t)b * V], &ref[((size_t)b * steps + s) * V], (size_t)V * 4) != 0) bad++;
        }
        size_t own = 0;
        for (int b = 0; b < B; b++) own += model.kv_slice(slice[b]).bytes();
        char line[256];
        std::snprintf(line, sizeof line, "shared-prefix B=%d: prompt %d, %d steps, %ld/%d logit rows differ from solo T=1; branch KV %.1f MB (full cache %.1f MB) %s\n",
                      B, prompt_len, steps, bad, B * steps, own / 1e6, model.kv().bytes() / 1e6, bad ? "FAIL" : "ok");
        report += line;
        if (bad) fails++;
    }
    model.clear_kv_slices();
    return fails;
}

}  // namespace lb
