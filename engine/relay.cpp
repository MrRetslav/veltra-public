// engine/relay.cpp - see relay.h. In --relay-only mode every byte that arrives here is hostile internet traffic.
//
// Rules this file keeps:
//   * the only thing a job carries is a suite id from the compiled-in allowlist (engine/test_runner.cpp);
//   * nothing is written to disk unless the admin dispatched a job to that pairing code + node token and the job is
//     still open; exactly one final result per job; the results directory is capped in files and bytes;
//   * every stored or echoed string is stripped of control characters; request data is never logged;
//   * everything kept in memory is capped and expires; a parsed body is also capped in JSON values, one result per job
//     is in flight at a time and at most two large result bodies are being read or handled at once.
#include "relay.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "remote_common.h"
#include "test_runner.h"

#if defined(_WIN32)
#  include <bcrypt.h>
#else
#  include <fcntl.h>
#  include <unistd.h>
#endif

namespace veltra::rc {

bool csprng(void* buf, size_t n) {
#if defined(_WIN32)
    return BCryptGenRandom(nullptr, static_cast<PUCHAR>(buf), static_cast<ULONG>(n), BCRYPT_USE_SYSTEM_PREFERRED_RNG) == 0;
#else
    std::FILE* f = std::fopen("/dev/urandom", "rb");
    if (!f) return false;
    const bool ok = std::fread(buf, 1, n, f) == n;
    std::fclose(f);
    return ok;
#endif
}

}  // namespace veltra::rc

namespace veltra::relay {
namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using lbhttp::Json;
using lbhttp::json_quote;
using lbhttp::sock_t;

constexpr size_t kMaxNodes = 64;
constexpr auto kNodeIdle = std::chrono::seconds(120);
constexpr auto kQueuedMaxAge = std::chrono::minutes(5);    // dispatched but never picked up
constexpr auto kPendingMaxAge = std::chrono::minutes(7);   // the client itself rejects after 5 minutes
constexpr auto kOpenMaxAge = std::chrono::hours(3);
constexpr auto kFinishedKeep = std::chrono::hours(1);
constexpr size_t kMaxFinishedJobs = 40;
constexpr double kRegisterBurst = 20.0;
constexpr double kRegisterPerSecond = 1.0 / 3.0;
constexpr size_t kMaxRegisterBody = 16u << 10;
constexpr size_t kMaxProgressBody = 8u << 10;
constexpr size_t kMaxResultBody = 1u << 20;
constexpr size_t kMaxDispatchBody = 16u << 10;
constexpr size_t kMaxResultFiles = 500;
constexpr uintmax_t kMaxResultDirBytes = 100u << 20;
constexpr size_t kMaxResultFileBytes = 2u << 20;
constexpr size_t kMaxListed = 50;
// A real report has a few hundred JSON values (long model answers are single strings).
constexpr size_t kMaxJsonNodes = 20000;
constexpr size_t kLargeBody = 64u << 10;
constexpr int kMaxLargeInFlight = 2;

struct Node {
    std::string code, token;
    Clock::time_point seen;
};

struct Job {
    std::string id, code, token, suite;
    std::string status = "queued";   // queued | delivered | pending | running | done | rejected | cancelled | error | expired
    std::string progress = "null", result = "null", reason, saved_file;
    Clock::time_point created, state_since, finished;
    bool open = true;
    bool receiving = false;   // a result body for this job is being read or handled right now
};

std::mutex g_mu;
Config g_cfg;
std::vector<Node> g_nodes;
std::vector<std::unique_ptr<Job>> g_jobs;
double g_bucket = kRegisterBurst;
Clock::time_point g_bucket_at = Clock::now();

Response fail(int status, const char* type, const std::string& message, const char* code = "") {
    return {status, "{\"error\":{\"message\":" + json_quote(message) + ",\"type\":" + json_quote(type) +
                        ",\"code\":" + (*code ? json_quote(code) : std::string("null")) + "}}"};
}
Response denied_node() { return fail(403, "permission_error", "unknown pairing code or wrong node token", "unknown_node"); }
Response job_closed() { return fail(409, "conflict_error", "no open job with this id for this node", "job_closed"); }

void close_job(Job& j, const char* status, const char* reason, Clock::time_point now) {
    j.open = false;
    j.status = status;
    if (reason && *reason) j.reason = reason;
    j.finished = now;
}

// Expires idle nodes, stale jobs and old finished jobs. g_mu held.
void sweep_locked() {
    const auto now = Clock::now();
    g_nodes.erase(std::remove_if(g_nodes.begin(), g_nodes.end(), [&](const Node& n) { return now - n.seen > kNodeIdle; }),
                  g_nodes.end());
    for (auto& jp : g_jobs) {
        Job& j = *jp;
        if (!j.open) continue;
        const bool node_alive = std::any_of(g_nodes.begin(), g_nodes.end(), [&](const Node& n) { return n.token == j.token; });
        if (!node_alive) close_job(j, "error", "the client stopped polling (offline)", now);
        else if ((j.status == "queued" || j.status == "delivered") && now - j.created > kQueuedMaxAge)
            close_job(j, "expired", "the client never confirmed the job", now);
        else if (j.status == "pending" && now - j.state_since > kPendingMaxAge)
            close_job(j, "expired", "no decision from the user", now);
        else if (now - j.created > kOpenMaxAge)
            close_job(j, "expired", "the job ran too long", now);
    }
    size_t finished = 0;
    for (const auto& j : g_jobs)
        if (!j->open) ++finished;
    for (size_t i = 0; i < g_jobs.size();) {   // oldest first (insertion order)
        const Job& j = *g_jobs[i];
        if (!j.open && (now - j.finished > kFinishedKeep || finished > kMaxFinishedJobs)) {
            g_jobs.erase(g_jobs.begin() + static_cast<std::ptrdiff_t>(i));
            --finished;
        } else {
            ++i;
        }
    }
}

// The node that has both this code and this token (constant-time token comparison against every node).
Node* find_node_locked(const std::string& code, const std::string& token) {
    Node* hit = nullptr;
    for (Node& n : g_nodes) {
        const bool token_ok = veltra::rc::ct_equal(n.token, token);
        if (token_ok && n.code == code) hit = &n;
    }
    return hit;
}

bool token_is_live_locked(const std::string& token, bool need_open_job) {
    bool live = false;
    for (const Node& n : g_nodes)
        if (veltra::rc::ct_equal(n.token, token)) live = true;
    if (!live || !need_open_job) return live;
    return std::any_of(g_jobs.begin(), g_jobs.end(),
                       [&](const std::unique_ptr<Job>& j) { return j->open && veltra::rc::ct_equal(j->token, token); });
}

Job* find_job_locked(const std::string& id) {
    for (auto& j : g_jobs)
        if (j->id == id) return j.get();
    return nullptr;
}

bool admin_ok(const lbhttp::Request& req, bool local_admin) {
    if (!g_cfg.relay_only && local_admin) return true;
    // the public mode is never reached through the proxy for admin routes: refuse anything that carries proxy headers
    if (g_cfg.relay_only &&
        (!req.header("x-forwarded-for").empty() || !req.header("forwarded").empty() || !req.header("x-real-ip").empty()))
        return false;
    if (g_cfg.admin_token.size() < (g_cfg.relay_only ? 32u : 1u)) return false;
    std::string given = req.header("x-admin-token");
    if (given.empty()) {
        const std::string auth = req.header("authorization");
        if (auth.size() > 7 && lbhttp::lower_ascii(std::string_view(auth).substr(0, 7)) == "bearer ") given = auth.substr(7);
    }
    return !given.empty() && veltra::rc::ct_equal(given, g_cfg.admin_token);
}

Response admin_denied() {
    if (g_cfg.relay_only && g_cfg.admin_token.empty())
        return fail(403, "permission_error", "admin routes are disabled: start the relay with VELTRA_ADMIN_TOKEN (at least 32 characters)",
                    "admin_disabled");
    return fail(403, "permission_error", "a valid admin token is required (X-Admin-Token or Authorization: Bearer)", "admin_required");
}

bool parse_body(const lbhttp::Request& req, Json& root) {
    std::string err;
    return lbhttp::parse_json(req.body, root, err, kMaxJsonNodes) && root.kind == Json::Kind::Object;
}

std::string raw_str(const Json& o, const char* key) {
    const Json* v = o.get(key);
    return v && v->kind == Json::Kind::String && v->str.size() <= 128 ? v->str : std::string();
}

Response do_register(const lbhttp::Request& req) {
    const std::string token = req.header("x-veltra-node-token");
    if (!veltra::rc::valid_node_token(token))
        return fail(400, "invalid_request_error", "X-Veltra-Node-Token must be 32..128 hex characters");
    if (!req.body.empty()) {
        Json root;
        if (!parse_body(req, root)) return fail(400, "invalid_request_error", "body must be empty or a JSON object");
    }
    std::lock_guard<std::mutex> lk(g_mu);
    sweep_locked();
    const auto now = Clock::now();
    auto ok = [](const std::string& code) {
        return Response{200, "{\"code\":" + json_quote(code) + ",\"poll_ms\":2000,\"expires_idle_s\":120}"};
    };
    for (Node& n : g_nodes)
        if (veltra::rc::ct_equal(n.token, token)) {
            n.seen = now;
            return ok(n.code);
        }
    g_bucket = std::min(kRegisterBurst, g_bucket + std::chrono::duration<double>(now - g_bucket_at).count() * kRegisterPerSecond);
    g_bucket_at = now;
    if (g_bucket < 1.0) return fail(429, "rate_limit_error", "too many registrations; try again shortly", "rate_limited");
    if (g_nodes.size() >= kMaxNodes) return fail(503, "server_error", "the relay has no free node slots", "relay_full");
    g_bucket -= 1.0;
    std::string code;
    for (int tries = 0; tries < 32 && code.empty(); ++tries) {
        uint32_t v = 0;
        if (!veltra::rc::csprng(&v, sizeof v)) return fail(500, "server_error", "no random source");
        if (v >= 4294000000u) continue;   // unbiased modulo
        char buf[16];
        std::snprintf(buf, sizeof buf, "VLT-%06u", static_cast<unsigned>(v % 1000000u));
        const bool used = std::any_of(g_nodes.begin(), g_nodes.end(), [&](const Node& n) { return n.code == buf; }) ||
                          std::any_of(g_jobs.begin(), g_jobs.end(), [&](const std::unique_ptr<Job>& j) { return j->code == buf; });
        if (!used) code = buf;
    }
    if (code.empty()) return fail(500, "server_error", "could not allocate a pairing code");
    g_nodes.push_back(Node{code, token, now});
    return ok(code);
}

Response do_poll(const lbhttp::Request& req) {
    if (req.query.size() != 15 || req.query.compare(0, 5, "code=") != 0 || !veltra::rc::valid_code(std::string_view(req.query).substr(5)))
        return fail(400, "invalid_request_error", "use ?code=VLT-123456");
    const std::string code = req.query.substr(5), token = req.header("x-veltra-node-token");
    std::lock_guard<std::mutex> lk(g_mu);
    sweep_locked();
    Node* n = find_node_locked(code, token);
    if (!n) return denied_node();
    n->seen = Clock::now();
    for (auto& j : g_jobs)
        if (j->open && j->code == code && j->token == n->token && (j->status == "queued" || j->status == "delivered")) {
            j->status = "delivered";   // repeated until the client confirms with progress or a result
            return {200, "{\"job\":{\"id\":" + json_quote(j->id) + ",\"suite\":" + json_quote(j->suite) + "}}"};
        }
    return {200, "{\"job\":null}"};
}

std::string progress_json(const Json& p) {
    if (p.kind != Json::Kind::Object) return "null";
    auto integer = [&](const char* k) {
        const Json* v = p.get(k);
        int64_t x = 0;
        return v && v->as_int(x) && x >= 0 && x <= 1000 ? std::to_string(x) : std::string("0");
    };
    const Json* t = p.get("tok_s");
    const bool has_t = t && t->kind == Json::Kind::Number && std::isfinite(t->number) && t->number >= 0 && t->number < 1e9;
    return "{\"step\":" + integer("step") + ",\"total\":" + integer("total") + ",\"name\":" + json_quote(veltra::rc::str_field(p, "name", 80)) +
           ",\"message\":" + json_quote(veltra::rc::str_field(p, "message", 300)) +
           ",\"tok_s\":" + (has_t ? lbhttp::json_number(t->number) : std::string("null")) + "}";
}

Response do_progress(const lbhttp::Request& req) {
    Json root;
    if (!parse_body(req, root)) return fail(400, "invalid_request_error", "body must be a JSON object");
    const std::string code = raw_str(root, "code"), id = raw_str(root, "job_id"), state = raw_str(root, "state");
    if (!veltra::rc::valid_code(code)) return denied_node();
    std::lock_guard<std::mutex> lk(g_mu);
    sweep_locked();
    Node* n = find_node_locked(code, req.header("x-veltra-node-token"));
    if (!n) return denied_node();
    n->seen = Clock::now();
    Job* j = veltra::rc::valid_job_id(id) ? find_job_locked(id) : nullptr;
    if (!j || !j->open || j->code != code || j->token != n->token) return job_closed();
    if (state != "pending" && state != "running") return fail(400, "invalid_request_error", "state must be pending or running");
    const std::string next = j->status == "running" ? std::string("running") : state;
    if (next != j->status) {
        j->status = next;
        j->state_since = Clock::now();
    }
    if (const Json* p = root.get("progress")) j->progress = progress_json(*p);
    return {200, "{\"ok\":true}"};
}

bool is_result_name(const std::string& n) {   // <yyyymmddThhmmssZ>_VLT-######_<suite>[-N].json
    if (n.size() < 16 + 1 + 10 + 1 + 1 + 5 || n.size() > 100 || n.compare(n.size() - 5, 5, ".json") != 0) return false;
    for (size_t i = 0; i < 16; ++i) {
        const bool ok = i == 8 ? n[i] == 'T' : i == 15 ? n[i] == 'Z' : veltra::rc::is_digit(n[i]);
        if (!ok) return false;
    }
    if (n[16] != '_' || !veltra::rc::valid_code(std::string_view(n).substr(17, 10)) || n[27] != '_') return false;
    for (size_t i = 28; i < n.size() - 5; ++i)
        if (!((n[i] >= 'a' && n[i] <= 'z') || n[i] == '_' || n[i] == '-' || veltra::rc::is_digit(n[i]))) return false;
    return true;
}

// A new file that only its owner can read (reports describe the user's hardware; the relay host may be shared).
std::FILE* open_private(const fs::path& p) {
#if defined(_WIN32)
    return std::fopen(p.string().c_str(), "wb");
#else
    ::unlink(p.c_str());   // a leftover from a crash
    const int fd = ::open(p.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) return nullptr;
    std::FILE* f = ::fdopen(fd, "wb");
    if (!f) ::close(fd);
    return f;
#endif
}

// Writes the report of one finished, admin-dispatched job. The file name is built only from validated parts.
// Returns the file name, or "" with *why set.
std::string save_result_locked(const Job& j, const std::string& content, std::string* why) {
    const veltra::tests::SuiteInfo* suite = veltra::tests::find_suite(j.suite);
    if (!suite || !veltra::rc::valid_code(j.code)) {
        *why = "internal: invalid job";
        return {};
    }
    std::error_code ec;
    const fs::path dir(g_cfg.results_dir);
    size_t files = 0;
    uintmax_t bytes = 0;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        std::error_code fe;
        if (!it->is_regular_file(fe)) continue;
        ++files;
        const uintmax_t s = it->file_size(fe);
        if (!fe) bytes += s;
    }
    if (ec) {
        *why = "the results directory cannot be read";
        return {};
    }
    if (files >= kMaxResultFiles || bytes + content.size() > kMaxResultDirBytes) {
        *why = "the results directory is full (500 files / 100 MiB); remove old results";
        return {};
    }
    const std::string base = veltra::rc::utc_stamp("%Y%m%dT%H%M%SZ") + "_" + j.code + "_" + suite->id;
    std::string name = base + ".json";
    for (int k = 2; fs::exists(dir / name, ec) && k < 100; ++k) name = base + "-" + std::to_string(k) + ".json";
    const fs::path tmp = dir / (name + ".tmp"), dst = dir / name;
    std::FILE* f = open_private(tmp);
    if (!f) {
        *why = "cannot write into the results directory";
        return {};
    }
    bool ok = std::fwrite(content.data(), 1, content.size(), f) == content.size() && std::fflush(f) == 0;
    ok = std::fclose(f) == 0 && ok;
    if (ok) fs::rename(tmp, dst, ec);
    if (!ok || ec) {
        fs::remove(tmp, ec);
        *why = "cannot write the result file";
        return {};
    }
    return name;
}

Response do_result(const lbhttp::Request& req) {
    Json root;
    if (!parse_body(req, root)) return fail(400, "invalid_request_error", "body must be a JSON object (at most 20000 values)");
    const std::string code = raw_str(root, "code"), id = raw_str(root, "job_id"), status = raw_str(root, "status");
    if (!veltra::rc::valid_code(code)) return denied_node();
    std::string report = "null";
    if (const Json* r = root.get("report"); r && r->kind != Json::Kind::Null) {
        if (r->kind != Json::Kind::Object) return fail(400, "invalid_request_error", "report must be an object");
        report.clear();
        veltra::rc::dump_clean(report, *r);
        if (report.size() > kMaxResultBody + (kMaxResultBody / 2)) return fail(413, "invalid_request_error", "report too large");
    }
    std::lock_guard<std::mutex> lk(g_mu);
    sweep_locked();
    Node* n = find_node_locked(code, req.header("x-veltra-node-token"));
    if (!n) return denied_node();
    n->seen = Clock::now();
    Job* j = veltra::rc::valid_job_id(id) ? find_job_locked(id) : nullptr;
    if (!j || !j->open || j->code != code || j->token != n->token) return job_closed();
    const char* st = status == "done" ? "done" : status == "rejected" ? "rejected" : status == "cancelled" ? "cancelled"
                     : status == "error" ? "error" : nullptr;
    if (!st) return fail(400, "invalid_request_error", "status must be done, rejected, cancelled or error");
    const std::string reason = veltra::rc::str_field(root, "reason", 200);
    close_job(*j, st, reason.c_str(), Clock::now());   // exactly one final result per job
    j->result = report;
    const std::string content = "{\"received_utc\":" + json_quote(veltra::rc::utc_stamp("%Y-%m-%dT%H:%M:%SZ")) + ",\"code\":" + json_quote(j->code) +
                                ",\"suite\":" + json_quote(j->suite) + ",\"job_id\":" + json_quote(j->id) + ",\"status\":" + json_quote(j->status) +
                                ",\"reason\":" + (j->reason.empty() ? std::string("null") : json_quote(j->reason)) + ",\"report\":" + report + "}\n";
    std::string why;
    j->saved_file = content.size() <= kMaxResultFileBytes ? save_result_locked(*j, content, &why) : std::string();
    if (j->saved_file.empty()) std::fprintf(stderr, "[veltra-relay] result not saved: %s\n", why.empty() ? "too large" : why.c_str());
    return {200, std::string("{\"ok\":true,\"saved\":") + (j->saved_file.empty() ? "false" : "true") + "}"};
}

Response do_dispatch(const lbhttp::Request& req) {
    Json root;
    if (!parse_body(req, root)) return fail(400, "invalid_request_error", "body must be a JSON object {\"code\",\"suite\"}");
    const std::string code = raw_str(root, "code"), suite = raw_str(root, "suite");
    if (!veltra::rc::valid_code(code)) return fail(400, "invalid_request_error", "code must look like VLT-123456", "bad_code");
    const veltra::tests::SuiteInfo* s = veltra::tests::find_suite(suite);
    if (!s) {
        std::string known;
        for (const auto& x : veltra::tests::suites()) known += (known.empty() ? "" : ", ") + std::string(x.id);
        return fail(400, "invalid_request_error", "unknown suite; allowed: " + known, "unknown_suite");
    }
    std::lock_guard<std::mutex> lk(g_mu);
    sweep_locked();
    const Node* node = nullptr;
    for (const Node& n : g_nodes)
        if (n.code == code) node = &n;
    if (!node) return fail(404, "not_found_error", "no live client with this pairing code", "node_not_found");
    for (const auto& j : g_jobs)
        if (j->open && j->code == code) return fail(409, "conflict_error", "this client already has an open job", "job_open");
    const std::string rnd = veltra::rc::random_hex(8);
    if (rnd.empty()) return fail(500, "server_error", "no random source");
    auto j = std::make_unique<Job>();
    j->id = "job-" + rnd;
    j->code = code;
    j->token = node->token;
    j->suite = s->id;
    j->created = j->state_since = Clock::now();
    const std::string body = "{\"job_id\":" + json_quote(j->id) + ",\"code\":" + json_quote(code) + ",\"suite\":" + json_quote(j->suite) +
                             ",\"status\":\"queued\"}";
    g_jobs.push_back(std::move(j));
    return {200, body};
}

Response do_job(const lbhttp::Request& req) {
    if (req.query.compare(0, 3, "id=") != 0 || !veltra::rc::valid_job_id(std::string_view(req.query).substr(3)))
        return fail(400, "invalid_request_error", "use ?id=<job id>");
    std::lock_guard<std::mutex> lk(g_mu);
    sweep_locked();
    const Job* j = find_job_locked(req.query.substr(3));
    if (!j) return fail(404, "not_found_error", "unknown job (finished jobs are kept for one hour)", "job_not_found");
    const auto age = std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - j->created).count();
    return {200, "{\"job_id\":" + json_quote(j->id) + ",\"code\":" + json_quote(j->code) + ",\"suite\":" + json_quote(j->suite) +
                     ",\"status\":" + json_quote(j->status) + ",\"open\":" + (j->open ? "true" : "false") + ",\"age_s\":" + std::to_string(age) +
                     ",\"reason\":" + (j->reason.empty() ? std::string("null") : json_quote(j->reason)) + ",\"progress\":" + j->progress +
                     ",\"saved_file\":" + (j->saved_file.empty() ? std::string("null") : json_quote(j->saved_file)) +
                     ",\"result\":" + j->result + "}"};
}

Response do_results() {
    std::lock_guard<std::mutex> lk(g_mu);
    std::error_code ec;
    const fs::path dir(g_cfg.results_dir);
    std::vector<std::string> names;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end && names.size() < 4 * kMaxResultFiles; it.increment(ec)) {
        std::error_code fe;
        const std::string n = it->path().filename().string();
        if (it->is_regular_file(fe) && is_result_name(n)) names.push_back(n);
    }
    std::sort(names.rbegin(), names.rend());
    const size_t total = names.size();
    if (names.size() > kMaxListed) names.resize(kMaxListed);
    std::string out = "{\"total\":" + std::to_string(total) + ",\"results\":[";
    bool first = true;
    for (const std::string& n : names) {
        std::string text;
        if (std::FILE* f = std::fopen((dir / n).string().c_str(), "rb")) {
            char buf[65536];
            size_t got;
            while ((got = std::fread(buf, 1, sizeof buf, f)) > 0 && text.size() <= kMaxResultFileBytes) text.append(buf, got);
            std::fclose(f);
        }
        Json j;
        std::string jerr;
        if (text.size() > kMaxResultFileBytes || !lbhttp::parse_json(text, j, jerr, kMaxJsonNodes + 16) || j.kind != Json::Kind::Object) continue;
        std::string headline, device;
        if (const Json* r = j.get("report")) {
            headline = veltra::rc::str_field(*r, "headline", 300);
            if (const Json* d = r->get("device")) device = veltra::rc::str_field(*d, "engine_device", 80);
        }
        out += std::string(first ? "" : ",") + "{\"file\":" + json_quote(n) + ",\"received_utc\":" +
               json_quote(veltra::rc::str_field(j, "received_utc", 32)) + ",\"code\":" + json_quote(veltra::rc::str_field(j, "code", 16)) +
               ",\"suite\":" + json_quote(veltra::rc::str_field(j, "suite", 32)) + ",\"status\":" +
               json_quote(veltra::rc::str_field(j, "status", 16)) + ",\"headline\":" + json_quote(headline) + ",\"device\":" + json_quote(device) + "}";
        first = false;
    }
    return {200, out + "]}"};
}

enum class Route { Register, Poll, Progress, Result, Dispatch, Job, Results, Unknown };

struct RouteInfo {
    const char* path;
    Route route;
    const char* method;
    size_t max_body;
    const char* label;
};
constexpr RouteInfo kRoutes[] = {
    {"/api/relay/register", Route::Register, "POST", kMaxRegisterBody, "register"},
    {"/api/relay/poll", Route::Poll, "GET", 0, "poll"},
    {"/api/relay/progress", Route::Progress, "POST", kMaxProgressBody, "progress"},
    {"/api/relay/result", Route::Result, "POST", kMaxResultBody, "result"},
    {"/api/relay/dispatch", Route::Dispatch, "POST", kMaxDispatchBody, "dispatch"},
    {"/api/relay/job", Route::Job, "GET", 0, "job"},
    {"/api/relay/results", Route::Results, "GET", 0, "results"},
};

const RouteInfo* find_route(const std::string& path) {
    for (const RouteInfo& r : kRoutes)
        if (path == r.path) return &r;
    return nullptr;
}

bool is_admin_route(Route r) { return r == Route::Dispatch || r == Route::Job || r == Route::Results; }

// ---- --relay-only: in-flight result bodies ------------------------------------------------------

std::atomic<int> g_large_bodies{0};

// Held from the header check until the result has been handled; released on every exit path (also a dropped upload).
struct ResultSlot {
    std::string job_id;
    bool large = false;
    ResultSlot() = default;
    ResultSlot(const ResultSlot&) = delete;
    ResultSlot& operator=(const ResultSlot&) = delete;
    ~ResultSlot() {
        if (large) g_large_bodies--;
        if (job_id.empty()) return;
        std::lock_guard<std::mutex> lk(g_mu);
        if (Job* j = find_job_locked(job_id)) j->receiving = false;
    }
};

// ---- --relay-only: strict HTTP/1.x reader ------------------------------------------------------

constexpr int kMaxConns = 64;
constexpr int kHeaderDeadlineMs = 10000;
constexpr int kBodyDeadlineMs = 20000;
constexpr size_t kMaxHead = 8u << 10;
constexpr int kMaxHeaderCount = 48;
std::atomic<int> g_conns{0};

int ms_left(Clock::time_point deadline) {
    return static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count());
}

// Appends what the socket has to `buf`, never waiting past `deadline`. False on close, error or deadline.
bool read_more(sock_t c, std::string& buf, Clock::time_point deadline) {
    const int left = ms_left(deadline);
    if (left <= 0) return false;
    lbhttp::set_timeouts(c, left, 10000);
    char tmp[16384];
    const long n = lbhttp::recv_some(c, tmp, sizeof tmp);
    if (n <= 0) return false;
    buf.append(tmp, static_cast<size_t>(n));
    return true;
}

void reply(sock_t c, const Response& r) {
    const char* reason = r.status == 429 ? "Too Many Requests" : lbhttp::reason_phrase(r.status);
    std::string h = "HTTP/1.1 " + std::to_string(r.status) + " " + reason +
                    "\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(r.body.size()) +
                    "\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n\r\n";
    h += r.body;
    lbhttp::set_timeouts(c, 1000, 10000);
    lbhttp::send_all(c, h);
}

// Half-close, then swallow unread input for at most 500 ms / 2 MiB so the peer sees the reply instead of a reset.
void close_bounded(sock_t c) {
    ::shutdown(c, lbhttp::kShutWr);
    const auto deadline = Clock::now() + std::chrono::milliseconds(500);
    std::string sink;
    size_t total = 0;
    while (total < (2u << 20) && read_more(c, sink, deadline)) {
        total += sink.size();
        sink.clear();
    }
    lbhttp::close_sock(c);
}

bool token_char(char ch) {
    return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || veltra::rc::is_digit(ch) ||
           std::string_view("!#$%&'*+-.^_`|~").find(ch) != std::string_view::npos;
}

// Serves one connection. Returns false when nothing should be sent (peer gone or too slow).
bool serve_request(sock_t c, Response& out, std::string& log_method, const char*& log_route, std::string& log_code) {
    const auto start = Clock::now();
    const auto head_deadline = start + std::chrono::milliseconds(kHeaderDeadlineMs);
    std::string buf;
    size_t head_end;
    while ((head_end = buf.find("\r\n\r\n")) == std::string::npos) {
        if (buf.size() > kMaxHead) {
            out = fail(431, "invalid_request_error", "headers too large");
            return true;
        }
        if (!read_more(c, buf, head_deadline)) return false;
    }
    if (head_end > kMaxHead) {
        out = fail(431, "invalid_request_error", "headers too large");
        return true;
    }
    auto bad = [&](const char* m) {
        out = fail(400, "invalid_request_error", m);
        return true;
    };
    lbhttp::Request req;
    const std::string_view head(buf.data(), head_end + 2);
    size_t pos = 0;
    bool first = true;
    int n_headers = 0, n_cl = 0;
    bool has_te = false;
    std::string cl;
    while (pos < head.size()) {
        const size_t eol = head.find("\r\n", pos);
        const std::string_view line = head.substr(pos, eol - pos);
        pos = eol + 2;
        for (unsigned char ch : line)
            if ((ch < 0x20 && ch != '\t') || ch == 0x7F) return bad("control character in request");
        if (first) {
            first = false;
            const size_t a = line.find(' ');
            const size_t b = a == std::string_view::npos ? a : line.find(' ', a + 1);
            if (b == std::string_view::npos || line.find(' ', b + 1) != std::string_view::npos) return bad("malformed request line");
            req.method = std::string(line.substr(0, a));
            req.target = std::string(line.substr(a + 1, b - a - 1));
            const std::string_view ver = line.substr(b + 1);
            if (ver != "HTTP/1.1" && ver != "HTTP/1.0") return bad("only HTTP/1.x is supported");
            if (req.target.empty() || req.target[0] != '/' || req.target.size() > 512) return bad("malformed request target");
            for (unsigned char ch : req.target)
                if (ch <= 0x20 || ch >= 0x7F) return bad("malformed request target");
            continue;
        }
        if (line.empty()) continue;
        if (++n_headers > kMaxHeaderCount) return bad("too many headers");
        const size_t colon = line.find(':');
        if (colon == std::string_view::npos || colon == 0) return bad("malformed header line");
        for (size_t i = 0; i < colon; ++i)
            if (!token_char(line[i])) return bad("malformed header name");   // also rejects folded lines and "Name : v"
        std::string name = lbhttp::lower_ascii(line.substr(0, colon));
        std::string value(lbhttp::trim_ws(line.substr(colon + 1)));
        if (name == "content-length") {
            ++n_cl;
            cl = value;
        }
        if (name == "transfer-encoding") has_te = true;
        req.headers.emplace_back(std::move(name), std::move(value));
    }
    log_method = req.method == "GET" ? "GET" : req.method == "POST" ? "POST" : "other";
    if (has_te) return bad("Transfer-Encoding is not accepted; send Content-Length");
    if (n_cl > 1) return bad("duplicate Content-Length");
    size_t body_len = 0;
    if (n_cl == 1 && !lbhttp::parse_decimal(cl, body_len)) return bad("invalid Content-Length");

    const size_t qm = req.target.find('?');
    req.path = req.target.substr(0, qm);
    req.query = qm == std::string::npos ? std::string() : req.target.substr(qm + 1);

    if (req.path == "/health" && req.method == "GET" && body_len == 0) {
        log_route = "health";
        out = {200, "{\"status\":\"ok\",\"relay_only\":true}"};
        return true;
    }
    const RouteInfo* r = find_route(req.path);
    if (!r) {
        out = fail(404, "not_found_error", "not found");
        return true;
    }
    log_route = r->label;
    if (req.method != r->method) {
        out = fail(405, "invalid_request_error", std::string("use ") + r->method);
        return true;
    }
    if (std::string_view(r->method) == "POST" && n_cl != 1) {
        out = fail(411, "invalid_request_error", "Content-Length required");
        return true;
    }
    if (body_len > r->max_body) {
        out = fail(413, "invalid_request_error", "request body too large for this route");
        return true;
    }
    // authenticate from the headers before a single body byte is buffered
    if (is_admin_route(r->route)) {
        if (!admin_ok(req, false)) {
            out = admin_denied();
            return true;
        }
    }
    ResultSlot slot;
    if (r->route == Route::Progress || r->route == Route::Result) {
        const std::string token = req.header("x-veltra-node-token");
        std::lock_guard<std::mutex> lk(g_mu);
        sweep_locked();
        if (!token_is_live_locked(token, false)) {
            out = denied_node();
            return true;
        }
        if (!token_is_live_locked(token, true)) {
            out = job_closed();
            return true;
        }
        if (r->route == Route::Result) {   // one result per open job at a time: a body costs a multiple of its size
            Job* open = nullptr;
            for (const auto& j : g_jobs)
                if (j->open && veltra::rc::ct_equal(j->token, token)) open = j.get();
            if (!open || open->receiving) {
                out = fail(409, "conflict_error", "a result for this job is already being received", "result_in_flight");
                return true;
            }
            if (body_len > kLargeBody) {
                if (g_large_bodies.fetch_add(1) >= kMaxLargeInFlight) {
                    g_large_bodies--;
                    out = fail(503, "server_error", "too many large results in flight; try again shortly", "busy");
                    return true;
                }
                slot.large = true;
            }
            open->receiving = true;
            slot.job_id = open->id;
        }
    }
    if (body_len) {
        const auto body_deadline = start + std::chrono::milliseconds(kBodyDeadlineMs);
        std::string body = buf.substr(head_end + 4);
        if (lbhttp::lower_ascii(req.header("expect")).find("100-continue") != std::string::npos && body.empty())
            if (!lbhttp::send_all(c, std::string_view("HTTP/1.1 100 Continue\r\n\r\n"))) return false;
        body.reserve(body_len);
        while (body.size() < body_len)
            if (!read_more(c, body, body_deadline)) return false;
        if (body.size() > body_len) return bad("more body bytes than Content-Length");
        req.body = std::move(body);
    } else if (buf.size() > head_end + 4) {
        return bad("unexpected request body");
    }
    if (r->route == Route::Poll && veltra::rc::valid_code(std::string_view(req.query).substr(std::min<size_t>(5, req.query.size()))))
        log_code = req.query.substr(5);
    if (r->route == Route::Progress || r->route == Route::Result || r->route == Route::Dispatch) {
        Json j;
        if (req.body.size() <= kMaxProgressBody && parse_body(req, j)) {
            const std::string code = raw_str(j, "code");
            if (veltra::rc::valid_code(code)) log_code = code;
        }
    }
    out = handle(req, false);
    return true;
}

// At most 60 request lines per minute, so anonymous traffic cannot flood the system log; the rest is only counted.
bool log_allowed() {
    static std::mutex mu;
    static auto window = Clock::now();
    static int used = 0;
    static unsigned long long suppressed = 0;
    std::lock_guard<std::mutex> lk(mu);
    if (Clock::now() - window > std::chrono::minutes(1)) {
        if (suppressed) std::fprintf(stderr, "[veltra-relay] %llu log lines suppressed\n", suppressed);
        window = Clock::now();
        used = 0;
        suppressed = 0;
    }
    if (used >= 60) {
        ++suppressed;
        return false;
    }
    ++used;
    return true;
}

void serve_connection(sock_t c) {
    Response res;
    std::string method = "?", code = "-";
    const char* route = "other";
    bool send = false;
    try {
        send = serve_request(c, res, method, route, code);
    } catch (const std::exception&) {
        res = fail(500, "server_error", "internal error");
        send = true;
    } catch (...) {
        res = fail(500, "server_error", "internal error");
        send = true;
    }
    if (!send) {
        lbhttp::close_sock(c);
        return;
    }
    try {
        reply(c, res);
        const bool quiet = res.status == 200 && (std::string_view(route) == "poll" || std::string_view(route) == "progress" ||
                                                 std::string_view(route) == "health" || std::string_view(route) == "job");
        // a stored result and a dispatch are always logged
        const bool always = res.status == 200 && (std::string_view(route) == "result" || std::string_view(route) == "dispatch");
        if (!quiet && (always || log_allowed()))
            std::fprintf(stderr, "[veltra-relay] %s %s %s %d\n", method.c_str(), route, code.c_str(), res.status);
    } catch (...) {
    }
    close_bounded(c);
}

}  // namespace

std::string init(const Config& cfg) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_cfg = cfg;
    if (g_cfg.results_dir.empty()) g_cfg.results_dir = "remote_gpu_results";
    std::error_code ec;
    fs::create_directories(fs::path(g_cfg.results_dir), ec);
    std::error_code ec2;
    if (!fs::is_directory(fs::path(g_cfg.results_dir), ec2)) return "cannot create the results directory " + g_cfg.results_dir;
    const fs::path probe = fs::path(g_cfg.results_dir) / ".write-test";
    std::FILE* f = std::fopen(probe.string().c_str(), "wb");
    if (!f) return "the results directory is not writable: " + g_cfg.results_dir;
    std::fclose(f);
    fs::remove(probe, ec2);
    return {};
}

Response handle(const lbhttp::Request& req, bool local_admin) {
    try {
        const RouteInfo* r = find_route(req.path);
        if (!r) return fail(404, "not_found_error", "not found");
        if (req.method != r->method) return fail(405, "invalid_request_error", std::string("use ") + r->method);
        if (req.body.size() > r->max_body) return fail(413, "invalid_request_error", "request body too large for this route");
        if (is_admin_route(r->route) && !admin_ok(req, local_admin)) return admin_denied();
        switch (r->route) {
            case Route::Register: return do_register(req);
            case Route::Poll: return do_poll(req);
            case Route::Progress: return do_progress(req);
            case Route::Result: return do_result(req);
            case Route::Dispatch: return do_dispatch(req);
            case Route::Job: return do_job(req);
            case Route::Results: return do_results();
            default: break;
        }
        return fail(404, "not_found_error", "not found");
    } catch (const std::exception&) {
        return fail(500, "server_error", "internal error");
    }
}

void serve(sock_t listener) {
    for (;;) {
        const sock_t c = lbhttp::accept_client(listener);
        if (c == lbhttp::kBadSock) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        if (g_conns.load() >= kMaxConns) {
            lbhttp::set_timeouts(c, 100, 200);
            reply(c, fail(503, "server_error", "too many connections"));
            ::shutdown(c, lbhttp::kShutWr);
            lbhttp::close_sock(c);
            continue;
        }
        g_conns++;
        try {
            std::thread([c]() {
                serve_connection(c);
                g_conns--;
            }).detach();
        } catch (...) {
            g_conns--;
            lbhttp::close_sock(c);
        }
    }
}

}  // namespace veltra::relay
