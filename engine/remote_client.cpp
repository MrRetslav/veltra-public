// engine/remote_client.cpp - see remote_client.h.
#include "remote_client.h"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <ctime>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

#include "cluster.h"
#include "platform.h"
#include "remote_common.h"

#if defined(_WIN32)
#  include <wininet.h>
#endif

// Consent bounds. Overridable at compile time only, so the tests can build a client with short intervals.
#ifndef VELTRA_RC_PROMPT_GAP_S
#  define VELTRA_RC_PROMPT_GAP_S 30      // at most one new consent prompt per this many seconds
#endif
#ifndef VELTRA_RC_GRANT_GAP_S
#  define VELTRA_RC_GRANT_GAP_S 600      // "this session": no automatic run sooner after the previous test than this
#endif
#ifndef VELTRA_RC_GRANT_TTL_S
#  define VELTRA_RC_GRANT_TTL_S 3600     // "this session": the grant ends after this long
#endif
#ifndef VELTRA_RC_MAX_SEEN
#  define VELTRA_RC_MAX_SEEN 4096        // job ids remembered per session; at the cap the session stops
#endif

namespace veltra::remote {
namespace {

using Clock = std::chrono::steady_clock;
using Wall = std::chrono::system_clock;
using lbhttp::Json;
using lbhttp::json_quote;
namespace rc = veltra::rc;

constexpr auto kPollEvery = std::chrono::seconds(2);
constexpr auto kPendingTtl = std::chrono::minutes(5);
constexpr auto kProgressEvery = std::chrono::milliseconds(1000);
constexpr int kNetTimeoutMs = 10000;
constexpr size_t kMaxRelayReply = 64u << 10;
constexpr auto kPromptGap = std::chrono::seconds(VELTRA_RC_PROMPT_GAP_S);
constexpr auto kGrantGap = std::chrono::seconds(VELTRA_RC_GRANT_GAP_S);
constexpr auto kGrantTtl = std::chrono::seconds(VELTRA_RC_GRANT_TTL_S);
constexpr int kGrantRuns = 3;               // automatic runs per "this session" answer
constexpr size_t kMaxSeen = VELTRA_RC_MAX_SEEN;
constexpr int kMaxDroppedPrompts = 5;       // prompts the relay may withdraw before the session stops

const char* const kSends[] = {
    "Hardware: processortype en aantal cores, geheugen (RAM), besturingssysteem (Windows/Linux), naam en VRAM van de videokaart",
    "Van het model alleen de bestandsnaam, de modelnaam en de maximale contextlengte",
    "Meetresultaten: snelheden (tok/s, GB/s), tijden en geslaagd/mislukt per test",
    "De antwoorden van het model op de ingebouwde testvragen (nooit je eigen chats)",
    "Niet verzonden: gebruikersnaam, computernaam, mappen of paden, chats, omgevingsvariabelen",
    "De beheerder van de relayserver ziet, zoals bij elke internetverbinding, je IP-adres",
};

// ---- HTTP (outbound only) -------------------------------------------------------------------------

struct NetResult {
    int status = 0;   // 0 = transport failure, see err
    std::string body, err;
};

#if defined(_WIN32)
// WinINet's timeouts apply per operation, so a relay that trickles bytes could keep one call busy for hours. At the
// total deadline this closes the request handle, which makes the blocked call return.
class RequestDeadline {
public:
    void arm(HINTERNET h, int ms) {
        t_ = std::thread([this, h, ms] {
            std::unique_lock<std::mutex> lk(mu_);
            if (!cv_.wait_for(lk, std::chrono::milliseconds(ms), [this] { return finished_; })) {
                fired_ = true;
                lk.unlock();
                InternetCloseHandle(h);
            }
        });
    }
    // Stops the timer. True when it fired: the request handle is then already closed.
    bool disarm() {
        if (t_.joinable()) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                finished_ = true;
            }
            cv_.notify_all();
            t_.join();
        }
        return fired_;
    }
    ~RequestDeadline() { disarm(); }

private:
    std::mutex mu_;
    std::condition_variable cv_;
    bool finished_ = false, fired_ = false;
    std::thread t_;
};

// HTTPS through WinINet with its normal certificate validation (no INTERNET_FLAG_IGNORE_* flag anywhere), no
// redirects (the node token must never follow one to another host), no cookies, no cache, no UI.
NetResult https_request(const rc::RelayUrl& u, const char* method, const std::string& path, const std::string& headers,
                        const std::string& body) {
    NetResult r;
    HINTERNET net = InternetOpenA("Veltra/1.0", INTERNET_OPEN_TYPE_PRECONFIG, nullptr, nullptr, 0);
    HINTERNET con = nullptr, rq = nullptr;
    RequestDeadline limit;
    auto done = [&]() {
        if (limit.disarm()) rq = nullptr;
        if (rq) InternetCloseHandle(rq);
        if (con) InternetCloseHandle(con);
        if (net) InternetCloseHandle(net);
    };
    auto failed = [&](const char* what) {
        r.err = std::string(what) + " (WinINet-fout " + std::to_string(GetLastError()) + ")";
        r.status = 0;
        done();
        return r;
    };
    if (!net) return failed("netwerk starten mislukt");
    DWORD to = kNetTimeoutMs;
    InternetSetOptionA(net, INTERNET_OPTION_CONNECT_TIMEOUT, &to, sizeof to);
    InternetSetOptionA(net, INTERNET_OPTION_SEND_TIMEOUT, &to, sizeof to);
    InternetSetOptionA(net, INTERNET_OPTION_RECEIVE_TIMEOUT, &to, sizeof to);
    con = InternetConnectA(net, u.host.c_str(), static_cast<INTERNET_PORT>(u.port), nullptr, nullptr, INTERNET_SERVICE_HTTP, 0, 0);
    if (!con) return failed("verbinden mislukt");
    const DWORD flags = INTERNET_FLAG_SECURE | INTERNET_FLAG_NO_AUTO_REDIRECT | INTERNET_FLAG_NO_COOKIES | INTERNET_FLAG_NO_AUTH |
                        INTERNET_FLAG_NO_CACHE_WRITE | INTERNET_FLAG_RELOAD | INTERNET_FLAG_PRAGMA_NOCACHE | INTERNET_FLAG_NO_UI;
    rq = HttpOpenRequestA(con, method, path.c_str(), nullptr, nullptr, nullptr, flags, 0);
    if (!rq) return failed("verzoek openen mislukt");
    try {
        limit.arm(rq, body.size() > (64u << 10) ? 60000 : 20000);
    } catch (const std::exception&) {
        return failed("tijdbewaking starten mislukt");
    }
    if (!HttpSendRequestA(rq, headers.c_str(), static_cast<DWORD>(headers.size()),
                          body.empty() ? nullptr : const_cast<char*>(body.data()), static_cast<DWORD>(body.size())))
        return failed("verzenden mislukt (relay onbereikbaar of ongeldig certificaat)");
    DWORD status = 0, len = sizeof status;
    if (!HttpQueryInfoA(rq, HTTP_QUERY_STATUS_CODE | HTTP_QUERY_FLAG_NUMBER, &status, &len, nullptr)) return failed("geen antwoord");
    char buf[4096];
    DWORD got = 0;
    for (;;) {
        if (!InternetReadFile(rq, buf, sizeof buf, &got)) return failed("lezen mislukt");
        if (got == 0) break;
        if (r.body.size() + got > kMaxRelayReply) {
            r.err = "antwoord van de relay is te groot";
            done();
            return r;
        }
        r.body.append(buf, got);
    }
    if (limit.disarm()) {   // the body may be cut off
        rq = nullptr;
        return failed("de relay antwoordt te traag");
    }
    r.status = static_cast<int>(status);
    done();
    return r;
}
#endif

NetResult relay_call(const rc::RelayUrl& u, const char* method, const std::string& route, const std::string& token,
                     const std::string& body) {
    const std::string path = u.path + "/api/relay/" + route;
    std::string headers = "X-Veltra-Node-Token: " + token + "\r\n";
    if (std::string(method) == "POST") headers += "Content-Type: application/json\r\n";
    if (!u.https) {   // parse_relay_url allows plain http for loopback only
        const veltra::cluster::HttpResult h =
            veltra::cluster::http_request(u.host, u.port, method, path, body, headers, kNetTimeoutMs, kMaxRelayReply);
        NetResult r;
        r.status = h.status;
        r.body = h.body;
        if (h.status == 0) r.err = "relay onbereikbaar: " + rc::clean_text(h.err, 120);
        return r;
    }
#if defined(_WIN32)
    return https_request(u, method, path, headers, body);
#else
    NetResult r;
    r.err = "https wordt in deze Linux-build niet ondersteund; gebruik http op 127.0.0.1 via een SSH-tunnel";
    return r;
#endif
}

// error.code of a relay error reply ("" when absent); only ever compared, never shown.
std::string relay_error_code(const std::string& body) {
    Json j;
    std::string err;
    if (!lbhttp::parse_json(body, j, err)) return {};
    const Json* e = j.get("error");
    return e ? rc::str_field(*e, "code", 32) : std::string();
}

// ---- state ----------------------------------------------------------------------------------------

struct Job {
    std::string id;
    std::string nonce;            // made here, never by the relay: a decision must name the job exactly as it was shown
    const tests::SuiteInfo* suite = nullptr;
    const tests::SuiteInfo* approved_suite = nullptr;   // what was approved (user or grant), checked again before the run
    std::string approved_nonce;
    std::string hardware;
    enum class Phase { Pending, Approved, Running, Finished } phase = Phase::Pending;
    Clock::time_point deadline;
    Wall::time_point wall_deadline;
    std::atomic<bool> cancel{false};
    bool acked = false;
    bool dropped = false;         // the relay closed it: nothing to report
    std::string reject_reason;    // set when a pending job is refused (user / timeout / disabled)
    tests::Progress progress;
    uint64_t progress_seq = 0;
    tests::Report report;
};

struct Session {
    rc::RelayUrl url;
    std::string token;
    std::atomic<bool> stop{false};
    // all below under g_mu
    std::string code, error;
    std::shared_ptr<Job> job;
    bool finished_one = false;
    int dropped_prompts = 0;
};

// "Yes for this session": automatic repeats of the approved suite only, bounded in number, spacing and time.
struct Grant {
    const tests::SuiteInfo* suite = nullptr;   // null = no grant
    int runs_left = 0;
    std::string date;
    Clock::time_point deadline;
    Wall::time_point wall_deadline;
};

std::mutex g_mu;
Host g_host;
bool g_url_loaded = false;
std::string g_relay_url;
std::shared_ptr<Session> g_cur;   // non-null exactly while enabled
Grant g_grant;
bool g_ran_before = false;        // g_last_run_end is valid
Clock::time_point g_last_run_end;
std::string g_last_result;        // JSON object without its closing brace ("" = none)
std::string g_last_nonce;         // the job g_last_result belongs to
bool g_last_sent = false;         // the relay acknowledged that job's final result
bool g_last_send_failed = false;  // it never will
std::string g_config_error;

std::mutex g_hook_mu;
std::function<void(bool)> g_hook;

void notify(bool pending) {
    std::function<void(bool)> h;
    {
        std::lock_guard<std::mutex> lk(g_hook_mu);
        h = g_hook;
    }
    if (h) h(pending);
}

std::string local_date() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[16];
    std::strftime(buf, sizeof buf, "%Y-%m-%d", &tm);
    return buf;
}

bool grant_active_locked() {
    if (g_grant.suite && (g_grant.runs_left <= 0 || g_grant.date != local_date() ||   // a new day: ask again
                          Clock::now() >= g_grant.deadline || Wall::now() >= g_grant.wall_deadline))
        g_grant = Grant{};
    return g_grant.suite != nullptr;
}

long long cooldown_left_s_locked() {
    if (!g_ran_before) return 0;
    const auto left = std::chrono::ceil<std::chrono::seconds>(g_last_run_end + kGrantGap - Clock::now()).count();
    return left > 0 ? static_cast<long long>(left) : 0;
}

std::string grant_json_locked() {
    if (!grant_active_locked()) return "null";
    const auto left = std::chrono::duration_cast<std::chrono::seconds>(g_grant.deadline - Clock::now()).count();
    return "{\"suite\":" + json_quote(g_grant.suite->id) + ",\"title\":" + json_quote(g_grant.suite->title) +
           ",\"runs_left\":" + std::to_string(g_grant.runs_left) + ",\"expires_in_s\":" + std::to_string(left > 0 ? left : 0) +
           ",\"cooldown_s\":" + std::to_string(cooldown_left_s_locked()) + "}";
}

std::string settings_path() {
    const std::string d = veltra::plat::app_data_dir();
    if (d.empty()) return {};
#if defined(_WIN32)
    return d + "\\remote.json";
#else
    return d + "/remote.json";
#endif
}

void load_url_locked() {
    if (g_url_loaded) return;
    g_url_loaded = true;
    const std::string path = settings_path();
    if (path.empty()) return;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return;
    char buf[4096];
    const size_t n = std::fread(buf, 1, sizeof buf, f);
    std::fclose(f);
    Json j;
    std::string err;
    if (!lbhttp::parse_json(std::string_view(buf, n), j, err)) return;
    const std::string url = rc::str_field(j, "relay_url", 200);
    rc::RelayUrl u;
    if (rc::parse_relay_url(url, u, err)) g_relay_url = u.normalized;
}

bool save_url_locked() {
    const std::string path = settings_path();
    if (path.empty()) return false;
    const std::string tmp = path + ".tmp", data = "{\"relay_url\":" + json_quote(g_relay_url) + "}\n";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    bool ok = std::fwrite(data.data(), 1, data.size(), f) == data.size();
    ok = std::fclose(f) == 0 && ok;
#if defined(_WIN32)
    ok = ok && MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
#else
    ok = ok && std::rename(tmp.c_str(), path.c_str()) == 0;
#endif
    if (!ok) std::remove(tmp.c_str());
    return ok;
}

std::string steps_json(const std::vector<tests::StepSummary>& steps) {
    std::string o = "[";
    for (size_t i = 0; i < steps.size(); ++i)
        o += std::string(i ? "," : "") + "{\"id\":" + json_quote(steps[i].id) + ",\"name\":" + json_quote(steps[i].name) +
             ",\"status\":" + json_quote(steps[i].status) + ",\"summary\":" + json_quote(steps[i].summary) + "}";
    return o + "]";
}

void set_last_result_locked(const Job& j, const std::string& status, const std::string& headline,
                            const std::vector<tests::StepSummary>& steps) {
    g_last_result = "{\"job_id\":" + json_quote(j.id) + ",\"suite\":" + json_quote(j.suite->id) + ",\"status\":" + json_quote(status) +
                    ",\"headline\":" + json_quote(headline) + ",\"steps\":" + steps_json(steps);
    g_last_nonce = j.nonce;
    g_last_sent = g_last_send_failed = false;
}

// The relay took back a request the user had not answered yet. g_mu held.
void withdraw_locked(Session& s) {
    s.job->dropped = true;
    s.job->phase = Job::Phase::Finished;
    ++s.dropped_prompts;
    set_last_result_locked(*s.job, "cancelled", "Verzoek ingetrokken door de relay; er is niets uitgevoerd", {});
}

// Refuses a pending job (user, timeout or toggle off). g_mu held; the worker reports it to the relay.
void reject_locked(Session& s, const char* reason) {
    if (!s.job || s.job->phase != Job::Phase::Pending) return;
    s.job->phase = Job::Phase::Finished;
    s.job->reject_reason = reason;
    const std::string r = reason;
    set_last_result_locked(*s.job, "rejected",
                           r == "timeout"    ? "Verzoek verlopen: geen antwoord binnen 5 minuten; er is niets uitgevoerd"
                           : r == "disabled" ? "Verzoek geweigerd: de externe testmodus is uitgezet; er is niets uitgevoerd"
                                             : "Verzoek geweigerd; er is niets uitgevoerd",
                           {});
    s.finished_one = true;
}

std::string result_body(const std::string& code, const std::string& job_id, const std::string& status, const std::string& reason,
                        const std::string& report) {
    return "{\"code\":" + json_quote(code) + ",\"job_id\":" + json_quote(job_id) + ",\"status\":" + json_quote(status) +
           (reason.empty() ? std::string() : ",\"reason\":" + json_quote(reason)) +
           (report.empty() ? std::string() : ",\"report\":" + report) + "}";
}

void run_job(std::shared_ptr<Session> s, std::shared_ptr<Job> job, const tests::SuiteInfo* suite, Host host) {
    tests::Report rep;
    try {
        if (!host.run_suite) throw std::runtime_error("no runner");
        rep = host.run_suite(*suite, job->cancel, [&](const tests::Progress& p) {
            std::lock_guard<std::mutex> lk(g_mu);
            job->progress = p;
            ++job->progress_seq;
        });
    } catch (const std::exception&) {
        rep = tests::Report{};
        rep.status = "error";
        rep.headline = "De testrunner is onverwacht gestopt";
    }
    std::lock_guard<std::mutex> lk(g_mu);
    job->report = std::move(rep);
    job->phase = Job::Phase::Finished;
    g_last_run_end = Clock::now();
    g_ran_before = true;
    set_last_result_locked(*job, job->report.status, job->report.headline, job->report.steps);
    s->finished_one = true;
}

struct Outgoing {
    std::string body;
    std::string nonce;   // of the job in last_result ("" = not shown there)
    int tries = 0;
    Clock::time_point next = Clock::now();
};

// Toggle off. g_mu held; returns true when a pending notification has to be withdrawn.
bool stop_session_locked();

// The relay behaves in a way no honest relay does: switch the test mode off and say why.
void stop_misbehaving(const std::shared_ptr<Session>& s, const char* why) {
    bool unpend = false;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_cur != s) return;
        g_config_error = std::string("De relay gedraagt zich verkeerd (") + why + "); de externe testmodus is uitgezet.";
        unpend = stop_session_locked();
    }
    if (unpend) notify(false);
}

void worker(std::shared_ptr<Session> s) {
    // Job ids already handled: a redelivered job is never prompted or run twice. Nothing is ever forgotten; when the
    // set is full the session stops instead.
    std::unordered_set<std::string> seen;
    std::deque<Outgoing> outbox;    // final results
    Clock::time_point next_net = Clock::now(), next_progress = Clock::now(), last_prompt{};
    bool prompted = false;
    uint64_t sent_seq = 0;
    int sent_step = -1, failures = 0;

    auto net_ok = [&] {
        failures = 0;
        std::lock_guard<std::mutex> lk(g_mu);
        s->error.clear();
    };
    auto net_failed = [&](const std::string& why) {
        failures = failures < 5 ? failures + 1 : 5;
        next_net = Clock::now() + std::chrono::seconds(1 << failures);   // 2, 4, 8, 16, 32 s
        std::lock_guard<std::mutex> lk(g_mu);
        s->error = why;
    };
    auto status_error = [](const NetResult& r) {
        if (r.status == 0) return r.err;
        if (r.status == 429) return std::string("de relay vraagt om even te wachten (429)");
        return "de relay antwoordde met HTTP " + std::to_string(r.status);
    };

    for (;;) {
        const bool stopping = s->stop.load();
        const auto now = Clock::now();

        // 1) job transitions (local only)
        std::shared_ptr<Job> job;
        std::string code;
        const tests::SuiteInfo* run_suite = nullptr;
        Host host;
        bool unpend = false, start = false;
        {
            std::lock_guard<std::mutex> lk(g_mu);
            job = s->job;
            code = s->code;
            if (job) {
                if (job->phase == Job::Phase::Pending && (now >= job->deadline || Wall::now() >= job->wall_deadline)) {
                    reject_locked(*s, "timeout");
                    unpend = true;
                }
                if (stopping) {
                    job->cancel = true;
                    if (job->phase == Job::Phase::Pending) {
                        reject_locked(*s, "disabled");
                        unpend = true;
                    } else if (job->phase == Job::Phase::Approved) {   // approved but not started: never start it
                        job->phase = Job::Phase::Finished;
                        job->report.status = "cancelled";
                        job->report.headline = "Geannuleerd voordat de test begon";
                        set_last_result_locked(*job, "cancelled", job->report.headline, {});
                    }
                } else if (job->phase == Job::Phase::Approved) {
                    // last check: only exactly what was approved may start
                    if (job->approved_suite && job->approved_suite == job->suite && !job->approved_nonce.empty() &&
                        job->approved_nonce == job->nonce && tests::find_suite(job->approved_suite->id) == job->approved_suite) {
                        job->phase = Job::Phase::Running;
                        run_suite = job->approved_suite;
                        host = g_host;
                        start = true;
                    } else {
                        job->phase = Job::Phase::Finished;
                        job->report.status = "error";
                        job->report.headline = "Interne controle mislukt; er is niets uitgevoerd";
                        set_last_result_locked(*job, "error", job->report.headline, {});
                    }
                }
            }
        }
        if (unpend) notify(false);
        if (start) {
            sent_seq = 0;
            sent_step = -1;
            next_progress = now;
            try {
                std::thread(run_job, s, job, run_suite, host).detach();
            } catch (const std::exception&) {
                std::lock_guard<std::mutex> lk(g_mu);
                job->phase = Job::Phase::Finished;
                job->report.status = "error";
                job->report.headline = "Kon de testthread niet starten";
                set_last_result_locked(*job, "error", job->report.headline, {});
            }
        }

        // 2) a finished job moves to the outbox
        if (job) {
            std::lock_guard<std::mutex> lk(g_mu);
            if (job->phase == Job::Phase::Finished && s->job == job) {
                if (!job->dropped && !code.empty()) {
                    Outgoing o;
                    o.nonce = job->nonce;
                    if (!job->reject_reason.empty())
                        o.body = result_body(code, job->id, "rejected", job->reject_reason, "");
                    else
                        o.body = result_body(code, job->id, job->report.status, "", job->report.json);
                    outbox.push_back(std::move(o));
                } else if (g_last_nonce == job->nonce) {
                    g_last_send_failed = true;
                }
                s->job.reset();
                job.reset();
            }
        }

        // 3) send final results (bounded retries)
        if (!outbox.empty() && now >= outbox.front().next) {
            Outgoing& o = outbox.front();
            const NetResult r = relay_call(s->url, "POST", "result", s->token, o.body);
            const bool retry_409 = r.status == 409 && relay_error_code(r.body) == "result_in_flight";
            const bool refused = r.status >= 400 && r.status < 500 && r.status != 429 && !retry_409;
            if (r.status == 200 || refused || ++o.tries >= (stopping ? 2 : 8)) {
                std::lock_guard<std::mutex> lk(g_mu);
                const bool shown = !o.nonce.empty() && o.nonce == g_last_nonce;
                if (r.status == 200) {
                    if (shown) g_last_sent = true;
                } else {
                    if (shown) g_last_send_failed = true;
                    s->error = "het testresultaat kon niet worden verzonden: " + status_error(r);
                }
                outbox.pop_front();
            } else {
                o.next = Clock::now() + std::chrono::seconds(stopping ? 1 : 1 << (o.tries < 5 ? o.tries : 5));
            }
        }

        if (stopping) {
            if (!job && outbox.empty()) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // 4) confirm a pending job / send progress of a running one
        if (job && !code.empty()) {
            bool send = false;
            std::string body;
            uint64_t seq = 0;
            int step = -1;
            {
                std::lock_guard<std::mutex> lk(g_mu);
                if (job->phase == Job::Phase::Pending && !job->acked && now >= next_progress) {
                    send = true;
                    body = "{\"code\":" + json_quote(code) + ",\"job_id\":" + json_quote(job->id) + ",\"state\":\"pending\"}";
                } else if (job->phase == Job::Phase::Running && (job->progress_seq != sent_seq || sent_step < 0) &&
                           (now >= next_progress || job->progress.step != sent_step)) {
                    send = true;
                    const tests::Progress& p = job->progress;
                    seq = job->progress_seq;
                    step = p.step;
                    body = "{\"code\":" + json_quote(code) + ",\"job_id\":" + json_quote(job->id) +
                           ",\"state\":\"running\",\"progress\":{\"step\":" + std::to_string(p.step) + ",\"total\":" + std::to_string(p.total) +
                           ",\"name\":" + json_quote(p.name) + ",\"message\":" + json_quote(rc::clean_text(p.message, 300)) +
                           ",\"tok_s\":" + (p.tok_s >= 0 ? lbhttp::json_number(p.tok_s) : std::string("null")) + "}}";
                }
            }
            if (send) {
                const NetResult r = relay_call(s->url, "POST", "progress", s->token, body);
                next_progress = Clock::now() + kProgressEvery;
                if (r.status == 200) {
                    std::lock_guard<std::mutex> lk(g_mu);
                    if (step < 0) job->acked = true;
                    else sent_seq = seq, sent_step = step;
                } else if (r.status == 409 && relay_error_code(r.body) == "job_closed") {
                    bool was_pending = false;
                    int dropped = 0;
                    {
                        std::lock_guard<std::mutex> lk(g_mu);
                        job->dropped = true;
                        if (s->job == job && job->phase == Job::Phase::Pending) {
                            was_pending = true;
                            withdraw_locked(*s);
                        }
                        dropped = s->dropped_prompts;
                    }
                    if (was_pending) notify(false);
                    if (dropped >= kMaxDroppedPrompts) stop_misbehaving(s, "trok te vaak een verzoek weer in");
                }
            }
        }

        // 5) register / poll (also the heartbeat while a suite runs)
        if (now >= next_net) {
            if (code.empty()) {
                const NetResult r = relay_call(s->url, "POST", "register", s->token, "{}");
                Json j;
                std::string jerr;
                std::string got;
                if (r.status == 200 && lbhttp::parse_json(r.body, j, jerr)) got = rc::str_field(j, "code", 16);
                if (rc::valid_code(got)) {
                    {
                        std::lock_guard<std::mutex> lk(g_mu);
                        s->code = got;
                    }
                    net_ok();
                    next_net = Clock::now();
                } else {
                    net_failed(r.status == 200 ? "de relay gaf een ongeldig antwoord" : status_error(r));
                }
            } else {
                const NetResult r = relay_call(s->url, "GET", "poll?code=" + code, s->token, "");
                next_net = Clock::now() + kPollEvery;
                if (r.status == 200) {
                    Json j;
                    std::string jerr;
                    if (!lbhttp::parse_json(r.body, j, jerr) || j.kind != Json::Kind::Object) {
                        net_failed("de relay gaf een ongeldig antwoord");
                    } else {
                        net_ok();
                        const Json* jj = j.get("job");
                        if (jj && jj->kind == Json::Kind::Object) {
                            // only these two fields are read; everything else the relay sends is ignored
                            const Json* idj = jj->get("id");
                            const Json* sj = jj->get("suite");
                            const std::string id = idj && idj->kind == Json::Kind::String ? idj->str : std::string();
                            const std::string sid = sj && sj->kind == Json::Kind::String ? sj->str : std::string();
                            const bool known_id = seen.count(id) != 0;
                            bool busy;
                            {
                                std::lock_guard<std::mutex> lk(g_mu);
                                busy = s->job != nullptr;
                            }
                            if (rc::valid_job_id(id) && !known_id && !busy) {
                                const tests::SuiteInfo* suite = tests::find_suite(sid);
                                bool defer = false;   // too soon for another prompt: left unseen, a later poll offers it again
                                if (suite && prompted && Clock::now() - last_prompt < kPromptGap) {
                                    std::lock_guard<std::mutex> lk(g_mu);
                                    defer = !(grant_active_locked() && g_grant.suite == suite && cooldown_left_s_locked() == 0);
                                }
                                const std::string nonce = rc::random_hex(16);
                                if (seen.size() >= kMaxSeen) {
                                    stop_misbehaving(s, "stuurde te veel verschillende verzoeken");
                                } else if (defer) {
                                } else if (!suite) {
                                    seen.insert(id);
                                    Outgoing o;
                                    o.body = result_body(code, id, "error", "unknown suite for this app version", "");
                                    outbox.push_back(std::move(o));
                                } else if (nonce.empty()) {
                                    net_failed("geen veilige willekeurige getallen beschikbaar");
                                } else {
                                    seen.insert(id);
                                    std::function<std::string()> hwf;
                                    {
                                        std::lock_guard<std::mutex> lk(g_mu);
                                        hwf = g_host.hardware_line;
                                    }
                                    auto nj = std::make_shared<Job>();
                                    nj->id = id;
                                    nj->nonce = nonce;
                                    nj->suite = suite;
                                    nj->hardware = hwf ? hwf() : std::string();
                                    nj->deadline = Clock::now() + kPendingTtl;
                                    nj->wall_deadline = Wall::now() + kPendingTtl;
                                    bool pending = false;
                                    {
                                        std::lock_guard<std::mutex> lk(g_mu);
                                        if (!s->stop.load() && !s->job) {
                                            if (grant_active_locked() && g_grant.suite == suite && cooldown_left_s_locked() == 0) {
                                                --g_grant.runs_left;
                                                nj->approved_suite = g_grant.suite;
                                                nj->approved_nonce = nj->nonce;
                                                nj->phase = Job::Phase::Approved;
                                            } else {
                                                pending = true;
                                            }
                                            s->job = nj;
                                        }
                                    }
                                    next_progress = Clock::now();
                                    if (pending) {
                                        last_prompt = Clock::now();
                                        prompted = true;
                                        notify(true);
                                        bool still;
                                        {
                                            std::lock_guard<std::mutex> lk(g_mu);
                                            still = !s->stop.load() && nj->phase == Job::Phase::Pending;
                                        }
                                        if (!still) notify(false);   // answered or switched off while the hook ran
                                    }
                                }
                            }
                        }
                    }
                } else if (r.status == 403 && relay_error_code(r.body) == "unknown_node") {
                    bool was_pending = false;
                    int dropped = 0;
                    {
                        std::lock_guard<std::mutex> lk(g_mu);
                        s->code.clear();   // register again (new pairing code)
                        if (s->job && s->job->phase == Job::Phase::Pending) {
                            was_pending = true;
                            withdraw_locked(*s);
                        }
                        dropped = s->dropped_prompts;
                    }
                    if (was_pending) notify(false);
                    if (dropped >= kMaxDroppedPrompts) stop_misbehaving(s, "trok te vaak een verzoek weer in");
                    next_net = Clock::now() + std::chrono::seconds(1);
                } else {
                    net_failed(status_error(r));
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

bool stop_session_locked() {
    g_grant = Grant{};
    if (!g_cur) return false;
    bool unpend = false;
    g_cur->stop = true;
    if (g_cur->job) {
        if (g_cur->job->phase == Job::Phase::Pending) {
            reject_locked(*g_cur, "disabled");
            unpend = true;
        } else {
            g_cur->job->cancel = true;
        }
    }
    g_cur.reset();
    return unpend;
}

}  // namespace

void set_host(Host host) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_host = std::move(host);
}

void set_pending_hook(std::function<void(bool)> hook) {
    std::lock_guard<std::mutex> lk(g_hook_mu);
    g_hook = std::move(hook);
}

std::string status_json() {
    std::lock_guard<std::mutex> lk(g_mu);
    load_url_locked();
    const std::shared_ptr<Session> s = g_cur;
    std::string state = "off", code, error, pending = "null", progress = "null";
    if (s) {
        code = s->code;
        error = s->error;
        const Job* j = s->job.get();
        if (j && j->phase == Job::Phase::Pending) {
            state = "pending";
            std::string steps = "[", sends = "[";
            for (size_t i = 0; i < j->suite->steps.size(); ++i) steps += (i ? "," : "") + json_quote(j->suite->steps[i].name);
            for (size_t i = 0; i < sizeof kSends / sizeof kSends[0]; ++i) sends += (i ? "," : "") + json_quote(kSends[i]);
            const auto left = std::chrono::duration_cast<std::chrono::seconds>(j->deadline - Clock::now()).count();
            pending = "{\"job_id\":" + json_quote(j->id) + ",\"nonce\":" + json_quote(j->nonce) + ",\"suite\":" + json_quote(j->suite->id) +
                      ",\"title\":" + json_quote(j->suite->title) +
                      ",\"steps\":" + steps + "],\"est_minutes\":" + std::to_string(j->suite->est_minutes) +
                      ",\"hardware\":" + json_quote(j->hardware) + ",\"sends\":" + sends + "],\"expires_in_s\":" +
                      std::to_string(left > 0 ? left : 0) + "}";
        } else if (j && (j->phase == Job::Phase::Approved || j->phase == Job::Phase::Running)) {
            state = "running";
            const tests::Progress& p = j->progress;
            progress = "{\"step\":" + std::to_string(p.step) + ",\"total\":" +
                       std::to_string(p.total ? p.total : static_cast<int>(j->suite->steps.size())) + ",\"name\":" + json_quote(p.name) +
                       ",\"message\":" + json_quote(j->cancel.load() ? std::string("bezig met annuleren") : p.message) +
                       ",\"tok_s\":" + (p.tok_s >= 0 ? lbhttp::json_number(p.tok_s) : std::string("null")) + "}";
        } else if (!error.empty()) {
            state = "error";
        } else if (code.empty()) {
            state = "connecting";
        } else {
            state = s->finished_one ? "done" : "waiting";
        }
    }
    if (error.empty()) error = g_config_error;
    const std::string last = g_last_result.empty()
                                 ? std::string("null")
                                 : g_last_result + ",\"sent\":" + (g_last_sent ? "true" : "false") +
                                       ",\"send_failed\":" + (g_last_send_failed ? "true" : "false") + "}";
    const std::string grant = grant_json_locked();
    return "{\"enabled\":" + std::string(s ? "true" : "false") + ",\"relay_url\":" + json_quote(g_relay_url) +
           ",\"code\":" + (code.empty() ? std::string("null") : json_quote(code)) + ",\"state\":" + json_quote(state) +
           ",\"error\":" + (error.empty() ? std::string("null") : json_quote(error)) +
           ",\"session_allow\":" + (grant == "null" ? "false" : "true") + ",\"grant\":" + grant + ",\"pending\":" + pending +
           ",\"progress\":" + progress + ",\"last_result\":" + last + "}";
}

int configure(const std::string& body, std::string& err) {
    Json root;
    std::string jerr;
    if (!lbhttp::parse_json(body, root, jerr) || root.kind != Json::Kind::Object) {
        err = "body must be a JSON object {\"enabled\"?: bool, \"relay_url\"?: string}";
        return 400;
    }
    const Json* en = root.get("enabled");
    const Json* ur = root.get("relay_url");
    if ((en && en->kind != Json::Kind::Bool) || (ur && ur->kind != Json::Kind::String)) {
        err = "\"enabled\" must be a boolean and \"relay_url\" a string";
        return 400;
    }
    bool unpend = false;
    std::shared_ptr<Session> started;
    {
        std::lock_guard<std::mutex> lk(g_mu);
        load_url_locked();
        std::string url = g_relay_url;
        rc::RelayUrl parsed;
        if (ur) {
            if (ur->str.empty()) url.clear();
            else if (rc::parse_relay_url(ur->str, parsed, err)) url = parsed.normalized;
            else return 400;
        }
        const bool want = en ? en->boolean : (g_cur != nullptr);
        if (want && !rc::parse_relay_url(url, parsed, err)) {
            if (url.empty()) err = "stel eerst een relay-adres in";
            return 400;
        }
        const bool url_changed = url != g_relay_url;
        g_relay_url = url;
        g_config_error.clear();
        if (ur && !save_url_locked()) g_config_error = "het relay-adres kon niet worden opgeslagen";
        if (!want || url_changed) unpend = stop_session_locked();
        if (want && !g_cur) {
            auto s = std::make_shared<Session>();
            s->url = parsed;
            s->token = rc::random_hex(32);
            if (s->token.empty()) {
                err = "geen veilige willekeurige getallen beschikbaar";
                return 500;
            }
            g_cur = s;
            started = s;
        }
    }
    if (unpend) notify(false);
    if (started) {
        try {
            std::thread(worker, started).detach();
        } catch (const std::exception&) {
            std::lock_guard<std::mutex> lk(g_mu);
            if (g_cur == started) g_cur.reset();
            err = "kon de verbindingsthread niet starten";
            return 500;
        }
    }
    return 200;
}

int decide(const std::string& body, std::string& err) {
    Json root;
    std::string jerr;
    if (!lbhttp::parse_json(body, root, jerr) || root.kind != Json::Kind::Object) {
        err = "body must be a JSON object {\"job_id\", \"nonce\", \"suite\", \"decision\": \"yes\"|\"yes_session\"|\"no\"}";
        return 400;
    }
    const std::string id = rc::str_field(root, "job_id", 80), d = rc::str_field(root, "decision", 16);
    const std::string nonce = rc::str_field(root, "nonce", 64), suite = rc::str_field(root, "suite", 40);
    if (d != "yes" && d != "yes_session" && d != "no") {
        err = "\"decision\" must be \"yes\", \"yes_session\" or \"no\"";
        return 400;
    }
    if (id.empty() || nonce.empty() || suite.empty()) {
        err = "\"job_id\", \"nonce\" and \"suite\" of the pending request are required (GET /api/remote/status)";
        return 400;
    }
    {
        std::lock_guard<std::mutex> lk(g_mu);
        // the answer must describe the request that is waiting now, exactly as it was shown
        if (!g_cur || !g_cur->job || g_cur->job->phase != Job::Phase::Pending || g_cur->job->id != id ||
            !rc::ct_equal(g_cur->job->nonce, nonce) || suite != g_cur->job->suite->id) {
            err = "dit antwoord hoort niet bij het wachtende verzoek; bekijk het verzoek opnieuw";
            return 409;
        }
        if (Clock::now() >= g_cur->job->deadline || Wall::now() >= g_cur->job->wall_deadline) {
            err = "dit verzoek is verlopen";
            return 409;
        }
        if (d == "no") {
            reject_locked(*g_cur, "user");
        } else {
            if (d == "yes_session")
                g_grant = Grant{g_cur->job->suite, kGrantRuns, local_date(), Clock::now() + kGrantTtl, Wall::now() + kGrantTtl};
            g_cur->job->approved_suite = g_cur->job->suite;
            g_cur->job->approved_nonce = g_cur->job->nonce;
            g_cur->job->phase = Job::Phase::Approved;
        }
    }
    notify(false);
    return 200;
}

int cancel(std::string& err) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (!g_cur || !g_cur->job || (g_cur->job->phase != Job::Phase::Running && g_cur->job->phase != Job::Phase::Approved)) {
        err = "er draait geen test";
        return 409;
    }
    g_cur->job->cancel = true;
    return 200;
}

}  // namespace veltra::remote
