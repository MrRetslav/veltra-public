// engine/server.cpp - OpenAI-compatible HTTP/1.1 server on top of lb::Engine (no third-party code).
//
//   veltra-server [--model x.vlt] [--models-dir DIR] [--host 127.0.0.1] [--port 8080] [--threads N] [--no-spec] [--no-browser]
//   veltra.exe    (Windows desktop app, engine/win_app.cpp: WebView2 window on an ephemeral loopback port; --headless = this server)
//             [--ui hub|desktop] [--require-token TOKEN]
//             [--int8-head] [--exact-head [0|1]] [--head-vocab N] [--head-vocab-file PATH] [--prefix-cache [0|1]]
//
//   GET  /desktop             the embedded desktop UI (desktop.html; also GET / with --ui desktop)
//   GET  /health              liveness and counters; answers while a generation is running
//   GET  /v1/models           the loaded model
//   POST /v1/chat/completions JSON response, or text/event-stream when "stream": true
//   OPTIONS *                 CORS preflight
//   GET  /api/benchmarks      the --bench-json file (404 when missing)
//   GET  /benchmarks          <--web-dir>/benchmarks.html
//   GET  /arena, /chat        <--web-dir>/arena.html, <--web-dir>/chat.html
//   POST /compare/llama/v1/chat/completions  forwarded to --compare-upstream (loopback only; 404 when not configured)
//   GET  /, /index.html       Veltra launcher / model hub (embedded UI; <--web-dir>/index.html overrides)
//   GET  /api/hardware        CPU / RAM / GPU + VRAM of this machine (engine/hw_detect.cpp)
//   GET  /api/catalog         built-in .vlt catalog with a per-machine fit status + the local models folder
//   GET  /api/status          loaded model, device, load progress
//   POST /api/models/load     {"file": name in --models-dir} or {"path": absolute .vlt}: (re)load in the background   (loopback/admin)
//   POST /api/models/download {"id": catalog id} or {"url","file"}: background download into --models-dir             (loopback/admin)
//   GET  /api/models/download/status
//   POST /api/benchmarks/run  in-process quick benchmark (loopback peer or X-Admin-Token / Bearer == env VELTRA_ADMIN_TOKEN / LB_ADMIN_TOKEN);
//                             updates "quick" in the --bench-json file; 409 while one is running
//   POST /api/benchmarks/local  decode tok/s, prefill tok/s + TTFT, peak RSS on this machine (loopback/admin; 409 busy,
//                             503 no model); appended to <app data>/local_benchmarks.json (last 50)
//   GET  /api/benchmarks/local  that history
//   GET  /api/compute         compute selection + live node health (GET host:port/health, 1.5 s), effective device, layer_plan
//   POST /api/compute/select  {"mode":"gpu"|"cpu"|"cluster","gpu_indices":[..],"nodes":["host:port",..]} (loopback/admin,
//                             <= 8 nodes); persisted to <app data>/compute.json. Cluster mode forwards POST /v1/chat/completions
//                             per request to an online node (round-robin); layer splitting is a plan only, not executed
//   GET  /api/chats           stored chats without messages (loopback/admin; <app data>/chats.json)
//   POST /api/chats           upsert {id?, title?, messages:[{role, content, meta?}]} (2 MiB body, 500 chats, 1000 messages)
//   GET  /api/chats/{id}      one chat with messages;   DELETE /api/chats/{id}
//   <app data> = %APPDATA%/Veltra (Windows), $XDG_DATA_HOME/veltra or ~/.local/share/veltra; env VELTRA_DATA_DIR overrides
//   GET  /api/remote/status   opt-in remote GPU test mode (engine/remote_client.cpp, docs/REMOTE_TESTS.md); all /api/remote/*
//   POST /api/remote/config   routes need the desktop token or a local client (loopback + own origin) / admin token.
//   POST /api/remote/decision {"enabled"?,"relay_url"?}; {"job_id","nonce","suite","decision":"yes"|"yes_session"|"no"}; the toggle is never
//   POST /api/remote/cancel   persisted and nothing runs before the local user approved it
//   /api/relay/*              only with --relay (adds the relay to this server, local use) or --relay-only (nothing but
//                             /api/relay/* and GET /health through the hardened reader in engine/relay.cpp; no model; may sit
//                             behind a public HTTPS reverse proxy). --relay-results DIR (default ./remote_gpu_results).
//                             register/poll/progress/result: pairing code + X-Veltra-Node-Token; dispatch/job/results:
//                             env VELTRA_ADMIN_TOKEN (--relay also accepts a direct local client)
//
// Concurrency: one thread per connection, one request per connection (Connection: close). The engine
// owns a single KV-cache, so generations run under one mutex. A request waiting for that mutex keeps its
// socket open and is dropped without generating if its client goes away meanwhile.
//
// Cancellation: every token callback checks the socket. A failed send() or a FIN/RST seen by poll()
// makes the callback return false, which stops the engine. A matched stop string is removed from the
// output and also stops generation.

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "backend.h"
#include "cluster.h"
#include "engine.h"
#include "http.h"
#include <filesystem>

#include "hf_convert.h"
#include "json.h"
#include "hub.h"
#include "platform.h"
#include "rest_index.h"
#include "speculative.h"
#include "lb_common.h"
#include "relay.h"
#include "remote_client.h"
#include "test_runner.h"
#include "server_main.h"
#if defined(VELTRA_EMBED_UI)
#include "embedded_ui.h"   // generated by cmake/embed_ui.cmake from chat/*.html + chat/benchmarks_live.json
#endif

namespace {

using lbhttp::Json;
using lbhttp::sock_t;
using Clock = std::chrono::steady_clock;

constexpr int kMaxConnections = 64;
constexpr int kMaxStopStrings = 8;
constexpr int kMaxThreads = 256;
constexpr int kMaxTokensCap = 2147483647;
constexpr int kDefaultMaxTokens = 256;
constexpr int kTelemetryEvery = 8;   // x_telemetry chunk cadence in streams
constexpr int kQueuePollMs = 100;    // how often a queued request checks whether its client left

struct Options {
    std::string model_path;   // optional: without --model the server starts empty and the hub UI loads one
    std::string host = "127.0.0.1";
#if defined(VELTRA_STANDALONE)
    int port = 8099;          // veltra.exe --headless (the desktop window starts its own server on an ephemeral port)
    bool open_browser = false;   // --open-browser opens http://127.0.0.1:PORT/ once listening
#else
    int port = 8080;
    bool open_browser = false;
#endif
    std::string convert_hf;       // --convert-hf DIR OUT.vlt: convert a downloaded dense Qwen3 checkpoint, then exit
    std::string convert_out;
    bool cuda_selftest = false;   // --cuda-selftest: check every CUDA kernel against the CPU reference, then exit
    std::string models_dir;   // --models-dir DIR (default: <exe dir>/models for veltra.exe, ./models otherwise)
    int threads = 0;        // compute threads per generation; 0 = engine default
    int ctx = 0;            // --ctx N: context to allocate; 0 = the UI setting ("ctx" in /api/ui-state), else the engine default
    bool no_spec = false;   // default for requests that do not send "speculative"
    bool rest = false;      // --rest [idx]: static REST draft index + calibrated gate (greedy requests only, opt-in)
    std::string rest_path = "rest.idx";
    int rest_k = 2;         // --rest-k N (byte accountant caps it at 2 = T <= 3 unless VELTRA_/LB_REST_MAXK)
    float rest_min_gain = 1.30f;   // --rest-min-gain X
    bool mtp = false;       // --mtp [0|1]: native MTP drafts for greedy requests (models with an MTP block; request field "mtp" overrides)
    int search_n = 1;       // --search-n N: default test-time search width (request field "search_n" overrides; 1 = off)
    int lookup_n = 0;       // --lookup-n K: prompt-lookup (n-gram) drafting with up to K tokens per verify pass (0 = engine default)
    int prefill_chunk = 0;  // --prefill-chunk N: tokens per prefill forward (0 = default 64)
    bool pareto_max = false;
    bool screen_given = false, moe_k_given = false;   // explicit flags win over --pareto-max
    lb::EngineOptions eng;  // --int8-head / --exact-head / --head-vocab / --head-vocab-file (+ threads)
    std::string bench_json; // --bench-json PATH: served at GET /api/benchmarks, "quick" updated by POST /api/benchmarks/run
    std::string web_dir;    // --web-dir DIR: GET /benchmarks, /arena, /chat serve DIR/{benchmarks,arena,chat}.html (no other files)
    std::string compare_host;   // --compare-upstream http://127.0.0.1:PORT: POST /compare/llama/v1/chat/completions proxy
    int compare_port = 0;       // 0 = proxy disabled
    bool ui_desktop = false;    // --ui desktop: GET / serves the embedded desktop.html instead of the hub index.html
    std::string require_token;  // --require-token TOKEN: every /api/*, /v1/* and /compare/* request needs X-Veltra-Token
    bool relay = false;         // --relay: also serve /api/relay/* (local use; never expose such a server)
    bool relay_only = false;    // --relay-only: serve nothing but the relay (the mode that may be exposed behind HTTPS)
    std::string relay_results = "remote_gpu_results";   // --relay-results DIR
};

// --require-token (set once before listening): when non-empty even loopback peers must send X-Veltra-Token == it.
std::string g_require_token;
struct Server;
std::atomic<Server*> g_server{nullptr};   // for veltra_request_unload() (the desktop app's tray menu)

struct Server {
    Options opt;
    // engine / model_id / isa / max_ctx / model_path change only while gen_mu AND info_mu are held (model (re)load);
    // generations read them under gen_mu, the status routes take a snapshot under info_mu.
    std::unique_ptr<lb::Engine> engine;
    std::string model_id;
    std::string isa;
    std::string model_path;
    int max_ctx = 0;  // 0 = no context check
    mutable std::mutex info_mu;
    std::atomic<bool> loading{false};
    std::string load_error;      // last failed (re)load, under info_mu
    std::string loading_path;    // under info_mu
    std::string parked_path;     // under info_mu: model released by unload / idle timer; reloaded by the next request
    std::atomic<uint64_t> load_done{0}, load_total{0};   // bytes of the model file read into the page cache
    std::atomic<int> load_phase{0};                      // 0 idle, 1 reading the file, 2 building the engine
    std::atomic<int64_t> load_t0_ms{0}, load_read_ms{0}, load_all_ms{0};
    uint64_t model_bytes = 0, kv_bytes = 0;              // under info_mu: file size and KV cache of the loaded model
    std::atomic<int> idle_unload_min{0};     // 0 = off (setting "idle_unload_min" of /api/ui-state)
    std::atomic<int64_t> last_used_ms{0};    // steady clock: last model load or finished generation
    int64_t started_epoch = 0;
    Clock::time_point started = Clock::now();

    std::timed_mutex gen_mu;  // one generation at a time
    std::atomic<int> connections{0};
    std::atomic<int> queued{0};
    std::atomic<bool> busy{false};
    std::atomic<uint64_t> requests{0};
    std::atomic<uint64_t> gens_completed{0};
    std::atomic<uint64_t> gens_cancelled{0};
    std::atomic<uint64_t> tokens_out{0};
    std::atomic<uint64_t> id_counter{0};
    std::atomic<bool> bench_running{false};  // POST /api/benchmarks/run: one at a time (409 otherwise)
    std::mutex bench_file_mu;                // serialises read-modify-write of --bench-json
};

void log(const std::string& msg) { std::fprintf(stderr, "[veltra-server] %s\n", msg.c_str()); }

// Aborted HTTP connections also go to <app data>/veltra.log (the Windows GUI exe has no console): one timestamped line each, rotated at 1 MB.
void net_log(const std::string& msg) {
    log(msg);
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    try {
        const std::string dir = veltra::plat::app_data_dir();
        if (dir.empty()) return;
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        const std::string path = dir + "/veltra.log";
        const auto size = std::filesystem::file_size(path, ec);
        if (!ec && size > (1u << 20)) std::filesystem::rename(path, path + ".1", ec);
        if (std::FILE* f = std::fopen(path.c_str(), "ab")) {
            char ts[32] = "";
            const std::time_t now = std::time(nullptr);
            if (const std::tm* t = std::localtime(&now)) std::strftime(ts, sizeof ts, "%Y-%m-%d %H:%M:%S", t);
            std::fprintf(f, "%s %s\n", ts, msg.c_str());
            std::fclose(f);
        }
    } catch (...) {   // logging must never break a request
    }
}
void clear_socket_error() {
#if defined(_WIN32)
    WSASetLastError(0);
#else
    errno = 0;
#endif
}
bool has_socket_error() {
#if defined(_WIN32)
    return WSAGetLastError() != 0;
#else
    return errno != 0;
#endif
}

std::string q(std::string_view s) { return lbhttp::json_quote(s); }
std::string num(double v) { return lbhttp::json_number(v); }
std::string itos(long long v) { return std::to_string(v); }
int64_t steady_ms();                    // defined next to the model loading code
void reload_parked(Server& srv);
void preset_context();

// ---- errors ------------------------------------------------------------------------------------

struct ApiError {
    int status;
    std::string type;     // "invalid_request_error", "not_found_error", "server_error"
    std::string message;
    std::string code;     // empty => null
};

ApiError bad_request(std::string message, std::string code = {}) {
    return ApiError{400, "invalid_request_error", std::move(message), std::move(code)};
}

std::string error_json(const ApiError& e) {
    std::string out = "{\"error\":{\"message\":" + q(e.message) + ",\"type\":" + q(e.type) + ",\"param\":null,\"code\":";
    out += e.code.empty() ? std::string("null") : q(e.code);
    out += "}}";
    return out;
}

void send_error(sock_t c, const ApiError& e) {
    lbhttp::send_response(c, e.status, "application/json", error_json(e));
}

// ---- request body ------------------------------------------------------------------------------

struct ChatRequest {
    std::vector<lb::ChatMessage> messages;
    lb::GenOptions gen;
    std::vector<std::string> stops;
    bool stream = false;
    bool raw = false;         // /v1/completions style: plain text prompt, no chat template
    std::string raw_prompt;
};

const Json* present(const Json& o, std::string_view key) {
    const Json* v = o.get(key);
    return (v && v->kind != Json::Kind::Null) ? v : nullptr;  // JSON null means "not set"
}

std::optional<int64_t> opt_int(const Json& o, std::string_view key, int64_t lo, int64_t hi) {
    const Json* v = present(o, key);
    if (!v) return std::nullopt;
    int64_t x = 0;
    if (!v->as_int(x) || x < lo || x > hi)
        throw bad_request("'" + std::string(key) + "' must be an integer in [" + itos(lo) + ", " + itos(hi) + "]");
    return x;
}

std::optional<double> opt_num(const Json& o, std::string_view key, double lo, double hi) {
    const Json* v = present(o, key);
    if (!v) return std::nullopt;
    if (v->kind != Json::Kind::Number || !std::isfinite(v->number) || v->number < lo || v->number > hi)
        throw bad_request("'" + std::string(key) + "' must be a number in [" + num(lo) + ", " + num(hi) + "]");
    return v->number;
}

std::optional<bool> opt_bool(const Json& o, std::string_view key) {
    const Json* v = present(o, key);
    if (!v) return std::nullopt;
    if (v->kind != Json::Kind::Bool) throw bad_request("'" + std::string(key) + "' must be a boolean");
    return v->boolean;
}

// Message content: a string, null, or an array of {"type":"text","text":...} parts.
std::string message_text(const Json& m) {
    const Json* c = present(m, "content");
    if (!c) return {};
    if (c->kind == Json::Kind::String) return c->str;
    if (c->kind == Json::Kind::Array) {
        std::string text;
        for (const Json& part : c->items) {
            const Json* type = part.get("type");
            const Json* t = part.get("text");
            if (part.kind != Json::Kind::Object || !type || type->kind != Json::Kind::String ||
                type->str != "text" || !t || t->kind != Json::Kind::String)
                throw bad_request("only text content parts are supported");
            text += t->str;
        }
        return text;
    }
    throw bad_request("'content' must be a string or an array of text parts");
}

ChatRequest parse_chat(const Json& root, const Server& srv) {
    if (root.kind != Json::Kind::Object) throw bad_request("the request body must be a JSON object");

    ChatRequest cr;
    if (const Json* m = present(root, "model")) {
        if (m->kind != Json::Kind::String) throw bad_request("'model' must be a string");
        std::string loaded_id;
        {
            std::lock_guard<std::mutex> lk(srv.info_mu);
            loaded_id = srv.model_id;
        }
        if (!loaded_id.empty() && m->str != loaded_id)
            throw ApiError{404, "not_found_error",
                           "model '" + m->str + "' is not served here; the loaded model is '" + loaded_id + "'",
                           "model_not_found"};
    }

    const Json* raw_prompt = present(root, "prompt");
    if (raw_prompt) {  // plain completion: the model just continues this text (no chat template)
        if (raw_prompt->kind != Json::Kind::String || raw_prompt->str.empty())
            throw bad_request("'prompt' must be a non-empty string");
        cr.raw = true;
        cr.raw_prompt = raw_prompt->str;
    } else {
        const Json* msgs = present(root, "messages");
        if (!msgs || msgs->kind != Json::Kind::Array || msgs->items.empty())
            throw bad_request("'messages' must be a non-empty array (or 'prompt' a string)");
        for (const Json& m : msgs->items) {
            if (m.kind != Json::Kind::Object) throw bad_request("each entry of 'messages' must be an object");
            const Json* role = present(m, "role");
            if (!role || role->kind != Json::Kind::String || role->str.empty())
                throw bad_request("each message needs a non-empty string 'role'");
            cr.messages.push_back(lb::ChatMessage{role->str, message_text(m)});
        }
    }

    (void)opt_int(root, "n", 1, 1);  // only one completion per request is supported

    auto max_tokens = opt_int(root, "max_tokens", 1, kMaxTokensCap);
    if (!max_tokens) max_tokens = opt_int(root, "max_completion_tokens", 1, kMaxTokensCap);
    cr.gen.max_new_tokens = max_tokens ? static_cast<int>(*max_tokens) : kDefaultMaxTokens;

    if (auto t = opt_num(root, "temperature", 0.0, 2.0)) cr.gen.sampling.temperature = static_cast<float>(*t);
    if (auto p = opt_num(root, "top_p", 0.0, 1.0)) cr.gen.sampling.top_p = static_cast<float>(*p);
    if (auto k = opt_int(root, "top_k", 0, 1000000)) cr.gen.sampling.top_k = static_cast<int>(*k);
    if (auto s = opt_int(root, "seed", INT64_MIN, INT64_MAX)) cr.gen.sampling.seed = static_cast<uint64_t>(*s);

    // Extension fields. Defaults come from the command line.
    cr.gen.spec.enabled = opt_bool(root, "speculative").value_or(!srv.opt.no_spec);
    cr.gen.spec.rest = opt_bool(root, "rest").value_or(srv.opt.rest);   // REST self-disables for temperature > 0 / top_p < 1 / top_k > 0 (SpecDecoder)
    cr.gen.spec.rest_k = static_cast<int>(opt_int(root, "rest_k", 1, 8).value_or(srv.opt.rest_k));
    cr.gen.spec.rest_min_gain = srv.opt.rest_min_gain;
    cr.gen.spec.mtp = opt_bool(root, "mtp").value_or(srv.opt.mtp);   // MTP self-disables for sampling requests and models without an MTP block (SpecDecoder)
    cr.gen.n_threads = static_cast<int>(opt_int(root, "n_threads", 0, kMaxThreads).value_or(srv.opt.threads));
    if (srv.opt.lookup_n > 0) {
        cr.gen.spec.ngram = true;
        cr.gen.spec.k_max = srv.opt.lookup_n;
        cr.gen.spec.k_min = std::min(cr.gen.spec.k_min, srv.opt.lookup_n);
        cr.gen.spec.k_init = std::min(cr.gen.spec.k_init, srv.opt.lookup_n);
    }

    if (const Json* s = present(root, "stop")) {
        auto add = [&cr](const Json& item) {
            if (item.kind != Json::Kind::String) throw bad_request("'stop' must be a string or an array of strings");
            if (!item.str.empty()) cr.stops.push_back(item.str);
        };
        if (s->kind == Json::Kind::Array) {
            for (const Json& item : s->items) add(item);
        } else {
            add(*s);
        }
        if (cr.stops.size() > kMaxStopStrings)
            throw bad_request("at most " + itos(kMaxStopStrings) + " stop strings are allowed");
    }

    cr.stream = opt_bool(root, "stream").value_or(false);
    cr.gen.search.n = static_cast<int>(opt_int(root, "search_n", 1, 16).value_or(srv.opt.search_n));
    if (auto t = opt_num(root, "search_temperature", 0.0, 2.0)) cr.gen.search.temperature = static_cast<float>(*t);
    if (auto p = opt_num(root, "search_top_p", 0.0, 1.0)) cr.gen.search.top_p = static_cast<float>(*p);
    if (cr.gen.sampling.seed) cr.gen.search.base_seed = cr.gen.sampling.seed;
    cr.gen.search.stop_strings = cr.stops;
    return cr;
}

// What the request itself demands of the answer (the only thing the search verifier may know about it): response_format, or a REQUIRED tool call
// (tool_choice "required" or a named function; with "auto" a plain answer is allowed, so nothing is gated).
void parse_verify_spec(const std::string& body, lb::VerifySpec& spec) {
    lb::json::Value v;
    std::string e;
    if (!lb::json::parse(body, &v, &e)) return;
    const std::string rf = v["response_format"]["type"].as_string("");
    if (rf == "json_object") spec.format = lb::OutputFormat::JsonObject;
    if (rf == "json_schema") {
        spec.format = lb::OutputFormat::JsonSchema;
        const lb::json::Value& js = v["response_format"]["json_schema"];
        spec.schema_json = (js.find("schema") ? js["schema"] : js).dump();
    }
    const lb::json::Value& choice = v["tool_choice"];
    const std::string named = choice["function"]["name"].as_string("");
    if (choice.as_string("") == "required" || !named.empty()) {
        spec.format = lb::OutputFormat::ToolCall;
        if (!named.empty()) spec.tool_names.push_back(named);
        else for (const auto& t : v["tools"].as_array()) {
            const std::string n = (t.find("function") ? t["function"] : t)["name"].as_string("");
            if (!n.empty()) spec.tool_names.push_back(n);
        }
    }
}

// ---- JSON responses ----------------------------------------------------------------------------

std::string make_id(Server& s) {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
    char buf[80];
    std::snprintf(buf, sizeof buf, "chatcmpl-%llx%llx", static_cast<unsigned long long>(ns),
                  static_cast<unsigned long long>(s.id_counter++));
    return buf;
}

std::string usage_json(int prompt, int completion) {
    return "{\"prompt_tokens\":" + itos(prompt) + ",\"completion_tokens\":" + itos(completion) +
           ",\"total_tokens\":" + itos(prompt + completion) + "}";
}

std::string timings_json(const lb::GenStats& st, int prompt, int completion, const std::string& isa) {
    const double accept = st.drafted > 0 ? static_cast<double>(st.accepted) / st.drafted : 0.0;
    return "{\"prompt_tokens\":" + itos(prompt) + ",\"completion_tokens\":" + itos(completion) +
           ",\"tokens_per_second\":" + num(st.tokens_per_s) + ",\"prefill_s\":" + num(st.prefill_s) +
           ",\"decode_s\":" + num(st.decode_s) + ",\"drafted\":" + itos(st.drafted) +
           ",\"accepted\":" + itos(st.accepted) + ",\"acceptance_rate\":" + num(accept) +
           ",\"cached_tokens\":" + itos(st.cached_tokens) + ",\"head_screened\":" + itos(st.head_screened) +
           ",\"head_full\":" + itos(st.head_full) + ",\"rest_lookups\":" + itos(st.rest_lookups) + ",\"rest_hits\":" + itos(st.rest_hits) +
           ",\"rest_passes\":" + itos(st.rest_passes) + ",\"rest_drafted\":" + itos(st.rest_drafted) + ",\"rest_accepted\":" + itos(st.rest_accepted) +
           ",\"rest_draft_s\":" + num(st.rest_draft_s) + ",\"verify_passes\":" + itos(st.verify_passes) +
           ",\"mtp_drafted\":" + itos(st.mtp_drafted[0] + st.mtp_drafted[1]) + ",\"mtp_accepted\":" + itos(st.mtp_accepted[0] + st.mtp_accepted[1]) +
           (st.search_branches > 0 ? ",\"search\":{\"branches_evaluated\":" + itos(st.search_branches) + ",\"valid_branches\":" + itos(st.search_valid) +
                                         ",\"pruned_branches\":" + itos(st.search_pruned) + ",\"winner\":" + itos(st.search_winner) +
                                         ",\"consensus_ratio\":" + num(st.search_consensus) + ",\"total_tokens\":" + itos(st.search_total_tokens) +
                                         ",\"steps\":" + itos(st.search_steps) + ",\"branches\":" + st.search_detail + "}"
                                   : std::string()) +
           ",\"isa\":" + q(isa) + "}";
}

std::string sse(const std::string& json) { return "data: " + json + "\n\n"; }

bool is_continuation(char c) { return (static_cast<unsigned char>(c) & 0xC0) == 0x80; }

// Holds back the tail of the generated text that could still turn into a stop string, so stop strings
// never reach the client. Text before a match is returned; after a match nothing more is returned.
class StopFilter {
public:
    struct Step {
        std::string text;
        bool stop = false;
    };

    explicit StopFilter(std::vector<std::string> stops) : stops_(std::move(stops)) {
        for (const auto& s : stops_) hold_ = std::max(hold_, s.size() - 1);
    }

    Step push(std::string_view piece) {
        Step out;
        pending_.append(piece);
        size_t hit = std::string::npos;
        for (const auto& s : stops_) {
            const size_t p = pending_.find(s);
            if (p != std::string::npos && p < hit) hit = p;
        }
        if (hit != std::string::npos) {
            out.text = pending_.substr(0, hit);
            out.stop = true;
            pending_.clear();
            return out;
        }
        if (pending_.size() > hold_) {
            size_t cut = pending_.size() - hold_;
            while (cut > 0 && is_continuation(pending_[cut])) --cut;  // never split a UTF-8 character
            out.text = pending_.substr(0, cut);
            pending_.erase(0, cut);
        }
        return out;
    }

    std::string flush() { return std::exchange(pending_, std::string()); }

private:
    std::vector<std::string> stops_;
    size_t hold_ = 0;
    std::string pending_;
};

// One chat completion's output: either buffered (JSON) or framed as SSE chunks. It is the engine's
// token callback target, so every token passes through on_token().
class Reply {
public:
    Reply(Server& srv, sock_t c, bool stream, std::vector<std::string> stops)
        : c_(c),
          stream_(stream),
          model_(srv.model_id),
          id_(make_id(srv)),
          created_(static_cast<int64_t>(std::time(nullptr))),
          filter_(std::move(stops)) {}

    // Engine callback. Returns false to cancel generation (client gone, or a stop string matched).
    bool on_token(std::string_view piece) {
        clear_socket_error();
        if (lbhttp::peer_gone(c_)) return gone();
        if (!have_first_) {
            first_ = Clock::now();
            have_first_ = true;
        }
        ++tokens_;
        const StopFilter::Step step = filter_.push(piece);
        if (!emit(step.text)) return gone();
        if (step.stop) {
            stop_hit_ = true;
            return false;
        }
        if (stream_ && tokens_ % kTelemetryEvery == 0 && !send_telemetry()) return gone();
        return true;
    }

    // Opens an SSE response and sends the role chunk. False if the client is gone.
    bool start_stream() {
        const bool ok = lbhttp::send_sse_head(c_) &&
                        lbhttp::send_chunk(c_, sse(chunk_head() + ",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}"));
        return ok ? true : gone();
    }

    // Releases text held back for stop-string matching once generation ends normally.
    bool finish_text() { return emit(filter_.flush()); }

    // Ends an SSE response: finish chunk, usage chunk, [DONE], terminating chunk.
    bool finish_stream(const std::string& finish, const std::string& usage, const std::string& timings) {
        const std::string head = chunk_head();
        return lbhttp::send_chunk(c_, sse(head + ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":" +
                                          q(finish) + "}]}")) &&
               lbhttp::send_chunk(c_, sse(head + ",\"choices\":[],\"usage\":" + usage + ",\"timings\":" + timings + "}")) &&
               lbhttp::send_chunk(c_, "data: [DONE]\n\n") && lbhttp::send_last_chunk(c_);
    }

    // Reports an engine failure inside an already-open SSE response.
    void fail_stream(const ApiError& e) {
        lbhttp::send_chunk(c_, sse(error_json(e)));
        lbhttp::send_chunk(c_, "data: [DONE]\n\n");
        lbhttp::send_last_chunk(c_);
    }

    std::string completion_json(const std::string& finish, const std::string& usage, const std::string& timings) const {
        return "{\"id\":" + q(id_) + ",\"object\":\"chat.completion\",\"created\":" + itos(created_) +
               ",\"model\":" + q(model_) +
               ",\"choices\":[{\"index\":0,\"text\":" + q(text_) + ",\"message\":{\"role\":\"assistant\",\"content\":" + q(text_) +
               "},\"finish_reason\":" + q(finish) + "}],\"usage\":" + usage + ",\"timings\":" + timings + "}";
    }

    bool client_gone() const { return client_gone_; }
    const std::string& gone_why() const { return gone_why_; }
    bool stop_hit() const { return stop_hit_; }
    int tokens() const { return tokens_; }

private:
    bool gone() {
        if (!client_gone_) gone_why_ = has_socket_error() ? lbhttp::last_socket_error() : "peer closed the connection";
        client_gone_ = true;
        return false;
    }
    std::string gone_why_;

    std::string chunk_head() const {
        return "{\"id\":" + q(id_) + ",\"object\":\"chat.completion.chunk\",\"created\":" + itos(created_) +
               ",\"model\":" + q(model_);
    }

    bool emit(std::string_view text) {
        if (text.empty()) return true;
        if (!stream_) {
            text_.append(text);
            return true;
        }
        return lbhttp::send_chunk(c_, sse(chunk_head() + ",\"choices\":[{\"index\":0,\"text\":" + q(text) +
                                          ",\"delta\":{\"content\":" + q(text) + "},\"finish_reason\":null}]}"));
    }

    // Live progress: "x_telemetry" on an empty-delta chunk, every kTelemetryEvery tokens.
    bool send_telemetry() {
        const double elapsed = std::chrono::duration<double>(Clock::now() - first_).count();
        const double tps = (tokens_ > 1 && elapsed > 0) ? (tokens_ - 1) / elapsed : 0.0;
        return lbhttp::send_chunk(
            c_, sse(chunk_head() + ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":null}],\"x_telemetry\":{"
                                   "\"completion_tokens\":" +
                        itos(tokens_) + ",\"tokens_per_second\":" + num(tps) + ",\"elapsed_s\":" + num(elapsed) + "}}"));
    }

    sock_t c_;
    bool stream_;
    std::string model_;
    std::string id_;
    int64_t created_;
    StopFilter filter_;
    std::string text_;  // JSON mode accumulates the completion here
    int tokens_ = 0;
    bool have_first_ = false;
    Clock::time_point first_{};
    bool client_gone_ = false;
    bool stop_hit_ = false;
};

// Runs one generation (the caller holds the engine lock) and writes the reply.
void generate_and_reply(Server& srv, sock_t c, const ChatRequest& cr, const std::vector<int>& prompt) {
    const int prompt_tokens = static_cast<int>(prompt.size());
    Reply reply(srv, c, cr.stream, cr.stops);
    if (cr.stream && !reply.start_stream()) {
        srv.gens_cancelled++;
        log("client left before the first token");
        return;
    }

    lb::GenStats st;
    bool failed = false;
    std::string failure;
    lb::GenOptions gen = cr.gen;
    if (gen.search.n > 1) gen.search.keep_going = [c]() { return !lbhttp::peer_gone(c); };
    try {
        st = srv.engine->generate(prompt, gen,
                                  [&reply](int /*token*/, const std::string& piece) { return reply.on_token(piece); });
    } catch (const std::exception& e) {
        failed = true;
        failure = e.what();
    }
    srv.tokens_out += static_cast<uint64_t>(reply.tokens());

    if (failed) {
        log("generation failed: " + failure);
        const ApiError err{500, "server_error", "engine error: " + failure, ""};
        if (cr.stream) reply.fail_stream(err);
        else send_error(c, err);
        return;
    }
    if (reply.client_gone()) {
        srv.gens_cancelled++;
        net_log("client disconnected after " + itos(reply.tokens()) + " tokens; generation cancelled (" + reply.gone_why() + ")");
        return;
    }
    if (st.finish_reason == "cancelled" && gen.search.n > 1) {
        srv.gens_cancelled++;
        net_log("client disconnected during test-time search; all branches cancelled");
        return;
    }
    clear_socket_error();
    if (!reply.stop_hit() && !reply.finish_text()) {
        srv.gens_cancelled++;
        net_log("sending the last text failed after " + itos(reply.tokens()) + " tokens: " + lbhttp::last_socket_error());
        return;
    }

    const std::string finish = (!reply.stop_hit() && st.finish_reason == "length") ? "length" : "stop";
    const int completion = st.generated_tokens > 0 ? st.generated_tokens : reply.tokens();
    const std::string usage = usage_json(prompt_tokens, completion);
    const std::string timings = timings_json(st, prompt_tokens, completion, srv.isa);

    const bool ok = cr.stream
                        ? reply.finish_stream(finish, usage, timings)
                        : lbhttp::send_response(c, 200, "application/json", reply.completion_json(finish, usage, timings));
    if (ok) srv.gens_completed++;
    else {
        srv.gens_cancelled++;
        net_log(std::string("sending the ") + (cr.stream ? "stream end" : "reply") + " failed after " + itos(completion) + " tokens: " + lbhttp::last_socket_error());
    }
    log("chat " + std::string(cr.stream ? "stream" : "json") + ": " + itos(prompt_tokens) + " prompt / " +
        itos(completion) + " completion tokens, finish=" + finish + ", " + num(st.tokens_per_s) + " tok/s");
}

struct BusyFlag {
    std::atomic<bool>& flag;
    ~BusyFlag() { flag = false; }
};

bool forward_chat_to_cluster(Server& srv, sock_t c, const lbhttp::Request& req);   // cluster mode (defined below)

void handle_chat(Server& srv, sock_t c, const lbhttp::Request& req) {
    if (forward_chat_to_cluster(srv, c, req)) return;
    ChatRequest cr;
    try {
        Json root;
        std::string jerr;
        if (!lbhttp::parse_json(req.body, root, jerr)) throw bad_request("invalid JSON: " + jerr, "invalid_json");
        cr = parse_chat(root, srv);
        if (cr.gen.search.n > 1) parse_verify_spec(req.body, cr.gen.search.verify);
    } catch (const ApiError& e) {
        send_error(c, e);
        return;
    }

    // Wait for the engine. A client that leaves while queued is dropped without generating.
    srv.queued++;
    std::unique_lock<std::timed_mutex> lock(srv.gen_mu, std::defer_lock);
    const auto slice = std::chrono::milliseconds(kQueuePollMs);
    bool locked = lock.try_lock_for(slice);
    while (!locked && !lbhttp::peer_gone(c)) locked = lock.try_lock_for(slice);
    srv.queued--;
    if (!locked) {
        log("client left while queued; request dropped");
        return;
    }

    reload_parked(srv);
    if (!srv.engine) {
        send_error(c, ApiError{503, "server_error", srv.loading.load() ? "a model is being loaded; try again shortly"
                                                                         : "no model loaded: pick one on the start page (GET /)", "model_not_loaded"});
        return;
    }
    srv.busy = true;
    BusyFlag busy_flag{srv.busy};
    struct Touch { Server& s; ~Touch() { s.last_used_ms = steady_ms(); } } touch{srv};
    touch.s.last_used_ms = steady_ms();

    std::vector<int> prompt;
    try {
        srv.engine->reset();  // every request carries its whole history; no KV-cache reuse
        prompt = cr.raw ? srv.engine->encode(cr.raw_prompt, false) : srv.engine->apply_chat_template(cr.messages, true);
        if (prompt.empty()) throw bad_request("the prompt produced no tokens");
        if (srv.max_ctx > 0) {
            const int n = static_cast<int>(prompt.size());
            if (n > srv.max_ctx - 2)   // the generator needs room for at least one more position (engine/speculative.cpp)
                throw ApiError{400, "invalid_request_error",
                               "the prompt is " + itos(n) + " tokens, which leaves no room in the " +
                                   itos(srv.max_ctx) + "-token context window",
                               "context_length_exceeded"};
            cr.gen.max_new_tokens = std::min(cr.gen.max_new_tokens, srv.max_ctx - n);
        }
    } catch (const ApiError& e) {
        send_error(c, e);
        return;
    } catch (const std::exception& e) {
        send_error(c, ApiError{500, "server_error", std::string("engine error: ") + e.what(), ""});
        return;
    }

    generate_and_reply(srv, c, cr, prompt);
}

// Strict decimal in [0, 999999999] from a string_view (no sign, no spaces).
bool parse_int_arg_view(std::string_view s, int& out) {
    if (s.empty() || s.size() > 9) return false;
    int v = 0;
    for (char ch : s) {
        if (ch < '0' || ch > '9') return false;
        v = v * 10 + (ch - '0');
    }
    out = v;
    return true;
}

// ---- benchmarks: --bench-json / --web-dir ------------------------------------------------------

constexpr size_t kMaxServedFile = 8u << 20;   // benchmarks JSON / page size cap
constexpr int kBenchPrefill = 500;            // quick bench: prompt tokens for TTFT / prefill tok/s
constexpr int kBenchDecode = 64;              // quick bench: greedy decode tokens

// Fixed real-text prompt of the quick bench (repeated until it covers kBenchPrefill tokens).
constexpr const char kBenchText[] =
    "De gemeente heeft gisteren een nieuw plan gepresenteerd voor het oude havengebied. Op de plek van de "
    "leegstaande loodsen moeten de komende tien jaar ongeveer twaalfhonderd woningen komen, waarvan een derde "
    "in de sociale huur. Daarnaast is er ruimte voor kleine bedrijven, een basisschool en een park langs het "
    "water. Bewoners van de aangrenzende wijk reageerden verdeeld. Sommigen zijn blij dat er eindelijk iets "
    "gebeurt met het terrein, dat al jaren verwaarloosd is. Anderen maken zich zorgen over de drukte op de "
    "smalle toegangswegen en over de hoogte van de geplande torens. De wethouder benadrukte dat het plan nog "
    "niet definitief is en dat er in het voorjaar drie inloopavonden komen. Ook wordt onderzocht of er een "
    "extra busverbinding naar het station kan komen. Volgens de planning begint de sanering van de grond "
    "volgend jaar; de eerste woningen zouden over vier jaar klaar moeten zijn. Het project wordt gefinancierd "
    "door de gemeente, een woningcorporatie en twee private ontwikkelaars, die samen ruim driehonderd miljoen "
    "euro investeren.\n\n";
constexpr const char kBenchDecodePrompt[] =
    "Hieronder volgt een uitgebreide beschrijving van de geschiedenis van de Nederlandse waterbouw, van de "
    "eerste terpen tot de Deltawerken:\n\n";

bool read_small_file(const std::string& path, std::string& out) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    out.clear();
    char buf[65536];
    size_t n;
    bool ok = true;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) {
        out.append(buf, n);
        if (out.size() > kMaxServedFile) {
            ok = false;
            break;
        }
    }
    if (std::ferror(f)) ok = false;
    std::fclose(f);
    return ok;
}

// Writes `path` via a temporary file and a rename, so readers never see a partial file.
bool write_file_atomic(const std::string& path, const std::string& data) {
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(data.data(), 1, data.size(), f) == data.size() && std::fflush(f) == 0;
    if (std::fclose(f) != 0 || !ok) {
        std::remove(tmp.c_str());
        return false;
    }
#if defined(_WIN32)
    const bool moved = MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
#else
    const bool moved = std::rename(tmp.c_str(), path.c_str()) == 0;
#endif
    if (!moved) std::remove(tmp.c_str());
    return moved;
}

void json_dump(std::string& out, const Json& j) {
    switch (j.kind) {
        case Json::Kind::Null: out += "null"; break;
        case Json::Kind::Bool: out += j.boolean ? "true" : "false"; break;
        case Json::Kind::Number:
            if (j.integral) {
                out += std::to_string(j.integer);
            } else if (!std::isfinite(j.number)) {
                out += "0";
            } else {
                char buf[40];
                std::snprintf(buf, sizeof buf, "%.10g", j.number);
                out += buf;
            }
            break;
        case Json::Kind::String: out += q(j.str); break;
        case Json::Kind::Array:
            out += '[';
            for (size_t i = 0; i < j.items.size(); ++i) {
                if (i) out += ',';
                json_dump(out, j.items[i]);
            }
            out += ']';
            break;
        case Json::Kind::Object:
            out += '{';
            for (size_t i = 0; i < j.items.size(); ++i) {
                if (i) out += ',';
                out += q(j.keys[i]);
                out += ':';
                json_dump(out, j.items[i]);
            }
            out += '}';
            break;
    }
}

// "[1m,5m,15m]" from /proc/loadavg, or null where it does not exist.
std::string loadavg_json() {
    std::string s;
    double a = 0, b = 0, c = 0;
    if (!read_small_file("/proc/loadavg", s) || std::sscanf(s.c_str(), "%lf %lf %lf", &a, &b, &c) != 3) return "null";
    return "[" + num(a) + "," + num(b) + "," + num(c) + "]";
}

std::string iso_utc_now() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

bool constant_time_equal(std::string_view a, std::string_view b) {
    const size_t n = std::max(a.size(), b.size());
    unsigned char diff = a.size() == b.size() ? 0 : 1;
    for (size_t i = 0; i < n; ++i) {
        const unsigned char x = i < a.size() ? static_cast<unsigned char>(a[i]) : 0;
        const unsigned char y = i < b.size() ? static_cast<unsigned char>(b[i]) : 0;
        diff |= static_cast<unsigned char>(x ^ y);
    }
    return diff == 0;
}

// POST /api/benchmarks/run: a matching admin token (X-Admin-Token or "Authorization: Bearer", env VELTRA_ADMIN_TOKEN / LB_ADMIN_TOKEN
// non-empty), or a direct loopback peer (a request carrying proxy headers is not treated as local).
// Loopback trust must not extend to web pages that merely run on this machine: a browser on any site can send requests to
// 127.0.0.1 (CORS "*" would even let it read the replies, e.g. saved chats). So loopback trust needs a loopback Host header
// (blocks DNS rebinding) and, when a browser sends an Origin, that Origin must be this server itself.
bool local_browser_ok(const lbhttp::Request& req) {
    const std::string host = lbhttp::lower_ascii(req.header("host"));
    std::string name = host;
    if (!name.empty() && name[0] == '[') name = name.substr(0, name.find(']') + 1);
    else name = name.substr(0, name.find(':'));
    if (!host.empty() && name != "127.0.0.1" && name != "localhost" && name != "[::1]") return false;
    const std::string origin = lbhttp::lower_ascii(req.header("origin"));
    return origin.empty() || origin == "http://" + host || origin == "https://" + host;
}

bool bench_authorized(sock_t c, const lbhttp::Request& req) {
    if (!g_require_token.empty()) return constant_time_equal(req.header("x-veltra-token"), g_require_token);
    const char* tok = lb::env_get("ADMIN_TOKEN");
    if (tok && *tok) {
        std::string given = req.header("x-admin-token");
        if (given.empty()) {
            const std::string auth = req.header("authorization");
            if (auth.size() > 7 && lbhttp::lower_ascii(std::string_view(auth).substr(0, 7)) == "bearer ") given = auth.substr(7);
        }
        if (!given.empty() && constant_time_equal(given, tok)) return true;
    }
    if (!req.header("x-forwarded-for").empty() || !req.header("forwarded").empty() || !req.header("x-real-ip").empty())
        return false;
    return lbhttp::peer_is_loopback(c) && local_browser_ok(req);
}

// Quick self-benchmark on the loaded engine; the caller holds gen_mu. Returns the "quick" JSON object.
std::string run_quick_bench(Server& srv) {
    const std::string nonce = "Meting " + itos(static_cast<long long>(std::time(nullptr))) + "-" +
                              itos(static_cast<long long>(srv.id_counter++)) + ".\n";  // defeats the prefix cache
    std::string text = nonce;
    std::vector<int> ids;
    for (int i = 0; i < 16 && static_cast<int>(ids.size()) < kBenchPrefill; ++i) {
        text += kBenchText;
        ids = srv.engine->encode(text, false);
    }
    if (static_cast<int>(ids.size()) > kBenchPrefill) ids.resize(kBenchPrefill);

    lb::GenOptions g;
    g.spec.enabled = false;  // strict T=1 decode, no REST: comparable with veltra-gen --spec 0
    g.spec.rest = false;
    g.n_threads = srv.opt.threads;

    // 1) prefill: TTFT = wall time until the first generated token.
    g.max_new_tokens = 1;
    srv.engine->reset();
    const auto t0 = Clock::now();
    double ttft = 0;
    bool have_first = false;
    const lb::GenStats a = srv.engine->generate(ids, g, [&](int, const std::string&) {
        if (!have_first) {
            ttft = std::chrono::duration<double>(Clock::now() - t0).count();
            have_first = true;
        }
        return true;
    });
    if (!have_first) ttft = std::chrono::duration<double>(Clock::now() - t0).count();
    const int pp_tokens = a.prompt_tokens > 0 ? a.prompt_tokens : static_cast<int>(ids.size());
    const double pp_tps = a.prefill_s > 0 ? (pp_tokens - a.cached_tokens) / a.prefill_s : 0.0;

    // 2) decode: 64 greedy tokens after a short prompt (ctx ~0).
    const std::vector<int> ids2 = srv.engine->encode(nonce + kBenchDecodePrompt, false);
    g.max_new_tokens = kBenchDecode;
    srv.engine->reset();
    const lb::GenStats b = srv.engine->generate(ids2, g, [](int, const std::string&) { return true; });
    srv.engine->reset();

    return "{\"timestamp\":" + q(iso_utc_now()) + ",\"source\":\"veltra-server in-process (POST /api/benchmarks/run)\"" +
           ",\"model\":" + q(srv.model_id) + ",\"isa\":" + q(srv.isa) + ",\"threads\":" + itos(srv.opt.threads) +
           ",\"prefill_tokens\":" + itos(pp_tokens) + ",\"cached_tokens\":" + itos(a.cached_tokens) +
           ",\"prefill_s\":" + num(a.prefill_s) + ",\"prefill_tok_s\":" + num(pp_tps) + ",\"ttft_s\":" + num(ttft) +
           ",\"decode_prompt_tokens\":" + itos(static_cast<long long>(ids2.size())) +
           ",\"decode_tokens\":" + itos(b.generated_tokens) + ",\"decode_tok_s\":" + num(b.tokens_per_s) +
           ",\"load_avg\":" + loadavg_json() + "}";
}

// Replaces (or adds) the top-level "quick" member of the --bench-json file. Empty string on success.
std::string store_quick(Server& srv, const std::string& quick) {
    std::lock_guard<std::mutex> lk(srv.bench_file_mu);
    Json root, qj;
    std::string jerr;
    if (!lbhttp::parse_json(quick, qj, jerr)) return "internal: quick result is not JSON: " + jerr;
    std::string old;
    if (read_small_file(srv.opt.bench_json, old)) {
        if (!lbhttp::parse_json(old, root, jerr) || root.kind != Json::Kind::Object)
            return "the existing benchmark file is not a JSON object; not overwritten";
    } else {
        root.kind = Json::Kind::Object;
    }
    for (size_t i = root.keys.size(); i-- > 0;)
        if (root.keys[i] == "quick") {
            root.keys.erase(root.keys.begin() + static_cast<std::ptrdiff_t>(i));
            root.items.erase(root.items.begin() + static_cast<std::ptrdiff_t>(i));
        }
    root.keys.push_back("quick");
    root.items.push_back(std::move(qj));
    std::string out;
    json_dump(out, root);
    out += '\n';
    if (!write_file_atomic(srv.opt.bench_json, out)) return "cannot write " + srv.opt.bench_json;
    return {};
}

void handle_bench_run(Server& srv, sock_t c, const lbhttp::Request& req) {
    if (!bench_authorized(c, req)) {
        send_error(c, ApiError{403, "permission_error", "benchmark runs need a loopback client or a valid X-Admin-Token", ""});
        return;
    }
    if (srv.opt.bench_json.empty()) {
        send_error(c, ApiError{404, "not_found_error", "no benchmark file configured (--bench-json)", ""});
        return;
    }
    bool expected = false;
    if (!srv.bench_running.compare_exchange_strong(expected, true)) {
        send_error(c, ApiError{409, "conflict_error", "a benchmark run is already in progress", "bench_running"});
        return;
    }
    BusyFlag running{srv.bench_running};

    // Same engine lock as the chat endpoints: waits for a running generation, never races it.
    srv.queued++;
    std::unique_lock<std::timed_mutex> lock(srv.gen_mu, std::defer_lock);
    const auto slice = std::chrono::milliseconds(kQueuePollMs);
    bool locked = lock.try_lock_for(slice);
    while (!locked && !lbhttp::peer_gone(c)) locked = lock.try_lock_for(slice);
    srv.queued--;
    if (!locked) return;
    if (!srv.engine) {
        send_error(c, ApiError{503, "server_error", "no model loaded", "model_not_loaded"});
        return;
    }

    std::string quick;
    {
        srv.busy = true;
        BusyFlag busy_flag{srv.busy};
        try {
            quick = run_quick_bench(srv);
        } catch (const std::exception& e) {
            send_error(c, ApiError{500, "server_error", std::string("benchmark failed: ") + e.what(), ""});
            return;
        }
    }
    lock.unlock();
    const std::string err = store_quick(srv, quick);
    if (!err.empty()) {
        send_error(c, ApiError{500, "server_error", err, ""});
        return;
    }
    log("benchmark run: " + quick);
    lbhttp::send_response(c, 200, "application/json", "{\"quick\":" + quick + ",\"saved\":true}", "Cache-Control: no-store\r\n");
}

// GET of one fixed file; never takes a path from the request.
void serve_file(sock_t c, const std::string& path, std::string_view type, const char* missing) {
    std::string body;
    if (path.empty() || !read_small_file(path, body)) {
        send_error(c, ApiError{404, "not_found_error", missing, ""});
        return;
    }
    lbhttp::send_response(c, 200, type, body, "Cache-Control: no-store\r\n");
}

void send_json(sock_t c, int status, const std::string& body, std::string_view extra = {});

// ---- launcher / model hub: embedded UI, /api/hardware, /api/catalog, model load + download ---------------

// Headers for a served file. Pages may only load from this server: injected markup or a crafted link cannot pull in a
// script or send data elsewhere. The arena and the plain chat page may also talk to other loopback ports (llama.cpp).
std::string ui_headers(std::string_view name, std::string_view type) {
    std::string h = "Cache-Control: no-store\r\n";
    if (type.substr(0, 9) != "text/html") return h;
    const bool loopback_ports = name == "arena.html" || name == "chat.html";
    h += "Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
         "img-src 'self' data:; connect-src 'self'";
    if (loopback_ports) h += " http://127.0.0.1:* http://localhost:*";
    h += "; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'self'\r\n"
         "Referrer-Policy: no-referrer\r\nX-Content-Type-Options: nosniff\r\n";
    return h;
}

// The desktop UI's own script and style files (static, no data in them). An exact list: the path is also the file name.
bool is_ui_asset(std::string_view p) { return p == "/pro.js" || p == "/pro.css" || p == "/market.js"; }

// Web page or data file: <--web-dir>/<name> when --web-dir is set (server layout), else the copy compiled into the
// binary (veltra.exe needs no folder next to it). Never takes a path from the request.
void serve_ui(Server& srv, sock_t c, const char* name, std::string_view type) {
    if (!srv.opt.web_dir.empty()) {
        std::string body;
        if (read_small_file(srv.opt.web_dir + "/" + name, body)) {
            lbhttp::send_response(c, 200, type, body, ui_headers(name, type));
            return;
        }
    }
#if defined(VELTRA_EMBED_UI)
    if (const veltra::ui::EmbeddedFile* f = veltra::ui::find(name)) {
        lbhttp::send_response(c, 200, f->type, std::string(reinterpret_cast<const char*>(f->data), f->size), ui_headers(name, f->type));
        return;
    }
#endif
    send_error(c, ApiError{404, "not_found_error", std::string(name) + " not available (--web-dir, or a build with VELTRA_EMBED_UI)", ""});
}

std::string file_name_of(const std::string& path) {
    const size_t k = path.find_last_of("/\\");
    return k == std::string::npos ? path : path.substr(k + 1);
}

std::string device_label(const std::string& isa) {
    std::string u;
    for (char ch : isa) u += (ch == '_') ? '-' : static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    return u.empty() ? std::string("-") : "CPU " + u;
}

std::string status_json(Server& srv) {
    std::lock_guard<std::mutex> lk(srv.info_mu);
    const bool loaded = !srv.model_path.empty();
    return "{\"loaded\":" + std::string(loaded ? "true" : "false") + ",\"model\":" + q(srv.model_id) +
           ",\"file\":" + q(file_name_of(srv.model_path)) + ",\"isa\":" + q(srv.isa) +
           ",\"device\":" + q(loaded ? device_label(srv.isa) : std::string("-")) + ",\"max_context\":" + itos(srv.max_ctx) +
           ",\"loading\":" + (srv.loading.load() ? "true" : "false") + ",\"loading_file\":" + q(file_name_of(srv.loading_path)) +
           ",\"load_error\":" + (srv.load_error.empty() ? std::string("null") : q(srv.load_error)) +
           ",\"busy\":" + (srv.busy.load() ? "true" : "false") + ",\"models_dir\":" + q(srv.opt.models_dir) +
           ",\"model_bytes\":" + itos(loaded ? (long long)srv.model_bytes : 0) + ",\"kv_bytes\":" + itos(loaded ? (long long)srv.kv_bytes : 0) +
           ",\"parked_file\":" + q(file_name_of(srv.parked_path)) + ",\"idle_unload_min\":" + itos(srv.idle_unload_min.load()) +
           ",\"cuda_built\":" + (lb::cuda_backend_built() ? "true" : "false") + "}";
}

int64_t steady_ms() { return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count(); }

std::string model_id_from_path(const std::string& path);

// The context a request can really use: the engine may allocate less than the model's max_position_embeddings.
int usable_context(const lb::Engine& e) {
    const int model = e.config().max_position_embeddings, engine = e.context_limit();
    if (model > 0 && engine > 0) return std::min(model, engine);
    return std::max(std::max(model, engine), 0);
}

// Loads `path` with gen_mu held (waits for a running generation); the old engine is freed first so two models never
// sit in RAM together. On failure the previous model is loaded again. Returns "" or the error.
std::string local_bench_history(Server& srv) {
    const std::string d = veltra::plat::app_data_dir();
    if (d.empty()) return "[]";
    std::lock_guard<std::mutex> lk(srv.bench_file_mu);
    std::FILE* f = std::fopen((d + "/local_benchmarks.json").c_str(), "rb");
    if (!f) return "[]";
    std::string text;
    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0 && text.size() < (4u << 20)) text.append(buf, n);
    std::fclose(f);
    return text;
}

// The context length the catalog's memory figures are computed for: ?ctx=N, else the current engine setting.
int catalog_context(const lbhttp::Request& req) {
    const size_t k = req.query.find("ctx=");
    const int v = k == std::string::npos ? 0 : std::atoi(req.query.c_str() + k + 4);
    return v >= 512 && v <= 32768 ? v : lb::engine_default_options().max_context;
}

// Reads the model file once, front to back, before the engine maps it: the page cache is warm afterwards and the UI can show
// real progress and the real read speed. Skipped when the file would not fit in free memory.
void prefetch_model(Server& srv, const std::string& path) {
    std::error_code ec;
    const uint64_t size = std::filesystem::file_size(path, ec);
    srv.load_done = 0;
    srv.load_total = ec ? 0 : size;
    if (ec || size / (1024.0 * 1024.0 * 1024.0) > veltra::detect_hardware().ram_free_gib * 0.9) return;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return;
    std::vector<char> buf(8u << 20);
    size_t n;
    while ((n = std::fread(buf.data(), 1, buf.size(), f)) > 0) srv.load_done += n;
    std::fclose(f);
}

std::string load_status_json(Server& srv) {
    std::lock_guard<std::mutex> lk(srv.info_mu);
    const int phase = srv.load_phase.load();
    const int64_t read_ms = phase == 1 ? steady_ms() - srv.load_t0_ms.load() : srv.load_read_ms.load();
    const uint64_t done = srv.load_done.load();
    const char* state = phase ? "loading" : !srv.load_error.empty() ? "error" : srv.load_all_ms.load() ? "done" : "idle";
    return std::string("{\"state\":\"") + state + "\",\"phase\":\"" + (phase == 1 ? "read" : phase == 2 ? "init" : "") +
           "\",\"file\":" + q(file_name_of(phase ? srv.loading_path : srv.model_path)) + ",\"bytes_done\":" + itos((long long)done) +
           ",\"bytes_total\":" + itos((long long)srv.load_total.load()) +
           ",\"gb_per_s\":" + (read_ms > 0 && done ? num(done / 1e9 / (read_ms / 1000.0)) : std::string("null")) +
           ",\"seconds\":" + num((phase ? steady_ms() - srv.load_t0_ms.load() : srv.load_all_ms.load()) / 1000.0) +
           ",\"error\":" + (srv.load_error.empty() ? std::string("null") : q(srv.load_error)) + "}";
}

std::string load_model_holding_gen(Server& srv, const std::string& path) {   // caller holds gen_mu
    const std::string prev = srv.model_path;
    {
        std::lock_guard<std::mutex> lk(srv.info_mu);
        srv.engine.reset();
        srv.model_path.clear(); srv.model_id.clear(); srv.isa.clear(); srv.max_ctx = 0;
    }
    std::string err;
    {
        std::lock_guard<std::mutex> lk(srv.info_mu);
        srv.loading_path = path;
    }
    struct Phase { Server& s; ~Phase() { s.load_phase = 0; s.load_all_ms = steady_ms() - s.load_t0_ms.load(); } } phase_guard{srv};
    srv.load_t0_ms = steady_ms();
    srv.load_read_ms = 0;
    srv.load_phase = 1;
    prefetch_model(srv, path);
    srv.load_read_ms = steady_ms() - srv.load_t0_ms.load();
    srv.load_phase = 2;
    std::unique_ptr<lb::Engine> e = lb::Engine::load(path, &err);
    std::string used = path;
    if (!e && !prev.empty() && prev != path) {
        std::string err2;
        e = lb::Engine::load(prev, &err2);
        used = prev;
    }
    std::lock_guard<std::mutex> lk(srv.info_mu);
    if (e) {
        const lb::ModelConfig& cfg = e->config();
        srv.model_id = cfg.model_name.empty() ? model_id_from_path(used) : cfg.model_name;
        srv.isa = e->isa_name();
        srv.max_ctx = usable_context(*e);
        srv.model_path = used;
        srv.engine = std::move(e);
        srv.kv_bytes = srv.engine->kv_bytes();
        std::error_code size_ec;
        const auto bytes = std::filesystem::file_size(used, size_ec);
        srv.model_bytes = size_ec ? 0 : static_cast<uint64_t>(bytes);
        srv.parked_path.clear();
        srv.last_used_ms = steady_ms();
    }
    if (used != path || !srv.engine) return "cannot load " + path + ": " + err;
    return {};
}

std::string load_model_locked(Server& srv, const std::string& path) {
    std::lock_guard<std::timed_mutex> gen(srv.gen_mu);
    return load_model_holding_gen(srv, path);
}

// Releases the model (weights and KV cache) but remembers it. 0 = done, 1 = busy generating / loading, 2 = nothing loaded.
int unload_model(Server& srv) {
    if (srv.loading.load()) return 1;
    std::unique_lock<std::timed_mutex> gen(srv.gen_mu, std::defer_lock);
    if (!gen.try_lock_for(std::chrono::milliseconds(300))) return 1;
    std::lock_guard<std::mutex> lk(srv.info_mu);
    if (!srv.engine) return 2;
    srv.parked_path = srv.model_path;
    srv.engine.reset();
    srv.model_path.clear(); srv.model_id.clear(); srv.isa.clear(); srv.max_ctx = 0;
    return 0;
}

// A request arrived while the model is parked: bring it back first. Caller holds gen_mu.
void reload_parked(Server& srv) {
    std::string path;
    {
        std::lock_guard<std::mutex> lk(srv.info_mu);
        if (srv.engine || srv.parked_path.empty()) return;
        path = srv.parked_path;
    }
    log("reloading the released model: " + path);
    try {
        const std::string e = load_model_holding_gen(srv, path);
        if (!e.empty()) log(e);
    } catch (const std::exception& ex) {
        log(std::string("reload failed: ") + ex.what());
    }
}

void handle_model_unload(Server& srv, sock_t c, const lbhttp::Request& req) {
    if (!bench_authorized(c, req)) {
        send_error(c, ApiError{403, "permission_error", "releasing the model needs a loopback client or a valid X-Admin-Token", ""});
        return;
    }
    const int rc = unload_model(srv);
    if (rc == 1) {
        send_error(c, ApiError{409, "conflict_error", "the model is busy; try again when the answer is finished", "busy"});
        return;
    }
    send_json(c, 200, std::string("{\"unloaded\":") + (rc == 0 ? "true" : "false") + "}");
}

void handle_model_load(Server& srv, sock_t c, const lbhttp::Request& req) {
    if (!bench_authorized(c, req)) {
        send_error(c, ApiError{403, "permission_error", "loading a model needs a loopback client or a valid X-Admin-Token", ""});
        return;
    }
    Json root;
    std::string jerr;
    if (!lbhttp::parse_json(req.body, root, jerr) || root.kind != Json::Kind::Object) {
        send_error(c, bad_request("body must be a JSON object {\"file\": ...} or {\"path\": ...}"));
        return;
    }
    const Json* f = present(root, "file");
    const Json* pth = present(root, "path");
    const Json* pick = f ? f : pth;
    if (!pick || pick->kind != Json::Kind::String) {
        send_error(c, bad_request("give \"file\" (a name in the models folder) or \"path\" (absolute .vlt path)"));
        return;
    }
    std::string err;
    const std::string path = veltra::resolve_model_path(pick->str, srv.opt.models_dir, &err);
    if (path.empty()) {
        send_error(c, ApiError{404, "not_found_error", err, "model_file_not_found"});
        return;
    }
    bool expected = false;
    if (!srv.loading.compare_exchange_strong(expected, true)) {
        send_error(c, ApiError{409, "conflict_error", "a model is already being loaded", "loading"});
        return;
    }
    {
        std::lock_guard<std::mutex> lk(srv.info_mu);
        srv.loading_path = path;
        srv.load_error.clear();
    }
    Server* s = &srv;
    std::thread([s, path]() {
        std::string e;
        try {
            e = load_model_locked(*s, path);
        } catch (const std::exception& ex) {
            e = std::string("load failed: ") + ex.what();
        }
        {
            std::lock_guard<std::mutex> lk(s->info_mu);
            s->load_error = e;
            s->loading_path.clear();
        }
        s->loading = false;
        log(e.empty() ? "loaded " + path : e);
    }).detach();
    send_json(c, 202, "{\"loading\":true,\"path\":" + q(path) + "}");
}

void handle_model_download(Server& srv, sock_t c, const lbhttp::Request& req) {
    if (!bench_authorized(c, req)) {
        send_error(c, ApiError{403, "permission_error", "downloads need a loopback client or a valid X-Admin-Token", ""});
        return;
    }
    Json root;
    std::string jerr;
    if (!lbhttp::parse_json(req.body, root, jerr) || root.kind != Json::Kind::Object) {
        send_error(c, bad_request("body must be a JSON object {\"id\": ...} or {\"url\": ..., \"file\": ...}"));
        return;
    }
    auto str = [&](const char* k) -> std::string {
        const Json* v = present(root, k);
        return v && v->kind == Json::Kind::String ? v->str : std::string();
    };
    std::string err;
    if (!veltra::start_download(str("id"), str("url"), str("file"), srv.opt.models_dir, &err)) {
        send_error(c, ApiError{400, "invalid_request_error", err, "download_refused"});
        return;
    }
    send_json(c, 202, veltra::download_status_json());
}

// ---- arena: POST /compare/llama/v1/chat/completions -> --compare-upstream ----------------------------
// A same-origin path to a loopback llama.cpp server, so the arena page needs no second tunnel. Only this one
// path is forwarded (always to <upstream>/v1/chat/completions); the upstream must be loopback (checked at startup).

constexpr int kCompareTimeoutMs = 300000;          // whole upstream exchange: a long CPU generation is fine
constexpr size_t kMaxUpstreamBytes = 16u << 20;    // upstream response cap (headers + body)

lbhttp::sock_t connect_upstream(const std::string& host, int port, std::string& err) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_NUMERICHOST;
    addrinfo* res = nullptr;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || !res) {
        err = "bad upstream address";
        return lbhttp::kBadSock;
    }
    sock_t s = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (s == lbhttp::kBadSock) {
        err = "socket: " + lbhttp::last_socket_error();
        freeaddrinfo(res);
        return lbhttp::kBadSock;
    }
    if (::connect(s, res->ai_addr, static_cast<socklen_t>(res->ai_addrlen)) != 0) {
        err = "upstream unreachable (" + lbhttp::last_socket_error() + ")";
        lbhttp::close_sock(s);
        freeaddrinfo(res);
        return lbhttp::kBadSock;
    }
    freeaddrinfo(res);
    return s;
}

// Decodes a chunked body in place; false when malformed or incomplete.
bool dechunk(const std::string& in, std::string& out) {
    out.clear();
    size_t pos = 0;
    for (;;) {
        const size_t eol = in.find("\r\n", pos);
        if (eol == std::string::npos) return false;
        std::string_view line(in.data() + pos, eol - pos);
        const size_t semi = line.find(';');
        if (semi != std::string_view::npos) line = line.substr(0, semi);
        line = lbhttp::trim_ws(line);
        if (line.empty()) return false;
        size_t n = 0;
        for (char ch : line) {
            const int v = lbhttp::hex_value(ch);
            if (v < 0 || n > (kMaxUpstreamBytes >> 4)) return false;
            n = n * 16 + static_cast<size_t>(v);
        }
        pos = eol + 2;
        if (n == 0) return true;
        if (in.size() < pos + n + 2) return false;
        out.append(in, pos, n);
        pos += n + 2;
    }
}

void handle_compare_proxy(Server& srv, sock_t c, const lbhttp::Request& req) {
    if (srv.opt.compare_port == 0) {
        send_error(c, ApiError{404, "not_found_error", "no comparison upstream configured (--compare-upstream)", ""});
        return;
    }
    std::string err;
    const sock_t u = connect_upstream(srv.opt.compare_host, srv.opt.compare_port, err);
    if (u == lbhttp::kBadSock) {
        send_error(c, ApiError{502, "server_error", "comparison upstream: " + err, "upstream_unreachable"});
        return;
    }
    lbhttp::set_timeouts(u, kCompareTimeoutMs, kCompareTimeoutMs);
    const std::string host_hdr = (srv.opt.compare_host.find(':') != std::string::npos ? "[" + srv.opt.compare_host + "]"
                                                                                       : srv.opt.compare_host) +
                                 ":" + itos(srv.opt.compare_port);
    std::string out = "POST /v1/chat/completions HTTP/1.1\r\nHost: " + host_hdr +
                      "\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\n"
                      "Connection: close\r\nContent-Length: " + itos(static_cast<long long>(req.body.size())) + "\r\n\r\n";
    out += req.body;
    std::string raw;
    bool ok = lbhttp::send_all(u, out);
    const auto t0 = Clock::now();
    char buf[65536];
    while (ok) {
        const long n = lbhttp::recv_some(u, buf, sizeof buf);
        if (n == 0) break;
        if (n < 0 || raw.size() + static_cast<size_t>(n) > kMaxUpstreamBytes ||
            Clock::now() - t0 > std::chrono::milliseconds(kCompareTimeoutMs)) {
            ok = false;
            break;
        }
        raw.append(buf, static_cast<size_t>(n));
    }
    lbhttp::close_sock(u);
    const size_t hdr_end = raw.find("\r\n\r\n");
    int status = 0;
    if (ok && hdr_end != std::string::npos && raw.rfind("HTTP/1.", 0) == 0 && raw.size() > 12)
        parse_int_arg_view(std::string_view(raw).substr(9, 3), status);
    if (status < 100 || status > 599) {
        send_error(c, ApiError{502, "server_error", ok ? "comparison upstream sent no valid HTTP response"
                                                       : "comparison upstream failed, timed out or answered too much",
                               "upstream_error"});
        return;
    }
    std::string ctype = "application/json";
    bool chunked = false;
    long long clen = -1;
    size_t ls = raw.find("\r\n") + 2;
    while (ls < hdr_end) {
        const size_t le = raw.find("\r\n", ls);
        const std::string_view line(raw.data() + ls, le - ls);
        const size_t colon = line.find(':');
        if (colon != std::string_view::npos) {
            const std::string name = lbhttp::lower_ascii(line.substr(0, colon));
            const std::string_view val = lbhttp::trim_ws(line.substr(colon + 1));
            if (name == "content-type" && !val.empty()) ctype = std::string(val);
            else if (name == "transfer-encoding") chunked = lbhttp::lower_ascii(val).find("chunked") != std::string::npos;
            else if (name == "content-length") {
                size_t v = 0;
                if (lbhttp::parse_decimal(val, v)) clen = static_cast<long long>(v);
            }
        }
        ls = le + 2;
    }
    std::string body = raw.substr(hdr_end + 4);
    if (chunked) {
        std::string dec;
        if (!dechunk(body, dec)) {
            send_error(c, ApiError{502, "server_error", "comparison upstream sent a malformed chunked body", "upstream_error"});
            return;
        }
        body.swap(dec);
    } else if (clen >= 0 && static_cast<size_t>(clen) < body.size()) {
        body.resize(static_cast<size_t>(clen));
    }
    lbhttp::send_response(c, status, ctype, body, "Cache-Control: no-store\r\n");
}

// ---- opt-in remote GPU test mode ---------------------------------------------------------------

std::string run_device_label(const std::string& isa, bool gpu_active) {
    return gpu_active ? device_label(isa) + " + CUDA (MoE-experts in VRAM op de GPU, de rest op de CPU)" : device_label(isa);
}

// One approved suite on this process's engine, under the same lock as chat requests; the engine is reset afterwards.
veltra::tests::Report run_remote_suite(Server& srv, const veltra::tests::SuiteInfo& suite, const std::atomic<bool>& cancel,
                                       const veltra::tests::ProgressFn& progress) {
    veltra::tests::Progress wait;
    wait.total = static_cast<int>(suite.steps.size());
    wait.message = "wachten tot de engine vrij is";
    progress(wait);
    std::unique_lock<std::timed_mutex> lock(srv.gen_mu, std::defer_lock);
    while (!lock.try_lock_for(std::chrono::milliseconds(kQueuePollMs))) {
        if (cancel.load()) break;
    }
    if (cancel.load()) {
        veltra::tests::Report r;
        r.status = "cancelled";
        r.headline = "Geannuleerd voordat de test begon";
        return r;
    }
    srv.busy = true;
    BusyFlag busy_flag{srv.busy};
    veltra::tests::EngineContext ctx;
    ctx.threads = srv.opt.threads;
    ctx.spec_default = !srv.opt.no_spec;
    ctx.rest = srv.opt.rest;
    ctx.mtp = srv.opt.mtp;
    if (srv.engine) {
        ctx.engine = srv.engine.get();
        ctx.model_path = srv.model_path;
        ctx.model_file = file_name_of(srv.model_path);
        ctx.model_id = file_name_of(srv.model_id);   // a model name stored in the file may be a path: keep the last part
        ctx.device = run_device_label(srv.isa, srv.engine->gpu_active());
        ctx.max_context = srv.engine->context_limit();
        if (srv.max_ctx > 0) ctx.max_context = std::min(ctx.max_context, srv.max_ctx);
        ctx.chat_options = [&srv](int max_new) {   // exactly what a temperature-0 /v1/chat/completions request gets
            Json body;
            std::string err;
            lbhttp::parse_json("{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"temperature\":0,\"max_tokens\":" +
                                   itos(max_new) + "}", body, err);
            return parse_chat(body, srv).gen;
        };
    } else {
        ctx.device = "geen model geladen";
    }
    struct ResetEngine {   // on every exit path
        lb::Engine* e;
        ~ResetEngine() { if (e) e->reset(); }
    } reset{srv.engine.get()};
    if (srv.engine) srv.engine->reset();
    veltra::tests::Report report = veltra::tests::run_suite(suite, ctx, cancel, progress);
    log(std::string("remote test ") + suite.id + ": " + report.status);
    return report;
}

// Shown in the consent popup: what was detected and what will really run the model.
std::string remote_hardware_line(Server& srv) {
    const veltra::HwInfo hw = veltra::detect_hardware();
    std::string gpus;
    for (const auto& g : hw.gpus) {
        char vram[32];
        std::snprintf(vram, sizeof vram, " (%.1f GB)", g.vram_total_gib);
        gpus += (gpus.empty() ? "" : ", ") + g.name + vram;
    }
    std::string isa;
    {
        std::lock_guard<std::mutex> lk(srv.info_mu);
        isa = srv.isa;
    }
    return "Videokaart: " + (gpus.empty() ? std::string("geen gevonden") : gpus) + " \xc2\xb7 het model draait op: " +
           (isa.empty() ? std::string("geen model geladen")
                        : device_label(isa) + (lb::cuda_backend_built() ? " (CUDA-build: mogelijk deels op de videokaart)"
                                                                        : " (deze build gebruikt de videokaart niet voor het model)"));
}

void handle_remote(Server& srv, sock_t c, const lbhttp::Request& req) {
    (void)srv;
    if (!bench_authorized(c, req)) {
        send_error(c, ApiError{403, "permission_error", "the remote test settings are only available to this app / a local client", ""});
        return;
    }
    const std::string& m = req.method;
    const std::string& p = req.path;
    if (p != "/api/remote/status" && p != "/api/remote/config" && p != "/api/remote/decision" && p != "/api/remote/cancel") {
        send_error(c, ApiError{404, "not_found_error", "no route for " + m + " " + p, ""});
        return;
    }
    const char* want = p == "/api/remote/status" ? "GET" : "POST";
    if (m != want) {
        send_json(c, 405, error_json({405, "invalid_request_error", std::string("use ") + want, ""}), "Allow: GET, POST, OPTIONS\r\n");
        return;
    }
    std::string err;
    int status = 200;
    if (p == "/api/remote/config") status = veltra::remote::configure(req.body, err);
    else if (p == "/api/remote/decision") status = veltra::remote::decide(req.body, err);
    else if (p == "/api/remote/cancel") status = veltra::remote::cancel(err);
    if (status != 200) {
        send_error(c, ApiError{status, status == 409 ? "conflict_error" : status == 400 ? "invalid_request_error" : "server_error", err, ""});
        return;
    }
    send_json(c, 200, veltra::remote::status_json(), "Cache-Control: no-store\r\n");
}

// ---- routing -----------------------------------------------------------------------------------

void send_json(sock_t c, int status, const std::string& body, std::string_view extra) {
    lbhttp::send_response(c, status, "application/json", body, extra);
}

// ---- app data files -----------------------------------------------------------------------------

std::string data_file(const char* name) {
    const std::string d = veltra::plat::app_data_dir();
    if (d.empty()) return {};
#if defined(_WIN32)
    return d + "\\" + name;
#else
    return d + "/" + name;
#endif
}

// Like read_small_file, with its own cap. 1 = read, 0 = missing, -1 = unreadable / too large.
int read_capped_file(const std::string& path, size_t cap, std::string& out) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return 0;
    out.clear();
    char buf[65536];
    size_t n;
    int rc = 1;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) {
        out.append(buf, n);
        if (out.size() > cap) {
            rc = -1;
            break;
        }
    }
    if (std::ferror(f)) rc = -1;
    std::fclose(f);
    return rc;
}

Json json_str(std::string s) {
    Json j;
    j.kind = Json::Kind::String;
    j.str = std::move(s);
    return j;
}
Json json_int(int64_t v) {
    Json j;
    j.kind = Json::Kind::Number;
    j.integral = true;
    j.integer = v;
    j.number = static_cast<double>(v);
    return j;
}
void json_set(Json& obj, const std::string& key, Json v) {
    for (size_t i = 0; i < obj.keys.size(); ++i)
        if (obj.keys[i] == key) {
            obj.items[i] = std::move(v);
            return;
        }
    obj.keys.push_back(key);
    obj.items.push_back(std::move(v));
}
std::string json_get_str(const Json& o, std::string_view key) {
    const Json* v = o.get(key);
    return v && v->kind == Json::Kind::String ? v->str : std::string();
}
int64_t json_get_int(const Json& o, std::string_view key) {
    const Json* v = o.get(key);
    int64_t x = 0;
    return v && v->as_int(x) ? x : 0;
}

// ---- compute selection / cluster (engine/cluster.cpp) -------------------------------------------

constexpr int kForwardTimeoutMs = 300000;           // one forwarded chat request (long CPU generations are fine)
constexpr size_t kMaxForwardBytes = 16u << 20;

// Equal-share split of the loaded model's layers over this machine + the online nodes. A proposal only.
std::string layer_plan_json(Server& srv, const std::vector<std::string>& online) {
    int n_layers = 0;
    std::string mid;
    {
        std::lock_guard<std::mutex> lk(srv.info_mu);
        if (srv.engine) n_layers = srv.engine->config().num_layers;
        mid = srv.model_id;
    }
    std::vector<std::string> parts{"local"};
    parts.insert(parts.end(), online.begin(), online.end());
    std::string o = "{\"status\":\"plan only: layer-split execution is not implemented\",\"model\":" +
                    (mid.empty() ? std::string("null") : q(mid)) + ",\"n_layers\":" + (n_layers > 0 ? itos(n_layers) : "null") +
                    ",\"parts\":[";
    if (n_layers > 0) {
        const int k = static_cast<int>(parts.size());
        int start = 0;
        for (int i = 0; i < k; ++i) {
            const int cnt = n_layers / k + (i < n_layers % k ? 1 : 0);
            if (i) o += ',';
            o += "{\"device\":" + q(parts[static_cast<size_t>(i)]) + ",\"first_layer\":" + itos(start) +
                 ",\"last_layer\":" + itos(start + cnt - 1) + ",\"n_layers\":" + itos(cnt) +
                 ",\"share\":" + num(static_cast<double>(cnt) / n_layers) + "}";
            start += cnt;
        }
    }
    return o + "]}";
}

std::string compute_json(Server& srv) {
    namespace vc = veltra::cluster;
    const vc::Selection sel = vc::current();
    const std::vector<vc::NodeStatus> st = sel.nodes.empty() ? std::vector<vc::NodeStatus>{} : vc::ping_all(sel.nodes);
    std::vector<std::string> online;
    for (const auto& s : st)
        if (s.online) online.push_back(s.node);
    std::string effective = "cpu", reason;
    if (sel.mode == "gpu") {
        if (lb::cuda_backend_built()) effective = "gpu";
        else reason = "CUDA backend not in this build (docs/GPU_RTX3060_GUIDE.md)";
    } else if (sel.mode == "cluster") {
        if (!online.empty()) {
            effective = "cluster";
            reason = "POST /v1/chat/completions is forwarded per request to an online node (round-robin); "
                     "one model is not split across machines";
        } else {
            reason = "no cluster node online; requests run on this machine";
        }
    }
    std::string nodes = "[";
    for (size_t i = 0; i < st.size(); ++i) nodes += (i ? "," : "") + vc::node_status_json(st[i]);
    nodes += "]";
    return "{\"selection\":" + vc::selection_json(sel) + ",\"mode\":" + q(sel.mode) + ",\"effective\":" + q(effective) +
           ",\"reason\":" + (reason.empty() ? std::string("null") : q(reason)) +
           ",\"cuda_backend_built\":" + (lb::cuda_backend_built() ? "true" : "false") + ",\"nodes\":" + nodes +
           ",\"online_nodes\":" + itos(static_cast<long long>(online.size())) +
           ",\"routing\":" + (effective == "cluster" ? q("request-level round-robin over online nodes") : std::string("null")) +
           ",\"layer_plan\":" + (sel.mode == "cluster" ? layer_plan_json(srv, online) : std::string("null")) + "}";
}

void handle_compute_select(Server& srv, sock_t c, const lbhttp::Request& req) {
    if (!bench_authorized(c, req)) {
        send_error(c, ApiError{403, "permission_error", "changing the compute selection needs a loopback client or a valid X-Admin-Token", ""});
        return;
    }
    veltra::cluster::Selection sel;
    std::string err;
    if (!veltra::cluster::parse_selection(req.body, static_cast<int>(veltra::detect_hardware().gpus.size()), sel, err)) {
        send_error(c, bad_request(err, "invalid_compute_selection"));
        return;
    }
    const std::string warn = veltra::cluster::set_current(sel);
    log("compute selection: " + veltra::cluster::selection_json(sel) + (warn.empty() ? "" : " (" + warn + ")"));
    std::string body = compute_json(srv);
    if (!warn.empty()) body = "{\"warning\":" + q(warn) + "," + body.substr(1);
    send_json(c, 200, body, "Cache-Control: no-store\r\n");
}

// Cluster mode: forwards one POST /v1/chat/completions to an online node. True when a response was sent; false means
// "run it locally" (not cluster mode, already forwarded once, no node online, or every node failed at transport level).
// A stream request is forwarded as non-stream and its answer re-emitted as SSE chunks.
bool forward_chat_to_cluster(Server& srv, sock_t c, const lbhttp::Request& req) {
    namespace vc = veltra::cluster;
    if (req.path != "/v1/chat/completions" || !req.header("x-veltra-forwarded").empty()) return false;
    const vc::Selection sel = vc::current();
    if (sel.mode != "cluster" || sel.nodes.empty()) return false;
    Json root;
    std::string jerr;
    if (!lbhttp::parse_json(req.body, root, jerr) || root.kind != Json::Kind::Object) return false;   // local path reports it
    const Json* sv = root.get("stream");
    const bool stream = sv && sv->kind == Json::Kind::Bool && sv->boolean;
    std::string body = req.body;
    if (stream) {
        Json copy;
        copy.kind = Json::Kind::Object;
        for (size_t i = 0; i < root.keys.size(); ++i)
            if (root.keys[i] != "stream" && root.keys[i] != "stream_options") {
                copy.keys.push_back(root.keys[i]);
                copy.items.push_back(root.items[i]);
            }
        Json f;
        f.kind = Json::Kind::Bool;
        f.boolean = false;
        copy.keys.push_back("stream");
        copy.items.push_back(f);
        body.clear();
        json_dump(body, copy);
    }
    for (size_t attempt = 0; attempt < sel.nodes.size(); ++attempt) {
        const std::string node = vc::pick_online_node(sel.nodes);
        if (node.empty()) break;
        std::string host, perr;
        int port = 0;
        if (!vc::parse_node(node, &host, &port, &perr)) break;
        const vc::HttpResult r = vc::http_request(host, port, "POST", "/v1/chat/completions", body,
                                                  "Content-Type: application/json\r\nAccept: application/json\r\nX-Veltra-Forwarded: 1\r\n",
                                                  kForwardTimeoutMs, kMaxForwardBytes);
        if (r.status == 0) {
            log("cluster node " + node + " failed: " + r.err + "; trying the next node");
            vc::mark_offline(node);
            continue;
        }
        log("chat forwarded to cluster node " + node + " (HTTP " + itos(r.status) + ", " + num(r.ms) + " ms)");
        const std::string extra = "Cache-Control: no-store\r\nX-Veltra-Node: " + node + "\r\n";
        if (!stream || r.status != 200) {
            lbhttp::send_response(c, r.status, r.content_type.empty() ? "application/json" : r.content_type, r.body, extra);
            return true;
        }
        Json resp;
        if (!lbhttp::parse_json(r.body, resp, jerr) || resp.kind != Json::Kind::Object) {
            send_error(c, ApiError{502, "server_error", "cluster node " + node + " sent no valid JSON completion", "upstream_error"});
            return true;
        }
        std::string content, finish = "stop";
        if (const Json* ch = resp.get("choices"); ch && ch->kind == Json::Kind::Array && !ch->items.empty()) {
            if (const Json* msg = ch->items[0].get("message")) content = json_get_str(*msg, "content");
            const std::string fr = json_get_str(ch->items[0], "finish_reason");
            if (!fr.empty()) finish = fr;
        }
        std::string usage = "null";
        if (const Json* u = resp.get("usage")) {
            usage.clear();
            json_dump(usage, *u);
        }
        std::string id = json_get_str(resp, "id");
        if (id.empty()) id = make_id(srv);
        const std::string model = json_get_str(resp, "model");
        const int64_t created = json_get_int(resp, "created");
        const std::string head = "{\"id\":" + q(id) + ",\"object\":\"chat.completion.chunk\",\"created\":" +
                                 itos(created ? created : static_cast<long long>(std::time(nullptr))) + ",\"model\":" + q(model) +
                                 ",\"x_veltra_node\":" + q(node);
        const bool sent =
            lbhttp::send_sse_head(c) &&
            lbhttp::send_chunk(c, sse(head + ",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}")) &&
            lbhttp::send_chunk(c, sse(head + ",\"choices\":[{\"index\":0,\"text\":" + q(content) + ",\"delta\":{\"content\":" + q(content) +
                                      "},\"finish_reason\":null}]}")) &&
            lbhttp::send_chunk(c, sse(head + ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":" + q(finish) + "}]}")) &&
            lbhttp::send_chunk(c, sse(head + ",\"choices\":[],\"usage\":" + usage + "}")) &&
            lbhttp::send_chunk(c, "data: [DONE]\n\n") && lbhttp::send_last_chunk(c);
        (void)sent;
        return true;
    }
    log("cluster mode: no node reachable; running the request locally");
    return false;
}

// ---- chat storage: <app data>/chats.json ---------------------------------------------------------

constexpr size_t kMaxChats = 500;
constexpr size_t kMaxChatMessages = 1000;
constexpr size_t kMaxChatsFile = 64u << 20;
constexpr size_t kMaxChatTitle = 200;

struct ChatStore {
    std::mutex mu;
    bool loaded = false;
    std::string load_error;   // chats.json exists but could not be read: never overwritten then
    std::vector<Json> chats;  // objects {id, title, created, updated, messages}
};
ChatStore g_chats;

bool valid_chat_id(const std::string& id) {
    if (id.empty() || id.size() > 64) return false;
    for (char ch : id)
        if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) return false;
    return true;
}

std::string random_hex16() {
    static std::atomic<uint64_t> ctr{0};
    uint64_t x = 0;
    try {
        std::random_device rd;
        x = (static_cast<uint64_t>(rd()) << 32) ^ rd();
    } catch (...) {
    }
    x ^= static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) * 0x9E3779B97F4A7C15ull;
    x += (++ctr) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    x ^= x >> 31;
    char buf[24];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(x));
    return buf;
}

// Caller holds g_chats.mu. False (with load_error) when chats.json exists but is unusable.
bool chats_load_locked() {
    if (g_chats.loaded) return g_chats.load_error.empty();
    g_chats.loaded = true;
    const std::string path = data_file("chats.json");
    if (path.empty()) {
        g_chats.load_error = "no writable app data directory";
        return false;
    }
    std::string text;
    const int rc = read_capped_file(path, kMaxChatsFile, text);
    if (rc == 0) return true;
    Json root;
    std::string jerr;
    if (rc < 0 || !lbhttp::parse_json(text, root, jerr) || root.kind != Json::Kind::Array) {
        g_chats.load_error = path + " is unreadable or not a JSON array; it is left untouched";
        return false;
    }
    for (Json& ch : root.items)
        if (ch.kind == Json::Kind::Object && valid_chat_id(json_get_str(ch, "id"))) g_chats.chats.push_back(std::move(ch));
    return true;
}

// Caller holds g_chats.mu. "" on success.
std::string chats_save_locked() {
    std::string out = "[";
    for (size_t i = 0; i < g_chats.chats.size(); ++i) {
        if (i) out += ',';
        json_dump(out, g_chats.chats[i]);
    }
    out += "]\n";
    if (out.size() > kMaxChatsFile) return "too_large";
    const std::string path = data_file("chats.json");
    if (path.empty() || !write_file_atomic(path, out)) return "cannot write " + (path.empty() ? std::string("chats.json") : path);
    return {};
}

std::string chat_summary_json(const Json& ch) {
    const Json* m = ch.get("messages");
    const size_t n = m && m->kind == Json::Kind::Array ? m->items.size() : 0;
    return "{\"id\":" + q(json_get_str(ch, "id")) + ",\"title\":" + q(json_get_str(ch, "title")) +
           ",\"created\":" + itos(json_get_int(ch, "created")) + ",\"updated\":" + itos(json_get_int(ch, "updated")) +
           ",\"n_messages\":" + itos(static_cast<long long>(n)) + "}";
}

// UTF-8-safe prefix of at most `max` bytes.
std::string utf8_prefix(const std::string& s, size_t max) {
    if (s.size() <= max) return s;
    size_t k = max;
    while (k > 0 && (static_cast<unsigned char>(s[k]) & 0xC0) == 0x80) --k;
    return s.substr(0, k);
}

constexpr size_t kMaxUiState = 256u << 10;

void apply_ui_state(Server& srv, const Json& root) {
    // the context slider: a loaded model with another size is released and comes back with the new size on the next request
    const Json* cx = present(root, "ctx");
    if (srv.opt.ctx == 0 && cx && cx->kind == Json::Kind::Number && cx->number >= 2048 && cx->number <= 32768) {
        const int want = static_cast<int>(cx->number);
        int& cur = lb::engine_default_options().max_context;   // written under the ui-state lock, read when a model is loaded
        if (cur != want) {
            cur = want;
            bool differs = false;
            {
                std::lock_guard<std::mutex> info(srv.info_mu);
                differs = srv.engine && srv.engine->context_limit() != std::min(want, std::max(srv.engine->config().max_position_embeddings, 512));
            }
            if (differs && unload_model(srv) == 0) log("context set to " + itos(want) + ": the model reloads on the next request");
        }
    }
    const Json* v = present(root, "idle_unload_min");
    if (v && v->kind == Json::Kind::Number && v->number >= 0 && v->number <= 240) {
        const int m = static_cast<int>(v->number);
        if (m != srv.idle_unload_min.exchange(m)) srv.last_used_ms = steady_ms();   // the countdown starts now
    }
}

// GET / POST /api/ui-state: one JSON object with the UI's settings and unsent drafts. The desktop app listens on a new port
// every start, so browser storage (per origin) would forget everything.
void handle_ui_state(Server& srv, sock_t c, const lbhttp::Request& req) {
    if (!bench_authorized(c, req)) {
        send_error(c, ApiError{403, "permission_error", "ui state needs a loopback client or a valid X-Admin-Token", ""});
        return;
    }
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    const std::string path = data_file("ui_state.json");
    if (req.method == "GET") {
        std::string text;
        if (path.empty() || read_capped_file(path, kMaxUiState, text) != 1 || text.empty()) text = "{}";
        send_json(c, 200, text, "Cache-Control: no-store\r\n");
        return;
    }
    if (req.method != "POST") {
        send_json(c, 405, error_json({405, "invalid_request_error", "use GET or POST", ""}), "Allow: GET, POST, OPTIONS\r\n");
        return;
    }
    Json root;
    std::string jerr;
    if (req.body.size() > kMaxUiState || !lbhttp::parse_json(req.body, root, jerr) || root.kind != Json::Kind::Object) {
        send_error(c, bad_request("body must be a JSON object of at most 256 KiB"));
        return;
    }
    apply_ui_state(srv, root);
    if (path.empty() || !write_file_atomic(path, req.body)) {
        send_error(c, ApiError{500, "server_error", "cannot write ui_state.json", ""});
        return;
    }
    send_json(c, 200, "{\"saved\":true}");
}

// POST /api/tokenize {"text": ...} -> {"tokens": n, "exact": true}: the loaded model's own tokenizer (thread-safe, so it
// does not wait for a running generation). Without a loaded model the count is an estimate and says so.
void handle_tokenize(Server& srv, sock_t c, const lbhttp::Request& req) {
    if (!bench_authorized(c, req)) {
        send_error(c, ApiError{403, "permission_error", "tokenize needs a loopback client or a valid X-Admin-Token", ""});
        return;
    }
    Json root;
    std::string jerr;
    const Json* t = lbhttp::parse_json(req.body, root, jerr) && root.kind == Json::Kind::Object ? present(root, "text") : nullptr;
    if (!t || t->kind != Json::Kind::String) {
        send_error(c, bad_request("body must be {\"text\": \"...\"}"));
        return;
    }
    long long n = -1;
    int ctx = 0;
    {
        std::lock_guard<std::mutex> lk(srv.info_mu);   // the engine cannot be freed while this is held
        if (srv.engine) {
            n = static_cast<long long>(srv.engine->encode(t->str, false).size());
            ctx = srv.max_ctx;
        }
    }
    const bool exact = n >= 0;
    if (!exact) n = static_cast<long long>(t->str.size() / 3.6) + 1;
    send_json(c, 200, "{\"tokens\":" + itos(n) + ",\"exact\":" + (exact ? "true" : "false") + ",\"max_context\":" + itos(ctx) + "}",
              "Cache-Control: no-store\r\n");
}

// Applies the saved "ctx" setting to the engine options before the first model load.
void preset_context() {
    std::string text;
    const std::string path = data_file("ui_state.json");
    Json root;
    std::string jerr;
    if (path.empty() || read_capped_file(path, kMaxUiState, text) != 1 || !lbhttp::parse_json(text, root, jerr) || root.kind != Json::Kind::Object) return;
    const Json* cx = present(root, "ctx");
    if (cx && cx->kind == Json::Kind::Number && cx->number >= 2048 && cx->number <= 32768)
        lb::engine_default_options().max_context = static_cast<int>(cx->number);
}

void start_idle_unloader(Server& srv) {
    std::string text;
    const std::string path = data_file("ui_state.json");
    Json root;
    std::string jerr;
    if (!path.empty() && read_capped_file(path, kMaxUiState, text) == 1 && lbhttp::parse_json(text, root, jerr) &&
        root.kind == Json::Kind::Object)
        apply_ui_state(srv, root);
    srv.last_used_ms = steady_ms();
    Server* s = &srv;
    std::thread([s]() {
        // VELTRA_IDLE_UNLOAD_SEC: test override for the length of one "minute"
        const char* o = lb::env_get("IDLE_UNLOAD_SEC");
        const int64_t minute_ms = o && std::atoi(o) > 0 ? 1000ll * std::atoi(o) : 60000ll;
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(std::min<int64_t>(5000, minute_ms)));
            const int m = s->idle_unload_min.load();
            if (m <= 0 || s->busy.load() || s->bench_running.load() || steady_ms() - s->last_used_ms.load() < m * minute_ms) continue;
            if (unload_model(*s) == 0) log("model released after " + itos(m) + " min without requests");
        }
    }).detach();
}

void handle_chats(Server&, sock_t c, const lbhttp::Request& req) {
    const std::string& m = req.method;
    const std::string& p = req.path;
    const bool one = p.size() > 11;   // "/api/chats/<id>"
    if (one ? (m != "GET" && m != "DELETE") : (m != "GET" && m != "POST")) {
        const char* allow = one ? "Allow: GET, DELETE, OPTIONS\r\n" : "Allow: GET, POST, OPTIONS\r\n";
        send_json(c, 405, error_json({405, "invalid_request_error", one ? "use GET or DELETE" : "use GET or POST", ""}), allow);
        return;
    }
    // Chats are private: reading needs the same loopback / admin-token check as changing them.
    if (!bench_authorized(c, req)) {
        send_error(c, ApiError{403, "permission_error", "chat storage needs a loopback client or a valid X-Admin-Token", ""});
        return;
    }
    std::string id;
    if (one) {
        id = lbhttp::lower_ascii(std::string_view(p).substr(11));
        if (!valid_chat_id(id)) {
            send_error(c, bad_request("chat ids are 1-64 hex characters", "invalid_chat_id"));
            return;
        }
    }
    std::lock_guard<std::mutex> lk(g_chats.mu);
    const bool ok = chats_load_locked();
    if (!ok && (m != "GET" || g_chats.load_error.rfind("no writable", 0) != 0)) {
        send_error(c, ApiError{500, "server_error", g_chats.load_error, "chat_store_unavailable"});
        return;
    }
    auto find = [&](const std::string& want) -> std::ptrdiff_t {
        for (size_t i = 0; i < g_chats.chats.size(); ++i)
            if (json_get_str(g_chats.chats[i], "id") == want) return static_cast<std::ptrdiff_t>(i);
        return -1;
    };

    if (!one && m == "GET") {
        std::vector<const Json*> v;
        for (const Json& ch : g_chats.chats) v.push_back(&ch);
        std::stable_sort(v.begin(), v.end(), [](const Json* a, const Json* b) {
            return json_get_int(*a, "updated") > json_get_int(*b, "updated");
        });
        std::string out = "{\"object\":\"list\",\"data\":[";
        for (size_t i = 0; i < v.size(); ++i) out += (i ? "," : "") + chat_summary_json(*v[i]);
        send_json(c, 200, out + "]}", "Cache-Control: no-store\r\n");
        return;
    }
    if (one && m == "GET") {
        const std::ptrdiff_t k = find(id);
        if (k < 0) {
            send_error(c, ApiError{404, "not_found_error", "no chat " + id, "chat_not_found"});
            return;
        }
        std::string out;
        json_dump(out, g_chats.chats[static_cast<size_t>(k)]);
        send_json(c, 200, out, "Cache-Control: no-store\r\n");
        return;
    }
    if (one) {   // DELETE
        const std::ptrdiff_t k = find(id);
        if (k < 0) {
            send_error(c, ApiError{404, "not_found_error", "no chat " + id, "chat_not_found"});
            return;
        }
        Json removed = std::move(g_chats.chats[static_cast<size_t>(k)]);
        g_chats.chats.erase(g_chats.chats.begin() + k);
        const std::string err = chats_save_locked();
        if (!err.empty()) {
            g_chats.chats.insert(g_chats.chats.begin() + k, std::move(removed));
            send_error(c, ApiError{500, "server_error", err, ""});
            return;
        }
        send_json(c, 200, "{\"id\":" + q(id) + ",\"deleted\":true}", "Cache-Control: no-store\r\n");
        return;
    }

    // POST: upsert one chat.
    Json root;
    std::string jerr;
    if (!lbhttp::parse_json(req.body, root, jerr) || root.kind != Json::Kind::Object) {
        send_error(c, bad_request("body must be a JSON object {id?, title?, messages: [{role, content, meta?}]}", "invalid_json"));
        return;
    }
    if (const Json* v = root.get("id"); v && v->kind != Json::Kind::Null) {
        if (v->kind != Json::Kind::String || !valid_chat_id(lbhttp::lower_ascii(v->str))) {
            send_error(c, bad_request("\"id\" must be 1-64 hex characters", "invalid_chat_id"));
            return;
        }
        id = lbhttp::lower_ascii(v->str);
    }
    Json msgs;
    msgs.kind = Json::Kind::Array;
    if (const Json* v = root.get("messages"); v && v->kind != Json::Kind::Null) {
        if (v->kind != Json::Kind::Array) {
            send_error(c, bad_request("\"messages\" must be an array"));
            return;
        }
        if (v->items.size() > kMaxChatMessages) {
            send_error(c, bad_request("at most " + itos(static_cast<long long>(kMaxChatMessages)) + " messages per chat", "too_many_messages"));
            return;
        }
        for (const Json& it : v->items) {
            const std::string role = json_get_str(it, "role");
            const Json* content = it.get("content");
            if (it.kind != Json::Kind::Object || (role != "system" && role != "user" && role != "assistant" && role != "tool") ||
                !content || (content->kind != Json::Kind::String && content->kind != Json::Kind::Null)) {
                send_error(c, bad_request("each message needs \"role\" (system|user|assistant|tool) and a string \"content\"", "invalid_message"));
                return;
            }
            Json msg;
            msg.kind = Json::Kind::Object;
            json_set(msg, "role", json_str(role));
            json_set(msg, "content", json_str(content->kind == Json::Kind::String ? content->str : std::string()));
            if (const Json* meta = it.get("meta"); meta && meta->kind != Json::Kind::Null) json_set(msg, "meta", *meta);
            msgs.items.push_back(std::move(msg));
        }
    }
    std::string title;
    if (const Json* v = root.get("title"); v && v->kind == Json::Kind::String) title = utf8_prefix(v->str, kMaxChatTitle);
    if (title.empty()) {
        for (const Json& mm : msgs.items)
            if (json_get_str(mm, "role") == "user") {
                title = utf8_prefix(json_get_str(mm, "content"), 60);
                break;
            }
        for (char& ch : title)
            if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
        if (title.empty()) title = "New chat";
    }
    const int64_t now = static_cast<int64_t>(std::time(nullptr));
    std::ptrdiff_t k = id.empty() ? -1 : find(id);
    const bool created_new = k < 0;
    if (created_new && g_chats.chats.size() >= kMaxChats) {
        send_error(c, ApiError{409, "conflict_error", "at most " + itos(static_cast<long long>(kMaxChats)) + " chats; delete old ones first",
                               "too_many_chats"});
        return;
    }
    if (id.empty())
        do id = random_hex16(); while (find(id) >= 0);
    int64_t created = created_new ? json_get_int(root, "created") : json_get_int(g_chats.chats[static_cast<size_t>(k)], "created");
    if (created <= 0) created = now;
    Json chat;
    chat.kind = Json::Kind::Object;
    json_set(chat, "id", json_str(id));
    json_set(chat, "title", json_str(title));
    json_set(chat, "created", json_int(created));
    json_set(chat, "updated", json_int(now));
    json_set(chat, "messages", std::move(msgs));
    Json previous;
    if (created_new) g_chats.chats.push_back(std::move(chat));
    else {
        previous = std::move(g_chats.chats[static_cast<size_t>(k)]);
        g_chats.chats[static_cast<size_t>(k)] = std::move(chat);
    }
    const std::string err = chats_save_locked();
    if (!err.empty()) {
        if (created_new) g_chats.chats.pop_back();
        else g_chats.chats[static_cast<size_t>(k)] = std::move(previous);
        if (err == "too_large") send_error(c, ApiError{413, "invalid_request_error", "the chat store would exceed 64 MiB; delete old chats", "store_full"});
        else send_error(c, ApiError{500, "server_error", err, ""});
        return;
    }
    const Json& saved = created_new ? g_chats.chats.back() : g_chats.chats[static_cast<size_t>(k)];
    std::string out = chat_summary_json(saved);
    out.insert(out.size() - 1, std::string(",\"created_new\":") + (created_new ? "true" : "false"));
    send_json(c, created_new ? 201 : 200, out, "Cache-Control: no-store\r\n");
}

// ---- local benchmark history: <app data>/local_benchmarks.json ---------------------------------

constexpr size_t kMaxLocalBenchRuns = 50;

void handle_bench_local_get(Server& srv, sock_t c) {
    const std::string path = data_file("local_benchmarks.json");
    std::string text;
    {
        std::lock_guard<std::mutex> lk(srv.bench_file_mu);
        if (path.empty() || read_capped_file(path, kMaxServedFile, text) <= 0) text = "[]";
    }
    Json root;
    std::string jerr;
    if (!lbhttp::parse_json(text, root, jerr) || root.kind != Json::Kind::Array) text = "[]";
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
    send_json(c, 200, "{\"object\":\"list\",\"path\":" + (path.empty() ? std::string("null") : q(path)) + ",\"data\":" + text + "}",
              "Cache-Control: no-store\r\n");
}

void handle_bench_local_run(Server& srv, sock_t c, const lbhttp::Request& req) {
    if (!bench_authorized(c, req)) {
        send_error(c, ApiError{403, "permission_error", "benchmark runs need a loopback client or a valid X-Admin-Token", ""});
        return;
    }
    {
        std::lock_guard<std::mutex> lk(srv.info_mu);
        if (!srv.engine) {
            send_error(c, ApiError{503, "server_error", srv.loading.load() ? "a model is being loaded; try again shortly" : "no model loaded",
                                   "model_not_loaded"});
            return;
        }
    }
    bool expected = false;
    if (!srv.bench_running.compare_exchange_strong(expected, true)) {
        send_error(c, ApiError{409, "conflict_error", "a benchmark run is already in progress", "bench_running"});
        return;
    }
    BusyFlag running{srv.bench_running};
    srv.queued++;
    std::unique_lock<std::timed_mutex> lock(srv.gen_mu, std::defer_lock);
    const auto slice = std::chrono::milliseconds(kQueuePollMs);
    bool locked = lock.try_lock_for(slice);
    while (!locked && !lbhttp::peer_gone(c)) locked = lock.try_lock_for(slice);
    srv.queued--;
    if (!locked) return;
    if (!srv.engine) {
        send_error(c, ApiError{503, "server_error", "no model loaded", "model_not_loaded"});
        return;
    }
    std::string quick, isa = srv.isa, model = srv.model_id;
    {
        srv.busy = true;
        BusyFlag busy_flag{srv.busy};
        try {
            quick = run_quick_bench(srv);
        } catch (const std::exception& e) {
            send_error(c, ApiError{500, "server_error", std::string("benchmark failed: ") + e.what(), ""});
            return;
        }
    }
    lock.unlock();

    Json qj;
    std::string jerr;
    lbhttp::parse_json(quick, qj, jerr);
    const auto qnum = [&](const char* k) {
        const Json* v = qj.get(k);
        return v && v->kind == Json::Kind::Number ? num(v->number) : std::string("null");
    };
    const uint64_t peak = veltra::plat::peak_rss_bytes();
    const std::string peak_mb = peak ? itos(static_cast<long long>(peak / (1024 * 1024))) : std::string("null");
    const veltra::HwInfo hw = veltra::detect_hardware();
    std::string gpus = "[";
    for (size_t i = 0; i < hw.gpus.size(); ++i) gpus += (i ? "," : "") + q(hw.gpus[i].name);
    gpus += "]";
    const veltra::cluster::Selection sel = veltra::cluster::current();
    // The quick bench always runs on this machine's engine (CPU in every current build), whatever the compute mode.
    const std::string entry =
        "{\"timestamp\":" + q(iso_utc_now()) + ",\"model\":" + q(model) + ",\"mode\":" + q(sel.mode) +
        ",\"device\":" + q(lb::cuda_backend_built() ? "gpu" : "cpu") + ",\"device_label\":" + q(device_label(isa)) +
        ",\"isa\":" + q(isa) + ",\"threads\":" + itos(srv.opt.threads) + ",\"cpu\":" + q(hw.cpu_brand) +
        ",\"cpu_arch\":" + q(hw.cpu_arch) + ",\"gpus\":" + gpus +
        ",\"decode_tok_s\":" + qnum("decode_tok_s") + ",\"prefill_tok_s\":" + qnum("prefill_tok_s") + ",\"ttft_s\":" + qnum("ttft_s") +
        ",\"peak_rss_mb\":" + peak_mb + ",\"vram_peak_mb\":null" +
        ",\"steps\":[{\"step\":\"decode\",\"tok_s\":" + qnum("decode_tok_s") + ",\"tokens\":" + qnum("decode_tokens") +
        "},{\"step\":\"prefill\",\"tok_s\":" + qnum("prefill_tok_s") + ",\"tokens\":" + qnum("prefill_tokens") +
        ",\"ttft_s\":" + qnum("ttft_s") + "},{\"step\":\"memory\",\"peak_rss_mb\":" + peak_mb + ",\"vram_peak_mb\":null}]" +
        ",\"quick\":" + quick + "}";

    std::string warn;
    const std::string path = data_file("local_benchmarks.json");
    if (path.empty()) warn = "no writable app data directory; result not saved";
    else {
        std::lock_guard<std::mutex> lk(srv.bench_file_mu);
        std::string text;
        const int rc = read_capped_file(path, kMaxServedFile, text);
        Json arr, ej;
        if (rc <= 0 || !lbhttp::parse_json(text, arr, jerr) || arr.kind != Json::Kind::Array) {
            if (rc != 0) log("local_benchmarks.json was unreadable; starting a new history");
            arr = Json{};
            arr.kind = Json::Kind::Array;
        }
        lbhttp::parse_json(entry, ej, jerr);
        arr.items.push_back(std::move(ej));
        if (arr.items.size() > kMaxLocalBenchRuns)
            arr.items.erase(arr.items.begin(), arr.items.end() - static_cast<std::ptrdiff_t>(kMaxLocalBenchRuns));
        std::string out;
        json_dump(out, arr);
        out += '\n';
        if (!write_file_atomic(path, out)) warn = "cannot write " + path;
    }
    log("local benchmark: " + entry);
    std::string body = entry;
    body.insert(body.size() - 1, ",\"saved\":" + std::string(warn.empty() ? "true" : "false") +
                                     (warn.empty() ? std::string() : ",\"warning\":" + q(warn)));
    send_json(c, 200, body, "Cache-Control: no-store\r\n");
}


void route(Server& srv, sock_t c, const lbhttp::Request& req) {
    const std::string& m = req.method;
    const std::string& p = req.path;

    if (m == "OPTIONS") {
        lbhttp::send_response(c, 204, "text/plain", "", "Allow: GET, POST, DELETE, OPTIONS\r\n");
        return;
    }
    // --require-token (desktop app): only the static pages and /health are open; the token itself is never served over
    // HTTP (the WebView2 host injects it), so other local programs and web pages cannot use the API.
    if (!g_require_token.empty() && p != "/" && p != "/index.html" && p != "/desktop" && p != "/health" &&
        p != "/benchmarks" && p != "/arena" && p != "/chat" && p != "/overlay" && !is_ui_asset(p) && !constant_time_equal(req.header("x-veltra-token"), g_require_token)) {
        log("403 " + m + " " + p + ": missing or wrong X-Veltra-Token");
        send_error(c, ApiError{403, "permission_error", "missing or wrong X-Veltra-Token", ""});
        return;
    }
    if (p == "/health") {
        if (m != "GET") {
            send_json(c, 405, error_json({405, "invalid_request_error", "use GET", ""}), "Allow: GET, OPTIONS\r\n");
            return;
        }
        const double uptime = std::chrono::duration<double>(Clock::now() - srv.started).count();
        std::string mid, isa;
        int mctx = 0;
        {
            std::lock_guard<std::mutex> lk(srv.info_mu);
            mid = srv.model_id; isa = srv.isa; mctx = srv.max_ctx;
        }
        send_json(c, 200,
                  "{\"status\":\"ok\",\"model\":" + q(mid) + ",\"isa\":" + q(isa) + ",\"device\":" + q(device_label(isa)) +
                      ",\"busy\":" + (srv.busy.load() ? "true" : "false") + ",\"queued\":" + itos(srv.queued.load()) +
                      ",\"uptime_s\":" + num(uptime) + ",\"requests\":" + itos(static_cast<long long>(srv.requests.load())) +
                      ",\"generations_completed\":" + itos(static_cast<long long>(srv.gens_completed.load())) +
                      ",\"generations_cancelled\":" + itos(static_cast<long long>(srv.gens_cancelled.load())) +
                      ",\"tokens_generated\":" + itos(static_cast<long long>(srv.tokens_out.load())) +
                      ",\"max_context\":" + itos(mctx) + "}");
        return;
    }
    if (p == "/v1/models") {
        if (m != "GET") {
            send_json(c, 405, error_json({405, "invalid_request_error", "use GET", ""}), "Allow: GET, OPTIONS\r\n");
            return;
        }
        std::string mid;
        {
            std::lock_guard<std::mutex> lk(srv.info_mu);
            mid = srv.model_id;
        }
        send_json(c, 200,
                  mid.empty() ? std::string("{\"object\":\"list\",\"data\":[]}")
                              : "{\"object\":\"list\",\"data\":[{\"id\":" + q(mid) +
                                    ",\"object\":\"model\",\"created\":" + itos(srv.started_epoch) +
                                    ",\"owned_by\":\"veltra\"}]}");
        return;
    }
    if (p == "/v1/chat/completions" || p == "/v1/completions") {
        if (m != "POST") {
            send_json(c, 405, error_json({405, "invalid_request_error", "use POST", ""}), "Allow: POST, OPTIONS\r\n");
            return;
        }
        handle_chat(srv, c, req);
        return;
    }
    if (p == "/" || p == "/index.html" || p == "/desktop" || p == "/api/benchmarks" || p == "/benchmarks" || p == "/arena" || p == "/chat" ||
        p == "/api/hardware" || p == "/api/catalog" || p == "/api/status" || p == "/api/models/download/status" || p == "/api/models/load/status" || p == "/overlay" || is_ui_asset(p)) {
        if (m != "GET") {
            send_json(c, 405, error_json({405, "invalid_request_error", "use GET", ""}), "Allow: GET, OPTIONS\r\n");
            return;
        }
        const char* kHtml = "text/html; charset=utf-8";
        if (p == "/desktop" || ((p == "/" || p == "/index.html") && srv.opt.ui_desktop))
            serve_ui(srv, c, "desktop.html", kHtml);
        else if (p == "/overlay")
            serve_ui(srv, c, "overlay.html", kHtml);
        else if (p == "/api/models/load/status")
            send_json(c, 200, load_status_json(srv), "Cache-Control: no-store\r\n");
        else if (is_ui_asset(p))
            serve_ui(srv, c, p.c_str() + 1, p.back() == 's' && p[p.size() - 2] == 'j' ? "text/javascript; charset=utf-8" : "text/css; charset=utf-8");
        else if (p == "/" || p == "/index.html")
            serve_ui(srv, c, "index.html", kHtml);
        else if (p == "/api/benchmarks") {
            if (!srv.opt.bench_json.empty()) serve_file(c, srv.opt.bench_json, "application/json", "no benchmark results yet");
            else serve_ui(srv, c, "benchmarks_live.json", "application/json");   // snapshot compiled into the binary
        } else if (p == "/benchmarks")
            serve_ui(srv, c, "benchmarks.html", kHtml);
        else if (p == "/arena")
            serve_ui(srv, c, "arena.html", kHtml);
        else if (p == "/chat")
            serve_ui(srv, c, "chat.html", kHtml);
        else if (p == "/api/hardware")
            send_json(c, 200, veltra::hardware_json(veltra::detect_hardware(), lb::cuda_backend_built()), "Cache-Control: no-store\r\n");
        else if (p == "/api/catalog") {
            std::string loaded;
            {
                std::lock_guard<std::mutex> lk(srv.info_mu);
                loaded = file_name_of(srv.model_path);
            }
            send_json(c, 200, veltra::catalog_json(veltra::detect_hardware(), srv.opt.models_dir, loaded, lb::cuda_backend_built(), catalog_context(req), local_bench_history(srv), srv.busy.load() || srv.bench_running.load()),
                      "Cache-Control: no-store\r\n");
        } else if (p == "/api/status")
            send_json(c, 200, status_json(srv), "Cache-Control: no-store\r\n");
        else
            send_json(c, 200, veltra::download_status_json(), "Cache-Control: no-store\r\n");
        return;
    }
    if (p == "/api/model/unload") {
        if (m != "POST") {
            send_json(c, 405, error_json({405, "invalid_request_error", "use POST", ""}), "Allow: POST, OPTIONS\r\n");
            return;
        }
        handle_model_unload(srv, c, req);
        return;
    }
    if (p == "/api/tokenize") {
        if (m != "POST") {
            send_json(c, 405, error_json({405, "invalid_request_error", "use POST", ""}), "Allow: POST, OPTIONS\r\n");
            return;
        }
        handle_tokenize(srv, c, req);
        return;
    }
    if (p == "/api/ui-state") {
        handle_ui_state(srv, c, req);
        return;
    }
    if (p == "/api/models/download/cancel") {
        if (m != "POST") {
            send_json(c, 405, error_json({405, "invalid_request_error", "use POST", ""}), "Allow: POST, OPTIONS\r\n");
            return;
        }
        if (!bench_authorized(c, req)) {
            send_error(c, ApiError{403, "permission_error", "downloads need a loopback client or a valid X-Admin-Token", ""});
            return;
        }
        veltra::cancel_download();
        send_json(c, 200, veltra::download_status_json());
        return;
    }
    if (p == "/api/models/load" || p == "/api/models/download") {
        if (m != "POST") {
            send_json(c, 405, error_json({405, "invalid_request_error", "use POST", ""}), "Allow: POST, OPTIONS\r\n");
            return;
        }
        if (p == "/api/models/load") handle_model_load(srv, c, req);
        else handle_model_download(srv, c, req);
        return;
    }
    if (p == "/compare/llama/v1/chat/completions") {
        if (m != "POST") {
            send_json(c, 405, error_json({405, "invalid_request_error", "use POST", ""}), "Allow: POST, OPTIONS\r\n");
            return;
        }
        handle_compare_proxy(srv, c, req);
        return;
    }
    if (p == "/api/benchmarks/run") {
        if (m != "POST") {
            send_json(c, 405, error_json({405, "invalid_request_error", "use POST", ""}), "Allow: POST, OPTIONS\r\n");
            return;
        }
        handle_bench_run(srv, c, req);
        return;
    }
    if (p == "/api/compute") {
        if (m != "GET") {
            send_json(c, 405, error_json({405, "invalid_request_error", "use GET", ""}), "Allow: GET, OPTIONS\r\n");
            return;
        }
        send_json(c, 200, compute_json(srv), "Cache-Control: no-store\r\n");
        return;
    }
    if (p == "/api/compute/select") {
        if (m != "POST") {
            send_json(c, 405, error_json({405, "invalid_request_error", "use POST", ""}), "Allow: POST, OPTIONS\r\n");
            return;
        }
        handle_compute_select(srv, c, req);
        return;
    }
    if (p == "/api/chats" || p.rfind("/api/chats/", 0) == 0) {
        handle_chats(srv, c, req);
        return;
    }
    if (p.rfind("/api/remote/", 0) == 0) {
        handle_remote(srv, c, req);
        return;
    }
    if (srv.opt.relay && p.rfind("/api/relay/", 0) == 0) {
        const veltra::relay::Response r = veltra::relay::handle(req, bench_authorized(c, req));
        lbhttp::send_response(c, r.status, "application/json", r.body, "Cache-Control: no-store\r\n");
        return;
    }
    if (p == "/api/benchmarks/local") {
        if (m == "GET") handle_bench_local_get(srv, c);
        else if (m == "POST") handle_bench_local_run(srv, c, req);
        else send_json(c, 405, error_json({405, "invalid_request_error", "use GET or POST", ""}), "Allow: GET, POST, OPTIONS\r\n");
        return;
    }
    send_error(c, ApiError{404, "not_found_error", "no route for " + m + " " + p, ""});
}

// Owns one connection; closes it on every path, including exceptions.
struct Conn {
    sock_t s;
    bool drain = false;
    ~Conn() { lbhttp::graceful_close(s, drain); }
};

void handle_connection(Server& srv, sock_t c) {
    Conn conn{c, false};
    lbhttp::set_timeouts(c, lbhttp::kRecvTimeoutMs, lbhttp::kSendTimeoutMs);
    lbhttp::set_nodelay(c);

    lbhttp::Request req;
    lbhttp::HttpError err;
    clear_socket_error();
    if (!lbhttp::read_request(c, req, err)) {
        if (!err.status && has_socket_error()) net_log("connection dropped while reading a request: " + lbhttp::last_socket_error());
        if (err.status) {
            conn.drain = true;  // unread request bytes may remain; let the reply get through first
            send_error(c, ApiError{err.status, "invalid_request_error", err.message, ""});
        }
        return;
    }
    srv.requests++;
    log(req.method + " " + req.target);
    route(srv, c, req);
}

// ---- command line ------------------------------------------------------------------------------

enum class ArgResult { Ok, Help, Error };

void usage(std::FILE* f) {
    std::fprintf(f,
                 "usage: veltra-server [--model x.vlt] [--models-dir DIR] [--host 127.0.0.1] [--port 8080] [--threads N] [--no-spec]\n"
                 "  without --model the server starts empty: open http://HOST:PORT/ and load or download a model there\n"
                 "  --models-dir DIR  folder scanned for .vlt / .lbt files and used for downloads (default: <exe dir>/models on Windows, ./models)\n"
                 "  --cuda-selftest   check every CUDA kernel against a CPU reference (plus real expert linears of --model) and exit (CUDA builds)\n"
                 "  --no-browser      do not open the start page in the default browser (default)\n"
                 "  --open-browser    open the start page once listening (Windows only)\n"
                 "  --ui hub|desktop  page served at GET / (default hub; desktop = the desktop app UI, also at GET /desktop)\n"
                 "  --require-token T  every /api/*, /v1/* and /compare/* request (loopback too) needs header X-Veltra-Token: T;\n"
                 "                    CORS is then limited to the server's own origin (veltra.exe desktop mode sets this)\n"
                 "                     [--int8-head] [--exact-head [0|1]] [--head-vocab N] [--head-vocab-file PATH]\n"
                 "  model files: .vlt (or legacy .lbt); env knobs: VELTRA_<NAME>, legacy LB_<NAME> still honoured\n"
                 "  --threads N  size of the compute thread pool (0 = hardware concurrency; VELTRA_THREADS / LB_THREADS env as fallback)\n"
                 "  --int8-head  INT8 output head with int8 activations (halves head bandwidth; native for int8-embedding files)\n"
                 "  --screen-head [0|1|2]  greedy head scored from a 4-bit (1) or 2-bit (2) copy + exact re-scoring of the few surviving rows (int8 heads only; ~4 ms/token faster, probabilistic 6-sigma margin)\n"
                 "  --screen-sigma2 S  margin of the 2-bit screener in sigmas (default 7)\n"
                 "                       greedy requests only (temperature 0 or absent, no top_p < 1 / top_k); sampled requests always score every id exactly\n"
                 "  --exact-head [0|1]  lossless greedy head over all ids (default on for f16 heads on desktop; 0 disables)\n"
                 "  --head-vocab N        only token ids < N (plus eos/special ids 151643..151699 and --head-vocab-file ids) get logits\n"
                 "  --head-vocab-file P   text file of token ids (one per line or comma/space separated) added to the head vocabulary\n"
                 "  --prefix-cache [0|1]  keep the KV state of the last 2 prompts and reuse the longest common token prefix (default on; bit-identical output)\n"
                 "  --mode strict|turbo  turbo = MoE expert pruning in decode (T=1): keep experts j>=--moe-min-k (turbo 6; default 4) only while p[j] >= --moe-alpha (turbo 0.50) * p[0]\n"
                 "                 [--moe-kmax N] [--moe-mass M] [--moe-edge N --moe-edge-k K] [--moe-prefill-prune 0|1]; env VELTRA_MOE_* / LB_MOE_*; default strict = all experts\n"
                 "  --no-spec    disable self-speculative decoding unless a request sets \"speculative\"\n"
                 "  --mtp [0|1]  native MTP drafts (Qwen3.6 MTP block) for greedy requests; lossless, byte-accounted; request field \"mtp\" overrides\n"
                 "  --pareto-max CPU speed preset for MoE models (top-6 experts, screened head, 512-token prefill, prompt lookup); trades a\n"
                 "               little quality for speed. Flags given explicitly win. Off by default.\n"
                 "  --moe-k N / --lookup-n K / --prefill-chunk N / --teal-sparsity P   the individual knobs\n"
                 "  --search-n N test-time search: N sampled branches over one shared prompt, a blind verifier picks the answer (1 = off);\n"
                 "               request fields \"search_n\", \"search_temperature\" (0.65), \"search_top_p\" (0.95), \"seed\"\n"
                 "  --rest [idx] static REST draft index (mmap, scripts/rest_build.py) + calibrated cost gate; lossless, greedy requests only (opt-in)\n"
                 "  --rest-k N   maximum REST draft tokens per verify pass (default 2, capped at 2: verify T <= 3)   --rest-min-gain X   minimum calibrated bytes-adjusted speedup (default 1.30)\n"
                 "  --bench-json PATH  serve PATH at GET /api/benchmarks; POST /api/benchmarks/run (loopback or env VELTRA_ADMIN_TOKEN / LB_ADMIN_TOKEN) updates its \"quick\" entry\n"
                 "  --web-dir DIR      serve DIR/benchmarks.html, DIR/arena.html, DIR/chat.html at GET /benchmarks, /arena, /chat (no other files)\n"
                 "  --compare-upstream URL  enable POST /compare/llama/v1/chat/completions, forwarded to URL (loopback http only,\n"
                 "                     e.g. http://127.0.0.1:8098 = the llama.cpp server of the arena page; timeout 300 s)\n"
                 "  --relay            also serve /api/relay/* (remote GPU test mode, docs/REMOTE_TESTS.md; local use only)\n"
                 "  --relay-only       serve only /api/relay/* and /health, no model (may sit behind an HTTPS reverse proxy);\n"
                 "                     admin routes need env VELTRA_ADMIN_TOKEN (at least 32 characters)\n"
                 "  --relay-results D  folder for received test reports (default ./remote_gpu_results)\n"
                 "  --port 0     let the OS pick a port; the chosen port is printed as 'LISTENING <port>'\n");
}

bool parse_int_arg(const char* s, long lo, long hi, int& out) {
    char* end = nullptr;
    errno = 0;
    const long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || errno != 0 || v < lo || v > hi) return false;
    out = static_cast<int>(v);
    return true;
}

// --compare-upstream http://HOST[:PORT][/]: HOST must be loopback (127.0.0.1, localhost or [::1]); no path.
bool parse_compare_upstream(const std::string& url, std::string& host, int& port, std::string& err) {
    const std::string pre = "http://";
    if (url.rfind(pre, 0) != 0) {
        err = "--compare-upstream must be an http:// URL (e.g. http://127.0.0.1:8098)";
        return false;
    }
    std::string rest = url.substr(pre.size());
    const size_t slash = rest.find('/');
    if (slash != std::string::npos) {
        if (rest.substr(slash) != "/") {
            err = "--compare-upstream takes no path (POST /v1/chat/completions is always used)";
            return false;
        }
        rest.resize(slash);
    }
    std::string h, ps;
    if (!rest.empty() && rest[0] == '[') {
        const size_t close = rest.find(']');
        if (close == std::string::npos || (close + 1 < rest.size() && rest[close + 1] != ':')) {
            err = "--compare-upstream: bad IPv6 address";
            return false;
        }
        h = rest.substr(1, close - 1);
        if (close + 1 < rest.size()) ps = rest.substr(close + 2);
    } else {
        const size_t colon = rest.find(':');
        h = rest.substr(0, colon);
        if (colon != std::string::npos) ps = rest.substr(colon + 1);
    }
    if (h == "localhost") h = "127.0.0.1";
    if (h != "127.0.0.1" && h != "::1") {
        err = "--compare-upstream must be a loopback address (127.0.0.1, localhost or [::1])";
        return false;
    }
    port = 80;
    if (!ps.empty() && !parse_int_arg(ps.c_str(), 1, 65535, port)) {
        err = "--compare-upstream: port must be 1..65535";
        return false;
    }
    host = h;
    return true;
}

ArgResult parse_args(int argc, char** argv, Options& o, std::string& err) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                err = a + " needs a value";
                return nullptr;
            }
            return argv[++i];
        };
        if (a == "--model") {
            const char* v = next();
            if (!v) return ArgResult::Error;
            o.model_path = v;
        } else if (a == "--models-dir") {
            const char* v = next();
            if (!v) return ArgResult::Error;
            o.models_dir = v;
        } else if (a == "--convert-hf") {
            const char* v = next();
            if (!v) return ArgResult::Error;
            o.convert_hf = v;
            v = next();
            if (!v) return ArgResult::Error;
            o.convert_out = v;
        } else if (a == "--ctx") {
            const char* v = next();
            if (!v) return ArgResult::Error;
            o.ctx = std::max(512, std::min(std::atoi(v), 32768));
        } else if (a == "--cuda-selftest") {
            o.cuda_selftest = true;
        } else if (a == "--no-browser") {
            o.open_browser = false;
        } else if (a == "--open-browser") {
            o.open_browser = true;
        } else if (a == "--host") {
            const char* v = next();
            if (!v) return ArgResult::Error;
            o.host = v;
        } else if (a == "--port") {
            const char* v = next();
            if (!v) return ArgResult::Error;
            if (!parse_int_arg(v, 0, 65535, o.port)) {
                err = "--port must be 0..65535";
                return ArgResult::Error;
            }
        } else if (a == "--threads") {
            const char* v = next();
            if (!v) return ArgResult::Error;
            if (!parse_int_arg(v, 0, kMaxThreads, o.threads)) {
                err = "--threads must be 0.." + itos(kMaxThreads);
                return ArgResult::Error;
            }
        } else if (a == "--no-spec") {
            o.no_spec = true;
        } else if (a == "--mtp") {   // bare flag = on; optional 0/1 value
            o.mtp = true;
            if (i + 1 < argc && (std::string(argv[i + 1]) == "0" || std::string(argv[i + 1]) == "1")) o.mtp = argv[++i][0] == '1';
        } else if (a == "--pareto-max") {
            o.pareto_max = true;
        } else if (a == "--moe-k-layers") {
            o.eng.moe_k_layers = i + 1 < argc ? argv[++i] : "";
            if (o.eng.moe_k_layers.empty()) { err = "--moe-k-layers needs a comma separated list"; return ArgResult::Error; }
        } else if (a == "--shadow-full" || a == "--shadow-masks") {   // experimental quality emulation
            const std::string v = i + 1 < argc ? argv[++i] : "";
            if (v.empty()) { err = a + " needs a value"; return ArgResult::Error; }
            if (a == "--shadow-masks") o.eng.shadow_masks = v;
            else o.eng.shadow_full = std::max(0, std::atoi(v.c_str()));
        } else if (a == "--moe-k" || a == "--lookup-n"|| a == "--prefill-chunk" || a == "--teal-sparsity") {
            const char* v = i + 1 < argc ? argv[++i] : "";
            const double x = std::atof(v);
            if (!*v || x < 0 || (a == "--teal-sparsity" ? x > 0.9 : x > 4096)) {
                err = a + " needs a value (--teal-sparsity 0..0.9)";
                return ArgResult::Error;
            }
            if (a == "--moe-k") { o.eng.moe_k = (int)x; o.moe_k_given = true; }
            else if (a == "--lookup-n") o.lookup_n = std::min(16, (int)x);
            else if (a == "--prefill-chunk") o.prefill_chunk = (int)x;
            else o.eng.teal_sparsity = (float)x;
        } else if (a == "--search-n") {
            const char* v = i + 1 < argc ? argv[++i] : "";
            o.search_n = std::atoi(v);
            if (o.search_n < 1 || o.search_n > 16) {
                err = "--search-n must be 1..16";
                return ArgResult::Error;
            }
        } else if (a == "--rest") {   // bare flag = on (index rest.idx next to the cwd); optional index path
            o.rest = true;
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) o.rest_path = argv[++i];
        } else if (a == "--rest-k") {
            const char* v = next();
            if (!v) return ArgResult::Error;
            o.rest_k = std::max(1, std::min(8, std::atoi(v)));
        } else if (a == "--rest-min-gain") {
            const char* v = next();
            if (!v) return ArgResult::Error;
            o.rest_min_gain = (float)std::atof(v);
        } else if (a == "--int8-head" || a == "--exact-head" || a == "--screen-head") {  // bare flag = on; optional 0/1 value
            int v = 1;
            if (i + 1 < argc && (std::string(argv[i + 1]) == "0" || std::string(argv[i + 1]) == "1" || (a == "--screen-head" && std::string(argv[i + 1]) == "2"))) v = argv[++i][0] - '0';
            if (a == "--int8-head") o.eng.int8_head = v == 1; else if (a == "--screen-head") { o.eng.screen_head = v; o.screen_given = true; } else o.eng.exact_head = v;
        } else if (a == "--screen-sigma2") {
            const char* v = next();
            if (!v) return ArgResult::Error;
            o.eng.screen_sigma2 = (float)std::atof(v);
        } else if (a == "--thinking") {   // chat-template enable_thinking (default 1; 0 for evals)
            int v = 1;
            if (i + 1 < argc && (std::string(argv[i + 1]) == "0" || std::string(argv[i + 1]) == "1")) v = argv[++i][0] - '0';
            o.eng.enable_thinking = v;
        } else if (a == "--mode") {   // strict (default) | turbo: MoE expert pruning in decode
            const char* v = next();
            if (!v) return ArgResult::Error;
            if (std::string(v) == "turbo") o.eng.turbo = true;
            else if (std::string(v) != "strict") { err = "--mode must be strict or turbo"; return ArgResult::Error; }
        } else if (a == "--moe-alpha" || a == "--moe-mass" || a == "--moe-min-k" || a == "--moe-kmax" || a == "--moe-edge" || a == "--moe-edge-k" || a == "--moe-prefill-prune") {
            const char* v = next();
            if (!v) return ArgResult::Error;
            if (a == "--moe-alpha") o.eng.moe_alpha = (float)std::atof(v);
            else if (a == "--moe-mass") o.eng.moe_mass = (float)std::atof(v);
            else if (a == "--moe-min-k") o.eng.moe_min_k = std::atoi(v);
            else if (a == "--moe-kmax") o.eng.moe_kmax = std::atoi(v);
            else if (a == "--moe-edge") o.eng.moe_edge = std::atoi(v);
            else if (a == "--moe-edge-k") o.eng.moe_edge_k = std::atoi(v);
            else o.eng.moe_prefill_prune = std::atoi(v);
        } else if (a == "--prefix-cache") {  // bare flag = on; optional 0/1 value
            int v = 1;
            if (i + 1 < argc && (std::string(argv[i + 1]) == "0" || std::string(argv[i + 1]) == "1")) v = argv[++i][0] - '0';
            o.eng.prefix_cache = v;
        } else if (a == "--head-vocab") {
            const char* v = next();
            if (!v) return ArgResult::Error;
            if (!parse_int_arg(v, 0, 10000000, o.eng.head_vocab)) {
                err = "--head-vocab must be a non-negative integer";
                return ArgResult::Error;
            }
        } else if (a == "--head-vocab-file") {
            const char* v = next();
            if (!v) return ArgResult::Error;
            o.eng.head_vocab_file = v;
        } else if (a == "--compare-upstream") {
            const char* v = next();
            if (!v) return ArgResult::Error;
            if (!parse_compare_upstream(v, o.compare_host, o.compare_port, err)) return ArgResult::Error;
        } else if (a == "--bench-json" || a == "--web-dir") {
            const char* v = next();
            if (!v) return ArgResult::Error;
            (a == "--bench-json" ? o.bench_json : o.web_dir) = v;
        } else if (a == "--ui") {
            const char* v = next();
            if (!v) return ArgResult::Error;
            if (std::string(v) != "hub" && std::string(v) != "desktop") {
                err = "--ui must be hub or desktop";
                return ArgResult::Error;
            }
            o.ui_desktop = std::string(v) == "desktop";
        } else if (a == "--relay") {
            o.relay = true;
        } else if (a == "--relay-only") {
            o.relay_only = true;
        } else if (a == "--relay-results") {
            const char* v = next();
            if (!v) return ArgResult::Error;
            o.relay_results = v;
        } else if (a == "--require-token") {
            const char* v = next();
            if (!v) return ArgResult::Error;
            if (std::strlen(v) < 16) {
                err = "--require-token needs at least 16 characters";
                return ArgResult::Error;
            }
            o.require_token = v;
        } else if (a == "-h" || a == "--help") {
            return ArgResult::Help;
        } else {
            err = "unknown argument: " + a;
            return ArgResult::Error;
        }
    }
    return ArgResult::Ok;   // --model is optional: the hub UI (GET /) loads one later
}

// "models/tiny.lbt" -> "tiny"; used as the model id when the file names none.
std::string model_id_from_path(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    std::string base = slash == std::string::npos ? path : path.substr(slash + 1);
    const size_t dot = base.rfind('.');
    if (dot != std::string::npos && dot > 0) base.resize(dot);
    return base.empty() ? "model" : base;
}

void (*g_listen_cb)(int, void*) = nullptr;
void* g_listen_user = nullptr;

}  // namespace

int veltra_request_unload() {
    Server* s = g_server.load();
    return s ? unload_model(*s) : 2;
}

void veltra_set_listen_callback(void (*cb)(int port, void* user), void* user) {
    g_listen_cb = cb;
    g_listen_user = user;
}

int veltra_server_main(int argc, char** argv) {
    Options opt;
    std::string err;
    switch (parse_args(argc, argv, opt, err)) {
        case ArgResult::Ok: break;
        case ArgResult::Help: usage(stdout); return 0;
        case ArgResult::Error:
            if (!err.empty()) std::fprintf(stderr, "veltra-server: %s\n", err.c_str());
            usage(stderr);
            return 2;
    }
    if (opt.pareto_max) {
        using P = lb::ParetoMaxPreset;
        if (!opt.moe_k_given) opt.eng.moe_k = P::moe_k;
        if (!opt.screen_given) opt.eng.screen_head = P::screen_head;
        if (opt.prefill_chunk <= 0) opt.prefill_chunk = P::prefill_chunk;
        if (opt.lookup_n <= 0) opt.lookup_n = P::lookup_n;
        std::fprintf(stderr, "[veltra-server] --pareto-max: moe-k %d, screen-head %d, prefill-chunk %d, lookup-n %d\n", opt.eng.moe_k, opt.eng.screen_head,
                     opt.prefill_chunk, opt.lookup_n);
    }
    lb::set_prefill_chunk(opt.prefill_chunk);

    if (!opt.convert_hf.empty()) {
        std::string cerr;
        if (!veltra::convert_hf_qwen3(opt.convert_hf, opt.convert_out, std::filesystem::path(opt.convert_out).stem().string(), opt.threads, nullptr, &cerr)) {
            std::fprintf(stderr, "veltra-server: convert failed: %s\n", cerr.c_str());
            return 1;
        }
        std::printf("converted to %s\n", opt.convert_out.c_str());
        return 0;
    }

    if (opt.cuda_selftest) {   // docs/GPU_RTX3060_GUIDE.md step 4
        const std::string rep = lb::cuda_selftest(opt.model_path);
        std::fputs(rep.c_str(), stdout);
        return rep.find("all PASS") != std::string::npos ? 0 : 1;
    }
    if (!lbhttp::net_startup(&err)) {
        std::fprintf(stderr, "veltra-server: %s\n", err.c_str());
        return 1;
    }

    veltra::relay::Config relay_cfg;
    relay_cfg.results_dir = opt.relay_results;
    relay_cfg.relay_only = opt.relay_only;
    if (const char* t = lb::env_get("ADMIN_TOKEN")) relay_cfg.admin_token = t;
    if (opt.relay_only) {   // no model, no models folder, no app data folder, no HOME: only the results folder is written
        if (!opt.model_path.empty()) {
            std::fprintf(stderr, "veltra-server: --relay-only does not load a model (drop --model)\n");
            return 2;
        }
        if (!relay_cfg.admin_token.empty() && relay_cfg.admin_token.size() < 32) {
            std::fprintf(stderr, "veltra-server: --relay-only needs VELTRA_ADMIN_TOKEN of at least 32 characters (or unset: admin routes off)\n");
            return 2;
        }
    }
    if (opt.relay || opt.relay_only) {
        const std::string why = veltra::relay::init(relay_cfg);
        if (!why.empty()) {
            std::fprintf(stderr, "veltra-server: %s\n", why.c_str());
            return 1;
        }
    }
    if (opt.relay_only) {
        int relay_port = 0;
        const sock_t l = lbhttp::listen_tcp(opt.host, opt.port, &relay_port, &err);
        if (l == lbhttp::kBadSock) {
            std::fprintf(stderr, "veltra-server: %s\n", err.c_str());
            return 1;
        }
        std::printf("LISTENING %d\n", relay_port);
        std::fflush(stdout);
        if (g_listen_cb) g_listen_cb(relay_port, g_listen_user);
        std::fprintf(stderr, "veltra-server: relay-only on http://%s:%d (admin routes %s)\n", opt.host.c_str(), relay_port,
                     relay_cfg.admin_token.empty() ? "disabled: VELTRA_ADMIN_TOKEN not set" : "need the admin token");
        veltra::relay::serve(l);
    }

    if (opt.models_dir.empty()) {
#if defined(VELTRA_STANDALONE)
        opt.models_dir = veltra::plat::exe_dir() + "/models";   // models\ next to veltra.exe
#else
        opt.models_dir = "models";
#endif
    }
    auto srv = std::make_unique<Server>();
    srv->opt = opt;
    opt.eng.threads = opt.threads;
    lb::engine_default_options() = opt.eng;
    if (opt.ctx > 0) lb::engine_default_options().max_context = opt.ctx;
    else preset_context();   // the saved context slider must apply to the --model loaded below
    if (!opt.model_path.empty()) {
        srv->engine = lb::Engine::load(opt.model_path, &err);
        if (!srv->engine) {
            std::fprintf(stderr, "veltra-server: cannot load %s: %s\n", opt.model_path.c_str(), err.c_str());
            return 1;
        }
    }
    if (opt.rest) {
        if (!lb::rest_load(opt.rest_path, &err)) { std::fprintf(stderr, "veltra-server: %s (REST runs with session lookup only)\n", err.c_str()); }
        else std::fprintf(stderr, "veltra-server: REST index %s: %llu entries, %.1f MB\n", opt.rest_path.c_str(), (unsigned long long)lb::rest_index()->n_entries(), lb::rest_index()->mapped_bytes() / 1e6);
    }
    if (srv->engine) {
        const lb::ModelConfig& cfg = srv->engine->config();
        srv->model_id = cfg.model_name.empty() ? model_id_from_path(opt.model_path) : cfg.model_name;
        srv->isa = srv->engine->isa_name();
        srv->max_ctx = usable_context(*srv->engine);
        srv->model_path = opt.model_path;
        srv->kv_bytes = srv->engine->kv_bytes();
        std::error_code size_ec;
        const auto bytes = std::filesystem::file_size(opt.model_path, size_ec);
        srv->model_bytes = size_ec ? 0 : static_cast<uint64_t>(bytes);
    }
    srv->started_epoch = static_cast<int64_t>(std::time(nullptr));
    g_server = srv.get();
    start_idle_unloader(*srv);

    int bound = 0;
    const sock_t listener = lbhttp::listen_tcp(opt.host, opt.port, &bound, &err);
    if (listener == lbhttp::kBadSock) {
        std::fprintf(stderr, "veltra-server: %s\n", err.c_str());
        return 1;
    }
    g_require_token = opt.require_token;
    {
        Server* host_srv = srv.get();   // lives for the whole process
        veltra::remote::Host host;
        host.run_suite = [host_srv](const veltra::tests::SuiteInfo& s, const std::atomic<bool>& cancel, const veltra::tests::ProgressFn& pf) {
            return run_remote_suite(*host_srv, s, cancel, pf);
        };
        host.hardware_line = [host_srv] { return remote_hardware_line(*host_srv); };
        veltra::remote::set_host(std::move(host));
    }
    if (!opt.require_token.empty()) {   // pin CORS to our own origin: other web pages get no readable answers
        const std::string host = (opt.host == "0.0.0.0" || opt.host == "::") ? std::string("127.0.0.1") : opt.host;
        lbhttp::cors_allow_origin() = "http://" + host + ":" + itos(bound);
    }
    std::printf("LISTENING %d\n", bound);
    std::fflush(stdout);
    if (g_listen_cb) g_listen_cb(bound, g_listen_user);
    if (srv->engine)
        std::fprintf(stderr, "veltra-server: serving '%s' (isa %s, context %d) on http://%s:%d\n", srv->model_id.c_str(),
                     srv->isa.c_str(), srv->max_ctx, opt.host.c_str(), bound);
    else
        std::fprintf(stderr, "veltra-server: no model loaded yet; open http://%s:%d/ (models folder: %s)\n", opt.host.c_str(), bound,
                     opt.models_dir.c_str());
    if (opt.open_browser) {
        const std::string host = (opt.host == "0.0.0.0" || opt.host == "::") ? std::string("127.0.0.1") : opt.host;
        const std::string url = "http://" + host + ":" + itos(bound) + "/";
        if (!veltra::plat::open_url(url)) std::fprintf(stderr, "veltra-server: open %s in your browser\n", url.c_str());
    }

    Server* s = srv.get();  // lives for the whole process
    for (;;) {
        const sock_t c = lbhttp::accept_client(listener);
        if (c == lbhttp::kBadSock) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        if (s->connections.load() >= kMaxConnections) {
            lbhttp::set_timeouts(c, 2000, 2000);
            send_json(c, 503, error_json({503, "server_error", "too many connections", ""}));
            lbhttp::graceful_close(c, false);
            continue;
        }
        s->connections++;
        try {
            std::thread([s, c]() {
                try {
                    handle_connection(*s, c);
                } catch (const std::exception& e) {
                    log(std::string("connection error: ") + e.what());
                } catch (...) {
                    log("connection error");
                }
                s->connections--;
            }).detach();
        } catch (const std::exception& e) {
            s->connections--;
            log(std::string("cannot start a thread: ") + e.what());
            lbhttp::graceful_close(c, false);
        }
    }
}

#if !defined(VELTRA_DESKTOP)
int main(int argc, char** argv) { return veltra_server_main(argc, argv); }
#endif
