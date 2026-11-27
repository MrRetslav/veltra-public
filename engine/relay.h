// engine/relay.h - the relay of the opt-in remote GPU test mode (docs/REMOTE_TESTS.md).
//
// The relay only passes a suite id from the compiled-in allowlist (engine/test_runner.cpp) from the admin to a
// registered client and stores the JSON report the client sends back. It cannot make a client run anything else:
// there is no command, path, URL, flag or parameter in the protocol.
//
//   POST /api/relay/register   (X-Veltra-Node-Token)            -> {"code":"VLT-123456",...}
//   GET  /api/relay/poll?code= (X-Veltra-Node-Token)            -> {"job":null | {"id","suite"}}
//   POST /api/relay/progress   (X-Veltra-Node-Token) {code, job_id, state:"pending"|"running", progress?}
//   POST /api/relay/result     (X-Veltra-Node-Token) {code, job_id, status, reason?, report?}
//   POST /api/relay/dispatch   (admin) {code, suite}            -> {"job_id",...}
//   GET  /api/relay/job?id=    (admin)                          -> status + progress + result
//   GET  /api/relay/results    (admin)                          -> summaries of the saved reports
//
// Two ways to run it: `--relay` adds these routes to a normal server (local use only), `--relay-only` serves nothing but
// these routes plus GET /health through its own hardened HTTP reader (serve()); that is the mode that may sit behind a
// public HTTPS reverse proxy.
#pragma once
#include <string>

#include "http.h"

namespace veltra::relay {

struct Config {
    std::string results_dir = "remote_gpu_results";
    std::string admin_token;   // env VELTRA_ADMIN_TOKEN ("" = none)
    bool relay_only = false;
};

// Creates the results directory. "" or the reason the relay cannot start.
std::string init(const Config& cfg);

struct Response {
    int status = 200;
    std::string body;
};

// One /api/relay/* request. `local_admin`: the surrounding server already trusts this request as its local admin
// (--relay on a normal server only; ignored in relay-only mode, which has no loopback trust).
Response handle(const lbhttp::Request& req, bool local_admin);

// --relay-only accept loop (never returns): strict request reader, per-route body limits, a total read deadline,
// a bounded number of connections. Logs only method, route name, pairing code and status.
[[noreturn]] void serve(lbhttp::sock_t listener);

}  // namespace veltra::relay
