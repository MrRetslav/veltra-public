// engine/remote_client.h - the app side of the opt-in remote GPU test mode (docs/REMOTE_TESTS.md).
//
// Outbound requests only: while the user has the toggle on, a background thread registers at the relay and polls it.
// The relay is untrusted: all it can hand over is a job id and a suite id from the compiled-in allowlist. Nothing runs
// before the local user approved it through decide() (the local API), unless that user granted "yes for this session":
// that only repeats the approved suite, at most 3 times, not within 10 minutes of the previous test, for at most 1 hour
// (memory only; also ends with the toggle, the process or the local calendar day). A decision is bound to the pending
// job's client-made nonce and suite. The toggle is never persisted.
#pragma once
#include <atomic>
#include <functional>
#include <string>

#include "test_runner.h"

namespace veltra::remote {

struct Host {
    // Runs an approved suite on this process's engine (takes the generation lock).
    std::function<tests::Report(const tests::SuiteInfo&, const std::atomic<bool>& cancel, const tests::ProgressFn&)> run_suite;
    // One line for the consent popup: detected GPU and the device that will really run the model.
    std::function<std::string()> hardware_line;
};
void set_host(Host host);

// Called (on any thread, no lock held) with true when a job starts waiting for the user's decision and with false when
// that wait ends. The Windows app uses it to flash the window and show a notification.
void set_pending_hook(std::function<void(bool pending)> hook);

// The local API behind /api/remote/*. Each returns the HTTP status; on != 200 `err` says why.
std::string status_json();
int configure(const std::string& body, std::string& err);   // {"enabled"?:bool,"relay_url"?:string}
int decide(const std::string& body, std::string& err);      // {"job_id","nonce","suite","decision":"yes"|"yes_session"|"no"}
int cancel(std::string& err);                              // cancels the running suite

}  // namespace veltra::remote
