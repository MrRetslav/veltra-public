// engine/server_main.h - the HTTP server's entry point as a function, so the Windows desktop app (engine/win_app.cpp)
// can run it on a background thread in-process. veltra-server's main() just calls veltra_server_main().
#pragma once

// Runs the server with veltra-server's command line (argv[0] is ignored). Returns only on start-up errors (exit code).
int veltra_server_main(int argc, char** argv);
// Called once, from the server thread, right after the listening socket is bound (port = the real port, also for --port 0).
void veltra_set_listen_callback(void (*cb)(int port, void* user), void* user);
// Releases the loaded model (the next request loads it again). Callable from any thread of the same process.
// 0 = released, 1 = busy generating or loading, 2 = nothing loaded / server not up yet.
int veltra_request_unload();
