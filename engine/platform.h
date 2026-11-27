// engine/platform.h - the OS layer of Veltra (Linux / Android POSIX and Windows _WIN32).
// Declarations only: no OS header leaks into the engine translation units (windows.h macros stay
// inside engine/platform.cpp). Socket *types* live in engine/http.h (Winsock2 vs BSD sockets);
// process-wide socket start-up (WSAStartup / SIGPIPE) is net_init() below.
//
//   memory-mapped files : mmap / munmap            <-> CreateFileA / CreateFileMappingA / MapViewOfFile
//   sockets             : SIGPIPE ignore           <-> WSAStartup / WSACleanup (closesocket in http.h)
//   threads + affinity  : pthread_setaffinity_np   <-> SetThreadAffinityMask
//   misc                : exe directory, open a URL in the default browser (ShellExecuteA / no-op)
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace veltra::plat {

// Read-only zero-copy mapping of a whole file. Pages fault in lazily unless `populate` is set.
struct FileMap {
    const uint8_t* data = nullptr;
    size_t size = 0;
    void* h_file = nullptr;   // Win32 HANDLEs (unused on POSIX)
    void* h_map = nullptr;
};
struct MapHints {
    bool populate = false;          // POSIX MAP_POPULATE (ignored on Windows)
    size_t willneed_below = 0;      // madvise(WILLNEED) when the file is smaller than this (0 = never)
};
bool map_file_ro(const std::string& path, FileMap* out, std::string* err, MapHints hints = {});
void unmap_file(FileMap* m);   // idempotent

// Process-wide network start-up (WSAStartup on Windows, SIGPIPE ignore on POSIX). Idempotent.
bool net_init(std::string* err);

// Pins the calling thread to one logical CPU. Returns false when unsupported or refused.
bool pin_current_thread(int cpu);
int hardware_threads();

// Directory of the running executable (no trailing slash); "." when unknown.
std::string exe_dir();
// Opens url in the user's default browser (Windows: ShellExecuteA "open"). No-op elsewhere; returns false then.
bool open_url(const std::string& url);

// Per-user data directory of the app, created on first call (and re-checked on every call):
// env VELTRA_DATA_DIR when set, else %APPDATA%\Veltra (Windows), $XDG_DATA_HOME/veltra or ~/.local/share/veltra.
// "" when it cannot be created.
std::string app_data_dir();
// Peak resident set size of this process in bytes (Linux VmHWM, Windows PeakWorkingSetSize); 0 when unknown.
uint64_t peak_rss_bytes();

}  // namespace veltra::plat
