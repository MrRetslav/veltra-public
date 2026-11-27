// engine/platform.cpp - see platform.h. The only engine translation unit that includes OS headers for these calls.
#include "platform.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <winsock2.h>
#  include <windows.h>
#  include <shellapi.h>
#  ifndef PSAPI_VERSION
#    define PSAPI_VERSION 2   // GetProcessMemoryInfo -> K32GetProcessMemoryInfo (kernel32)
#  endif
#  include <psapi.h>
#else
#  include <cerrno>
#  include <csignal>
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#  if defined(__linux__)
#    include <pthread.h>
#    include <sched.h>
#  endif
#endif

namespace veltra::plat {

#if defined(_WIN32)

bool map_file_ro(const std::string& path, FileMap* out, std::string* err, MapHints) {
    *out = FileMap{};
    HANDLE f = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) { if (err) *err = "cannot open " + path; return false; }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(f, &sz) || sz.QuadPart == 0) { CloseHandle(f); if (err) *err = "empty or unreadable file " + path; return false; }
    HANDLE m = CreateFileMappingA(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!m) { CloseHandle(f); if (err) *err = "CreateFileMapping failed"; return false; }
    void* p = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (!p) { CloseHandle(m); CloseHandle(f); if (err) *err = "MapViewOfFile failed"; return false; }
    out->data = (const uint8_t*)p; out->size = (size_t)sz.QuadPart; out->h_file = f; out->h_map = m;
    return true;
}
void unmap_file(FileMap* m) {
    if (m->data) UnmapViewOfFile(m->data);
    if (m->h_map) CloseHandle((HANDLE)m->h_map);
    if (m->h_file) CloseHandle((HANDLE)m->h_file);
    *m = FileMap{};
}

bool net_init(std::string* err) {
    static int rc = [] { WSADATA wsa; return WSAStartup(MAKEWORD(2, 2), &wsa); }();
    if (rc != 0 && err) *err = "WSAStartup failed: " + std::to_string(rc);
    return rc == 0;
}

bool pin_current_thread(int cpu) {
    if (cpu < 0 || cpu >= 64) return false;   // one processor group (<= 64 logical CPUs)
    return SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpu) != 0;
}

std::string exe_dir() {
    char buf[MAX_PATH];
    const DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return ".";
    std::string s(buf, n);
    const size_t k = s.find_last_of("\\/");
    return k == std::string::npos ? "." : s.substr(0, k);
}

bool open_url(const std::string& url) {
    return (INT_PTR)ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL) > 32;
}

namespace {
bool make_dirs(const std::string& path) {   // mkdir -p
    for (size_t i = 1; i <= path.size(); i++) {
        if (i < path.size() && path[i] != '\\' && path[i] != '/') continue;
        const std::string part = path.substr(0, i);
        if (part.size() == 2 && part[1] == ':') continue;   // "C:"
        if (!CreateDirectoryA(part.c_str(), nullptr) && GetLastError() != ERROR_ALREADY_EXISTS) return false;
    }
    const DWORD a = GetFileAttributesA(path.c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
}
}  // namespace

std::string app_data_dir() {
    std::string d;
    const char* o = std::getenv("VELTRA_DATA_DIR");
    if (o && *o) d = o;
    else {
        const char* a = std::getenv("APPDATA");
        if (a && *a) d = std::string(a) + "\\Veltra";
        else d = exe_dir() + "\\data";
    }
    return make_dirs(d) ? d : std::string();
}

uint64_t peak_rss_bytes() {
    PROCESS_MEMORY_COUNTERS pmc;
    std::memset(&pmc, 0, sizeof pmc);
    pmc.cb = sizeof pmc;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) return 0;
    return (uint64_t)pmc.PeakWorkingSetSize;
}

#else  // POSIX

bool map_file_ro(const std::string& path, FileMap* out, std::string* err, MapHints hints) {
    *out = FileMap{};
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { if (err) *err = "cannot open " + path; return false; }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size == 0) { ::close(fd); if (err) *err = "empty or unreadable file " + path; return false; }
    int flags = MAP_PRIVATE;
#ifdef MAP_POPULATE
    if (hints.populate) flags |= MAP_POPULATE;
#endif
    void* p = mmap(nullptr, (size_t)st.st_size, PROT_READ, flags, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) { if (err) *err = "mmap failed"; return false; }
#ifdef MADV_WILLNEED
    if ((size_t)st.st_size < hints.willneed_below) madvise(p, (size_t)st.st_size, MADV_WILLNEED);
#endif
    out->data = (const uint8_t*)p; out->size = (size_t)st.st_size;
    return true;
}
void unmap_file(FileMap* m) {
    if (m->data) munmap((void*)m->data, m->size);
    *m = FileMap{};
}

bool net_init(std::string*) {
    // A write to a client that vanished must fail with EPIPE, not kill the process.
    std::signal(SIGPIPE, SIG_IGN);
    return true;
}

bool pin_current_thread(int cpu) {
#if defined(__linux__)
    if (cpu < 0 || cpu >= CPU_SETSIZE) return false;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
    (void)cpu;
    return false;
#endif
}

std::string exe_dir() {
#if defined(__linux__)
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        std::string s(buf, (size_t)n);
        const size_t k = s.find_last_of('/');
        if (k != std::string::npos) return s.substr(0, k);
    }
#endif
    return ".";
}

bool open_url(const std::string&) { return false; }

namespace {
bool make_dirs(const std::string& path) {   // mkdir -p
    for (size_t i = 1; i <= path.size(); i++) {
        if (i < path.size() && path[i] != '/') continue;
        const std::string part = path.substr(0, i);
        if (::mkdir(part.c_str(), 0700) != 0 && errno != EEXIST) return false;
    }
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}
}  // namespace

std::string app_data_dir() {
    std::string d;
    const char* o = std::getenv("VELTRA_DATA_DIR");
    const char* x = std::getenv("XDG_DATA_HOME");
    const char* h = std::getenv("HOME");
    if (o && *o) d = o;
    else if (x && *x == '/') d = std::string(x) + "/veltra";
    else if (h && *h) d = std::string(h) + "/.local/share/veltra";
    else return {};
    while (d.size() > 1 && d.back() == '/') d.pop_back();
    return make_dirs(d) ? d : std::string();
}

uint64_t peak_rss_bytes() {
#if defined(__linux__)
    std::FILE* f = std::fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    uint64_t kb = 0;
    while (std::fgets(line, sizeof line, f)) {
        if (std::strncmp(line, "VmHWM:", 6) == 0) {
            kb = std::strtoull(line + 6, nullptr, 10);
            break;
        }
    }
    std::fclose(f);
    return kb * 1024;
#else
    return 0;
#endif
}

#endif

int hardware_threads() {
    const unsigned n = std::thread::hardware_concurrency();
    return n ? (int)n : 1;
}

}  // namespace veltra::plat
