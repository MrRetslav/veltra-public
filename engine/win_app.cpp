// engine/win_app.cpp - veltra.exe as a native Windows desktop app (only built for the Windows `veltra` target).
//
//   veltra.exe [server args]        Win32 window hosting the UI in Microsoft Edge WebView2; the engine + HTTP server run
//                                   in-process on a background thread, bound to 127.0.0.1 on an ephemeral port, with
//                                   --ui desktop --require-token <random> (the token reaches the page through
//                                   WebView2's AddScriptToExecuteOnDocumentCreated, or in the browser fallback through
//                                   the #veltra_token= URL fragment; the server never serves it).
//   veltra.exe --headless [args]    no window: exactly veltra-server (console output; attaches to the parent console).
//
// WebView2 without WebView2Loader.dll: the Evergreen runtime's location comes from the EdgeUpdate registry keys and its
// EBWebView\<arch>\EmbeddedBrowserWebView.dll exports CreateWebViewEnvironmentWithOptionsInternal, which is what the
// official loader calls (same approach as the open-source OpenWebView2Loader). A WebView2Loader.dll next to the exe is
// used as a second option. Without any runtime: a Dutch message box + the UI in the default browser (token still required).
//
// Page <-> host messages (window.chrome.webview.postMessage(obj) / addEventListener('message', e => e.data)):
//   {"type":"pick_model"}  -> {"type":"picked_model","path":"C:\\..."}  | {"type":"picked_model","cancelled":true}
//   {"type":"pick_folder"} -> {"type":"picked_folder","path":"C:\\..."} | {"type":"picked_folder","cancelled":true}
//   {"type":"hello"}       -> window_state + hotkey_state
//   {"type":"window","action":"minimize"|"maximize"|"close"|"drag"|"resize_top"|"resize_topleft"|"resize_topright"}
//                          <- {"type":"window_state","maximized":bool}
//   {"type":"hotkey","enabled":bool}   <- {"type":"hotkey_state","enabled":bool,"registered":bool}   (Alt+Space overlay)
//   overlay page: {"type":"overlay","action":"hide"|"resize"|"open_main","height":px,"chat":id}
//                 <- {"type":"overlay_shown"};  main page <- {"type":"open_chat","id":id}
// Window: frameless (the page draws the title bar; app-region: drag, or the "drag" message on old runtimes), position kept
// in <app data>\window_state.json, closing hides to the tray (one instance only: a second start shows the running one).
// Env: VELTRA_DEVTOOLS=1 enables the dev tools, VELTRA_LOG_FILE=path appends stdout/stderr (server log) to a file.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <windows.h>
#include <bcrypt.h>
#include <dwmapi.h>
#include <fcntl.h>
#include <io.h>
#include <objbase.h>
#include <shellapi.h>
#include <shobjidl.h>
#include <windowsx.h>

#include "WebView2.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cwchar>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include "json.h"
#include "platform.h"
#include "remote_client.h"
#include "server_main.h"

namespace {

constexpr UINT WM_APP_LISTENING = WM_APP + 1;   // wParam = bound port
constexpr UINT WM_APP_SERVER_EXIT = WM_APP + 2; // wParam = veltra_server_main exit code
constexpr UINT WM_APP_PICK = WM_APP + 3;        // wParam = 0 model file, 1 folder
constexpr UINT WM_APP_REMOTE = WM_APP + 4;      // wParam = 1 a remote test request waits for the user, 0 that ended
constexpr UINT WM_APP_TRAY = WM_APP + 5;        // tray icon callback of that notification
constexpr UINT kTrayId = 1;
constexpr UINT WM_APP_SHOW = WM_APP + 6;        // a second veltra.exe asks the running one to show itself
constexpr UINT WM_APP_UNLOADED = WM_APP + 7;    // wParam = veltra_request_unload() result
constexpr int kHotkeyId = 1;
constexpr UINT kMenuOpen = 101, kMenuUnload = 102, kMenuQuit = 103;
constexpr int kOverlayW = 680, kOverlayMinH = 96, kOverlayMaxH = 540;   // DIPs
constexpr DWORD kDwmaCornerPreference = 33;     // DWMWA_WINDOW_CORNER_PREFERENCE (Windows 11)
const wchar_t kWindowClass[] = L"VeltraDesktopWindow";
const wchar_t kOverlayClass[] = L"VeltraOverlayWindow";
const wchar_t kMutexName[] = L"Local\\VeltraDesktopSingleInstance";
constexpr COLORREF kBg = RGB(0x07, 0x07, 0x0C);   // the page's --base
// window backdrop; the DWM constants are defined here because SDK headers may be old
constexpr DWORD kDwmaBackdrop = 38;                 // DWMWA_SYSTEMBACKDROP_TYPE (build >= 22621)
constexpr DWORD kDwmaCaptionColor = 35;             // DWMWA_CAPTION_COLOR (Windows 11)
constexpr COLORREF kDwmColorDefault = 0xFFFFFFFF;
constexpr DWORD kMinBackdropBuild = 22621;          // Windows 11 22H2
constexpr int kMinW = 1100, kMinH = 720;        // DIPs
constexpr int kInitW = 1280, kInitH = 800;      // DIPs
const wchar_t kInstallUrl[] = L"https://go.microsoft.com/fwlink/p/?LinkId=2124703";

// ---- strings ------------------------------------------------------------------------------------------------------

std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

std::string narrow(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    std::string s(static_cast<size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), s.data(), n, nullptr, nullptr);
    return s;
}

// JSON string literal (UTF-16 in, UTF-16 out): quotes, backslashes and control characters escaped.
std::wstring json_quote(const std::wstring& s) {
    std::wstring o = L"\"";
    for (wchar_t ch : s) {
        if (ch == L'"' || ch == L'\\') {
            o += L'\\';
            o += ch;
        } else if (ch < 0x20) {
            wchar_t buf[8];
            swprintf(buf, 8, L"\\u%04x", static_cast<unsigned>(ch));
            o += buf;
        } else {
            o += ch;
        }
    }
    return o + L"\"";
}

std::wstring take_co_string(LPWSTR p) {   // CoTaskMem string -> std::wstring (frees it)
    std::wstring s = p ? p : L"";
    if (p) CoTaskMemFree(p);
    return s;
}

const char* env(const char* name) {
    const char* v = std::getenv(name);
    return v && *v ? v : nullptr;
}

void dlog(const std::string& msg) {
    const std::string line = "[veltra-desktop] " + msg + "\n";
    std::fwrite(line.data(), 1, line.size(), stderr);   // one write: no interleaving with the server threads
}

// ---- console / log file -------------------------------------------------------------------------------------------

// The C runtime of a GUI-subsystem program does not always wire fd 1/2 to inherited pipe/file handles (stderr stayed
// silent in tests): bind an OS handle (duplicated) to the fd explicitly.
void bind_handle(HANDLE h, int fd, FILE* f) {
    HANDLE dup = nullptr;
    if (!DuplicateHandle(GetCurrentProcess(), h, GetCurrentProcess(), &dup, 0, FALSE, DUPLICATE_SAME_ACCESS)) return;
    const int nfd = _open_osfhandle(reinterpret_cast<intptr_t>(dup), _O_TEXT);
    if (nfd < 0) {
        CloseHandle(dup);
        return;
    }
    std::fflush(f);
    // a GUI-subsystem program started without redirection has no stdout / stderr: give the FILE a real descriptor first
    if (_fileno(f) < 0 && !std::freopen("NUL", "w", f)) {
        _close(nfd);
        return;
    }
    fd = _fileno(f);
    if (nfd != fd) {
        _dup2(nfd, fd);
        _close(nfd);
    }
    std::setvbuf(f, nullptr, _IONBF, 0);   // Windows has no real line buffering: unbuffered, or log lines stay in memory
}

// VELTRA_LOG_FILE=path: stdout + stderr appended to that file (both modes). One append-only handle, so lines written
// by the UI thread and the server threads never overwrite each other.
bool redirect_to_log_file() {
    const char* path = env("VELTRA_LOG_FILE");
    if (!path) return false;
    const HANDLE h = CreateFileW(widen(path).c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS,
                                 FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    bind_handle(h, 1, stdout);
    bind_handle(h, 2, stderr);
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    CloseHandle(h);
    return true;
}

bool std_handle_usable(DWORD which) {
    const HANDLE h = GetStdHandle(which);
    return h && h != INVALID_HANDLE_VALUE && GetFileType(h) != FILE_TYPE_UNKNOWN;
}

// The exe is a GUI-subsystem program: for --headless, keep redirected handles (pipes/files) as they are, otherwise
// attach to the parent's console (cmd / PowerShell) or open a new one (started from Explorer).
void attach_console() {
    if (redirect_to_log_file()) return;
    if (std_handle_usable(STD_OUTPUT_HANDLE) && std_handle_usable(STD_ERROR_HANDLE)) {
        bind_handle(GetStdHandle(STD_OUTPUT_HANDLE), 1, stdout);
        bind_handle(GetStdHandle(STD_ERROR_HANDLE), 2, stderr);
        return;
    }
    if (!AttachConsole(ATTACH_PARENT_PROCESS) && !AllocConsole()) return;
    std::freopen("CONOUT$", "w", stdout);
    std::freopen("CONOUT$", "w", stderr);
    std::freopen("CONIN$", "r", stdin);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    SetConsoleOutputCP(CP_UTF8);
}

// ---- random token -------------------------------------------------------------------------------------------------

std::string random_token_hex32() {
    unsigned char b[16] = {};
    if (BCryptGenRandom(nullptr, b, sizeof b, BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0) {
        // never expected; still unpredictable enough for a loopback-only session token
        LARGE_INTEGER t;
        QueryPerformanceCounter(&t);
        unsigned long long x = static_cast<unsigned long long>(t.QuadPart) ^ (static_cast<unsigned long long>(GetCurrentProcessId()) << 32) ^
                               reinterpret_cast<uintptr_t>(&t);
        for (unsigned char& c : b) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            c = static_cast<unsigned char>(x);
        }
    }
    static const char* hex = "0123456789abcdef";
    std::string s;
    for (unsigned char c : b) {
        s += hex[c >> 4];
        s += hex[c & 15];
    }
    return s;
}

// ---- WebView2 runtime discovery + loader --------------------------------------------------------------------------

const wchar_t kRuntimeClientKey[] = L"SOFTWARE\\Microsoft\\EdgeUpdate\\Clients\\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}";

bool reg_string(HKEY root, const wchar_t* sub, REGSAM view, const wchar_t* name, std::wstring* out) {
    HKEY k;
    if (RegOpenKeyExW(root, sub, 0, KEY_READ | view, &k) != ERROR_SUCCESS) return false;
    wchar_t buf[1024];
    DWORD type = 0, size = sizeof(buf) - sizeof(wchar_t);
    const LSTATUS st = RegQueryValueExW(k, name, nullptr, &type, reinterpret_cast<BYTE*>(buf), &size);
    RegCloseKey(k);
    if (st != ERROR_SUCCESS || (type != REG_SZ && type != REG_EXPAND_SZ)) return false;
    buf[size / sizeof(wchar_t)] = 0;
    *out = buf;
    return !out->empty();
}

bool version_less(const std::wstring& a, const std::wstring& b) {
    unsigned va[4] = {}, vb[4] = {};
    swscanf(a.c_str(), L"%u.%u.%u.%u", &va[0], &va[1], &va[2], &va[3]);
    swscanf(b.c_str(), L"%u.%u.%u.%u", &vb[0], &vb[1], &vb[2], &vb[3]);
    for (int i = 0; i < 4; ++i)
        if (va[i] != vb[i]) return va[i] < vb[i];
    return false;
}

bool file_exists(const std::wstring& p) {
    const DWORD a = GetFileAttributesW(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

std::wstring known_env_dir(const wchar_t* var, const wchar_t* tail) {
    wchar_t buf[MAX_PATH * 2];
    const DWORD n = GetEnvironmentVariableW(var, buf, MAX_PATH * 2);
    if (!n || n >= MAX_PATH * 2) return {};
    return std::wstring(buf) + tail;
}

// Full path of the installed Evergreen runtime's EmbeddedBrowserWebView.dll (highest version of the machine-wide and
// per-user installs), or "" when none is installed.
std::wstring find_runtime_dll(std::wstring* version) {
    struct Cand { HKEY root; REGSAM view; std::wstring fallback_dir; };
    const Cand cands[] = {
        {HKEY_LOCAL_MACHINE, KEY_WOW64_32KEY, known_env_dir(L"ProgramFiles(x86)", L"\\Microsoft\\EdgeWebView\\Application")},
        {HKEY_LOCAL_MACHINE, KEY_WOW64_64KEY, known_env_dir(L"ProgramFiles(x86)", L"\\Microsoft\\EdgeWebView\\Application")},
        {HKEY_CURRENT_USER, 0, known_env_dir(L"LOCALAPPDATA", L"\\Microsoft\\EdgeWebView\\Application")},
    };
#if defined(_M_ARM64) || defined(__aarch64__)
    const wchar_t* arch = L"arm64";
#else
    const wchar_t* arch = L"x64";
#endif
    std::wstring best, best_ver;
    for (const Cand& c : cands) {
        std::wstring pv, loc;
        if (!reg_string(c.root, kRuntimeClientKey, c.view, L"pv", &pv) || pv == L"0.0.0.0") continue;
        if (!reg_string(c.root, kRuntimeClientKey, c.view, L"location", &loc)) loc = c.fallback_dir;
        if (loc.empty()) continue;
        const std::wstring dll = loc + L"\\" + pv + L"\\EBWebView\\" + arch + L"\\EmbeddedBrowserWebView.dll";
        if (!file_exists(dll)) continue;
        if (best.empty() || version_less(best_ver, pv)) {
            best = dll;
            best_ver = pv;
        }
    }
    if (version) *version = best_ver;
    return best;
}

using CreateEnvInternalFn = HRESULT(STDMETHODCALLTYPE*)(bool checkRunningInstance, int runtimeType, PCWSTR userDataFolder,
                                                        ICoreWebView2EnvironmentOptions* options,
                                                        ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler* handler);
using CreateEnvWithOptionsFn = HRESULT(STDMETHODCALLTYPE*)(PCWSTR browserExecutableFolder, PCWSTR userDataFolder,
                                                           ICoreWebView2EnvironmentOptions* options,
                                                           ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler* handler);

struct Loader {
    CreateEnvInternalFn internal = nullptr;   // EmbeddedBrowserWebView.dll of the installed runtime
    CreateEnvWithOptionsFn official = nullptr; // WebView2Loader.dll next to veltra.exe
    std::wstring what;                         // for the log

    bool available() const { return internal || official; }
    HRESULT create(const std::wstring& udf, ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler* h) const {
        if (internal) return internal(true, 0, udf.c_str(), nullptr, h);
        if (official) return official(nullptr, udf.c_str(), nullptr, h);
        return E_NOTIMPL;
    }
};

Loader find_loader() {
    Loader l;
    std::wstring ver;
    const std::wstring dll = find_runtime_dll(&ver);
    if (!dll.empty()) {
        // LOAD_WITH_ALTERED_SEARCH_PATH: the DLL's own folder resolves its dependencies
        if (HMODULE m = LoadLibraryExW(dll.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH)) {
            l.internal = reinterpret_cast<CreateEnvInternalFn>(
                reinterpret_cast<void*>(GetProcAddress(m, "CreateWebViewEnvironmentWithOptionsInternal")));
            if (l.internal) l.what = L"runtime " + ver + L" (" + dll + L")";
        }
    }
    if (!l.internal) {
        const std::wstring side = widen(veltra::plat::exe_dir()) + L"\\WebView2Loader.dll";
        if (file_exists(side))
            if (HMODULE m = LoadLibraryW(side.c_str())) {
                l.official = reinterpret_cast<CreateEnvWithOptionsFn>(
                    reinterpret_cast<void*>(GetProcAddress(m, "CreateCoreWebView2EnvironmentWithOptions")));
                if (l.official) l.what = side;
            }
    }
    return l;
}

// ---- tiny COM callback objects ------------------------------------------------------------------------------------

template <class I, class Sig>
class Callback;
template <class I, class C, class... A>
class Callback<I, HRESULT (C::*)(A...)> final : public I {
public:
    Callback(const IID& iid, std::function<HRESULT(A...)> f) : iid_(iid), f_(std::move(f)) {}
    ULONG STDMETHODCALLTYPE AddRef() override { return static_cast<ULONG>(InterlockedIncrement(&ref_)); }
    ULONG STDMETHODCALLTYPE Release() override {
        const LONG r = InterlockedDecrement(&ref_);
        if (r == 0) delete this;
        return static_cast<ULONG>(r);
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        if (!out) return E_POINTER;
        if (IsEqualIID(riid, IID_IUnknown) || IsEqualIID(riid, iid_)) {
            *out = static_cast<I*>(this);
            AddRef();
            return S_OK;
        }
        *out = nullptr;
        return E_NOINTERFACE;
    }
    HRESULT STDMETHODCALLTYPE Invoke(A... a) override { return f_(a...); }

private:
    LONG ref_ = 1;
    IID iid_;
    std::function<HRESULT(A...)> f_;
};

// make<ICoreWebView2XxxHandler>(IID_ICoreWebView2XxxHandler, lambda): a new object with one reference (the caller's).
template <class I, class F>
I* make(const IID& iid, F&& f) {
    return new Callback<I, decltype(&I::Invoke)>(iid, std::forward<F>(f));
}

template <class T>
void release(T*& p) {
    if (p) p->Release();
    p = nullptr;
}

// ---- the app ------------------------------------------------------------------------------------------------------

enum class Backdrop { None, Mica, Acrylic };   // Acrylic: only through VELTRA_BACKDROP, to verify see-through

struct App {
    HINSTANCE inst = nullptr;
    HWND hwnd = nullptr;
    HBRUSH bg = nullptr;
    Backdrop backdrop = Backdrop::None;
    std::string token;          // per-process; the server always runs with --require-token <token>
    std::wstring models_dir;
    std::wstring user_data;
    Loader loader;
    bool env_requested = false;
    bool webview_failed = false;
    int port = 0;
    ICoreWebView2Environment* env = nullptr;
    ICoreWebView2Controller* controller = nullptr;
    ICoreWebView2* webview = nullptr;
    bool navigated = false;
    bool tray = false;          // the tray icon is present (closing the window hides to it)
    std::wstring status;        // text painted in the window while there is no WebView
    bool frameless = false;     // the page draws the title bar
    bool app_region = false;    // the runtime handles CSS app-region: drag itself
    bool quitting = false;
    UINT taskbar_created = 0;   // Explorer restarted: add the tray icon again
    // persisted in <app data>\window_state.json
    std::string state_path;
    bool hotkey_wanted = false;
    bool hotkey_on = false;     // RegisterHotKey succeeded
    bool hotkey_ctrl = false;   // Alt+Space was taken: registered as Ctrl+Alt+Space
    bool tray_hint_shown = false;
    // Alt+Space overlay (created on first use)
    HWND overlay = nullptr;
    ICoreWebView2Controller* ov_controller = nullptr;
    ICoreWebView2* ov_webview = nullptr;
    bool ov_creating = false;
    int ov_height = kOverlayMinH;
};
App g_app;
UINT window_dpi(HWND h);
void show_main();
void toggle_overlay();
void hide_overlay();
void set_hotkey(bool enabled);

void post_json(const std::wstring& json);

std::wstring origin() { return L"http://127.0.0.1:" + std::to_wstring(g_app.port); }

void set_status(const std::wstring& s) {
    g_app.status = s;
    if (g_app.hwnd) InvalidateRect(g_app.hwnd, nullptr, TRUE);
}

void open_in_browser(const std::wstring& url) {
    ShellExecuteW(nullptr, L"open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

void set_frameless(bool on);
std::wstring startup_script();
void save_window_state();
void resize_webview();

bool is_own_url(const std::wstring& uri) {
    const std::wstring o = origin() + L"/";
    return uri.compare(0, o.size(), o) == 0 || uri == origin() || uri == L"about:blank";
}

// The address for the system browser. The server always requires the token; the page takes it from the fragment (never
// sent over HTTP) and removes it from the address bar.
std::wstring browser_url() { return origin() + L"/#veltra_token=" + widen(g_app.token); }

// WebView2 missing or broken: tell the user (Dutch), keep the server running and open the UI in the default browser.
void fallback_to_browser(const wchar_t* why) {
    g_app.webview_failed = true;
    set_frameless(false);
    std::wstring msg = std::wstring(why) +
                       L"\n\nVeltra gebruikt de Microsoft Edge WebView2 Runtime voor zijn venster. Installeer die gratis via:\n" +
                       kInstallUrl +
                       L"\n\nStart Veltra daarna opnieuw. Voor nu opent Veltra in je standaardbrowser; dit venster openhouden "
                       L"zolang je Veltra gebruikt.";
    set_status(L"Veltra draait in je browser.\nKlik in dit venster om het opnieuw te openen (het adres zelf typen werkt niet).\n"
               L"Sluit dit venster om Veltra te stoppen.");
    MessageBoxW(g_app.hwnd, msg.c_str(), L"Veltra", MB_OK | MB_ICONINFORMATION);
    open_in_browser(browser_url());
}

void resize_webview() {
    if (!g_app.controller) return;
    RECT rc;
    GetClientRect(g_app.hwnd, &rc);
    g_app.controller->put_Bounds(rc);
}

void post_json(const std::wstring& json) {
    if (g_app.webview) g_app.webview->PostWebMessageAsJson(json.c_str());
}

void post_window_state() {
    post_json(std::wstring(L"{\"type\":\"window_state\",\"maximized\":") + (g_app.hwnd && IsZoomed(g_app.hwnd) ? L"true}" : L"false}"));
}

void post_hotkey_state() {
    post_json(std::wstring(L"{\"type\":\"hotkey_state\",\"enabled\":") + (g_app.hotkey_wanted ? L"true" : L"false") +
              L",\"registered\":" + (g_app.hotkey_on ? L"true" : L"false") + L",\"keys\":\"" +
              (g_app.hotkey_ctrl ? L"Ctrl + Alt + Spatie" : L"Alt + Spatie") + L"\"}");
}

// ---- window state file --------------------------------------------------------------------------------------------

struct SavedWindow {
    bool valid = false, maximized = false;
    RECT rect{};
};

SavedWindow read_window_state() {
    SavedWindow w;
    if (g_app.state_path.empty()) return w;
    std::FILE* f = _wfopen(widen(g_app.state_path).c_str(), L"rb");
    if (!f) return w;
    char buf[4096];
    const size_t n = std::fread(buf, 1, sizeof buf, f);
    std::fclose(f);
    lb::json::Value v;
    if (n == sizeof buf || !lb::json::parse(std::string(buf, n), &v) || !v.is_object()) return w;
    g_app.hotkey_wanted = v["hotkey"].as_bool(false);
    g_app.tray_hint_shown = v["tray_hint_shown"].as_bool(false);
    const int64_t x = v["x"].as_int(0), y = v["y"].as_int(0), cw = v["w"].as_int(0), ch = v["h"].as_int(0);
    if (cw < 200 || ch < 150 || cw > 20000 || ch > 20000 || x < -32000 || x > 32000 || y < -32000 || y > 32000) return w;
    w.rect = RECT{static_cast<LONG>(x), static_cast<LONG>(y), static_cast<LONG>(x + cw), static_cast<LONG>(y + ch)};
    w.maximized = v["maximized"].as_bool(false);
    w.valid = true;
    return w;
}

void save_window_state() {
    if (g_app.state_path.empty() || !g_app.hwnd) return;
    WINDOWPLACEMENT wp{};
    wp.length = sizeof wp;
    if (!GetWindowPlacement(g_app.hwnd, &wp)) return;
    const RECT& r = wp.rcNormalPosition;
    const bool maximized = wp.showCmd == SW_SHOWMAXIMIZED || (wp.showCmd == SW_SHOWMINIMIZED && (wp.flags & WPF_RESTORETOMAXIMIZED));
    MONITORINFOEXW mi{};
    mi.cbSize = sizeof mi;
    std::string monitor;
    if (GetMonitorInfoW(MonitorFromWindow(g_app.hwnd, MONITOR_DEFAULTTONEAREST), &mi)) {
        for (const wchar_t* c = mi.szDevice; *c; ++c)
            if (*c < 128) monitor += (*c == L'\\') ? std::string("\\\\") : std::string(1, static_cast<char>(*c));
    }
    const std::string json = "{\"monitor\":\"" + monitor + "\",\"x\":" + std::to_string(r.left) + ",\"y\":" + std::to_string(r.top) +
                             ",\"w\":" + std::to_string(r.right - r.left) + ",\"h\":" + std::to_string(r.bottom - r.top) +
                             ",\"maximized\":" + (maximized ? "true" : "false") + ",\"hotkey\":" + (g_app.hotkey_wanted ? "true" : "false") +
                             ",\"tray_hint_shown\":" + (g_app.tray_hint_shown ? "true" : "false") + "}";
    const std::wstring path = widen(g_app.state_path), tmp = path + L".tmp";
    std::FILE* f = _wfopen(tmp.c_str(), L"wb");
    if (!f) return;
    const bool ok = std::fwrite(json.data(), 1, json.size(), f) == json.size();
    if (std::fclose(f) == 0 && ok) MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING);
}

// A saved rectangle is only used when its title bar is on a monitor that exists now; its size is clamped to that screen.
bool clamp_to_monitor(RECT* r) {
    const POINT title{(r->left + r->right) / 2, r->top + 20};
    const HMONITOR mon = MonitorFromPoint(title, MONITOR_DEFAULTTONULL);
    MONITORINFO mi{};
    mi.cbSize = sizeof mi;
    if (!mon || !GetMonitorInfoW(mon, &mi)) return false;
    const RECT& wa = mi.rcWork;
    const LONG w = std::min(r->right - r->left, wa.right - wa.left), h = std::min(r->bottom - r->top, wa.bottom - wa.top);
    const LONG x = std::max(wa.left, std::min(r->left, wa.right - w)), y = std::max(wa.top, std::min(r->top, wa.bottom - h));
    *r = RECT{x, y, x + w, y + h};
    return true;
}

// ---- frameless window ---------------------------------------------------------------------------------------------

void set_frameless(bool on) {
    if (!g_app.hwnd || on == g_app.frameless) return;
    g_app.frameless = on;
    // WS_SYSMENU is what makes DWM draw its own caption buttons into the extended frame: only keep it with the standard frame
    LONG_PTR style = GetWindowLongPtrW(g_app.hwnd, GWL_STYLE);
    style = on ? (style & ~static_cast<LONG_PTR>(WS_SYSMENU)) : (style | WS_SYSMENU);
    SetWindowLongPtrW(g_app.hwnd, GWL_STYLE, style);
    SetWindowPos(g_app.hwnd, nullptr, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED);
}

void window_action(const std::string& a) {
    const HWND h = g_app.hwnd;
    if (a == "minimize") ShowWindow(h, SW_MINIMIZE);
    else if (a == "maximize") ShowWindow(h, IsZoomed(h) ? SW_RESTORE : SW_MAXIMIZE);
    else if (a == "close") PostMessageW(h, WM_CLOSE, 0, 0);
    else {
        // the mouse button is still down in the page: hand the drag / resize over to Windows (keeps Aero Snap)
        WPARAM hit = 0;
        if (a == "drag") hit = HTCAPTION;
        else if (a == "resize_top") hit = HTTOP;
        else if (a == "resize_topleft") hit = HTTOPLEFT;
        else if (a == "resize_topright") hit = HTTOPRIGHT;
        if (!hit || (hit != HTCAPTION && IsZoomed(h)) || !(GetAsyncKeyState(GetSystemMetrics(SM_SWAPBUTTON) ? VK_RBUTTON : VK_LBUTTON) & 0x8000))
            return;
        POINT pt;
        GetCursorPos(&pt);
        ReleaseCapture();
        PostMessageW(h, WM_NCLBUTTONDOWN, hit, MAKELPARAM(pt.x, pt.y));
    }
}

// ---- window backdrop (Mica) ---------------------------------------------------------------------------------------

const char* backdrop_name(Backdrop b) { return b == Backdrop::Mica ? "mica" : b == Backdrop::Acrylic ? "acrylic" : "none"; }

DWORD os_build() {   // RtlGetVersion: GetVersionEx lies to programs without a matching manifest
    using Fn = LONG(WINAPI*)(OSVERSIONINFOW*);
    HMODULE nt = GetModuleHandleW(L"ntdll.dll");
    const Fn fn = nt ? reinterpret_cast<Fn>(reinterpret_cast<void*>(GetProcAddress(nt, "RtlGetVersion"))) : nullptr;
    OSVERSIONINFOW v{};
    v.dwOSVersionInfoSize = sizeof v;
    return fn && fn(&v) == 0 ? v.dwBuildNumber : 0;
}

Backdrop wanted_backdrop() {
    if (os_build() < kMinBackdropBuild) return Backdrop::None;
    if (const char* o = env("VELTRA_BACKDROP")) {
        if (std::strcmp(o, "none") == 0) return Backdrop::None;
        if (std::strcmp(o, "mica") == 0) return Backdrop::Mica;
        if (std::strcmp(o, "acrylic") == 0) return Backdrop::Acrylic;
    }
    DWORD transparency = 1, size = sizeof transparency;   // a missing value means "on"
    if (RegGetValueW(HKEY_CURRENT_USER, L"Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize", L"EnableTransparency",
                     RRF_RT_REG_DWORD, nullptr, &transparency, &size) == ERROR_SUCCESS &&
        transparency == 0)
        return Backdrop::None;
    HIGHCONTRASTW hc{};
    hc.cbSize = sizeof hc;
    if (SystemParametersInfoW(SPI_GETHIGHCONTRAST, sizeof hc, &hc, 0) && (hc.dwFlags & HCF_HIGHCONTRASTON)) return Backdrop::None;
    return Backdrop::Mica;
}

// The WebView2 colour behind the page. Only alpha 0 or 255 is accepted: all tinting is done in CSS.
void apply_webview_background() {
    if (!g_app.controller) return;
    ICoreWebView2Controller2* c2 = nullptr;
    if (FAILED(g_app.controller->QueryInterface(IID_ICoreWebView2Controller2, reinterpret_cast<void**>(&c2)))) return;
    const bool glass = g_app.backdrop != Backdrop::None;
    const COREWEBVIEW2_COLOR color = glass ? COREWEBVIEW2_COLOR{0, 0, 0, 0}
                                           : COREWEBVIEW2_COLOR{255, GetRValue(kBg), GetGValue(kBg), GetBValue(kBg)};
    const HRESULT hr = c2->put_DefaultBackgroundColor(color);
    if (FAILED(hr)) dlog("put_DefaultBackgroundColor failed: " + std::to_string(static_cast<long>(hr)));
    c2->Release();
}

void apply_backdrop(Backdrop b) {
    const HWND h = g_app.hwnd;
    if (b != Backdrop::None) {
        const MARGINS glass{-1, -1, -1, -1};
        const int type = b == Backdrop::Mica ? 2 /* DWMSBT_MAINWINDOW */ : 3 /* DWMSBT_TRANSIENTWINDOW */;
        if (FAILED(DwmExtendFrameIntoClientArea(h, &glass)) || FAILED(DwmSetWindowAttribute(h, kDwmaBackdrop, &type, sizeof type)))
            b = Backdrop::None;
    }
    COLORREF caption = kDwmColorDefault;   // native: the caption merges into the backdrop
    HBRUSH brush = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));   // black in the extended frame is see-through
    if (b == Backdrop::None) {
        const MARGINS none{0, 0, 0, 0};
        DwmExtendFrameIntoClientArea(h, &none);
        const int type = 1;   // DWMSBT_NONE (fails harmlessly on older builds)
        DwmSetWindowAttribute(h, kDwmaBackdrop, &type, sizeof type);
        caption = kBg;
        brush = g_app.bg;
    }
    DwmSetWindowAttribute(h, kDwmaCaptionColor, &caption, sizeof caption);
    SetClassLongPtrW(h, GCLP_HBRBACKGROUND, reinterpret_cast<LONG_PTR>(brush));
    g_app.backdrop = b;
    apply_webview_background();
    InvalidateRect(h, nullptr, TRUE);
}

// A Windows setting changed (transparency effects, high contrast, theme, composition): follow it and tell the page.
void refresh_backdrop() {
    const Backdrop before = g_app.backdrop, want = wanted_backdrop();
    if (want == before) return;
    apply_backdrop(want);
    if (g_app.backdrop == before) return;
    dlog(std::string("window backdrop: ") + backdrop_name(g_app.backdrop));
    post_json(L"{\"type\":\"backdrop\",\"mode\":\"" + widen(backdrop_name(g_app.backdrop)) + L"\"}");
}

// the token only for documents of our own origin (an external page opened by accident never sees it)
std::wstring startup_script() {
    return L"if (location.origin === '" + origin() + L"') { Object.defineProperty(window, '__VELTRA_TOKEN', { value: '" +
           widen(g_app.token) + L"', writable: false, configurable: false }); window.__VELTRA_DESKTOP = true; window.__VELTRA_BACKDROP = '" +
           widen(backdrop_name(g_app.backdrop)) + L"'; window.__VELTRA_FRAMELESS = " + (g_app.frameless ? L"true" : L"false") +
           L"; window.__VELTRA_APPREGION = " + (g_app.app_region ? L"true" : L"false") + L"; }";
}

void navigate_when_ready() {
    if (g_app.navigated || !g_app.webview || g_app.port <= 0) return;
    g_app.navigated = true;
    const std::wstring js = startup_script();
    auto* done = make<ICoreWebView2AddScriptToExecuteOnDocumentCreatedCompletedHandler>(
        IID_ICoreWebView2AddScriptToExecuteOnDocumentCreatedCompletedHandler, [](HRESULT hr, LPCWSTR) -> HRESULT {
            if (FAILED(hr)) dlog("AddScriptToExecuteOnDocumentCreated failed: " + std::to_string(static_cast<long>(hr)));
            if (g_app.webview) g_app.webview->Navigate((origin() + L"/").c_str());
            return S_OK;
        });
    const HRESULT hr = g_app.webview->AddScriptToExecuteOnDocumentCreated(js.c_str(), done);
    done->Release();
    if (FAILED(hr)) g_app.webview->Navigate((origin() + L"/").c_str());
}

bool base64_decode(const std::string& in, std::vector<unsigned char>* out) {
    static const char* al = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    int val = 0, bits = 0;
    out->clear();
    out->reserve(in.size() / 4 * 3);
    for (char c : in) {
        if (c == '=') break;
        const char* p = std::strchr(al, c);
        if (!c || !p) return false;
        val = ((val << 6) | static_cast<int>(p - al)) & 0xFFFFFF;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out->push_back(static_cast<unsigned char>((val >> bits) & 0xFF));
        }
    }
    return true;
}

HGLOBAL global_copy(const void* data, size_t n) {
    HGLOBAL g = GlobalAlloc(GMEM_MOVEABLE, n);
    if (!g) return nullptr;
    if (void* p = GlobalLock(g)) {
        std::memcpy(p, data, n);
        GlobalUnlock(g);
        return g;
    }
    GlobalFree(g);
    return nullptr;
}

// The page's result card on the clipboard: as a bitmap (every program) and as PNG (keeps exact pixels where supported).
// bgra64: width * height * 4 bytes, bottom-up rows, as a DIB wants them.
bool copy_image(int w, int h, const std::string& bgra64, const std::string& png64) {
    if (w <= 0 || h <= 0 || w > 4096 || h > 4096) return false;
    std::vector<unsigned char> px, png;
    if (!base64_decode(bgra64, &px) || px.size() != static_cast<size_t>(w) * h * 4 || !base64_decode(png64, &png) || png.size() > (32u << 20))
        return false;
    BITMAPINFOHEADER bi{};
    bi.biSize = sizeof bi;
    bi.biWidth = w;
    bi.biHeight = h;
    bi.biPlanes = 1;
    bi.biBitCount = 32;
    bi.biCompression = BI_RGB;
    bi.biSizeImage = static_cast<DWORD>(px.size());
    std::vector<unsigned char> dib(sizeof bi + px.size());
    std::memcpy(dib.data(), &bi, sizeof bi);
    std::memcpy(dib.data() + sizeof bi, px.data(), px.size());
    if (!OpenClipboard(g_app.hwnd)) return false;
    EmptyClipboard();
    bool ok = false;
    if (HGLOBAL d = global_copy(dib.data(), dib.size())) {
        ok = SetClipboardData(CF_DIB, d) != nullptr;
        if (!ok) GlobalFree(d);
    }
    if (!png.empty())
        if (HGLOBAL p = global_copy(png.data(), png.size()))
            if (!SetClipboardData(RegisterClipboardFormatW(L"PNG"), p)) GlobalFree(p);
    CloseClipboard();
    return ok;
}

void on_web_message(ICoreWebView2WebMessageReceivedEventArgs* args) {
    LPWSTR src = nullptr;
    if (FAILED(args->get_Source(&src)) || !is_own_url(take_co_string(src))) return;   // only our own pages
    LPWSTR raw = nullptr;
    std::wstring json;
    if (SUCCEEDED(args->TryGetWebMessageAsString(&raw))) json = take_co_string(raw);   // postMessage(JSON.stringify(..))
    else if (SUCCEEDED(args->get_WebMessageAsJson(&raw))) json = take_co_string(raw);  // postMessage({...})
    lb::json::Value v;
    if (!lb::json::parse(narrow(json), &v) || !v.is_object()) return;
    const lb::json::Value* t = v.find("type");
    if (!t || !t->is_string()) return;
    if (t->as_string() == "pick_model") PostMessageW(g_app.hwnd, WM_APP_PICK, 0, 0);
    else if (t->as_string() == "pick_folder") PostMessageW(g_app.hwnd, WM_APP_PICK, 1, 0);
    else if (t->as_string() == "window") window_action(v["action"].as_string(""));
    else if (t->as_string() == "copy_png") {
        const bool ok = copy_image(static_cast<int>(v["width"].as_int(0)), static_cast<int>(v["height"].as_int(0)), v["bgra"].as_string(), v["png"].as_string());
        post_json(std::wstring(L"{\"type\":\"copied\",\"ok\":") + (ok ? L"true}" : L"false}"));
    }
    else if (t->as_string() == "hotkey") {
        set_hotkey(v["enabled"].as_bool(false));
        save_window_state();
        post_hotkey_state();
    } else if (t->as_string() == "hello") {
        post_window_state();
        post_hotkey_state();
    }
    // unknown types are ignored
}

// Native file / folder picker (runs from the message loop, not inside a WebView2 event handler).
void pick(bool folder) {
    const wchar_t* reply = folder ? L"picked_folder" : L"picked_model";
    std::wstring path;
    IFileOpenDialog* dlg = nullptr;
    if (SUCCEEDED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_IFileOpenDialog,
                                   reinterpret_cast<void**>(&dlg)))) {
        DWORD opts = 0;
        dlg->GetOptions(&opts);
        dlg->SetOptions(opts | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST | (folder ? FOS_PICKFOLDERS : FOS_FILEMUSTEXIST));
        if (folder) {
            dlg->SetTitle(L"Kies een map");
        } else {
            const COMDLG_FILTERSPEC types[] = {{L"Veltra modellen (*.vlt;*.lbt)", L"*.vlt;*.lbt"}, {L"Alle bestanden (*.*)", L"*.*"}};
            dlg->SetFileTypes(2, types);
            dlg->SetTitle(L"Kies een Veltra model");
        }
        IShellItem* start = nullptr;
        if (!g_app.models_dir.empty() &&
            SUCCEEDED(SHCreateItemFromParsingName(g_app.models_dir.c_str(), nullptr, IID_IShellItem, reinterpret_cast<void**>(&start)))) {
            dlg->SetDefaultFolder(start);
            start->Release();
        }
        if (SUCCEEDED(dlg->Show(g_app.hwnd))) {
            IShellItem* item = nullptr;
            if (SUCCEEDED(dlg->GetResult(&item))) {
                LPWSTR p = nullptr;
                if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &p))) path = take_co_string(p);
                item->Release();
            }
        }
        dlg->Release();
    }
    if (path.empty()) post_json(std::wstring(L"{\"type\":\"") + reply + L"\",\"cancelled\":true}");
    else post_json(std::wstring(L"{\"type\":\"") + reply + L"\",\"path\":" + json_quote(path) + L"}");
}

HRESULT on_controller(HRESULT hr, ICoreWebView2Controller* ctl) {
    if (FAILED(hr) || !ctl) {
        dlog("CreateCoreWebView2Controller failed: " + std::to_string(static_cast<long>(hr)));
        if (g_app.port > 0) fallback_to_browser(L"Het Veltra-venster kon niet worden geopend.");
        else g_app.webview_failed = true;
        return S_OK;
    }
    g_app.controller = ctl;
    ctl->AddRef();
    apply_webview_background();   // before the first paint: no white flash
    ctl->get_CoreWebView2(&g_app.webview);
    if (!g_app.webview) return S_OK;
    ICoreWebView2Settings* st = nullptr;
    if (SUCCEEDED(g_app.webview->get_Settings(&st)) && st) {
        const char* dt = env("VELTRA_DEVTOOLS");
        st->put_AreDevToolsEnabled(dt && dt[0] == '1' ? TRUE : FALSE);
        st->put_IsStatusBarEnabled(FALSE);
        st->put_IsZoomControlEnabled(TRUE);
        ICoreWebView2Settings9* st9 = nullptr;
        if (SUCCEEDED(st->QueryInterface(IID_ICoreWebView2Settings9, reinterpret_cast<void**>(&st9))) && st9) {
            g_app.app_region = SUCCEEDED(st9->put_IsNonClientRegionSupportEnabled(TRUE));
            st9->Release();
        }
        dlog(std::string("title bar drag: ") + (g_app.app_region ? "app-region" : "message fallback"));
        st->Release();
    }
    EventRegistrationToken tok;
    auto* msg = make<ICoreWebView2WebMessageReceivedEventHandler>(
        IID_ICoreWebView2WebMessageReceivedEventHandler, [](ICoreWebView2*, ICoreWebView2WebMessageReceivedEventArgs* a) -> HRESULT {
            on_web_message(a);
            return S_OK;
        });
    g_app.webview->add_WebMessageReceived(msg, &tok);
    msg->Release();
    // links to other sites open in the default browser, never inside the app window
    auto* nav = make<ICoreWebView2NavigationStartingEventHandler>(
        IID_ICoreWebView2NavigationStartingEventHandler, [](ICoreWebView2*, ICoreWebView2NavigationStartingEventArgs* a) -> HRESULT {
            LPWSTR u = nullptr;
            const std::wstring uri = SUCCEEDED(a->get_Uri(&u)) ? take_co_string(u) : std::wstring();
            if (!is_own_url(uri)) {
                a->put_Cancel(TRUE);
                if (uri.rfind(L"http://", 0) == 0 || uri.rfind(L"https://", 0) == 0) open_in_browser(uri);
            }
            return S_OK;
        });
    g_app.webview->add_NavigationStarting(nav, &tok);
    nav->Release();
    auto* loaded = make<ICoreWebView2NavigationCompletedEventHandler>(
        IID_ICoreWebView2NavigationCompletedEventHandler, [](ICoreWebView2*, ICoreWebView2NavigationCompletedEventArgs* a) -> HRESULT {
            BOOL ok = FALSE;
            a->get_IsSuccess(&ok);
            COREWEBVIEW2_WEB_ERROR_STATUS why = COREWEBVIEW2_WEB_ERROR_STATUS_UNKNOWN;
            a->get_WebErrorStatus(&why);
            RECT b{};
            if (g_app.controller) g_app.controller->get_Bounds(&b);
            dlog(std::string("page ") + (ok ? "loaded" : "failed, web error " + std::to_string(static_cast<int>(why))) + ", view " +
                 std::to_string(b.right - b.left) + "x" + std::to_string(b.bottom - b.top));
            return S_OK;
        });
    g_app.webview->add_NavigationCompleted(loaded, &tok);
    loaded->Release();
    auto* nw = make<ICoreWebView2NewWindowRequestedEventHandler>(
        IID_ICoreWebView2NewWindowRequestedEventHandler, [](ICoreWebView2*, ICoreWebView2NewWindowRequestedEventArgs* a) -> HRESULT {
            LPWSTR u = nullptr;
            const std::wstring uri = SUCCEEDED(a->get_Uri(&u)) ? take_co_string(u) : std::wstring();
            a->put_Handled(TRUE);
            if (is_own_url(uri)) {
                if (g_app.webview) g_app.webview->Navigate(uri.c_str());
            } else if (uri.rfind(L"http://", 0) == 0 || uri.rfind(L"https://", 0) == 0) {
                open_in_browser(uri);
            }
            return S_OK;
        });
    g_app.webview->add_NewWindowRequested(nw, &tok);
    nw->Release();
    resize_webview();
    ctl->put_IsVisible(TRUE);
    ctl->MoveFocus(COREWEBVIEW2_MOVE_FOCUS_REASON_PROGRAMMATIC);
    set_status(L"");
    navigate_when_ready();
    return S_OK;
}

HRESULT on_environment(HRESULT hr, ICoreWebView2Environment* e) {
    if (FAILED(hr) || !e) {
        dlog("WebView2 environment failed: " + std::to_string(static_cast<long>(hr)));
        if (g_app.port > 0) fallback_to_browser(L"De WebView2 Runtime kon niet worden gestart.");
        else g_app.webview_failed = true;
        return S_OK;
    }
    g_app.env = e;
    e->AddRef();
    auto* h = make<ICoreWebView2CreateCoreWebView2ControllerCompletedHandler>(IID_ICoreWebView2CreateCoreWebView2ControllerCompletedHandler,
                                                                              on_controller);
    const HRESULT r = e->CreateCoreWebView2Controller(g_app.hwnd, h);
    h->Release();
    if (FAILED(r)) on_controller(r, nullptr);
    return S_OK;
}

void start_webview() {
    if (!g_app.loader.available()) return;
    g_app.env_requested = true;
    dlog("WebView2 loader: " + narrow(g_app.loader.what));
    auto* h = make<ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler>(IID_ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler,
                                                                               on_environment);
    const HRESULT hr = g_app.loader.create(g_app.user_data, h);
    h->Release();
    if (FAILED(hr)) on_environment(hr, nullptr);
}

// ---- Alt+Space overlay: a small always-on-top window with its own page (/overlay) -----------------------------------

void place_overlay() {
    POINT pt;
    GetCursorPos(&pt);
    MONITORINFO mi{};
    mi.cbSize = sizeof mi;
    GetMonitorInfoW(MonitorFromPoint(pt, MONITOR_DEFAULTTOPRIMARY), &mi);
    // move first so the DPI is that of the target monitor, then size
    SetWindowPos(g_app.overlay, HWND_TOPMOST, mi.rcWork.left, mi.rcWork.top, 0, 0, SWP_NOSIZE | SWP_NOACTIVATE);
    const int dpi = static_cast<int>(window_dpi(g_app.overlay));
    const int w = std::min<int>(MulDiv(kOverlayW, dpi, 96), mi.rcWork.right - mi.rcWork.left);
    const int h = std::min<int>(MulDiv(g_app.ov_height, dpi, 96), mi.rcWork.bottom - mi.rcWork.top);
    const int x = mi.rcWork.left + (mi.rcWork.right - mi.rcWork.left - w) / 2;
    const int y = mi.rcWork.top + (mi.rcWork.bottom - mi.rcWork.top) / 5;
    SetWindowPos(g_app.overlay, HWND_TOPMOST, x, y, w, h, SWP_NOACTIVATE);
}

void resize_overlay(int height) {
    if (!g_app.overlay) return;
    g_app.ov_height = std::max(kOverlayMinH, std::min(kOverlayMaxH, height));
    RECT r;
    GetWindowRect(g_app.overlay, &r);
    const int dpi = static_cast<int>(window_dpi(g_app.overlay));
    SetWindowPos(g_app.overlay, nullptr, 0, 0, r.right - r.left, MulDiv(g_app.ov_height, dpi, 96), SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
    if (g_app.ov_controller) {
        RECT rc;
        GetClientRect(g_app.overlay, &rc);
        g_app.ov_controller->put_Bounds(rc);
    }
}

void hide_overlay() {
    if (!g_app.overlay) return;
    ShowWindow(g_app.overlay, SW_HIDE);
    if (g_app.ov_controller) g_app.ov_controller->put_IsVisible(FALSE);
}

void reveal_overlay() {
    place_overlay();
    ShowWindow(g_app.overlay, SW_SHOW);
    SetForegroundWindow(g_app.overlay);
    if (!g_app.ov_controller) return;
    RECT rc;
    GetClientRect(g_app.overlay, &rc);
    g_app.ov_controller->put_Bounds(rc);
    g_app.ov_controller->put_IsVisible(TRUE);
    g_app.ov_controller->MoveFocus(COREWEBVIEW2_MOVE_FOCUS_REASON_PROGRAMMATIC);
    if (g_app.ov_webview) g_app.ov_webview->PostWebMessageAsJson(L"{\"type\":\"overlay_shown\"}");
}

void on_overlay_message(ICoreWebView2WebMessageReceivedEventArgs* args) {
    LPWSTR src = nullptr;
    if (FAILED(args->get_Source(&src)) || !is_own_url(take_co_string(src))) return;
    LPWSTR raw = nullptr;
    if (FAILED(args->get_WebMessageAsJson(&raw))) return;
    lb::json::Value v;
    if (!lb::json::parse(narrow(take_co_string(raw)), &v) || v["type"].as_string("") != "overlay") return;
    const std::string a = v["action"].as_string("");
    if (a == "hide") hide_overlay();
    else if (a == "resize") resize_overlay(static_cast<int>(std::max<int64_t>(0, std::min<int64_t>(4000, v["height"].as_int(0)))));
    else if (a == "open_main") {
        hide_overlay();
        show_main();
        const std::string id = v["chat"].as_string("");
        if (!id.empty() && id.size() <= 64 &&
            id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_") == std::string::npos)
            post_json(L"{\"type\":\"open_chat\",\"id\":" + json_quote(widen(id)) + L"}");
    }
}

HRESULT on_overlay_controller(HRESULT hr, ICoreWebView2Controller* ctl) {
    g_app.ov_creating = false;
    if (FAILED(hr) || !ctl || !g_app.overlay) {
        dlog("overlay: controller failed: " + std::to_string(static_cast<long>(hr)));
        hide_overlay();
        return S_OK;
    }
    g_app.ov_controller = ctl;
    ctl->AddRef();
    ctl->get_CoreWebView2(&g_app.ov_webview);
    if (!g_app.ov_webview) return S_OK;
    ICoreWebView2Settings* st = nullptr;
    if (SUCCEEDED(g_app.ov_webview->get_Settings(&st)) && st) {
        st->put_AreDevToolsEnabled(FALSE);
        st->put_IsStatusBarEnabled(FALSE);
        st->put_AreDefaultContextMenusEnabled(TRUE);
        st->Release();
    }
    EventRegistrationToken tok;
    auto* msg = make<ICoreWebView2WebMessageReceivedEventHandler>(
        IID_ICoreWebView2WebMessageReceivedEventHandler, [](ICoreWebView2*, ICoreWebView2WebMessageReceivedEventArgs* a) -> HRESULT {
            on_overlay_message(a);
            return S_OK;
        });
    g_app.ov_webview->add_WebMessageReceived(msg, &tok);
    msg->Release();
    auto* nav = make<ICoreWebView2NavigationStartingEventHandler>(
        IID_ICoreWebView2NavigationStartingEventHandler, [](ICoreWebView2*, ICoreWebView2NavigationStartingEventArgs* a) -> HRESULT {
            LPWSTR u = nullptr;
            if (!is_own_url(SUCCEEDED(a->get_Uri(&u)) ? take_co_string(u) : std::wstring())) a->put_Cancel(TRUE);
            return S_OK;
        });
    g_app.ov_webview->add_NavigationStarting(nav, &tok);
    nav->Release();
    auto* nw = make<ICoreWebView2NewWindowRequestedEventHandler>(
        IID_ICoreWebView2NewWindowRequestedEventHandler, [](ICoreWebView2*, ICoreWebView2NewWindowRequestedEventArgs* a) -> HRESULT {
            a->put_Handled(TRUE);
            return S_OK;
        });
    g_app.ov_webview->add_NewWindowRequested(nw, &tok);
    nw->Release();
    auto* done = make<ICoreWebView2AddScriptToExecuteOnDocumentCreatedCompletedHandler>(
        IID_ICoreWebView2AddScriptToExecuteOnDocumentCreatedCompletedHandler, [](HRESULT, LPCWSTR) -> HRESULT {
            if (g_app.ov_webview) g_app.ov_webview->Navigate((origin() + L"/overlay").c_str());
            return S_OK;
        });
    g_app.ov_webview->AddScriptToExecuteOnDocumentCreated(startup_script().c_str(), done);
    done->Release();
    reveal_overlay();
    return S_OK;
}

void toggle_overlay() {
    if (!g_app.env || g_app.port <= 0) {   // engine / WebView not ready: just bring the main window forward
        show_main();
        return;
    }
    if (g_app.overlay && IsWindowVisible(g_app.overlay)) {
        hide_overlay();
        return;
    }
    if (!g_app.overlay) {
        g_app.overlay = CreateWindowExW(WS_EX_TOPMOST | WS_EX_TOOLWINDOW, kOverlayClass, L"Veltra snelstart", WS_POPUP, 0, 0, 100, 100,
                                        nullptr, nullptr, g_app.inst, nullptr);
        if (!g_app.overlay) return;
        const int round = 2;   // DWMWCP_ROUND
        DwmSetWindowAttribute(g_app.overlay, kDwmaCornerPreference, &round, sizeof round);
    }
    reveal_overlay();
    if (!g_app.ov_controller && !g_app.ov_creating) {
        g_app.ov_creating = true;
        auto* h = make<ICoreWebView2CreateCoreWebView2ControllerCompletedHandler>(
            IID_ICoreWebView2CreateCoreWebView2ControllerCompletedHandler, on_overlay_controller);
        const HRESULT r = g_app.env->CreateCoreWebView2Controller(g_app.overlay, h);
        h->Release();
        if (FAILED(r)) on_overlay_controller(r, nullptr);
    }
}

LRESULT CALLBACK overlay_proc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_CLOSE) {
        hide_overlay();
        return 0;
    }
    if (msg == WM_SETFOCUS && g_app.ov_controller) g_app.ov_controller->MoveFocus(COREWEBVIEW2_MOVE_FOCUS_REASON_PROGRAMMATIC);
    return DefWindowProcW(h, msg, wp, lp);
}

void set_hotkey(bool enabled) {
    g_app.hotkey_wanted = enabled;
    if (g_app.hotkey_on) {
        UnregisterHotKey(g_app.hwnd, kHotkeyId);
        g_app.hotkey_on = false;
    }
    g_app.hotkey_ctrl = false;
    if (enabled) {
        g_app.hotkey_on = RegisterHotKey(g_app.hwnd, kHotkeyId, MOD_ALT | MOD_NOREPEAT, VK_SPACE) != 0;
        if (!g_app.hotkey_on) {   // e.g. PowerToys Run owns Alt+Space
            g_app.hotkey_on = RegisterHotKey(g_app.hwnd, kHotkeyId, MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, VK_SPACE) != 0;
            g_app.hotkey_ctrl = g_app.hotkey_on;
            dlog(g_app.hotkey_on ? "Alt+Space is taken by another program: using Ctrl+Alt+Space" : "Alt+Space and Ctrl+Alt+Space are both taken");
        }
    } else {
        hide_overlay();
    }
}

// ---- tray ---------------------------------------------------------------------------------------------------------

NOTIFYICONDATAW tray_data() {
    NOTIFYICONDATAW nid{};
    nid.cbSize = sizeof nid;
    nid.hWnd = g_app.hwnd;
    nid.uID = kTrayId;
    return nid;
}

void add_tray_icon() {
    NOTIFYICONDATAW nid = tray_data();
    nid.uFlags = NIF_ICON | NIF_TIP | NIF_MESSAGE;
    nid.uCallbackMessage = WM_APP_TRAY;
    nid.hIcon = static_cast<HICON>(LoadImageW(g_app.inst, MAKEINTRESOURCEW(1), IMAGE_ICON, GetSystemMetrics(SM_CXSMICON),
                                              GetSystemMetrics(SM_CYSMICON), LR_DEFAULTCOLOR | LR_SHARED));
    lstrcpynW(nid.szTip, L"Veltra", ARRAYSIZE(nid.szTip));
    g_app.tray = Shell_NotifyIconW(NIM_ADD, &nid) != 0;
}

void remove_tray_icon() {
    if (!g_app.tray) return;
    NOTIFYICONDATAW nid = tray_data();
    Shell_NotifyIconW(NIM_DELETE, &nid);
    g_app.tray = false;
}

void tray_balloon(const wchar_t* text) {
    if (!g_app.tray) return;
    NOTIFYICONDATAW nid = tray_data();
    nid.uFlags = NIF_INFO;
    lstrcpynW(nid.szInfoTitle, L"Veltra", ARRAYSIZE(nid.szInfoTitle));
    lstrcpynW(nid.szInfo, text, ARRAYSIZE(nid.szInfo));
    nid.dwInfoFlags = NIIF_INFO;
    Shell_NotifyIconW(NIM_MODIFY, &nid);
}

void show_main() {
    const HWND h = g_app.hwnd;
    if (!IsWindowVisible(h)) ShowWindow(h, SW_SHOW);
    if (IsIconic(h)) ShowWindow(h, SW_RESTORE);
    SetForegroundWindow(h);
    if (g_app.controller) {
        g_app.controller->put_IsVisible(TRUE);
        resize_webview();
        g_app.controller->MoveFocus(COREWEBVIEW2_MOVE_FOCUS_REASON_PROGRAMMATIC);
    }
}

// The close button: keep the engine (and the loaded model) running in the tray.
void hide_to_tray() {
    save_window_state();
    if (g_app.controller) g_app.controller->put_IsVisible(FALSE);
    ShowWindow(g_app.hwnd, SW_HIDE);
    if (!g_app.tray_hint_shown) {
        g_app.tray_hint_shown = true;
        tray_balloon(L"Veltra draait door in het systeemvak. Rechtsklik op het icoon om volledig af te sluiten.");
        save_window_state();
    }
}

void quit_app() {
    g_app.quitting = true;
    save_window_state();
    DestroyWindow(g_app.hwnd);
}

void tray_menu() {
    HMENU m = CreatePopupMenu();
    if (!m) return;
    AppendMenuW(m, MF_STRING, kMenuOpen, L"Openen");
    AppendMenuW(m, MF_STRING, kMenuUnload, L"VRAM vrijgeven");
    AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
    AppendMenuW(m, MF_STRING, kMenuQuit, L"Volledig afsluiten");
    SetMenuDefaultItem(m, kMenuOpen, FALSE);
    POINT pt;
    GetCursorPos(&pt);
    SetForegroundWindow(g_app.hwnd);   // otherwise the menu does not close on an outside click
    const UINT cmd = TrackPopupMenu(m, TPM_RIGHTBUTTON | TPM_RETURNCMD | TPM_NONOTIFY, pt.x, pt.y, 0, g_app.hwnd, nullptr);
    PostMessageW(g_app.hwnd, WM_NULL, 0, 0);
    DestroyMenu(m);
    if (cmd == kMenuOpen) show_main();
    else if (cmd == kMenuUnload)
        std::thread([] { PostMessageW(g_app.hwnd, WM_APP_UNLOADED, static_cast<WPARAM>(veltra_request_unload()), 0); }).detach();
    else if (cmd == kMenuQuit) quit_app();
}

UINT window_dpi(HWND h) {
    const UINT d = GetDpiForWindow(h);
    return d ? d : 96;
}

// Attention for a pending remote test request (opt-in test mode): taskbar flash while the window is minimised or not in
// focus, plus a tray balloon. A window hidden in the tray comes back minimised so there is a taskbar button to flash.
// Approving happens only in the app UI; this just points there. Runs on the UI thread (WM_APP_REMOTE).
void remote_notice(HWND h, bool pending) {
    FLASHWINFO fi{};
    fi.cbSize = sizeof fi;
    fi.hwnd = h;
    if (pending) {
        if (!IsWindowVisible(h)) ShowWindow(h, SW_SHOWMINNOACTIVE);
        if (IsIconic(h) || GetForegroundWindow() != h) {
            fi.dwFlags = FLASHW_ALL | FLASHW_TIMERNOFG;   // until the window comes to the foreground
            FlashWindowEx(&fi);
            tray_balloon(L"Nieuw GPU-testverzoek ontvangen");
        }
    } else {
        fi.dwFlags = FLASHW_STOP;
        FlashWindowEx(&fi);
    }
}

LRESULT CALLBACK wnd_proc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    if (g_app.taskbar_created && msg == g_app.taskbar_created) {
        add_tray_icon();
        return 0;
    }
    switch (msg) {
        case WM_NCCALCSIZE:
            if (wp && g_app.frameless) {
                // keep the (invisible) native resize borders left / right / bottom, drop the caption at the top
                auto* p = reinterpret_cast<NCCALCSIZE_PARAMS*>(lp);
                const LONG top = p->rgrc[0].top;
                DefWindowProcW(h, msg, wp, lp);
                p->rgrc[0].top = top;
                if (IsZoomed(h)) {   // a maximised window reaches past the screen edge by the frame thickness
                    const UINT dpi = window_dpi(h);
                    p->rgrc[0].top += GetSystemMetricsForDpi(SM_CYSIZEFRAME, dpi) + GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi);
                }
                return 0;
            }
            break;
        case WM_NCHITTEST:
            if (g_app.frameless) {
                // only reached where no WebView covers the window (start-up screen): top strip = resize / drag
                const LRESULT def = DefWindowProcW(h, msg, wp, lp);
                if (def != HTCLIENT) return def;
                POINT pt{GET_X_LPARAM(lp), GET_Y_LPARAM(lp)};
                ScreenToClient(h, &pt);
                const int dpi = static_cast<int>(window_dpi(h));
                if (!IsZoomed(h) && pt.y < MulDiv(6, dpi, 96)) return HTTOP;
                if (pt.y < MulDiv(40, dpi, 96)) return HTCAPTION;
                return HTCLIENT;
            }
            break;
        case WM_CLOSE:
            if (!g_app.quitting && g_app.tray && !g_app.webview_failed) {
                hide_to_tray();
                return 0;
            }
            quit_app();
            return 0;
        case WM_QUERYENDSESSION:
            save_window_state();
            return TRUE;
        case WM_EXITSIZEMOVE:
            save_window_state();
            break;
        case WM_HOTKEY:
            if (wp == kHotkeyId) toggle_overlay();
            return 0;
        case WM_APP_SHOW:
            show_main();
            return 0;
        case WM_APP_UNLOADED:
            tray_balloon(wp == 0 ? L"Model uit het geheugen gehaald. Bij de volgende vraag laadt Veltra het weer."
                         : wp == 1 ? L"Veltra is bezig. Probeer het zo nog eens."
                                   : L"Er was geen model geladen.");
            return 0;
        case WM_GETMINMAXINFO: {
            auto* mm = reinterpret_cast<MINMAXINFO*>(lp);
            const UINT dpi = window_dpi(h);
            RECT r{0, 0, MulDiv(kMinW, static_cast<int>(dpi), 96), MulDiv(kMinH, static_cast<int>(dpi), 96)};
            AdjustWindowRectExForDpi(&r, WS_OVERLAPPEDWINDOW, FALSE, 0, dpi);
            mm->ptMinTrackSize.x = r.right - r.left;
            mm->ptMinTrackSize.y = r.bottom - r.top;
            return 0;
        }
        case WM_SIZE: {
            // minimised: stop the renderer painting (no TrySuspend, it would stall an active stream)
            if (g_app.controller) g_app.controller->put_IsVisible(wp == SIZE_MINIMIZED ? FALSE : TRUE);
            if (wp != SIZE_MINIMIZED) resize_webview();
            static WPARAM last = SIZE_RESTORED;
            if (wp != SIZE_MINIMIZED && wp != last) {
                last = wp;
                post_window_state();
                if (IsWindowVisible(h)) save_window_state();
            }
            return 0;
        }
        case WM_SETTINGCHANGE:
        case WM_THEMECHANGED:
        case WM_DWMCOMPOSITIONCHANGED:
            refresh_backdrop();
            break;
        case WM_MOVE:
        case WM_MOVING:
            if (g_app.controller) g_app.controller->NotifyParentWindowPositionChanged();
            break;
        case WM_DPICHANGED: {
            const RECT* r = reinterpret_cast<const RECT*>(lp);
            SetWindowPos(h, nullptr, r->left, r->top, r->right - r->left, r->bottom - r->top, SWP_NOZORDER | SWP_NOACTIVATE);
            resize_webview();
            return 0;
        }
        case WM_LBUTTONUP:   // browser fallback only (no WebView covers the window): open the UI again, with the token
            if (g_app.webview_failed && g_app.port > 0) open_in_browser(browser_url());
            return 0;
        case WM_SETFOCUS:
            if (g_app.controller) g_app.controller->MoveFocus(COREWEBVIEW2_MOVE_FOCUS_REASON_PROGRAMMATIC);
            return 0;
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC dc = BeginPaint(h, &ps);
            if (!g_app.status.empty() && !g_app.controller) {
                RECT rc;
                GetClientRect(h, &rc);
                SetBkMode(dc, TRANSPARENT);
                SetTextColor(dc, RGB(0xE4, 0xE4, 0xE7));
                HFONT f = CreateFontW(-MulDiv(11, static_cast<int>(window_dpi(h)), 72), 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, 0, 0,
                                      CLEARTYPE_QUALITY, 0, L"Segoe UI");
                HGDIOBJ old = SelectObject(dc, f);
                RECT tr = rc;
                DrawTextW(dc, g_app.status.c_str(), -1, &tr, DT_CENTER | DT_WORDBREAK | DT_CALCRECT);
                const int th = tr.bottom - tr.top;
                tr = rc;
                tr.top = (rc.bottom - th) / 2;
                DrawTextW(dc, g_app.status.c_str(), -1, &tr, DT_CENTER | DT_WORDBREAK);
                SelectObject(dc, old);
                DeleteObject(f);
            }
            EndPaint(h, &ps);
            return 0;
        }
        case WM_APP_LISTENING:
            g_app.port = static_cast<int>(wp);
            dlog("server listening on 127.0.0.1:" + std::to_string(g_app.port));
            if (!g_app.loader.available()) {
                fallback_to_browser(L"De Microsoft Edge WebView2 Runtime is niet gevonden op deze computer.");
            } else if (g_app.webview_failed) {
                fallback_to_browser(L"Het Veltra-venster kon niet worden geopend.");
            } else {
                if (!g_app.controller) set_status(L"Veltra wordt geladen\u2026");
                navigate_when_ready();
            }
            return 0;
        case WM_APP_SERVER_EXIT: {
            const std::wstring m = L"De Veltra-engine kon niet starten (foutcode " + std::to_wstring(static_cast<int>(wp)) +
                                   L").\n\nTip: start veltra.exe --headless vanuit een opdrachtprompt om de foutmelding te zien.";
            MessageBoxW(h, m.c_str(), L"Veltra", MB_OK | MB_ICONERROR);
            DestroyWindow(h);
            return 0;
        }
        case WM_APP_PICK:
            pick(wp == 1);
            return 0;
        case WM_APP_REMOTE:
            remote_notice(h, wp != 0);
            return 0;
        case WM_APP_TRAY:
            if (LOWORD(lp) == WM_LBUTTONUP || LOWORD(lp) == WM_LBUTTONDBLCLK || LOWORD(lp) == NIN_BALLOONUSERCLICK) show_main();
            else if (LOWORD(lp) == WM_RBUTTONUP || LOWORD(lp) == WM_CONTEXTMENU) tray_menu();
            return 0;
        case WM_DESTROY:
            remote_notice(h, false);
            if (g_app.hotkey_on) UnregisterHotKey(h, kHotkeyId);
            remove_tray_icon();
            PostQuitMessage(0);
            return 0;
        default:
            break;
    }
    return DefWindowProcW(h, msg, wp, lp);
}

void dark_title_bar(HWND h) {
    const BOOL on = TRUE;
    if (FAILED(DwmSetWindowAttribute(h, 20 /* DWMWA_USE_IMMERSIVE_DARK_MODE */, &on, sizeof on)))
        DwmSetWindowAttribute(h, 19 /* the same attribute before Windows 10 20H1 */, &on, sizeof on);
}

bool create_window(int show) {
    WNDCLASSEXW wc{};
    wc.cbSize = sizeof wc;
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = wnd_proc;
    wc.hInstance = g_app.inst;
    wc.hIcon = static_cast<HICON>(LoadImageW(g_app.inst, MAKEINTRESOURCEW(1), IMAGE_ICON, GetSystemMetrics(SM_CXICON),
                                              GetSystemMetrics(SM_CYICON), LR_DEFAULTCOLOR));
    wc.hIconSm = static_cast<HICON>(LoadImageW(g_app.inst, MAKEINTRESOURCEW(1), IMAGE_ICON, GetSystemMetrics(SM_CXSMICON),
                                                GetSystemMetrics(SM_CYSMICON), LR_DEFAULTCOLOR));
    wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512) /* IDC_ARROW */);
    wc.hbrBackground = g_app.bg = CreateSolidBrush(kBg);
    wc.lpszClassName = kWindowClass;
    if (!RegisterClassExW(&wc)) return false;
    WNDCLASSEXW ov = wc;
    ov.lpfnWndProc = overlay_proc;
    ov.lpszClassName = kOverlayClass;
    if (!RegisterClassExW(&ov)) return false;
    g_app.taskbar_created = RegisterWindowMessageW(L"TaskbarCreated");
    const SavedWindow saved = read_window_state();

    // monitor under the cursor: its work area and DPI decide the initial size and position
    POINT pt;
    GetCursorPos(&pt);
    HMONITOR mon = MonitorFromPoint(pt, MONITOR_DEFAULTTOPRIMARY);
    MONITORINFO mi{};
    mi.cbSize = sizeof mi;
    GetMonitorInfoW(mon, &mi);
    const RECT wa = mi.rcWork;
    g_app.hwnd = CreateWindowExW(0, wc.lpszClassName, L"Veltra", WS_OVERLAPPEDWINDOW, wa.left, wa.top, 100, 100, nullptr, nullptr,
                                 g_app.inst, nullptr);
    if (!g_app.hwnd) return false;
    dark_title_bar(g_app.hwnd);
    apply_backdrop(wanted_backdrop());   // also sets the caption colour
    dlog(std::string("window backdrop: ") + backdrop_name(g_app.backdrop) + " (Windows build " + std::to_string(os_build()) + ")");
    const UINT dpi = window_dpi(g_app.hwnd);
    RECT r{0, 0, MulDiv(kInitW, static_cast<int>(dpi), 96), MulDiv(kInitH, static_cast<int>(dpi), 96)};
    AdjustWindowRectExForDpi(&r, WS_OVERLAPPEDWINDOW, FALSE, 0, dpi);
    const int ww = std::min<int>(r.right - r.left, wa.right - wa.left);
    const int wh = std::min<int>(r.bottom - r.top, wa.bottom - wa.top);
    SetWindowPos(g_app.hwnd, nullptr, wa.left + (wa.right - wa.left - ww) / 2, wa.top + (wa.bottom - wa.top - wh) / 2, ww, wh,
                 SWP_NOZORDER | SWP_NOACTIVATE);
    if (g_app.loader.available()) set_frameless(true);
    RECT restored = saved.rect;
    if (saved.valid && clamp_to_monitor(&restored)) {
        WINDOWPLACEMENT pl{};
        pl.length = sizeof pl;
        pl.rcNormalPosition = restored;
        pl.showCmd = SW_HIDE;
        SetWindowPlacement(g_app.hwnd, &pl);
        if (saved.maximized && (show == SW_SHOWNORMAL || show == SW_SHOWDEFAULT || show == SW_SHOW)) show = SW_SHOWMAXIMIZED;
    }
    add_tray_icon();
    if (g_app.hotkey_wanted) set_hotkey(true);
    ShowWindow(g_app.hwnd, show);
    UpdateWindow(g_app.hwnd);
    return true;
}

// The in-process server: argv for veltra_server_main, kept alive for the whole process.
struct ServerArgs {
    std::vector<std::string> args;
    std::vector<char*> argv;
};

void start_server_thread(ServerArgs* sa) {
    for (std::string& s : sa->args) sa->argv.push_back(s.data());
    sa->argv.push_back(nullptr);
    veltra_set_listen_callback([](int port, void*) { PostMessageW(g_app.hwnd, WM_APP_LISTENING, static_cast<WPARAM>(port), 0); },
                               nullptr);
    std::thread([sa]() {
        const int rc = veltra_server_main(static_cast<int>(sa->args.size()), sa->argv.data());
        PostMessageW(g_app.hwnd, WM_APP_SERVER_EXIT, static_cast<WPARAM>(rc), 0);
    }).detach();
}

int run_desktop(HINSTANCE inst, int show, const std::vector<std::string>& user_args) {
    redirect_to_log_file();
    g_app.inst = inst;
    // One desktop instance per user session (VELTRA_ALLOW_MULTI=1 for tests): the window may be hidden in the tray, and a
    // second engine would load the model a second time.
    if (!env("VELTRA_ALLOW_MULTI")) {
        CreateMutexW(nullptr, FALSE, kMutexName);
        if (GetLastError() == ERROR_ALREADY_EXISTS) {
            for (int i = 0; i < 20; ++i) {
                if (HWND other = FindWindowW(kWindowClass, nullptr)) {
                    DWORD pid = 0;
                    GetWindowThreadProcessId(other, &pid);
                    AllowSetForegroundWindow(pid);
                    PostMessageW(other, WM_APP_SHOW, 0, 0);
                    break;
                }
                Sleep(100);
            }
            return 0;
        }
    }
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    const std::string exe_dir = veltra::plat::exe_dir();
    g_app.models_dir = widen(exe_dir + "\\models");
    const std::string data = veltra::plat::app_data_dir();
    g_app.user_data = widen((data.empty() ? exe_dir + "\\data" : data) + "\\WebView2");
    if (!data.empty()) g_app.state_path = data + "\\window_state.json";
    CreateDirectoryW(g_app.user_data.c_str(), nullptr);
    g_app.loader = find_loader();
    // Always: also without a WebView2 runtime, where the browser gets the token in the URL fragment (fallback_to_browser).
    // The API is never served on loopback trust alone.
    g_app.token = random_token_hex32();
    if (!g_app.loader.available()) dlog("no WebView2 runtime found");

    if (!create_window(show)) {
        MessageBoxW(nullptr, L"Veltra kon geen venster openen.", L"Veltra", MB_OK | MB_ICONERROR);
        return 1;
    }
    set_status(L"Veltra wordt gestart\u2026");

    // called on the remote-test worker / HTTP threads: only post to the UI thread
    veltra::remote::set_pending_hook([](bool pending) { PostMessageW(g_app.hwnd, WM_APP_REMOTE, pending ? 1 : 0, 0); });

    static ServerArgs sa;
    sa.args = {"veltra", "--ui", "desktop", "--models-dir", exe_dir + "\\models"};
    sa.args.insert(sa.args.end(), user_args.begin(), user_args.end());
    // always last (the last value wins): loopback only, ephemeral port, the session token
    sa.args.insert(sa.args.end(), {"--host", "127.0.0.1", "--port", "0", "--no-browser"});
    sa.args.insert(sa.args.end(), {"--require-token", g_app.token});
    start_server_thread(&sa);
    start_webview();   // in parallel with the model load / socket bind

    MSG m;
    while (GetMessageW(&m, nullptr, 0, 0) > 0) {
        TranslateMessage(&m);
        DispatchMessageW(&m);
    }
    // Window closed: shut the browser processes down, then end the process without running static destructors
    // (the server threads may be mid-request or mid-generation; the OS unmaps the model file).
    if (g_app.ov_controller) g_app.ov_controller->Close();
    release(g_app.ov_webview);
    release(g_app.ov_controller);
    if (g_app.controller) g_app.controller->Close();
    release(g_app.webview);
    release(g_app.controller);
    release(g_app.env);
    std::fflush(nullptr);
    TerminateProcess(GetCurrentProcess(), 0);
    return 0;
}

std::vector<std::string> utf8_args() {
    std::vector<std::string> out;
    int n = 0;
    if (LPWSTR* w = CommandLineToArgvW(GetCommandLineW(), &n)) {
        for (int i = 1; i < n; ++i) out.push_back(narrow(w[i]));
        LocalFree(w);
    }
    return out;
}

}  // namespace

int WINAPI WinMain(HINSTANCE inst, HINSTANCE, LPSTR, int show) {
    std::vector<std::string> args = utf8_args();
    bool headless = false;
    std::vector<std::string> rest;
    for (const std::string& a : args) {
        if (a == "--headless") headless = true;
        else rest.push_back(a);
    }
    if (!headless) return run_desktop(inst, show, rest);

    attach_console();
    std::vector<std::string> all = {"veltra"};
    all.insert(all.end(), rest.begin(), rest.end());
    std::vector<char*> argv;
    for (std::string& s : all) argv.push_back(s.data());
    argv.push_back(nullptr);
    const int rc = veltra_server_main(static_cast<int>(all.size()), argv.data());
    std::fflush(nullptr);
    return rc;
}
