// engine/cluster.cpp - see cluster.h. Compute selection store (compute.json in the app data dir), node health pings and
// a small blocking HTTP/1.1 client (non-blocking connect with a deadline, then blocking I/O with socket timeouts).
#include "cluster.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <thread>

#include "http.h"
#include "platform.h"

#if !defined(_WIN32)
#  include <fcntl.h>
#endif

namespace veltra::cluster {
namespace {

using Clock = std::chrono::steady_clock;
using lbhttp::Json;
using lbhttp::sock_t;

double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

bool set_nonblocking(sock_t s, bool on) {
#if defined(_WIN32)
    u_long m = on ? 1 : 0;
    return ioctlsocket(s, FIONBIO, &m) == 0;
#else
    const int fl = fcntl(s, F_GETFL, 0);
    if (fl < 0) return false;
    return fcntl(s, F_SETFL, on ? (fl | O_NONBLOCK) : (fl & ~O_NONBLOCK)) == 0;
#endif
}

bool connect_in_progress() {
#if defined(_WIN32)
    const int e = WSAGetLastError();
    return e == WSAEWOULDBLOCK || e == WSAEINPROGRESS;
#else
    return errno == EINPROGRESS || errno == EINTR;
#endif
}

// Waits until `s` is writable (connect finished) or `ms` elapsed. True when connected without error.
bool wait_connected(sock_t s, int ms, std::string& err) {
#if defined(_WIN32)
    fd_set wr, ex;
    FD_ZERO(&wr);
    FD_ZERO(&ex);
    FD_SET(s, &wr);
    FD_SET(s, &ex);
    timeval tv{ms / 1000, (ms % 1000) * 1000};
    const int r = select(0, nullptr, &wr, &ex, &tv);
#else
    pollfd pfd{};
    pfd.fd = s;
    pfd.events = POLLOUT;
    int r;
    do r = poll(&pfd, 1, ms); while (r < 0 && errno == EINTR);
#endif
    if (r == 0) {
        err = "connect timed out";
        return false;
    }
    if (r < 0) {
        err = "connect: " + lbhttp::last_socket_error();
        return false;
    }
    int so = 0;
    socklen_t len = sizeof so;
    if (getsockopt(s, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&so), &len) != 0 || so != 0) {
#if defined(_WIN32)
        err = "connection refused or unreachable (error " + std::to_string(so) + ")";
#else
        err = std::string("connect: ") + std::strerror(so ? so : errno);
#endif
        return false;
    }
    return true;
}

sock_t connect_deadline(const std::string& host, int port, Clock::time_point deadline, std::string& err) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || !res) {
        err = "cannot resolve " + host;
        return lbhttp::kBadSock;
    }
    sock_t out = lbhttp::kBadSock;
    err = "no address";
    for (addrinfo* ai = res; ai && out == lbhttp::kBadSock; ai = ai->ai_next) {
        const int left = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count());
        if (left <= 0) {
            err = "connect timed out";
            break;
        }
        sock_t s = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (s == lbhttp::kBadSock) {
            err = "socket: " + lbhttp::last_socket_error();
            continue;
        }
        bool ok = set_nonblocking(s, true);
        if (ok && ::connect(s, ai->ai_addr, static_cast<socklen_t>(ai->ai_addrlen)) != 0)
            ok = connect_in_progress() ? wait_connected(s, left, err) : (err = "connect: " + lbhttp::last_socket_error(), false);
        if (ok) ok = set_nonblocking(s, false);
        if (ok) out = s;
        else lbhttp::close_sock(s);
    }
    freeaddrinfo(res);
    return out;
}

bool dechunk_body(const std::string& in, std::string& out) {
    out.clear();
    size_t pos = 0;
    for (;;) {
        const size_t eol = in.find("\r\n", pos);
        if (eol == std::string::npos) return false;
        std::string_view line(in.data() + pos, eol - pos);
        const size_t semi = line.find(';');
        if (semi != std::string_view::npos) line = line.substr(0, semi);
        line = lbhttp::trim_ws(line);
        if (line.empty() || line.size() > 8) return false;
        size_t n = 0;
        for (char ch : line) {
            const int v = lbhttp::hex_value(ch);
            if (v < 0) return false;
            n = n * 16 + static_cast<size_t>(v);
        }
        pos = eol + 2;
        if (n == 0) return true;
        if (in.size() < pos + n + 2) return false;
        out.append(in, pos, n);
        pos += n + 2;
    }
}

bool valid_hostname(const std::string& h) {
    if (h.empty() || h.size() > 253 || h.front() == '-' || h.front() == '.' || h.back() == '-') return false;
    size_t label = 0;
    for (char c : h) {
        if (c == '.') {
            if (label == 0) return false;
            label = 0;
            continue;
        }
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-')) return false;
        if (++label > 63) return false;
    }
    return true;
}

bool all_digits_dots(const std::string& h) {
    return !h.empty() && std::all_of(h.begin(), h.end(), [](char c) { return (c >= '0' && c <= '9') || c == '.'; });
}

// ---- selection store --------------------------------------------------------------------------

std::mutex g_mu;
bool g_loaded = false;
Selection g_sel;

std::string compute_path() {
    const std::string d = veltra::plat::app_data_dir();
    if (d.empty()) return {};
#if defined(_WIN32)
    return d + "\\compute.json";
#else
    return d + "/compute.json";
#endif
}

void load_locked() {
    if (g_loaded) return;
    g_loaded = true;
    const std::string path = compute_path();
    if (path.empty()) return;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return;
    std::string text;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0 && text.size() < 65536) text.append(buf, n);
    std::fclose(f);
    Selection s;
    std::string err;
    // GPU indices are re-checked against the current machine on every select; here any index list is kept.
    if (parse_selection(text, 1 << 20, s, err)) g_sel = s;
}

bool write_atomic(const std::string& path, const std::string& data) {
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

// ---- routing cache ----------------------------------------------------------------------------

constexpr int kCacheMs = 5000;
std::mutex g_cache_mu;
std::vector<NodeStatus> g_cache;
Clock::time_point g_cache_at;
std::atomic<unsigned> g_rr{0};

NodeStatus ping_one(const std::string& node) {
    NodeStatus st;
    st.node = node;
    std::string host, err;
    int port = 0;
    if (!parse_node(node, &host, &port, &err)) {
        st.error = err;
        return st;
    }
    const HttpResult r = http_request(host, port, "GET", "/health", "", "Accept: application/json\r\n", kPingTimeoutMs, 64 * 1024);
    st.latency_ms = r.ms;
    if (r.status == 0) {
        st.error = r.err;
        return st;
    }
    if (r.status != 200) {
        st.error = "health answered HTTP " + std::to_string(r.status);
        return st;
    }
    st.online = true;
    Json j;
    std::string jerr;
    if (lbhttp::parse_json(r.body, j, jerr)) {
        const Json* m = j.get("model");
        if (m && m->kind == Json::Kind::String) st.model = m->str;
    }
    return st;
}

}  // namespace

bool parse_node(const std::string& s, std::string* host, int* port, std::string* err) {
    auto fail = [&](const std::string& m) {
        if (err) *err = "node \"" + s.substr(0, 80) + "\": " + m;
        return false;
    };
    if (s.empty() || s.size() > 270) return fail("expected host:port");
    std::string h, p;
    if (s.front() == '[') {
        const size_t rb = s.find(']');
        if (rb == std::string::npos || rb + 1 >= s.size() || s[rb + 1] != ':') return fail("IPv6 needs the form [addr]:port");
        h = s.substr(1, rb - 1);
        p = s.substr(rb + 2);
        in6_addr a6{};
        if (h.empty() || inet_pton(AF_INET6, h.c_str(), &a6) != 1) return fail("invalid IPv6 address");
    } else {
        const size_t colon = s.rfind(':');
        if (colon == std::string::npos || colon == 0) return fail("expected host:port");
        h = s.substr(0, colon);
        p = s.substr(colon + 1);
        if (h.find(':') != std::string::npos) return fail("put an IPv6 address in brackets: [addr]:port");
        if (all_digits_dots(h)) {
            in_addr a4{};
            if (inet_pton(AF_INET, h.c_str(), &a4) != 1) return fail("invalid IPv4 address");
        } else if (!valid_hostname(h)) {
            return fail("invalid host name");
        }
    }
    if (p.empty() || p.size() > 5 || !std::all_of(p.begin(), p.end(), [](char c) { return c >= '0' && c <= '9'; }))
        return fail("port must be 1..65535");
    const int v = std::stoi(p);
    if (v < 1 || v > 65535) return fail("port must be 1..65535");
    if (host) *host = h;
    if (port) *port = v;
    return true;
}

bool parse_selection(const std::string& body, int n_gpus, Selection& out, std::string& err) {
    Json root;
    std::string jerr;
    if (!lbhttp::parse_json(body, root, jerr)) {
        err = "invalid JSON: " + jerr;
        return false;
    }
    if (root.kind != Json::Kind::Object) {
        err = "body must be a JSON object {\"mode\": \"gpu\"|\"cpu\"|\"cluster\", ...}";
        return false;
    }
    Selection s;
    const Json* m = root.get("mode");
    if (!m || m->kind != Json::Kind::String || (m->str != "gpu" && m->str != "cpu" && m->str != "cluster")) {
        err = "\"mode\" must be \"gpu\", \"cpu\" or \"cluster\"";
        return false;
    }
    s.mode = m->str;
    if (const Json* g = root.get("gpu_indices"); g && g->kind != Json::Kind::Null) {
        if (g->kind != Json::Kind::Array || g->items.size() > 16) {
            err = "\"gpu_indices\" must be an array of at most 16 GPU indices";
            return false;
        }
        for (const Json& it : g->items) {
            int64_t v = -1;
            if (!it.as_int(v) || v < 0 || v >= n_gpus) {
                err = "gpu index out of range (this machine has " + std::to_string(n_gpus) + " GPU(s); see GET /api/hardware)";
                return false;
            }
            if (std::find(s.gpu_indices.begin(), s.gpu_indices.end(), static_cast<int>(v)) == s.gpu_indices.end())
                s.gpu_indices.push_back(static_cast<int>(v));
        }
    }
    if (const Json* n = root.get("nodes"); n && n->kind != Json::Kind::Null) {
        if (n->kind != Json::Kind::Array) {
            err = "\"nodes\" must be an array of \"host:port\" strings";
            return false;
        }
        for (const Json& it : n->items) {
            if (it.kind != Json::Kind::String) {
                err = "\"nodes\" must be an array of \"host:port\" strings";
                return false;
            }
            if (!parse_node(it.str, nullptr, nullptr, &err)) return false;
            if (std::find(s.nodes.begin(), s.nodes.end(), it.str) == s.nodes.end()) s.nodes.push_back(it.str);
            if (static_cast<int>(s.nodes.size()) > kMaxNodes) {
                err = "at most " + std::to_string(kMaxNodes) + " cluster nodes";
                return false;
            }
        }
    }
    if (s.mode == "gpu") {
        if (n_gpus <= 0) {
            err = "no GPU detected on this machine (GET /api/hardware)";
            return false;
        }
        if (s.gpu_indices.empty()) s.gpu_indices.push_back(0);
    }
    if (s.mode == "cluster" && s.nodes.empty()) {
        err = "mode \"cluster\" needs at least one node \"host:port\"";
        return false;
    }
    out = s;
    return true;
}

Selection current() {
    std::lock_guard<std::mutex> lk(g_mu);
    load_locked();
    return g_sel;
}

std::string set_current(const Selection& s) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_loaded = true;
    g_sel = s;
    {
        std::lock_guard<std::mutex> ck(g_cache_mu);
        g_cache.clear();
    }
    const std::string path = compute_path();
    if (path.empty()) return "no writable app data directory; selection kept in memory only";
    if (!write_atomic(path, selection_json(s) + "\n")) return "cannot write " + path + "; selection kept in memory only";
    return {};
}

std::string selection_json(const Selection& s) {
    std::string o = "{\"mode\":" + lbhttp::json_quote(s.mode) + ",\"gpu_indices\":[";
    for (size_t i = 0; i < s.gpu_indices.size(); i++) o += (i ? "," : "") + std::to_string(s.gpu_indices[i]);
    o += "],\"nodes\":[";
    for (size_t i = 0; i < s.nodes.size(); i++) o += (i ? "," : "") + lbhttp::json_quote(s.nodes[i]);
    return o + "]}";
}

HttpResult http_request(const std::string& host, int port, const std::string& method, const std::string& path,
                        const std::string& body, const std::string& extra_headers, int timeout_ms, size_t max_bytes) {
    HttpResult r;
    const auto t0 = Clock::now();
    const auto deadline = t0 + std::chrono::milliseconds(timeout_ms);
    const sock_t s = connect_deadline(host, port, deadline, r.err);
    if (s == lbhttp::kBadSock) {
        r.ms = ms_since(t0);
        return r;
    }
    const auto left_ms = [&] {
        return std::max<int>(1, static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count()));
    };
    lbhttp::set_timeouts(s, left_ms(), left_ms());
    const std::string host_hdr = (host.find(':') != std::string::npos ? "[" + host + "]" : host) + ":" + std::to_string(port);
    std::string req = method + " " + path + " HTTP/1.1\r\nHost: " + host_hdr + "\r\nConnection: close\r\n" + extra_headers;
    if (!body.empty() || method == "POST") req += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    req += "\r\n";
    req += body;
    std::string raw;
    bool ok = lbhttp::send_all(s, req);
    if (!ok) r.err = "send failed";
    char buf[65536];
    while (ok) {
        if (Clock::now() >= deadline) {
            ok = false;
            r.err = "timed out";
            break;
        }
        lbhttp::set_timeouts(s, left_ms(), left_ms());
        const long n = lbhttp::recv_some(s, buf, sizeof buf);
        if (n == 0) break;
        if (n < 0) {
            ok = false;
            r.err = "receive failed or timed out";
            break;
        }
        if (raw.size() + static_cast<size_t>(n) > max_bytes) {
            ok = false;
            r.err = "response too large";
            break;
        }
        raw.append(buf, static_cast<size_t>(n));
    }
    lbhttp::close_sock(s);
    r.ms = ms_since(t0);
    if (!ok) return r;
    const size_t hdr_end = raw.find("\r\n\r\n");
    int status = 0;
    if (hdr_end != std::string::npos && raw.rfind("HTTP/1.", 0) == 0 && raw.size() > 12) {
        const std::string code = raw.substr(9, 3);
        if (std::all_of(code.begin(), code.end(), [](char c) { return c >= '0' && c <= '9'; })) status = std::stoi(code);
    }
    if (status < 100 || status > 599) {
        r.err = "no valid HTTP response";
        return r;
    }
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
            if (name == "content-type") r.content_type = std::string(val);
            else if (name == "transfer-encoding") chunked = lbhttp::lower_ascii(val).find("chunked") != std::string::npos;
            else if (name == "content-length") {
                size_t v = 0;
                if (lbhttp::parse_decimal(val, v)) clen = static_cast<long long>(v);
            }
        }
        ls = le + 2;
    }
    r.body = raw.substr(hdr_end + 4);
    if (chunked) {
        std::string dec;
        if (!dechunk_body(r.body, dec)) {
            r.err = "malformed chunked body";
            r.body.clear();
            return r;
        }
        r.body.swap(dec);
    } else if (clen >= 0 && static_cast<size_t>(clen) < r.body.size()) {
        r.body.resize(static_cast<size_t>(clen));
    }
    r.status = status;
    return r;
}

std::vector<NodeStatus> ping_all(const std::vector<std::string>& nodes) {
    std::vector<NodeStatus> out(nodes.size());
    std::vector<std::thread> th;
    th.reserve(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++) th.emplace_back([&out, &nodes, i] { out[i] = ping_one(nodes[i]); });
    for (auto& t : th) t.join();
    std::lock_guard<std::mutex> ck(g_cache_mu);
    g_cache = out;
    g_cache_at = Clock::now();
    return out;
}

std::string node_status_json(const NodeStatus& s) {
    char lat[32];
    std::snprintf(lat, sizeof lat, "%.1f", s.latency_ms);
    return "{\"node\":" + lbhttp::json_quote(s.node) + ",\"status\":" + (s.online ? "\"online\"" : "\"offline\"") +
           ",\"latency_ms\":" + (s.online ? std::string(lat) : std::string("null")) +
           ",\"model\":" + (s.model.empty() ? std::string("null") : lbhttp::json_quote(s.model)) +
           (s.error.empty() ? std::string() : ",\"error\":" + lbhttp::json_quote(s.error)) + "}";
}

std::string pick_online_node(const std::vector<std::string>& nodes) {
    if (nodes.empty()) return {};
    std::vector<NodeStatus> st;
    {
        std::lock_guard<std::mutex> ck(g_cache_mu);
        bool fresh = g_cache.size() == nodes.size() &&
                     std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - g_cache_at).count() < kCacheMs;
        for (size_t i = 0; fresh && i < nodes.size(); i++) fresh = g_cache[i].node == nodes[i];
        if (fresh) st = g_cache;
    }
    if (st.empty()) st = ping_all(nodes);
    std::vector<std::string> online;
    for (const NodeStatus& s : st)
        if (s.online) online.push_back(s.node);
    if (online.empty()) return {};
    return online[g_rr++ % online.size()];
}

void mark_offline(const std::string& node) {
    std::lock_guard<std::mutex> ck(g_cache_mu);
    for (NodeStatus& s : g_cache)
        if (s.node == node) {
            s.online = false;
            s.error = "request failed";
        }
}

}  // namespace veltra::cluster
