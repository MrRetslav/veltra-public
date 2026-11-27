// engine/http.h - header-only plumbing for engine/server.cpp (no third-party code).
//   * sockets: Winsock2 on Windows, BSD sockets on POSIX (Linux, Android)
//   * HTTP/1.1 request reader: request line, headers, Content-Length or chunked bodies and
//     Expect: 100-continue; size limits are enforced while reading
//   * response helpers: plain responses and chunked framing (used for SSE)
//   * a small JSON DOM parser (RFC 8259, depth-limited) and a UTF-8-safe JSON string escaper
// Everything is `inline`; include it from one translation unit.
#pragma once

#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "platform.h"

#if defined(_WIN32)
#  ifndef _WIN32_WINNT
#    define _WIN32_WINNT 0x0A00
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  if defined(_MSC_VER)
#    pragma comment(lib, "ws2_32")  // other toolchains: link ws2_32 explicitly
#  endif
#  define LBH_WINDOWS 1
#else
#  include <arpa/inet.h>
#  include <csignal>
#  include <netdb.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <poll.h>
#  include <sys/socket.h>
#  include <sys/types.h>
#  include <unistd.h>
#  define LBH_WINDOWS 0
#endif

namespace lbhttp {

#if LBH_WINDOWS
using sock_t = SOCKET;
inline constexpr sock_t kBadSock = INVALID_SOCKET;
inline constexpr int kShutWr = SD_SEND;
#else
using sock_t = int;
inline constexpr sock_t kBadSock = -1;
inline constexpr int kShutWr = SHUT_WR;
#endif

inline constexpr size_t kMaxLineBytes = 8 * 1024;        // one request line or header line
inline constexpr size_t kMaxHeaderBytes = 16 * 1024;     // all header lines together
inline constexpr size_t kMaxBodyBytes = 1024 * 1024;     // request body
inline constexpr size_t kMaxLargeBodyBytes = 2 * 1024 * 1024;   // request body of POST /api/chats (one stored chat)
inline constexpr int kRecvTimeoutMs = 15000;
inline constexpr int kSendTimeoutMs = 30000;

// Allowed CORS origin: "*" by default; a server started with --require-token pins it to its own origin
// (set once before the first connection is accepted, read-only afterwards).
inline std::string& cors_allow_origin() {
    static std::string origin = "*";
    return origin;
}
inline std::string cors_headers() {
    std::string h = "Access-Control-Allow-Origin: " + cors_allow_origin() + "\r\n";
    if (cors_allow_origin() != "*") h += "Vary: Origin\r\n";
    h += "Access-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n"
         "Access-Control-Allow-Headers: Content-Type, Authorization, X-Veltra-Token\r\n"
         "Access-Control-Max-Age: 86400\r\n";
    return h;
}

// ---- sockets ---------------------------------------------------------------------------------

inline bool net_startup(std::string* err) {
    return veltra::plat::net_init(err);   // WSAStartup on Windows, SIGPIPE ignore on POSIX (engine/platform.cpp)
}

inline std::string last_socket_error() {
#if LBH_WINDOWS
    return "WSA error " + std::to_string(WSAGetLastError());
#else
    return std::strerror(errno);
#endif
}

inline void close_sock(sock_t s) {
#if LBH_WINDOWS
    closesocket(s);
#else
    ::close(s);
#endif
}

inline void set_timeouts(sock_t s, int recv_ms, int send_ms) {
#if LBH_WINDOWS
    DWORD r = static_cast<DWORD>(recv_ms), w = static_cast<DWORD>(send_ms);
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&r), sizeof r);
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&w), sizeof w);
#else
    timeval r{recv_ms / 1000, (recv_ms % 1000) * 1000};
    timeval w{send_ms / 1000, (send_ms % 1000) * 1000};
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &r, sizeof r);
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &w, sizeof w);
#endif
}

inline void set_nodelay(sock_t s) {
    int one = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof one);
}

// Binds host:port (port 0 lets the OS choose; *bound_port receives the real port) and listens.
inline sock_t listen_tcp(const std::string& host, int port, int* bound_port, std::string* err) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    addrinfo* res = nullptr;
    const std::string port_str = std::to_string(port);
    const int rc = getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res);
    if (rc != 0) {
        if (err) *err = "cannot resolve " + host + ": " + gai_strerror(rc);
        return kBadSock;
    }
    sock_t s = kBadSock;
    std::string why = "no usable address";
    for (addrinfo* ai = res; ai && s == kBadSock; ai = ai->ai_next) {
        s = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (s == kBadSock) {
            why = last_socket_error();
            continue;
        }
#if !LBH_WINDOWS
        int one = 1;
        setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
#endif
        if (bind(s, ai->ai_addr, static_cast<socklen_t>(ai->ai_addrlen)) != 0 || listen(s, 64) != 0) {
            why = last_socket_error();
            close_sock(s);
            s = kBadSock;
        }
    }
    freeaddrinfo(res);
    if (s == kBadSock) {
        if (err) *err = "cannot listen on " + host + ":" + port_str + " (" + why + ")";
        return kBadSock;
    }
    sockaddr_storage ss{};
    socklen_t len = sizeof ss;
    getsockname(s, reinterpret_cast<sockaddr*>(&ss), &len);
    if (bound_port) {
        *bound_port = ss.ss_family == AF_INET6
                          ? ntohs(reinterpret_cast<sockaddr_in6*>(&ss)->sin6_port)
                          : ntohs(reinterpret_cast<sockaddr_in*>(&ss)->sin_port);
    }
    return s;
}

inline sock_t accept_client(sock_t listener) { return ::accept(listener, nullptr, nullptr); }

// True when the connected peer has a loopback address (127.0.0.0/8, ::1 or ::ffff:127.x.x.x).
inline bool peer_is_loopback(sock_t s) {
    sockaddr_storage ss{};
#if LBH_WINDOWS
    int len = static_cast<int>(sizeof ss);
#else
    socklen_t len = sizeof ss;
#endif
    if (::getpeername(s, reinterpret_cast<sockaddr*>(&ss), &len) != 0) return false;
    if (ss.ss_family == AF_INET) {
        const auto* a = reinterpret_cast<const sockaddr_in*>(&ss);
        return (ntohl(a->sin_addr.s_addr) >> 24) == 127;
    }
    if (ss.ss_family == AF_INET6) {
        const unsigned char* b = reinterpret_cast<const sockaddr_in6*>(&ss)->sin6_addr.s6_addr;
        static constexpr unsigned char kLo[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
        static constexpr unsigned char kMapped[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};
        return std::memcmp(b, kLo, 16) == 0 || (std::memcmp(b, kMapped, 12) == 0 && b[12] == 127);
    }
    return false;
}

// Sends everything or reports failure (peer gone, timeout, reset).
inline bool send_all(sock_t s, const void* data, size_t n) {
    const char* p = static_cast<const char*>(data);
    while (n > 0) {
#if LBH_WINDOWS
        const int chunk = n > 0x40000000u ? 0x40000000 : static_cast<int>(n);
        const int w = ::send(s, p, chunk, 0);
        if (w == SOCKET_ERROR) return false;
#else
        const ssize_t w = ::send(s, p, n, MSG_NOSIGNAL);
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
#endif
        if (w == 0) return false;
        p += w;
        n -= static_cast<size_t>(w);
    }
    return true;
}

inline bool send_all(sock_t s, std::string_view data) { return send_all(s, data.data(), data.size()); }

// >0: bytes read, 0: peer closed, <0: error or timeout.
inline long recv_some(sock_t s, char* buf, size_t cap) {
#if LBH_WINDOWS
    return ::recv(s, buf, static_cast<int>(cap), 0);
#else
    for (;;) {
        const ssize_t n = ::recv(s, buf, cap, 0);
        if (n < 0 && errno == EINTR) continue;
        return static_cast<long>(n);
    }
#endif
}

// True when the peer has closed the connection (FIN or reset). Never blocks. Bytes that are
// merely waiting to be read (a pipelined request, say) count as "still here".
inline bool peer_gone(sock_t s) {
#if LBH_WINDOWS
    WSAPOLLFD pfd{};
    pfd.fd = s;
    pfd.events = POLLRDNORM;
    const int ready = WSAPoll(&pfd, 1, 0);
#else
    pollfd pfd{};
    pfd.fd = s;
    pfd.events = POLLIN;
    const int ready = poll(&pfd, 1, 0);
#endif
    if (ready <= 0) return false;
    if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) return true;
    char c = 0;
#if LBH_WINDOWS
    const int n = ::recv(s, &c, 1, MSG_PEEK);
#else
    const ssize_t n = ::recv(s, &c, 1, MSG_PEEK);
#endif
    return n <= 0;
}

// Ends the response: shuts down the write side, optionally drains unread input for a moment (so the
// peer receives our reply instead of a TCP reset), then closes.
inline void graceful_close(sock_t s, bool drain) {
    if (s == kBadSock) return;
    ::shutdown(s, kShutWr);
    if (drain) {
        set_timeouts(s, 200, 200);
        char buf[4096];
        size_t total = 0;
        while (total < 256 * 1024) {
            const long n = recv_some(s, buf, sizeof buf);
            if (n <= 0) break;
            total += static_cast<size_t>(n);
        }
    }
    close_sock(s);
}

// ---- small string helpers ----------------------------------------------------------------------

inline std::string lower_ascii(std::string_view s) {
    std::string r(s);
    for (char& c : r)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return r;
}

inline std::string_view trim_ws(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.remove_suffix(1);
    return s;
}

inline bool parse_decimal(std::string_view s, size_t& out) {
    if (s.empty() || s.size() > 18) return false;
    size_t v = 0;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
        v = v * 10 + static_cast<size_t>(c - '0');
    }
    out = v;
    return true;
}

inline int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// ---- HTTP request ------------------------------------------------------------------------------

struct Request {
    std::string method, target, path, query, version;
    std::vector<std::pair<std::string, std::string>> headers;  // names lower-cased, values trimmed
    std::string body;

    // First header with this (lower-case) name, or "".
    std::string header(std::string_view name) const {
        for (const auto& [k, v] : headers)
            if (k == name) return v;
        return {};
    }
};

// status 0 means "no response": the peer went away or the read timed out.
struct HttpError {
    int status = 0;
    std::string message;
};

// Buffered reader over a socket. Read-only; bytes beyond what the caller asks for stay buffered.
class SockReader {
public:
    enum class Line { Ok, Closed, TooLong };

    explicit SockReader(sock_t s) : s_(s) {}

    // One line without its CRLF (a bare LF is accepted too).
    Line read_line(std::string& out, size_t max) {
        for (;;) {
            const size_t nl = buf_.find('\n', pos_);
            if (nl != std::string::npos) {
                size_t end = nl;
                if (end > pos_ && buf_[end - 1] == '\r') --end;
                if (end - pos_ > max) return Line::TooLong;
                out.assign(buf_, pos_, end - pos_);
                pos_ = nl + 1;
                return Line::Ok;
            }
            if (buf_.size() - pos_ > max + 2) return Line::TooLong;
            if (!fill()) return Line::Closed;
        }
    }

    bool read_exact(size_t n, std::string& out) {
        while (buf_.size() - pos_ < n) {
            if (!fill()) return false;
        }
        out.assign(buf_, pos_, n);
        pos_ += n;
        return true;
    }

private:
    bool fill() {
        if (pos_ > 0) {
            buf_.erase(0, pos_);
            pos_ = 0;
        }
        char tmp[8192];
        const long n = recv_some(s_, tmp, sizeof tmp);
        if (n <= 0) return false;
        buf_.append(tmp, static_cast<size_t>(n));
        return true;
    }

    sock_t s_;
    std::string buf_;
    size_t pos_ = 0;
};

inline bool read_chunked_body(SockReader& rd, std::string& body, HttpError& err, size_t max_body = kMaxBodyBytes) {
    std::string line;
    for (;;) {
        if (rd.read_line(line, 256) != SockReader::Line::Ok) return false;
        const std::string_view size_text = trim_ws(std::string_view(line).substr(0, line.find(';')));
        if (size_text.empty() || size_text.size() > 8) {
            err = {400, "malformed chunk size"};
            return false;
        }
        size_t size = 0;
        for (char c : size_text) {
            const int d = hex_value(c);
            if (d < 0) {
                err = {400, "malformed chunk size"};
                return false;
            }
            size = size * 16 + static_cast<size_t>(d);
        }
        if (size == 0) break;
        if (body.size() + size > max_body) {
            err = {413, max_body == kMaxBodyBytes ? "request body exceeds 1 MiB" : "request body exceeds 2 MiB"};
            return false;
        }
        std::string part;
        if (!rd.read_exact(size, part)) return false;
        body += part;
        std::string crlf;
        if (rd.read_line(crlf, 0) != SockReader::Line::Ok) {
            err = {400, "malformed chunk framing"};
            return false;
        }
    }
    // Trailer section: ends with an empty line.
    for (;;) {
        if (rd.read_line(line, kMaxLineBytes) != SockReader::Line::Ok) return false;
        if (line.empty()) return true;
    }
}

// Reads one request. On false, err.status tells the caller what to send (0 = nothing).
inline bool read_request(sock_t s, Request& req, HttpError& err) {
    SockReader rd(s);
    using L = SockReader::Line;
    std::string line;

    switch (rd.read_line(line, kMaxLineBytes)) {
        case L::Ok: break;
        case L::TooLong: err = {414, "request line too long"}; return false;
        case L::Closed: return false;
    }
    const size_t sp1 = line.find(' ');
    const size_t sp2 = sp1 == std::string::npos ? std::string::npos : line.find(' ', sp1 + 1);
    if (sp2 == std::string::npos) {
        err = {400, "malformed request line"};
        return false;
    }
    req.method = line.substr(0, sp1);
    req.target = line.substr(sp1 + 1, sp2 - sp1 - 1);
    req.version = line.substr(sp2 + 1);
    if (req.version.rfind("HTTP/1.", 0) != 0) {
        err = {505, "only HTTP/1.x is supported"};
        return false;
    }
    if (req.target.empty() || (req.target[0] != '/' && req.target != "*")) {
        err = {400, "malformed request target"};
        return false;
    }
    const size_t q = req.target.find('?');
    req.path = req.target.substr(0, q);
    req.query = q == std::string::npos ? std::string() : req.target.substr(q + 1);

    size_t header_bytes = 0;
    for (;;) {
        const L r = rd.read_line(line, kMaxLineBytes);
        if (r == L::Closed) return false;
        if (r == L::TooLong) {
            err = {431, "header line too long"};
            return false;
        }
        if (line.empty()) break;
        header_bytes += line.size() + 2;
        if (header_bytes > kMaxHeaderBytes) {
            err = {431, "headers too large"};
            return false;
        }
        const size_t colon = line.find(':');
        if (colon == std::string::npos || colon == 0) {
            err = {400, "malformed header line"};
            return false;
        }
        req.headers.emplace_back(lower_ascii(trim_ws(std::string_view(line).substr(0, colon))),
                                 std::string(trim_ws(std::string_view(line).substr(colon + 1))));
    }

    const size_t max_body = req.path == "/api/chats" ? kMaxLargeBodyBytes : kMaxBodyBytes;
    const bool chunked = lower_ascii(req.header("transfer-encoding")).find("chunked") != std::string::npos;
    const bool expect_continue = lower_ascii(req.header("expect")).find("100-continue") != std::string::npos;
    const std::string cl = req.header("content-length");

    if (chunked) {
        if (expect_continue && !send_all(s, std::string_view("HTTP/1.1 100 Continue\r\n\r\n")))
            return false;
        return read_chunked_body(rd, req.body, err, max_body);
    }
    if (!cl.empty()) {
        size_t n = 0;
        if (!parse_decimal(cl, n)) {
            err = {400, "invalid Content-Length"};
            return false;
        }
        if (n > max_body) {
            err = {413, max_body == kMaxBodyBytes ? "request body exceeds 1 MiB" : "request body exceeds 2 MiB"};
            return false;
        }
        if (n > 0) {
            if (expect_continue && !send_all(s, std::string_view("HTTP/1.1 100 Continue\r\n\r\n")))
                return false;
            if (!rd.read_exact(n, req.body)) return false;
        }
        return true;
    }
    if (req.method == "POST" || req.method == "PUT") {
        err = {411, "Content-Length required"};
        return false;
    }
    return true;
}

// ---- HTTP response -----------------------------------------------------------------------------

inline const char* reason_phrase(int status) {
    switch (status) {
        case 200: return "OK";
        case 204: return "No Content";
        case 400: return "Bad Request";
        case 403: return "Forbidden";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 409: return "Conflict";
        case 411: return "Length Required";
        case 413: return "Payload Too Large";
        case 414: return "URI Too Long";
        case 431: return "Request Header Fields Too Large";
        case 500: return "Internal Server Error";
        case 502: return "Bad Gateway";
        case 503: return "Service Unavailable";
        case 504: return "Gateway Timeout";
        case 505: return "HTTP Version Not Supported";
        default: return "Unknown";
    }
}

// Complete response with Content-Length. `extra_headers` must be full "Name: value\r\n" lines.
inline bool send_response(sock_t s, int status, std::string_view content_type, std::string_view body,
                          std::string_view extra_headers = {}) {
    std::string h;
    h.reserve(256 + body.size());
    h += "HTTP/1.1 ";
    h += std::to_string(status);
    h += ' ';
    h += reason_phrase(status);
    h += "\r\n";
    if (status != 204) {  // RFC 9110: no Content-Length on 204
        h += "Content-Type: ";
        h += content_type;
        h += "\r\nContent-Length: ";
        h += std::to_string(body.size());
        h += "\r\n";
    }
    h += "Connection: close\r\n";
    h += cors_headers();
    h += extra_headers;
    h += "\r\n";
    h += body;
    return send_all(s, h);
}

// Head of a chunked text/event-stream response.
inline bool send_sse_head(sock_t s) {
    std::string h =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream; charset=utf-8\r\n"
        "Cache-Control: no-cache\r\n"
        "Transfer-Encoding: chunked\r\n"
        "Connection: close\r\n";
    h += cors_headers();
    h += "\r\n";
    return send_all(s, h);
}

// One HTTP chunk. Empty data is skipped (a zero-length chunk would end the body).
inline bool send_chunk(sock_t s, std::string_view data) {
    if (data.empty()) return true;
    char size[32];
    const int n = std::snprintf(size, sizeof size, "%llx\r\n", static_cast<unsigned long long>(data.size()));
    std::string frame;
    frame.reserve(static_cast<size_t>(n) + data.size() + 2);
    frame.append(size, static_cast<size_t>(n));
    frame.append(data);
    frame += "\r\n";
    return send_all(s, frame);
}

inline bool send_last_chunk(sock_t s) { return send_all(s, std::string_view("0\r\n\r\n", 5)); }

// ---- UTF-8 and JSON output ---------------------------------------------------------------------

inline bool utf8_cont(std::string_view s, size_t i, unsigned char lo = 0x80, unsigned char hi = 0xBF) {
    if (i >= s.size()) return false;
    const unsigned char c = static_cast<unsigned char>(s[i]);
    return c >= lo && c <= hi;
}

// Length of the well-formed UTF-8 sequence starting at s[i] (RFC 3629), or 0 if it is not one.
inline size_t utf8_seq_len(std::string_view s, size_t i) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c >= 0xC2 && c <= 0xDF) return utf8_cont(s, i + 1) ? 2 : 0;
    if (c == 0xE0) return (utf8_cont(s, i + 1, 0xA0, 0xBF) && utf8_cont(s, i + 2)) ? 3 : 0;
    if ((c >= 0xE1 && c <= 0xEC) || c == 0xEE || c == 0xEF)
        return (utf8_cont(s, i + 1) && utf8_cont(s, i + 2)) ? 3 : 0;
    if (c == 0xED) return (utf8_cont(s, i + 1, 0x80, 0x9F) && utf8_cont(s, i + 2)) ? 3 : 0;
    if (c == 0xF0)
        return (utf8_cont(s, i + 1, 0x90, 0xBF) && utf8_cont(s, i + 2) && utf8_cont(s, i + 3)) ? 4 : 0;
    if (c >= 0xF1 && c <= 0xF3)
        return (utf8_cont(s, i + 1) && utf8_cont(s, i + 2) && utf8_cont(s, i + 3)) ? 4 : 0;
    if (c == 0xF4)
        return (utf8_cont(s, i + 1, 0x80, 0x8F) && utf8_cont(s, i + 2) && utf8_cont(s, i + 3)) ? 4 : 0;
    return 0;
}

// Appends s as the body of a JSON string (no quotes). Control characters are escaped, valid UTF-8 is
// copied through, and invalid bytes become U+FFFD so the output is always valid JSON.
inline void json_escape_append(std::string& out, std::string_view s) {
    static constexpr char kHex[] = "0123456789abcdef";
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c >= 0x80) {
            const size_t n = utf8_seq_len(s, i);
            if (n) {
                out.append(s.data() + i, n);
                i += n;
            } else {
                out += "\\ufffd";
                ++i;
            }
            continue;
        }
        ++i;
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) {
                    out += "\\u00";
                    out += kHex[c >> 4];
                    out += kHex[c & 0xF];
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
}

// Quoted JSON string.
inline std::string json_quote(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 2);
    out += '"';
    json_escape_append(out, s);
    out += '"';
    return out;
}

// JSON number; NaN and infinities become 0 (JSON cannot represent them).
inline std::string json_number(double v) {
    if (!std::isfinite(v)) return "0";
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.6g", v);
    return buf;
}

// ---- JSON input --------------------------------------------------------------------------------

struct Json {
    enum class Kind : uint8_t { Null, Bool, Number, String, Array, Object };

    Kind kind = Kind::Null;
    bool boolean = false;
    double number = 0.0;
    bool integral = false;   // literal was an integer that fits comfortably in int64
    int64_t integer = 0;
    std::string str;
    std::vector<Json> items;     // array elements, or object values (parallel to keys)
    std::vector<std::string> keys;

    // Object member (the last one wins for duplicate keys); nullptr when absent or not an object.
    const Json* get(std::string_view key) const {
        if (kind != Kind::Object) return nullptr;
        for (size_t i = keys.size(); i-- > 0;)
            if (keys[i] == key) return &items[i];
        return nullptr;
    }

    // Integer-valued number (an integral literal, or a finite double with no fraction).
    bool as_int(int64_t& out) const {
        if (kind != Kind::Number) return false;
        if (integral) {
            out = integer;
            return true;
        }
        if (!std::isfinite(number) || std::floor(number) != number || std::fabs(number) > 9e15) return false;
        out = static_cast<int64_t>(number);
        return true;
    }
};

inline void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

class JsonParser {
public:
    // `max_nodes` bounds the number of values: one Json is ~112 bytes, so a few bytes of input ("0,") can cost a lot.
    explicit JsonParser(std::string_view text, size_t max_nodes = SIZE_MAX) : s_(text), max_nodes_(max_nodes) {}

    bool parse(Json& out, std::string& err) {
        skip_ws();
        bool ok = value(out, 0);
        if (ok) {
            skip_ws();
            if (pos_ != s_.size()) ok = fail("unexpected trailing characters");
        }
        if (!ok) err = err_;
        return ok;
    }

private:
    static constexpr int kMaxDepth = 64;

    std::string_view s_;
    size_t max_nodes_;
    size_t nodes_ = 0;
    size_t pos_ = 0;
    std::string err_;

    bool fail(const char* what) {
        if (err_.empty()) err_ = std::string(what) + " at byte " + std::to_string(pos_);
        return false;
    }
    void skip_ws() {
        while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\n' || s_[pos_] == '\r'))
            ++pos_;
    }
    bool at(char c) const { return pos_ < s_.size() && s_[pos_] == c; }
    bool at_digit() const { return pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9'; }

    bool value(Json& v, int depth) {
        if (depth > kMaxDepth) return fail("nesting too deep");
        if (++nodes_ > max_nodes_) return fail("too many values");
        if (pos_ >= s_.size()) return fail("unexpected end of input");
        const char c = s_[pos_];
        if (c == '{') return object(v, depth);
        if (c == '[') return array(v, depth);
        if (c == '"') {
            v.kind = Json::Kind::String;
            return parse_string(v.str);
        }
        if (c == 't' || c == 'f' || c == 'n') return literal(v);
        if (c == '-' || (c >= '0' && c <= '9')) return number(v);
        return fail("unexpected character");
    }

    bool literal(Json& v) {
        if (s_.substr(pos_, 4) == "true") {
            pos_ += 4;
            v.kind = Json::Kind::Bool;
            v.boolean = true;
            return true;
        }
        if (s_.substr(pos_, 5) == "false") {
            pos_ += 5;
            v.kind = Json::Kind::Bool;
            v.boolean = false;
            return true;
        }
        if (s_.substr(pos_, 4) == "null") {
            pos_ += 4;
            v.kind = Json::Kind::Null;
            return true;
        }
        return fail("invalid literal");
    }

    bool object(Json& v, int depth) {
        v.kind = Json::Kind::Object;
        ++pos_;  // '{'
        skip_ws();
        if (at('}')) {
            ++pos_;
            return true;
        }
        for (;;) {
            skip_ws();
            if (!at('"')) return fail("expected object key");
            std::string key;
            if (!parse_string(key)) return false;
            skip_ws();
            if (!at(':')) return fail("expected ':'");
            ++pos_;
            skip_ws();
            Json child;
            if (!value(child, depth + 1)) return false;
            v.keys.push_back(std::move(key));
            v.items.push_back(std::move(child));
            skip_ws();
            if (at(',')) {
                ++pos_;
                continue;
            }
            if (at('}')) {
                ++pos_;
                return true;
            }
            return fail("expected ',' or '}'");
        }
    }

    bool array(Json& v, int depth) {
        v.kind = Json::Kind::Array;
        ++pos_;  // '['
        skip_ws();
        if (at(']')) {
            ++pos_;
            return true;
        }
        for (;;) {
            skip_ws();
            Json child;
            if (!value(child, depth + 1)) return false;
            v.items.push_back(std::move(child));
            skip_ws();
            if (at(',')) {
                ++pos_;
                continue;
            }
            if (at(']')) {
                ++pos_;
                return true;
            }
            return fail("expected ',' or ']'");
        }
    }

    bool hex4(uint32_t& cp) {
        if (pos_ + 4 > s_.size()) return false;
        cp = 0;
        for (int i = 0; i < 4; ++i) {
            const int d = hex_value(s_[pos_++]);
            if (d < 0) return false;
            cp = cp * 16 + static_cast<uint32_t>(d);
        }
        return true;
    }

    bool parse_string(std::string& out) {
        ++pos_;  // opening quote
        out.clear();
        for (;;) {
            if (pos_ >= s_.size()) return fail("unterminated string");
            const unsigned char c = static_cast<unsigned char>(s_[pos_++]);
            if (c == '"') return true;
            if (c < 0x20) return fail("control character in string");
            if (c != '\\') {
                out.push_back(static_cast<char>(c));
                continue;
            }
            if (pos_ >= s_.size()) return fail("unterminated escape");
            switch (s_[pos_++]) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    uint32_t cp = 0;
                    if (!hex4(cp)) return fail("bad \\u escape");
                    if (cp >= 0xD800 && cp <= 0xDBFF && s_.substr(pos_, 2) == "\\u") {
                        const size_t save = pos_;
                        pos_ += 2;
                        uint32_t lo = 0;
                        if (!hex4(lo)) return fail("bad \\u escape");
                        if (lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        } else {
                            pos_ = save;  // lone high surrogate; parse the next escape normally
                            cp = 0xFFFD;
                        }
                    } else if (cp >= 0xD800 && cp <= 0xDFFF) {
                        cp = 0xFFFD;  // lone surrogate
                    }
                    append_utf8(out, cp);
                    break;
                }
                default: return fail("bad escape in string");
            }
        }
    }

    bool number(Json& v) {
        const size_t start = pos_;
        bool integral = true;
        if (at('-')) ++pos_;
        if (at('0')) {
            ++pos_;
        } else if (at_digit()) {
            while (at_digit()) ++pos_;
        } else {
            return fail("bad number");
        }
        if (at('.')) {
            integral = false;
            ++pos_;
            if (!at_digit()) return fail("bad number");
            while (at_digit()) ++pos_;
        }
        if (at('e') || at('E')) {
            integral = false;
            ++pos_;
            if (at('+') || at('-')) ++pos_;
            if (!at_digit()) return fail("bad number");
            while (at_digit()) ++pos_;
        }
        const std::string lit(s_.substr(start, pos_ - start));
        v.kind = Json::Kind::Number;
        v.number = std::strtod(lit.c_str(), nullptr);  // the program never sets a locale, so '.' is the separator
        if (integral && lit.size() <= 18) {
            v.integral = true;
            v.integer = std::strtoll(lit.c_str(), nullptr, 10);
        }
        return true;
    }
};

// Parses a complete JSON document. On failure, err describes the problem and its byte offset.
inline bool parse_json(std::string_view text, Json& out, std::string& err, size_t max_nodes = SIZE_MAX) {
    return JsonParser(text, max_nodes).parse(out, err);
}

}  // namespace lbhttp
