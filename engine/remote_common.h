// engine/remote_common.h - validation and crypto helpers shared by the relay (engine/relay.cpp) and the outbound client
// (engine/remote_client.cpp) of the opt-in remote GPU test mode. Everything that crosses the relay is untrusted input.
// Header-only; no OS header leaks (CSPRNG lives in relay.cpp).
#pragma once
#include <cstddef>
#include <ctime>
#include <string>
#include <string_view>

#include "http.h"

namespace veltra::rc {

// Fills `buf` from the OS CSPRNG (BCryptGenRandom / /dev/urandom). False when it fails: callers must then refuse.
bool csprng(void* buf, size_t n);

inline std::string random_hex(size_t bytes) {
    unsigned char b[64];
    if (bytes > sizeof b || !csprng(b, bytes)) return {};
    static const char* k = "0123456789abcdef";
    std::string out;
    out.reserve(bytes * 2);
    for (size_t i = 0; i < bytes; ++i) {
        out += k[b[i] >> 4];
        out += k[b[i] & 15];
    }
    return out;
}

inline bool ct_equal(std::string_view a, std::string_view b) {
    const size_t n = a.size() > b.size() ? a.size() : b.size();
    unsigned char diff = a.size() == b.size() ? 0 : 1;
    for (size_t i = 0; i < n; ++i) {
        const unsigned char x = i < a.size() ? static_cast<unsigned char>(a[i]) : 0;
        const unsigned char y = i < b.size() ? static_cast<unsigned char>(b[i]) : 0;
        diff |= static_cast<unsigned char>(x ^ y);
    }
    return diff == 0;
}

inline bool is_digit(char c) { return c >= '0' && c <= '9'; }

// "VLT-" + 6 digits
inline bool valid_code(std::string_view s) {
    if (s.size() != 10 || s.substr(0, 4) != "VLT-") return false;
    for (size_t i = 4; i < 10; ++i)
        if (!is_digit(s[i])) return false;
    return true;
}

inline bool valid_job_id(std::string_view s) {
    if (s.empty() || s.size() > 64) return false;
    for (char c : s)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || is_digit(c) || c == '_' || c == '-')) return false;
    return true;
}

// 32..128 hex characters
inline bool valid_node_token(std::string_view s) {
    if (s.size() < 32 || s.size() > 128) return false;
    for (char c : s)
        if (lbhttp::hex_value(c) < 0) return false;
    return true;
}

// Copy of `s` without control characters (C0, DEL, C1) and without the Unicode bidi / line-separator controls;
// newline and tab become a space; invalid UTF-8 bytes are dropped; at most `max_bytes` bytes (never splits a character).
inline std::string clean_text(std::string_view s, size_t max_bytes) {
    std::string out;
    out.reserve(s.size() < max_bytes ? s.size() : max_bytes);
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) {
            ++i;
            char put = static_cast<char>(c);
            if (c == '\n' || c == '\t' || c == '\r') put = ' ';
            else if (c < 0x20 || c == 0x7F) continue;
            if (out.size() + 1 > max_bytes) break;
            out += put;
            continue;
        }
        const size_t len = lbhttp::utf8_seq_len(s, i);
        if (len < 2) {
            ++i;
            continue;
        }
        unsigned cp = c & (0xFF >> (len + 1));
        for (size_t k = 1; k < len; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
        const bool drop = (cp >= 0x80 && cp <= 0x9F) || (cp >= 0x202A && cp <= 0x202E) || (cp >= 0x2066 && cp <= 0x2069) ||
                          cp == 0x2028 || cp == 0x2029 || cp == 0x200E || cp == 0x200F;
        if (!drop) {
            if (out.size() + len > max_bytes) break;
            out.append(s.substr(i, len));
        }
        i += len;
    }
    return out;
}

// Re-serialises a parsed JSON value with every key and string passed through clean_text.
inline void dump_clean(std::string& out, const lbhttp::Json& j, size_t max_str = 4000) {
    using K = lbhttp::Json::Kind;
    switch (j.kind) {
        case K::Null: out += "null"; break;
        case K::Bool: out += j.boolean ? "true" : "false"; break;
        case K::Number:
            if (j.integral) {
                out += std::to_string(j.integer);
            } else if (!std::isfinite(j.number)) {
                out += "null";
            } else {
                char buf[40];
                std::snprintf(buf, sizeof buf, "%.10g", j.number);
                out += buf;
            }
            break;
        case K::String: out += lbhttp::json_quote(clean_text(j.str, max_str)); break;
        case K::Array:
            out += '[';
            for (size_t i = 0; i < j.items.size(); ++i) {
                if (i) out += ',';
                dump_clean(out, j.items[i], max_str);
            }
            out += ']';
            break;
        case K::Object:
            out += '{';
            for (size_t i = 0; i < j.items.size(); ++i) {
                if (i) out += ',';
                out += lbhttp::json_quote(clean_text(j.keys[i], 128));
                out += ':';
                dump_clean(out, j.items[i], max_str);
            }
            out += '}';
            break;
    }
}

inline std::string str_field(const lbhttp::Json& o, std::string_view key, size_t max_bytes) {
    const lbhttp::Json* v = o.get(key);
    return v && v->kind == lbhttp::Json::Kind::String ? clean_text(v->str, max_bytes) : std::string();
}

// Replaces local paths in a message with "<pad>": drive paths (C:\x), UNC paths (\\host\share), home-relative paths
// (~/x, ~\x, ~name/x) and Unix paths with at least two '/' (not units such as "tok/s"). A path with spaces is followed
// for up to three more words when a later word clearly continues it. Over-scrubbing is fine, leaking is not.
inline void scrub_paths(std::string& s) {
    auto stop = [](char c) { return c == ' ' || c == '"' || c == '\'' || c == '\n' || c == '\r' || c == '\t'; };
    auto word = [](char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || is_digit(c); };
    auto token_end = [&](size_t i) {
        while (i < s.size() && !stop(s[i])) ++i;
        return i;
    };
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i], prev = i ? s[i - 1] : ' ';
        const bool drive = i + 2 < s.size() && word(c) && !is_digit(c) && s[i + 1] == ':' && (s[i + 2] == '\\' || s[i + 2] == '/') &&
                           !word(prev);
        const bool unc = c == '\\' && i + 2 < s.size() && s[i + 1] == '\\' && s[i + 2] != '\\' && !stop(s[i + 2]);
        bool tilde = false;
        if (c == '~' && !word(prev)) {
            size_t k = i + 1;
            while (k < s.size() && (word(s[k]) || s[k] == '.' || s[k] == '_' || s[k] == '-')) ++k;
            tilde = k < s.size() && (s[k] == '/' || s[k] == '\\');
        }
        bool unix_path = s.compare(i, 6, "/home/") == 0 || s.compare(i, 6, "<home>/") == 0 || s.compare(i, 7, "/Users/") == 0;
        if (!unix_path && c == '/' && !word(prev) && prev != '.' && prev != '%' && prev != ')') {
            const size_t second = s.find('/', i + 1);
            unix_path = second != std::string::npos && second < token_end(i);
        }
        if (!drive && !unc && !tilde && !unix_path) continue;
        size_t e = token_end(i);
        for (size_t k = e, words = 0; k < s.size() && s[k] == ' ' && words < 3; ++words) {   // "C:\Users\Some User Name\x"
            const size_t from = k + 1, to = token_end(from);
            const std::string_view w = std::string_view(s).substr(from, to - from);
            const size_t sep = w.find(unix_path ? '/' : '\\');
            const bool continues = sep != std::string_view::npos &&
                                   (!unix_path || w.find('/', sep + 1) != std::string_view::npos || w.find('.', sep) != std::string_view::npos);
            if (continues) e = to;
            k = to;
        }
        s.replace(i, e - i, "<pad>");
        i += 4;
    }
}

inline std::string utc_stamp(const char* fmt) {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[40];
    std::strftime(buf, sizeof buf, fmt, &tm);
    return buf;
}

// ---- relay URL (client side) ----------------------------------------------------------------------------------

struct RelayUrl {
    bool https = false;
    std::string host;   // lower case, IPv6 without brackets
    int port = 0;
    std::string path;   // "" or "/prefix" without a trailing slash
    std::string normalized;
};

inline bool is_loopback_name(std::string_view h) { return h == "127.0.0.1" || h == "localhost" || h == "::1"; }

// https:// for any host; plain http:// only for 127.0.0.1, localhost or [::1] (SSH tunnel). An optional path prefix of
// [A-Za-z0-9._/-] (no "..", no "//") is allowed; userinfo, query, fragment, percent escapes and backslashes are not.
inline bool parse_relay_url(const std::string& in, RelayUrl& u, std::string& err) {
    auto fail = [&](const char* m) {
        err = m;
        return false;
    };
    if (in.empty() || in.size() > 200) return fail("relay-adres is leeg of te lang (maximaal 200 tekens)");
    for (unsigned char c : in)
        if (c <= 0x20 || c >= 0x7F) return fail("relay-adres bevat spaties of ongeldige tekens");
    if (in.find_first_of("@?#%\\") != std::string::npos)
        return fail("relay-adres mag geen gebruikersnaam (@), query (?), fragment (#), % of \\ bevatten");
    const std::string low = lbhttp::lower_ascii(in);
    size_t at;
    u = RelayUrl{};
    if (low.rfind("https://", 0) == 0) {
        u.https = true;
        at = 8;
    } else if (low.rfind("http://", 0) == 0) {
        at = 7;
    } else {
        return fail("relay-adres moet met https:// beginnen (http:// alleen voor 127.0.0.1, localhost of [::1])");
    }
    const size_t slash = in.find('/', at);
    const std::string auth = in.substr(at, slash == std::string::npos ? std::string::npos : slash - at);
    std::string path = slash == std::string::npos ? std::string() : in.substr(slash);
    std::string host, port;
    bool has_port = false, v6 = false;
    if (!auth.empty() && auth[0] == '[') {
        const size_t rb = auth.find(']');
        if (rb == std::string::npos) return fail("ongeldig IPv6-adres");
        host = auth.substr(1, rb - 1);
        v6 = true;
        if (rb + 1 < auth.size()) {
            if (auth[rb + 1] != ':') return fail("ongeldig adres na ]");
            has_port = true;
            port = auth.substr(rb + 2);
        }
        for (char c : host)
            if (lbhttp::hex_value(c) < 0 && c != ':' && c != '.') return fail("ongeldig IPv6-adres");
    } else {
        const size_t colon = auth.find(':');
        host = auth.substr(0, colon);
        if (colon != std::string::npos) {
            has_port = true;
            port = auth.substr(colon + 1);
        }
        for (char c : host)
            if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || is_digit(c) || c == '.' || c == '-'))
                return fail("ongeldige hostnaam");
    }
    if (host.empty() || host.front() == '-' || host.front() == '.') return fail("ongeldige hostnaam");
    int pn = u.https ? 443 : 80;
    if (has_port) {
        if (port.empty() || port.size() > 5) return fail("ongeldige poort");
        pn = 0;
        for (char c : port) {
            if (!is_digit(c)) return fail("ongeldige poort");
            pn = pn * 10 + (c - '0');
        }
        if (pn < 1 || pn > 65535) return fail("ongeldige poort");
    }
    for (char c : path)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || is_digit(c) || c == '.' || c == '_' || c == '-' || c == '/'))
            return fail("het pad in het relay-adres mag alleen letters, cijfers, . _ - en / bevatten");
    if (path.find("..") != std::string::npos || path.find("//") != std::string::npos)
        return fail("het pad in het relay-adres mag geen \"..\" of \"//\" bevatten");
    while (!path.empty() && path.back() == '/') path.pop_back();
    u.host = lbhttp::lower_ascii(host);
    u.port = pn;
    u.path = path;
    if (!u.https && !is_loopback_name(u.host))
        return fail("onversleuteld http:// is alleen toegestaan voor 127.0.0.1, localhost of [::1]; gebruik https://");
    if (v6 && u.host.find(':') == std::string::npos) return fail("ongeldig IPv6-adres");
    u.normalized = std::string(u.https ? "https://" : "http://") + (v6 ? "[" + u.host + "]" : u.host) +
                   (has_port ? ":" + std::to_string(pn) : std::string()) + path;
    return true;
}

}  // namespace veltra::rc
