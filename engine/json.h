// Small, fast, dependency-free JSON parser / value type (header only).
//  * full RFC 8259 grammar, \uXXXX escapes incl. surrogate pairs (lone surrogates -> U+FFFD)
//  * objects keep insertion order (vector of pairs), duplicate keys are kept (find() returns the last)
//  * ~10 MB tokenizer.json parses in well under 100 ms
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace lb::json {

class Value;
using Array = std::vector<Value>;
using Object = std::vector<std::pair<std::string, Value>>;

class Value {
public:
    enum class Type : uint8_t { Null, Bool, Int, Double, String, Array, Object };

    Value() = default;
    Value(std::nullptr_t) {}
    Value(bool b) : v_(b) {}
    Value(int64_t i) : v_(i) {}
    Value(int i) : v_(static_cast<int64_t>(i)) {}
    Value(double d) : v_(d) {}
    Value(std::string s) : v_(std::move(s)) {}
    Value(const char* s) : v_(std::string(s)) {}
    Value(Array a) : v_(std::move(a)) {}
    Value(Object o) : v_(std::move(o)) {}

    Type type() const { return static_cast<Type>(v_.index()); }
    bool is_null() const { return type() == Type::Null; }
    bool is_bool() const { return type() == Type::Bool; }
    bool is_int() const { return type() == Type::Int; }
    bool is_number() const { return type() == Type::Int || type() == Type::Double; }
    bool is_string() const { return type() == Type::String; }
    bool is_array() const { return type() == Type::Array; }
    bool is_object() const { return type() == Type::Object; }

    bool as_bool(bool def = false) const { return is_bool() ? std::get<bool>(v_) : def; }
    int64_t as_int(int64_t def = 0) const {
        if (is_int()) return std::get<int64_t>(v_);
        if (type() == Type::Double) return static_cast<int64_t>(std::get<double>(v_));
        return def;
    }
    double as_double(double def = 0) const {
        if (type() == Type::Double) return std::get<double>(v_);
        if (is_int()) return static_cast<double>(std::get<int64_t>(v_));
        return def;
    }
    const std::string& as_string() const { return is_string() ? std::get<std::string>(v_) : empty_string(); }
    std::string as_string(const std::string& def) const { return is_string() ? std::get<std::string>(v_) : def; }
    const Array& as_array() const { return is_array() ? std::get<Array>(v_) : empty_array(); }
    const Object& as_object() const { return is_object() ? std::get<Object>(v_) : empty_object(); }
    Array& array() { return std::get<Array>(v_); }
    Object& object() { return std::get<Object>(v_); }
    std::string& string() { return std::get<std::string>(v_); }

    size_t size() const {
        if (is_array()) return std::get<Array>(v_).size();
        if (is_object()) return std::get<Object>(v_).size();
        return 0;
    }

    // Object lookup; returns nullptr when absent / not an object. Last duplicate wins.
    const Value* find(std::string_view key) const {
        if (!is_object()) return nullptr;
        const Object& o = std::get<Object>(v_);
        for (size_t i = o.size(); i-- > 0;)
            if (o[i].first == key) return &o[i].second;
        return nullptr;
    }
    // Never throws: missing key / index / wrong type yields a shared null Value.
    const Value& operator[](std::string_view key) const {
        const Value* p = find(key);
        return p ? *p : null_value();
    }
    const Value& operator[](const char* key) const { return (*this)[std::string_view(key)]; }
    const Value& operator[](size_t i) const {
        if (!is_array()) return null_value();
        const Array& a = std::get<Array>(v_);
        return i < a.size() ? a[i] : null_value();
    }

    static const Value& null_value() { static const Value v; return v; }

    // Compact serialisation (mostly for debugging / tests).
    std::string dump() const {
        std::string out;
        dump_to(out);
        return out;
    }
    static void escape_to(std::string& out, std::string_view s) {
        out.push_back('"');
        for (unsigned char c : s) {
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
                        char buf[8];
                        snprintf(buf, sizeof buf, "\\u%04x", c);
                        out += buf;
                    } else {
                        out.push_back(static_cast<char>(c));
                    }
            }
        }
        out.push_back('"');
    }
    void dump_to(std::string& out) const {
        switch (type()) {
            case Type::Null: out += "null"; break;
            case Type::Bool: out += std::get<bool>(v_) ? "true" : "false"; break;
            case Type::Int: out += std::to_string(std::get<int64_t>(v_)); break;
            case Type::Double: {
                char buf[40];
                snprintf(buf, sizeof buf, "%.17g", std::get<double>(v_));
                out += buf;
                break;
            }
            case Type::String: escape_to(out, std::get<std::string>(v_)); break;
            case Type::Array: {
                out.push_back('[');
                bool first = true;
                for (const Value& e : std::get<Array>(v_)) {
                    if (!first) out.push_back(',');
                    first = false;
                    e.dump_to(out);
                }
                out.push_back(']');
                break;
            }
            case Type::Object: {
                out.push_back('{');
                bool first = true;
                for (const auto& kv : std::get<Object>(v_)) {
                    if (!first) out.push_back(',');
                    first = false;
                    escape_to(out, kv.first);
                    out.push_back(':');
                    kv.second.dump_to(out);
                }
                out.push_back('}');
                break;
            }
        }
    }

private:
    static const std::string& empty_string() { static const std::string s; return s; }
    static const Array& empty_array() { static const Array a; return a; }
    static const Object& empty_object() { static const Object o; return o; }

    std::variant<std::monostate, bool, int64_t, double, std::string, Array, Object> v_;
};

namespace detail {

class Parser {
public:
    Parser(const char* b, const char* e) : begin_(b), p_(b), end_(e) {}

    bool parse(Value* out, std::string* err) {
        skip_ws();
        // tolerate a UTF-8 BOM
        if (end_ - p_ >= 3 && (unsigned char)p_[0] == 0xEF && (unsigned char)p_[1] == 0xBB && (unsigned char)p_[2] == 0xBF) {
            p_ += 3;
            skip_ws();
        }
        if (!value(*out, 0)) return fail_out(err);
        skip_ws();
        if (p_ != end_) {
            error_ = "trailing characters after JSON value";
            return fail_out(err);
        }
        return true;
    }

private:
    const char* begin_;
    const char* p_;
    const char* end_;
    std::string error_;

    bool fail_out(std::string* err) {
        if (err) {
            // line/column of the failure
            size_t line = 1, col = 1;
            for (const char* q = begin_; q < p_ && q < end_; ++q) {
                if (*q == '\n') { ++line; col = 1; } else ++col;
            }
            *err = "JSON parse error at line " + std::to_string(line) + ", col " + std::to_string(col) + ": " + error_;
        }
        return false;
    }
    bool fail(const char* msg) {
        if (error_.empty()) error_ = msg;
        return false;
    }
    void skip_ws() {
        while (p_ < end_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\t' || *p_ == '\r')) ++p_;
    }

    bool value(Value& out, int depth) {
        if (depth > 512) return fail("nesting too deep");
        if (p_ >= end_) return fail("unexpected end of input");
        switch (*p_) {
            case '{': return object(out, depth);
            case '[': return array(out, depth);
            case '"': {
                std::string s;
                if (!string(s)) return false;
                out = Value(std::move(s));
                return true;
            }
            case 't':
                if (end_ - p_ >= 4 && memcmp(p_, "true", 4) == 0) { p_ += 4; out = Value(true); return true; }
                return fail("invalid literal");
            case 'f':
                if (end_ - p_ >= 5 && memcmp(p_, "false", 5) == 0) { p_ += 5; out = Value(false); return true; }
                return fail("invalid literal");
            case 'n':
                if (end_ - p_ >= 4 && memcmp(p_, "null", 4) == 0) { p_ += 4; out = Value(); return true; }
                return fail("invalid literal");
            default:
                return number(out);
        }
    }

    bool object(Value& out, int depth) {
        ++p_;  // {
        Object o;
        skip_ws();
        if (p_ < end_ && *p_ == '}') { ++p_; out = Value(std::move(o)); return true; }
        for (;;) {
            skip_ws();
            if (p_ >= end_ || *p_ != '"') return fail("expected string key");
            std::string key;
            if (!string(key)) return false;
            skip_ws();
            if (p_ >= end_ || *p_ != ':') return fail("expected ':'");
            ++p_;
            skip_ws();
            o.emplace_back(std::move(key), Value());
            if (!value(o.back().second, depth + 1)) return false;
            skip_ws();
            if (p_ >= end_) return fail("unterminated object");
            if (*p_ == ',') { ++p_; continue; }
            if (*p_ == '}') { ++p_; break; }
            return fail("expected ',' or '}'");
        }
        out = Value(std::move(o));
        return true;
    }

    bool array(Value& out, int depth) {
        ++p_;  // [
        Array a;
        skip_ws();
        if (p_ < end_ && *p_ == ']') { ++p_; out = Value(std::move(a)); return true; }
        for (;;) {
            skip_ws();
            a.emplace_back();
            if (!value(a.back(), depth + 1)) return false;
            skip_ws();
            if (p_ >= end_) return fail("unterminated array");
            if (*p_ == ',') { ++p_; continue; }
            if (*p_ == ']') { ++p_; break; }
            return fail("expected ',' or ']'");
        }
        out = Value(std::move(a));
        return true;
    }

    static void put_utf8(std::string& s, uint32_t cp) {
        if (cp < 0x80) {
            s.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    bool hex4(uint32_t& v) {
        if (end_ - p_ < 4) return fail("truncated \\u escape");
        v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = p_[i];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0';
            else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
            else return fail("bad hex digit in \\u escape");
        }
        p_ += 4;
        return true;
    }

    bool string(std::string& out) {
        ++p_;  // opening quote
        const char* start = p_;
        // fast path: scan to closing quote with no escapes
        while (p_ < end_) {
            unsigned char c = static_cast<unsigned char>(*p_);
            if (c == '"') {
                out.assign(start, p_ - start);
                ++p_;
                return true;
            }
            if (c == '\\') break;
            if (c < 0x20) return fail("control character in string");
            ++p_;
        }
        if (p_ >= end_) return fail("unterminated string");
        out.assign(start, p_ - start);
        while (p_ < end_) {
            unsigned char c = static_cast<unsigned char>(*p_);
            if (c == '"') { ++p_; return true; }
            if (c < 0x20) return fail("control character in string");
            if (c != '\\') {
                // copy a run of plain bytes
                const char* s = p_;
                while (p_ < end_ && *p_ != '"' && *p_ != '\\' && static_cast<unsigned char>(*p_) >= 0x20) ++p_;
                out.append(s, p_ - s);
                continue;
            }
            ++p_;
            if (p_ >= end_) return fail("unterminated escape");
            char e = *p_++;
            switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    uint32_t cp;
                    if (!hex4(cp)) return false;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        // high surrogate: needs a following \uDC00..DFFF
                        if (end_ - p_ >= 6 && p_[0] == '\\' && p_[1] == 'u') {
                            const char* save = p_;
                            p_ += 2;
                            uint32_t lo;
                            if (!hex4(lo)) return false;
                            if (lo >= 0xDC00 && lo <= 0xDFFF) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            } else {
                                p_ = save;  // not a pair: emit replacement for the lone high surrogate
                                cp = 0xFFFD;
                            }
                        } else {
                            cp = 0xFFFD;
                        }
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        cp = 0xFFFD;
                    }
                    put_utf8(out, cp);
                    break;
                }
                default: return fail("invalid escape");
            }
        }
        return fail("unterminated string");
    }

    bool number(Value& out) {
        const char* s = p_;
        bool neg = false;
        if (p_ < end_ && *p_ == '-') { neg = true; ++p_; }
        if (p_ >= end_) return fail("invalid number");
        const char* digits = p_;
        if (*p_ == '0') {
            ++p_;
        } else if (*p_ >= '1' && *p_ <= '9') {
            while (p_ < end_ && *p_ >= '0' && *p_ <= '9') ++p_;
        } else {
            return fail("unexpected character");
        }
        bool is_float = false;
        const char* int_end = p_;
        if (p_ < end_ && *p_ == '.') {
            is_float = true;
            ++p_;
            const char* f = p_;
            while (p_ < end_ && *p_ >= '0' && *p_ <= '9') ++p_;
            if (p_ == f) return fail("digits expected after '.'");
        }
        if (p_ < end_ && (*p_ == 'e' || *p_ == 'E')) {
            is_float = true;
            ++p_;
            if (p_ < end_ && (*p_ == '+' || *p_ == '-')) ++p_;
            const char* f = p_;
            while (p_ < end_ && *p_ >= '0' && *p_ <= '9') ++p_;
            if (p_ == f) return fail("digits expected in exponent");
        }
        if (!is_float && int_end - digits <= 18) {
            int64_t v = 0;
            for (const char* q = digits; q < int_end; ++q) v = v * 10 + (*q - '0');
            out = Value(neg ? -v : v);
            return true;
        }
        std::string tmp(s, p_ - s);
        double d = strtod(tmp.c_str(), nullptr);
        if (!is_float && std::isfinite(d) && std::fabs(d) < 9.2e18) {
            out = Value(static_cast<int64_t>(strtoll(tmp.c_str(), nullptr, 10)));
        } else {
            out = Value(d);
        }
        return true;
    }
};

}  // namespace detail

// Parse `text` into *out. Returns false and fills *err (when non-null) on malformed input.
inline bool parse(std::string_view text, Value* out, std::string* err = nullptr) {
    detail::Parser p(text.data(), text.data() + text.size());
    return p.parse(out, err);
}

}  // namespace lb::json
