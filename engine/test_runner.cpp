// engine/test_runner.cpp - see test_runner.h. Every number in a report is measured here; nothing is estimated.
#include "test_runner.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <utility>

#include "backend.h"
#include "hub.h"
#include "remote_common.h"
#include "unicode_tables.h"

#if defined(_WIN32)
#  ifndef LOAD_LIBRARY_SEARCH_SYSTEM32
#    define LOAD_LIBRARY_SEARCH_SYSTEM32 0x00000800
#  endif
#else
#  include <dlfcn.h>
#endif

namespace veltra::tests {
namespace {

using Clock = std::chrono::steady_clock;
using lbhttp::Json;
using lbhttp::json_number;
using lbhttp::json_quote;
namespace rc = veltra::rc;

double secs(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }
std::string num(double v) { return json_number(v); }
std::string fx(double v, int digits = 1) {
    char b[40];
    std::snprintf(b, sizeof b, "%.*f", digits, v);
    for (char* p = b; *p; ++p)
        if (*p == '.') *p = ',';   // Dutch decimal comma in the summaries
    return b;
}
double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// ============================================================ suite table (the allowlist)

const StepInfo kCuda{"cuda_kernels", "CUDA-kernels controleren"};
const StepInfo kVram{"vram_bandwidth", "Videogeheugen- en PCIe-snelheid meten"};
const StepInfo kDecode{"decode_sweep", "Tekst genereren bij verschillende contextlengtes"};
const StepInfo kPrefill{"prefill_sweep", "Prompt inlezen (500 en 2000 tokens)"};
const StepInfo kBatch{"batch_and_search", "Meerdere antwoorden tegelijk (batch 1, 2, 4, 8)"};
const StepInfo kQuality{"quality_gate", "Kwaliteitscontrole met ingebouwde vragen"};

// ============================================================ text / Unicode helpers (Python str semantics)

using U = std::u32string;

U to_u32(std::string_view s) {
    U out;
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) {
            out += c;
            ++i;
            continue;
        }
        const size_t len = lbhttp::utf8_seq_len(s, i);
        if (len < 2) {
            out += 0xFFFD;
            ++i;
            continue;
        }
        char32_t cp = c & (0xFF >> (len + 1));
        for (size_t k = 1; k < len; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
        out += cp;
        i += len;
    }
    return out;
}

template <size_t N>
bool in_ranges(const uint32_t (&t)[N][2], char32_t c) {
    size_t lo = 0, hi = N;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (c < t[mid][0]) hi = mid;
        else if (c > t[mid][1]) lo = mid + 1;
        else return true;
    }
    return false;
}
bool py_alnum(char32_t c) {
    if (c < 0x80) return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
    return in_ranges(lb::uni::kLetter, c) || in_ranges(lb::uni::kNumber, c);
}
bool py_word(char32_t c) { return c == '_' || py_alnum(c); }
bool py_space(char32_t c) {
    return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
           c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}
bool py_digit(char32_t c) {   // \d: general category Nd
    static const char32_t starts[] = {0x30,    0x660,   0x6F0,   0x7C0,   0x966,   0x9E6,   0xA66,   0xAE6,   0xB66,   0xBE6,   0xC66,
                                      0xCE6,   0xD66,   0xDE6,   0xE50,   0xED0,   0xF20,   0x1040,  0x1090,  0x17E0,  0x1810,  0x1946,
                                      0x19D0,  0x1A80,  0x1A90,  0x1B50,  0x1BB0,  0x1C40,  0x1C50,  0xA620,  0xA8D0,  0xA900,  0xA9D0,
                                      0xA9F0,  0xAA50,  0xABF0,  0xFF10,  0x104A0, 0x10D30, 0x11066, 0x110F0, 0x11136, 0x111D0, 0x112F0,
                                      0x11450, 0x114D0, 0x11650, 0x116C0, 0x11730, 0x118E0, 0x11950, 0x11C50, 0x11D50, 0x11DA0, 0x11F50,
                                      0x16A60, 0x16AC0, 0x16B50, 0x1E140, 0x1E2F0, 0x1E4F0, 0x1E950, 0x1FBF0};
    for (char32_t s : starts)
        if (c >= s && c < s + 10) return true;
    return c >= 0x1D7CE && c <= 0x1D7FF;
}
// str.lower() for everything that can reach the needles used below
U py_lower(const U& s) {
    U o;
    o.reserve(s.size());
    for (char32_t c : s) {
        if (c >= 'A' && c <= 'Z') o += c + 32;
        else if (c >= 0xC0 && c <= 0xDE && c != 0xD7) o += c + 32;
        else if (c == 0x178) o += 0xFF;
        else if (c == 0x212A) o += U'k';
        else if (c == 0x212B) o += 0xE5;
        else if (c == 0x130) o += U"i̇";
        else o += c;
    }
    return o;
}
U py_strip(const U& s) {
    size_t a = 0, b = s.size();
    while (a < b && py_space(s[a])) ++a;
    while (b > a && py_space(s[b - 1])) --b;
    return s.substr(a, b - a);
}
bool has(const U& hay, const U& needle) { return hay.find(needle) != U::npos; }
// re.search with optional \b on either side of a literal that starts and ends with a word character
bool has_word(const U& hay, const U& w, bool left, bool right) {
    for (size_t p = hay.find(w); p != U::npos; p = hay.find(w, p + 1)) {
        if (left && p > 0 && py_word(hay[p - 1])) continue;
        if (right && p + w.size() < hay.size() && py_word(hay[p + w.size()])) continue;
        return true;
    }
    return false;
}
U after_last_think(const U& t) {   // text.split("</think>")[-1]
    static const U end = U"</think>";
    size_t last = U::npos;
    for (size_t p = t.find(end); p != U::npos; p = t.find(end, p + end.size())) last = p;
    return last == U::npos ? t : t.substr(last + end.size());
}

// ============================================================ checks of scripts/run_sanity10.py and scripts/eval_raptor.py

struct Verdict {
    bool ok = false;
    std::string why;
};
Verdict pass() { return {true, "ok"}; }
Verdict nope(std::string why) { return {false, std::move(why)}; }

// run_sanity10.py json_object(): the substring from the first "{" to the last "}" as one JSON object
// (json.loads also accepts NaN / Infinity outside strings).
bool python_loads(const std::string& text, Json& out) {
    std::string s;
    bool in_str = false;
    for (size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        if (in_str) {
            s += c;
            if (c == '\\' && i + 1 < text.size()) s += text[++i];
            else if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') {
            in_str = true;
            s += c;
        } else if (text.compare(i, 3, "NaN") == 0) {
            s += '0';
            i += 2;
        } else if (text.compare(i, 9, "-Infinity") == 0) {
            s += '0';
            i += 8;
        } else if (text.compare(i, 8, "Infinity") == 0) {
            s += '0';
            i += 7;
        } else {
            s += c;
        }
    }
    std::string err;
    return lbhttp::parse_json(s, out, err);
}

std::string to_utf8(const U& s) {
    std::string o;
    for (char32_t c : s) lbhttp::append_utf8(o, static_cast<uint32_t>(c));
    return o;
}

Verdict check_json_keys(const std::string& raw, const std::vector<const char*>& keys) {
    std::string t = to_utf8(after_last_think(to_u32(raw)));
    std::string stripped;
    for (size_t i = 0; i < t.size();) {
        if (t.compare(i, 3, "```") == 0) {
            i += 3;
            if (t.compare(i, 4, "json") == 0) i += 4;
        } else {
            stripped += t[i++];
        }
    }
    const std::string s = to_utf8(py_strip(to_u32(stripped)));
    const size_t i = s.find('{'), j = s.rfind('}');
    Json obj;
    if (i == std::string::npos || j == std::string::npos || j <= i || !python_loads(s.substr(i, j - i + 1), obj) || obj.kind != Json::Kind::Object)
        return nope("no JSON object");
    std::string miss;
    for (const char* k : keys)
        if (!obj.get(k)) miss += (miss.empty() ? "" : ", ") + std::string(k);
    return miss.empty() ? pass() : nope("missing keys " + miss);
}

Verdict check_contains_all(const std::string& raw, const std::vector<const char32_t*>& values, bool ignore_case) {
    const U t = after_last_think(to_u32(raw));
    const U hay = ignore_case ? py_lower(t) : t;
    std::string miss;
    for (const char32_t* v : values)
        if (!has(hay, ignore_case ? py_lower(U(v)) : U(v))) miss += (miss.empty() ? "" : ", ") + to_utf8(U(v));
    return miss.empty() ? pass() : nope("missing " + miss);
}

// (?<!\d)156(?!\d)
bool re_number_156(const U& t) {
    for (size_t p = t.find(U"156"); p != U::npos; p = t.find(U"156", p + 1))
        if (!(p > 0 && py_digit(t[p - 1])) && !(p + 3 < t.size() && py_digit(t[p + 3]))) return true;
    return false;
}
// def\s+max_of_two\s*\([^)]*\)\s*:[\s\S]*return
bool re_max_of_two(const U& t) {
    for (size_t p = t.find(U"def"); p != U::npos; p = t.find(U"def", p + 1)) {
        size_t i = p + 3;
        size_t spaces = 0;
        while (i < t.size() && py_space(t[i])) ++i, ++spaces;
        if (!spaces || t.compare(i, 10, U"max_of_two") != 0) continue;
        i += 10;
        while (i < t.size() && py_space(t[i])) ++i;
        if (i >= t.size() || t[i] != '(') continue;
        const size_t close = t.find(U')', i + 1);
        if (close == U::npos) continue;
        i = close + 1;
        while (i < t.size() && py_space(t[i])) ++i;
        if (i >= t.size() || t[i] != ':') continue;
        if (t.find(U"return", i + 1) != U::npos) return true;
    }
    return false;
}
// (?i)^\W*<word>\b
bool re_starts_with_word(const U& t, const char* word) {
    size_t i = 0;
    while (i < t.size() && !py_word(t[i])) ++i;
    const size_t n = std::strlen(word);
    if (i + n > t.size()) return false;
    for (size_t k = 0; k < n; ++k) {
        char32_t c = t[i + k];
        if (c >= 'A' && c <= 'Z') c += 32;
        if (c != static_cast<char32_t>(word[k])) return false;
    }
    return i + n == t.size() || !py_word(t[i + n]);
}

// eval_raptor.py strip_think(): (visible text, closed)
std::pair<U, bool> strip_think(const U& t) {
    if (has(t, U"<think>") && !has(t, U"</think>")) return {t, false};
    return {py_strip(after_last_think(t)), true};
}

// eval_raptor.py garbage(): "" = fine
std::string garbage(const U& text) {
    if (py_strip(text).empty()) return "empty output";
    for (char32_t c : text) {
        if (c == 0xFFFD) return "replacement char";
    }
    for (char32_t c : text)
        if ((c >= 0x3000 && c <= 0x9FFF) || (c >= 0xAC00 && c <= 0xD7AF) || (c >= 0x400 && c <= 0x4FF) || (c >= 0x600 && c <= 0x6FF))
            return "non-latin script";
    for (char32_t c : text)
        if (c <= 0x08 || c == 0x0B || c == 0x0C || (c >= 0x0E && c <= 0x1F)) return "control chars";
    U no_end = text;
    for (size_t p = no_end.find(U"<|im_end|>"); p != U::npos; p = no_end.find(U"<|im_end|>")) no_end.erase(p, 10);
    if (has(no_end, U"<|")) return "special token leaked";
    static const U extra = U"€°…–—‘’“” ";
    size_t odd = 0;
    for (char32_t c : text)
        if (!(py_alnum(c) || py_space(c) || extra.find(c) != U::npos || c < 128)) ++odd;
    if (text.size() >= 20 && static_cast<double>(odd) / static_cast<double>(text.size()) > 0.10) return "too many odd characters";
    std::vector<U> words;
    U cur;
    for (char32_t c : text) {
        if (py_space(c)) {
            if (!cur.empty()) words.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) words.push_back(cur);
    if (words.size() >= 12) {
        std::map<U, int> grams;
        int worst = 0;
        for (size_t i = 0; i + 2 < words.size(); ++i)
            worst = std::max(worst, ++grams[words[i] + U"\n" + words[i + 1] + U"\n" + words[i + 2]]);
        if (worst >= 4) return "repetition loop";
        size_t long_words = 0;
        for (const U& w : words)
            if (w.size() > 25) ++long_words;
        if (long_words >= 3) return "very long tokens";
    }
    return {};
}

int dutch_word_count(const U& text) {
    static const char* const stop[] = {"de", "het", "een", "en", "is", "van", "in", "dat", "te", "op", "voor", "met", "niet", "zijn", "ik",
                                       "je", "jij", "u", "dit", "om", "als", "maar", "ze", "ook", "aan", "door", "er", "naar", "bij", "of",
                                       "kan", "kunt", "wordt", "worden", "heeft", "hebben", "gaat", "hoe", "wat", "dus", "dan", "jouw", "uw",
                                       "mijn", "graag", "helaas", "goede", "goedemorgen", "ben", "bent", "waar", "omdat", "lucht", "licht",
                                       "heel", "meer", "geen"};
    static const U accents = U"àâäçèéêëîïôöùûüÿ'";
    const U low = py_lower(text);
    int count = 0;
    U cur;
    auto flush = [&] {
        if (cur.empty()) return;
        const std::string w = to_utf8(cur);
        for (const char* s : stop)
            if (w == s) {
                ++count;
                break;
            }
        cur.clear();
    };
    for (char32_t c : low) {
        if ((c >= 'a' && c <= 'z') || accents.find(c) != U::npos) cur += c;
        else flush();
    }
    flush();
    return count;
}

// F1: chk_contains(r"\bParis\b") (case sensitive: the pattern is not all lower case)
Verdict check_paris(const std::string& raw) {
    const auto [t, ok] = strip_think(to_u32(raw));
    if (!ok) return nope("unclosed <think>");
    const std::string g = garbage(t);
    if (!g.empty()) return nope(g);
    return has_word(t, U"Paris", true, true) ? pass() : nope("missing /\\bParis\\b/");
}

// F5: chk_number(366): re.findall(r"-?\d[\d,]*\.?\d*")
Verdict check_number(const std::string& raw, const std::string& expected) {
    const auto [t, ok] = strip_think(to_u32(raw));
    if (!ok) return nope("unclosed <think>");
    const std::string g = garbage(t);
    if (!g.empty()) return nope(g);
    size_t i = 0;
    while (i < t.size()) {
        size_t j = i;
        if (t[j] == '-') ++j;
        if (j >= t.size() || !py_digit(t[j])) {
            ++i;
            continue;
        }
        ++j;
        while (j < t.size() && (py_digit(t[j]) || t[j] == ',')) ++j;
        if (j < t.size() && t[j] == '.') ++j;
        while (j < t.size() && py_digit(t[j])) ++j;
        U n;
        for (size_t k = i; k < j; ++k)
            if (t[k] != ',') n += t[k];
        while (!n.empty() && n.back() == '.') n.pop_back();
        if (to_utf8(n) == expected) return pass();
        i = j;
    }
    return nope("expected " + expected);
}

// chk_dutch: every group of keys_all needs one of its alternatives; keys_any needs one; at least min_stop Dutch words
Verdict check_dutch(const std::string& raw, const std::vector<std::vector<const char32_t*>>& all, const std::vector<const char32_t*>& any,
                    int min_stop) {
    const auto [t, ok] = strip_think(to_u32(raw));
    if (!ok) return nope("unclosed <think>");
    const std::string g = garbage(t);
    if (!g.empty()) return nope(g);
    const U low = py_lower(t);
    for (const auto& alts : all) {
        bool found = false;
        for (const char32_t* a : alts) found = found || has(low, a);
        if (!found) return nope("missing /" + to_utf8(alts.front()) + "/");
    }
    if (!any.empty()) {
        bool found = false;
        for (const char32_t* a : any) found = found || has(low, a);
        if (!found) return nope("missing an expected word");
    }
    const int n = dutch_word_count(t);
    return n >= min_stop ? pass() : nope("not Dutch (" + std::to_string(n) + " Dutch function words)");
}

void dumps_py(std::string& o, const Json& j) {   // json.dumps separators, enough for substring checks
    if (j.kind == Json::Kind::Array || j.kind == Json::Kind::Object) {
        const bool obj = j.kind == Json::Kind::Object;
        o += obj ? '{' : '[';
        for (size_t i = 0; i < j.items.size(); ++i) {
            if (i) o += ", ";
            if (obj) o += json_quote(j.keys[i]) + ": ";
            dumps_py(o, j.items[i]);
        }
        o += obj ? '}' : ']';
    } else {
        rc::dump_clean(o, j, 100000);
    }
}

// eval_raptor.py tool_checker(accept) with acc_df (T1) and acc_read (T2)
Verdict check_tool(const std::string& raw, bool df) {
    const U u = to_u32(raw);
    if (has(u, U"<think>") && !has(u, U"</think>")) return nope("unclosed <think> before JSON");
    const std::string t = to_utf8(after_last_think(u));
    Json obj;
    bool found = false;
    for (size_t i = t.find('{'); i != std::string::npos && !found; i = t.find('{', i + 1)) {
        int depth = 0;
        bool in_str = false;
        size_t j = i;
        for (; j < t.size(); ++j) {   // the matching brace = where JSONDecoder.raw_decode would stop
            const char c = t[j];
            if (in_str) {
                if (c == '\\') ++j;
                else if (c == '"') in_str = false;
            } else if (c == '"') {
                in_str = true;
            } else if (c == '{' || c == '[') {
                ++depth;
            } else if (c == '}' || c == ']') {
                if (--depth == 0) break;
            }
        }
        if (j >= t.size()) continue;
        Json cand;
        if (python_loads(t.substr(i, j - i + 1), cand) && cand.kind == Json::Kind::Object && cand.get("tool")) {
            obj = std::move(cand);
            found = true;
        }
        if (t.size() > 20000 && i > 20000) break;
    }
    if (!found) return nope("no JSON object found");
    const Json* tool = obj.get("tool");
    const std::string name = tool && tool->kind == Json::Kind::String ? tool->str : std::string();
    if (name != "bash" && name != "read_file" && name != "write_file" && name != "http_get" && name != "python" && name != "final")
        return nope("unknown tool");
    const Json* args = obj.get("args");
    if (!args || args->kind != Json::Kind::Object) return nope("args is not an object");
    auto arg = [&](const char* k) {
        const Json* v = args->get(k);
        if (!v) return std::string();
        if (v->kind == Json::Kind::String) return v->str;
        std::string o;
        dumps_py(o, *v);
        return o;
    };
    const U cmd = to_u32(arg("cmd"));
    bool ok;
    if (df) {
        ok = name == "bash" && has_word(cmd, U"df", true, true) && has(cmd, U"/var");
    } else {
        ok = (name == "read_file" && to_utf8(py_strip(to_u32(arg("path")))) == "/etc/hostname") ||
             (name == "bash" && has_word(cmd, U"cat", true, true) && has(cmd, U"/etc/hostname"));
    }
    return ok ? pass() : nope("wrong tool/args");
}

const char* const kToolSystem =
    "You are RAPTOR, an operations agent. You can use these tools by replying with ONE JSON object on its own line, "
    "after a short plan: {\"tool\":\"<name>\",\"args\":{...}}.\n"
    "Tools: bash(cmd), read_file(path), write_file(path, content), http_get(url), python(code), final(answer).\n"
    "Tool results arrive in <tool_response> tags. When you are done, call final with the answer. Never invent tool output.";
const char* const kDutchSystem = "Je bent een behulpzame assistent. Antwoord kort en duidelijk in het Nederlands.";
const char* const kGeneralSystem = "You are a helpful assistant.";

struct Case {
    const char* id;
    const char* system;   // nullptr = user message only (sanity10)
    const char* user;
    bool quick;           // part of the quick_smoke subset
    Verdict (*check)(const std::string&);
};

const Case kCases[] = {
    // scripts/sanity10.json
    {"nl1", nullptr, "Wat is de hoofdstad van Australi\xc3\xab? Antwoord met alleen de naam van de stad.", true,
     [](const std::string& s) { return check_contains_all(s, {U"Canberra"}, true); }},
    {"nl2", nullptr, "Hoeveel is 12 keer 13? Geef alleen het getal.", true,
     [](const std::string& s) { return re_number_156(after_last_think(to_u32(s))) ? pass() : nope("pattern not found"); }},
    {"nl3", nullptr, "Zet deze zin in het Nederlands: 'The cat is sleeping on the table.'", false,
     [](const std::string& s) { return check_contains_all(s, {U"kat", U"tafel"}, true); }},
    {"code1", nullptr, "Schrijf een Python-functie is_even(n) die True teruggeeft als n even is, en anders False. Geef alleen de code.", false,
     [](const std::string& s) { return check_contains_all(s, {U"def", U"is_even", U"%"}, false); }},
    {"code2", nullptr, "Schrijf een Python-functie max_of_two(a, b) die het grootste van twee getallen teruggeeft. Geef alleen de code, zonder uitleg.", false,
     [](const std::string& s) { return re_max_of_two(after_last_think(to_u32(s))) ? pass() : nope("pattern not found"); }},
    {"logic1", nullptr, "Alle mensen zijn sterfelijk. Socrates is een mens. Is Socrates sterfelijk? Antwoord met ja of nee.", false,
     [](const std::string& s) { return re_starts_with_word(after_last_think(to_u32(s)), "ja") ? pass() : nope("pattern not found"); }},
    {"logic2", nullptr, "Anna is ouder dan Bas. Bas is ouder dan Cees. Cees is ouder dan Dirk. Wie is de jongste? Antwoord alleen met de naam.", true,
     [](const std::string& s) { return re_starts_with_word(after_last_think(to_u32(s)), "dirk") ? pass() : nope("pattern not found"); }},
    {"json1", nullptr, "Geef alleen een JSON-object voor een fictief persoon, met de sleutels naam, leeftijd en stad. Geen uitleg.", true,
     [](const std::string& s) { return check_json_keys(s, {"naam", "leeftijd", "stad"}); }},
    {"json2", nullptr, "Geef alleen een JSON-object voor een fictief boek, met de sleutels titel, auteur en jaar. Geen uitleg.", false,
     [](const std::string& s) { return check_json_keys(s, {"titel", "auteur", "jaar"}); }},
    {"json3", nullptr, "Geef alleen een JSON-object voor een fictief product, met de sleutels product, prijs_eur en op_voorraad. Geen uitleg.", false,
     [](const std::string& s) { return check_json_keys(s, {"product", "prijs_eur", "op_voorraad"}); }},
    // core set of scripts/eval_raptor.py
    {"F1", kGeneralSystem, "What is the capital of France? Answer in one sentence.", true,
     [](const std::string& s) { return check_paris(s); }},
    {"F5", kGeneralSystem, "How many days does a leap year have? Answer with the number.", false,
     [](const std::string& s) { return check_number(s, "366"); }},
    {"D1", kDutchSystem, "Vertaal naar het Nederlands: 'Good morning, how are you today?'", false,
     [](const std::string& s) { return check_dutch(s, {}, {U"goedemorgen", U"goede morgen"}, 1); }},
    {"D3", kDutchSystem, "Noem de vier jaargetijden in het Nederlands.", false,
     [](const std::string& s) {
         return check_dutch(s, {{U"lente", U"voorjaar"}, {U"zomer"}, {U"herfst", U"najaar"}, {U"winter"}}, {}, 0);
     }},
    {"T1", kToolSystem, "Show how much disk space is free on /var.", false, [](const std::string& s) { return check_tool(s, true); }},
    {"T2", kToolSystem, "Read the file /etc/hostname.", false, [](const std::string& s) { return check_tool(s, false); }},
};

constexpr int kQualityMaxTokens = 250;   // as run_sanity10.py / eval_raptor.py

// ============================================================ run context

struct Cancelled {};

struct Run {
    EngineContext& ctx;
    const std::atomic<bool>& cancel;
    const ProgressFn& report_progress;
    const SuiteInfo& suite;
    Progress progress;

    void message(const std::string& m, double tok_s = -1) {
        progress.message = m;
        if (tok_s >= 0) progress.tok_s = tok_s;
        if (report_progress) report_progress(progress);
    }
    void check_cancel() const {
        if (cancel.load()) throw Cancelled{};
    }
};

struct StepOut {
    std::string status = "error", summary, data = "{}";
};
StepOut skipped(const std::string& why) { return {"skipped", why, "{}"}; }

// Removes local paths from engine / driver messages before they enter a report.
std::string scrub(std::string s, const EngineContext& ctx) {
    auto replace_all = [&s](const std::string& a, const std::string& b) {
        if (a.size() < 2) return;
        for (size_t p = s.find(a); p != std::string::npos; p = s.find(a, p + b.size())) s.replace(p, a.size(), b);
    };
    replace_all(ctx.model_path, ctx.model_file);
    const size_t slash = ctx.model_path.find_last_of("/\\");
    if (slash != std::string::npos && slash > 1) replace_all(ctx.model_path.substr(0, slash), "<modelmap>");
    rc::scrub_paths(s);
    return rc::clean_text(s, 400);
}

// ---- engine measurements ----------------------------------------------------------------------

constexpr const char* kText =
    "De gemeente heeft gisteren een nieuw plan gepresenteerd voor het oude havengebied. Op de plek van de "
    "leegstaande loodsen moeten de komende tien jaar ongeveer twaalfhonderd woningen komen, waarvan een derde "
    "in de sociale huur. Daarnaast is er ruimte voor kleine bedrijven, een basisschool en een park langs het "
    "water. Bewoners van de aangrenzende wijk reageerden verdeeld. Sommigen zijn blij dat er eindelijk iets "
    "gebeurt met het terrein, dat al jaren verwaarloosd is. Anderen maken zich zorgen over de drukte op de "
    "smalle toegangswegen en over de hoogte van de geplande torens. De wethouder benadrukte dat het plan nog "
    "niet definitief is en dat er in het voorjaar drie inloopavonden komen. Ook wordt onderzocht of er een "
    "extra busverbinding naar het station kan komen. Volgens de planning begint de sanering van de grond "
    "volgend jaar; de eerste woningen zouden over vier jaar klaar moeten zijn. Het project wordt gefinancierd "
    "door de gemeente, een woningcorporatie en twee private ontwikkelaars, die samen ruim driehonderd miljoen "
    "euro investeren.\n\n";
constexpr const char* kDecodePrompt =
    "Hieronder volgt een uitgebreide beschrijving van de geschiedenis van de Nederlandse waterbouw, van de "
    "eerste terpen tot de Deltawerken:\n\n";

std::string nonce() { return rc::random_hex(6) + ".\n"; }   // a different first line defeats the prefix cache

// Exactly n token ids of built-in Dutch text, starting at a random line.
std::vector<int> text_prompt(lb::Engine& e, int n, const std::string& head) {
    std::string text = head;
    std::vector<int> ids;
    for (int i = 0; i < 64 && static_cast<int>(ids.size()) < n; ++i) {
        text += kText;
        ids = e.encode(text, false);
    }
    if (static_cast<int>(ids.size()) > n) ids.resize(static_cast<size_t>(n));
    return ids;
}

lb::GenOptions plain_options(const EngineContext& ctx, int max_new) {
    lb::GenOptions g;
    g.max_new_tokens = max_new;
    g.spec.enabled = false;   // plain greedy decode: no speculative drafts, REST or MTP in the speed numbers
    g.spec.rest = false;
    g.spec.mtp = false;
    g.n_threads = ctx.threads;
    return g;
}

constexpr int kDecodeTokens = 32;
constexpr int kDecodeReps = 3;

StepOut step_decode(Run& r) {
    lb::Engine* e = r.ctx.engine;
    if (!e) return skipped("geen model geladen");
    r.message("opwarmen");
    e->reset();
    e->generate(text_prompt(*e, 32, nonce()), plain_options(r.ctx, 8), [&](int, const std::string&) { return !r.cancel.load(); });
    r.check_cancel();
    std::string points, text, skips;
    int measured = 0, requested = 0, failed = 0;
    for (int ctx_len : r.suite.decode_ctx) {
        ++requested;
        if (ctx_len + kDecodeTokens + 2 > r.ctx.max_context) {
            const std::string why = "boven de contextlimiet van " + std::to_string(r.ctx.max_context) + " tokens";
            points += std::string(points.empty() ? "" : ",") + "{\"ctx\":" + std::to_string(ctx_len) + ",\"status\":\"skipped\",\"reason\":" +
                      json_quote(why) + "}";
            skips += (skips.empty() ? "" : "; ") + ("ctx " + std::to_string(ctx_len) + " overgeslagen: " + why);
            continue;
        }
        r.message("context " + std::to_string(ctx_len) + ": prompt inlezen");
        const std::vector<int> prompt = ctx_len > 0 ? text_prompt(*e, ctx_len, nonce()) : e->encode(nonce() + kDecodePrompt, false);
        std::vector<double> speeds;
        std::string reps;
        for (int rep = 0; rep < kDecodeReps; ++rep) {
            r.check_cancel();
            e->reset();
            int count = 0;
            Clock::time_point first;
            const lb::GenStats st = e->generate(prompt, plain_options(r.ctx, kDecodeTokens), [&](int, const std::string&) {
                if (count++ == 0) first = Clock::now();
                else if (count % 8 == 0)
                    r.message("context " + std::to_string(ctx_len) + ": meting " + std::to_string(rep + 1) + "/" + std::to_string(kDecodeReps),
                              (count - 1) / std::max(1e-9, secs(first)));
                return !r.cancel.load();
            });
            r.check_cancel();
            const bool usable = st.generated_tokens >= 8 && st.tokens_per_s > 0;   // an early end-of-text is no speed measurement
            if (usable) speeds.push_back(st.tokens_per_s);
            reps += std::string(reps.empty() ? "" : ",") + "{\"tokens\":" + std::to_string(st.generated_tokens) + ",\"tok_s\":" +
                    (usable ? num(st.tokens_per_s) : std::string("null")) + "}";
        }
        const std::string head = "{\"ctx\":" + std::to_string(ctx_len) + ",\"prompt_tokens\":" + std::to_string(prompt.size());
        points += std::string(points.empty() ? "" : ",") + head;
        if (speeds.empty()) {
            ++failed;
            points += ",\"status\":\"error\",\"reason\":\"het model stopte telkens binnen 8 tokens\",\"reps\":[" + reps + "]}";
            continue;
        }
        const double m = median(speeds);
        ++measured;
        points += ",\"status\":\"pass\",\"tok_s_median\":" + num(m) + ",\"runs\":" + std::to_string(speeds.size()) + ",\"reps\":[" + reps + "]}";
        text += (text.empty() ? "" : " | ") + ("ctx " + std::to_string(ctx_len) + ": " + fx(m) + " tok/s");
        r.message("context " + std::to_string(ctx_len) + " klaar", m);
    }
    e->reset();
    StepOut o;
    o.status = measured == 0 ? (failed ? "error" : "skipped") : failed ? "fail" : "pass";
    o.summary = (measured ? "Genereren op " + r.ctx.device + " (mediaan van " + std::to_string(kDecodeReps) + "): " + text : std::string("Niets gemeten")) +
                (skips.empty() ? "" : " (" + skips + ")");
    o.data = "{\"device\":" + json_quote(r.ctx.device) + ",\"tokens_per_run\":" + std::to_string(kDecodeTokens) +
             ",\"speculative\":false,\"requested\":" + std::to_string(requested) + ",\"points\":[" + points + "]}";
    return o;
}

StepOut step_prefill(Run& r) {
    lb::Engine* e = r.ctx.engine;
    if (!e) return skipped("geen model geladen");
    std::string points, text, skips;
    int measured = 0;
    for (int n : {500, 2000}) {
        r.check_cancel();
        if (n + 4 > r.ctx.max_context) {
            points += std::string(points.empty() ? "" : ",") + "{\"prompt_tokens\":" + std::to_string(n) +
                      ",\"status\":\"skipped\",\"reason\":\"boven de contextlimiet\"}";
            skips += (skips.empty() ? "" : "; ") + ("pp" + std::to_string(n) + " overgeslagen: boven de contextlimiet");
            continue;
        }
        r.message("prompt van " + std::to_string(n) + " tokens inlezen");
        const std::vector<int> prompt = text_prompt(*e, n, nonce());
        e->reset();
        const auto t0 = Clock::now();
        double ttft = -1;
        const lb::GenStats st = e->generate(prompt, plain_options(r.ctx, 1), [&](int, const std::string&) {
            if (ttft < 0) ttft = secs(t0);
            return !r.cancel.load();
        });
        if (ttft < 0) ttft = secs(t0);
        r.check_cancel();
        const int fresh = static_cast<int>(prompt.size()) - st.cached_tokens;
        const double speed = st.prefill_s > 0 ? fresh / st.prefill_s : 0;
        if (speed <= 0) continue;
        ++measured;
        points += std::string(points.empty() ? "" : ",") + "{\"prompt_tokens\":" + std::to_string(prompt.size()) +
                  ",\"status\":\"pass\",\"cached_tokens\":" + std::to_string(st.cached_tokens) + ",\"prefill_s\":" + num(st.prefill_s) +
                  ",\"prefill_tok_s\":" + num(speed) + ",\"ttft_s\":" + num(ttft) + "}";
        text += (text.empty() ? "" : " | ") + ("pp" + std::to_string(n) + ": " + fx(speed) + " tok/s, eerste token na " + fx(ttft) + " s");
        r.message("pp" + std::to_string(n) + " klaar", speed);
    }
    e->reset();
    StepOut o;
    o.status = measured ? "pass" : "skipped";
    o.summary = (measured ? "Inlezen op " + r.ctx.device + " (1 meting): " + text : std::string("Niets gemeten")) + (skips.empty() ? "" : " (" + skips + ")");
    o.data = "{\"device\":" + json_quote(r.ctx.device) + ",\"runs_per_size\":1,\"points\":[" + points + "]}";
    return o;
}

constexpr int kBatchPromptLen = 64;
constexpr int kBatchSteps = 24;

StepOut step_batch(Run& r) {
    // There is no verifier-guided search in this engine (no --search-n anywhere in engine/): report it, never fake it.
    const std::string search = "{\"status\":\"skipped\",\"reason\":\"Test-Time Search is niet ge\xc3\xaf" "mplementeerd in de engine\"}";
    const std::string search_note = "Test-Time Search overgeslagen: niet ge\xc3\xaf" "mplementeerd in de engine";
    lb::Engine* e = r.ctx.engine;
    if (!e) return {"skipped", "geen model geladen. " + search_note, "{\"search\":" + search + "}"};
    std::vector<std::vector<int>> prompts;
    for (int i = 0; i < 8; ++i) {
        std::vector<int> p = text_prompt(*e, kBatchPromptLen + 8 * i, nonce());
        if (static_cast<int>(p.size()) > kBatchPromptLen) p.erase(p.begin(), p.end() - kBatchPromptLen);   // distinct contexts
        prompts.push_back(std::move(p));
    }
    e->reset();
    std::vector<lb::BatchPoint> pts;
    std::string why;
    int stage = -1, calls = 0;
    bool supported;
    try {
        supported = e->batch_probe(prompts, {1, 2, 4, 8}, kBatchSteps, [&](int b) {
            if (b != stage || ++calls % 8 == 0) {
                stage = b;
                r.message(b == 0 ? std::string("8 prompts inlezen") : "batch " + std::to_string(b) + " meten",
                          pts.empty() ? -1 : pts.back().agg_tok_s);
            }
            return !r.cancel.load();
        }, &pts, &why);
    } catch (const std::exception& ex) {
        supported = false;
        why = ex.what();
    }
    e->reset();
    r.check_cancel();
    if (!supported) return {"skipped", "Batch niet mogelijk: " + scrub(why, r.ctx) + ". " + search_note,
                            "{\"batch_reason\":" + json_quote(scrub(why, r.ctx)) + ",\"search\":" + search + "}"};
    std::string points, text;
    for (const lb::BatchPoint& p : pts) {
        points += std::string(points.empty() ? "" : ",") + "{\"batch\":" + std::to_string(p.batch) + ",\"steps\":" + std::to_string(p.steps) +
                  ",\"step_ms_median\":" + num(p.step_ms_median) + ",\"aggregate_tok_s\":" + num(p.agg_tok_s) +
                  ",\"per_stream_tok_s\":" + num(p.agg_tok_s / std::max(1, p.batch)) + "}";
        text += (text.empty() ? "" : " | ") + ("B=" + std::to_string(p.batch) + ": " + fx(p.agg_tok_s) + " tok/s totaal");
    }
    StepOut o;
    o.status = pts.size() == 4 ? "pass" : "error";
    o.summary = "Batch op " + r.ctx.device + ": " + (text.empty() ? std::string("niets gemeten") : text) + ". " + search_note;
    o.data = "{\"device\":" + json_quote(r.ctx.device) + ",\"prompt_tokens\":" + std::to_string(kBatchPromptLen) + ",\"points\":[" + points +
             "],\"search\":" + search + "}";
    return o;
}

StepOut step_quality(Run& r) {
    lb::Engine* e = r.ctx.engine;
    if (!e) return skipped("geen model geladen");
    std::vector<const Case*> cases;
    for (const Case& c : kCases)
        if (!r.suite.quality_subset || c.quick) cases.push_back(&c);
    int passed = 0, done = 0;
    std::string rows, failed, failed_json;
    for (const Case* c : cases) {
        r.check_cancel();
        r.message("vraag " + std::to_string(done + 1) + "/" + std::to_string(cases.size()) + " (" + c->id + ")");
        std::vector<lb::ChatMessage> msgs;
        if (c->system) msgs.push_back({"system", c->system});
        msgs.push_back({"user", c->user});
        std::string text, why;
        bool ok = false;
        int tokens = 0;
        try {
            e->reset();
            const std::vector<int> prompt = e->apply_chat_template(msgs, true);
            lb::GenOptions g = r.ctx.chat_options ? r.ctx.chat_options(kQualityMaxTokens) : plain_options(r.ctx, kQualityMaxTokens);
            g.max_new_tokens = std::min(kQualityMaxTokens, r.ctx.max_context - static_cast<int>(prompt.size()) - 2);
            if (g.max_new_tokens < 1) throw std::runtime_error("prompt past niet in de context");
            Clock::time_point first;
            const lb::GenStats st = e->generate(prompt, g, [&](int, const std::string& piece) {
                if (tokens++ == 0) first = Clock::now();
                else if (tokens % 16 == 0)
                    r.message("vraag " + std::to_string(done + 1) + "/" + std::to_string(cases.size()) + " (" + c->id + ")",
                              (tokens - 1) / std::max(1e-9, secs(first)));
                text += piece;
                const size_t stop = text.find("<|im_end|>");
                if (stop != std::string::npos) {
                    text.resize(stop);
                    return false;
                }
                return !r.cancel.load();
            });
            (void)st;
            r.check_cancel();
            const Verdict v = c->check(text);
            ok = v.ok;
            why = v.why;
        } catch (const std::exception& ex) {
            why = scrub(std::string("generatie mislukt: ") + ex.what(), r.ctx);
        }
        ++done;
        if (ok) ++passed;
        else {
            failed += (failed.empty() ? "" : ", ") + std::string(c->id);
            failed_json += std::string(failed_json.empty() ? "" : ",") + json_quote(c->id);
        }
        rows += std::string(rows.empty() ? "" : ",") + "{\"id\":" + json_quote(c->id) + ",\"pass\":" + (ok ? "true" : "false") +
                ",\"reason\":" + json_quote(rc::clean_text(why, 200)) + ",\"tokens\":" + std::to_string(tokens) +
                ",\"output\":" + json_quote(std::string(text, 0, 1500)) + "}";
    }
    e->reset();
    const int total = static_cast<int>(cases.size());
    StepOut o;
    o.status = passed == total ? "pass" : "fail";
    o.summary = std::to_string(passed) + "/" + std::to_string(total) + " vragen goed (antwoorden berekend op " + r.ctx.device + ")" +
                (failed.empty() ? "" : "; fout: " + failed);
    o.data = "{\"device\":" + json_quote(r.ctx.device) + ",\"passed\":" + std::to_string(passed) + ",\"total\":" + std::to_string(total) +
             ",\"failed\":[" + failed_json + "],\"max_tokens\":" + std::to_string(kQualityMaxTokens) + ",\"cases\":[" + rows + "]}";
    return o;
}

StepOut step_cuda(Run& r) {
    if (!lb::cuda_backend_built()) return skipped("CUDA-backend zit niet in deze build");
    r.message("CUDA-selftest draait");
    const std::string rep = lb::cuda_selftest(r.ctx.model_path);
    if (rep.find("no CUDA device") != std::string::npos) return skipped("geen CUDA-videokaart gevonden");
    int ok = 0, bad = 0;
    std::string lines;
    size_t pos = 0;
    while (pos < rep.size()) {
        size_t eol = rep.find('\n', pos);
        if (eol == std::string::npos) eol = rep.size();
        const std::string line = rep.substr(pos, eol - pos);
        pos = eol + 1;
        const bool p = line.find("  PASS  ") != std::string::npos, f = line.find("  FAIL  ") != std::string::npos;
        ok += p;
        bad += f;
        if (!line.empty()) lines += std::string(lines.empty() ? "" : ",") + json_quote(scrub(line, r.ctx));
    }
    const bool all = rep.find("all PASS") != std::string::npos && bad == 0;
    StepOut o;
    o.status = all ? "pass" : "fail";
    o.summary = "CUDA-selftest: " + std::to_string(ok) + " PASS, " + std::to_string(bad) + " FAIL" + (all ? "" : " (niet geslaagd)");
    o.data = "{\"passed\":" + std::to_string(ok) + ",\"failed\":" + std::to_string(bad) + ",\"lines\":[" + lines + "]}";
    return o;
}

// ---- vram_bandwidth: the NVIDIA driver API, loaded at run time (no CUDA toolkit needed) --------

#if defined(_WIN32)
#  define VELTRA_CUDAAPI __stdcall
#else
#  define VELTRA_CUDAAPI
#endif

struct CudaDriver {
    using Ptr = unsigned long long;   // CUdeviceptr (64-bit builds)
    void* lib = nullptr;
    int(VELTRA_CUDAAPI* Init)(unsigned) = nullptr;
    int(VELTRA_CUDAAPI* DeviceGetCount)(int*) = nullptr;
    int(VELTRA_CUDAAPI* DeviceGet)(int*, int) = nullptr;
    int(VELTRA_CUDAAPI* DeviceGetName)(char*, int, int) = nullptr;
    int(VELTRA_CUDAAPI* CtxCreate)(void**, unsigned, int) = nullptr;
    int(VELTRA_CUDAAPI* CtxDestroy)(void*) = nullptr;
    int(VELTRA_CUDAAPI* CtxSynchronize)() = nullptr;
    int(VELTRA_CUDAAPI* MemGetInfo)(size_t*, size_t*) = nullptr;
    int(VELTRA_CUDAAPI* MemAlloc)(Ptr*, size_t) = nullptr;
    int(VELTRA_CUDAAPI* MemFree)(Ptr) = nullptr;
    int(VELTRA_CUDAAPI* MemHostAlloc)(void**, size_t, unsigned) = nullptr;   // optional: pinned host buffers
    int(VELTRA_CUDAAPI* MemFreeHost)(void*) = nullptr;
    int(VELTRA_CUDAAPI* HtoD)(Ptr, const void*, size_t) = nullptr;
    int(VELTRA_CUDAAPI* DtoH)(void*, Ptr, size_t) = nullptr;
    int(VELTRA_CUDAAPI* DtoD)(Ptr, Ptr, size_t) = nullptr;

    void* symbol(const char* name) const {
#if defined(_WIN32)
        return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
        return dlsym(lib, name);
#endif
    }
    template <class F>
    bool load(F& fn, const char* name) {
        fn = reinterpret_cast<F>(symbol(name));
        return fn != nullptr;
    }
    // "" or the reason the driver API cannot be used. The library stays loaded (drivers do not like being unloaded).
    std::string open() {
#if defined(_WIN32)
        lib = LoadLibraryExA("nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);   // only the driver's copy in System32
        const char* file = "nvcuda.dll";
#else
        lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        const char* file = "libcuda.so.1";
#endif
        if (!lib) return std::string("NVIDIA-driver niet gevonden (") + file + "); geen NVIDIA-videokaart of geen driver";
        const bool ok = load(Init, "cuInit") && load(DeviceGetCount, "cuDeviceGetCount") && load(DeviceGet, "cuDeviceGet") &&
                        load(DeviceGetName, "cuDeviceGetName") && load(CtxCreate, "cuCtxCreate_v2") && load(CtxDestroy, "cuCtxDestroy_v2") &&
                        load(CtxSynchronize, "cuCtxSynchronize") && load(MemGetInfo, "cuMemGetInfo_v2") && load(MemAlloc, "cuMemAlloc_v2") &&
                        load(MemFree, "cuMemFree_v2") && load(HtoD, "cuMemcpyHtoD_v2") && load(DtoH, "cuMemcpyDtoH_v2") &&
                        load(DtoD, "cuMemcpyDtoD_v2");
        if (!ok) return "de NVIDIA-driver mist een benodigde functie (te oude driver?)";
        load(MemHostAlloc, "cuMemHostAlloc");
        load(MemFreeHost, "cuMemFreeHost");
        return {};
    }
};

struct CudaError {
    const char* call;
    int code;
};
void cu(int rc, const char* call) {
    if (rc != 0) throw CudaError{call, rc};
}

constexpr int kBandwidthReps = 7;

StepOut step_vram(Run& r) {
    static CudaDriver drv;
    static std::string open_error = drv.open();
    if (!open_error.empty()) return skipped(open_error);
    const int init = drv.Init(0);
    if (init == 100) return skipped("de NVIDIA-driver meldt geen videokaart (CUDA_ERROR_NO_DEVICE)");
    if (init != 0) return {"error", "cuInit mislukt met CUDA-foutcode " + std::to_string(init), "{}"};
    int count = 0;
    if (drv.DeviceGetCount(&count) != 0 || count <= 0) return skipped("geen CUDA-videokaart gevonden");

    struct Resources {   // freed in reverse order on every exit path
        CudaDriver& d;
        void* ctx = nullptr;
        CudaDriver::Ptr a = 0, b = 0;
        void* host = nullptr;
        void* back = nullptr;
        bool pinned = false;
        ~Resources() {
            if (a) d.MemFree(a);
            if (b) d.MemFree(b);
            for (void* p : {host, back}) {
                if (!p) continue;
                if (pinned) d.MemFreeHost(p);
                else std::free(p);
            }
            if (ctx) d.CtxDestroy(ctx);
        }
    } res{drv};
    try {
        int dev = 0;
        cu(drv.DeviceGet(&dev, 0), "cuDeviceGet");
        char name[128] = {0};
        cu(drv.DeviceGetName(name, sizeof name - 1, dev), "cuDeviceGetName");
        cu(drv.CtxCreate(&res.ctx, 0, dev), "cuCtxCreate_v2");
        size_t free_b = 0, total_b = 0;
        cu(drv.MemGetInfo(&free_b, &total_b), "cuMemGetInfo_v2");
        const size_t size = std::min<size_t>(512u << 20, free_b / 4) & ~static_cast<size_t>(0xFFFFF);
        const std::string device = rc::clean_text(name, 120);
        if (size < (16u << 20)) return skipped("te weinig vrij videogeheugen voor een betrouwbare meting op " + device);
        cu(drv.MemAlloc(&res.a, size), "cuMemAlloc_v2");
        cu(drv.MemAlloc(&res.b, size), "cuMemAlloc_v2");
        if (drv.MemHostAlloc && drv.MemFreeHost && drv.MemHostAlloc(&res.host, size, 0) == 0) {
            res.pinned = true;
            if (drv.MemHostAlloc(&res.back, size, 0) != 0) {
                drv.MemFreeHost(res.host);
                res.host = res.back = nullptr;
                res.pinned = false;
            }
        }
        if (!res.pinned) {
            res.host = std::malloc(size);
            res.back = std::malloc(size);
            if (!res.host || !res.back) return {"error", "niet genoeg werkgeheugen voor de testbuffer", "{}"};
        }
        uint64_t* words = static_cast<uint64_t*>(res.host);
        for (size_t i = 0; i < size / 8; ++i) words[i] = i * 0x9E3779B97F4A7C15ull + 1;
        std::memset(res.back, 0, size);

        auto timed = [&](const char* label, const std::function<int()>& copy) {
            std::vector<double> gbs;
            cu(copy(), label);   // warm-up
            cu(drv.CtxSynchronize(), "cuCtxSynchronize");
            for (int i = 0; i < kBandwidthReps; ++i) {
                r.check_cancel();
                const auto t0 = Clock::now();
                cu(copy(), label);
                cu(drv.CtxSynchronize(), "cuCtxSynchronize");   // device copies are asynchronous: wait before stopping the clock
                const double s = secs(t0);
                if (s > 0) gbs.push_back(static_cast<double>(size) / s / 1e9);
            }
            return median(gbs);
        };
        r.message("PCIe: computer naar videokaart (" + std::to_string(size >> 20) + " MiB)");
        const double h2d = timed("cuMemcpyHtoD_v2", [&] { return drv.HtoD(res.a, res.host, size); });
        r.message("kopie binnen het videogeheugen");
        const double d2d = timed("cuMemcpyDtoD_v2", [&] { return drv.DtoD(res.b, res.a, size); });
        r.message("PCIe: videokaart naar computer");
        const double d2h = timed("cuMemcpyDtoH_v2", [&] { return drv.DtoH(res.back, res.b, size); });
        const bool verified = std::memcmp(res.host, res.back, size) == 0;   // proves the bytes really made the round trip
        if (!verified || h2d <= 0 || d2h <= 0 || d2d <= 0)   // timings of copies that did not happen mean nothing
            return {"fail", device + (verified ? ": een kopie was niet meetbaar" : ": de teruggelezen data klopt niet") + "; geen snelheden gerapporteerd",
                    "{\"device\":" + json_quote(device) + ",\"data_verified\":" + (verified ? "true" : "false") + "}"};
        StepOut o;
        o.status = "pass";
        o.summary = device + ": naar GPU " + fx(h2d) + " GB/s, van GPU " + fx(d2h) + " GB/s, kopie in VRAM " + fx(d2d) +
                    " GB/s gekopieerd (= " + fx(2 * d2d) + " GB/s geheugenverkeer, lezen + schrijven)";
        o.data = "{\"device\":" + json_quote(device) + ",\"device_count\":" + std::to_string(count) +
                 ",\"vram_total_mib\":" + std::to_string(total_b >> 20) + ",\"vram_free_mib\":" + std::to_string(free_b >> 20) +
                 ",\"buffer_mib\":" + std::to_string(size >> 20) + ",\"repetitions\":" + std::to_string(kBandwidthReps) +
                 ",\"host_buffer\":" + (res.pinned ? "\"pinned\"" : "\"pageable\"") + ",\"h2d_gb_s\":" + num(h2d) + ",\"d2h_gb_s\":" + num(d2h) +
                 ",\"d2d_copied_gb_s\":" + num(d2d) + ",\"d2d_memory_traffic_gb_s\":" + num(2 * d2d) +
                 ",\"data_verified\":true,\"unit\":\"GB = 1e9 bytes; median\"}";
        return o;
    } catch (const CudaError& e) {
        return {"error", std::string(e.call) + " mislukt met CUDA-foutcode " + std::to_string(e.code), "{}"};
    }
}

// ---- report -------------------------------------------------------------------------------------

std::string hardware_report() {
    const veltra::HwInfo hw = veltra::detect_hardware();
    std::string o = "{\"os\":" + json_quote(rc::clean_text(hw.os, 40)) + ",\"cpu\":" + json_quote(rc::clean_text(hw.cpu_brand, 120)) +
                    ",\"cores_physical\":" + std::to_string(hw.cores_physical) + ",\"cores_logical\":" + std::to_string(hw.cores_logical) +
                    ",\"ram_total_gib\":" + num(hw.ram_total_gib) + ",\"gpus\":[";
    for (size_t i = 0; i < hw.gpus.size(); ++i) {
        const auto& g = hw.gpus[i];
        o += std::string(i ? "," : "") + "{\"name\":" + json_quote(rc::clean_text(g.name, 120)) + ",\"vram_total_gib\":" + num(g.vram_total_gib) +
             ",\"cuda_compute_capability\":" +
             (g.cc_major >= 0 ? json_quote(std::to_string(g.cc_major) + "." + std::to_string(g.cc_minor)) : std::string("null")) + "}";
    }
    return o + "]}";
}

std::string engine_options_report(const EngineContext& ctx) {
    const lb::EngineOptions& o = lb::engine_default_options();
    auto b = [](bool v) { return std::string(v ? "true" : "false"); };
    return "{\"screen_head\":" + std::to_string(o.screen_head) + ",\"int8_head\":" + b(o.int8_head) + ",\"exact_head\":" +
           (o.exact_head < 0 ? std::string("\"platform_default\"") : std::to_string(o.exact_head)) + ",\"turbo\":" + b(o.turbo) +
           ",\"head_vocab\":" + std::to_string(o.head_vocab) + ",\"prefix_cache\":" + b(o.prefix_cache != 0) +
           ",\"speculative_default\":" + b(ctx.spec_default) + ",\"rest\":" + b(ctx.rest) + ",\"mtp_flag\":" + b(ctx.mtp) +
           ",\"note\":\"snelheidsmetingen zonder speculative decoding, REST en MTP; de kwaliteitsvragen met de standaard chatinstellingen\"}";
}

// Only measured numbers, always with the device that produced them.
struct Headline {
    std::string decode, prefill, batch, quality, vram, cuda;
};

void collect_headline(Headline& h, const std::string& step, const StepOut& o) {
    Json d;
    std::string err;
    if (!lbhttp::parse_json(o.data, d, err)) return;
    auto number = [](const Json& obj, const char* k) {
        const Json* v = obj.get(k);
        return v && v->kind == Json::Kind::Number ? v->number : -1.0;
    };
    const Json* points = d.get("points");
    if (step == "decode_sweep" && points) {
        for (const Json& p : points->items) {
            const double v = number(p, "tok_s_median"), c = number(p, "ctx");
            if (v >= 0 && (h.decode.empty() || c <= 500)) h.decode = "genereren " + fx(v) + " tok/s (ctx " + std::to_string(static_cast<int>(c)) + ")";
        }
    } else if (step == "prefill_sweep" && points) {
        for (const Json& p : points->items) {
            const double v = number(p, "prefill_tok_s"), n = number(p, "prompt_tokens");
            if (v >= 0) h.prefill = "inlezen " + fx(v) + " tok/s (" + std::to_string(static_cast<int>(n)) + " tokens)";
        }
    } else if (step == "batch_and_search" && points && !points->items.empty()) {
        const Json& p = points->items.back();
        h.batch = "batch " + std::to_string(static_cast<int>(number(p, "batch"))) + ": " + fx(number(p, "aggregate_tok_s")) + " tok/s totaal";
    } else if (step == "quality_gate" && number(d, "total") >= 0) {
        h.quality = "kwaliteit " + std::to_string(static_cast<int>(number(d, "passed"))) + "/" + std::to_string(static_cast<int>(number(d, "total")));
    } else if (step == "vram_bandwidth" && number(d, "h2d_gb_s") >= 0) {
        h.vram = rc::str_field(d, "device", 80) + ": naar GPU " + fx(number(d, "h2d_gb_s")) + " GB/s, VRAM-kopie " + fx(number(d, "d2d_copied_gb_s")) + " GB/s";
    } else if (step == "cuda_kernels" && o.status != "skipped") {
        h.cuda = std::string("CUDA-kernels ") + (o.status == "pass" ? "PASS" : "FAIL");
    }
}

}  // namespace

const std::vector<SuiteInfo>& suites() {
    static const std::vector<SuiteInfo> all = {
        {"quick_smoke", "Snelle controle", 2, {kVram, kDecode, kQuality}, {0, 500}, true},
        {"cuda_verify", "CUDA-controle", 4, {kCuda, kVram, kQuality}, {}, false},
        {"full_gpu_sweep", "Volledige test", 10, {kCuda, kVram, kDecode, kPrefill, kBatch, kQuality}, {0, 500, 2000, 8000}, false},
        {"search_benchmark", "Batch- en searchbenchmark", 2, {kBatch}, {}, false},
    };
    return all;
}

const SuiteInfo* find_suite(std::string_view id) {
    for (const SuiteInfo& s : suites())
        if (id == s.id) return &s;
    return nullptr;
}

std::vector<std::string> quality_case_ids() {
    std::vector<std::string> out;
    for (const Case& c : kCases) out.push_back(c.id);
    return out;
}

bool quality_check(const std::string& id, const std::string& output, std::string* reason) {
    for (const Case& c : kCases)
        if (id == c.id) {
            const Verdict v = c.check(output);
            if (reason) *reason = v.why;
            return v.ok;
        }
    if (reason) *reason = "unknown case";
    return false;
}

Report run_suite(const SuiteInfo& suite, EngineContext& ctx, const std::atomic<bool>& cancel, const ProgressFn& progress) {
    Run run{ctx, cancel, progress, suite, {}};
    run.progress.total = static_cast<int>(suite.steps.size());
    const auto t0 = Clock::now();
    const std::string started = rc::utc_stamp("%Y-%m-%dT%H:%M:%SZ");
    Report rep;
    rep.status = "done";
    Headline head;
    std::string steps_json;
    bool cancelled = false;
    for (size_t i = 0; i < suite.steps.size(); ++i) {
        const StepInfo& si = suite.steps[i];
        run.progress.step = static_cast<int>(i) + 1;
        run.progress.name = si.name;
        StepOut out;
        const auto s0 = Clock::now();
        if (cancelled || cancel.load()) {
            cancelled = true;
            out = skipped("geannuleerd");
        } else {
            run.message("bezig");
            try {
                const std::string id = si.id;
                if (id == "cuda_kernels") out = step_cuda(run);
                else if (id == "vram_bandwidth") out = step_vram(run);
                else if (id == "decode_sweep") out = step_decode(run);
                else if (id == "prefill_sweep") out = step_prefill(run);
                else if (id == "batch_and_search") out = step_batch(run);
                else if (id == "quality_gate") out = step_quality(run);
            } catch (const Cancelled&) {
                cancelled = true;
                out = skipped("geannuleerd");
            } catch (const std::exception& e) {
                out = {"error", "fout: " + scrub(e.what(), ctx), "{}"};
            }
            if (ctx.engine) ctx.engine->reset();
        }
        const std::string summary = rc::clean_text(out.summary, 600);
        rep.steps.push_back({si.id, si.name, out.status, summary});
        collect_headline(head, si.id, out);
        steps_json += std::string(i ? "," : "") + "{\"id\":" + json_quote(si.id) + ",\"name\":" + json_quote(si.name) +
                      ",\"status\":" + json_quote(out.status) + ",\"summary\":" + json_quote(summary) +
                      ",\"duration_s\":" + num(secs(s0)) + ",\"data\":" + out.data + "}";
    }
    if (cancelled) rep.status = "cancelled";
    std::string parts;
    for (const std::string* p : {&head.decode, &head.prefill, &head.batch, &head.quality}) {
        if (!p->empty()) parts += (parts.empty() ? "" : " \xc2\xb7 ") + *p;
    }
    std::string text;
    if (!parts.empty()) text = "Model op " + ctx.device + ": " + parts;
    for (const std::string* p : {&head.vram, &head.cuda})
        if (!p->empty()) text += (text.empty() ? "" : " \xc2\xb7 ") + *p;
    if (cancelled) text = "Geannuleerd. " + (text.empty() ? std::string("Geen enkele meetstap was afgerond") : text);
    else if (text.empty()) text = "Niets gemeten (zie de redenen per stap)";
    rep.headline = rc::clean_text(text, 400);

    const std::string model = ctx.engine ? "{\"file\":" + json_quote(rc::clean_text(ctx.model_file, 160)) + ",\"id\":" +
                                               json_quote(rc::clean_text(ctx.model_id, 160)) + ",\"context_limit\":" +
                                               std::to_string(ctx.max_context) + ",\"threads\":" + std::to_string(ctx.threads) + "}"
                                         : std::string("null");
    rep.json = "{\"schema\":1,\"suite\":" + json_quote(suite.id) + ",\"title\":" + json_quote(suite.title) + ",\"status\":" +
               json_quote(rep.status) + ",\"started_utc\":" + json_quote(started) + ",\"finished_utc\":" +
               json_quote(rc::utc_stamp("%Y-%m-%dT%H:%M:%SZ")) + ",\"duration_s\":" + num(secs(t0)) + ",\"headline\":" +
               json_quote(rep.headline) + ",\"device\":{\"engine_device\":" + json_quote(ctx.engine ? ctx.device : std::string("geen model geladen")) +
               ",\"gpu_used_by_engine\":" + (ctx.engine && ctx.engine->gpu_active() ? "true" : "false") + ",\"cuda_backend_built\":" +
               (lb::cuda_backend_built() ? "true" : "false") + "},\"hardware\":" + hardware_report() + ",\"model\":" + model +
               ",\"engine_options\":" + engine_options_report(ctx) + ",\"steps\":[" + steps_json + "]}";
    return rep;
}

}  // namespace veltra::tests
