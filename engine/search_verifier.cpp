#include "search_verifier.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>

#include "json.h"

namespace lb {
namespace {

// ---------------------------------------------------------------- JSON prefix scanner
struct Scan {
    const std::string& s;
    size_t i;
    int depth = 0;
    bool eof() const { return i >= s.size(); }
    void ws() { while (!eof() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) i++; }
    JsonState string() {
        i++;   // opening quote
        while (!eof()) {
            const unsigned char c = (unsigned char)s[i++];
            if (c == '"') return JsonState::Complete;
            if (c < 0x20) return JsonState::Invalid;
            if (c == '\\') {
                if (eof()) return JsonState::Incomplete;
                const char e = s[i++];
                if (e == 'u') {
                    for (int k = 0; k < 4; k++) {
                        if (eof()) return JsonState::Incomplete;
                        if (!std::isxdigit((unsigned char)s[i++])) return JsonState::Invalid;
                    }
                } else if (!std::strchr("\"\\/bfnrt", e) || !e) return JsonState::Invalid;
            }
        }
        return JsonState::Incomplete;
    }
    JsonState literal(const char* w) {
        for (; *w; w++) {
            if (eof()) return JsonState::Incomplete;
            if (s[i++] != *w) return JsonState::Invalid;
        }
        return JsonState::Complete;
    }
    JsonState digits() {
        if (eof()) return JsonState::Incomplete;
        if (!std::isdigit((unsigned char)s[i])) return JsonState::Invalid;
        while (!eof() && std::isdigit((unsigned char)s[i])) i++;
        return JsonState::Complete;
    }
    JsonState number() {
        if (s[i] == '-') i++;
        if (eof()) return JsonState::Incomplete;
        if (s[i] == '0') i++;
        else if (JsonState r = digits(); r != JsonState::Complete) return r;
        if (!eof() && s[i] == '.') { i++; if (JsonState r = digits(); r != JsonState::Complete) return r; }
        if (!eof() && (s[i] == 'e' || s[i] == 'E')) {
            i++;
            if (!eof() && (s[i] == '+' || s[i] == '-')) i++;
            if (JsonState r = digits(); r != JsonState::Complete) return r;
        }
        return JsonState::Complete;
    }
    JsonState value() {
        ws();
        if (eof()) return JsonState::Incomplete;
        const char c = s[i];
        if (c == '"') return string();
        if (c == '-' || std::isdigit((unsigned char)c)) return number();
        if (c == 't') return literal("true");
        if (c == 'f') return literal("false");
        if (c == 'n') return literal("null");
        if (c != '{' && c != '[') return JsonState::Invalid;
        if (++depth > 64) return JsonState::Invalid;
        const bool obj = c == '{';
        const char close = obj ? '}' : ']';
        i++;
        ws();
        if (eof()) return JsonState::Incomplete;
        if (s[i] == close) { i++; depth--; return JsonState::Complete; }
        for (;;) {
            if (obj) {
                ws();
                if (eof()) return JsonState::Incomplete;
                if (s[i] != '"') return JsonState::Invalid;
                if (JsonState r = string(); r != JsonState::Complete) return r;
                ws();
                if (eof()) return JsonState::Incomplete;
                if (s[i++] != ':') return JsonState::Invalid;
            }
            if (JsonState r = value(); r != JsonState::Complete) return r;
            ws();
            if (eof()) return JsonState::Incomplete;
            const char d = s[i++];
            if (d == close) { depth--; return JsonState::Complete; }
            if (d != ',') return JsonState::Invalid;
        }
    }
};

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) a++;
    while (b > a && std::isspace((unsigned char)s[b - 1])) b--;
    return s.substr(a, b - a);
}
std::string lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}
// Text after a closed think block, trimmed. An unclosed block leaves nothing visible.
std::string visible_text(const std::string& t) {
    const size_t e = t.rfind("</think>");
    if (e != std::string::npos) return trim(t.substr(e + 8));
    if (t.find("<think>") != std::string::npos) return {};
    return trim(t);
}
// Strips one surrounding ``` / ```json fence (the closing fence may still be missing mid-stream).
std::string unfenced(const std::string& v) {
    if (v.rfind("```", 0) != 0) return v;
    const size_t nl = v.find('\n');
    if (nl == std::string::npos) return {};
    std::string body = v.substr(nl + 1);
    const size_t close = body.rfind("```");
    if (close != std::string::npos) body = body.substr(0, close);
    return trim(body);
}

// Compact dump with sorted object keys, so equal values compare equal.
void canonical(const json::Value& v, std::string& out) {
    if (v.is_object()) {
        std::vector<std::pair<std::string, const json::Value*>> kv;
        for (const auto& e : v.as_object()) kv.push_back({e.first, &e.second});
        std::sort(kv.begin(), kv.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        out += '{';
        for (size_t i = 0; i < kv.size(); i++) {
            if (i) out += ',';
            json::Value::escape_to(out, kv[i].first);
            out += ':';
            canonical(*kv[i].second, out);
        }
        out += '}';
    } else if (v.is_array()) {
        out += '[';
        for (size_t i = 0; i < v.size(); i++) { if (i) out += ','; canonical(v[i], out); }
        out += ']';
    } else if (v.is_number() && std::nearbyint(v.as_double()) == v.as_double() && std::fabs(v.as_double()) < 9e15) {
        out += std::to_string((long long)v.as_double());   // 2 and 2.0 are the same result
    } else v.dump_to(out);
}

// The tool call of a JSON object: (name, arguments). Accepts the common shapes {"name"|"tool"|"function": N, "arguments"|"args"|"parameters": {...}}.
bool as_tool_call(const json::Value& v, std::string* name, const json::Value** args) {
    if (!v.is_object()) return false;
    for (const char* nk : {"name", "tool", "function"}) {
        const json::Value* n = v.find(nk);
        if (!n || !n->is_string()) continue;
        for (const char* ak : {"arguments", "args", "parameters"}) {
            const json::Value* a = v.find(ak);
            if (a && a->is_object()) { *name = n->as_string(); *args = a; return true; }
        }
    }
    return false;
}

bool find_tool_call(const std::string& t, std::string* name, std::string* args_canon) {
    size_t pos = 0;
    for (int tries = 0; tries < 64; tries++) {
        const size_t b = t.find('{', pos);
        if (b == std::string::npos) return false;
        size_t end = 0;
        if (scan_json(t, b, &end) == JsonState::Complete) {
            json::Value v;
            std::string e;
            const json::Value* a = nullptr;
            if (json::parse(t.substr(b, end - b), &v, &e) && as_tool_call(v, name, &a)) {
                args_canon->clear();
                canonical(*a, *args_canon);
                return true;
            }
        }
        pos = b + 1;
    }
    return false;
}

bool type_is(const json::Value& v, const std::string& t) {
    if (t == "object") return v.is_object();
    if (t == "array") return v.is_array();
    if (t == "string") return v.is_string();
    if (t == "boolean") return v.is_bool();
    if (t == "null") return v.is_null();
    if (t == "number") return v.is_number();
    if (t == "integer") return v.is_number() && std::nearbyint(v.as_double()) == v.as_double();
    return true;
}

// JSON-schema subset: type, enum, const, properties, required, additionalProperties:false, items, anyOf/oneOf (as "any"). Other keywords are ignored.
bool schema_ok(const json::Value& s, const json::Value& v, const std::string& at, std::string* why, int depth) {
    if (!s.is_object() || depth > 32) return true;
    if (const json::Value* t = s.find("type")) {
        bool ok = false;
        if (t->is_string()) ok = type_is(v, t->as_string());
        else for (const auto& e : t->as_array()) ok = ok || type_is(v, e.as_string());
        if (!ok) { *why = at + ": wrong type"; return false; }
    }
    std::string cv;
    canonical(v, cv);
    if (const json::Value* c = s.find("const")) {
        std::string cc;
        canonical(*c, cc);
        if (cc != cv) { *why = at + ": not the const value"; return false; }
    }
    if (const json::Value* en = s.find("enum")) {
        bool ok = false;
        for (const auto& e : en->as_array()) { std::string c; canonical(e, c); ok = ok || c == cv; }
        if (!ok) { *why = at + ": not in enum"; return false; }
    }
    for (const char* k : {"anyOf", "oneOf"}) {
        const json::Value* a = s.find(k);
        if (!a || !a->is_array()) continue;
        bool ok = false;
        std::string tmp;
        for (const auto& e : a->as_array()) ok = ok || schema_ok(e, v, at, &tmp, depth + 1);
        if (!ok) { *why = at + ": matches no alternative"; return false; }
    }
    if (v.is_object()) {
        const json::Value* props = s.find("properties");
        if (const json::Value* req = s.find("required"))
            for (const auto& r : req->as_array())
                if (!v.find(r.as_string())) { *why = at + ": missing " + r.as_string(); return false; }
        const json::Value* extra = s.find("additionalProperties");
        for (const auto& e : v.as_object()) {
            const json::Value* ps = props ? props->find(e.first) : nullptr;
            if (ps) { if (!schema_ok(*ps, e.second, at + "." + e.first, why, depth + 1)) return false; }
            else if (extra && extra->is_bool() && !extra->as_bool(true)) { *why = at + ": unexpected key " + e.first; return false; }
            else if (extra && extra->is_object() && !schema_ok(*extra, e.second, at + "." + e.first, why, depth + 1)) return false;
        }
    }
    if (v.is_array())
        if (const json::Value* it = s.find("items"))
            for (size_t i = 0; i < v.size(); i++)
                if (!schema_ok(*it, v[i], at + "[]", why, depth + 1)) return false;
    return true;
}

std::string squash(const std::string& s) {   // lower case, punctuation to spaces, single spaces
    std::string o;
    bool sp = false;
    for (unsigned char c : s) {
        if (c >= 0x80 || std::isalnum(c)) {
            if (sp && !o.empty()) o += ' ';
            sp = false;
            o += (char)std::tolower(c);
        } else sp = true;
    }
    return o;
}
std::string last_number(const std::string& s) {
    std::string last;
    for (size_t i = 0; i < s.size();) {
        if (!std::isdigit((unsigned char)s[i])) { i++; continue; }
        std::string n = (i > 0 && s[i - 1] == '-') ? "-" : "";
        while (i < s.size() && (std::isdigit((unsigned char)s[i]) ||
                                ((s[i] == ',' || s[i] == '.') && i + 1 < s.size() && std::isdigit((unsigned char)s[i + 1])))) {
            if (s[i] != ',') n += s[i];
            i++;
        }
        last = n;
    }
    return last;
}

}  // namespace

JsonState scan_json(const std::string& s, size_t pos, size_t* end) {
    Scan sc{s, pos};
    const JsonState r = sc.value();
    if (end) *end = sc.i;
    return r;
}

bool token_loop(const std::vector<int>& ids) {
    const int n = (int)ids.size();
    for (int p = 1; p <= 64; p++) {
        const int need = std::max(48, 3 * p);   // `need` tokens equal to the token one period earlier
        if (n < need + p) continue;
        int k = 0;
        while (k < need && ids[n - 1 - k] == ids[n - 1 - k - p]) k++;
        if (k == need) return true;
    }
    return false;
}

std::string prune_reason(const std::vector<int>& ids, const std::string& text, const VerifySpec& spec) {
    if (token_loop(ids)) return "repetition loop";
    if (spec.format == OutputFormat::JsonObject || spec.format == OutputFormat::JsonSchema) {
        std::string v = visible_text(text);
        if (v.empty() || (v.size() < 3 && v[0] == '`')) return "";
        v = unfenced(v);
        if (v.empty()) return "";
        if (v[0] != '{' && v[0] != '[') return "not JSON";
        size_t end = 0;
        if (scan_json(v, 0, &end) == JsonState::Invalid) return "broken JSON";
    }
    return "";
}

bool check_format(const std::string& text, const VerifySpec& spec, std::string* why) {
    why->clear();
    if (spec.format == OutputFormat::Free) return true;
    const std::string v = visible_text(text);
    if (spec.format == OutputFormat::ToolCall) {
        std::string name, args;
        if (!find_tool_call(v, &name, &args)) { *why = "no well-formed tool call"; return false; }
        if (!spec.tool_names.empty() && std::find(spec.tool_names.begin(), spec.tool_names.end(), name) == spec.tool_names.end()) {
            *why = "unknown tool " + name;
            return false;
        }
        return true;
    }
    const std::string j = unfenced(v);
    size_t end = 0;
    if (j.empty() || scan_json(j, 0, &end) != JsonState::Complete || !trim(j.substr(end)).empty()) { *why = "invalid JSON"; return false; }
    json::Value val;
    std::string e;
    if (!json::parse(j, &val, &e)) { *why = "invalid JSON"; return false; }
    if (spec.format == OutputFormat::JsonObject) {
        if (!val.is_object()) { *why = "not a JSON object"; return false; }
        return true;
    }
    json::Value schema;
    if (!json::parse(spec.schema_json, &schema, &e)) return true;   // an unreadable schema must not reject every branch
    return schema_ok(schema, val, "$", why, 0);
}

std::string normalized_outcome(const std::string& text) {
    const std::string v = visible_text(text);
    if (v.empty()) return "";
    std::string name, args;
    if (find_tool_call(v, &name, &args)) return "tool:" + name + args;
    {
        const std::string j = unfenced(v);
        size_t end = 0;
        if (!j.empty() && (j[0] == '{' || j[0] == '[') && scan_json(j, 0, &end) == JsonState::Complete && trim(j.substr(end)).empty()) {
            json::Value val;
            std::string e;
            if (json::parse(j, &val, &e)) {
                std::string c = "json:";
                canonical(val, c);
                return c;
            }
        }
    }
    const size_t f = v.find("```");
    if (f != std::string::npos) {
        const size_t nl = v.find('\n', f), close = nl == std::string::npos ? std::string::npos : v.find("```", nl);
        if (close != std::string::npos) {
            std::string code;
            for (char c : v.substr(nl + 1, close - nl - 1))
                if (!std::isspace((unsigned char)c)) code += c;
            return "code:" + code;
        }
    }
    const size_t bx = v.rfind("\\boxed{");
    if (bx != std::string::npos) {
        int depth = 1;
        size_t i = bx + 7;
        for (; i < v.size(); i++) {
            if (v[i] == '{') depth++;
            else if (v[i] == '}' && --depth == 0) break;
        }
        return "ans:" + squash(v.substr(bx + 7, i - bx - 7));
    }
    const std::string low = lower(v);
    const std::string first = squash(v).substr(0, squash(v).find(' '));
    if (first == "yes" || first == "no" || first == "ja" || first == "nee") return "yn:" + first;
    for (const char* key : {"answer is", "antwoord is", "final answer:", "answer:", "antwoord:"}) {
        const size_t p = low.rfind(key);
        if (p == std::string::npos) continue;
        std::string rest = v.substr(p + std::strlen(key));
        rest = trim(rest);
        const size_t nl = rest.find('\n');
        if (nl != std::string::npos) rest = rest.substr(0, nl);
        while (!rest.empty() && (rest.back() == '.' || rest.back() == '*')) rest.pop_back();
        const std::string a = squash(rest);
        if (!a.empty() && a.size() <= 80) return "ans:" + a;
    }
    const std::string sq = squash(v);
    if (sq.size() <= 60) return "txt:" + sq;
    const std::string num = last_number(v);
    return num.empty() ? "" : "num:" + num;
}

SearchVerdict judge_branches(std::vector<BranchResult>& br, const VerifySpec& spec) {
    SearchVerdict out;
    out.branches_evaluated = (int)br.size();
    for (BranchResult& b : br) {
        b.valid = true;
        b.why.clear();
        if (b.pruned) { b.valid = false; b.why = b.why.empty() ? "pruned" : b.why; out.pruned_branches++; }
        else if (visible_text(b.text).empty()) { b.valid = false; b.why = "empty answer"; }
        else if (!check_format(b.text, spec, &b.why)) b.valid = false;
        b.outcome = b.valid ? normalized_outcome(b.text) : std::string();
        if (b.valid) out.valid_branches++;
    }
    for (BranchResult& b : br) {
        if (!b.valid) { b.score = kInvalidScore; continue; }
        int same = 0;
        for (const BranchResult& o : br)
            if (&o != &b && o.valid && !b.outcome.empty() && o.outcome == b.outcome) same++;
        b.score = b.cum_logprob / std::max(1, b.token_count) + kConsensusBonus * same;
    }
    out.winner = 0;
    if (out.valid_branches) {
        for (int i = 1; i < (int)br.size(); i++)
            if (br[i].score > br[out.winner].score) out.winner = i;
    } else {   // nothing passed: fall back to the most probable branch (unpruned ones first)
        auto key = [&](int i) { return std::make_pair(br[i].pruned ? 0 : 1, br[i].cum_logprob / std::max(1, br[i].token_count)); };
        for (int i = 1; i < (int)br.size(); i++)
            if (key(i) > key(out.winner)) out.winner = i;
    }
    if (out.valid_branches && !br.empty()) {
        const BranchResult& w = br[out.winner];
        int agree = 0;
        for (const BranchResult& o : br)
            if (o.valid && !w.outcome.empty() && o.outcome == w.outcome) agree++;
        out.consensus_ratio = (double)agree / out.valid_branches;
    }
    return out;
}

int search_verifier_selftest(std::string* report) {
    int fails = 0, checks = 0;
    auto expect = [&](bool ok, const char* what) {
        checks++;
        if (!ok) { fails++; if (report) *report += std::string("search verifier FAIL: ") + what + "\n"; }
    };
    size_t end = 0;
    expect(scan_json("{\"a\":[1,2.5e3,true,null,\"x\\n\"]}", 0, &end) == JsonState::Complete, "complete object");
    expect(scan_json("{\"a\":[1,", 0, &end) == JsonState::Incomplete, "incomplete array");
    expect(scan_json("{\"a\" 1}", 0, &end) == JsonState::Invalid, "missing colon");
    expect(scan_json("{\"a\":tru", 0, &end) == JsonState::Incomplete, "partial literal");
    expect(scan_json("{\"a\":01}", 0, &end) == JsonState::Invalid, "leading zero");
    std::vector<int> ids;
    for (int i = 0; i < 80; i++) ids.push_back(i);
    expect(!token_loop(ids), "no loop in distinct tokens");
    for (int i = 0; i < 80; i++) ids.push_back(7 + i % 5);
    expect(token_loop(ids), "period 5 loop");
    VerifySpec js;
    js.format = OutputFormat::JsonSchema;
    js.schema_json = "{\"type\":\"object\",\"required\":[\"x\"],\"additionalProperties\":false,\"properties\":{\"x\":{\"type\":\"integer\"}}}";
    std::string why;
    expect(check_format("```json\n{\"x\": 3}\n```", js, &why), "schema accepts");
    expect(!check_format("{\"x\": \"3\"}", js, &why), "schema type");
    expect(!check_format("{\"x\": 3, \"y\": 1}", js, &why), "schema extra key");
    expect(!check_format("{\"x\": 3} trailing", js, &why), "trailing text");
    expect(prune_reason({}, "Sure, here", js) == "not JSON", "prune prose in JSON mode");
    expect(prune_reason({}, "{\"x\": ", js).empty(), "keep JSON prefix");
    VerifySpec tool;
    tool.format = OutputFormat::ToolCall;
    tool.tool_names = {"lookup"};
    expect(check_format("<tool_call>\n{\"name\":\"lookup\",\"arguments\":{\"q\":1}}\n</tool_call>", tool, &why), "tool call accepts");
    expect(!check_format("{\"name\":\"other\",\"arguments\":{}}", tool, &why), "unknown tool");
    expect(normalized_outcome("{\"b\":1,\"a\":2.0}") == normalized_outcome("{ \"a\": 2, \"b\": 1 }"), "canonical json");
    expect(normalized_outcome("So the total is 12. The answer is 42.") == "ans:42", "answer phrase");
    auto mk = [](const char* t, double lp) {
        BranchResult b;
        b.text = t; b.cum_logprob = lp * 10; b.token_count = 10; b.finished = true; b.finish_reason = "stop";
        return b;
    };
    std::vector<BranchResult> br = {mk("The answer is 5.", -0.10), mk("The answer is 8.", -0.20), mk("Answer: 8", -0.25), mk("The answer is 8", -0.30)};
    const SearchVerdict v = judge_branches(br, VerifySpec{});
    expect(v.winner == 1 && v.valid_branches == 4 && std::fabs(v.consensus_ratio - 0.75) < 1e-9, "consensus beats a single likelier branch");
    std::vector<BranchResult> bad = {mk("nope", -0.01), mk("{\"x\":1}", -0.9)};
    const SearchVerdict v2 = judge_branches(bad, js);
    expect(v2.winner == 1 && v2.valid_branches == 1 && bad[0].score == kInvalidScore, "hard format gate");
    if (report) *report += "search verifier selftest: " + std::to_string(checks) + " checks, " + std::to_string(fails) + " failures\n";
    return fails;
}

}  // namespace lb
