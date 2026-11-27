// See tokenizer.h. Byte-level BPE (Qwen/Llama-3/GPT-2) and SentencePiece-BPE (Llama-2) from tokenizer.json.
#include "tokenizer.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <queue>
#include <string_view>
#include <unordered_map>

#include "json.h"
#include "unicode_tables.h"

namespace lb {

// ============================================================================================
// UTF-8 helpers
// ============================================================================================
namespace {

inline void put_utf8(std::string& s, uint32_t cp) {
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

// Decode one code point from *valid* UTF-8 at s[i]; bounds-safe (truncated input yields 0xFFFD).
inline uint32_t dec_cp(const uint8_t* s, size_t i, size_t n, int& len) {
    uint8_t b = s[i];
    if (b < 0x80) { len = 1; return b; }
    if (b < 0xE0) {
        if (i + 1 < n) { len = 2; return ((b & 0x1Fu) << 6) | (s[i + 1] & 0x3Fu); }
    } else if (b < 0xF0) {
        if (i + 2 < n) { len = 3; return ((b & 0x0Fu) << 12) | ((s[i + 1] & 0x3Fu) << 6) | (s[i + 2] & 0x3Fu); }
    } else {
        if (i + 3 < n)
            { len = 4; return ((b & 0x07u) << 18) | ((s[i + 1] & 0x3Fu) << 12) | ((s[i + 2] & 0x3Fu) << 6) | (s[i + 3] & 0x3Fu); }
    }
    len = 1;
    return 0xFFFD;
}

// Length of the valid UTF-8 sequence starting at s[i] (0 when invalid), Unicode Table 3-7 rules.
inline int valid_seq_len(const uint8_t* s, size_t i, size_t n) {
    uint8_t b = s[i];
    if (b < 0x80) return 1;
    int need;
    uint8_t lo = 0x80, hi = 0xBF;
    if (b >= 0xC2 && b <= 0xDF) need = 1;
    else if (b == 0xE0) { need = 2; lo = 0xA0; }
    else if (b >= 0xE1 && b <= 0xEC) need = 2;
    else if (b == 0xED) { need = 2; hi = 0x9F; }
    else if (b == 0xEE || b == 0xEF) need = 2;
    else if (b == 0xF0) { need = 3; lo = 0x90; }
    else if (b >= 0xF1 && b <= 0xF3) need = 3;
    else if (b == 0xF4) { need = 3; hi = 0x8F; }
    else return 0;
    for (int k = 1; k <= need; ++k) {
        if (i + k >= n) return 0;
        uint8_t c = s[i + k];
        if (k == 1) { if (c < lo || c > hi) return 0; }
        else if (c < 0x80 || c > 0xBF) return 0;
    }
    return need + 1;
}

}  // namespace

bool utf8_valid(const std::string& str) {
    const uint8_t* s = reinterpret_cast<const uint8_t*>(str.data());
    size_t n = str.size(), i = 0;
    while (i < n) {
        if (s[i] < 0x80) { ++i; continue; }
        int l = valid_seq_len(s, i, n);
        if (!l) return false;
        i += l;
    }
    return true;
}

std::string utf8_sanitize(const std::string& str) {
    if (utf8_valid(str)) return str;
    const uint8_t* s = reinterpret_cast<const uint8_t*>(str.data());
    size_t n = str.size(), i = 0;
    std::string out;
    out.reserve(n + 8);
    while (i < n) {
        uint8_t b = s[i];
        if (b < 0x80) { out.push_back(static_cast<char>(b)); ++i; continue; }
        int l = valid_seq_len(s, i, n);
        if (l) { out.append(str, i, l); i += l; continue; }
        // maximal invalid subpart: lead byte plus the continuation bytes that were still acceptable
        int need = 0;
        uint8_t lo = 0x80, hi = 0xBF;
        if (b >= 0xC2 && b <= 0xDF) need = 1;
        else if (b == 0xE0) { need = 2; lo = 0xA0; }
        else if (b >= 0xE1 && b <= 0xEC) need = 2;
        else if (b == 0xED) { need = 2; hi = 0x9F; }
        else if (b == 0xEE || b == 0xEF) need = 2;
        else if (b == 0xF0) { need = 3; lo = 0x90; }
        else if (b >= 0xF1 && b <= 0xF3) need = 3;
        else if (b == 0xF4) { need = 3; hi = 0x8F; }
        size_t j = i + 1;
        for (int k = 1; k <= need && j < n; ++k) {
            uint8_t c = s[j];
            bool ok = (k == 1) ? (c >= lo && c <= hi) : (c >= 0x80 && c <= 0xBF);
            if (!ok) break;
            ++j;
        }
        out += "\xEF\xBF\xBD";
        i = j;
    }
    return out;
}

// ============================================================================================
// Unicode classification (letters / numbers / whitespace) + NFC
// ============================================================================================
namespace {

constexpr uint8_t kL = 1, kN = 2, kS = 4, kM = 8;
constexpr uint32_t kFlagTableSize = 0x35000;

struct FlagTable {
    std::vector<uint8_t> t;
    FlagTable() : t(kFlagTableSize, 0) {
        auto fill = [&](const uint32_t (*r)[2], size_t cnt, uint8_t f) {
            for (size_t i = 0; i < cnt; ++i)
                for (uint32_t c = r[i][0]; c <= r[i][1] && c < kFlagTableSize; ++c) t[c] |= f;
        };
        fill(uni::kLetter, uni::kLetterCount, kL);
        fill(uni::kNumber, uni::kNumberCount, kN);
        fill(uni::kSpace, uni::kSpaceCount, kS);
        fill(uni::kMark, uni::kMarkCount, kM);
    }
};
const FlagTable& flag_table() {
    static const FlagTable ft;
    return ft;
}

uint8_t slow_flags(uint32_t cp) {
    auto in = [&](const uint32_t (*r)[2], size_t cnt) {
        size_t lo = 0, hi = cnt;
        while (lo < hi) {
            size_t mid = (lo + hi) / 2;
            if (cp < r[mid][0]) hi = mid;
            else if (cp > r[mid][1]) lo = mid + 1;
            else return true;
        }
        return false;
    };
    uint8_t f = 0;
    if (in(uni::kLetter, uni::kLetterCount)) f |= kL;
    if (in(uni::kNumber, uni::kNumberCount)) f |= kN;
    if (in(uni::kSpace, uni::kSpaceCount)) f |= kS;
    if (in(uni::kMark, uni::kMarkCount)) f |= kM;
    return f;
}

// ---- NFC ------------------------------------------------------------------------------------
int ccc_of(uint32_t cp) {
    if (cp < 0x300) return 0;
    size_t lo = 0, hi = uni::kCccCount;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < uni::kCcc[mid][0]) hi = mid;
        else if (cp > uni::kCcc[mid][1]) lo = mid + 1;
        else return static_cast<int>(uni::kCcc[mid][2]);
    }
    return 0;
}

const uint32_t* decomp_of(uint32_t cp, uint32_t* len) {
    size_t lo = 0, hi = uni::kDecompCount;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        uint32_t k = uni::kDecomp[mid][0];
        if (cp < k) hi = mid;
        else if (cp > k) lo = mid + 1;
        else {
            *len = uni::kDecomp[mid][2];
            return uni::kDecompPool + uni::kDecomp[mid][1];
        }
    }
    return nullptr;
}

uint32_t compose_pair(uint32_t a, uint32_t b) {
    // Hangul
    if (a >= 0x1100 && a <= 0x1112 && b >= 0x1161 && b <= 0x1175)
        return 0xAC00 + ((a - 0x1100) * 21 + (b - 0x1161)) * 28;
    if (a >= 0xAC00 && a <= 0xD7A3 && (a - 0xAC00) % 28 == 0 && b >= 0x11A8 && b <= 0x11C2)
        return a + (b - 0x11A7);
    size_t lo = 0, hi = uni::kComposeCount;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        uint32_t x = uni::kCompose[mid][0], y = uni::kCompose[mid][1];
        if (a < x || (a == x && b < y)) hi = mid;
        else if (a > x || b > y) lo = mid + 1;
        else return uni::kCompose[mid][2];
    }
    return 0;
}

inline bool no_decomp_block(uint32_t cp) {
    return (cp >= 0x3400 && cp <= 0x9FFF) || (cp >= 0x20000 && cp < 0x2F800) || (cp >= 0x30000);
}

std::string nfc(const std::string& in) {
    const uint8_t* s = reinterpret_cast<const uint8_t*>(in.data());
    size_t n = in.size();
    bool slow = false;
    for (size_t i = 0; i < n; ++i)
        if (s[i] >= 0xCC) { slow = true; break; }
    if (!slow) return in;

    std::vector<uint32_t> d;
    d.reserve(n);
    for (size_t i = 0; i < n;) {
        int l;
        uint32_t cp = dec_cp(s, i, n, l);
        i += l;
        if (cp < 0xC0 || no_decomp_block(cp)) { d.push_back(cp); continue; }
        if (cp >= 0xAC00 && cp <= 0xD7A3) {
            uint32_t si = cp - 0xAC00;
            d.push_back(0x1100 + si / 588);
            d.push_back(0x1161 + (si % 588) / 28);
            if (si % 28) d.push_back(0x11A7 + si % 28);
            continue;
        }
        uint32_t dl;
        const uint32_t* dp = decomp_of(cp, &dl);
        if (dp) for (uint32_t k = 0; k < dl; ++k) d.push_back(dp[k]);
        else d.push_back(cp);
    }
    // canonical ordering
    for (size_t i = 0; i < d.size();) {
        if (ccc_of(d[i]) == 0) { ++i; continue; }
        size_t j = i;
        while (j < d.size() && ccc_of(d[j]) != 0) ++j;
        if (j - i > 1)
            std::stable_sort(d.begin() + i, d.begin() + j, [](uint32_t a, uint32_t b) { return ccc_of(a) < ccc_of(b); });
        i = j;
    }
    // canonical composition
    std::vector<uint32_t> o;
    o.reserve(d.size());
    long starter = -1;
    int last = -1;  // ccc of last char appended after the starter; -1 when the starter is last
    for (uint32_t c : d) {
        int cc = ccc_of(c);
        if (starter >= 0 && (last == -1 || (last != 0 && last < cc))) {
            uint32_t comp = compose_pair(o[starter], c);
            if (comp) { o[starter] = comp; continue; }
        }
        if (cc == 0) { starter = static_cast<long>(o.size()); last = -1; }
        else last = cc;
        o.push_back(c);
    }
    std::string out;
    out.reserve(n);
    for (uint32_t c : o) put_utf8(out, c);
    return out;
}

}  // namespace

// ============================================================================================
// Tokenizer implementation
// ============================================================================================
namespace {

// GPT-2 byte <-> unicode map.
struct ByteMap {
    uint32_t b2u[256];
    int16_t u2b[324];  // code points 0..323
    ByteMap() {
        std::fill(std::begin(u2b), std::end(u2b), -1);
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            uint32_t cp = printable ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + n++);
            b2u[b] = cp;
            u2b[cp] = static_cast<int16_t>(b);
        }
    }
};
const ByteMap& byte_map() {
    static const ByteMap m;
    return m;
}

enum class PreTokKind { None, Qwen, Llama3, Gpt2, Qwen36 };

struct NormOp {
    enum Kind { Nfc, Prepend, Replace } kind;
    std::string a, b;
};

struct Added {
    int id = -1;
    std::string content;
    std::string match;  // pattern used for matching (content, or normalized content)
    bool special = false, lstrip = false, rstrip = false, single_word = false, normalized = false;
};

struct TokSet {
    bool first[256] = {};
    std::vector<int> cand[256];  // indices into Impl::added, longest match first
    bool empty = true;
};

struct PairVal { uint32_t rank; int32_t id; };

inline bool is_ascii_ws(unsigned char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r' || c == '\f' || c == '\v'; }
inline bool is_word_byte(unsigned char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || c >= 0x80;
}

}  // namespace

struct Tokenizer::Impl {
    // model
    std::unordered_map<std::string, int> vocab;
    std::vector<const std::string*> id2tok;
    std::vector<std::string> tbytes;    // id -> decoded bytes
    std::vector<uint8_t> tspecial;      // id -> special flag
    std::vector<uint8_t> tvalid;        // id -> exists
    int n_vocab = 0;
    bool byte_level = true;
    bool ignore_merges = false;
    bool byte_fallback = false;
    bool fuse_unk = false;
    int unk_id = -1;
    int byte_tok[256];
    int sp_byte_tok[256];
    // merges: open addressing pair table
    std::vector<uint64_t> pk;
    std::vector<PairVal> pv;
    uint32_t pmask = 0;
    int pshift = 0;
    // normalizer / pre-tokenizer
    std::vector<NormOp> norm;
    PreTokKind pretok = PreTokKind::None;
    // added tokens
    std::vector<Added> added;
    TokSet set_raw, set_norm;
    // post processor
    std::vector<int> prefix_ids, suffix_ids;
    // decoder
    bool dec_sp = false;
    bool dec_strip_start = false;
    // config
    int bos = -1, eos = -1, pad = -1;
    // cache
    mutable std::mutex mu;
    mutable std::unordered_map<std::string, std::vector<int>> cache;

    // ---------------- pair table ----------------
    static inline uint64_t pair_key(int a, int b) { return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b); }
    inline uint32_t slot_of(uint64_t k) const { return static_cast<uint32_t>((k * 0x9E3779B97F4A7C15ull) >> pshift); }

    void init_pairs(size_t count) {
        size_t cap = 16;
        int bits = 4;
        while (cap < count * 2 + 2) { cap <<= 1; ++bits; }
        pk.assign(cap, ~0ull);
        pv.assign(cap, PairVal{0, 0});
        pmask = static_cast<uint32_t>(cap - 1);
        pshift = 64 - bits;
    }
    void insert_pair(int a, int b, uint32_t rank, int id) {
        uint64_t k = pair_key(a, b);
        uint32_t s = slot_of(k);
        while (pk[s] != ~0ull) {
            if (pk[s] == k) return;  // first (lowest rank) wins
            s = (s + 1) & pmask;
        }
        pk[s] = k;
        pv[s] = PairVal{rank, id};
    }
    inline const PairVal* find_pair(int a, int b) const {
        uint64_t k = pair_key(a, b);
        uint32_t s = slot_of(k);
        while (pk[s] != ~0ull) {
            if (pk[s] == k) return &pv[s];
            s = (s + 1) & pmask;
        }
        return nullptr;
    }

    // ---------------- BPE core ----------------
    void merge_symbols(std::vector<int>& s) const {
        size_t n = s.size();
        if (n < 2) return;
        if (n <= 40) {
            for (;;) {
                uint32_t best = 0xFFFFFFFFu;
                size_t bi = 0;
                int bid = 0;
                for (size_t i = 0; i + 1 < s.size(); ++i) {
                    const PairVal* p = find_pair(s[i], s[i + 1]);
                    if (p && p->rank < best) { best = p->rank; bi = i; bid = p->id; }
                }
                if (best == 0xFFFFFFFFu) break;
                s[bi] = bid;
                s.erase(s.begin() + bi + 1);
            }
            return;
        }
        // linked-list + priority queue
        struct Cand { uint32_t rank; int left, right, lid, rid, res; };
        struct Cmp {
            bool operator()(const Cand& a, const Cand& b) const {
                return a.rank != b.rank ? a.rank > b.rank : a.left > b.left;
            }
        };
        std::vector<int> prv(n), nxt(n);
        std::vector<uint8_t> alive(n, 1);
        for (size_t i = 0; i < n; ++i) { prv[i] = static_cast<int>(i) - 1; nxt[i] = (i + 1 < n) ? static_cast<int>(i + 1) : -1; }
        std::priority_queue<Cand, std::vector<Cand>, Cmp> pq;
        auto push = [&](int l, int r) {
            if (l < 0 || r < 0) return;
            const PairVal* p = find_pair(s[l], s[r]);
            if (p) pq.push(Cand{p->rank, l, r, s[l], s[r], p->id});
        };
        for (size_t i = 0; i + 1 < n; ++i) push(static_cast<int>(i), static_cast<int>(i + 1));
        while (!pq.empty()) {
            Cand c = pq.top();
            pq.pop();
            if (!alive[c.left] || !alive[c.right] || nxt[c.left] != c.right || s[c.left] != c.lid || s[c.right] != c.rid) continue;
            s[c.left] = c.res;
            alive[c.right] = 0;
            int nr = nxt[c.right];
            nxt[c.left] = nr;
            if (nr >= 0) prv[nr] = c.left;
            push(prv[c.left], c.left);
            push(c.left, nr);
        }
        size_t w = 0;
        for (int i = 0; i >= 0; i = nxt[i]) s[w++] = s[i];
        s.resize(w);
    }

    // ---------------- byte-level piece ----------------
    void encode_piece(const uint8_t* p, size_t n, std::vector<int>& out) const {
        if (n == 0) return;
        if (n == 1) {
            int id = byte_tok[p[0]];
            if (id >= 0) { out.push_back(id); return; }
        }
        const bool cacheable = n <= 64;
        std::string key;
        if (cacheable) {
            key.assign(reinterpret_cast<const char*>(p), n);
            std::lock_guard<std::mutex> lk(mu);
            auto it = cache.find(key);
            if (it != cache.end()) { out.insert(out.end(), it->second.begin(), it->second.end()); return; }
        }
        std::vector<int> syms;
        if (ignore_merges) {
            const ByteMap& bm = byte_map();
            std::string mapped;
            mapped.reserve(n * 2);
            for (size_t i = 0; i < n; ++i) put_utf8(mapped, bm.b2u[p[i]]);
            auto it = vocab.find(mapped);
            if (it != vocab.end()) syms.push_back(it->second);
        }
        if (syms.empty()) {
            syms.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                int id = byte_tok[p[i]];
                if (id >= 0) syms.push_back(id);
            }
            merge_symbols(syms);
        }
        out.insert(out.end(), syms.begin(), syms.end());
        if (cacheable) {
            std::lock_guard<std::mutex> lk(mu);
            if (cache.size() < 300000) cache.emplace(std::move(key), std::move(syms));
        }
    }

    // ---------------- pre-tokenizer ----------------
    inline uint8_t flags(uint32_t cp) const {
        return cp < kFlagTableSize ? flag_table().t[cp] : slow_flags(cp);
    }

    // end offset of the regex match starting at i (always > i)
    size_t match_at(const uint8_t* s, size_t n, size_t i) const {
        const bool gpt2 = pretok == PreTokKind::Gpt2;
        const bool ci = !gpt2;
        int l0;
        uint32_t c = dec_cp(s, i, n, l0);
        const uint8_t f = flags(c);
        // contractions
        if (c == '\'' && i + 1 < n) {
            auto low = [&](uint8_t x) -> uint8_t { return (ci && x >= 'A' && x <= 'Z') ? static_cast<uint8_t>(x | 0x20) : x; };
            uint8_t a = low(s[i + 1]);
            if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
            if (ci && s[i + 1] == 0xC5 && i + 2 < n && s[i + 2] == 0xBF) return i + 3;  // 'ſ' case-folds to 's'
            if (i + 2 < n) {
                uint8_t b = low(s[i + 2]);
                if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return i + 3;
            }
        }
        size_t j;
        if (!gpt2) {
            // [^\r\n\p{L}\p{N}]?\p{L}+
            const uint8_t kW = pretok == PreTokKind::Qwen36 ? (uint8_t)(kL | kM) : kL;   // Qwen3.5/3.6: [\p{L}\p{M}]+ and the punctuation class also excludes \p{M}
            if (f & kW) {
                j = i;
                while (j < n) { int l; uint32_t d = dec_cp(s, j, n, l); if (!(flags(d) & kW)) break; j += l; }
                return j;
            }
            if (c != '\r' && c != '\n' && !(f & kN)) {
                size_t k = i + l0;
                if (k < n) {
                    int l;
                    uint32_t d = dec_cp(s, k, n, l);
                    if (flags(d) & kW) {
                        j = k;
                        while (j < n) { int l2; uint32_t e = dec_cp(s, j, n, l2); if (!(flags(e) & kW)) break; j += l2; }
                        return j;
                    }
                }
            }
            // \p{N} / \p{N}{1,3}
            if (f & kN) {
                int maxd = (pretok == PreTokKind::Llama3) ? 3 : 1;
                j = i;
                for (int cnt = 0; cnt < maxd && j < n; ++cnt) {
                    int l;
                    uint32_t d = dec_cp(s, j, n, l);
                    if (!(flags(d) & kN)) break;
                    j += l;
                }
                return j;
            }
            //  ?[^\s\p{L}\p{N}]+[\r\n]*
            j = i;
            if (c == ' ') j += 1;
            if (j < n) {
                int l;
                uint32_t d = dec_cp(s, j, n, l);
                const uint8_t kP = (uint8_t)(kS | kW | kN);
                if (!(flags(d) & kP)) {
                    while (j < n) { int l2; uint32_t e = dec_cp(s, j, n, l2); if (flags(e) & kP) break; j += l2; }
                    while (j < n && (s[j] == '\r' || s[j] == '\n')) ++j;
                    return j;
                }
            }
        } else {
            // GPT-2:  ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+
            j = i;
            if (c == ' ') j += 1;
            if (j < n) {
                int l;
                uint32_t d = dec_cp(s, j, n, l);
                uint8_t fd = flags(d);
                uint8_t want = (fd & kL) ? kL : (fd & kN) ? kN : (fd & kS) ? 0 : 255;
                if (want) {
                    while (j < n) {
                        int l2;
                        uint32_t e = dec_cp(s, j, n, l2);
                        uint8_t fe = flags(e);
                        bool ok = (want == 255) ? !(fe & (kS | kL | kN)) : (fe & want) != 0;
                        if (!ok) break;
                        j += l2;
                    }
                    return j;
                }
            }
        }
        if (f & kS) {
            // whitespace run
            size_t run_end = i, prev_start = i, last_nl_end = 0, count = 0;
            while (run_end < n) {
                int l;
                uint32_t d = dec_cp(s, run_end, n, l);
                if (!(flags(d) & kS)) break;
                prev_start = run_end;
                run_end += l;
                if (d == '\r' || d == '\n') last_nl_end = run_end;
                ++count;
            }
            if (!gpt2 && last_nl_end) return last_nl_end;     // \s*[\r\n]+
            if (run_end == n) return run_end;                 // \s+(?!\S) at end of text
            if (count >= 2) return prev_start;                // \s+(?!\S) leaves the last char
            return run_end;                                   // \s+
        }
        return i + l0;
    }

    void encode_segment_bytelevel(const std::string& seg, std::vector<int>& out) const {
        const uint8_t* s = reinterpret_cast<const uint8_t*>(seg.data());
        size_t n = seg.size();
        if (pretok == PreTokKind::None) {
            encode_piece(s, n, out);
            return;
        }
        size_t i = 0;
        while (i < n) {
            size_t e = match_at(s, n, i);
            encode_piece(s + i, e - i, out);
            i = e;
        }
    }

    // ---------------- SentencePiece-style segment ----------------
    void encode_segment_sp(const std::string& seg, std::vector<int>& out) const {
        const uint8_t* s = reinterpret_cast<const uint8_t*>(seg.data());
        size_t n = seg.size();
        std::vector<int> syms;
        syms.reserve(n);
        bool last_unk = false;
        std::string tmp;
        for (size_t i = 0; i < n;) {
            int l;
            dec_cp(s, i, n, l);
            tmp.assign(seg, i, l);
            auto it = vocab.find(tmp);
            if (it != vocab.end()) {
                syms.push_back(it->second);
                last_unk = false;
            } else {
                bool done = false;
                if (byte_fallback) {
                    bool all = true;
                    for (int k = 0; k < l; ++k) if (sp_byte_tok[s[i + k]] < 0) { all = false; break; }
                    if (all) {
                        for (int k = 0; k < l; ++k) syms.push_back(sp_byte_tok[s[i + k]]);
                        done = true;
                        last_unk = false;
                    }
                }
                if (!done && unk_id >= 0) {
                    if (!(fuse_unk && last_unk)) syms.push_back(unk_id);
                    last_unk = true;
                }
            }
            i += l;
        }
        merge_symbols(syms);
        out.insert(out.end(), syms.begin(), syms.end());
    }

    void encode_segment(const std::string& seg, std::vector<int>& out) const {
        if (seg.empty()) return;
        if (byte_level) encode_segment_bytelevel(seg, out);
        else encode_segment_sp(seg, out);
    }

    // ---------------- normalizer ----------------
    std::string normalize(const std::string& in) const {
        if (norm.empty()) return in;
        std::string cur = in;
        for (const NormOp& op : norm) {
            switch (op.kind) {
                case NormOp::Nfc: cur = nfc(cur); break;
                case NormOp::Prepend:
                    if (!cur.empty()) cur = op.a + cur;
                    break;
                case NormOp::Replace: {
                    if (op.a.empty()) break;
                    std::string o;
                    o.reserve(cur.size());
                    size_t pos = 0;
                    for (;;) {
                        size_t f = cur.find(op.a, pos);
                        if (f == std::string::npos) { o.append(cur, pos, std::string::npos); break; }
                        o.append(cur, pos, f - pos);
                        o += op.b;
                        pos = f + op.a.size();
                    }
                    cur.swap(o);
                    break;
                }
            }
        }
        return cur;
    }

    // ---------------- added-token splitting ----------------
    template <class FText, class FTok>
    void split_added(const TokSet& set, const std::string& text, FText on_text, FTok on_tok) const {
        if (set.empty) { if (!text.empty()) on_text(text); return; }
        const uint8_t* s = reinterpret_cast<const uint8_t*>(text.data());
        size_t n = text.size();
        size_t chunk = 0, pos = 0;
        while (pos < n) {
            if (!set.first[s[pos]]) { ++pos; continue; }
            const Added* hit = nullptr;
            for (int idx : set.cand[s[pos]]) {
                const Added& a = added[idx];
                if (pos + a.match.size() > n || memcmp(s + pos, a.match.data(), a.match.size()) != 0) continue;
                if (a.single_word) {
                    bool lb = pos > 0 && is_word_byte(s[pos - 1]);
                    size_t e = pos + a.match.size();
                    bool rb = e < n && is_word_byte(s[e]);
                    if (lb || rb) continue;
                }
                hit = &a;
                break;
            }
            if (!hit) { ++pos; continue; }
            size_t tend = chunk > pos ? pos : pos;
            if (hit->lstrip) while (tend > chunk && is_ascii_ws(s[tend - 1])) --tend;
            if (tend > chunk) on_text(text.substr(chunk, tend - chunk));
            on_tok(*hit);
            pos += hit->match.size();
            if (hit->rstrip) while (pos < n && is_ascii_ws(s[pos])) ++pos;
            chunk = pos;
        }
        if (chunk < n) on_text(text.substr(chunk));
    }

    void build_set(TokSet& set, bool normalized_flag) {
        set = TokSet();
        std::vector<int> idx;
        for (size_t i = 0; i < added.size(); ++i)
            if (added[i].normalized == normalized_flag && !added[i].match.empty()) idx.push_back(static_cast<int>(i));
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return added[a].match.size() > added[b].match.size(); });
        for (int i : idx) {
            uint8_t f = static_cast<uint8_t>(added[i].match[0]);
            set.first[f] = true;
            set.cand[f].push_back(i);
            set.empty = false;
        }
    }
};

Tokenizer::Tokenizer() : impl_(new Impl()) {}
Tokenizer::~Tokenizer() = default;

namespace {

bool fail(std::string* err, const std::string& m) {
    if (err) *err = m;
    return false;
}

bool parse_normalizer(const json::Value& v, std::vector<NormOp>& ops, std::string* err) {
    if (v.is_null()) return true;
    const std::string& t = v["type"].as_string();
    if (t == "Sequence") {
        for (const json::Value& e : v["normalizers"].as_array())
            if (!parse_normalizer(e, ops, err)) return false;
        return true;
    }
    if (t == "NFC") { ops.push_back({NormOp::Nfc, "", ""}); return true; }
    if (t == "Prepend") { ops.push_back({NormOp::Prepend, v["prepend"].as_string(), ""}); return true; }
    if (t == "Replace") {
        const json::Value* p = v.find("pattern");
        if (!p || !p->find("String")) return fail(err, "unsupported Replace normalizer (regex pattern)");
        ops.push_back({NormOp::Replace, (*p)["String"].as_string(), v["content"].as_string()});
        return true;
    }
    return fail(err, "unsupported normalizer type: " + t);
}

bool parse_pretokenizer(const json::Value& v, PreTokKind& kind, bool& has_bytelevel, std::string* err) {
    if (v.is_null()) return true;
    const std::string& t = v["type"].as_string();
    if (t == "Sequence") {
        for (const json::Value& e : v["pretokenizers"].as_array())
            if (!parse_pretokenizer(e, kind, has_bytelevel, err)) return false;
        return true;
    }
    if (t == "ByteLevel") {
        has_bytelevel = true;
        if (v["use_regex"].as_bool(true) && kind == PreTokKind::None) kind = PreTokKind::Gpt2;
        return true;
    }
    if (t == "Split") {
        const json::Value* p = v.find("pattern");
        if (!p || !p->find("Regex")) return fail(err, "unsupported Split pre-tokenizer (non-regex pattern)");
        if (v["behavior"].as_string() != "Isolated") return fail(err, "unsupported Split behavior: " + v["behavior"].as_string());
        if (v["invert"].as_bool(false)) return fail(err, "unsupported Split invert=true");
        const std::string& rx = (*p)["Regex"].as_string();
        if (rx.find("\\p{N}{1,3}") != std::string::npos && rx.find("(?i:'s|'t|'re|'ve|'m|'ll|'d)") == 0) kind = PreTokKind::Llama3;
        else if (rx.find("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") == 0) kind = PreTokKind::Qwen;
        else if (rx.find("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") == 0) kind = PreTokKind::Qwen36;
        else if (rx == "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+") kind = PreTokKind::Gpt2;
        else return fail(err, "unsupported pre-tokenizer regex: " + rx);
        return true;
    }
    return fail(err, "unsupported pre-tokenizer type: " + t);
}

bool decoder_has(const json::Value& v, const char* type) {
    if (v.is_null()) return false;
    if (v["type"].as_string() == type) return true;
    for (const json::Value& e : v["decoders"].as_array())
        if (decoder_has(e, type)) return true;
    return false;
}

std::string token_field(const json::Value& v) {
    if (v.is_string()) return v.as_string();
    if (v.is_object()) return v["content"].as_string();
    return "";
}

}  // namespace

std::unique_ptr<Tokenizer> Tokenizer::from_json(const std::string& tokenizer_json, const std::string& tokenizer_config_json,
                                                std::string* err) {
    json::Value root;
    std::string perr;
    if (!json::parse(tokenizer_json, &root, &perr)) { fail(err, "tokenizer.json: " + perr); return nullptr; }
    if (!root.is_object()) { fail(err, "tokenizer.json: not an object"); return nullptr; }
    const json::Value& model = root["model"];
    const std::string mtype = model["type"].as_string("BPE");
    if (mtype != "BPE") { fail(err, "unsupported model type: " + mtype); return nullptr; }
    if (!model["vocab"].is_object()) { fail(err, "tokenizer.json: model.vocab missing"); return nullptr; }

    std::unique_ptr<Tokenizer> tk(new Tokenizer());
    Impl& m = *tk->impl_;

    // ---- normalizer / pre-tokenizer ----
    std::string e2;
    if (!parse_normalizer(root["normalizer"], m.norm, &e2)) { fail(err, e2); return nullptr; }
    bool has_bytelevel = false;
    if (!parse_pretokenizer(root["pre_tokenizer"], m.pretok, has_bytelevel, &e2)) { fail(err, e2); return nullptr; }
    m.byte_level = has_bytelevel;
    m.ignore_merges = model["ignore_merges"].as_bool(false);
    m.byte_fallback = model["byte_fallback"].as_bool(false);
    m.fuse_unk = model["fuse_unk"].as_bool(false);
    if (m.byte_level && m.byte_fallback) { fail(err, "byte-level + byte_fallback combination not supported"); return nullptr; }
    if (!m.byte_level && m.pretok != PreTokKind::None) { fail(err, "unsupported: regex pre-tokenizer without ByteLevel"); return nullptr; }

    // ---- vocab ----
    const json::Object& vobj = model["vocab"].as_object();
    m.vocab.reserve(vobj.size() * 2 + 512);
    int maxid = -1;
    for (const auto& kv : vobj) {
        int id = static_cast<int>(kv.second.as_int(-1));
        if (id < 0) continue;
        m.vocab.emplace(kv.first, id);
        maxid = std::max(maxid, id);
    }
    // ---- added tokens ----
    for (const json::Value& a : root["added_tokens"].as_array()) {
        Added t;
        t.id = static_cast<int>(a["id"].as_int(-1));
        t.content = a["content"].as_string();
        t.special = a["special"].as_bool(false);
        t.lstrip = a["lstrip"].as_bool(false);
        t.rstrip = a["rstrip"].as_bool(false);
        t.single_word = a["single_word"].as_bool(false);
        t.normalized = a["normalized"].as_bool(false);
        if (t.id < 0 || t.content.empty()) continue;
        m.vocab[t.content] = t.id;
        maxid = std::max(maxid, t.id);
        m.added.push_back(std::move(t));
    }
    m.n_vocab = maxid + 1;
    m.id2tok.assign(m.n_vocab, nullptr);
    for (const auto& kv : m.vocab)
        if (kv.second >= 0 && kv.second < m.n_vocab && !m.id2tok[kv.second]) m.id2tok[kv.second] = &kv.first;

    // ---- decoding bytes per token ----
    m.tbytes.assign(m.n_vocab, std::string());
    m.tspecial.assign(m.n_vocab, 0);
    m.tvalid.assign(m.n_vocab, 0);
    m.dec_sp = !m.byte_level;
    m.dec_strip_start = !m.byte_level && decoder_has(root["decoder"], "Strip");
    {
        const ByteMap& bm = byte_map();
        for (int id = 0; id < m.n_vocab; ++id) {
            const std::string* ts = m.id2tok[id];
            if (!ts) continue;
            m.tvalid[id] = 1;
            std::string& out = m.tbytes[id];
            if (m.byte_level) {
                const uint8_t* s = reinterpret_cast<const uint8_t*>(ts->data());
                size_t n = ts->size();
                bool all = true;
                for (size_t i = 0; i < n;) {
                    int l;
                    uint32_t cp = dec_cp(s, i, n, l);
                    if (cp >= 324 || bm.u2b[cp] < 0) { all = false; break; }
                    out.push_back(static_cast<char>(bm.u2b[cp]));
                    i += l;
                }
                if (!all) out = *ts;
            } else {
                // <0xXX> byte tokens
                if (ts->size() == 6 && (*ts)[0] == '<' && (*ts)[1] == '0' && (*ts)[2] == 'x' && (*ts)[5] == '>' &&
                    isxdigit((unsigned char)(*ts)[3]) && isxdigit((unsigned char)(*ts)[4])) {
                    out.push_back(static_cast<char>(std::stoi(ts->substr(3, 2), nullptr, 16)));
                } else {
                    size_t pos = 0;
                    while (pos < ts->size()) {
                        if (ts->compare(pos, 3, "\xE2\x96\x81") == 0) { out.push_back(' '); pos += 3; }
                        else out.push_back((*ts)[pos++]);
                    }
                }
            }
        }
        for (const Added& a : m.added) {
            if (a.id < m.n_vocab) {
                m.tbytes[a.id] = a.content;  // added tokens decode verbatim
                m.tspecial[a.id] = a.special ? 1 : 0;
                m.tvalid[a.id] = 1;
            }
        }
    }

    // ---- byte tokens ----
    std::fill(std::begin(m.byte_tok), std::end(m.byte_tok), -1);
    std::fill(std::begin(m.sp_byte_tok), std::end(m.sp_byte_tok), -1);
    if (m.byte_level) {
        const ByteMap& bm = byte_map();
        for (int b = 0; b < 256; ++b) {
            std::string k;
            put_utf8(k, bm.b2u[b]);
            auto it = m.vocab.find(k);
            if (it != m.vocab.end()) m.byte_tok[b] = it->second;
        }
    } else {
        for (int b = 0; b < 256; ++b) {
            char buf[8];
            snprintf(buf, sizeof buf, "<0x%02X>", b);
            auto it = m.vocab.find(buf);
            if (it != m.vocab.end()) m.sp_byte_tok[b] = it->second;
        }
        std::string unk = model["unk_token"].as_string();
        if (!unk.empty()) {
            auto it = m.vocab.find(unk);
            if (it != m.vocab.end()) m.unk_id = it->second;
        }
    }

    // ---- merges ----
    const json::Array& merges = model["merges"].as_array();
    m.init_pairs(merges.size());
    {
        std::string ab;
        for (size_t r = 0; r < merges.size(); ++r) {
            const json::Value& mv = merges[r];
            const std::string* a = nullptr;
            const std::string* b = nullptr;
            std::string sa, sb;
            if (mv.is_string()) {
                const std::string& s = mv.as_string();
                size_t sp = s.find(' ', 1);
                if (sp == std::string::npos) continue;
                sa = s.substr(0, sp);
                sb = s.substr(sp + 1);
                a = &sa;
                b = &sb;
            } else if (mv.is_array() && mv.size() == 2) {
                a = &mv[static_cast<size_t>(0)].as_string();
                b = &mv[static_cast<size_t>(1)].as_string();
            } else {
                continue;
            }
            auto ia = m.vocab.find(*a), ib = m.vocab.find(*b);
            if (ia == m.vocab.end() || ib == m.vocab.end()) continue;
            ab.assign(*a);
            ab.append(*b);
            auto ir = m.vocab.find(ab);
            if (ir == m.vocab.end()) continue;
            m.insert_pair(ia->second, ib->second, static_cast<uint32_t>(r), ir->second);
        }
    }

    // ---- added token matching sets ----
    for (Added& a : m.added) a.match = a.normalized ? m.normalize(a.content) : a.content;
    m.build_set(m.set_raw, false);
    m.build_set(m.set_norm, true);

    // ---- post processor (TemplateProcessing "single") ----
    {
        std::vector<const json::Value*> procs;
        const json::Value& pp = root["post_processor"];
        if (pp["type"].as_string() == "Sequence")
            for (const json::Value& e : pp["processors"].as_array()) procs.push_back(&e);
        else
            procs.push_back(&pp);
        for (const json::Value* p : procs) {
            if ((*p)["type"].as_string() != "TemplateProcessing") continue;
            bool after_seq = false;
            for (const json::Value& item : (*p)["single"].as_array()) {
                if (item.find("Sequence")) { after_seq = true; continue; }
                const json::Value* st = item.find("SpecialToken");
                if (!st) continue;
                std::string name = (*st)["id"].as_string();
                std::vector<int> ids;
                const json::Value& spec = (*p)["special_tokens"][name];
                for (const json::Value& x : spec["ids"].as_array()) ids.push_back(static_cast<int>(x.as_int()));
                if (ids.empty()) {
                    auto it = m.vocab.find(name);
                    if (it != m.vocab.end()) ids.push_back(it->second);
                }
                auto& dst = after_seq ? m.suffix_ids : m.prefix_ids;
                dst.insert(dst.end(), ids.begin(), ids.end());
            }
        }
    }

    // ---- config ----
    if (!tokenizer_config_json.empty()) {
        json::Value cfg;
        if (json::parse(tokenizer_config_json, &cfg, nullptr) && cfg.is_object()) {
            auto lookup = [&](const char* key) {
                std::string s = token_field(cfg[key]);
                if (s.empty()) return -1;
                auto it = m.vocab.find(s);
                return it == m.vocab.end() ? -1 : it->second;
            };
            m.bos = lookup("bos_token");
            m.eos = lookup("eos_token");
            m.pad = lookup("pad_token");
        }
    }
    return tk;
}

std::vector<int> Tokenizer::encode(const std::string& text_in, bool add_special) const {
    const Impl& m = *impl_;
    std::vector<int> out;
    out.reserve(text_in.size() / 3 + 8);
    if (add_special) out.insert(out.end(), m.prefix_ids.begin(), m.prefix_ids.end());
    const std::string& text_ref = text_in;
    std::string clean;
    const std::string* text = &text_ref;
    if (!utf8_valid(text_ref)) { clean = utf8_sanitize(text_ref); text = &clean; }

    m.split_added(
        m.set_raw, *text,
        [&](const std::string& raw) {
            std::string nrm = m.normalize(raw);
            m.split_added(
                m.set_norm, nrm, [&](const std::string& seg) { m.encode_segment(seg, out); },
                [&](const Added& a) { out.push_back(a.id); });
        },
        [&](const Added& a) { out.push_back(a.id); });
    if (add_special) out.insert(out.end(), m.suffix_ids.begin(), m.suffix_ids.end());
    return out;
}

void Tokenizer::append_token_bytes(int id, std::string* out, bool skip_special) const {
    const Impl& m = *impl_;
    if (id < 0 || id >= m.n_vocab || !m.tvalid[id]) return;
    if (skip_special && m.tspecial[id]) return;
    out->append(m.tbytes[id]);
}

bool Tokenizer::strips_leading_space() const { return impl_->dec_strip_start; }

std::string Tokenizer::decode(const std::vector<int>& ids, bool skip_special) const {
    std::string raw;
    raw.reserve(ids.size() * 4);
    for (int id : ids) append_token_bytes(id, &raw, skip_special);
    if (impl_->dec_strip_start && !raw.empty() && raw[0] == ' ') raw.erase(0, 1);
    return utf8_sanitize(raw);
}

int Tokenizer::vocab_size() const { return impl_->n_vocab; }
int Tokenizer::token_to_id(const std::string& token) const {
    auto it = impl_->vocab.find(token);
    return it == impl_->vocab.end() ? -1 : it->second;
}
std::string Tokenizer::id_to_token(int id) const {
    if (id < 0 || id >= impl_->n_vocab || !impl_->id2tok[id]) return std::string();
    return *impl_->id2tok[id];
}
int Tokenizer::bos_token_id() const { return impl_->bos; }
int Tokenizer::eos_token_id() const { return impl_->eos; }
int Tokenizer::pad_token_id() const { return impl_->pad; }
bool Tokenizer::is_special(int id) const { return id >= 0 && id < impl_->n_vocab && impl_->tspecial[id]; }
bool Tokenizer::is_byte_level() const { return impl_->byte_level; }

// ============================================================================================
// Detok
// ============================================================================================
std::string Detok::push(int id) {
    std::string tmp;
    tok_->append_token_bytes(id, &tmp, skip_special_);
    if (tmp.empty()) return std::string();
    if (!started_) {
        started_ = true;
        if (tok_->strips_leading_space() && tmp[0] == ' ') tmp.erase(0, 1);
    }
    pending_ += tmp;
    // hold back an incomplete multi-byte tail
    const uint8_t* s = reinterpret_cast<const uint8_t*>(pending_.data());
    size_t n = pending_.size();
    size_t cut = n;
    for (size_t back = 1; back <= 3 && back <= n; ++back) {
        uint8_t b = s[n - back];
        if ((b & 0xC0) == 0x80) continue;  // continuation byte, keep looking for the lead
        int need = (b >= 0xF0 && b <= 0xF4) ? 4 : (b >= 0xE0 && b < 0xF0) ? 3 : (b >= 0xC2 && b < 0xE0) ? 2 : 1;
        if (need > static_cast<int>(back)) {
            bool prefix_ok = true;
            if (back >= 2) {  // second byte must be inside the legal range for this lead
                uint8_t c1 = s[n - back + 1];
                uint8_t lo = 0x80, hi = 0xBF;
                if (b == 0xE0) lo = 0xA0; else if (b == 0xED) hi = 0x9F; else if (b == 0xF0) lo = 0x90; else if (b == 0xF4) hi = 0x8F;
                if (c1 < lo || c1 > hi) prefix_ok = false;
            }
            if (prefix_ok) cut = n - back;
        }
        break;
    }
    std::string emit = utf8_sanitize(pending_.substr(0, cut));
    pending_.erase(0, cut);
    return emit;
}

std::string Detok::flush() {
    std::string emit = utf8_sanitize(pending_);
    pending_.clear();
    return emit;
}

void Detok::reset() {
    pending_.clear();
    started_ = false;
}

}  // namespace lb
