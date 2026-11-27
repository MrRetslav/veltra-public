// engine/hf_convert.cpp - see hf_convert.h. The checkpoint is treated as untrusted input: every size and offset is checked.
#include "q4k_quant.h"
#include "hf_convert.h"

#include "json.h"
#include "lb_common.h"
#include "lbt_model.h"
#include "platform.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
namespace json = lb::json;
using lb::DType;

namespace veltra {
namespace {

constexpr int kSuper = 256;   // weights per Q4_K superblock
constexpr int kSb = 148;      // bytes per (row, superblock)

struct Src {
    const uint8_t* data = nullptr;
    uint32_t dtype = 0;
    std::vector<uint64_t> shape;
    uint64_t numel = 0;
};

struct Checkpoint {
    std::vector<plat::FileMap> maps;
    std::map<std::string, Src> tensors;
    ~Checkpoint() {
        for (auto& m : maps) plat::unmap_file(&m);
    }

    bool add(const std::string& path, std::string* err) {
        plat::FileMap m;
        if (!plat::map_file_ro(path, &m, err)) return false;
        maps.push_back(m);
        auto bad = [&](const char* why) { *err = path + ": " + why; return false; };
        if (m.size < 8) return bad("too small");
        uint64_t hlen = 0;
        std::memcpy(&hlen, m.data, 8);
        if (hlen == 0 || hlen > (64u << 20) || hlen > m.size - 8) return bad("bad safetensors header length");
        json::Value h;
        std::string e;
        if (!json::parse(std::string((const char*)m.data + 8, (size_t)hlen), &h, &e) || !h.is_object()) return bad("bad safetensors header");
        const uint64_t base = 8 + hlen, avail = m.size - base;
        for (const auto& kv : h.as_object()) {
            if (kv.first == "__metadata__") continue;
            const std::string& dt = kv.second["dtype"].as_string();
            Src s;
            size_t esz = 0;
            if (dt == "BF16") { s.dtype = lb::DT_BF16; esz = 2; }
            else if (dt == "F16") { s.dtype = lb::DT_F16; esz = 2; }
            else if (dt == "F32") { s.dtype = lb::DT_F32; esz = 4; }
            else continue;   // not a weight type we use; a required tensor is then reported as missing
            s.numel = 1;
            for (const auto& d : kv.second["shape"].as_array()) {
                const int64_t v = d.as_int(-1);
                if (v <= 0 || v > (1 << 24) || s.numel > (1ull << 36) / (uint64_t)v) return bad("bad tensor shape");
                s.shape.push_back((uint64_t)v);
                s.numel *= (uint64_t)v;
            }
            const auto& off = kv.second["data_offsets"].as_array();
            if (off.size() != 2) return bad("bad data_offsets");
            const int64_t a = off[0].as_int(-1), b = off[1].as_int(-1);
            if (a < 0 || b < a || (uint64_t)b > avail || (uint64_t)(b - a) != s.numel * esz) return bad("tensor outside the file (truncated download?)");
            s.data = m.data + base + (uint64_t)a;
            tensors[kv.first] = std::move(s);
        }
        return true;
    }
    const Src* get(const std::string& name, std::vector<uint64_t> shape, std::string* err) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) { *err = "checkpoint misses " + name; return nullptr; }
        if (it->second.shape != shape) { *err = "unexpected shape of " + name; return nullptr; }
        return &it->second;
    }
};

void read_row(const Src& s, uint64_t first, size_t n, float* out) {
    if (s.dtype == lb::DT_F32) { std::memcpy(out, s.data + 4 * first, 4 * n); return; }
    for (size_t i = 0; i < n; i++) {
        uint16_t h;
        std::memcpy(&h, s.data + 2 * (first + i), 2);
        out[i] = s.dtype == lb::DT_BF16 ? lb::bf16_to_f32(h) : lb::f16_to_f32(h);
    }
}

void parallel_rows(int rows, int threads, const std::function<void(int, int)>& fn) {
    const int t = std::max(1, std::min(threads, rows));
    std::vector<std::thread> pool;
    for (int k = 1; k < t; k++) pool.emplace_back(fn, (int)((int64_t)rows * k / t), (int)((int64_t)rows * (k + 1) / t));
    fn(0, rows / t);
    for (auto& th : pool) th.join();
}

struct OutTensor {
    std::string name;
    uint32_t dtype;
    std::vector<uint64_t> shape;
    uint64_t nbytes;
    std::function<bool(std::vector<uint8_t>&, std::string*)> make;   // fills a buffer of nbytes
};

bool read_small(const fs::path& p, size_t limit, std::string* out) {
    std::FILE* f = std::fopen(p.string().c_str(), "rb");
    if (!f) return false;
    out->clear();
    char buf[1 << 16];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0 && out->size() <= limit) out->append(buf, n);
    std::fclose(f);
    return out->size() <= limit;
}

}  // namespace

bool convert_hf_qwen3(const std::string& src_dir, const std::string& out_path, const std::string& model_name, int threads,
                      ConvertProgress* progress, std::string* err) {
    ConvertProgress local;
    if (!progress) progress = &local;
    if (threads <= 0) threads = plat::hardware_threads();
    const fs::path dir(src_dir);
    std::string cfg_s, tj, tc, gen_s, e;
    if (!read_small(dir / "config.json", 1u << 20, &cfg_s)) { *err = "config.json missing"; return false; }
    if (!read_small(dir / "tokenizer.json", 64u << 20, &tj)) { *err = "tokenizer.json missing"; return false; }
    if (!read_small(dir / "tokenizer_config.json", 4u << 20, &tc)) { *err = "tokenizer_config.json missing"; return false; }
    json::Value cfg, tcv, gen;
    if (!json::parse(cfg_s, &cfg, &e) || !json::parse(tc, &tcv, &e)) { *err = "bad config json: " + e; return false; }
    if (read_small(dir / "generation_config.json", 1u << 20, &gen_s)) json::parse(gen_s, &gen, &e);

    if (cfg["model_type"].as_string() != "qwen3" || cfg["attention_bias"].as_bool(false)) {
        *err = "only dense Qwen3 checkpoints can be converted in the app";
        return false;
    }
    const int L = (int)cfg["num_hidden_layers"].as_int(), H = (int)cfg["hidden_size"].as_int(), I = (int)cfg["intermediate_size"].as_int();
    const int heads = (int)cfg["num_attention_heads"].as_int(), kvh = (int)cfg["num_key_value_heads"].as_int();
    const int hd = (int)cfg["head_dim"].as_int(heads > 0 ? H / heads : 0), V = (int)cfg["vocab_size"].as_int();
    const bool tied = cfg["tie_word_embeddings"].as_bool(false);
    if (L <= 0 || L > 200 || H <= 0 || H > 16384 || I <= 0 || I > 65536 || heads <= 0 || kvh <= 0 || hd <= 0 || V <= 0 || V > (1 << 20) ||
        H % kSuper || I % kSuper || (heads * hd) % kSuper || (kvh * hd) % 4) {
        *err = "unsupported model dimensions";
        return false;
    }

    Checkpoint ck;
    std::vector<std::string> shards;
    std::error_code ec;
    for (const auto& de : fs::directory_iterator(dir, ec))
        if (de.path().extension() == ".safetensors") shards.push_back(de.path().string());
    std::sort(shards.begin(), shards.end());
    if (shards.empty()) { *err = "no .safetensors file in " + src_dir; return false; }
    for (const auto& s : shards)
        if (!ck.add(s, err)) return false;

    std::vector<OutTensor> out;
    uint64_t weights = 0;
    auto vec_f32 = [&](const std::string& name, const std::string& hf, int n) {
        const Src* s = ck.get(hf, {(uint64_t)n}, err);
        if (!s) return false;
        out.push_back({name, lb::DT_F32, {(uint64_t)n}, 4ull * n, [s, n](std::vector<uint8_t>& b, std::string*) {
                           read_row(*s, 0, (size_t)n, (float*)b.data());
                           return true;
                       }});
        return true;
    };
    // per-row INT8 with an f16 scale per row (embeddings and an untied head)
    auto rows_i8 = [&](const std::string& name, const std::string& hf) {
        const Src* s = ck.get(hf, {(uint64_t)V, (uint64_t)H}, err);
        if (!s) return false;
        weights += (uint64_t)V * H;
        auto scales = std::make_shared<std::vector<uint16_t>>(V);
        out.push_back({name, lb::DT_I8, {(uint64_t)V, (uint64_t)H}, (uint64_t)V * H,
                       [s, V, H, scales, threads, progress](std::vector<uint8_t>& b, std::string*) {
                           parallel_rows(V, threads, [&](int r0, int r1) {
                               std::vector<float> w(H);
                               for (int r = r0; r < r1 && !progress->cancel; r++) {
                                   read_row(*s, (uint64_t)r * H, H, w.data());
                                   float amax = 1e-8f;
                                   for (float x : w) amax = std::max(amax, std::fabs(x));
                                   uint16_t sh = lb::f32_to_f16(amax / 127.0f);
                                   if (!sh) sh = 1;
                                   (*scales)[r] = sh;
                                   const float sc = lb::f16_to_f32(sh);
                                   for (int i = 0; i < H; i++)
                                       b[(size_t)r * H + i] = (uint8_t)(int8_t)std::min(127.0f, std::max(-127.0f, std::nearbyintf(w[i] / sc)));
                                   progress->done += H;
                               }
                           });
                           return true;
                       }});
        out.push_back({name + ".scale", lb::DT_F16, {(uint64_t)V}, 2ull * V, [scales, V](std::vector<uint8_t>& b, std::string*) {
                           std::memcpy(b.data(), scales->data(), 2ull * V);
                           return true;
                       }});
        return true;
    };
    // Q4_K linear [m, n]: bytes are [group of 4 rows][superblock of 256 columns][row 0..3][148 bytes]
    auto linear_q4k = [&](const std::string& base, const std::string& hf, int m, int n) {
        const Src* s = ck.get(hf, {(uint64_t)m, (uint64_t)n}, err);
        if (!s) return false;
        weights += (uint64_t)m * n;
        const int nsb = n / kSuper;
        out.push_back({base + ".q4k", lb::DT_U8, {(uint64_t)m, (uint64_t)nsb * kSb}, (uint64_t)m * nsb * kSb,
                       [s, m, n, nsb, threads, progress](std::vector<uint8_t>& b, std::string*) {
                           parallel_rows(m, threads, [&](int r0, int r1) {
                               std::vector<float> w(n);
                               for (int r = r0; r < r1 && !progress->cancel; r++) {
                                   read_row(*s, (uint64_t)r * n, n, w.data());
                                   for (int t = 0; t < nsb; t++)
                                       quant_q4k(w.data() + t * kSuper, b.data() + (((size_t)(r / 4) * nsb + t) * 4 + r % 4) * kSb);
                                   progress->done += n;
                               }
                           });
                           return true;
                       }});
        return true;
    };

    if (!rows_i8("tok_embeddings", "model.embed_tokens.weight") || !vec_f32("output_norm", "model.norm.weight", H)) return false;
    if (!tied && !rows_i8("output", "lm_head.weight")) return false;
    for (int i = 0; i < L; i++) {
        const std::string b = "blk." + std::to_string(i) + ".", p = "model.layers." + std::to_string(i) + ".";
        if (!vec_f32(b + "attn_norm", p + "input_layernorm.weight", H) || !vec_f32(b + "ffn_norm", p + "post_attention_layernorm.weight", H) ||
            !vec_f32(b + "q_norm", p + "self_attn.q_norm.weight", hd) || !vec_f32(b + "k_norm", p + "self_attn.k_norm.weight", hd) ||
            !linear_q4k(b + "q", p + "self_attn.q_proj.weight", heads * hd, H) || !linear_q4k(b + "k", p + "self_attn.k_proj.weight", kvh * hd, H) ||
            !linear_q4k(b + "v", p + "self_attn.v_proj.weight", kvh * hd, H) || !linear_q4k(b + "o", p + "self_attn.o_proj.weight", H, heads * hd) ||
            !linear_q4k(b + "gate", p + "mlp.gate_proj.weight", I, H) || !linear_q4k(b + "up", p + "mlp.up_proj.weight", I, H) ||
            !linear_q4k(b + "down", p + "mlp.down_proj.weight", H, I))
            return false;
    }
    auto blob = [&](const std::string& name, const std::string* s) {
        out.push_back({name, lb::DT_U8, {s->size()}, s->size(), [s](std::vector<uint8_t>& b, std::string*) {
                           std::memcpy(b.data(), s->data(), s->size());
                           return true;
                       }});
    };
    blob("tokenizer.json", &tj);
    blob("tokenizer_config.json", &tc);

    // ---- meta ----
    std::vector<int64_t> eos;
    auto add_eos = [&](const json::Value& v) {
        auto one = [&](const json::Value& x) {
            const int64_t id = x.as_int(-1);
            if (id >= 0 && std::find(eos.begin(), eos.end(), id) == eos.end()) eos.push_back(id);
        };
        if (v.is_array()) for (const auto& x : v.as_array()) one(x);
        else one(v);
    };
    add_eos(gen["eos_token_id"]);
    add_eos(cfg["eos_token_id"]);
    std::string meta = "{\"format\":\"lbt\",\"version\":1,\"arch\":\"qwen3\",\"model_name\":";
    json::Value::escape_to(meta, model_name);
    auto num = [&](const char* k, double v) {
        char b[96];
        std::snprintf(b, sizeof b, ",\"%s\":%.12g", k, v);
        meta += b;
    };
    num("hidden_size", H); num("num_layers", L); num("num_heads", heads); num("num_kv_heads", kvh); num("head_dim", hd);
    num("intermediate_size", I); num("vocab_size", V);
    num("rope_theta", cfg["rope_theta"].as_double(1000000.0)); num("rms_norm_eps", cfg["rms_norm_eps"].as_double(1e-6));
    num("max_position_embeddings", (double)cfg["max_position_embeddings"].as_int(32768));
    num("bos_token_id", (double)cfg["bos_token_id"].as_int(-1));
    meta += std::string(",\"tie_word_embeddings\":") + (tied ? "true" : "false") + ",\"qk_norm\":true,\"attn_bias\":false,\"embed_dtype\":\"i8\"";
    meta += ",\"eos_token_ids\":[";
    for (size_t i = 0; i < eos.size(); i++) meta += (i ? "," : "") + std::to_string(eos[i]);
    meta += "],\"chat_template\":";
    json::Value::escape_to(meta, tcv["chat_template"].as_string(""));
    meta += ",\"converted_by\":\"veltra hf_convert\"}";

    // ---- layout, then tensors one by one (peak memory = the largest single output tensor) ----
    auto align = [](uint64_t x) { return (x + 63) / 64 * 64; };
    const uint64_t meta_off = sizeof(lb::LbtHeader), table_off = align(meta_off + meta.size());
    const uint64_t data_off = align(table_off + sizeof(lb::LbtEntry) * out.size());
    std::vector<lb::LbtEntry> table(out.size());
    uint64_t off = data_off;
    for (size_t i = 0; i < out.size(); i++) {
        lb::LbtEntry& t = table[i];
        std::memset(&t, 0, sizeof t);
        std::snprintf(t.name, sizeof t.name, "%s", out[i].name.c_str());
        t.dtype = out[i].dtype;
        t.ndim = (uint32_t)out[i].shape.size();
        for (size_t d = 0; d < out[i].shape.size() && d < 4; d++) t.shape[d] = out[i].shape[d];
        t.offset = off;
        t.nbytes = out[i].nbytes;
        off = align(off + out[i].nbytes);
    }
    lb::LbtHeader h;
    std::memset(&h, 0, sizeof h);
    std::memcpy(h.magic, "VELTRA\0\0", 8);
    h.version = 1;
    h.n_tensors = (uint32_t)out.size();
    h.meta_off = meta_off;
    h.meta_len = meta.size();
    h.table_off = table_off;
    h.data_off = data_off;

    progress->total = weights;
    progress->done = 0;
    const std::string part = out_path + ".part";
    std::FILE* f = std::fopen(part.c_str(), "wb");
    if (!f) { *err = "cannot write " + part; return false; }
    uint64_t pos = 0;
    bool ok = true;
    auto put = [&](const void* p, size_t n) {
        ok = ok && std::fwrite(p, 1, n, f) == n;
        pos += n;
    };
    auto pad_to = [&](uint64_t target) {
        static const char zeros[64] = {};
        while (pos < target && ok) put(zeros, (size_t)std::min<uint64_t>(64, target - pos));
    };
    put(&h, sizeof h);
    put(meta.data(), meta.size());
    pad_to(table_off);
    put(table.data(), sizeof(lb::LbtEntry) * table.size());
    std::vector<uint8_t> buf;
    for (size_t i = 0; i < out.size() && ok && !progress->cancel; i++) {
        pad_to(table[i].offset);
        buf.assign((size_t)out[i].nbytes, 0);
        if (!out[i].make(buf, err)) { ok = false; break; }
        put(buf.data(), buf.size());
    }
    if (ok) pad_to(off);
    ok = (std::fclose(f) == 0) && ok;
    if (progress->cancel) { *err = "cancelled"; ok = false; }
    else if (!ok && err->empty()) *err = "write failed (disk full?)";
    if (ok) {
        fs::rename(part, out_path, ec);
        if (ec) { *err = "rename failed: " + ec.message(); ok = false; }
    }
    if (!ok) fs::remove(part, ec);
    return ok;
}

}  // namespace veltra
