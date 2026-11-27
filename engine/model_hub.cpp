// engine/model_hub.cpp - built-in .vlt catalog with a per-machine fit status, local model scan, background downloader.
// See hub.h. Downloads: WinINet on Windows (system DLL), curl via posix_spawnp elsewhere (no shell involved).
#include "hub.h"
#include "catalog.h"
#include "hf_convert.h"
#include "lb_common.h"
#include "sha256.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <wininet.h>
#else
#  include <csignal>
#  include <spawn.h>
#  include <sys/wait.h>
extern char** environ;
#endif

namespace fs = std::filesystem;

namespace veltra {
namespace {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

std::string jq(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
        else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += (char)c;
    }
    return o + "\"";
}
std::string jn(double v, int prec = 2) { char b[32]; std::snprintf(b, sizeof b, "%.*f", prec, v); return b; }

// One catalog entry. size_gib is the .vlt file size (the RAM a CPU run needs on top of the KV cache);
// ref_tps is decode tok/s at ctx 0 on the reference box (Ryzen 9 3900, DDR4 ~29 GB/s, 8 threads):
// measured where `ref_measured`, otherwise a bandwidth estimate (active bytes per token / 29 GB/s).
struct CatalogEntry {
    const char* id;
    const char* name;
    const char* file;
    const char* params;     // total / active
    const char* quant;
    double size_gib;
    double ref_tps;
    bool ref_measured;
    const char* hf_source;  // upstream weights the .vlt is built from
    const char* url;        // direct .vlt download URL ("" = not published; env VELTRA_MODEL_BASE_URL + file is used then)
    const char* notes;
    // Set for official dense Qwen3 checkpoints: the app downloads them from Hugging Face and converts them here (hf_convert).
    const char* hf_repo = "";
    int hf_shards = 0;           // 1 = model.safetensors, n = model-0000i-of-0000n.safetensors
    uint64_t hf_bytes = 0;       // bytes of the files fetched (progress and the free-space check)
    // A ready-made .vlt at `url`, pinned by size and SHA-256: downloaded as is (resumable), nothing to convert.
    uint64_t url_bytes = 0;
    const char* url_sha256 = "";
};

// Sizes of the two MoE files are the real files on the reference server. The dense Qwen3 rows are
// built on the user's machine: size_gib = size of the converted INT4 + INT8-embedding file, hf_bytes = download size.
const CatalogEntry kCatalog[] = {
    {"qwen3-0.6b", "Qwen3-0.6B", "qwen3-0.6b.vlt", "0,75B dense", "Q4_K + INT8 embeddings", 0.40, 40.0, false,
     "Qwen/Qwen3-0.6B", "", "Piepklein en snel; voor testen en simpele vragen. Maakt vaker fouten.", "Qwen/Qwen3-0.6B", 1, 1514735660ull},
    {"qwen3-1.7b", "Qwen3-1.7B", "qwen3-1.7b.vlt", "2B dense", "Q4_K + INT8 embeddings", 1.07, 24.9, true,
     "Qwen/Qwen3-1.7B", "", "Klein allround model; past op bijna elke pc.", "Qwen/Qwen3-1.7B", 2, 4074988000ull},
    {"qwen3-4b", "Qwen3-4B", "qwen3-4b.vlt", "4B dense", "Q4_K + INT8 embeddings", 2.35, 10.0, false,
     "Qwen/Qwen3-4B", "", "Middenmaat voor een laptop met 8 GB RAM.", "Qwen/Qwen3-4B", 3, 8056428000ull},
    {"qwen3-8b", "Qwen3-8B", "qwen3-8b.vlt", "8,2B dense", "Q4_K + INT8 embeddings", 4.92, 5.0, false,
     "Qwen/Qwen3-8B", "", "Het sterkste dense model; reken op 7 GB vrij RAM.", "Qwen/Qwen3-8B", 5, 16394000000ull},
    {"qwen3-30b-a3b", "Qwen3-30B-A3B-Instruct-2507", "q30b-qk-best.vlt", "30.5B / 3.3B actief (MoE, 128 experts top-8)",
     "experts Q2_K/Q3_K/Q4_K (imatrix) + INT3-GPTQ attention", 11.14, 20.6, true, "Qwen/Qwen3-30B-A3B-Instruct-2507", "",
     "eval_raptor 20/20, sanity10 10/10."},
    {"qwen3.6-35b-a3b", "Qwen3.6-35B-A3B", "q36-35b-qk.vlt", "35B / 3B actief (hybride Gated DeltaNet + MoE, 256 experts)",
     "experts IQ2_XS/IQ3_XXS/IQ4_XS + Q-K attention", 11.90, 14.1, true, "Qwen/Qwen3.6-35B-A3B", "",
     "eval_raptor 20/20; --thinking 0|1."},
};

// Attention shape per catalog id (layers with a KV cache, KV heads, head size), read from the model files / configs.
struct KvShape { const char* id; int layers, kv_heads, head_dim; };
const KvShape kKvShapes[] = {
    {"qwen3-0.6b", 28, 8, 128}, {"qwen3-1.7b", 28, 8, 128}, {"qwen3-4b", 36, 8, 128}, {"qwen3-8b", 36, 8, 128},
    {"qwen3-30b-a3b", 48, 4, 128}, {"qwen3.6-35b-a3b", 10, 2, 256},
};
// int8 KV cache: per layer, KV head and position, head_dim bytes for K and V plus one f32 scale each per 16 dims.
double kv_gib(const char* id, int ctx) {
    for (const KvShape& k : kKvShapes)
        if (std::strcmp(k.id, id) == 0) return (double)k.layers * k.kv_heads * ctx * (2.0 * k.head_dim + 8.0 * (k.head_dim / 16)) / kGiB;
    return 0.0;
}

// Official files of the convertible checkpoints, pinned to one repository revision. sha256 = "" for the small plain files
// (their size is checked; the pinned revision over TLS fixes their content).
struct HfPin { const char* id; const char* rev; };
const HfPin kHfPins[] = {
    {"qwen3-0.6b", "c1899de289a04d12100db370d81485cdf75e47ca"}, {"qwen3-1.7b", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"},
    {"qwen3-4b", "1cfa9a7208912126459214e8b04321603b3df60c"}, {"qwen3-8b", "b968826d9c46dd6066d109eabc6255188de91218"},
};
struct HfFile { const char* id; const char* name; uint64_t size; const char* sha256; };
const char* const kTokSha = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4";
const HfFile kHfFiles[] = {
    {"qwen3-0.6b", "config.json", 726, ""}, {"qwen3-0.6b", "generation_config.json", 239, ""}, {"qwen3-0.6b", "tokenizer_config.json", 9732, ""},
    {"qwen3-0.6b", "tokenizer.json", 11422654, kTokSha},
    {"qwen3-0.6b", "model.safetensors", 1503300328ull, "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"},
    {"qwen3-1.7b", "config.json", 726, ""}, {"qwen3-1.7b", "generation_config.json", 239, ""}, {"qwen3-1.7b", "tokenizer_config.json", 9732, ""},
    {"qwen3-1.7b", "tokenizer.json", 11422654, kTokSha},
    {"qwen3-1.7b", "model-00001-of-00002.safetensors", 3441185608ull, "169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5"},
    {"qwen3-1.7b", "model-00002-of-00002.safetensors", 622329984ull, "912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9"},
    {"qwen3-4b", "config.json", 726, ""}, {"qwen3-4b", "generation_config.json", 239, ""}, {"qwen3-4b", "tokenizer_config.json", 9732, ""},
    {"qwen3-4b", "tokenizer.json", 11422654, kTokSha},
    {"qwen3-4b", "model-00001-of-00003.safetensors", 3957900840ull, "328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223"},
    {"qwen3-4b", "model-00002-of-00003.safetensors", 3987450520ull, "6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5"},
    {"qwen3-4b", "model-00003-of-00003.safetensors", 99630640ull, "e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1"},
    {"qwen3-8b", "config.json", 728, ""}, {"qwen3-8b", "generation_config.json", 239, ""}, {"qwen3-8b", "tokenizer_config.json", 9732, ""},
    {"qwen3-8b", "tokenizer.json", 11422654, kTokSha},
    {"qwen3-8b", "model-00001-of-00005.safetensors", 3996250744ull, "31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f"},
    {"qwen3-8b", "model-00002-of-00005.safetensors", 3993160032ull, "5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282"},
    {"qwen3-8b", "model-00003-of-00005.safetensors", 3959604768ull, "c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836"},
    {"qwen3-8b", "model-00004-of-00005.safetensors", 3187841392ull, "b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a"},
    {"qwen3-8b", "model-00005-of-00005.safetensors", 1244659840ull, "20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff"},
};

std::string entry_url(const CatalogEntry& e) {
    if (e.url[0]) return e.url;
    const char* base = lb::env_get("MODEL_BASE_URL");   // VELTRA_MODEL_BASE_URL (or LB_MODEL_BASE_URL)
    if (!base || !*base) return {};
    std::string b = base;
    if (b.back() != '/') b += '/';
    return b + e.file;
}

struct Fit { const char* status; const char* color; std::string reason; const char* runs_on; };

// Status rules (GiB): GPU READY  size + 1.2 <= free VRAM of the best GPU
//                     CPU READY  size + 2.0 <= free RAM and AVX2 (or NEON)
//                     HYBRID / TIGHT  fits in free VRAM + free RAM together, or in RAM with only 0.5 GiB headroom (low context)
//                     TOO LARGE  none of the above
Fit fit_for(const HwInfo& hw, double size, bool cuda_built) {
    double vram = 0;
    for (const GpuInfo& g : hw.gpus) vram = std::max(vram, g.vram_free_gib);
    const bool cpu_isa = hw.avx2 || hw.neon;
    if (size + 1.2 <= vram)
        return {"GPU READY", "green", "past volledig in " + jn(vram, 1) + " GiB vrij VRAM" +
                                          (cuda_built ? std::string() : std::string(" (CUDA-build nodig; deze build draait op CPU)")),
                cuda_built ? "cuda" : "cpu"};
    if (!cpu_isa)
        return {"CPU UNSUPPORTED", "gray", "CPU zonder AVX2: de kernels van Veltra hebben minimaal AVX2 nodig", "none"};
    if (size + 2.0 <= hw.ram_free_gib)
        return {"CPU READY", "blue", "past in " + jn(hw.ram_free_gib, 1) + " GiB vrij RAM", "cpu"};
    if (vram > 0 && size + 1.2 + 0.5 <= vram + hw.ram_free_gib)
        return {"HYBRID / TIGHT", "orange", "alleen met GPU+CPU-split (" + jn(vram, 1) + " GiB VRAM + " + jn(hw.ram_free_gib, 1) + " GiB RAM)",
                cuda_built ? "cuda+cpu" : "cpu"};
    if (size + 0.5 <= hw.ram_free_gib)
        return {"HYBRID / TIGHT", "orange", "past net in RAM; gebruik een lage context en sluit andere programma's", "cpu"};
    return {"TOO LARGE", "red", "nodig ~" + jn(size + 2.0, 1) + " GiB, vrij " + jn(hw.ram_free_gib, 1) + " GiB RAM" +
                                    (vram > 0 ? " + " + jn(vram, 1) + " GiB VRAM" : std::string()), "none"};
}

bool has_model_ext(const std::string& f) {
    auto ends = [&](const char* s) { const size_t n = std::strlen(s); return f.size() > n && f.compare(f.size() - n, n, s) == 0; };
    return ends(".vlt") || ends(".lbt");
}
bool safe_file_name(const std::string& f) {
    if (f.empty() || f.size() > 128 || f[0] == '.') return false;
    for (char c : f)
        if (!(std::isalnum((unsigned char)c) || c == '.' || c == '-' || c == '_')) return false;
    return has_model_ext(f);
}
bool safe_url(const std::string& u) {
    if (u.rfind("https://", 0) != 0 && u.rfind("http://", 0) != 0) return false;
    if (u.size() > 2048) return false;
    for (unsigned char c : u)
        if (c <= 0x20 || c >= 0x7f || c == '"' || c == '\'' || c == '`' || c == '\\' || c == '<' || c == '>') return false;
    return true;
}

// ---------------------------------------------------------------- downloader state (one download at a time)
struct Dl {
    std::mutex mu;
    std::atomic<bool> running{false};
    std::string id, file, url, part, error, state = "idle";   // idle | running | done | error
    std::string phase = "download";                            // download | verify | convert
    std::atomic<uint64_t> check_done{0}, check_total{0};        // checksum progress of the current file
    std::atomic<bool> verified{false};                         // every pinned checksum matched
    std::atomic<uint64_t> reused{0};                           // bytes that were already on disk: not part of the speed
    uint64_t total = 0;
    std::atomic<bool> cancel{false};
    ConvertProgress conv;
    std::atomic<uint64_t> done{0};
    std::chrono::steady_clock::time_point t0;
};
Dl& dl() { static Dl d; return d; }

#if defined(_WIN32)
bool fetch_to_file(const std::string& url, const std::string& part, uint64_t have, std::string* err) {
    HINTERNET net = InternetOpenA("Veltra/1.0", INTERNET_OPEN_TYPE_PRECONFIG, nullptr, nullptr, 0);
    if (!net) { *err = "InternetOpen failed"; return false; }
    const std::string range = have ? "Range: bytes=" + std::to_string(have) + "-\r\n" : std::string();
    HINTERNET h = InternetOpenUrlA(net, url.c_str(), range.empty() ? nullptr : range.c_str(), range.empty() ? 0 : (DWORD)-1L,
                                   INTERNET_FLAG_RELOAD | INTERNET_FLAG_NO_CACHE_WRITE | INTERNET_FLAG_NO_UI, 0);
    if (!h) { InternetCloseHandle(net); *err = "cannot open URL (" + std::to_string(GetLastError()) + ")"; return false; }
    char buf[64];
    DWORD len = sizeof buf, idx = 0;
    bool resumed = false;
    if (HttpQueryInfoA(h, HTTP_QUERY_STATUS_CODE, buf, &len, &idx)) {
        const int code = std::atoi(buf);
        resumed = have && code == 206;
        if (code >= 400) { InternetCloseHandle(h); InternetCloseHandle(net); *err = "HTTP " + std::to_string(code); return false; }
    }
    len = sizeof buf; idx = 0;
    uint64_t expect = 0, wrote = 0;
    if (HttpQueryInfoA(h, HTTP_QUERY_CONTENT_LENGTH, buf, &len, &idx)) {
        expect = std::strtoull(buf, nullptr, 10);
        std::lock_guard<std::mutex> lk(dl().mu);
        if (dl().total == 0) dl().total = expect;
    }
    if (have && !resumed) dl().done -= std::min<uint64_t>(dl().done.load(), have);   // the server sends the whole file again
    std::FILE* f = std::fopen(part.c_str(), resumed ? "ab" : "wb");
    if (!f) { InternetCloseHandle(h); InternetCloseHandle(net); *err = "cannot write " + part; return false; }
    std::vector<char> chunk(1 << 20);
    bool ok = true;
    for (;;) {
        DWORD got = 0;
        if (!InternetReadFile(h, chunk.data(), (DWORD)chunk.size(), &got)) { ok = false; *err = "read error"; break; }
        if (got == 0) break;
        if (dl().cancel) { ok = false; *err = "cancelled"; break; }
        if (std::fwrite(chunk.data(), 1, got, f) != got) { ok = false; *err = "disk full?"; break; }
        dl().done += got;
        wrote += got;
    }
    if (ok && expect && wrote != expect) { ok = false; *err = "download incomplete"; }
    std::fclose(f);
    InternetCloseHandle(h);
    InternetCloseHandle(net);
    return ok;
}
#else
bool fetch_to_file(const std::string& url, const std::string& part, uint64_t have, std::string* err) {
    // curl as a child process (argv, no shell). Progress = size of the .part file, polled while curl runs.
    // "-C -" continues an existing partial file with a Range request
    const char* argv[] = {"curl", "-fsSL", "--max-redirs", "5", "--retry", "2", "-C", "-", "-o", part.c_str(), url.c_str(), nullptr};
    pid_t pid = 0;
    if (posix_spawnp(&pid, "curl", nullptr, nullptr, const_cast<char* const*>(argv), environ) != 0) {
        *err = "curl not found (install curl or download the .vlt by hand into the models folder)";
        return false;
    }
    int status = 0;
    const uint64_t base = dl().done.load() - std::min<uint64_t>(dl().done.load(), have);
    for (;;) {
        const pid_t r = waitpid(pid, &status, WNOHANG);
        std::error_code ec;
        const auto sz = fs::file_size(part, ec);
        if (!ec) dl().done = base + sz;
        if (r == pid) break;
        if (dl().cancel) kill(pid, SIGTERM);
        if (r < 0) { *err = "waitpid failed"; return false; }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        *err = "curl failed (exit " + std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1) + ")";
        return false;
    }
    return true;
}
#endif

bool file_sha256(const std::string& path, std::string* hex, std::string* err) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { *err = "cannot read " + path; return false; }
    Dl& d = dl();
    Sha256 sha;
    std::vector<char> buf(4 << 20);
    size_t n;
    while ((n = std::fread(buf.data(), 1, buf.size(), f)) > 0 && !d.cancel) {
        sha.update(buf.data(), n);
        d.check_done += n;
    }
    const bool bad = std::ferror(f) != 0;
    std::fclose(f);
    if (d.cancel) { *err = "cancelled"; return false; }
    if (bad) { *err = "read error in " + path; return false; }
    *hex = sha.hex();
    return true;
}

void set_phase(const char* p) {
    std::lock_guard<std::mutex> lk(dl().mu);
    dl().phase = p;
}

// One pinned file to `final_path`, through <final_path>.part: continues a partial file with a Range request and gives the
// file its final name only after its size and, where pinned, its SHA-256 match.
bool fetch_pinned(const std::string& url, const fs::path& final_path, uint64_t size, const char* sha256, const char* name,
                  std::string* err) {
    Dl& d = dl();
    std::error_code ec;
    const std::string part = final_path.string() + ".part";
    uint64_t have = fs::file_size(part, ec);
    if (ec) have = 0;
    if (have > size) { fs::remove(part, ec); have = 0; }
    d.done += have;
    d.reused += have;
    if (have < size) {
        set_phase("download");
        if (!fetch_to_file(url, part, have, err)) {
            if (d.cancel) *err = "cancelled";
            else *err = std::string(name) + ": " + *err + " (het deel dat binnen is blijft bewaard; probeer opnieuw om te hervatten)";
            return false;
        }
    }
    const uint64_t got = fs::file_size(part, ec);
    if (ec || got != size) {
        *err = std::string(name) + ": verkeerde grootte na downloaden";
        if (!ec && got > size) fs::remove(part, ec);
        return false;
    }
    if (sha256[0]) {
        d.check_done = 0;
        d.check_total = size;
        set_phase("verify");
        std::string hex;
        if (!file_sha256(part, &hex, err)) return false;
        if (hex != sha256) {
            fs::remove(part, ec);
            *err = std::string(name) + ": SHA-256 klopt niet, bestand verwijderd";
            return false;
        }
    }
    fs::rename(part, final_path, ec);
    if (ec) { *err = "rename failed: " + ec.message(); return false; }
    return true;
}

// Downloads the pinned files of an official checkpoint into <models>/.hf-<id>/, converts them to `dst` and removes the
// originals. Partial files stay on disk (disconnect, cancel, app closed), so the next attempt continues at the same byte.
// A file gets its final name only after its size and, where pinned, its SHA-256 match.
bool install_from_hf(const CatalogEntry& e, const std::string& work, const std::string& dst, std::string* err) {
    const char* rev = nullptr;
    for (const HfPin& p : kHfPins)
        if (std::strcmp(p.id, e.id) == 0) rev = p.rev;
    if (!rev) { *err = "no pinned revision for this model"; return false; }
    std::error_code ec;
    fs::create_directories(work, ec);
    if (ec) { *err = "cannot create " + work; return false; }
    Dl& d = dl();
    for (const HfFile& f : kHfFiles) {
        if (std::strcmp(f.id, e.id) != 0) continue;
        const fs::path final_path = fs::path(work) / f.name;
        const uint64_t done_size = fs::file_size(final_path, ec);
        if (!ec && done_size == f.size) { d.done += done_size; d.reused += done_size; continue; }   // verified in an earlier attempt
        const char* mirror = lb::env_get("HF_BASE");   // VELTRA_HF_BASE: test mirror; the pinned checksums still apply
        const std::string hf_base = mirror && *mirror ? mirror : "https://huggingface.co";
        if (!fetch_pinned(hf_base + "/" + e.hf_repo + "/resolve/" + rev + "/" + f.name, final_path, f.size, f.sha256, f.name, err))
            return false;
    }
    d.verified = true;
    if (d.cancel) { *err = "cancelled"; return false; }
    set_phase("convert");
    // leave one hardware thread for the UI
    if (!convert_hf_qwen3(work, dst, e.hf_repo, std::max(1, (int)std::thread::hardware_concurrency() - 1), &d.conv, err)) return false;
    fs::remove_all(work, ec);
    return true;
}

}  // namespace

std::vector<LocalModel> scan_models(const std::string& dir) {
    std::vector<LocalModel> out;
    std::error_code ec;
    if (dir.empty() || !fs::is_directory(dir, ec)) return out;
    for (const auto& de : fs::directory_iterator(dir, fs::directory_options::skip_permission_denied, ec)) {
        const std::string name = de.path().filename().string();
        if (!has_model_ext(name)) continue;
        std::error_code e2;
        const uint64_t sz = fs::file_size(de.path(), e2);   // follows symlinks (.vlt -> .lbt)
        if (e2) continue;
        out.push_back({name, de.path().string(), sz});
    }
    std::sort(out.begin(), out.end(), [](const LocalModel& a, const LocalModel& b) { return a.file < b.file; });
    return out;
}

std::string catalog_json(const HwInfo& hw, const std::string& models_dir, const std::string& loaded_file, bool cuda_built, int ctx,
                         const std::string& bench_history_json, bool engine_busy) {
    const std::vector<LocalModel> local = scan_models(models_dir);
    // marketplace order: the models Veltra runs (kCatalog), then the ones it can only show
    std::vector<const market::ModelMeta*> metas;
    for (const CatalogEntry& e : kCatalog) metas.push_back(market::find(e.id));
    size_t n_meta = 0;
    const market::ModelMeta* every = market::all(&n_meta);
    for (size_t i = 0; i < n_meta; i++)
        if (std::none_of(metas.begin(), metas.end(), [&](const market::ModelMeta* m) { return m == &every[i]; })) metas.push_back(&every[i]);
    const market::Speed speed = market::speeds_for(metas, bench_history_json, engine_busy);
    size_t meta_at = 0;
    std::ostringstream o;
    o << "{\"models_dir\":" << jq(models_dir) << ",\"reference\":\"Ryzen 9 3900 (Zen 2), DDR4 ~29 GB/s, 8 threads, decode ctx 0\",\"bandwidth_gb_s\":" << jn(speed.bandwidth_gbs, 1) << ",\"calibrated_from\":" << speed.calibrated_from << ",\"context\":" << ctx << ",\"models\":[";
    bool first = true;
    auto local_of = [&](const std::string& f) -> const LocalModel* {
        for (const LocalModel& m : local) if (m.file == f) return &m;
        return nullptr;
    };
    for (const CatalogEntry& e : kCatalog) {
        const LocalModel* lm = local_of(e.file);
        const double size = lm ? lm->bytes / kGiB : e.size_gib;
        const double kv = kv_gib(e.id, ctx);
        Fit fit = fit_for(hw, size + kv, cuda_built);
        const bool loaded = !loaded_file.empty() && loaded_file == e.file;
        const std::string url = entry_url(e);
        if (!first) o << ',';
        first = false;
        o << "{\"id\":" << jq(e.id) << ",\"name\":" << jq(e.name) << ",\"file\":" << jq(e.file) << ",\"params\":" << jq(e.params)
          << ",\"quant\":" << jq(e.quant) << ",\"size_gib\":" << jn(size) << ",\"kv_gib\":" << jn(kv) << ",\"ram_needed_gib\":" << jn(size + kv + 2.0)
          << ",\"vram_needed_gib\":" << jn(size + kv + 1.2) << ",\"ref_tps\":" << jn(e.ref_tps, 1)
          << ",\"ref_tps_kind\":" << jq(e.ref_measured ? "measured" : "estimate") << ",\"hf_source\":" << jq(e.hf_source)
          << ",\"source_url\":" << jq(std::string("https://huggingface.co/") + e.hf_source)
          << ",\"download_url\":" << (url.empty() ? std::string("null") : jq(url)) << ",\"local\":" << (lm ? "true" : "false")
          << ",\"installable\":" << (e.hf_repo[0] || !url.empty() ? "true" : "false") << ",\"converts\":" << (e.hf_repo[0] ? "true" : "false")
          << ",\"download_gib\":" << (e.hf_repo[0] ? jn(e.hf_bytes / kGiB) : e.url_bytes ? jn(e.url_bytes / kGiB) : std::string("null"))
          << ",\"prebuilt\":" << (e.url_bytes ? "true" : "false")
          << ",\"loaded\":" << (loaded ? "true" : "false") << ",\"status\":" << jq(loaded ? "LOADED" : fit.status)
          << ",\"color\":" << jq(loaded ? "green" : fit.color) << ",\"reason\":" << jq(loaded ? std::string("draait nu") : fit.reason)
          << ",\"runs_on\":" << jq(fit.runs_on) << ",\"notes\":" << jq(e.notes) << ",\"supported\":true"
          << market::meta_fields(*metas[meta_at], speed, meta_at) << "}";
        meta_at++;
    }
    for (; meta_at < metas.size(); meta_at++) {
        const market::ModelMeta& m = *metas[meta_at];
        const double size = market::file_estimate_gib(m);
        const Fit fit = fit_for(hw, size, cuda_built);
        o << ",{\"id\":" << jq(m.id) << ",\"name\":" << jq(m.name) << ",\"file\":\"\",\"size_gib\":" << jn(size) << ",\"size_estimated\":true,\"kv_gib\":0"
          << ",\"ram_needed_gib\":" << jn(size + 2.0) << ",\"vram_needed_gib\":" << jn(size + 1.2) << ",\"local\":false,\"loaded\":false,\"installable\":false"
          << ",\"supported\":false,\"status\":" << jq(fit.status) << ",\"color\":" << jq(fit.color)
          << ",\"reason\":\"Veltra kan dit model nog niet draaien\",\"source_url\":" << jq(std::string("https://huggingface.co/") + m.source)
          << market::meta_fields(m, speed, meta_at) << "}";
    }
    o << "],\"local\":[";
    for (size_t i = 0; i < local.size(); i++) {
        const LocalModel& m = local[i];
        bool in_cat = false;
        for (const CatalogEntry& e : kCatalog) in_cat |= m.file == e.file;
        const Fit fit = fit_for(hw, m.bytes / kGiB, cuda_built);
        if (i) o << ',';
        o << "{\"file\":" << jq(m.file) << ",\"size_gib\":" << jn(m.bytes / kGiB) << ",\"in_catalog\":" << (in_cat ? "true" : "false")
          << ",\"loaded\":" << (m.file == loaded_file ? "true" : "false") << ",\"status\":" << jq(m.file == loaded_file ? "LOADED" : fit.status)
          << ",\"color\":" << jq(m.file == loaded_file ? "green" : fit.color) << "}";
    }
    o << "]}";
    return o.str();
}

bool start_download(const std::string& id, const std::string& url_in, const std::string& file_in, const std::string& models_dir,
                    std::string* err) {
    std::string url = url_in, file = file_in;
    uint64_t expect = 0;
    const CatalogEntry* hf = nullptr;
    const CatalogEntry* pre = nullptr;   // ready-made .vlt with pinned size and checksum
    if (!id.empty()) {
        const CatalogEntry* e = nullptr;
        for (const CatalogEntry& c : kCatalog) if (id == c.id) e = &c;
        if (!e) { *err = "unknown model id '" + id + "'"; return false; }
        url = entry_url(*e);
        file = e->file;
        expect = (uint64_t)(e->size_gib * kGiB);
        if (e->hf_repo[0]) { hf = e; url = std::string("https://huggingface.co/") + e->hf_repo; expect = e->hf_bytes; }
        else if (e->url_bytes) { pre = e; expect = e->url_bytes; }
        if (url.empty()) {
            *err = "no download URL for " + std::string(e->file) + ": this .vlt is not published; set VELTRA_MODEL_BASE_URL or place the file in " + models_dir;
            return false;
        }
    }
    if (!safe_url(url)) { *err = "url must be a plain http(s) URL"; return false; }
    if (!safe_file_name(file)) { *err = "file must be a plain name ending in .vlt or .lbt"; return false; }
    std::error_code ec;
    fs::create_directories(models_dir, ec);
    const std::string dst = (fs::path(models_dir) / file).string();
    if (fs::exists(dst, ec)) { *err = file + " already exists in " + models_dir; return false; }
    const std::string work = (fs::path(models_dir) / (std::string(".hf-") + (hf ? hf->id : ""))).string();
    if (hf) {
        uint64_t have = 0;
        if (fs::is_directory(work, ec))
            for (const auto& de : fs::directory_iterator(work, ec)) { std::error_code e2; const auto z = de.file_size(e2); if (!e2) have += z; }
        const double need = (hf->hf_bytes > have ? hf->hf_bytes - have : 0) / kGiB + hf->size_gib + 0.5;
        const auto sp = fs::space(models_dir, ec);
        if (!ec && sp.available / kGiB < need) {
            *err = "te weinig schijfruimte: ongeveer " + jn(need, 1) + " GiB nodig, " + jn(sp.available / kGiB, 1) + " GiB vrij";
            return false;
        }
    }
    if (pre) {
        uint64_t have = fs::file_size(dst + ".part", ec);
        if (ec) have = 0;
        const double need = (pre->url_bytes > have ? pre->url_bytes - have : 0) / kGiB + 0.5;
        const auto sp = fs::space(models_dir, ec);
        if (!ec && sp.available / kGiB < need) {
            *err = "te weinig schijfruimte: ongeveer " + jn(need, 1) + " GiB nodig, " + jn(sp.available / kGiB, 1) + " GiB vrij";
            return false;
        }
    }

    Dl& d = dl();
    bool expected = false;
    if (!d.running.compare_exchange_strong(expected, true)) { *err = "another download is running"; return false; }
    {
        std::lock_guard<std::mutex> lk(d.mu);
        d.id = id; d.file = file; d.url = url; d.part = dst + ".part"; d.error.clear(); d.state = "running";
        d.total = hf || pre ? expect : 0; d.done = 0; d.t0 = std::chrono::steady_clock::now();
        d.phase = "download"; d.cancel = false; d.conv.cancel = false; d.conv.done = 0; d.conv.total = 0;
        d.check_done = 0; d.check_total = 0; d.verified = false; d.reused = 0;
    }
    std::thread([dst, work, hf, pre]() {
        Dl& d = dl();
        std::string url, part, e;
        { std::lock_guard<std::mutex> lk(d.mu); url = d.url; part = d.part; }
        std::error_code ec;
        bool ok = false;
        if (hf) {
            try { ok = install_from_hf(*hf, work, dst, &e); }
            catch (const std::exception& ex) { e = ex.what(); }
        } else if (pre) {
            try { ok = fetch_pinned(url, dst, pre->url_bytes, pre->url_sha256, pre->file, &e); }
            catch (const std::exception& ex) { e = ex.what(); }
            if (ok) d.verified = true;
        } else {
            fs::remove(part, ec);   // no checksum for a custom URL: never continue an old partial file
            ok = fetch_to_file(url, part, 0, &e);
            if (ok) {
                fs::rename(part, dst, ec);
                if (ec) { ok = false; e = "rename failed: " + ec.message(); }
            }
            if (!ok) fs::remove(part, ec);
        }
        {
            std::lock_guard<std::mutex> lk(d.mu);
            d.state = ok ? "done" : "error";
            d.error = e;
        }
        d.running = false;
    }).detach();
    return true;
}

void cancel_download() {
    Dl& d = dl();
    if (d.running) { d.cancel = true; d.conv.cancel = true; }
}

std::string download_status_json() {
    Dl& d = dl();
    std::lock_guard<std::mutex> lk(d.mu);
    const bool converting = d.phase == "convert", checking = d.phase == "verify";
    const uint64_t done = converting ? d.conv.done.load() : checking ? d.check_done.load() : d.done.load();
    const uint64_t total = converting ? d.conv.total.load() : checking ? d.check_total.load() : d.total;
    const double secs = d.state == "idle" ? 0.0 : std::chrono::duration<double>(std::chrono::steady_clock::now() - d.t0).count();
    std::ostringstream o;
    o << "{\"state\":" << jq(d.state) << ",\"id\":" << jq(d.id) << ",\"file\":" << jq(d.file) << ",\"bytes_done\":" << done
      << ",\"bytes_total\":" << total << ",\"phase\":" << jq(d.phase) << ",\"sha256_ok\":" << (d.verified.load() ? "true" : "false") << ",\"percent\":"
      << (total ? jn(std::min(100.0, 100.0 * (double)done / (double)total), 1) : std::string("null"))
      << ",\"mb_per_s\":" << jn(secs > 0.5 && !converting && !checking && d.done.load() > d.reused.load() ? (d.done.load() - d.reused.load()) / 1e6 / secs : 0.0, 1) << ",\"error\":" << (d.error.empty() ? std::string("null") : jq(d.error))
      << "}";
    return o.str();
}

std::string resolve_model_path(const std::string& name_or_path, const std::string& models_dir, std::string* err) {
    std::error_code ec;
    if (safe_file_name(name_or_path)) {
        for (const LocalModel& m : scan_models(models_dir))
            if (m.file == name_or_path) return m.path;
        *err = name_or_path + " is not in " + models_dir;
        return {};
    }
    const fs::path p(name_or_path);
    if (!p.is_absolute() || !has_model_ext(p.filename().string())) {
        *err = "give a file name from the models folder or an absolute path to a .vlt / .lbt file";
        return {};
    }
    if (!fs::is_regular_file(p, ec)) { *err = "file not found: " + name_or_path; return {}; }
    return p.string();
}

}  // namespace veltra
