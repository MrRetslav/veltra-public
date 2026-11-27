// engine/hw_detect.cpp - GET /api/hardware: CPU brand / cores / ISA, RAM, GPUs + VRAM. See hub.h.
// Windows: __cpuid, GetLogicalProcessorInformationEx, GlobalMemoryStatusEx, dxgi.dll (CreateDXGIFactory1 -> GetDesc1,
//          IDXGIAdapter3::QueryVideoMemoryInfo) and nvml.dll for the CUDA compute capability - both via LoadLibraryA.
// Linux:   cpuid, /proc/cpuinfo, /proc/meminfo, libnvidia-ml.so.1 via dlopen.
#include "hub.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <utility>

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
#  include <windows.h>
#  include <dxgi1_4.h>
#else
#  include <dlfcn.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#  define VELTRA_X86 1
#  if defined(_MSC_VER)
#    include <intrin.h>
#  else
#    include <cpuid.h>
#  endif
#endif

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
std::string jn(double v) { char b[32]; std::snprintf(b, sizeof b, "%.2f", v); return b; }
std::string trim(std::string s) {
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\n' || s.back() == '\r' || s.back() == '\0')) s.pop_back();
    size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) i++;
    return s.substr(i);
}

// ---------------------------------------------------------------- CPU (cpuid)
#if VELTRA_X86
void cpuid(unsigned leaf, unsigned sub, unsigned r[4]) {
#  if defined(_MSC_VER)
    int a[4];
    __cpuidex(a, (int)leaf, (int)sub);
    for (int i = 0; i < 4; i++) r[i] = (unsigned)a[i];
#  else
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#  endif
}
unsigned long long xgetbv0() {
#  if defined(_MSC_VER)
    return _xgetbv(0);
#  else
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long)hi << 32) | lo;
#  endif
}
void detect_cpu_x86(HwInfo& hw) {
    unsigned r[4];
    cpuid(0x80000000u, 0, r);
    if (r[0] >= 0x80000004u) {
        char brand[49] = {0};
        for (unsigned i = 0; i < 3; i++) {
            cpuid(0x80000002u + i, 0, r);
            std::memcpy(brand + 16 * i, r, 16);
        }
        hw.cpu_brand = trim(brand);
    }
    cpuid(0, 0, r);
    const unsigned max_leaf = r[0];
    char vend[13] = {0};
    std::memcpy(vend, &r[1], 4); std::memcpy(vend + 4, &r[3], 4); std::memcpy(vend + 8, &r[2], 4);
    cpuid(1, 0, r);
    {   // family / model -> coarse micro-architecture
        const unsigned base_fam = (r[0] >> 8) & 0xF, base_model = (r[0] >> 4) & 0xF;
        const unsigned fam = base_fam == 0xF ? base_fam + ((r[0] >> 20) & 0xFF) : base_fam;
        const unsigned model = (base_fam == 0xF || base_fam == 0x6) ? (((r[0] >> 16) & 0xF) << 4) | base_model : base_model;
        if (std::strcmp(vend, "AuthenticAMD") == 0) {
            if (fam == 0x17) hw.cpu_arch = model >= 0x30 ? "zen2" : "zen1";
            else if (fam == 0x19) hw.cpu_arch = ((model >= 0x10 && model <= 0x1F) || (model >= 0x60 && model <= 0x7F) ||
                                                 (model >= 0xA0 && model <= 0xAF)) ? "zen4" : "zen3";
            else if (fam == 0x1A) hw.cpu_arch = "zen5";
        } else if (std::strcmp(vend, "GenuineIntel") == 0 && fam == 6) {
            hw.cpu_arch = "intel-core";
        }
    }
    const bool osxsave = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1;
    const unsigned long long xcr0 = osxsave ? xgetbv0() : 0;
    const bool os_avx = avx && (xcr0 & 0x6) == 0x6, os_avx512 = os_avx && (xcr0 & 0xE6) == 0xE6;
    hw.fma = os_avx && ((r[2] >> 12) & 1);
    if (max_leaf >= 7) {
        cpuid(7, 0, r);
        hw.avx2 = os_avx && ((r[1] >> 5) & 1);
        hw.avx512f = os_avx512 && ((r[1] >> 16) & 1);
        hw.avx512_vnni = os_avx512 && ((r[2] >> 11) & 1);
        cpuid(7, 1, r);
        hw.avx_vnni = os_avx && ((r[0] >> 4) & 1);
    }
}
#endif

// ---------------------------------------------------------------- NVML (optional, run-time loaded)
struct NvmlGpu { std::string name; double total = 0, free = 0; int ccM = -1, ccm = -1, load = -1; };
std::vector<NvmlGpu> query_nvml() {
    std::vector<NvmlGpu> out;
    using Fn0 = int (*)();
    using FnCount = int (*)(unsigned*);
    using FnHandle = int (*)(unsigned, void**);
    using FnName = int (*)(void*, char*, unsigned);
    struct Mem { unsigned long long total, free, used; };
    using FnMem = int (*)(void*, Mem*);
    using FnCc = int (*)(void*, int*, int*);
    struct Util { unsigned gpu, memory; };
    using FnUtil = int (*)(void*, Util*);
#if defined(_WIN32)
    HMODULE lib = LoadLibraryA("nvml.dll");
    if (!lib) lib = LoadLibraryA("C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
    if (!lib) return out;
    auto sym = [&](const char* n) { return (void*)GetProcAddress(lib, n); };
#else
    void* lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) return out;
    auto sym = [&](const char* n) { return dlsym(lib, n); };
#endif
    auto init = (Fn0)sym("nvmlInit_v2");
    auto shut = (Fn0)sym("nvmlShutdown");
    auto count = (FnCount)sym("nvmlDeviceGetCount_v2");
    auto handle = (FnHandle)sym("nvmlDeviceGetHandleByIndex_v2");
    auto name = (FnName)sym("nvmlDeviceGetName");
    auto mem = (FnMem)sym("nvmlDeviceGetMemoryInfo");
    auto cc = (FnCc)sym("nvmlDeviceGetCudaComputeCapability");
    auto util = (FnUtil)sym("nvmlDeviceGetUtilizationRates");
    if (init && shut && count && handle && name && mem && init() == 0) {
        unsigned n = 0;
        if (count(&n) == 0) {
            for (unsigned i = 0; i < n && i < 16; i++) {
                void* d = nullptr;
                if (handle(i, &d) != 0) continue;
                NvmlGpu g;
                char nb[96] = {0};
                if (name(d, nb, sizeof nb) == 0) g.name = nb;
                Mem m{};
                if (mem(d, &m) == 0) { g.total = m.total / kGiB; g.free = m.free / kGiB; }
                if (cc) cc(d, &g.ccM, &g.ccm);
                Util u{};
                if (util && util(d, &u) == 0 && u.gpu <= 100) g.load = (int)u.gpu;
                out.push_back(g);
            }
        }
        shut();
    }
#if defined(_WIN32)
    FreeLibrary(lib);
#else
    dlclose(lib);
#endif
    return out;
}

std::string vendor_of(unsigned id) {
    switch (id) {
        case 0x10DE: return "NVIDIA";
        case 0x1002: case 0x1022: return "AMD";
        case 0x8086: return "Intel";
        case 0x5143: return "Qualcomm";
        default: return "other";
    }
}

#if defined(_WIN32)
// ---------------------------------------------------------------- Windows
std::string utf8(const wchar_t* w) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1) return {};
    std::string s((size_t)n - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}
void detect_cores_win(HwInfo& hw) {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) return;
    std::vector<unsigned char> buf(len);
    auto* p = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)buf.data();
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, p, &len)) return;
    int phys = 0, logical = 0;
    for (DWORD off = 0; off < len;) {
        auto* e = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*)(buf.data() + off);
        if (e->Relationship == RelationProcessorCore) {
            phys++;
            for (WORD g = 0; g < e->Processor.GroupCount; g++) {
                unsigned long long mask = (unsigned long long)e->Processor.GroupMask[g].Mask;
                while (mask) { logical += (int)(mask & 1); mask >>= 1; }
            }
        }
        off += e->Size;
    }
    hw.cores_physical = phys;
    hw.cores_logical = logical;
}
void detect_ram_win(HwInfo& hw) {
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) { hw.ram_total_gib = ms.ullTotalPhys / kGiB; hw.ram_free_gib = ms.ullAvailPhys / kGiB; }
}
// IIDs spelled out so no dxguid import library is needed.
const GUID kIID_IDXGIFactory1 = {0x770aae78, 0xf26f, 0x4dba, {0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87}};
const GUID kIID_IDXGIAdapter3 = {0x645967a4, 0x1392, 0x4310, {0xa7, 0x98, 0x80, 0x53, 0xce, 0x3e, 0x93, 0xfd}};
void detect_gpus_win(HwInfo& hw) {
    HMODULE dxgi = LoadLibraryA("dxgi.dll");
    if (!dxgi) return;
    using CreateFn = HRESULT(WINAPI*)(REFIID, void**);
    auto create = (CreateFn)(void*)GetProcAddress(dxgi, "CreateDXGIFactory1");
    IDXGIFactory1* fac = nullptr;
    if (create && SUCCEEDED(create(kIID_IDXGIFactory1, (void**)&fac)) && fac) {
        IDXGIAdapter1* ad = nullptr;
        for (UINT i = 0; fac->EnumAdapters1(i, &ad) != DXGI_ERROR_NOT_FOUND && i < 16; i++) {
            DXGI_ADAPTER_DESC1 d;
            if (SUCCEEDED(ad->GetDesc1(&d)) && !(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && d.VendorId != 0x1414) {
                GpuInfo g;
                g.name = utf8(d.Description);
                g.vendor = vendor_of(d.VendorId);
                g.vendor_id = d.VendorId;
                g.vram_total_gib = d.DedicatedVideoMemory / kGiB;
                g.vram_free_gib = g.vram_total_gib;
                g.source = "dxgi";
                IDXGIAdapter3* a3 = nullptr;
                if (SUCCEEDED(ad->QueryInterface(kIID_IDXGIAdapter3, (void**)&a3)) && a3) {
                    DXGI_QUERY_VIDEO_MEMORY_INFO mi;
                    if (SUCCEEDED(a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mi)) && mi.Budget > 0) {
                        const double used = mi.CurrentUsage / kGiB, budget = mi.Budget / kGiB;
                        // clamp to dedicated memory: an integrated GPU's budget is shared system RAM and must not count as VRAM
                        g.vram_free_gib = std::min(budget > used ? budget - used : 0.0, g.vram_total_gib);
                        g.free_known = true;
                    }
                    a3->Release();
                }
                hw.gpus.push_back(g);   // every hardware adapter, integrated GPUs included (software adapters are skipped above)
            }
            ad->Release();
        }
        fac->Release();
    }
    FreeLibrary(dxgi);
}
#else
// ---------------------------------------------------------------- Linux / POSIX
void detect_cpu_proc(HwInfo& hw) {
    std::ifstream f("/proc/cpuinfo");
    std::string line, phys_id = "0";
    std::set<std::pair<std::string, std::string>> cores;
    int logical = 0;
    while (std::getline(f, line)) {
        const size_t c = line.find(':');
        if (c == std::string::npos) continue;
        const std::string k = trim(line.substr(0, c)), v = trim(line.substr(c + 1));
        if (k == "processor") logical++;
        else if (k == "physical id") phys_id = v;
        else if (k == "core id") cores.insert({phys_id, v});
        else if ((k == "model name" || k == "Hardware") && hw.cpu_brand.empty()) hw.cpu_brand = v;
        else if (k == "Features" && v.find("asimd") != std::string::npos) hw.neon = true;
    }
    hw.cores_logical = logical;
    hw.cores_physical = cores.empty() ? logical : (int)cores.size();
}
void detect_ram_proc(HwInfo& hw) {
    std::ifstream f("/proc/meminfo");
    std::string k;
    unsigned long long v = 0;
    std::string unit;
    while (f >> k >> v >> unit) {
        if (k == "MemTotal:") hw.ram_total_gib = v * 1024.0 / kGiB;
        else if (k == "MemAvailable:") hw.ram_free_gib = v * 1024.0 / kGiB;
    }
}
#endif

}  // namespace

HwInfo detect_hardware() {
    HwInfo hw;
#if defined(_WIN32)
    hw.os = "windows";
    detect_cores_win(hw);
    detect_ram_win(hw);
    detect_gpus_win(hw);
#else
    hw.os = "linux";
    detect_cpu_proc(hw);
    detect_ram_proc(hw);
#endif
#if VELTRA_X86
    detect_cpu_x86(hw);
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
    hw.neon = true;
    hw.cpu_arch = "arm64";
#endif
    // NVML: names / free memory / compute capability. Windows: merged into the DXGI NVIDIA adapters (same order);
    // Linux: the only GPU source.
    const std::vector<NvmlGpu> nv = query_nvml();
    size_t k = 0;
    for (GpuInfo& g : hw.gpus) {
        if (g.vendor != "NVIDIA" || k >= nv.size()) continue;
        const NvmlGpu& n = nv[k++];
        g.cc_major = n.ccM; g.cc_minor = n.ccm; g.load_pct = n.load;
        if (n.total > 0) { g.vram_free_gib = n.free; g.free_known = true; }
        g.source = "dxgi+nvml";
    }
    if (hw.gpus.empty()) {
        for (const NvmlGpu& n : nv) {
            GpuInfo g;
            g.name = n.name; g.vendor = "NVIDIA"; g.vendor_id = 0x10DE; g.vram_total_gib = n.total; g.vram_free_gib = n.free;
            g.free_known = n.total > 0; g.cc_major = n.ccM; g.cc_minor = n.ccm; g.source = "nvml"; g.load_pct = n.load;
            hw.gpus.push_back(g);
        }
    }
    return hw;
}

namespace {
std::string lower(std::string s) {
    for (char& c : s) if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    return s;
}
bool has(const std::string& s, const char* sub) { return s.find(sub) != std::string::npos; }
// First run of digits after `key` ("rtx 3060" / "rtx3060" -> 3060), or -1.
int number_after(const std::string& s, const char* key) {
    size_t p = s.find(key);
    if (p == std::string::npos) return -1;
    p += std::strlen(key);
    while (p < s.size() && (s[p] == ' ' || s[p] == '-')) p++;
    int v = 0, n = 0;
    while (p < s.size() && s[p] >= '0' && s[p] <= '9' && n < 6) { v = v * 10 + (s[p] - '0'); p++; n++; }
    return n ? v : -1;
}
}  // namespace

std::string gpu_arch(const GpuInfo& g) {
    const std::string n = lower(g.name);
    const unsigned vid = g.vendor_id ? g.vendor_id : g.vendor == "NVIDIA" ? 0x10DE : g.vendor == "AMD" ? 0x1002 : g.vendor == "Intel" ? 0x8086 : 0;
    const bool small = g.vram_total_gib < 1.0;
    if (vid == 0x10DE || has(n, "nvidia") || has(n, "geforce") || has(n, "rtx") || has(n, "gtx")) {
        const int rtx = number_after(n, "rtx");
        if (rtx >= 0) {
            if (rtx == 3060) return "rtx3060";
            if (rtx >= 5000 && rtx < 6000) return "rtx50";
            if (rtx >= 4000 && rtx < 5000) return "rtx40";
            if (rtx >= 3000 && rtx < 4000) return "rtx30";
            if (rtx >= 2000 && rtx < 3000) return "rtx20";
            return "other";   // RTX A-series / workstation names
        }
        if (has(n, "gtx")) return "gtx";
        return "other";
    }
    if (vid == 0x8086 || has(n, "intel")) {
        if (has(n, "arc")) return "arc";
        if (has(n, "uhd") || has(n, "iris") || has(n, "hd graphics") || small) return "igpu";
        return "other";
    }
    if (vid == 0x1002 || vid == 0x1022 || has(n, "radeon") || has(n, "amd")) {
        if (small && (has(n, "graphics") || has(n, "vega") || has(n, "radeon(tm)") || !has(n, " rx "))) return "igpu";
        return "radeon";
    }
    return "other";
}

std::string hardware_json(const HwInfo& hw, bool cuda_backend_built) {
    std::ostringstream o;
    o << "{\"os\":" << jq(hw.os) << ",\"cpu\":{\"brand\":" << jq(hw.cpu_brand.empty() ? "unknown CPU" : hw.cpu_brand)
      << ",\"cores_physical\":" << hw.cores_physical << ",\"cores_logical\":" << hw.cores_logical << ",\"isa\":{\"avx2\":"
      << (hw.avx2 ? "true" : "false") << ",\"fma\":" << (hw.fma ? "true" : "false") << ",\"avx512f\":" << (hw.avx512f ? "true" : "false")
      << ",\"avx512_vnni\":" << (hw.avx512_vnni ? "true" : "false") << ",\"avx_vnni\":" << (hw.avx_vnni ? "true" : "false")
      << ",\"neon\":" << (hw.neon ? "true" : "false") << "}},\"cpu_arch\":" << jq(hw.cpu_arch) << ",\"ram\":{\"total_gib\":" << jn(hw.ram_total_gib)
      << ",\"free_gib\":" << jn(hw.ram_free_gib) << "},\"gpus\":[";
    for (size_t i = 0; i < hw.gpus.size(); i++) {
        const GpuInfo& g = hw.gpus[i];
        if (i) o << ',';
        o << "{\"index\":" << i << ",\"name\":" << jq(g.name) << ",\"vendor\":" << jq(g.vendor)
          << ",\"vendor_id\":" << g.vendor_id << ",\"gpu_arch\":" << jq(gpu_arch(g))
          << ",\"vram_total_mb\":" << (long long)(g.vram_total_gib * 1024.0 + 0.5)
          << ",\"vram_free_mb\":" << (long long)(g.vram_free_gib * 1024.0 + 0.5) << ",\"vram_total_gib\":" << jn(g.vram_total_gib)
          << ",\"vram_free_gib\":" << jn(g.vram_free_gib) << ",\"vram_free_known\":" << (g.free_known ? "true" : "false")
          << ",\"cuda_cc\":" << (g.cc_major >= 0 ? jq(std::to_string(g.cc_major) + "." + std::to_string(g.cc_minor)) : std::string("null"))
          << ",\"load_pct\":" << (g.load_pct >= 0 ? std::to_string(g.load_pct) : std::string("null"))
          << ",\"source\":" << jq(g.source) << "}";
    }
    o << "],\"backends\":{\"cpu\":" << jq(hw.avx512f ? "avx512" : hw.avx2 ? "avx2" : hw.neon ? "neon" : "scalar")
      << ",\"cuda_built\":" << (cuda_backend_built ? "true" : "false") << "}}";
    return o.str();
}

}  // namespace veltra
