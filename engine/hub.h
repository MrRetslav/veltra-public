// engine/hub.h - hardware detection (engine/hw_detect.cpp) and the model hub: built-in .vlt catalog with a per-machine
// fit status, local model scan and a background downloader (engine/model_hub.cpp). Used by veltra-server / veltra.exe.
// No hard GPU dependency: dxgi.dll / nvml.dll / libnvidia-ml.so.1 are loaded at run time when present.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace veltra {

struct GpuInfo {
    std::string name;
    std::string vendor;          // "NVIDIA", "AMD", "Intel", ...
    unsigned vendor_id = 0;      // PCI vendor id (0x10DE NVIDIA, 0x1002 AMD, 0x8086 Intel), 0 = unknown
    double vram_total_gib = 0;   // dedicated video memory
    double vram_free_gib = 0;    // DXGI budget - usage (Windows) or NVML free; equals total when unknown
    bool free_known = false;
    int cc_major = -1, cc_minor = -1;   // CUDA compute capability (NVML), -1 = unknown / not NVIDIA
    int load_pct = -1;                  // GPU utilisation 0..100 (NVML), -1 = unknown
    std::string source;          // "dxgi", "nvml", "dxgi+nvml"
};

struct HwInfo {
    std::string os;
    std::string cpu_brand;
    std::string cpu_arch = "unknown";   // heuristic: "zen1".."zen5", "intel-core", "arm64", "unknown"
    int cores_physical = 0, cores_logical = 0;
    bool avx2 = false, fma = false, avx512f = false, avx512_vnni = false, avx_vnni = false, neon = false;
    double ram_total_gib = 0, ram_free_gib = 0;
    std::vector<GpuInfo> gpus;
};

HwInfo detect_hardware();
// Coarse GPU class from name + PCI vendor id: "rtx50", "rtx40", "rtx3060", "rtx30", "rtx20", "gtx", "radeon", "arc",
// "igpu" (Intel UHD/Iris or AMD integrated with < 1 GiB dedicated memory), "other".
std::string gpu_arch(const GpuInfo& g);
std::string hardware_json(const HwInfo& hw, bool cuda_backend_built);

struct LocalModel {
    std::string file;   // file name inside the models directory
    std::string path;
    uint64_t bytes = 0;
};
std::vector<LocalModel> scan_models(const std::string& dir);   // *.vlt and legacy *.lbt

// GET /api/catalog: every built-in model with its hardware status for this machine
// ("GPU READY", "CPU READY", "HYBRID / TIGHT", "TOO LARGE", "CPU UNSUPPORTED") and whether it is on disk / loaded.
std::string catalog_json(const HwInfo& hw, const std::string& models_dir, const std::string& loaded_file, bool cuda_backend_built, int ctx,
                         const std::string& bench_history_json, bool engine_busy);

// Background download of one model into models_dir (one at a time). `id` selects a catalog entry (its URL comes from
// env VELTRA_MODEL_BASE_URL + file name, or the entry's own URL); `url` + `file` download anything else (http/https only).
bool start_download(const std::string& id, const std::string& url, const std::string& file, const std::string& models_dir,
                    std::string* err);
std::string download_status_json();   // GET /api/models/download/status
void cancel_download();               // POST /api/models/download/cancel; no-op when idle

// Accepts a bare file name from scan_models(dir) or an absolute path to an existing .vlt / .lbt; returns the path or "".
std::string resolve_model_path(const std::string& name_or_path, const std::string& models_dir, std::string* err);

}  // namespace veltra
