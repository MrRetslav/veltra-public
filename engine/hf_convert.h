// engine/hf_convert.h - converts an official dense Qwen3 checkpoint (Hugging Face safetensors) into a .vlt on this machine:
// every attention / MLP matrix becomes Q4_K (llama.cpp's 4.6-bit format, run by engine/q2k_gemv.h), embeddings (and an
// untied head) become per-row INT8. Plain symmetric INT4 over 128 columns is NOT used: without calibration it turns the
// output of dense Qwen3 into garbage (measured on 1.7B, caused by the MLP matrices of the early layers).
// Only data is read: safetensors is a JSON header plus raw tensors, nothing in the download is ever executed.
#pragma once
#include <atomic>
#include <cstdint>
#include <string>

namespace veltra {

struct ConvertProgress {
    std::atomic<uint64_t> done{0}, total{0};   // weights converted / to convert
    std::atomic<bool> cancel{false};
};

// src_dir: config.json, tokenizer.json, tokenizer_config.json, generation_config.json (optional) and *.safetensors.
// Writes out_path + ".part" and renames it to out_path on success. threads <= 0: all hardware threads.
bool convert_hf_qwen3(const std::string& src_dir, const std::string& out_path, const std::string& model_name, int threads,
                      ConvertProgress* progress, std::string* err);

}  // namespace veltra
