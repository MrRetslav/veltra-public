// Built-in chat templates (no Jinja): Qwen3 / Qwen2.5 ChatML and Llama-3, mirroring the official HF
// templates for plain conversations (system / user / assistant / tool roles, no `tools=` schema).
#pragma once
#include <string>
#include <vector>

#include "engine.h"     // lb::ChatMessage
#include "tokenizer.h"

namespace lb {

// arch: "chatml" (Qwen3 Instruct-2507: plain ChatML, no default system, no <think>; engine.cpp picks it for arch qwen3 when the meta chat_template
// has no enable_thinking), "qwen3" (Qwen3 ChatML incl. <think> handling), "qwen2" (Qwen2/2.5 ChatML with the default system
// prompt), "llama" (Llama-3 header format, BOS first). Unknown archs fall back to the qwen2 format.
// enable_thinking=false makes the Qwen3 generation prompt end with the empty "<think>\n\n</think>\n\n"
// block (HF `enable_thinking=False`); true (default) is the plain HF behaviour. Ignored by other archs.
std::string render_chat_template(const std::string& arch, const std::vector<ChatMessage>& messages,
                                 bool add_generation_prompt = true, bool enable_thinking = true);

// render_chat_template() followed by tok.encode() (special tokens in the text map to single ids, the
// literal "<|begin_of_text|>" of the Llama-3 template yields the BOS id; no extra specials are added).
std::vector<int> apply_chat_template(const Tokenizer& tok, const std::string& arch,
                                     const std::vector<ChatMessage>& messages, bool add_generation_prompt = true,
                                     bool enable_thinking = true);

}  // namespace lb
