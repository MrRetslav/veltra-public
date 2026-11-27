#include "chat_template.h"

#include <cstdint>
#include <cstring>

namespace lb {
namespace {

// Python str.strip() / jinja `trim`: strips Unicode whitespace on both ends.
size_t ws_len_at(const std::string& s, size_t i) {  // length of a whitespace char starting at i, else 0
    unsigned char c = s[i];
    if (c == ' ' || (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x1F)) return 1;
    if (c == 0xC2 && i + 1 < s.size() && ((unsigned char)s[i + 1] == 0x85 || (unsigned char)s[i + 1] == 0xA0)) return 2;
    if (c == 0xE1 && i + 2 < s.size() && (unsigned char)s[i + 1] == 0x9A && (unsigned char)s[i + 2] == 0x80) return 3;
    if (c == 0xE2 && i + 2 < s.size()) {
        unsigned char b = s[i + 1], d = s[i + 2];
        if (b == 0x80 && ((d >= 0x80 && d <= 0x8A) || d == 0xA8 || d == 0xA9 || d == 0xAF)) return 3;
        if (b == 0x81 && d == 0x9F) return 3;
    }
    if (c == 0xE3 && i + 2 < s.size() && (unsigned char)s[i + 1] == 0x80 && (unsigned char)s[i + 2] == 0x80) return 3;
    return 0;
}

std::string py_strip(const std::string& s) {
    size_t b = 0;
    while (b < s.size()) {
        size_t l = ws_len_at(s, b);
        if (!l) break;
        b += l;
    }
    size_t e = s.size();
    while (e > b) {
        size_t st = e - 1;  // start of the last UTF-8 char
        while (st > b && ((unsigned char)s[st] & 0xC0) == 0x80) --st;
        size_t l = ws_len_at(s, st);
        if (!l || st + l != e) break;
        e = st;
    }
    return s.substr(b, e - b);
}

std::string strip_nl(const std::string& s, bool left, bool right) {
    size_t b = 0, e = s.size();
    if (left) while (b < e && s[b] == '\n') ++b;
    if (right) while (e > b && s[e - 1] == '\n') --e;
    return s.substr(b, e - b);
}

bool starts_with(const std::string& s, const char* p) { return s.compare(0, strlen(p), p) == 0; }
bool ends_with(const std::string& s, const char* p) {
    size_t n = strlen(p);
    return s.size() >= n && s.compare(s.size() - n, n, p) == 0;
}

std::string render_qwen3(const std::vector<ChatMessage>& msgs, bool gen, bool thinking) {
    std::string out;
    const long n = static_cast<long>(msgs.size());
    if (n > 0 && msgs[0].role == "system") out += "<|im_start|>system\n" + msgs[0].content + "<|im_end|>\n";
    long last_query = n - 1;
    for (long i = n - 1; i >= 0; --i) {
        const ChatMessage& m = msgs[i];
        if (m.role == "user" && !(starts_with(m.content, "<tool_response>") && ends_with(m.content, "</tool_response>"))) {
            last_query = i;
            break;
        }
    }
    for (long i = 0; i < n; ++i) {
        const ChatMessage& m = msgs[i];
        std::string content = m.content;
        if (m.role == "user" || (m.role == "system" && i != 0)) {
            out += "<|im_start|>" + m.role + "\n" + content + "<|im_end|>\n";
        } else if (m.role == "assistant") {
            std::string reasoning;
            size_t tp = content.find("</think>");
            if (tp != std::string::npos) {
                // reasoning = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
                std::string head = strip_nl(content.substr(0, tp), false, true);
                size_t ot = head.rfind("<think>");
                if (ot != std::string::npos) head = head.substr(ot + 7);
                reasoning = strip_nl(head, true, false);
                // content = content.split('</think>')[-1].lstrip('\n')
                size_t lt = content.rfind("</think>");
                content = strip_nl(content.substr(lt + 8), true, false);
            }
            if (i > last_query && (i == n - 1 || !reasoning.empty())) {
                out += "<|im_start|>" + m.role + "\n<think>\n" + strip_nl(reasoning, true, true) + "\n</think>\n\n" +
                       strip_nl(content, true, false);
            } else {
                out += "<|im_start|>" + m.role + "\n" + content;
            }
            out += "<|im_end|>\n";
        } else if (m.role == "tool") {
            if (i == 0 || msgs[i - 1].role != "tool") out += "<|im_start|>user";
            out += "\n<tool_response>\n" + content + "\n</tool_response>";
            if (i == n - 1 || msgs[i + 1].role != "tool") out += "<|im_end|>\n";
        }
    }
    if (gen) {
        out += "<|im_start|>assistant\n";
        if (!thinking) out += "<think>\n\n</think>\n\n";
    }
    return out;
}

std::string render_qwen2(const std::vector<ChatMessage>& msgs, bool gen) {
    std::string out;
    if (!msgs.empty() && msgs[0].role == "system")
        out += "<|im_start|>system\n" + msgs[0].content + "<|im_end|>\n";
    else
        out += "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n";
    const size_t n = msgs.size();
    for (size_t i = 0; i < n; ++i) {
        const ChatMessage& m = msgs[i];
        if (m.role == "user" || m.role == "assistant" || (m.role == "system" && i != 0)) {
            out += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
        } else if (m.role == "tool") {
            if (i == 0 || msgs[i - 1].role != "tool") out += "<|im_start|>user";
            out += "\n<tool_response>\n" + m.content + "\n</tool_response>";
            if (i == n - 1 || msgs[i + 1].role != "tool") out += "<|im_end|>\n";
        }
    }
    if (gen) out += "<|im_start|>assistant\n";
    return out;
}

// Qwen3 "Instruct-2507" template (also the tiny test models): plain ChatML, NO default system prompt, NO <think> handling.
std::string render_chatml(const std::vector<ChatMessage>& msgs, bool gen) {
    std::string out;
    const size_t n = msgs.size();
    for (size_t i = 0; i < n; ++i) {
        const ChatMessage& m = msgs[i];
        if (m.role == "user" || m.role == "assistant" || m.role == "system") {
            out += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
        } else if (m.role == "tool") {
            if (i == 0 || msgs[i - 1].role != "tool") out += "<|im_start|>user";
            out += "\n<tool_response>\n" + m.content + "\n</tool_response>";
            if (i == n - 1 || msgs[i + 1].role != "tool") out += "<|im_end|>\n";
        }
    }
    if (gen) out += "<|im_start|>assistant\n";
    return out;
}

// GraniteForCausalLM chat template (jinja) without tools: a system turn is always emitted (empty without a system message), history
// assistant turns lose their reasoning, no BOS. enable_thinking=false prefills a closed, empty think block ("<think></think>").
std::string render_granite(const std::vector<ChatMessage>& msgs, bool gen, bool thinking) {
    const size_t first = !msgs.empty() && msgs[0].role == "system" ? 1 : 0;
    std::string out = "<|im_start|>system\n" + (first ? msgs[0].content : std::string()) + "<|im_end|>\n";
    long last_user = -1;
    for (size_t i = first; i < msgs.size(); ++i)
        if (msgs[i].role == "user") last_user = static_cast<long>(i);
    for (size_t i = first; i < msgs.size(); ++i) {
        const ChatMessage& m = msgs[i];
        if (m.role == "assistant") {
            std::string c = m.content;
            if (c.find("<think>") == std::string::npos && c.find("</think>") == std::string::npos) c = "<think></think>" + c;
            if (static_cast<long>(i) < last_user && c.find("<think>") != std::string::npos && c.find("</think>") != std::string::npos)
                c = "<think></think>" + c.substr(c.rfind("</think>") + 8);
            out += "<|im_start|>assistant\n" + py_strip(c) + "<|im_end|>\n";
        } else if (m.role == "tool") {
            if (i > first && msgs[i - 1].role != "tool") out += "<|im_start|>user\n";
            out += "<tool_response>\n" + m.content + "\n</tool_response>\n";
            if (i + 1 == msgs.size() || msgs[i + 1].role != "tool") out += "<|im_end|>\n";
        } else {
            out += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
        }
    }
    if (gen) out += thinking ? "<|im_start|>assistant\n<think>\n" : "<|im_start|>assistant\n<think></think>";
    return out;
}

std::string render_llama3(const std::vector<ChatMessage>& msgs, bool gen) {
    std::string out;
    for (size_t i = 0; i < msgs.size(); ++i) {
        if (i == 0) out += "<|begin_of_text|>";
        out += "<|start_header_id|>" + msgs[i].role + "<|end_header_id|>\n\n" + py_strip(msgs[i].content) + "<|eot_id|>";
    }
    if (gen) out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
    return out;
}

}  // namespace

std::string render_chat_template(const std::string& arch, const std::vector<ChatMessage>& messages,
                                 bool add_generation_prompt, bool enable_thinking) {
    if (arch == "chatml") return render_chatml(messages, add_generation_prompt);
    if (arch == "qwen3") return render_qwen3(messages, add_generation_prompt, enable_thinking);
    // Qwen3.5 / Qwen3.6: same ChatML, but the generation prompt of the thinking mode already opens the think block (assistant, newline, <think>, newline)
    if (arch == "qwen35") return render_qwen3(messages, add_generation_prompt, enable_thinking) + (add_generation_prompt && enable_thinking ? "<think>\n" : "");
    if (arch == "granite") return render_granite(messages, add_generation_prompt, enable_thinking);
    if (arch == "llama") return render_llama3(messages, add_generation_prompt);
    return render_qwen2(messages, add_generation_prompt);
}

std::vector<int> apply_chat_template(const Tokenizer& tok, const std::string& arch, const std::vector<ChatMessage>& messages,
                                     bool add_generation_prompt, bool enable_thinking) {
    return tok.encode(render_chat_template(arch, messages, add_generation_prompt, enable_thinking), false);
}

}  // namespace lb
