// Zero-dependency tokenizer for the HF `tokenizer.json` formats we ship:
//   * byte-level BPE (Qwen2 / Qwen2.5 / Qwen3 / Llama-3 / GPT-2 regexes, hand-implemented, no regex engine)
//   * SentencePiece-style BPE (Llama-2: Prepend/Replace normalizer, byte fallback, metaspace)
// Encoding is bit-identical to HuggingFace `tokenizers` (verified by tests/verify_tokenizer.py).
// Thread-safety: encode()/decode() are const and thread-safe (the word cache is mutex-protected).
#pragma once
#include <memory>
#include <string>
#include <vector>

namespace lb {

class Detok;

class Tokenizer {
public:
    // Builds from the raw bytes of tokenizer.json (+ optional tokenizer_config.json, may be empty).
    // Returns nullptr and fills *err on failure (unsupported normalizer / pre-tokenizer / model).
    static std::unique_ptr<Tokenizer> from_json(const std::string& tokenizer_json,
                                                const std::string& tokenizer_config_json,
                                                std::string* err);
    ~Tokenizer();

    // Text -> ids. Added/special tokens occurring literally in `text` (e.g. "<|im_start|>") become single
    // ids. add_special applies the tokenizer.json post-processor template (e.g. Llama BOS). Invalid UTF-8
    // in `text` is replaced by U+FFFD first (like a lossy decode).
    std::vector<int> encode(const std::string& text, bool add_special = false) const;

    // ids -> UTF-8 text. Out-of-range ids are skipped; invalid byte sequences become U+FFFD.
    std::string decode(const std::vector<int>& ids, bool skip_special = false) const;

    int vocab_size() const;                          // max id + 1 (includes added tokens)
    int token_to_id(const std::string& token) const; // -1 when unknown. `token` is the vocab-form string
    std::string id_to_token(int id) const;           // vocab-form string ("" when invalid)
    int bos_token_id() const;                        // -1 when none (from tokenizer_config.json)
    int eos_token_id() const;
    int pad_token_id() const;
    bool is_special(int id) const;                   // id is an added token flagged "special"
    bool is_byte_level() const;

    // --- used by Detok -------------------------------------------------------------------------
    // Appends the raw decoded bytes of one token (no UTF-8 repair) to *out.
    void append_token_bytes(int id, std::string* out, bool skip_special) const;
    bool strips_leading_space() const;               // SentencePiece decoder drops one leading space

private:
    Tokenizer();
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class Detok;
};

// Streaming-safe incremental decoder: push() returns only complete UTF-8 characters and holds back an
// incomplete multi-byte tail (typical for byte-level BPE emoji / CJK split across tokens).
class Detok {
public:
    explicit Detok(const Tokenizer& tok, bool skip_special = false) : tok_(&tok), skip_special_(skip_special) {}
    std::string push(int id);
    std::string flush();     // emits any held-back bytes (lossy: invalid -> U+FFFD)
    void reset();

private:
    const Tokenizer* tok_;
    bool skip_special_;
    bool started_ = false;
    std::string pending_;
};

// UTF-8 helpers (exposed for tests / other modules)
// Replaces invalid sequences by U+FFFD (maximal-subpart policy, same as Rust's from_utf8_lossy).
std::string utf8_sanitize(const std::string& s);
bool utf8_valid(const std::string& s);

}  // namespace lb
