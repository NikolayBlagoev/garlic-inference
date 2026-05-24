#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Byte-Pair Encoding tokenizer compatible with Qwen3 / HuggingFace tokenizer.json.
// Uses GPT-2-style byte-level encoding and a tiktoken-style pre-tokenizer.
struct BPETokenizer {
    // Load from a HuggingFace checkpoint directory.
    // Reads tokenizer.json (vocab + merges + special tokens) and
    // tokenizer_config.json (bos/eos/pad overrides) if present.
    static BPETokenizer load(const std::string& hf_dir);

    // Encode UTF-8 text to token IDs.
    // add_special=true wraps with <|im_start|> / <|im_end|> when present.
    std::vector<uint32_t> encode(const std::string& text,
                                bool add_special = false) const;

    // Decode token IDs to UTF-8 text; special tokens are skipped.
    std::string decode(const std::vector<uint32_t>& ids) const;

    // Return the raw vocabulary string for a single token ID.
    std::string decode_token(uint32_t id) const;

    uint32_t bos_id()      const { return bos_id_; }
    uint32_t eos_id()      const { return eos_id_; }
    uint32_t pad_id()      const { return pad_id_; }
    uint32_t im_start_id() const { return im_start_id_; }
    uint32_t im_end_id()   const { return im_end_id_; }
    uint32_t vocab_size()  const { return (uint32_t)id_to_token_.size(); }

private:
    std::unordered_map<std::string, int32_t>  token_to_id_;
    std::vector<std::string>                  id_to_token_;
    // "left right" -> merge rank; lower rank is applied first.
    std::unordered_map<std::string, int32_t>  merge_rank_;

    uint32_t bos_id_      = -1;
    uint32_t eos_id_      = -1;
    uint32_t pad_id_      = -1;
    uint32_t im_start_id_ = -1;
    uint32_t im_end_id_   = -1;

    // GPT-2 byte <-> unicode-character tables.
    std::array<std::string, 256>             byte_enc_;  // byte -> UTF-8 str
    std::unordered_map<std::string, uint8_t> byte_dec_;  // UTF-8 str -> byte

    // Split text into pre-tokens (mimics Qwen3/tiktoken regex, ASCII-precise;
    // non-ASCII codepoints are treated as letter-class).
    std::vector<std::string> pre_tokenize(const std::string& text) const;

    // Byte-encode one pre-token and apply BPE merges; returns token IDs.
    std::vector<uint32_t> bpe_encode(const std::string& word) const;
};
