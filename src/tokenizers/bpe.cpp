#include "bpe.h"
#include "../external/json.hpp"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

using json = nlohmann::json;

// ── UTF-8 helpers ─────────────────────────────────────────────────────────────

static std::string cp_to_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += (char)cp;
    } else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    } else {
        s += (char)(0xF0 | (cp >> 18));
        s += (char)(0x80 | ((cp >> 12) & 0x3F));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
    return s;
}

// Advance i by one UTF-8 codepoint; return the byte span as a string.
static std::string utf8_next_char(const std::string& s, size_t& i) {
    unsigned char c = (unsigned char)s[i];
    size_t len;
    if      (c < 0x80) len = 1;
    else if (c < 0xE0) len = 2;
    else if (c < 0xF0) len = 3;
    else               len = 4;
    len = std::min(len, s.size() - i);
    std::string ch = s.substr(i, len);
    i += len;
    return ch;
}

// ── GPT-2 byte encoder ────────────────────────────────────────────────────────
// Maps each byte (0–255) to a unique printable unicode char (UTF-8 encoded).
// Printable Latin bytes (33–126, 161–172, 174–255) map to themselves.
// Control / non-printable bytes map to code points U+0100, U+0101, … in order.
static std::array<std::string, 256> build_byte_encoder() {
    std::array<std::string, 256> enc;
    for (int b = 33;  b <= 126; ++b) enc[b] = cp_to_utf8(b);
    for (int b = 161; b <= 172; ++b) enc[b] = cp_to_utf8(b);
    for (int b = 174; b <= 255; ++b) enc[b] = cp_to_utf8(b);
    int n = 0;
    for (int b = 0; b < 256; ++b)
        if (enc[b].empty()) enc[b] = cp_to_utf8(0x100 + n++);
    return enc;
}

// ── BPETokenizer::load ────────────────────────────────────────────────────────

BPETokenizer BPETokenizer::load(const std::string& hf_dir) {
    BPETokenizer tok;
    tok.byte_enc_ = build_byte_encoder();
    for (int b = 0; b < 256; ++b)
        tok.byte_dec_[tok.byte_enc_[b]] = (uint8_t)b;

    // ── tokenizer.json ───────────────────────────────────────────────────────
    std::string path = hf_dir + "/tokenizer.json";
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open: " + path);
    json tj;
    f >> tj;

    // Vocabulary
    const auto& vocab = tj.at("model").at("vocab");
    size_t max_id = 0;
    for (auto& [ts, id_val] : vocab.items())
        max_id = std::max(max_id, (size_t)id_val.get<uint32_t>());
    tok.id_to_token_.resize(max_id + 1);
    for (auto& [ts, id_val] : vocab.items()) {
        uint32_t id = id_val.get<uint32_t>();
        tok.token_to_id_[ts] = id;
        tok.id_to_token_[id] = ts;
    }

    // Added / special tokens (may extend id_to_token_ beyond vocab range)
    if (tj.contains("added_tokens")) {
        for (auto& at : tj["added_tokens"]) {
            std::string ts = at["content"].get<std::string>();
            uint32_t     id = at["id"].get<uint32_t>();
            if ((size_t)id >= tok.id_to_token_.size())
                tok.id_to_token_.resize(id + 1);
            tok.id_to_token_[id] = ts;
            tok.token_to_id_[ts] = id;
        }
    }

    // Merges: stored as "left right" strings; index is the merge rank.
    const auto& merges = tj.at("model").at("merges");
    tok.merge_rank_.reserve(merges.size());
    for (size_t i = 0; i < merges.size(); ++i) {
        std::string ms;
        if (merges[i].is_string()) {
            ms = merges[i].get<std::string>();
        } else if (merges[i].is_array() && merges[i].size() == 2) {
            ms = merges[i][0].get<std::string>() + " " + merges[i][1].get<std::string>();
        }
        if (!ms.empty())
            tok.merge_rank_[ms] = (uint32_t)i;
    }

    // Special token IDs from vocabulary
    auto find_id = [&](const std::string& s) -> uint32_t {
        auto it = tok.token_to_id_.find(s);
        return it != tok.token_to_id_.end() ? it->second : -1;
    };
    tok.eos_id_      = find_id("<|endoftext|>");
    tok.bos_id_      = find_id("<|im_start|>");
    tok.pad_id_      = find_id("<|endoftext|>");
    tok.im_start_id_ = find_id("<|im_start|>");
    tok.im_end_id_   = find_id("<|im_end|>");

    // ── tokenizer_config.json (optional bos/eos/pad overrides) ───────────────
    std::ifstream cfg_f(hf_dir + "/tokenizer_config.json");
    if (cfg_f) {
        json cfg;
        cfg_f >> cfg;
        auto read_special = [&](const std::string& key) -> uint32_t {
            if (!cfg.contains(key)) return -1;
            auto& v = cfg[key];
            if (v.is_string())                       return find_id(v.get<std::string>());
            if (v.is_object() && v.contains("content"))
                return find_id(v["content"].get<std::string>());
            return -1;
        };
        uint32_t bos = read_special("bos_token");
        uint32_t eos = read_special("eos_token");
        uint32_t pad = read_special("pad_token");
        if (bos != -1) tok.bos_id_ = bos;
        if (eos != -1) tok.eos_id_ = eos;
        if (pad != -1) tok.pad_id_ = pad;
    }

    return tok;
}

// ── Pre-tokenizer ─────────────────────────────────────────────────────────────
// Approximates the Qwen3 / tiktoken GPT-4 regex:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   | [^\r\n\p{L}\p{N}]?\p{L}+
//   | \p{N}
//   | ?[^\s\p{L}\p{N}]+[\r\n]*
//   | \s*[\r\n]+
//   | \s+(?!\S) | \s+
//
// \p{L} is approximated as [a-zA-Z] plus any byte >= 0x80 (UTF-8 continuation
// and lead bytes treated as letter-class). \p{N} is [0-9].
// Full Unicode property support requires PCRE2 or ICU.

std::vector<std::string> BPETokenizer::pre_tokenize(const std::string& text) const {
    std::vector<std::string> result;
    const size_t n = text.size();
    size_t i = 0;

    auto is_letter  = [](unsigned char c) { return std::isalpha(c) || c >= 0x80; };
    auto is_digit   = [](unsigned char c) { return std::isdigit(c) != 0; };
    auto is_space   = [](unsigned char c) { return std::isspace(c) != 0; };
    auto is_newline = [](unsigned char c) { return c == '\n' || c == '\r'; };
    auto is_word    = [&](unsigned char c) { return is_letter(c) || is_digit(c); };

    while (i < n) {
        unsigned char c = (unsigned char)text[i];
        size_t start = i;

        // ── Contractions: 's 't 're 've 'm 'll 'd ────────────────────────────
        if (c == '\'') {
            static const char* ctrs[] = { "'re", "'ve", "'ll", "'s", "'t", "'m", "'d" };
            bool matched = false;
            for (const char* ctr : ctrs) {
                size_t len = std::strlen(ctr);
                if (n - i >= len) {
                    bool ok = true;
                    for (size_t k = 0; k < len && ok; ++k)
                        ok = std::tolower((unsigned char)text[i + k]) == (unsigned char)ctr[k];
                    if (ok) {
                        result.push_back(text.substr(i, len));
                        i += len;
                        matched = true;
                        break;
                    }
                }
            }
            if (matched) continue;
        }

        // ── Whitespace runs containing newlines: \s*[\r\n]+ ──────────────────
        if (is_newline(c)) {
            while (i < n && is_space((unsigned char)text[i])) ++i;
            result.push_back(text.substr(start, i - start));
            continue;
        }

        // Consume optional single leading space (kept attached to next token).
        bool had_space = false;
        if (c == ' ' && i + 1 < n && !is_newline((unsigned char)text[i + 1])) {
            had_space = true;
            ++i;
            if (i >= n) { result.push_back(" "); break; }
            c = (unsigned char)text[i];
        }

        // ── Letter run: [^\r\n\p{L}\p{N}]?\p{L}+ ────────────────────────────
        if (is_letter(c)) {
            while (i < n && is_letter((unsigned char)text[i])) ++i;
            result.push_back(text.substr(start, i - start));
            continue;
        }

        // ── Single digit ──────────────────────────────────────────────────────
        if (is_digit(c)) {
            if (had_space) {
                result.push_back(" ");        // space doesn't attach to digits
                result.push_back(text.substr(i, 1));
            } else {
                result.push_back(text.substr(start, 1));
            }
            ++i;
            continue;
        }

        // ── Punctuation run: [ ?][^\s\p{L}\p{N}]+[\r\n]* ─────────────────────
        if (!is_space(c) && !is_word(c)) {
            while (i < n && !is_space((unsigned char)text[i]) &&
                   !is_word((unsigned char)text[i]) &&
                   !is_newline((unsigned char)text[i])) ++i;
            while (i < n && is_newline((unsigned char)text[i])) ++i;
            result.push_back(text.substr(start, i - start));
            continue;
        }

        // ── Trailing / isolated space ─────────────────────────────────────────
        if (had_space) {
            result.push_back(" ");
            // i is already past the consumed space; next char re-enters the loop.
            continue;
        }

        // ── Pure whitespace run ───────────────────────────────────────────────
        if (is_space(c)) {
            while (i < n && is_space((unsigned char)text[i]) &&
                   !is_newline((unsigned char)text[i])) ++i;
            result.push_back(text.substr(start, i - start));
            continue;
        }

        // Fallback: emit single byte.
        result.push_back(text.substr(i, 1));
        ++i;
    }

    return result;
}

// ── BPETokenizer::bpe_encode ──────────────────────────────────────────────────
// Converts each byte of `word` to its unicode symbol, then greedily applies
// the highest-priority (lowest-rank) merge until no more merges are possible.

std::vector<uint32_t> BPETokenizer::bpe_encode(const std::string& word) const {
    if (word.empty()) return {};

    // Initialise symbol list: one entry per byte.
    std::vector<std::string> syms;
    syms.reserve(word.size());
    for (unsigned char b : word)
        syms.push_back(byte_enc_[b]);

    // Merge loop: O(n²) but n is the number of bytes in a single pre-token
    // (typically < 30), so this is fast in practice.
    while (syms.size() > 1) {
        int best_rank = INT_MAX;
        int best_idx  = -1;
        for (int k = 0; k + 1 < (int)syms.size(); ++k) {
            auto it = merge_rank_.find(syms[k] + " " + syms[k + 1]);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx  = k;
            }
        }
        if (best_idx == -1) break;
        syms[best_idx] += syms[best_idx + 1];
        syms.erase(syms.begin() + best_idx + 1);
    }

    // Map symbols to token IDs.
    std::vector<uint32_t> ids;
    ids.reserve(syms.size());
    for (const auto& sym : syms) {
        auto it = token_to_id_.find(sym);
        if (it != token_to_id_.end()) {
            ids.push_back(it->second);
        } else {
            // Byte fallback: emit each unicode character individually.
            size_t pos = 0;
            while (pos < sym.size()) {
                std::string ch = utf8_next_char(sym, pos);
                auto cit = token_to_id_.find(ch);
                if (cit != token_to_id_.end())
                    ids.push_back(cit->second);
            }
        }
    }
    return ids;
}

// ── BPETokenizer::encode ──────────────────────────────────────────────────────

std::vector<uint32_t> BPETokenizer::encode(const std::string& text,
                                           bool add_special) const {
    std::vector<uint32_t> ids;
    if (add_special && im_start_id_ != -1)
        ids.push_back(im_start_id_);

    for (const auto& word : pre_tokenize(text)) {
        auto word_ids = bpe_encode(word);
        ids.insert(ids.end(), word_ids.begin(), word_ids.end());
    }

    if (add_special && im_end_id_ != -1)
        ids.push_back(im_end_id_);
    return ids;
}

// ── BPETokenizer::decode ──────────────────────────────────────────────────────

std::string BPETokenizer::decode(const std::vector<uint32_t>& ids) const {
    std::vector<uint8_t> bytes;
    bytes.reserve(ids.size() * 3);

    for (uint32_t id : ids) {
        if (id < 0 || (size_t)id >= id_to_token_.size()) continue;
        const std::string& tok_str = id_to_token_[id];

        // Attempt to byte-decode every unicode char in this token.
        std::vector<uint8_t> tok_bytes;
        bool all_byte = true;
        size_t pos = 0;
        while (pos < tok_str.size() && all_byte) {
            std::string ch = utf8_next_char(tok_str, pos);
            auto it = byte_dec_.find(ch);
            if (it != byte_dec_.end()) tok_bytes.push_back(it->second);
            else all_byte = false;
        }

        if (all_byte)
            bytes.insert(bytes.end(), tok_bytes.begin(), tok_bytes.end());
        // else: special token — skip in output
    }

    return std::string(bytes.begin(), bytes.end());
}

// ── BPETokenizer::decode_token ────────────────────────────────────────────────

std::string BPETokenizer::decode_token(uint32_t id) const {
    if (id < 0 || (size_t)id >= id_to_token_.size()) return "";
    const std::string& tok_str = id_to_token_[id];

    // Try byte-decode; if any char is not in the byte table (e.g. special token)
    // return the raw vocabulary string instead.
    std::string result;
    size_t pos = 0;
    while (pos < tok_str.size()) {
        std::string ch = utf8_next_char(tok_str, pos);
        auto it = byte_dec_.find(ch);
        if (it == byte_dec_.end()) return tok_str;   // special token
        result += (char)it->second;
    }
    return result;
}
