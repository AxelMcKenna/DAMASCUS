#pragma once

// Compartment 4 — Llama 3 tokenizer (GPT-2 byte-level BPE, "llama-bpe" pre-tok).
//
// Pipeline (encode):
//   text ──pretokenize──▶ chunks ──byte_encode──▶ byte-char strings
//        ──BPE merges──▶ pieces ──vocab lookup──▶ token ids
//
// Correctness target (Compartment 4 acceptance): exact round-trip vs HF, and
// matching the oracle token ids for a fixed prompt. This is pure C++ with no
// Metal dependency, so it runs under the Docker/Linux escape used for the
// rest of the CPU-verified path.

#include "gguf.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace damascus {

class Tokenizer {
public:
    // Build from the raw GGUF tokenizer payload (Compartment 3 extracts it).
    // Throws if the tokenizer model isn't a byte-level BPE we understand.
    explicit Tokenizer(const TokenizerData& data);

    // Encode text to token ids. `add_bos` prepends the begin-of-text token.
    std::vector<int32_t> encode(const std::string& text, bool add_bos = false) const;

    // Decode ids back to text. Inverse of encode for the byte-level mapping.
    std::string decode(const std::vector<int32_t>& ids) const;

    uint32_t bos_id() const { return bos_id_; }
    uint32_t eos_id() const { return eos_id_; }
    std::size_t vocab_size() const { return id_to_token_.size(); }

private:
    // Greedy BPE over one pre-tokenized chunk already mapped to byte-chars.
    std::vector<std::string> bpe(const std::string& chunk) const;

    // Split raw UTF-8 text into pre-token chunks (llama-3 regex, see .cpp).
    std::vector<std::string> pretokenize(const std::string& text) const;

    std::unordered_map<std::string, int32_t> token_to_id_;
    std::vector<std::string> id_to_token_;

    // merge rule (a,b) -> rank; lower rank merges first.
    std::unordered_map<std::string, int32_t> merge_rank_;

    // byte (0..255) <-> printable unicode codepoint, GPT-2 convention.
    std::string byte_to_unicode_[256];
    std::unordered_map<std::string, uint8_t> unicode_to_byte_;

    uint32_t bos_id_ = 0;
    uint32_t eos_id_ = 0;
};

}  // namespace damascus
