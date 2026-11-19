#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace damascus {

struct Config {
    std::string architecture;

    uint32_t vocab_size;
    uint32_t block_count;
    uint32_t embedding_length;

    uint32_t attention_head_count;
    uint32_t attention_head_count_kv;

    uint32_t feed_forward_length;

    float rms_norm_epsilon;
    float rope_freq_base;

    uint32_t context_length;
};

struct TensorInfo {
    std::string name;
    std::vector<int64_t> dims;
    uint32_t type;
    uint64_t offset; // relative to tensor blob start
};

// Raw tokenizer payload as it sits in the GGUF metadata. Compartment 4 turns
// this into a working byte-level BPE tokenizer; we only *extract* it here so
// the GGUF stays the single source of truth (no separate vocab file).
struct TokenizerData {
    std::string model;                  // ggml tokenizer model, e.g. "gpt2"
    std::string pre;                    // pre-tokenizer id, e.g. "llama-bpe"

    std::vector<std::string> tokens;    // id -> token piece (byte-level encoded)
    std::vector<int32_t> token_types;   // id -> ggml token type (1 normal, 3 control)
    std::vector<std::string> merges;    // rank-ordered "A B" BPE merge rules

    uint32_t bos_token_id = 0;
    uint32_t eos_token_id = 0;
};

class MappedFile {
public:
    MappedFile() = default;
    explicit MappedFile(const std::string& path);
    const std::byte* data() const { return ptr_.get(); }
    std::size_t size() const { return size_; }

private:
    std::shared_ptr<const std::byte> ptr_;
    std::size_t size_ = 0;
};

struct GGUFModel {
    Config config;
    TokenizerData tokenizer;
    std::vector<TensorInfo> tensors;

    // mmap of the whole GGUF file; TensorInfo::offset is relative to
    // data_blob_start, not file start.
    MappedFile data;
    uint64_t data_blob_start;
};

GGUFModel load_gguf(const std::string& path);
GGUFModel load_gguf(MappedFile file_buffer);

} // namespace damascus