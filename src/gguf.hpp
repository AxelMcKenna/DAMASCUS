#pragma once

#include <cstddef>
#include <cstdint>
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

struct GGUFModel {
    Config config;
    std::vector<TensorInfo> tensors;

    // Own the whole GGUF file buffer for now.
    // TensorInfo::offset is relative to data_blob_start, not file start.
    std::vector<std::byte> data;
    uint64_t data_blob_start;
};

GGUFModel load_gguf(const std::string& path);

} // namespace damascus