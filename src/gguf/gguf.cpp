#include "gguf.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <variant>

namespace damascus {

struct GGMLTypeTraits {
    uint64_t block_size;
    uint64_t type_size;
};

struct GGUFValue {
    using Array = std::vector<GGUFValue>;

    std::variant<
        uint8_t,
        int8_t,
        uint16_t,
        int16_t,
        uint32_t,
        int32_t,
        uint64_t,
        int64_t,
        float,
        double,
        bool,
        std::string,
        Array
    > value;
};

enum GGUFType : uint32_t {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

uint64_t align_up(uint64_t x, uint64_t alignment) {
    if (alignment == 0) {
        throw std::runtime_error("Error: alignment cannot be zero");
    }

    return ((x + alignment - 1) / alignment) * alignment;
}

template <typename T>
T read_primitive(const std::vector<std::byte>& buffer, size_t& cursor) {
    if (cursor + sizeof(T) > buffer.size()) {
        throw std::runtime_error("Error: attempted to read past end of file buffer");
    }

    T value;
    std::memcpy(&value, &buffer[cursor], sizeof(T));
    cursor += sizeof(T);

    return value;
}

std::string read_string(const std::vector<std::byte>& buffer, size_t& cursor) {
    uint64_t length = read_primitive<uint64_t>(buffer, cursor);

    if (cursor + length > buffer.size()) {
        throw std::runtime_error("Error: string length exceeds file buffer");
    }

    std::string value(
        reinterpret_cast<const char*>(&buffer[cursor]),
        static_cast<size_t>(length)
    );

    cursor += static_cast<size_t>(length);

    return value;
}

GGUFValue read_value(const std::vector<std::byte>& buffer,
                     size_t& cursor,
                     uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8: {
            return GGUFValue{read_primitive<uint8_t>(buffer, cursor)};
        }

        case GGUF_TYPE_INT8: {
            return GGUFValue{read_primitive<int8_t>(buffer, cursor)};
        }

        case GGUF_TYPE_UINT16: {
            return GGUFValue{read_primitive<uint16_t>(buffer, cursor)};
        }

        case GGUF_TYPE_INT16: {
            return GGUFValue{read_primitive<int16_t>(buffer, cursor)};
        }

        case GGUF_TYPE_UINT32: {
            return GGUFValue{read_primitive<uint32_t>(buffer, cursor)};
        }

        case GGUF_TYPE_INT32: {
            return GGUFValue{read_primitive<int32_t>(buffer, cursor)};
        }

        case GGUF_TYPE_FLOAT32: {
            return GGUFValue{read_primitive<float>(buffer, cursor)};
        }

        case GGUF_TYPE_BOOL: {
            uint8_t value = read_primitive<uint8_t>(buffer, cursor);
            return GGUFValue{value != 0};
        }

        case GGUF_TYPE_STRING: {
            return GGUFValue{read_string(buffer, cursor)};
        }

        case GGUF_TYPE_ARRAY: {
            uint32_t element_type = read_primitive<uint32_t>(buffer, cursor);
            uint64_t count = read_primitive<uint64_t>(buffer, cursor);

            GGUFValue::Array elements;
            elements.reserve(static_cast<size_t>(count));

            for (uint64_t i = 0; i < count; ++i) {
                elements.push_back(read_value(buffer, cursor, element_type));
            }

            return GGUFValue{std::move(elements)};
        }

        case GGUF_TYPE_UINT64: {
            return GGUFValue{read_primitive<uint64_t>(buffer, cursor)};
        }

        case GGUF_TYPE_INT64: {
            return GGUFValue{read_primitive<int64_t>(buffer, cursor)};
        }

        case GGUF_TYPE_FLOAT64: {
            return GGUFValue{read_primitive<double>(buffer, cursor)};
        }

        default: {
            throw std::runtime_error(
                "Error: unknown GGUF value type tag: " + std::to_string(type)
            );
        }
    }
}

template <typename T>
T get_metadata(const std::unordered_map<std::string, GGUFValue>& metadata,
               const std::string& key) {
    auto it = metadata.find(key);

    if (it == metadata.end()) {
        throw std::runtime_error("Error: missing required metadata field: " + key);
    }

    try {
        return std::get<T>(it->second.value);
    } catch (const std::bad_variant_access&) {
        throw std::runtime_error("Error: metadata field has wrong type: " + key);
    }
}

uint32_t read_vocab_size(const std::unordered_map<std::string, GGUFValue>& metadata) {
    auto explicit_vocab = metadata.find("llama.vocab_size");

    if (explicit_vocab != metadata.end()) {
        try {
            return std::get<uint32_t>(explicit_vocab->second.value);
        } catch (const std::bad_variant_access&) {
            throw std::runtime_error("Error: llama.vocab_size has wrong type");
        }
    }

    auto tokens_it = metadata.find("tokenizer.ggml.tokens");

    if (tokens_it == metadata.end()) {
        throw std::runtime_error(
            "Error: missing llama.vocab_size and tokenizer.ggml.tokens"
        );
    }

    try {
        const auto& tokens = std::get<GGUFValue::Array>(tokens_it->second.value);
        return static_cast<uint32_t>(tokens.size());
    } catch (const std::bad_variant_access&) {
        throw std::runtime_error("Error: tokenizer.ggml.tokens has wrong type");
    }
}

// Pull a homogeneous array out of the metadata map, projecting each element
// through `project`. Used for the tokenizer token/merge/type arrays.
template <typename Out, typename Elem>
std::vector<Out> get_metadata_array(
        const std::unordered_map<std::string, GGUFValue>& metadata,
        const std::string& key,
        Out (*project)(const Elem&)) {
    auto it = metadata.find(key);

    if (it == metadata.end()) {
        throw std::runtime_error("Error: missing required metadata array: " + key);
    }

    const auto* arr = std::get_if<GGUFValue::Array>(&it->second.value);

    if (arr == nullptr) {
        throw std::runtime_error("Error: metadata field is not an array: " + key);
    }

    std::vector<Out> out;
    out.reserve(arr->size());

    for (const GGUFValue& element : *arr) {
        const auto* typed = std::get_if<Elem>(&element.value);

        if (typed == nullptr) {
            throw std::runtime_error("Error: array element has wrong type in: " + key);
        }

        out.push_back(project(*typed));
    }

    return out;
}

TokenizerData load_tokenizer(const std::unordered_map<std::string, GGUFValue>& metadata) {
    TokenizerData tok;

    tok.model = get_metadata<std::string>(metadata, "tokenizer.ggml.model");
    tok.pre   = get_metadata<std::string>(metadata, "tokenizer.ggml.pre");

    tok.tokens = get_metadata_array<std::string, std::string>(
        metadata, "tokenizer.ggml.tokens",
        [](const std::string& s) { return s; });

    tok.token_types = get_metadata_array<int32_t, int32_t>(
        metadata, "tokenizer.ggml.token_type",
        [](const int32_t& v) { return v; });

    tok.merges = get_metadata_array<std::string, std::string>(
        metadata, "tokenizer.ggml.merges",
        [](const std::string& s) { return s; });

    if (tok.tokens.size() != tok.token_types.size()) {
        throw std::runtime_error(
            "Error: tokenizer tokens / token_type length mismatch");
    }

    tok.bos_token_id = get_metadata<uint32_t>(metadata, "tokenizer.ggml.bos_token_id");
    tok.eos_token_id = get_metadata<uint32_t>(metadata, "tokenizer.ggml.eos_token_id");

    return tok;
}

Config load_config(const std::unordered_map<std::string, GGUFValue>& metadata) {
    Config cfg;

    cfg.architecture = get_metadata<std::string>(
        metadata,
        "general.architecture"
    );

    cfg.vocab_size = read_vocab_size(metadata);

    cfg.block_count = get_metadata<uint32_t>(
        metadata,
        "llama.block_count"
    );

    cfg.embedding_length = get_metadata<uint32_t>(
        metadata,
        "llama.embedding_length"
    );

    cfg.attention_head_count = get_metadata<uint32_t>(
        metadata,
        "llama.attention.head_count"
    );

    cfg.attention_head_count_kv = get_metadata<uint32_t>(
        metadata,
        "llama.attention.head_count_kv"
    );

    cfg.feed_forward_length = get_metadata<uint32_t>(
        metadata,
        "llama.feed_forward_length"
    );

    cfg.rms_norm_epsilon = get_metadata<float>(
        metadata,
        "llama.attention.layer_norm_rms_epsilon"
    );

    cfg.rope_freq_base = get_metadata<float>(
        metadata,
        "llama.rope.freq_base"
    );

    cfg.context_length = get_metadata<uint32_t>(
        metadata,
        "llama.context_length"
    );

    return cfg;
}

GGMLTypeTraits get_type_traits(uint32_t type) {
    switch (type) {
        case 0:
            return {1, 4};

        case 12:
            return {256, 144};

        case 14:
            return {256, 210};

        default:
            throw std::runtime_error(
                "Error: unsupported ggml_type: " + std::to_string(type)
            );
    }
}

uint64_t byte_size(uint32_t type, uint64_t n) {
    GGMLTypeTraits traits = get_type_traits(type);

    if (n % traits.block_size != 0) {
        throw std::runtime_error(
            "Error: tensor element count is not a multiple of block size"
        );
    }

    return (n / traits.block_size) * traits.type_size;
}

void validate_tensor_tiling(const std::vector<TensorInfo>& tensors,
                            uint64_t data_blob_start,
                            uint64_t file_size) {
    if (tensors.empty()) {
        throw std::runtime_error("Error: no tensors found");
    }

    std::vector<TensorInfo> sorted_tensors = tensors;

    std::sort(
        sorted_tensors.begin(),
        sorted_tensors.end(),
        [](const TensorInfo& a, const TensorInfo& b) {
            return a.offset < b.offset;
        }
    );

    for (size_t i = 0; i + 1 < sorted_tensors.size(); ++i) {
        const TensorInfo& current = sorted_tensors[i];
        const TensorInfo& next = sorted_tensors[i + 1];

        uint64_t n_elements = 1;

        for (int64_t dim : current.dims) {
            n_elements *= static_cast<uint64_t>(dim);
        }

        uint64_t size = byte_size(current.type, n_elements);
        uint64_t computed_end = current.offset + size;

        if (computed_end > next.offset) {
            throw std::runtime_error(
                "Error: tensor overlap detected after " + current.name
            );
        }
    }

    const TensorInfo& last = sorted_tensors.back();

    uint64_t last_n = 1;

    for (int64_t dim : last.dims) {
        last_n *= static_cast<uint64_t>(dim);
    }

    uint64_t last_size = byte_size(last.type, last_n);
    uint64_t computed_file_end = data_blob_start + last.offset + last_size;

    if (computed_file_end > file_size) {
        throw std::runtime_error(
            "Error: last tensor exceeds file size: " + last.name
        );
    }
}

GGUFModel load_gguf(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);

    if (!file) {
        throw std::runtime_error("Error: could not open GGUF file: " + path);
    }

    std::streamsize file_size_signed = file.tellg();

    if (file_size_signed < 0) {
        throw std::runtime_error("Error: could not determine GGUF file size");
    }

    uint64_t file_size = static_cast<uint64_t>(file_size_signed);

    file.seekg(0, std::ios::beg);

    std::vector<std::byte> file_buffer(static_cast<size_t>(file_size));

    file.read(
        reinterpret_cast<char*>(file_buffer.data()),
        static_cast<std::streamsize>(file_buffer.size())
    );

    if (!file) {
        throw std::runtime_error("Error: could not read GGUF file into memory");
    }

    size_t cursor = 0;

    uint32_t magic = read_primitive<uint32_t>(file_buffer, cursor);
    uint32_t version = read_primitive<uint32_t>(file_buffer, cursor);
    uint64_t tensor_count = read_primitive<uint64_t>(file_buffer, cursor);
    uint64_t metadata_kv_count = read_primitive<uint64_t>(file_buffer, cursor);

    char magic_str[5] = {0};
    std::memcpy(magic_str, &magic, 4);

    if (std::string(magic_str) != "GGUF") {
        throw std::runtime_error("Error: file is not a GGUF file");
    }

    if (version != 3) {
        throw std::runtime_error(
            "Error: unsupported GGUF version: " + std::to_string(version)
        );
    }

    std::unordered_map<std::string, GGUFValue> metadata;
    metadata.reserve(static_cast<size_t>(metadata_kv_count));

    for (uint64_t i = 0; i < metadata_kv_count; ++i) {
        std::string key = read_string(file_buffer, cursor);
        uint32_t value_type = read_primitive<uint32_t>(file_buffer, cursor);

        GGUFValue value = read_value(file_buffer, cursor, value_type);

        metadata.emplace(std::move(key), std::move(value));
    }

    Config config = load_config(metadata);
    TokenizerData tokenizer = load_tokenizer(metadata);

    std::vector<TensorInfo> tensors;
    tensors.reserve(static_cast<size_t>(tensor_count));

    for (uint64_t i = 0; i < tensor_count; ++i) {
        TensorInfo tensor;

        tensor.name = read_string(file_buffer, cursor);

        uint32_t n_dims = read_primitive<uint32_t>(file_buffer, cursor);
        tensor.dims.reserve(n_dims);

        for (uint32_t d = 0; d < n_dims; ++d) {
            int64_t dim = read_primitive<int64_t>(file_buffer, cursor);
            tensor.dims.push_back(dim);
        }

        tensor.type = read_primitive<uint32_t>(file_buffer, cursor);
        tensor.offset = read_primitive<uint64_t>(file_buffer, cursor);

        tensors.push_back(std::move(tensor));
    }

    uint64_t alignment = 32;

    auto it = metadata.find("general.alignment");

    if (it != metadata.end()) {
        const uint32_t* value = std::get_if<uint32_t>(&it->second.value);

        if (value == nullptr) {
            throw std::runtime_error("Error: general.alignment has wrong type");
        }

        alignment = *value;
    }

    uint64_t data_blob_start = align_up(static_cast<uint64_t>(cursor), alignment);

    validate_tensor_tiling(tensors, data_blob_start, file_size);

    GGUFModel model;
    model.config = std::move(config);
    model.tokenizer = std::move(tokenizer);
    model.tensors = std::move(tensors);
    model.data = std::move(file_buffer);
    model.data_blob_start = data_blob_start;

    return model;
}

} // namespace damascus