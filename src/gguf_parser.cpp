#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>
#include <stdexcept>
#include <variant>
#include <unordered_map>


struct TensorInfo {
    std::string name;
    std::vector<int64_t> dims;
    uint32_t type;
    uint64_t offset;
};


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

struct Config {
    std::string architecture;

    uint32_t vocab_size;
    uint32_t block_count;
    uint32_t embedding_length;

    uint32_t attention_head_count;
    uint32_t attention_head_count_kv;

    uint32_t feed_forward_length;

    float rms_norm_elipson;
    float rope_freq_base;

    uint32_t context_length;
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

std::string type_name(uint32_t type) {
    switch (type) {
    case GGUF_TYPE_UINT8:   return "UINT8";
    case GGUF_TYPE_INT8:    return "INT8";
    case GGUF_TYPE_UINT16:  return "UINT16";
    case GGUF_TYPE_INT16:   return "INT16";
    case GGUF_TYPE_UINT32:  return "UINT32";
    case GGUF_TYPE_INT32:   return "INT32";
    case GGUF_TYPE_FLOAT32: return "FLOAT32";
    case GGUF_TYPE_BOOL:    return "BOOL";
    case GGUF_TYPE_STRING:  return "STRING";
    case GGUF_TYPE_ARRAY:   return "ARRAY";
    case GGUF_TYPE_UINT64:  return "UINT64";
    case GGUF_TYPE_INT64:   return "INT64";
    case GGUF_TYPE_FLOAT64: return "FLOAT64";
    default:                return "UNKNOWN";
    }
}


// Helper to align offsets
uint64_t align_up(uint64_t x, uint64_t alignment) {
    if (alignment == 0) {
        throw std::runtime_error("Error: Alignment cannot be zero");
    }

    return ((x + alignment - 1) / alignment) * alignment;
}

// Helper to read a primitive value out of our buffer and advance the cursor
template<typename T>
T read_primitive(const std::vector<std::byte>& buffer, size_t& cursor) {
    if (cursor + sizeof(T) > buffer.size()) {
        throw std::runtime_error("Error: Attempted to read past end of file buffer");
    }
    T value;
    std::memcpy(&value, &buffer[cursor], sizeof(T)); // Fixed: added address-of operator '&'
    cursor += sizeof(T);
    return value;
}

// Helper to read a length-prefixed string from the buffer and advance the cursor
std::string read_string(const std::vector<std::byte>& buffer, size_t& cursor) {

    uint64_t length = read_primitive<uint64_t>(buffer, cursor);

    if (cursor + length > buffer.size()) {
        throw std::runtime_error("Error: String length exceeds past end of file buffer");
    }

    std::string str(reinterpret_cast<const char*>(&buffer[cursor]), length);

    cursor += length;
    return str;
}

// Helper to read a GGUFValue 
GGUFValue read_value(const std::vector<std::byte>& buffer,
                     size_t& cursor,
                     uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8: {
            uint8_t value = read_primitive<uint8_t>(buffer, cursor);
            return GGUFValue{value};
        }

        case GGUF_TYPE_INT8: {
            int8_t value = read_primitive<int8_t>(buffer, cursor);
            return GGUFValue{value};
        }

        case GGUF_TYPE_UINT16: {
            uint16_t value = read_primitive<uint16_t>(buffer, cursor);
            return GGUFValue{value};
        }

        case GGUF_TYPE_INT16: {
            int16_t value = read_primitive<int16_t>(buffer, cursor);
            return GGUFValue{value};
        }

        case GGUF_TYPE_UINT32: {
            uint32_t value = read_primitive<uint32_t>(buffer, cursor);
            return GGUFValue{value};
        }

        case GGUF_TYPE_INT32: {
            int32_t value = read_primitive<int32_t>(buffer, cursor);
            return GGUFValue{value};
        }

        case GGUF_TYPE_FLOAT32: {
            float value = read_primitive<float>(buffer, cursor);
            return GGUFValue{value};
        }

        case GGUF_TYPE_BOOL: {
            uint8_t value = read_primitive<uint8_t>(buffer, cursor);
            return GGUFValue{value != 0};
        }

        case GGUF_TYPE_STRING: {
            std::string value = read_string(buffer, cursor);
            return GGUFValue{std::move(value)};
        }

        case GGUF_TYPE_ARRAY: {
            uint32_t element_type = read_primitive<uint32_t>(buffer, cursor);
            uint64_t count = read_primitive<uint64_t>(buffer, cursor);

            std::vector<GGUFValue> elements;
            elements.reserve(count);

            for (uint64_t i = 0; i < count; ++i) {
                elements.push_back(read_value(buffer, cursor, element_type));
            }

            return GGUFValue{std::move(elements)};
        }

        case GGUF_TYPE_UINT64: {
            uint64_t value = read_primitive<uint64_t>(buffer, cursor);
            return GGUFValue{value};
        }

        case GGUF_TYPE_INT64: {
            int64_t value = read_primitive<int64_t>(buffer, cursor);
            return GGUFValue{value};
        }

        case GGUF_TYPE_FLOAT64: {
            double value = read_primitive<double>(buffer, cursor);
            return GGUFValue{value};
        }

        default: {
            throw std::runtime_error(
                "Unknown GGUF value type tag: " + std::to_string(type)
            );
        }
    }
}


GGMLTypeTraits get_type_traits(uint32_t type) {
    switch (type) {
        case 0: return {1, 4};
        case 12: return {256, 144};
        case 14: return {256, 210};
        default:
            throw std::runtime_error("Error: type is unsupported by ggml: " + std::to_string(type));
    }
}

uint64_t byte_size(uint32_t type, uint64_t n) {
    GGMLTypeTraits traits = get_type_traits(type);

    if (n % traits.block_size != 0) {
        throw std::runtime_error("Error: Tensor element count (" + std::to_string(n) +
            ") is not a multiple of block size (" + std::to_string(traits.block_size) +
            ") for ggml_type " + std::to_string(type)
        );
    }

    return (n / traits.block_size) * traits.type_size;
}

template <typename T>
T get_metadata(const std::unordered_map<std::string, GGUFValue>& md, const std::string& key) {
    auto it = md.find(key);

    if (it == md.end()) {
        throw std::runtime_error("Error: Missing required metadata field: " + key);
    }

    try {
        return std::get<T>(it->second.value);
    } catch (const std::bad_variant_access&) {
        throw std::runtime_error("Error: Metadata field has wrong type: " + key);
    }
}


// --- Main loop ---

int main() {
    // Fixed: restored the complete, proper path to the llama-bpe vocabulary file
    const std::string file_path = "models/llama-3.2-3b-instruct.Q4_K_M.gguf";

    // Fixed: named the stream variable 'file' consistently throughout the program
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);

    if (!file) {
        std::cerr << "Error: Could not open file " << file_path << "\n";
        return 1;
    }

    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::cout << "Slurping file into memory (" << file_size << " bytes)\n";

    // Slurp entire file into a single contiguous block of memory
    std::vector<std::byte> file_buffer(file_size);

    // Overcoming the friction point with a reinterpret_cast to char*
    file.read(reinterpret_cast<char*>(file_buffer.data()), file_size);

    if (!file) {
        std::cerr << "Error, could not read file into the buffer.\n";
        return 1;
    }

    size_t cursor = 0;

    try {

        // --- Section 1: Header Parsing ---
        uint32_t magic = read_primitive<uint32_t>(file_buffer, cursor);
        uint32_t version = read_primitive<uint32_t>(file_buffer, cursor);
        uint64_t tensor_count = read_primitive<uint64_t>(file_buffer, cursor);
        uint64_t metadata_kv_count = read_primitive<uint64_t>(file_buffer, cursor);

        // Convert magic uint32_t safely into human-readable string
        char magic_str[5] = {0};
        std::memcpy(magic_str, &magic, 4);

        // Output
        std::cout << "\n--- GGUF Header Analysis ---\n";
        std::cout << "Magic Bytes:     " << magic_str << "\n";
        std::cout << "Format Version:  " << version << "\n";
        std::cout << "Tensor Count:    " << tensor_count << "\n";
        std::cout << "Metadata KVs:    " << metadata_kv_count << "\n";
        std::cout << "Current Cursor:  " << cursor << " bytes\n";
        std::cout << "------------------------------\n";

        // --- Section 2: Metadata Parsing ---
        std::cout << " --- Metadata Analysis ---";

        std::unordered_map<std::string, GGUFValue> metadata;
        metadata.reserve(metadata_kv_count);

        for (uint64_t i = 0; i < metadata_kv_count; ++i) {
            std::string key = read_string(file_buffer, cursor);
            uint32_t value_type = read_primitive<uint32_t>(file_buffer, cursor);

            GGUFValue value = read_value(file_buffer, cursor, value_type);

            metadata.emplace(std::move(key), std::move(value));
        }
        
        std::cout << "\nFinished metadata parsing.\n";
        std::cout << "Metadata values stored: " << metadata.size() << "\n";
        
        std::cout << "Cursor after metadata: " << cursor << " bytes\n";

        // --- Section 2.5: Config Load --- 
        Config cfg;

        cfg.architecture = get_metadata<std::string>(
            metadata,
            "general.architecture"
        );


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

        cfg.rms_norm_elipson = get_metadata<float>(
            metadata,
            "llama.attention.layer_norm_rms_elipson"
        );

        cfg.rope_freq_base = get_metadata<float>(
            metadata,
            "llama.rope.req_base"
        );

        cfg.context_length = get_metadata<uint32_t>(
            metadata,
            "llama.context_length"
        );

        // --- Section 3: Tensor Info Parsing ---
        std::cout <<"\n --- Tensor Info ---\n";

        std::vector<TensorInfo> tensors;
        tensors.reserve(tensor_count);

        for (uint64_t i = 0; i < tensor_count; ++i) {
            TensorInfo tensor;

            // 1. Tensor name: GGUF string
            tensor.name = read_string(file_buffer, cursor);

            // 2. Number of dimensions
            uint32_t n_dims = read_primitive<uint32_t>(file_buffer, cursor);

            // 3. Dimension sizes
            tensor.dims.reserve(n_dims);

            for (uint32_t d = 0; d < n_dims; ++d) {
                int64_t dim = read_primitive<int64_t>(file_buffer, cursor);
                tensor.dims.push_back(dim);
            }

            // 4. Tensor type: ggml type, not gguf type
            tensor.type = read_primitive<uint32_t>(file_buffer, cursor);

            // 5. Offset relative to tensor blob
            tensor.offset = read_primitive<uint64_t>(file_buffer, cursor);

            tensors.push_back(tensor);

            // Print a small summary
            std::cout << "\n[" << i << "] " << tensor.name << "\n";

            std::cout << "Dims:   [";
            for (size_t d = 0; d < tensor.dims.size(); ++d) {
                if (d > 0) {
                    std::cout << ", ";
                }
                std::cout << tensor.dims[d];
            }
            std::cout << "]\n";

            std::cout << "Type:   " << tensor.type << "  // raw ggml_type for now\n";
            std::cout << "Offset: " << tensor.offset << "\n";
            std::cout << "Cursor: " << cursor << " bytes\n";
        }

        std::cout << "\n Finished tensor info parsing \n";
        std::cout << "Tensors info read: " << tensors.size() << "\n";
        std::cout << "Cursor after tensor info: " << cursor << " bytes\n";


        // --- Section 4: Tensor Blob Parsing ---
        //
        // If general alignment is absent from metadata, GGUF uses the default alignment.
        // gguf.h defines this as 32

        uint64_t alignment = 32;

        uint64_t cursor_after_tensor_infos = cursor;
        uint64_t data_blob_start = align_up(cursor_after_tensor_infos, alignment);

        std::cout << "\n--- Tensor Data Blob ---\n";
        std::cout << "Alignment:                 " << alignment << " bytes\n";
        std::cout << "Cursor after tensor infos: " << cursor_after_tensor_infos << "\n";
        std::cout << "Data blob start:           " << data_blob_start << "\n";
        std::cout << "Padding bytes:             "
                  << (data_blob_start - cursor_after_tensor_infos) << "\n";

        if (tensors.empty()) {
            throw std::runtime_error("Error: No tensors found");
        }

        // Sort a copy by offset ascending to guarantee physical ordering
        std::vector<TensorInfo> sorted_tensors = tensors;
        std::sort(sorted_tensors.begin(), sorted_tensors.end(), [](const TensorInfo&a, const TensorInfo&b) {
            return a.offset < b.offset;
        });

        std::cout << "\n --- Full Tiling Check --- \n";

        bool has_overlaps = false;
        bool has_gaps = false;
        uint64_t total_padding_bytes = 0;


        // Check all interior boundaries
        for (size_t i = 0; i < sorted_tensors.size() - 1; ++i) {
            const TensorInfo& current = sorted_tensors[i];
            const TensorInfo& next = sorted_tensors[i + 1];

            // compute n elements per tensor
            uint64_t n_elements = 1;
            for (uint64_t dim : current.dims) {
                n_elements *= dim;
            }

            uint64_t size = byte_size(current.type, n_elements);
            uint64_t computed_end = current.offset + size;

            // byte checks
            if (computed_end > next.offset) {
                std::cout << "[FATAL] Overlap at tensor " << i << " (" << current.name << "):\n"
                          << "  Ends at: " << computed_end << " but next starts at " << next.offset << "\n";
                has_overlaps = true;
            } else if (computed_end < next.offset) {
                uint64_t gap = next.offset - computed_end;
                total_padding_bytes += gap;

                // Print the very first gap we find so you can observe the behavior
                if (!has_gaps) {
                    std::cout << "[NOTE] First gap detected after tensor " << i << " (" << current.name << "):\n"
                              << "  Ends at: " << computed_end << " | Next starts at: " << next.offset
                              << " | Gap: " << gap << " bytes\n";
                }
                has_gaps = true;
            }
        }

        // Boundary check last tensor against EOF
        const TensorInfo& last_tensor = sorted_tensors.back();

        uint64_t last_n = 1;
        for (int64_t dim : last_tensor.dims) {
            last_n *= dim;
        }

        uint64_t last_size = byte_size(last_tensor.type, last_n);
        uint64_t computed_file_end = data_blob_start + last_tensor.offset + last_size;

        std::cout << "\nLast tensor:           " << last_tensor.name << "\n";
        std::cout << "Computed file end:     " << computed_file_end << "\n";
        std::cout << "Actual file size:      " << static_cast<uint64_t>(file_size) << "\n";

        if (computed_file_end < static_cast<uint64_t>(file_size)) {
            uint64_t final_gap = static_cast<uint64_t>(file_size) - computed_file_end;
            total_padding_bytes += final_gap;
            has_gaps = true;
            std::cout << "EOF Padding Gap:       " << final_gap << " bytes\n";
        } else if (computed_file_end > static_cast<uint64_t>(file_size)) {
            std::cout << "[FATAL] Last tensor exceeds EOF by " << (computed_file_end - static_cast<uint64_t>(file_size)) << " bytes\n";
            has_overlaps = true;
        }

        // Tiling summmary
        std::cout << "\n--- Tiling Result ---\n";
        if (has_overlaps) {
            std::cout << "FAIL: Overlaps detected! Constants are wrong or file is corrupted.\n";
        } else if (has_gaps) {
            std::cout << "PASS WITH GAPS: No overlaps. Quantization constants are correct.\n";
            std::cout << "Total alignment padding counted: " << total_padding_bytes << " bytes.\n";
        } else {
            std::cout << "PASS PERFECTLY: Absolute contiguous tiling with zero gaps.\n";
        }


    } catch (const std::exception& e) {
        std::cerr << "Parsing failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
