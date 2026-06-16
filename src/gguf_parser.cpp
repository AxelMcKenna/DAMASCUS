#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>
#include <stdexcept>


struct TensorInfo {
    std::string name;
    std::vector<int64_t> dims;
    uint32_t type;
    uint64_t offset;
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

// Helper to read a value based on a pre-determined tag type and advance the cursor
void read_value(const std::vector<std::byte>& buffer, size_t& cursor, uint32_t type, bool print = true) {
    switch (type)
    {
        case GGUF_TYPE_UINT8: {
            uint8_t value = read_primitive<uint8_t>(buffer, cursor);
            if (print) std::cout << static_cast<uint32_t>(value);
            break;
        }

        case GGUF_TYPE_INT8: {
            int8_t value = read_primitive<int8_t>(buffer, cursor);
            if (print) std::cout << static_cast<int32_t>(value);
            break;
        }

        case GGUF_TYPE_UINT16: {
            uint16_t value = read_primitive<uint16_t>(buffer, cursor);
            if (print) std::cout << value;
            break;
        }

        case GGUF_TYPE_INT16: {
            int16_t value = read_primitive<int16_t>(buffer, cursor);
            if (print) std::cout << value;
            break;
        }

        case GGUF_TYPE_UINT32: {
            uint32_t value = read_primitive<uint32_t>(buffer, cursor);
            if (print) std::cout << value;
            break;
        }

        case GGUF_TYPE_INT32: {
            int32_t value = read_primitive<int32_t>(buffer, cursor);
            if (print) std::cout << value;
            break;
        }

        case GGUF_TYPE_FLOAT32: {
            float value = read_primitive<float>(buffer, cursor);
            if (print) std::cout << value;
            break;
        }

        case GGUF_TYPE_BOOL: {
            uint8_t value = read_primitive<uint8_t>(buffer, cursor);
            if (print) std::cout << (value ? "true" : "false");
            break;
        }

        case GGUF_TYPE_STRING: {
            std::string value = read_string(buffer, cursor);
            if (print) std::cout << "\"" << value << "\"";
            break;
        }

        case GGUF_TYPE_UINT64: {
            uint64_t value = read_primitive<uint64_t>(buffer, cursor);
            if (print) std::cout << value;
            break;
        }

        case GGUF_TYPE_INT64: {
            int64_t value = read_primitive<int64_t>(buffer, cursor);
            if (print) std::cout << value;
            break;
        }

        case GGUF_TYPE_FLOAT64: {
            double value = read_primitive<double>(buffer, cursor);
            if (print) std::cout << value;
            break;
        }

        case GGUF_TYPE_ARRAY: {
            uint32_t element_type = read_primitive<uint32_t>(buffer, cursor);
            uint64_t count = read_primitive<uint64_t>(buffer, cursor);

            if (print) {
                std::cout << "array<" << type_name(element_type) << ">[" << count << "]";
                std::cout << " = [";
            }

            const uint64_t preview_count = std::min<uint64_t>(count, 5);

            for (uint64_t i = 0; i < count; ++i) {
                bool should_print_element = print && i < preview_count;

                if (should_print_element) {
                    if (i > 0) {
                        std::cout << ", ";
                    }
                }

                read_value(buffer, cursor, element_type, should_print_element);
            }

            if (print) {
                if (count > preview_count) {
                    std::cout << ", ...";
                }

                std::cout << "]";
            }

            break;
        }

        default: {
            throw std::runtime_error("Unknown GGUF value type tag: " + std::to_string(type));
        }
    }
}

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
        std::cout << "\n--- GGUF Metadata ---\n";

        for (uint64_t i = 0; i < metadata_kv_count; ++i) {
            std::string key = read_string(file_buffer, cursor);
            uint32_t value_type = read_primitive<uint32_t>(file_buffer, cursor);

            std::cout << "\n[" << i << "] ";
            std::cout << key << "\n";
            std::cout << "Type:  " << type_name(value_type) << " (" << value_type << ")\n";
            std::cout << "Value: ";

            read_value(file_buffer, cursor, value_type);

            std::cout << "\nCursor: " << cursor << " bytes\n";
        }

        std::cout << "\nFinished metadata parsing.\n";
        std::cout << "Cursor after metadata: " << cursor << " bytes\n";

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

        // -- Close the loop check --
        //
        // Temporary hand-computed size for the known final tensor
        // output_norm.weight has shape [3072] and type [F32
        // F32 = 4 bytes, so byte size is 3072 times 4
        //
        // Later will replace with a real tensor byte-size function
        // That function must understand ggml type properly, including quantized formats
        // like Q4_K and Q6_K, because quantized tensors are block encoded
        // and cannot be sized as num-elements times sizeof(scalar)
        if (tensors.empty()) {
            throw std::runtime_error("Error: No tensors found");
        }

        const TensorInfo& last_tensor = tensors.back();

        uint64_t last_byte_size = 3072ull * 4ull;
        uint64_t computed_file_end = data_blob_start + last_tensor.offset + last_byte_size;

        std::cout << "\n--- Close-the-loop Check ---\n";
        std::cout << "Last tensor name:          " << last_tensor.name << "\n";
        std::cout << "Last tensor offset:        " << last_tensor.offset << "\n";
        std::cout << "Last tensor byte size:     " << last_byte_size << "\n";
        std::cout << "Computed file end:         " << computed_file_end << "\n";
        std::cout << "Actual file size:          " << static_cast<uint64_t>(file_size) << "\n";

        if (computed_file_end == static_cast<uint64_t>(file_size)) {
            std::cout << "Close-the-loop check:      PASS\n";
        } else {
            std::cout << "Close-the-loop check:      FAIL\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Parsing failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
