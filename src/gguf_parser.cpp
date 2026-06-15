#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>
#include <stdexcept>


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

    } catch (const std::exception& e) {
        std::cerr << "Parsing failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
