#include <iostream>
#include <fstream>
#include <vector>
#include <cstddef>
#include <cstring>
#include <string>
#include <stdexcept>

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

int main() {
    // Fixed: restored the complete, proper path to the llama-bpe vocabulary file
    const std::string file_path = "third_party/llama.cpp/models/ggml-vocab-llama-bpe.gguf";

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
        char magic_str[5] = {0}; // Fixed: added missing semicolon
        std::memcpy(magic_str, &magic, 4);

        // Output
        std::cout << "\n--- GGUF Header Analysis ---\n";
        std::cout << "Magic Bytes:     " << magic_str << "\n";
        std::cout << "Format Version:  " << version << "\n";
        std::cout << "Tensor Count:    " << tensor_count << "\n";
        std::cout << "Metadata KVs:    " << metadata_kv_count << "\n";
        std::cout << "Current Cursor:  " << cursor << " bytes\n";
        std::cout << "------------------------------\n";

        // --- Section 2: Isolated Metadata ---
        std::cout << "\nParsing first metadata key...\n";
        std::string first_key = read_string(file_buffer, cursor);

        std::cout << "Successfully extracted key\n";
        std::cout << "Key Content:     \"" << first_key << "\"\n";
        std::cout << "Key Size:        " << first_key.length() << " characters\n";
        std::cout << "Current Cursor:  " << cursor << " bytes\n";
        std::cout << "------------------------------\n";

        // --- Section 3: First Metadata Value
        std::cout << "\nParsing first metadata key...\n";

        // Step 1: read the unit32 type tag
        uint32_t value_type = read_primitive<uint32_t>(file_buffer,cursor);

        std::cout << "Value Type Tag:  " << value_type << "\n";
        std::cout << "Current Cursor:  " << cursor << " bytes\n";

        // Step 2: general.architecture should be a string GGUF type tag 8
        if (value_type == 8) {
            std::string value = read_string(file_buffer, cursor);

            std::cout << "Successfully extracted value!\n";
            std::cout << "Value Content:   \"" << value << "\"\n";
            std::cout << "Value Size:      " << value.length() << " characters\n";
            std::cout << "Current Cursor:  " << cursor << " bytes\n";
        } else {
            std::cout << "Unexpected value type for first metadata key.\n";
        }



    } catch (const std::exception& e) {
        std::cerr << "Parsing failed: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
