#pragma once

// Minimal NumPy .npy v1.0 reader/writer — just enough to interoperate with the
// oracle harness (read int32 token ids, write float32 arrays). C order only.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace damascus::npy {

inline std::vector<int32_t> read_i32(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("npy: cannot open " + path);

    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error("npy: bad magic");
    f.ignore(2);  // version
    uint16_t hlen;
    f.read(reinterpret_cast<char*>(&hlen), 2);
    std::string header(hlen, '\0');
    f.read(header.data(), hlen);
    if (header.find("'<i4'") == std::string::npos &&
        header.find("'|i4'") == std::string::npos) {
        throw std::runtime_error("npy: expected int32 dtype in " + path);
    }
    // remaining bytes are the data
    std::vector<int32_t> out;
    int32_t v;
    while (f.read(reinterpret_cast<char*>(&v), 4)) out.push_back(v);
    return out;
}

inline void write_f32(const std::string& path,
                      const std::vector<float>& data,
                      const std::vector<std::size_t>& shape) {
    std::string shape_str = "(";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        shape_str += std::to_string(shape[i]);
        if (shape.size() == 1 || i + 1 < shape.size()) shape_str += ", ";
    }
    shape_str += ")";

    std::string header =
        "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_str + ", }";
    // pad so that 10 + header.size() + 1(newline) is a multiple of 64
    std::size_t total = 10 + header.size() + 1;
    std::size_t pad = (64 - (total % 64)) % 64;
    header.append(pad, ' ');
    header.push_back('\n');

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("npy: cannot write " + path);
    f.write("\x93NUMPY", 6);
    const char ver[2] = {1, 0};
    f.write(ver, 2);
    auto hlen = static_cast<uint16_t>(header.size());
    f.write(reinterpret_cast<const char*>(&hlen), 2);
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    f.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(float)));
}

}  // namespace damascus::npy
