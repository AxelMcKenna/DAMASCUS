#pragma once

// Thin accessor over a loaded GGUF: locate a named weight tensor and hand back
// its dequantized f32 values plus shape. Compartment 6 uses this to pull each
// weight just-in-time for its matmul (keeps peak memory ~one big weight at a
// time rather than materializing all ~3B params in f32 at once).

#include "gguf.hpp"

#include <string>
#include <vector>

namespace damascus {

struct Weight {
    std::vector<float> data;
    std::vector<int64_t> dims;  // ggml ne order: dims[0] = in (contiguous)
};

class Model {
public:
    explicit Model(const GGUFModel& gguf) : gguf_(gguf) {}

    const Config& config() const { return gguf_.config; }

    bool has(const std::string& name) const;

    // Dequantize the named tensor to f32. Throws if absent.
    Weight get(const std::string& name) const;

private:
    const TensorInfo& find(const std::string& name) const;
    const GGUFModel& gguf_;
};

}  // namespace damascus
