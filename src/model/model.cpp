#include "model.hpp"
#include "quant.hpp"

#include <stdexcept>

namespace damascus {

const TensorInfo& Model::find(const std::string& name) const {
    for (const TensorInfo& t : gguf_.tensors) {
        if (t.name == name) return t;
    }
    throw std::runtime_error("Model: tensor not found: " + name);
}

bool Model::has(const std::string& name) const {
    for (const TensorInfo& t : gguf_.tensors) {
        if (t.name == name) return true;
    }
    return false;
}

Weight Model::get(const std::string& name) const {
    const TensorInfo& t = find(name);

    std::uint64_t n = 1;
    for (int64_t d : t.dims) n *= static_cast<std::uint64_t>(d);

    const std::byte* src = gguf_.data.data() + gguf_.data_blob_start + t.offset;

    Weight w;
    w.data = quant::dequantize(t.type, src, n);
    w.dims = t.dims;
    return w;
}

}  // namespace damascus
