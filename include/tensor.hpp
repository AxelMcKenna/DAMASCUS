#pragma once

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace damascus {

enum class Dtype {
    f32,
    f16,
    i8,
};

inline std::size_t dtype_size(Dtype dtype) {
    switch (dtype) {
    case Dtype::f32: return 4;
    case Dtype::f16: return 2;
    case Dtype::i8:  return 1;
    default:
        throw std::runtime_error("Error: unknown data type");
    }
}

struct TensorImpl;
struct MetalContext;

struct Tensor {
    std::vector<std::size_t> shape;
    Dtype dtype;

    Tensor(MetalContext* ctx, std::vector<std::size_t> s, Dtype dt);

    // Rule of 5
    ~Tensor();

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    std::size_t size_in_bytes() const;

    std::byte* data();
    const std::byte* data() const;

    float* as_f32();
    const float* as_f32() const;

private:
    TensorImpl* impl = nullptr;
};

}  // namespace damascus