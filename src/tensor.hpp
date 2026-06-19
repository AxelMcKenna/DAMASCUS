#pragma once

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <cassert>

namespace damascus {

enum class Dtype {
    f32,
    f16,
    i8,
};

// helper function to determine byte size based on data type
inline std::size_t dtype_size(Dtype dtype) {
    switch (dtype) {
    case Dtype::f32: return 4;
    case Dtype::f16: return 2;
    case Dtype::i8: return 1;
    default:
        throw std::runtime_error("Error: unknown data type");
    }
}

struct Tensor {
    std::byte* data;
    std::vector<std::size_t> shape;
    Dtype dtype;

    Tensor(std::vector<std::size_t> s, Dtype dt)    // initialises shape and dtype first
        : shape(s), dtype(dt) {
        data = new std::byte[size_in_bytes()];
    }

    // Ro5
    ~Tensor() {
        delete[] data;
    }

    Tensor(const Tensor&) = delete;             // no copy construction
    Tensor& operator=(const Tensor&) = delete;  // no copy assigment

    // move constructor
    Tensor(Tensor&& other) noexcept
        : data(std::exchange(other.data, nullptr)),
        shape(std::move(other.shape)),
        dtype(other.dtype)
    {}

    // move assigment operator
    Tensor& operator=(Tensor&& other) noexcept {
        if (this == &other) {
            return *this;
        }

        delete[] data; // clean up existing memory first

        data = std::exchange(other.data, nullptr);
        shape = std::move(other.shape);
        dtype = other.dtype;

        return *this;
    }


    std::size_t size_in_bytes() const {
        if (shape.empty()) return 0;

        std::size_t total_elements = 1;
        for (std::size_t dim: shape) {
            total_elements *= dim;
        }
        return total_elements * dtype_size(dtype);
    }

    float* as_f32() {
        assert(dtype == Dtype::f32);
        return reinterpret_cast<float*>(data);
    }

    const float* as_f32() const {
        assert(dtype == Dtype::f32);
        return reinterpret_cast<const float*>(data);
    }
};



}  // namespace damascus