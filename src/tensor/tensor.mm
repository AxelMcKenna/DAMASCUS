#include "tensor.hpp"
#include "metal/metal_ctx_internal.hpp"

#import <Metal/Metal.h>

#include <cassert>
#include <stdexcept>
#include <utility>

namespace damascus {

struct TensorImpl {
    id<MTLBuffer> buffer = nil;
};

Tensor::Tensor(MetalContext* ctx, std::vector<std::size_t> s, Dtype dt)
    : shape(std::move(s)), dtype(dt) {
    if (ctx == nullptr) {
        throw std::runtime_error("Error: MetalContext is null");
    }

    const std::size_t bytes = size_in_bytes();

    id<MTLDevice> device = metal_ctx_device(ctx);
    if (device == nil) {
        throw std::runtime_error("Error: Device is null");
    }

    impl = new TensorImpl{};

    impl->buffer = [device newBufferWithLength:bytes
                                       options:MTLResourceStorageModeShared];

    if (impl->buffer == nil) {
        delete impl;
        impl = nullptr;
        throw std::runtime_error("Error: failed to allocate Metal Buffer");
    }
}

Tensor::~Tensor() {
    delete impl;
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape(std::move(other.shape)),
      dtype(other.dtype),
      impl(std::exchange(other.impl, nullptr)) {}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    delete impl;

    shape = std::move(other.shape);
    dtype = other.dtype;
    impl = std::exchange(other.impl, nullptr);

    return *this;
}

std::size_t Tensor::size_in_bytes() const {
    if (shape.empty()) return 0;

    std::size_t total_elements = 1;
    for (std::size_t dim : shape) {
        total_elements *= dim;
    }

    return total_elements * dtype_size(dtype);
}

std::byte* Tensor::data() {
    if (impl == nullptr || impl->buffer == nil) return nullptr;

    return static_cast<std::byte*>([impl->buffer contents]);
}

const std::byte* Tensor::data() const {
    if (impl == nullptr || impl->buffer == nil) return nullptr;

    return static_cast<const std::byte*>([impl->buffer contents]);
}

float* Tensor::as_f32() {
    assert(dtype == Dtype::f32);

    return reinterpret_cast<float*>(data());
}

const float* Tensor::as_f32() const {
    assert(dtype == Dtype::f32);

    return reinterpret_cast<const float*>(data());
}

}  // namespace damascus