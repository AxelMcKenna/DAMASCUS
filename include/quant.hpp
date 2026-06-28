#pragma once

// Q4_K / Q6_K dequantization — a CPU reference pulled forward from Compartment 9
// because Milestone A (Compartment 6) needs f32 weights but the GGUF stores them
// quantized. This is the scalar reference; the fused dequant→AMX path (the
// project's headline) is built against it later. Byte layouts follow ggml's
// block_q4_K / block_q6_K exactly (super-blocks of 256, QK_K=256).

#include <cstddef>
#include <cstdint>
#include <vector>

namespace damascus::quant {

// ggml type tags we handle (see gguf.cpp get_type_traits).
enum GgmlType : uint32_t {
    GGML_F32  = 0,
    GGML_F16  = 1,
    GGML_Q4_K = 12,
    GGML_Q6_K = 14,
};

float half_to_float(uint16_t h);

// Dequantize `n_elements` values of the given ggml type from `src` into f32.
// `n_elements` must be a multiple of 256 for the K-quants.
std::vector<float> dequantize(uint32_t ggml_type, const std::byte* src,
                              std::uint64_t n_elements);

}  // namespace damascus::quant
