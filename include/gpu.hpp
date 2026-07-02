#pragma once

// Metal fused dequant matmul (perf spike). The whole GGUF blob lives in one
// shared MTLBuffer (unified memory — no copies after init); each kernel
// dequantizes Q4_K/Q6_K blocks in registers and dots against x, one simdgroup
// per output feature. fp32 accumulation: NOT bit-identical to the CPU fp64
// reference — correctness is gated on greedy-token identity and per-tensor
// max-rel-err < 1e-4 (--gpu-check).

#include <cstddef>
#include <cstdint>
#include <span>

namespace damascus::gpu {

// Idempotent. Uploads the blob on first call; returns false if Metal is
// unavailable or DAMASCUS_GPU=0. Safe to call every forward().
bool init(const std::byte* blob, std::size_t size);

bool available();

// y = x · Wᵀ, same contract as cpu::matmul_q. `w` is a host pointer into the
// blob passed to init() (the offset is derived from it). Returns false if the
// weight type has no GPU kernel — caller falls back to CPU.
bool matmul(std::span<float> out, std::span<const float> x, std::uint32_t wtype,
            const std::byte* w, std::size_t rows, std::size_t n, std::size_t k);

}  // namespace damascus::gpu
