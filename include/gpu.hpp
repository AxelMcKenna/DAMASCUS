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

namespace damascus {
class Model;
struct KVCache;
}

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

// Shared-storage MTLBuffer exposed as a plain float* (unified memory), for a
// KV cache both CPU prefill and GPU decode can touch. nullptr if GPU is off.
float* kv_alloc(std::size_t n_floats);

// One whole decode step (all layers + LM head) in a single command buffer.
// Requires a kv_alloc-backed cache and context <= 2048; returns false if it
// can't run (caller falls back to forward_chunk's CPU orchestration).
// On success writes the vocab logits and advances cache.len.
bool decode_step(const Model& model, KVCache& cache, std::int32_t token,
                 std::span<float> logits_out);

// Greedy-decode out_tokens.size() tokens starting from `token`, entirely on
// GPU: embedding lookup, all layers, and argmax sampling chain inside one
// command buffer with zero CPU round-trips. Matches the CPU argmax
// tie-break. Returns false if unsupported (caller falls back per-token).
bool decode_chain(const Model& model, KVCache& cache, std::int32_t token,
                  std::span<std::int32_t> out_tokens);

// Whole prompt chunk on GPU in one command buffer. GEMMs run in fp16 via MPS
// against a one-time dequantized weight arena (compute-bound regime — fp16
// matmul hardware); norms/RoPE/attention stay fp32, KV cache stays fp32.
// Writes the LAST position's logits and advances cache.len. Returns false if
// unsupported (caller falls back to CPU-orchestrated prefill).
bool prefill_chunk(const Model& model, KVCache& cache,
                   std::span<const std::int32_t> tokens,
                   std::span<float> logits_out);

}  // namespace damascus::gpu
