#pragma once

// Compartment 5 — compute kernels, CPU reference implementations.
//
// Per ARCHITECTURE.md the rule is: write a scalar CPU reference FIRST, then
// port each kernel to a Metal shader validated against this reference within
// 1e-4. These functions are that reference. They are deliberately plain
// (no SIMD, no Metal) so they are obviously correct and run anywhere, which
// is also what unblocks them under the Docker/Linux CPU path.
//
// Conventions (match HF/llama so the Milestone-A oracle lines up):
//   * Weight matrices are row-major [out_features, in_features]; y = x · Wᵀ.
//   * RoPE uses the half-split (NEOX/HF) layout, not interleaved pairs.
//   * All activations are f32 here; quantized paths arrive in Compartment 9.

#include <cstddef>
#include <cstdint>
#include <span>

namespace damascus::cpu {

// out[i] = x[i] / rms(x) * weight[i],  rms = sqrt(mean(x^2) + eps).
// Shapes: out, x, weight all length `dim`.
void rmsnorm(std::span<float> out, std::span<const float> x,
             std::span<const float> weight, float eps);

// out[m, n] = sum_k x[m, k] * w[n, k]   (y = x · Wᵀ, llama weight layout).
// x: [rows, k]   w: [n, k]   out: [rows, n]   (all row-major, contiguous).
void matmul(std::span<float> out, std::span<const float> x,
            std::span<const float> w, std::size_t rows, std::size_t n,
            std::size_t k);

// Copy row `token_id` of the embedding table into `out` (length `dim`).
void embedding_lookup(std::span<float> out, std::span<const float> table,
                      std::int32_t token_id, std::size_t dim);

// Rotary position embedding, applied in place to one head vector `x` of length
// `head_dim` at absolute position `pos`. Half-split layout: dims j and j+d/2
// rotate together by angle pos * theta^(-2j/head_dim).
void rope(std::span<float> x, std::size_t head_dim, std::size_t pos, float theta);

// RoPE with per-frequency divisors (Llama 3 "llama3" rope scaling). The angle
// for pair j becomes pos * theta^(-2j/head_dim) / freq_factors[j]. An empty
// `freq_factors` reproduces plain rope(). `freq_factors` length = head_dim/2.
void rope_scaled(std::span<float> x, std::size_t head_dim, std::size_t pos,
                 float theta, std::span<const float> freq_factors);

// RoPE in GGML's "NORM" convention: rotates *adjacent* pairs (x[2i], x[2i+1])
// rather than the half-split (x[j], x[j+d/2]). This is what raw GGUF q/k
// weights expect — llama.cpp's converter permutes those weights so interleaved
// rotation reproduces HF's rotate_half result. `freq_factors` (length d/2,
// may be empty) divides the per-pair frequency for llama3 scaling.
void rope_interleaved(std::span<float> x, std::size_t head_dim, std::size_t pos,
                      float theta, std::span<const float> freq_factors);

// Numerically-stable softmax over `x`, in place.
void softmax(std::span<float> x);

// SwiGLU feed-forward activation: out[i] = silu(gate[i]) * up[i],
// silu(v) = v * sigmoid(v). out may alias gate.
void swiglu(std::span<float> out, std::span<const float> gate,
            std::span<const float> up);

// out[i] += x[i]  (residual add, in place on `out`).
void residual_add(std::span<float> out, std::span<const float> x);

}  // namespace damascus::cpu
