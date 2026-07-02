#pragma once

// Compartment 6 — the decoder forward pass. Composes the Compartment 5 kernels
// into the full Llama graph (embed → 28×[RMSNorm, GQA attention w/ RoPE, SwiGLU
// FFN] → final RMSNorm → tied LM head) and returns every residual-stream
// snapshot plus the logits, in the same layout as the HF oracle's hidden.npy /
// logits.npy so compare.py can gate Milestone A.

#include "model.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace damascus {

struct ForwardResult {
    std::size_t seq = 0;
    std::size_t d = 0;
    std::size_t vocab = 0;
    std::size_t n_layers = 0;

    // hidden: (n_layers + 1) × seq × d, row-major.
    //   [0]      = token embeddings (layer-0 input)
    //   [k]      = residual stream after decoder layer k-1
    // (matches the oracle's hidden_index_legend; index -1 here is BEFORE the
    //  final norm — we also expose the post-final-norm stream as `final_norm`.)
    std::vector<float> hidden;
    std::vector<float> final_norm;  // seq × d, input to the LM head
    std::vector<float> logits;      // seq × vocab
};

ForwardResult forward(const Model& model, const std::vector<int32_t>& tokens);

// --- Compartment 8 (runtime) minimal decode path -------------------------
//
// KVCache holds the post-RoPE K and V rows for every layer and position, so
// a decode step recomputes nothing: the new token attends over cached rows.
// Layout: k/v are [n_layers, max_seq, kv_dim], row-major, preallocated.
struct KVCache {
    std::size_t max_seq = 0;
    std::size_t n_layers = 0;
    std::size_t kv_dim = 0;
    std::size_t len = 0;          // positions currently filled
    std::vector<float> k_store, v_store;  // backing when not GPU-backed
    float* kp = nullptr;
    float* vp = nullptr;

    // Pass gpu::kv_alloc pointers to make the cache GPU-backed (unified
    // memory: CPU prefill writes through the same pointers the GPU decode
    // step reads). Without them it allocates plain vectors.
    KVCache(std::size_t max_seq_, std::size_t n_layers_, std::size_t kv_dim_,
            float* kbuf = nullptr, float* vbuf = nullptr)
        : max_seq(max_seq_), n_layers(n_layers_), kv_dim(kv_dim_) {
        if (kbuf != nullptr && vbuf != nullptr) {
            kp = kbuf;
            vp = vbuf;
        } else {
            k_store.assign(n_layers_ * max_seq_ * kv_dim_, 0.0f);
            v_store.assign(n_layers_ * max_seq_ * kv_dim_, 0.0f);
            kp = k_store.data();
            vp = v_store.data();
        }
    }

    float* k_row(std::size_t layer, std::size_t pos) {
        return kp + (layer * max_seq + pos) * kv_dim;
    }
    float* v_row(std::size_t layer, std::size_t pos) {
        return vp + (layer * max_seq + pos) * kv_dim;
    }
};

// Process `tokens` starting at absolute position cache.len (prefill: the whole
// prompt at once; decode: one token). Appends the chunk's K/V to the cache and
// returns the LAST position's logits (vocab floats). Arithmetic per position is
// identical to forward(), so greedy decode through this path must reproduce the
// full-recompute reference token-for-token.
std::vector<float> forward_chunk(const Model& model, KVCache& cache,
                                 std::span<const int32_t> tokens);

}  // namespace damascus
