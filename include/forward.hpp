#pragma once

// Compartment 6 — the decoder forward pass. Composes the Compartment 5 kernels
// into the full Llama graph (embed → 28×[RMSNorm, GQA attention w/ RoPE, SwiGLU
// FFN] → final RMSNorm → tied LM head) and returns every residual-stream
// snapshot plus the logits, in the same layout as the HF oracle's hidden.npy /
// logits.npy so compare.py can gate Milestone A.

#include "model.hpp"

#include <cstdint>
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

}  // namespace damascus
