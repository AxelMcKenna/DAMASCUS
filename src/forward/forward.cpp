#include "forward.hpp"
#include "kernels.hpp"

#include <cmath>
#include <span>
#include <stdexcept>

namespace damascus {

namespace {

// y = x · Wᵀ for a GGUF weight stored in ne order [in, out] (in contiguous).
std::vector<float> mm(const Weight& w, std::span<const float> x, std::size_t rows) {
    const std::size_t k = static_cast<std::size_t>(w.dims.at(0));  // in_features
    const std::size_t n = static_cast<std::size_t>(w.dims.at(1));  // out_features
    std::vector<float> out(rows * n);
    cpu::matmul(out, x, w.data, rows, n, k);
    return out;
}

}  // namespace

ForwardResult forward(const Model& model, const std::vector<int32_t>& tokens) {
    const Config& cfg = model.config();

    const std::size_t seq      = tokens.size();
    const std::size_t d        = cfg.embedding_length;
    const std::size_t n_head   = cfg.attention_head_count;
    const std::size_t n_kv     = cfg.attention_head_count_kv;
    const std::size_t head_dim = d / n_head;
    const std::size_t kv_dim   = n_kv * head_dim;
    const std::size_t n_rep    = n_head / n_kv;
    const std::size_t n_layers = cfg.block_count;
    const float       eps      = cfg.rms_norm_epsilon;
    const float       theta    = cfg.rope_freq_base;
    const float       scale    = 1.0f / std::sqrt(static_cast<float>(head_dim));

    if (seq == 0) throw std::runtime_error("forward: empty prompt");

    // llama3 RoPE frequency divisors, if present (Llama 3.2 has them).
    std::vector<float> freq_factors;
    if (model.has("rope_freqs.weight")) freq_factors = model.get("rope_freqs.weight").data;

    const Weight embd = model.get("token_embd.weight");  // [d, vocab]
    const std::size_t vocab = static_cast<std::size_t>(embd.dims.at(1));

    ForwardResult res;
    res.seq = seq; res.d = d; res.vocab = vocab; res.n_layers = n_layers;
    res.hidden.assign((n_layers + 1) * seq * d, 0.0f);

    auto hidden_slot = [&](std::size_t idx) {
        return std::span<float>(res.hidden.data() + idx * seq * d, seq * d);
    };

    // --- token embeddings -> residual stream x [seq, d] ---
    std::vector<float> x(seq * d);
    for (std::size_t t = 0; t < seq; ++t) {
        cpu::embedding_lookup(std::span<float>(x.data() + t * d, d),
                              embd.data, tokens[t], d);
    }
    std::copy(x.begin(), x.end(), hidden_slot(0).begin());

    std::vector<float> normed(seq * d);

    for (std::size_t L = 0; L < n_layers; ++L) {
        const std::string p = "blk." + std::to_string(L) + ".";
        const Weight w_attn_norm = model.get(p + "attn_norm.weight");
        const Weight w_q  = model.get(p + "attn_q.weight");
        const Weight w_k  = model.get(p + "attn_k.weight");
        const Weight w_v  = model.get(p + "attn_v.weight");
        const Weight w_o  = model.get(p + "attn_output.weight");

        // --- RMSNorm -> QKV ---
        for (std::size_t t = 0; t < seq; ++t) {
            cpu::rmsnorm(std::span<float>(normed.data() + t * d, d),
                         std::span<const float>(x.data() + t * d, d),
                         w_attn_norm.data, eps);
        }
        std::vector<float> Q = mm(w_q, normed, seq);   // [seq, d]
        std::vector<float> K = mm(w_k, normed, seq);   // [seq, kv_dim]
        std::vector<float> V = mm(w_v, normed, seq);   // [seq, kv_dim]

        // --- RoPE on each head of Q and K, at absolute position t ---
        for (std::size_t t = 0; t < seq; ++t) {
            for (std::size_t h = 0; h < n_head; ++h) {
                cpu::rope_interleaved(std::span<float>(Q.data() + t * d + h * head_dim, head_dim),
                                      head_dim, t, theta, freq_factors);
            }
            for (std::size_t h = 0; h < n_kv; ++h) {
                cpu::rope_interleaved(std::span<float>(K.data() + t * kv_dim + h * head_dim, head_dim),
                                      head_dim, t, theta, freq_factors);
            }
        }

        // --- GQA attention (causal) -> attn_out [seq, d] ---
        std::vector<float> attn_out(seq * d, 0.0f);
        std::vector<float> scores(seq);
        for (std::size_t i = 0; i < seq; ++i) {          // query position
            for (std::size_t h = 0; h < n_head; ++h) {   // query head
                const std::size_t kv = h / n_rep;
                const float* q = Q.data() + i * d + h * head_dim;

                for (std::size_t j = 0; j <= i; ++j) {   // causal: keys 0..i
                    const float* kvec = K.data() + j * kv_dim + kv * head_dim;
                    double dot = 0.0;
                    for (std::size_t e = 0; e < head_dim; ++e) dot += static_cast<double>(q[e]) * kvec[e];
                    scores[j] = static_cast<float>(dot) * scale;
                }
                cpu::softmax(std::span<float>(scores.data(), i + 1));

                float* out = attn_out.data() + i * d + h * head_dim;
                for (std::size_t j = 0; j <= i; ++j) {
                    const float* vvec = V.data() + j * kv_dim + kv * head_dim;
                    const float a = scores[j];
                    for (std::size_t e = 0; e < head_dim; ++e) out[e] += a * vvec[e];
                }
            }
        }

        // --- output projection + residual ---
        std::vector<float> proj = mm(w_o, attn_out, seq);
        cpu::residual_add(x, proj);

        // --- FFN: RMSNorm -> SwiGLU -> down -> residual ---
        const Weight w_ffn_norm = model.get(p + "ffn_norm.weight");
        const Weight w_gate = model.get(p + "ffn_gate.weight");
        const Weight w_up   = model.get(p + "ffn_up.weight");
        const Weight w_down = model.get(p + "ffn_down.weight");

        for (std::size_t t = 0; t < seq; ++t) {
            cpu::rmsnorm(std::span<float>(normed.data() + t * d, d),
                         std::span<const float>(x.data() + t * d, d),
                         w_ffn_norm.data, eps);
        }
        std::vector<float> gate = mm(w_gate, normed, seq);  // [seq, ff]
        std::vector<float> up   = mm(w_up,   normed, seq);  // [seq, ff]
        cpu::swiglu(gate, gate, up);                        // gate <- silu(gate)*up
        std::vector<float> down = mm(w_down, gate, seq);    // [seq, d]
        cpu::residual_add(x, down);

        // snapshot residual stream after layers 0..n_layers-2 (matches oracle)
        if (L + 1 < n_layers) std::copy(x.begin(), x.end(), hidden_slot(L + 1).begin());
    }

    // --- final RMSNorm (lm_head input) + tied LM head -> logits ---
    const Weight w_out_norm = model.get("output_norm.weight");
    res.final_norm.assign(seq * d, 0.0f);
    for (std::size_t t = 0; t < seq; ++t) {
        cpu::rmsnorm(std::span<float>(res.final_norm.data() + t * d, d),
                     std::span<const float>(x.data() + t * d, d),
                     w_out_norm.data, eps);
    }
    std::copy(res.final_norm.begin(), res.final_norm.end(), hidden_slot(n_layers).begin());

    res.logits = mm(embd, res.final_norm, seq);  // tied: token_embd as LM head
    return res;
}

}  // namespace damascus
