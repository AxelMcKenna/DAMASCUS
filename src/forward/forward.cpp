#include "forward.hpp"
#include "gpu.hpp"
#include "kernels.hpp"
#include "quant.hpp"

#include <cmath>
#include <span>
#include <stdexcept>

namespace damascus {

namespace {

// y = x · Wᵀ for a GGUF weight stored in ne order [in, out] (in contiguous),
// dequantized row-by-row inside the matmul (never materialized in f32).
std::vector<float> mm(const WeightView& w, std::span<const float> x, std::size_t rows) {
    const std::size_t k = static_cast<std::size_t>(w.dims.at(0));  // in_features
    const std::size_t n = static_cast<std::size_t>(w.dims.at(1));  // out_features
    std::vector<float> out(rows * n);
    if (!gpu::matmul(out, x, w.type, w.data, rows, n, k)) {
        cpu::matmul_q(out, x, w.type, w.data, rows, n, k);
    }
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

    gpu::init(model.mapping());

    // llama3 RoPE frequency divisors, if present (Llama 3.2 has them).
    std::vector<float> freq_factors;
    if (model.has("rope_freqs.weight")) freq_factors = model.get("rope_freqs.weight").data;

    const WeightView embd = model.view("token_embd.weight");  // [d, vocab]
    const std::size_t vocab = static_cast<std::size_t>(embd.dims.at(1));
    const std::size_t embd_row = quant::row_bytes(embd.type, d);

    ForwardResult res;
    res.seq = seq; res.d = d; res.vocab = vocab; res.n_layers = n_layers;
    res.hidden.assign((n_layers + 1) * seq * d, 0.0f);

    auto hidden_slot = [&](std::size_t idx) {
        return std::span<float>(res.hidden.data() + idx * seq * d, seq * d);
    };

    // --- token embeddings -> residual stream x [seq, d] ---
    // Dequantize only the rows we look up (rows are block-aligned: d % 256 == 0).
    std::vector<float> x(seq * d);
    for (std::size_t t = 0; t < seq; ++t) {
        const int32_t id = tokens[t];
        if (id < 0 || static_cast<std::size_t>(id) >= vocab)
            throw std::runtime_error("forward: token id out of range");
        quant::dequantize_to(embd.type,
                             embd.data + static_cast<std::size_t>(id) * embd_row,
                             d, x.data() + t * d);
    }
    std::copy(x.begin(), x.end(), hidden_slot(0).begin());

    std::vector<float> normed(seq * d);

    for (std::size_t L = 0; L < n_layers; ++L) {
        const std::string p = "blk." + std::to_string(L) + ".";
        const Weight w_attn_norm = model.get(p + "attn_norm.weight");
        const WeightView w_q  = model.view(p + "attn_q.weight");
        const WeightView w_k  = model.view(p + "attn_k.weight");
        const WeightView w_v  = model.view(p + "attn_v.weight");
        const WeightView w_o  = model.view(p + "attn_output.weight");

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
        const WeightView w_gate = model.view(p + "ffn_gate.weight");
        const WeightView w_up   = model.view(p + "ffn_up.weight");
        const WeightView w_down = model.view(p + "ffn_down.weight");

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

std::vector<float> forward_chunk(const Model& model, KVCache& cache,
                                 std::span<const int32_t> tokens) {
    const Config& cfg = model.config();

    const std::size_t seq      = tokens.size();
    const std::size_t start    = cache.len;
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

    if (seq == 0) throw std::runtime_error("forward_chunk: empty chunk");

    gpu::init(model.mapping());
    if (cache.n_layers != n_layers || cache.kv_dim != kv_dim)
        throw std::runtime_error("forward_chunk: cache shape mismatch");
    if (start + seq > cache.max_seq)
        throw std::runtime_error("forward_chunk: cache overflow");

    std::vector<float> freq_factors;
    if (model.has("rope_freqs.weight")) freq_factors = model.get("rope_freqs.weight").data;

    const WeightView embd = model.view("token_embd.weight");  // [d, vocab]
    const std::size_t vocab = static_cast<std::size_t>(embd.dims.at(1));
    const std::size_t embd_row = quant::row_bytes(embd.type, d);

    // Decode fast path: the whole token step in one GPU command buffer.
    if (seq == 1) {
        std::vector<float> logits(vocab);
        if (gpu::decode_step(model, cache, tokens[0], logits)) return logits;
    }
    // Prefill fast path: whole chunk on GPU (fp16 MPS GEMMs). NOTE: fp16
    // matmuls deviate more from the fp64 CPU reference than the decode path;
    // gated on end-to-end output checks, not bit-identity.
    if (seq > 1) {
        std::vector<float> logits(vocab);
        if (gpu::prefill_chunk(model, cache, tokens, logits)) return logits;
    }

    std::vector<float> x(seq * d);
    for (std::size_t t = 0; t < seq; ++t) {
        const int32_t id = tokens[t];
        if (id < 0 || static_cast<std::size_t>(id) >= vocab)
            throw std::runtime_error("forward_chunk: token id out of range");
        quant::dequantize_to(embd.type,
                             embd.data + static_cast<std::size_t>(id) * embd_row,
                             d, x.data() + t * d);
    }

    std::vector<float> normed(seq * d);

    for (std::size_t L = 0; L < n_layers; ++L) {
        const std::string p = "blk." + std::to_string(L) + ".";
        const Weight w_attn_norm = model.get(p + "attn_norm.weight");
        const WeightView w_q  = model.view(p + "attn_q.weight");
        const WeightView w_k  = model.view(p + "attn_k.weight");
        const WeightView w_v  = model.view(p + "attn_v.weight");
        const WeightView w_o  = model.view(p + "attn_output.weight");

        for (std::size_t t = 0; t < seq; ++t) {
            cpu::rmsnorm(std::span<float>(normed.data() + t * d, d),
                         std::span<const float>(x.data() + t * d, d),
                         w_attn_norm.data, eps);
        }
        std::vector<float> Q  = mm(w_q, normed, seq);   // [seq, d]
        std::vector<float> Kc = mm(w_k, normed, seq);   // [seq, kv_dim]
        std::vector<float> Vc = mm(w_v, normed, seq);   // [seq, kv_dim]

        // RoPE at absolute positions, then append the chunk's K/V to the cache.
        for (std::size_t t = 0; t < seq; ++t) {
            const std::size_t pos = start + t;
            for (std::size_t h = 0; h < n_head; ++h) {
                cpu::rope_interleaved(std::span<float>(Q.data() + t * d + h * head_dim, head_dim),
                                      head_dim, pos, theta, freq_factors);
            }
            for (std::size_t h = 0; h < n_kv; ++h) {
                cpu::rope_interleaved(std::span<float>(Kc.data() + t * kv_dim + h * head_dim, head_dim),
                                      head_dim, pos, theta, freq_factors);
            }
            std::copy(Kc.begin() + t * kv_dim, Kc.begin() + (t + 1) * kv_dim,
                      cache.k_row(L, pos));
            std::copy(Vc.begin() + t * kv_dim, Vc.begin() + (t + 1) * kv_dim,
                      cache.v_row(L, pos));
        }

        // Causal GQA attention over the cache (positions 0..i for query at i).
        std::vector<float> attn_out(seq * d, 0.0f);
        std::vector<float> scores(start + seq);
        for (std::size_t ti = 0; ti < seq; ++ti) {
            const std::size_t i = start + ti;
            for (std::size_t h = 0; h < n_head; ++h) {
                const std::size_t kv = h / n_rep;
                const float* q = Q.data() + ti * d + h * head_dim;

                for (std::size_t j = 0; j <= i; ++j) {
                    const float* kvec = cache.k_row(L, j) + kv * head_dim;
                    double dot = 0.0;
                    for (std::size_t e = 0; e < head_dim; ++e) dot += static_cast<double>(q[e]) * kvec[e];
                    scores[j] = static_cast<float>(dot) * scale;
                }
                cpu::softmax(std::span<float>(scores.data(), i + 1));

                float* out = attn_out.data() + ti * d + h * head_dim;
                for (std::size_t j = 0; j <= i; ++j) {
                    const float* vvec = cache.v_row(L, j) + kv * head_dim;
                    const float a = scores[j];
                    for (std::size_t e = 0; e < head_dim; ++e) out[e] += a * vvec[e];
                }
            }
        }

        std::vector<float> proj = mm(w_o, attn_out, seq);
        cpu::residual_add(x, proj);

        const Weight w_ffn_norm = model.get(p + "ffn_norm.weight");
        const WeightView w_gate = model.view(p + "ffn_gate.weight");
        const WeightView w_up   = model.view(p + "ffn_up.weight");
        const WeightView w_down = model.view(p + "ffn_down.weight");

        for (std::size_t t = 0; t < seq; ++t) {
            cpu::rmsnorm(std::span<float>(normed.data() + t * d, d),
                         std::span<const float>(x.data() + t * d, d),
                         w_ffn_norm.data, eps);
        }
        std::vector<float> gate = mm(w_gate, normed, seq);
        std::vector<float> up   = mm(w_up,   normed, seq);
        cpu::swiglu(gate, gate, up);
        std::vector<float> down = mm(w_down, gate, seq);
        cpu::residual_add(x, down);
    }

    cache.len = start + seq;

    // Only the last position feeds sampling — skip the LM head for the rest.
    const Weight w_out_norm = model.get("output_norm.weight");
    std::vector<float> last_norm(d);
    cpu::rmsnorm(last_norm,
                 std::span<const float>(x.data() + (seq - 1) * d, d),
                 w_out_norm.data, eps);
    return mm(embd, last_norm, 1);  // [1, vocab]
}

}  // namespace damascus
