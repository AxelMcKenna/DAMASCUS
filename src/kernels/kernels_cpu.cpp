#include "kernels.hpp"

#include <cmath>
#include <stdexcept>

namespace damascus::cpu {

void rmsnorm(std::span<float> out, std::span<const float> x,
             std::span<const float> weight, float eps) {
    if (x.size() != out.size() || x.size() != weight.size()) {
        throw std::runtime_error("rmsnorm: shape mismatch");
    }
    const std::size_t dim = x.size();

    double sum_sq = 0.0;  // accumulate in double; the reference is the oracle
    for (float v : x) sum_sq += static_cast<double>(v) * v;

    const float inv_rms =
        static_cast<float>(1.0 / std::sqrt(sum_sq / static_cast<double>(dim) + eps));

    for (std::size_t i = 0; i < dim; ++i) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

void matmul(std::span<float> out, std::span<const float> x,
            std::span<const float> w, std::size_t rows, std::size_t n,
            std::size_t k) {
    if (x.size() != rows * k || w.size() != n * k || out.size() != rows * n) {
        throw std::runtime_error("matmul: shape mismatch");
    }
    for (std::size_t m = 0; m < rows; ++m) {
        const float* xr = x.data() + m * k;
        float* outr = out.data() + m * n;
        for (std::size_t j = 0; j < n; ++j) {
            const float* wr = w.data() + j * k;
            double acc = 0.0;
            for (std::size_t i = 0; i < k; ++i) {
                acc += static_cast<double>(xr[i]) * wr[i];
            }
            outr[j] = static_cast<float>(acc);
        }
    }
}

void embedding_lookup(std::span<float> out, std::span<const float> table,
                      std::int32_t token_id, std::size_t dim) {
    if (out.size() != dim) throw std::runtime_error("embedding_lookup: out size");
    if (token_id < 0) throw std::runtime_error("embedding_lookup: negative id");

    const std::size_t row = static_cast<std::size_t>(token_id);
    if ((row + 1) * dim > table.size()) {
        throw std::runtime_error("embedding_lookup: id out of range");
    }
    const float* src = table.data() + row * dim;
    for (std::size_t i = 0; i < dim; ++i) out[i] = src[i];
}

void rope_scaled(std::span<float> x, std::size_t head_dim, std::size_t pos,
                 float theta, std::span<const float> freq_factors) {
    if (x.size() != head_dim || (head_dim & 1U)) {
        throw std::runtime_error("rope: head_dim must be even and match x");
    }
    const std::size_t half = head_dim / 2;
    if (!freq_factors.empty() && freq_factors.size() != half) {
        throw std::runtime_error("rope: freq_factors length must be head_dim/2");
    }
    for (std::size_t j = 0; j < half; ++j) {
        double freq =
            std::pow(static_cast<double>(theta),
                     -2.0 * static_cast<double>(j) / static_cast<double>(head_dim));
        if (!freq_factors.empty()) freq /= static_cast<double>(freq_factors[j]);
        const double angle = static_cast<double>(pos) * freq;
        const float c = static_cast<float>(std::cos(angle));
        const float s = static_cast<float>(std::sin(angle));

        const float a = x[j];
        const float b = x[j + half];
        x[j]        = a * c - b * s;
        x[j + half] = b * c + a * s;
    }
}

void rope(std::span<float> x, std::size_t head_dim, std::size_t pos, float theta) {
    rope_scaled(x, head_dim, pos, theta, {});
}

void rope_interleaved(std::span<float> x, std::size_t head_dim, std::size_t pos,
                      float theta, std::span<const float> freq_factors) {
    if (x.size() != head_dim || (head_dim & 1U)) {
        throw std::runtime_error("rope: head_dim must be even and match x");
    }
    const std::size_t half = head_dim / 2;
    if (!freq_factors.empty() && freq_factors.size() != half) {
        throw std::runtime_error("rope: freq_factors length must be head_dim/2");
    }
    for (std::size_t i = 0; i < half; ++i) {
        double freq =
            std::pow(static_cast<double>(theta),
                     -2.0 * static_cast<double>(i) / static_cast<double>(head_dim));
        if (!freq_factors.empty()) freq /= static_cast<double>(freq_factors[i]);
        const double angle = static_cast<double>(pos) * freq;
        const float c = static_cast<float>(std::cos(angle));
        const float s = static_cast<float>(std::sin(angle));

        const float a = x[2 * i];
        const float b = x[2 * i + 1];
        x[2 * i]     = a * c - b * s;
        x[2 * i + 1] = a * s + b * c;
    }
}

void softmax(std::span<float> x) {
    if (x.empty()) return;

    float max_v = x[0];
    for (float v : x) max_v = v > max_v ? v : max_v;

    double sum = 0.0;
    for (float& v : x) {
        const float e = std::exp(v - max_v);
        v = e;
        sum += e;
    }
    const float inv = static_cast<float>(1.0 / sum);
    for (float& v : x) v *= inv;
}

void swiglu(std::span<float> out, std::span<const float> gate,
            std::span<const float> up) {
    if (gate.size() != up.size() || out.size() != gate.size()) {
        throw std::runtime_error("swiglu: shape mismatch");
    }
    for (std::size_t i = 0; i < gate.size(); ++i) {
        const float g = gate[i];
        const float silu = g / (1.0f + std::exp(-g));  // g * sigmoid(g)
        out[i] = silu * up[i];
    }
}

void residual_add(std::span<float> out, std::span<const float> x) {
    if (out.size() != x.size()) throw std::runtime_error("residual_add: shape mismatch");
    for (std::size_t i = 0; i < out.size(); ++i) out[i] += x[i];
}

}  // namespace damascus::cpu
