#include "kernels.hpp"
#include "quant.hpp"

#include <cmath>
#include <stdexcept>
#include <thread>
#include <vector>

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

void matmul_q(std::span<float> out, std::span<const float> x,
              std::uint32_t wtype, const std::byte* w, std::size_t rows,
              std::size_t n, std::size_t k) {
    if (x.size() != rows * k || out.size() != rows * n) {
        throw std::runtime_error("matmul_q: shape mismatch");
    }
    const std::size_t rb = quant::row_bytes(wtype, k);

    // Each output row j is an independent dequant+dot, so rows can be split
    // across threads without touching any dot's accumulation order.
    // NOTE: 8 independent fp64 accumulators break the serial add dependency
    // (and let clang emit 2-lane fp64 NEON FMAs). Reordering the summation
    // perturbs logits by ~1 ulp vs the scalar reference — NOT bit-identical,
    // but verified byte-identical on --predict output and greedy generations.
    auto run_rows = [&](std::size_t j0, std::size_t j1) {
        std::vector<float> wrow(k);
        for (std::size_t j = j0; j < j1; ++j) {
            quant::dequantize_to(wtype, w + j * rb, k, wrow.data());
            for (std::size_t m = 0; m < rows; ++m) {
                const float* xr = x.data() + m * k;
                double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
                double a4 = 0.0, a5 = 0.0, a6 = 0.0, a7 = 0.0;
                std::size_t i = 0;
                for (; i + 8 <= k; i += 8) {
                    a0 += static_cast<double>(xr[i + 0]) * wrow[i + 0];
                    a1 += static_cast<double>(xr[i + 1]) * wrow[i + 1];
                    a2 += static_cast<double>(xr[i + 2]) * wrow[i + 2];
                    a3 += static_cast<double>(xr[i + 3]) * wrow[i + 3];
                    a4 += static_cast<double>(xr[i + 4]) * wrow[i + 4];
                    a5 += static_cast<double>(xr[i + 5]) * wrow[i + 5];
                    a6 += static_cast<double>(xr[i + 6]) * wrow[i + 6];
                    a7 += static_cast<double>(xr[i + 7]) * wrow[i + 7];
                }
                double acc = ((a0 + a4) + (a1 + a5)) + ((a2 + a6) + (a3 + a7));
                for (; i < k; ++i) {
                    acc += static_cast<double>(xr[i]) * wrow[i];
                }
                out[m * n + j] = static_cast<float>(acc);
            }
        }
    };

    const std::size_t hw = std::thread::hardware_concurrency();
    const std::size_t nt = std::min<std::size_t>(hw ? hw : 1, n);
    if (nt <= 1 || rows * n * k < (1u << 20)) {
        run_rows(0, n);
        return;
    }

    std::vector<std::thread> pool;
    pool.reserve(nt);
    const std::size_t chunk = (n + nt - 1) / nt;
    for (std::size_t t = 0; t < nt; ++t) {
        const std::size_t j0 = t * chunk;
        const std::size_t j1 = std::min(n, j0 + chunk);
        if (j0 >= j1) break;
        pool.emplace_back(run_rows, j0, j1);
    }
    for (auto& th : pool) th.join();
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
