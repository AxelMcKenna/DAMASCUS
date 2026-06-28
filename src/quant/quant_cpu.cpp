#include "quant.hpp"

#include <cstring>
#include <stdexcept>

namespace damascus::quant {

namespace {

constexpr int QK_K = 256;

uint16_t read_u16(const std::byte* p) {
    uint16_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

// ggml block_q4_K: d(f16) dmin(f16) scales[12] qs[128]  -> 256 weights, 144 B.
// 6-bit scale + 6-bit min per 32-weight sub-block, packed in scales[12].
void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

void dequantize_q4_K(const std::byte* src, std::uint64_t n, float* out) {
    const std::uint64_t nb = n / QK_K;
    for (std::uint64_t i = 0; i < nb; ++i) {
        const std::byte* blk = src + i * 144;
        const float d    = half_to_float(read_u16(blk));
        const float dmin = half_to_float(read_u16(blk + 2));
        const auto* scales = reinterpret_cast<const uint8_t*>(blk + 4);
        const auto* q      = reinterpret_cast<const uint8_t*>(blk + 16);

        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, sc, m);
            const float d1 = d * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, sc, m);
            const float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32; ++l) *out++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *out++ = d2 * (q[l] >> 4)  - m2;
            q += 32;
            is += 2;
        }
    }
}

// ggml block_q6_K: ql[128] qh[64] scales[16](int8) d(f16) -> 256 weights, 210 B.
void dequantize_q6_K(const std::byte* src, std::uint64_t n, float* out) {
    const std::uint64_t nb = n / QK_K;
    for (std::uint64_t i = 0; i < nb; ++i) {
        const std::byte* blk = src + i * 210;
        const auto* ql = reinterpret_cast<const uint8_t*>(blk);
        const auto* qh = reinterpret_cast<const uint8_t*>(blk + 128);
        const auto* sc = reinterpret_cast<const int8_t*>(blk + 192);
        const float d  = half_to_float(read_u16(blk + 208));

        float* y = out + i * QK_K;
        for (int n0 = 0; n0 < QK_K; n0 += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int8_t q1 = static_cast<int8_t>((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = static_cast<int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = static_cast<int8_t>((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = static_cast<int8_t>((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

}  // namespace

float half_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 1;
            while ((mant & 0x400) == 0) { mant <<= 1; --exp; }
            mant &= 0x3FF;
            f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }

    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

std::vector<float> dequantize(uint32_t ggml_type, const std::byte* src,
                              std::uint64_t n_elements) {
    std::vector<float> out(n_elements);

    switch (ggml_type) {
        case GGML_F32:
            std::memcpy(out.data(), src, n_elements * sizeof(float));
            break;
        case GGML_F16:
            for (std::uint64_t i = 0; i < n_elements; ++i)
                out[i] = half_to_float(read_u16(src + i * 2));
            break;
        case GGML_Q4_K:
            if (n_elements % QK_K) throw std::runtime_error("q4_K: n % 256 != 0");
            dequantize_q4_K(src, n_elements, out.data());
            break;
        case GGML_Q6_K:
            if (n_elements % QK_K) throw std::runtime_error("q6_K: n % 256 != 0");
            dequantize_q6_K(src, n_elements, out.data());
            break;
        default:
            throw std::runtime_error("dequantize: unsupported ggml type " +
                                     std::to_string(ggml_type));
    }
    return out;
}

}  // namespace damascus::quant
