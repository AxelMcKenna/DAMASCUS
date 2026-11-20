// Metal fused dequant matmul + full-GPU decode step. Obj-C++ isolated here
// like metal_ctx.mm. Shader source is embedded so there is no metallib/build
// dance; it is compiled once at init (~100 ms).
//
// Two levels of GPU use:
//   1. gpu::matmul  — one fused dequant-GEMV/GEMM dispatch, synchronous.
//      Used by the CPU-orchestrated forward()/forward_chunk() (prefill).
//   2. gpu::decode_step — one WHOLE token step (28 layers: rmsnorm, QKV,
//      RoPE, attention over the cached KV, SwiGLU FFN, residuals, LM head)
//      encoded into a single command buffer: one commit+wait per token
//      instead of ~253. The KV cache lives in a shared MTLBuffer that the
//      CPU prefill writes into directly (unified memory).

#include "gpu.hpp"
#include <unistd.h>
#include "forward.hpp"
#include "quant.hpp"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

namespace damascus::gpu {

namespace {

constexpr std::uint32_t Q4K = 12;
constexpr std::uint32_t Q6K = 14;
constexpr std::uint32_t MAX_GPU_CTX = max_decode_context;  // threadgroup scores[] size

struct MMParams {
    std::uint32_t n;
    std::uint32_t k;
    std::uint32_t rows;
    std::uint32_t mode;  // 0: store, 1: y += acc (residual), 2: y = silu(y)*acc
    std::uint64_t w_off;
};

struct RopeFusedParams {
    std::uint32_t n_head;
    std::uint32_t n_kv;
    std::uint32_t head_dim;
    std::uint32_t pos;
    std::uint32_t kv_dim;
    std::uint32_t pad;
    std::uint64_t row_off;  // float offset of this position's cache row
};

struct NormParams {
    std::uint32_t d;
    float eps;
    std::uint64_t w_off;
};

struct RopeParams {
    std::uint32_t n_head;
    std::uint32_t head_dim;
    std::uint32_t pos;
    std::uint32_t pad;
    std::uint64_t dst_off;  // float offset of the cache row (rope_k_store)
};

struct AttnParams {
    std::uint32_t n_rep;
    std::uint32_t head_dim;
    std::uint32_t pos;
    std::uint32_t kv_dim;
    std::uint64_t layer_off;  // float offset of this layer's cache slab
    float scale;
    std::uint32_t S;          // chunk rows (flash prefill kernel only)
};

struct EmbedParams {
    std::uint32_t d;
    std::uint32_t slot;       // index into the token ring buffer
    std::uint64_t embd_off;
};

struct ArgmaxParams {
    std::uint32_t n;
    std::uint32_t dst;        // token ring slot to write
};

struct DqParams {
    std::uint64_t w_off;
    std::uint32_t n;
    std::uint32_t pad;
};

const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct MMParams  { uint n; uint k; uint rows; uint mode; ulong w_off; };
struct RopeFusedParams {
    uint n_head; uint n_kv; uint head_dim; uint pos; uint kv_dim; uint pad;
    ulong row_off;
};

// Epilogue shared by the gemv kernels: plain store, fused residual add, or
// fused SwiGLU (y holds the gate vector, acc is the up projection).
inline void store_mode(device float* y, ulong idx, float acc, uint mode) {
    if (mode == 1) {
        y[idx] += acc;
    } else if (mode == 2) {
        const float g = y[idx];
        y[idx] = (g / (1.0f + exp(-g))) * acc;
    } else {
        y[idx] = acc;
    }
}
struct NormParams{ uint d; float eps; ulong w_off; };
struct RopeParams{ uint n_head; uint head_dim; uint pos; uint pad; ulong dst_off; };
struct AttnParams{ uint n_rep; uint head_dim; uint pos; uint kv_dim; ulong layer_off; float scale; uint S; };

// ---- fused dequant GEMV/GEMM --------------------------------------------
// block_q4_K: d(half) dmin(half) scales[12] qs[128] = 144 B, 256 weights.
inline void scale_min_k4(uint s, device const uchar* q,
                         thread float& d, thread float& m) {
    if (s < 4) {
        d = float(q[s] & 63);
        m = float(q[s + 4] & 63);
    } else {
        d = float((q[s + 4] & 0xF) | ((q[s - 4] >> 6) << 4));
        m = float((q[s + 4] >> 4)  | ((q[s]     >> 6) << 4));
    }
}

kernel void gemv_q4k(device const uchar* blob [[buffer(0)]],
                     device const float* x    [[buffer(1)]],
                     device float*       y    [[buffer(2)]],
                     constant MMParams&  p    [[buffer(3)]],
                     uint2 tg   [[threadgroup_position_in_grid]],
                     uint  sid  [[simdgroup_index_in_threadgroup]],
                     uint  lane [[thread_index_in_simdgroup]]) {
    const uint j = tg.x * 4 + sid;   // 4 simdgroups per tg, one row each
    if (j >= p.n) return;
    const uint m = tg.y;
    const uint nb = p.k >> 8;
    device const uchar* wrow = blob + p.w_off + ulong(j) * ulong(nb) * 144;
    device const float* xr   = x + ulong(m) * ulong(p.k);

    float acc = 0.0f;
    // Each lane owns a 64-element pair: both nibble halves of the same 32 q
    // bytes, so every weight byte is loaded exactly once.
    const uint npair = p.k >> 6;
    for (uint c = lane; c < npair; c += 32) {
        const uint b  = c >> 2;                    // super-block
        const uint pr = c & 3;                     // 64-element pair within it
        device const uchar* blk = wrow + b * 144;
        const float d    = float(*(device const half*)(blk));
        const float dmin = float(*(device const half*)(blk + 2));
        float sd0, sm0, sd1, sm1;
        scale_min_k4(2 * pr,     blk + 4, sd0, sm0);
        scale_min_k4(2 * pr + 1, blk + 4, sd1, sm1);
        device const uchar4* q   = (device const uchar4*)(blk + 16 + pr * 32);
        device const float4* xlo = (device const float4*)(xr + b * 256 + pr * 64);
        device const float4* xhi = xlo + 8;

        float4 sq0 = 0.0f, sx0 = 0.0f, sq1 = 0.0f, sx1 = 0.0f;
        for (uint l = 0; l < 8; ++l) {
            const uchar4 qq = q[l];
            const float4 xl = xlo[l];
            const float4 xh = xhi[l];
            sq0 += xl * float4(qq & 0xF);
            sx0 += xl;
            sq1 += xh * float4(qq >> 4);
            sx1 += xh;
        }
        acc += (d * sd0) * (sq0.x + sq0.y + sq0.z + sq0.w)
             - (dmin * sm0) * (sx0.x + sx0.y + sx0.z + sx0.w)
             + (d * sd1) * (sq1.x + sq1.y + sq1.z + sq1.w)
             - (dmin * sm1) * (sx1.x + sx1.y + sx1.z + sx1.w);
    }
    acc = simd_sum(acc);
    if (lane == 0) store_mode(y, ulong(m) * ulong(p.n) + j, acc, p.mode);
}

// block_q6_K: ql[128] qh[64] scales[16](int8) d(half @208) = 210 B, 256 w.
kernel void gemv_q6k(device const uchar* blob [[buffer(0)]],
                     device const float* x    [[buffer(1)]],
                     device float*       y    [[buffer(2)]],
                     constant MMParams&  p    [[buffer(3)]],
                     uint2 tg   [[threadgroup_position_in_grid]],
                     uint  sid  [[simdgroup_index_in_threadgroup]],
                     uint  lane [[thread_index_in_simdgroup]]) {
    const uint j = tg.x * 4 + sid;
    if (j >= p.n) return;
    const uint m = tg.y;
    const uint nb = p.k >> 8;
    device const uchar* wrow = blob + p.w_off + ulong(j) * ulong(nb) * 210;
    device const float* xr   = x + ulong(m) * ulong(p.k);

    float acc = 0.0f;
    const uint nhalf = p.k >> 7;
    for (uint c = lane; c < nhalf; c += 32) {
        const uint b = c >> 1;
        const uint h = c & 1;
        device const uchar* blk = wrow + b * 210;
        device const uchar* ql  = blk + h * 64;
        device const uchar* qh  = blk + 128 + h * 32;
        device const char*  sc  = (device const char*)(blk + 192) + h * 8;
        const float d = float(*(device const half*)(blk + 208));
        device const float* xv = xr + b * 256 + h * 128;

        // 210-byte blocks are only 2-aligned -> packed vector loads.
        device const packed_uchar4* pl = (device const packed_uchar4*)ql;
        device const packed_uchar4* ph = (device const packed_uchar4*)qh;
        device const float4*        x4 = (device const float4*)xv;
        float4 a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f;
        for (uint g = 0; g < 2; ++g) {                 // scale group (is = g)
            const float s1 = float(sc[g]),     s2 = float(sc[g + 2]);
            const float s3 = float(sc[g + 4]), s4 = float(sc[g + 6]);
            float4 t1 = 0.0f, t2 = 0.0f, t3 = 0.0f, t4 = 0.0f;
            for (uint v = 0; v < 4; ++v) {
                const uint l4 = g * 4 + v;             // float4 index (l = 4*l4)
                const uchar4 lo = uchar4(pl[l4]);
                const uchar4 lo32 = uchar4(pl[l4 + 8]);
                const uchar4 hi = uchar4(ph[l4]);
                t1 += x4[l4]      * float4(int4((lo   & 0xF) | ((hi >> 0 & 3) << 4)) - 32);
                t2 += x4[l4 + 8]  * float4(int4((lo32 & 0xF) | ((hi >> 2 & 3) << 4)) - 32);
                t3 += x4[l4 + 16] * float4(int4((lo   >>  4) | ((hi >> 4 & 3) << 4)) - 32);
                t4 += x4[l4 + 24] * float4(int4((lo32 >>  4) | ((hi >> 6 & 3) << 4)) - 32);
            }
            a1 += s1 * t1; a2 += s2 * t2; a3 += s3 * t3; a4 += s4 * t4;
        }
        const float4 at = a1 + a2 + a3 + a4;
        acc += d * (at.x + at.y + at.z + at.w);
    }
    acc = simd_sum(acc);
    if (lane == 0) store_mode(y, ulong(m) * ulong(p.n) + j, acc, p.mode);
}

#if DAMASCUS_APPLE_MATRIX
// Each SIMD group multiplies eight output features by one or two tiles
// of eight prompt rows.
// Groups within a threadgroup partition K and reduce once after the dot.
constant bool prompt_q4 [[function_constant(0)]];
constant bool prompt_second_tile [[function_constant(1)]];

kernel void prompt_gemv(device const uchar* blob [[buffer(0)]],
                        device const float* x [[buffer(1)]],
                        device float* y [[buffer(2)]],
                        constant MMParams& p [[buffer(3)]],
                        uint2 tg [[threadgroup_position_in_grid]],
                        uint sid [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    // Apple 8x8 matrix fragments assign two adjacent columns to each lane.
    const uint m = ((lane / 4) & 4) + ((lane / 2) % 4);
    const uint n = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const uint feature = tg.x * 8 + m;
    // The right edge of a tile repeats the last valid input row. Only
    // in-range results are written after the split-K reduction.
    const uint row = n;
    const uint blocks = p.k / 256, bytes = prompt_q4 ? 144 : 210;
    device const uchar* weights = blob + p.w_off + ulong(min(feature, p.n - 1)) * blocks * bytes;
    device const float* x0 = x + ulong(min(row, p.rows - 1)) * p.k;
    device const float* x1 = x + ulong(min(row + 1, p.rows - 1)) * p.k;
    device const float* x8 = x + ulong(min(row + 8, p.rows - 1)) * p.k;
    device const float* x9 = x + ulong(min(row + 9, p.rows - 1)) * p.k;

    simdgroup_float8x8 sum = make_filled_simdgroup_matrix<float, 8>(0);
    simdgroup_float8x8 sum2 = make_filled_simdgroup_matrix<float, 8>(0);
    for (uint segment = sid; segment < p.k / 64; segment += 4) {
        device const uchar* w = weights + segment / 4 * bytes;
        const uint pair = segment % 4;
        float scales[2], minima[2];
        if (prompt_q4) {
            for (uint h = 0; h < 2; ++h) {
                scale_min_k4(2 * pair + h, w + 4, scales[h], minima[h]);
                scales[h] *= float(*(device const half*)w);
                minima[h] *= float(*(device const half*)(w + 2));
            }
        }
        // Each segment covers a pair of 32-weight quantization groups.
        #pragma clang loop unroll(full)
        for (uint s = 0; s < 8; ++s) {
            const uint index = 8 * s + n, within = index % 32;
            float2 a;
            if (prompt_q4) {
                uchar2 q = *(device const uchar2*)(w + 16 + pair * 32 + within);
                a = float2((q >> (4 * (s / 4))) & 15) * scales[s / 4] - minima[s / 4];
            } else {
                uint h = pair / 2, g = 2 * (pair % 2) + s / 4;
                uchar2 lo = *(device const uchar2*)(w + h * 64 + (g % 2) * 32 + within);
                uchar2 hi = *(device const uchar2*)(w + 128 + h * 32 + within);
                int2 q = int2(((lo >> (4 * (g / 2))) & 15) | (((hi >> (2 * g)) & 3) << 4)) - 32;
                float scale = float(*(device const half*)(w + 208)) *
                              float(((device const char*)(w + 192))[h * 8 + 2 * g + within / 16]);
                a = float2(q) * scale;
            }
            simdgroup_float8x8 wa, xb;
            wa.thread_elements()[0] = a.x;
            wa.thread_elements()[1] = a.y;
            xb.thread_elements()[0] = x0[segment * 64 + 8 * s + m];
            xb.thread_elements()[1] = x1[segment * 64 + 8 * s + m];
            simdgroup_multiply_accumulate(sum, wa, xb, sum);
            if (prompt_second_tile) {
                xb.thread_elements()[0] = x8[segment * 64 + 8 * s + m];
                xb.thread_elements()[1] = x9[segment * 64 + 8 * s + m];
                simdgroup_multiply_accumulate(sum2, wa, xb, sum2);
            }
        }
    }
    threadgroup float4 partial[4][32];
    partial[sid][lane] = float4(sum.thread_elements()[0], sum.thread_elements()[1],
                                sum2.thread_elements()[0], sum2.thread_elements()[1]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sid == 0 && feature < p.n) {
        float4 v = partial[0][lane] + partial[1][lane] + partial[2][lane] + partial[3][lane];
        if (row < p.rows) store_mode(y, ulong(row) * p.n + feature, v.x, p.mode);
        if (row + 1 < p.rows) store_mode(y, ulong(row + 1) * p.n + feature, v.y, p.mode);
        if (row + 8 < p.rows) store_mode(y, ulong(row + 8) * p.n + feature, v.z, p.mode);
        if (row + 9 < p.rows) store_mode(y, ulong(row + 9) * p.n + feature, v.w, p.mode);
    }
}

#endif

// Fused per-token RoPE + KV store: threads [0, n_head*hd/2) rotate q pairs in
// place; [.., +n_kv*hd/2) rotate k pairs into the cache row; [.., +kv_dim)
// copy v into the cache row.
// grid.y = chunk row; p.pos is the row-0 absolute position.
kernel void rope_kv(device float*       qin    [[buffer(0)]],
                    device const float* ktin   [[buffer(1)]],
                    device const float* vtin   [[buffer(2)]],
                    device float*       kcache [[buffer(3)]],
                    device float*       vcache [[buffer(4)]],
                    device const float* fr     [[buffer(5)]],
                    constant RopeFusedParams& p [[buffer(6)]],
                    uint2 tp [[thread_position_in_grid]]) {
    const uint t = tp.x;
    const uint row = tp.y;
    const uint pos = p.pos + row;
    const uint half_hd = p.head_dim >> 1;
    const uint nq = p.n_head * half_hd;
    const uint nk = p.n_kv * half_hd;
    device float*       q  = qin  + ulong(row) * p.n_head * p.head_dim;
    device const float* kt = ktin + ulong(row) * p.kv_dim;
    device const float* vt = vtin + ulong(row) * p.kv_dim;
    const ulong row_off = p.row_off + ulong(row) * p.kv_dim;
    if (t < nq) {
        const uint i = t % half_hd;
        const float angle = float(pos) * fr[i];
        const float c = cos(angle), s = sin(angle);
        device float* v = q + (t / half_hd) * p.head_dim + 2 * i;
        const float a = v[0], b = v[1];
        v[0] = a * c - b * s;
        v[1] = a * s + b * c;
    } else if (t < nq + nk) {
        const uint tk = t - nq;
        const uint i = tk % half_hd;
        const float angle = float(pos) * fr[i];
        const float c = cos(angle), s = sin(angle);
        const uint idx = (tk / half_hd) * p.head_dim + 2 * i;
        const float a = kt[idx], b = kt[idx + 1];
        kcache[row_off + idx]     = a * c - b * s;
        kcache[row_off + idx + 1] = a * s + b * c;
    } else if (t < nq + nk + p.kv_dim) {
        const uint tv = t - nq - nk;
        vcache[row_off + tv] = vt[tv];
    }
}

// ---- decode-step kernels -------------------------------------------------

// One threadgroup per row (grid.y); norm weight is an F32 tensor read
// straight from the blob.
kernel void rmsnorm_g(device const float* xin  [[buffer(0)]],
                      device float*       yout [[buffer(1)]],
                      device const uchar* blob [[buffer(2)]],
                      constant NormParams& p   [[buffer(3)]],
                      uint2 tg   [[threadgroup_position_in_grid]],
                      uint2 tid2 [[thread_position_in_threadgroup]],
                      uint2 tsz2 [[threads_per_threadgroup]],
                      uint sid  [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    const uint tid = tid2.x, tsz = tsz2.x;
    device const float* w = (device const float*)(blob + p.w_off);
    device const float* x = xin  + ulong(tg.y) * p.d;
    device float*       y = yout + ulong(tg.y) * p.d;
    threadgroup float red[32];
    threadgroup float inv;

    float s = 0.0f;
    for (uint i = tid; i < p.d; i += tsz) s += x[i] * x[i];
    s = simd_sum(s);
    if (lane == 0) red[sid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sid == 0) {
        float t = (lane < (tsz + 31) / 32) ? red[lane] : 0.0f;
        t = simd_sum(t);
        if (lane == 0) inv = rsqrt(t / float(p.d) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < p.d; i += tsz) y[i] = x[i] * inv * w[i];
}

// Interleaved RoPE on q, in place. One thread per (head, pair).
kernel void rope_q(device float*       q  [[buffer(0)]],
                   device const float* fr [[buffer(1)]],   // inv_freq[hd/2]
                   constant RopeParams& p [[buffer(2)]],
                   uint t [[thread_position_in_grid]]) {
    const uint half_hd = p.head_dim >> 1;
    const uint h = t / half_hd;
    const uint i = t % half_hd;
    if (h >= p.n_head) return;
    const float angle = float(p.pos) * fr[i];
    const float c = cos(angle), s = sin(angle);
    device float* v = q + h * p.head_dim + 2 * i;
    const float a = v[0], b = v[1];
    v[0] = a * c - b * s;
    v[1] = a * s + b * c;
}

// Interleaved RoPE on the fresh k vector, written into its cache row.
kernel void rope_k_store(device const float* kt [[buffer(0)]],
                         device float*   kcache [[buffer(1)]],
                         device const float* fr [[buffer(2)]],
                         constant RopeParams& p [[buffer(3)]],
                         uint t [[thread_position_in_grid]]) {
    const uint half_hd = p.head_dim >> 1;
    const uint h = t / half_hd;
    const uint i = t % half_hd;
    if (h >= p.n_head) return;   // n_head = n_kv heads here
    const float angle = float(p.pos) * fr[i];
    const float c = cos(angle), s = sin(angle);
    const uint idx = h * p.head_dim + 2 * i;
    const float a = kt[idx], b = kt[idx + 1];
    kcache[p.dst_off + idx]     = a * c - b * s;
    kcache[p.dst_off + idx + 1] = a * s + b * c;
}

kernel void copy_off(device const float* src [[buffer(0)]],
                     device float*       dst [[buffer(1)]],
                     constant ulong& dst_off [[buffer(2)]],
                     uint t [[thread_position_in_grid]]) {
    dst[dst_off + t] = src[t];
}

// One threadgroup per query head: scores over cached K (0..pos), softmax in
// threadgroup memory, weighted sum over cached V.
kernel void attn_head(device const float* q      [[buffer(0)]],
                      device const float* kcache [[buffer(1)]],
                      device const float* vcache [[buffer(2)]],
                      device float*       out    [[buffer(3)]],
                      constant AttnParams& p     [[buffer(4)]],
                      uint2 tg   [[threadgroup_position_in_grid]],
                      uint2 tid2 [[thread_position_in_threadgroup]],
                      uint2 tsz2 [[threads_per_threadgroup]]) {
    threadgroup float sc[2048];
    threadgroup float red[128];
    const uint tid = tid2.x, tsz = tsz2.x;
    const uint h   = tg.x;
    const uint row = tg.y;                     // chunk row; p.pos = row-0 pos
    const uint n_head = p.kv_dim * p.n_rep / p.head_dim;
    device const float* qh = q + ulong(row) * n_head * p.head_dim
                               + h * p.head_dim;
    const uint kvh = h / p.n_rep;
    const uint ctx = p.pos + row + 1;          // causal: keys 0..pos+row

    for (uint j = tid; j < ctx; j += tsz) {
        device const float* kv =
            kcache + p.layer_off + ulong(j) * p.kv_dim + kvh * p.head_dim;
        float dot = 0.0f;
        for (uint e = 0; e < p.head_dim; ++e) dot += qh[e] * kv[e];
        sc[j] = dot * p.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mx = -INFINITY;
    for (uint j = tid; j < ctx; j += tsz) mx = max(mx, sc[j]);
    red[tid] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s2 = tsz >> 1; s2 > 0; s2 >>= 1) {
        if (tid < s2) red[tid] = max(red[tid], red[tid + s2]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    mx = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum = 0.0f;
    for (uint j = tid; j < ctx; j += tsz) {
        const float e = exp(sc[j] - mx);
        sc[j] = e;
        sum += e;
    }
    red[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s2 = tsz >> 1; s2 > 0; s2 >>= 1) {
        if (tid < s2) red[tid] += red[tid + s2];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = 1.0f / red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = tid; e < p.head_dim; e += tsz) {
        float acc = 0.0f;
        for (uint j = 0; j < ctx; ++j)
            acc += sc[j] * vcache[p.layer_off + ulong(j) * p.kv_dim
                                  + kvh * p.head_dim + e];
        out[ulong(row) * n_head * p.head_dim + h * p.head_dim + e] = acc * inv;
    }
}

// One SIMD group per query head and 32-key partition. Lanes own four
// head coordinates each, so K/V reads within a SIMD group are contiguous.
kernel void decode_attention_128(device const float4* q [[buffer(0)]],
                                 device const float4* k [[buffer(1)]],
                                 device const float4* v [[buffer(2)]],
                                 device float* partial [[buffer(3)]],
                                 constant AttnParams& p [[buffer(4)]],
                                 uint3 tg [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    const uint head = tg.x, first = tg.y * 32;
    const uint ctx = p.pos + tg.z + 1;
    const uint count = first < ctx ? min(32u, ctx - first) : 0;
    const uint heads = p.kv_dim * p.n_rep / p.head_dim;
    const ulong offset = p.layer_off / 4 + (head / p.n_rep) * 32 + lane;
    const float4 query = q[(tg.z * heads + head) * 32 + lane];
    float score = -INFINITY;
    for (uint i = 0; i < count; ++i) {
        const float d = simd_sum(dot(query, k[offset + ulong(first + i) * (p.kv_dim / 4)])) * p.scale;
        if (i == lane) score = d;
    }
    const float maximum = simd_max(score);
    const float weight = count ? exp(score - maximum) : 0;
    const float total = simd_sum(weight);
    float4 ov = 0;
    for (uint i = 0; i < count; ++i)
        ov += simd_broadcast(weight, i) * v[offset + ulong(first + i) * (p.kv_dim / 4)];
    const uint parts = (p.pos + p.S + 31) / 32;
    device float* tile = partial + ((tg.z * heads + head) * parts + tg.y) * 132;
    ((device float4*)tile)[lane] = ov;
    if (lane == 0) { tile[128] = maximum; tile[129] = total; }
}

kernel void decode_attention_merge(device const float* partial [[buffer(0)]],
                                   device float4* output [[buffer(1)]],
                                   constant uint& parts [[buffer(2)]],
                                   uint head [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    device const float* tiles = partial + head * parts * 132;
    float maximum = -INFINITY;
    for (uint i = lane; i < parts; i += 32)
        maximum = max(maximum, tiles[i * 132 + 128]);
    maximum = simd_max(maximum);
    float total = 0;
    float4 out = 0;
    for (uint i = 0; i < parts; ++i) {
        device const float* t = tiles + i * 132;
        const float scale = exp(t[128] - maximum);
        out += ((device const float4*)t)[lane] * scale;
        total += t[129] * scale;
    }
    output[head * 32 + lane] = out / total;
}

struct EmbedParams { uint d; uint slot; ulong embd_off; };

// Dequantize row tokens[slot] of the Q6_K embedding table into x.
// One thread per element; the token id is read on-GPU so decode steps can
// chain inside one command buffer without CPU round-trips.
kernel void embed_q6k(device const uchar* blob   [[buffer(0)]],
                      device const int*   tokens [[buffer(1)]],
                      device float*       xout   [[buffer(2)]],
                      constant EmbedParams& p    [[buffer(3)]],
                      uint2 tp [[thread_position_in_grid]]) {
    const uint e = tp.x;
    if (e >= p.d) return;
    device float* x = xout + ulong(tp.y) * p.d;
    const uint row = uint(tokens[p.slot + tp.y]);
    const ulong row_off = p.embd_off + ulong(row) * ulong(p.d >> 8) * 210;
    const uint b  = e >> 8;          // super-block in row
    const uint r  = e & 255;
    const uint h  = r >> 7;          // 128-half
    const uint l2 = r & 127;
    const uint grp = l2 >> 5;        // q1..q4 group
    const uint l   = l2 & 31;
    device const uchar* blk = blob + row_off + b * 210;
    device const uchar* ql  = blk + h * 64;
    device const uchar* qh  = blk + 128 + h * 32;
    device const char*  sc  = (device const char*)(blk + 192) + h * 8;
    const float d = float(*(device const half*)(blk + 208));

    const uchar lo = (grp == 0 || grp == 2) ? ql[l] : ql[l + 32];
    const uchar nib = (grp < 2) ? (lo & 0xF) : (lo >> 4);
    const int  q  = int(nib | (((qh[l] >> (2 * grp)) & 3) << 4)) - 32;
    x[e] = d * float(sc[(l >> 4) + 2 * grp]) * float(q);
}

struct ArgmaxParams { uint n; uint dst; };

// Greedy sample on-GPU: first index of the max logit (same tie-break as the
// CPU argmax), written into tokens[dst]. One threadgroup.
kernel void argmax_g(device const float* logits [[buffer(0)]],
                     device int*         tokens [[buffer(1)]],
                     constant ArgmaxParams& p   [[buffer(2)]],
                     uint tid [[thread_position_in_threadgroup]],
                     uint tsz [[threads_per_threadgroup]]) {
    threadgroup float bv[256];
    threadgroup int   bi[256];
    float best = -INFINITY;
    int   idx  = 0;
    for (uint i = tid; i < p.n; i += tsz) {
        if (logits[i] > best) { best = logits[i]; idx = int(i); }
    }
    bv[tid] = best;
    bi[tid] = idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tsz >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            if (bv[tid + s] > bv[tid] ||
                (bv[tid + s] == bv[tid] && bi[tid + s] < bi[tid])) {
                bv[tid] = bv[tid + s];
                bi[tid] = bi[tid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) tokens[p.dst] = bi[0];
}

// ---- fp16 prefill support -------------------------------------------------

struct DqParams { ulong w_off; uint n; uint pad; };

// One-time on-GPU dequant of a whole Q4_K tensor to half (prefill weights).
kernel void dq_q4k_h(device const uchar* blob [[buffer(0)]],
                     device half*        dst  [[buffer(1)]],
                     constant DqParams&  p    [[buffer(2)]],
                     uint e [[thread_position_in_grid]]) {
    if (e >= p.n) return;
    const uint b  = e >> 8;
    const uint r  = e & 255;
    const uint p0 = r >> 6;          // 64-element pair
    const uint hi = (r >> 5) & 1;    // low/high nibble half
    const uint l  = r & 31;
    device const uchar* blk = blob + p.w_off + ulong(b) * 144;
    const float d    = float(*(device const half*)(blk));
    const float dmin = float(*(device const half*)(blk + 2));
    float sd, sm;
    scale_min_k4(2 * p0 + hi, blk + 4, sd, sm);
    const uchar qb = blk[16 + p0 * 32 + l];
    const float nib = float(hi ? (qb >> 4) : (qb & 0xF));
    dst[e] = half(d * sd * nib - dmin * sm);
}

kernel void dq_q6k_h(device const uchar* blob [[buffer(0)]],
                     device half*        dst  [[buffer(1)]],
                     constant DqParams&  p    [[buffer(2)]],
                     uint e [[thread_position_in_grid]]) {
    if (e >= p.n) return;
    const uint b   = e >> 8;
    const uint r   = e & 255;
    const uint h   = r >> 7;
    const uint l2  = r & 127;
    const uint grp = l2 >> 5;
    const uint l   = l2 & 31;
    device const uchar* blk = blob + p.w_off + ulong(b) * 210;
    device const uchar* ql  = blk + h * 64;
    device const uchar* qh  = blk + 128 + h * 32;
    device const char*  sc  = (device const char*)(blk + 192) + h * 8;
    const float d = float(*(device const half*)(blk + 208));
    const uchar lo = (grp == 0 || grp == 2) ? ql[l] : ql[l + 32];
    const uchar nib = (grp < 2) ? (lo & 0xF) : (lo >> 4);
    const int  q = int(nib | (((qh[l] >> (2 * grp)) & 3) << 4)) - 32;
    dst[e] = half(d * float(sc[(l >> 4) + 2 * grp]) * float(q));
}

// rmsnorm with half output (prefill GEMM feed).
kernel void rmsnorm_h(device const float* xin  [[buffer(0)]],
                      device half*        yout [[buffer(1)]],
                      device const uchar* blob [[buffer(2)]],
                      constant NormParams& p   [[buffer(3)]],
                      uint2 tg   [[threadgroup_position_in_grid]],
                      uint2 tid2 [[thread_position_in_threadgroup]],
                      uint2 tsz2 [[threads_per_threadgroup]],
                      uint sid  [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    const uint tid = tid2.x, tsz = tsz2.x;
    device const float* w = (device const float*)(blob + p.w_off);
    device const float* x = xin  + ulong(tg.y) * p.d;
    device half*        y = yout + ulong(tg.y) * p.d;
    threadgroup float red[32];
    threadgroup float inv;
    float s = 0.0f;
    for (uint i = tid; i < p.d; i += tsz) s += x[i] * x[i];
    s = simd_sum(s);
    if (lane == 0) red[sid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sid == 0) {
        float t = (lane < (tsz + 31) / 32) ? red[lane] : 0.0f;
        t = simd_sum(t);
        if (lane == 0) inv = rsqrt(t / float(p.d) + p.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < p.d; i += tsz) y[i] = half(x[i] * inv * w[i]);
}

// RoPE from a fused half QKV row ([q | k | v], stride p.pad): rotated q to
// fp32 for attention, rotated k and v into the fp32 caches.
kernel void rope_kv_h(device const half*  qkv    [[buffer(0)]],
                      device float*       qout   [[buffer(1)]],
                      device float*       kcache [[buffer(2)]],
                      device float*       vcache [[buffer(3)]],
                      device const float* fr     [[buffer(4)]],
                      constant RopeFusedParams& p [[buffer(5)]],
                      uint2 tp [[thread_position_in_grid]]) {
    const uint t = tp.x;
    const uint row = tp.y;
    const uint pos = p.pos + row;
    const uint half_hd = p.head_dim >> 1;
    const uint d = p.n_head * p.head_dim;
    const uint nq = p.n_head * half_hd;
    const uint nk = p.n_kv * half_hd;
    device const half* src = qkv + ulong(row) * p.pad;
    const ulong row_off = p.row_off + ulong(row) * p.kv_dim;
    if (t < nq) {
        const uint i = t % half_hd;
        const float angle = float(pos) * fr[i];
        const float c = cos(angle), s = sin(angle);
        const uint idx = (t / half_hd) * p.head_dim + 2 * i;
        const float a = float(src[idx]), b = float(src[idx + 1]);
        device float* dst = qout + ulong(row) * d + idx;
        dst[0] = a * c - b * s;
        dst[1] = a * s + b * c;
    } else if (t < nq + nk) {
        const uint tk = t - nq;
        const uint i = tk % half_hd;
        const float angle = float(pos) * fr[i];
        const float c = cos(angle), s = sin(angle);
        const uint idx = (tk / half_hd) * p.head_dim + 2 * i;
        const float a = float(src[d + idx]), b = float(src[d + idx + 1]);
        kcache[row_off + idx]     = a * c - b * s;
        kcache[row_off + idx + 1] = a * s + b * c;
    } else if (t < nq + nk + p.kv_dim) {
        const uint tv = t - nq - nk;
        vcache[row_off + tv] = float(src[d + p.kv_dim + tv]);
    }
}

// SwiGLU over a fused [gate | up] half row (stride 2*ff) -> compact [S, ff].
kernel void swiglu_cat(device const half* gu  [[buffer(0)]],
                       device half*       dst [[buffer(1)]],
                       constant uint2&    nf  [[buffer(2)]],  // {S*ff, ff}
                       uint t [[thread_position_in_grid]]) {
    if (t >= nf.x) return;
    const uint row = t / nf.y;
    const uint i   = t % nf.y;
    device const half* r = gu + ulong(row) * 2 * nf.y;
    const float g = float(r[i]);
    dst[t] = half((g / (1.0f + exp(-g))) * float(r[nf.y + i]));
}

kernel void cvt_f16(device const float* src [[buffer(0)]],
                    device half*        dst [[buffer(1)]],
                    constant uint&      n   [[buffer(2)]],
                    uint t [[thread_position_in_grid]]) {
    if (t < n) dst[t] = half(src[t]);
}

kernel void cvt_f32(device const half*  src [[buffer(0)]],
                    device float*       dst [[buffer(1)]],
                    constant uint&      n   [[buffer(2)]],
                    uint t [[thread_position_in_grid]]) {
    if (t < n) dst[t] = float(src[t]);
}

kernel void cvt_add_f16(device float*      x [[buffer(0)]],
                        device const half* h [[buffer(1)]],
                        constant uint&     n [[buffer(2)]],
                        uint t [[thread_position_in_grid]]) {
    if (t < n) x[t] += float(h[t]);
}

kernel void swiglu_h(device half*       gate [[buffer(0)]],
                     device const half* up   [[buffer(1)]],
                     constant uint&     n    [[buffer(2)]],
                     uint t [[thread_position_in_grid]]) {
    if (t >= n) return;
    const float g = float(gate[t]);
    gate[t] = half((g / (1.0f + exp(-g))) * float(up[t]));
}

// Flash-style prefill attention: one threadgroup per (head, tile of 8 query
// rows). K/V stream through threadgroup memory in 32-position chunks with
// online softmax, so each K/V element is read once per 8 queries instead of
// once per query. 128 threads = 4 simdgroups; simdgroup r handles query row
// r (two passes for 8 rows), lane = position within the chunk.
kernel void attn_flash(device const float* q      [[buffer(0)]],
                       device const float* kcache [[buffer(1)]],
                       device const float* vcache [[buffer(2)]],
                       device half*        out    [[buffer(3)]],
                       constant AttnParams& p     [[buffer(4)]],
                       uint2 tg   [[threadgroup_position_in_grid]],
                       uint2 tid2 [[thread_position_in_threadgroup]],
                       uint  sidx [[simdgroup_index_in_threadgroup]],
                       uint  lane [[thread_index_in_simdgroup]]) {
    constexpr uint QT = 8;    // query rows per tile
    constexpr uint CK = 32;   // kv positions per chunk
    constexpr uint KS = 132;  // kv row stride: 16B-aligned, breaks bank aliasing
    threadgroup float q8[QT * 128];
    threadgroup float o8[QT * 128];
    threadgroup float kv[CK * KS];
    threadgroup float sc8[QT * CK];
    threadgroup float m8[QT], l8[QT];

    const uint tid = tid2.x;
    const uint h   = tg.x;
    const uint i0  = tg.y * QT;                     // first tile row (chunk-rel)
    const uint n_head = p.kv_dim * p.n_rep / p.head_dim;
    const uint kvh = h / p.n_rep;
    const uint hd  = p.head_dim;

    // load q tile; zero accumulators
    for (uint t = tid; t < QT * hd; t += 128) {
        const uint r = t / hd;
        q8[t] = (i0 + r < p.S)
            ? q[ulong(i0 + r) * n_head * hd + h * hd + (t % hd)] : 0.0f;
        o8[t] = 0.0f;
    }
    if (tid < QT) { m8[tid] = -INFINITY; l8[tid] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint last_abs = p.pos + min(i0 + QT - 1, p.S - 1);  // causal bound
    for (uint j0 = 0; j0 <= last_abs; j0 += CK) {
        const uint jn = min(CK, last_abs + 1 - j0);

        // K chunk -> threadgroup
        for (uint t = tid; t < jn * hd; t += 128) {
            kv[(t / hd) * KS + (t % hd)] =
                kcache[p.layer_off + ulong(j0 + t / hd) * p.kv_dim
                       + kvh * hd + (t % hd)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // scores + online-softmax update, simdgroup per row, 2 passes
        for (uint pass = 0; pass < 2; ++pass) {
            const uint r = pass * 4 + sidx;
            const uint i_abs = p.pos + i0 + r;
            if (i0 + r < p.S) {
                float s = -INFINITY;
                const uint j = lane;
                if (j < jn && j0 + j <= i_abs) {
                    threadgroup const float4* q4 =
                        (threadgroup const float4*)(q8 + r * hd);
                    threadgroup const float4* k4 =
                        (threadgroup const float4*)(kv + j * KS);
                    float4 d4 = 0.0f;
                    for (uint e = 0; e < hd / 4; ++e) d4 += q4[e] * k4[e];
                    s = (d4.x + d4.y + d4.z + d4.w) * p.scale;
                }
                const float chunk_max = simd_max(s);
                const float m_old = m8[r];
                const float m_new = max(m_old, chunk_max);
                const float e_s = (s == -INFINITY) ? 0.0f : exp(s - m_new);
                const float corr = (m_old == -INFINITY) ? 0.0f
                                                        : exp(m_old - m_new);
                if (lane < CK) sc8[r * CK + lane] = e_s;
                const float sum = simd_sum(e_s);
                if (lane == 0) {
                    l8[r] = l8[r] * corr + sum;
                    m8[r] = m_new;
                }
                // rescale o by corr (lanes stride the head dim)
                for (uint e = lane; e < hd; e += 32) o8[r * hd + e] *= corr;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // V chunk -> threadgroup (reuse kv)
        for (uint t = tid; t < jn * hd; t += 128) {
            kv[(t / hd) * KS + (t % hd)] =
                vcache[p.layer_off + ulong(j0 + t / hd) * p.kv_dim
                       + kvh * hd + (t % hd)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint pass = 0; pass < 2; ++pass) {
            const uint r = pass * 4 + sidx;
            if (i0 + r < p.S) {
                for (uint e = lane; e < hd; e += 32) {
                    float acc = 0.0f;
                    for (uint j = 0; j < jn; ++j)
                        acc += sc8[r * CK + j] * kv[j * KS + e];
                    o8[r * hd + e] += acc;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint t = tid; t < QT * hd; t += 128) {
        const uint r = t / hd;
        if (i0 + r < p.S) {
            out[ulong(i0 + r) * n_head * hd + h * hd + (t % hd)] =
                half(o8[t] / l8[r]);
        }
    }
}

kernel void swiglu_g(device float*       gate [[buffer(0)]],
                     device const float* up   [[buffer(1)]],
                     constant uint&      n    [[buffer(2)]],
                     uint t [[thread_position_in_grid]]) {
    if (t >= n) return;
    const float g = gate[t];
    gate[t] = (g / (1.0f + exp(-g))) * up[t];
}

kernel void add_g(device float*       x [[buffer(0)]],
                  device const float* y [[buffer(1)]],
                  constant uint&      n [[buffer(2)]],
                  uint t [[thread_position_in_grid]]) {
    if (t >= n) return;
    x[t] += y[t];
}
)MSL";

// ---- host state -----------------------------------------------------------

id<MTLDevice>       g_dev;
id<MTLCommandQueue> g_queue;
id<MTLComputePipelineState> g_prompt[2][2];  // [Q4_K][second 8-row tile]
id<MTLComputePipelineState> g_attn_part, g_attn_merge;
id<MTLComputePipelineState> g_q4k, g_q6k, g_rms, g_ropeq, g_ropek, g_copy,
                            g_attn, g_swiglu, g_add, g_embed, g_argmax,
                            g_ropekv, g_dq4h, g_dq6h, g_cvt16, g_cvt32,
                            g_cvtadd, g_swigluh, g_attnf, g_rmsh, g_ropekvh,
                            g_swigluc;
id<MTLBuffer> g_tokens;          // int32 ring for on-GPU token chaining
std::size_t   g_tokens_cap = 0;
id<MTLBuffer> g_blob;
id<MTLBuffer> g_x, g_y;                 // scratch for synchronous matmul()
std::size_t   g_x_cap = 0, g_y_cap = 0;
const std::byte* g_blob_base = nullptr;
std::size_t      g_blob_size = 0;
bool g_ok = false, g_tried = false;

// kv_alloc registry: contents pointer -> buffer.
std::vector<std::pair<float*, id<MTLBuffer>>> g_kv_bufs;

// Per-model decode plan, built once by prepare().
struct LayerOffs {
    std::uint64_t attn_norm, ffn_norm;                  // F32
    std::uint64_t wq, wk, wv, wo, wgate, wup, wdown;
    std::uint32_t tq, tk, tv, to, tgate, tup, tdown;    // ggml types
};
constexpr std::uint32_t SHORT_PREFILL_ROWS = 16;

struct Layer16 {  // element offsets into the fp16 weight arena
    std::uint64_t q, k, v, o, gate, up, down;
};
struct DecodePlan {
    bool ready = false;
    std::uint32_t d = 0, n_head = 0, n_kv = 0, head_dim = 0, kv_dim = 0,
                  ff = 0, vocab = 0;
    float eps = 0.0f, scale = 0.0f;
    std::uint64_t out_norm = 0, embd = 0;
    std::uint32_t embd_type = 0;
    std::vector<LayerOffs> layers;
    id<MTLBuffer> inv_freq;
    // persistent activation buffers
    id<MTLBuffer> verify_logits;
    id<MTLBuffer> attn_parts;
    std::size_t parts_cap = 0;
    id<MTLBuffer> x, xb, xb2, q, kt, vt, attn, h1, h2, logits;

    // fp16 prefill state (built lazily by prepare16)
    bool w16_ready = false, w16_failed = false;
    id<MTLBuffer> w16;                 // all layer matmul weights, half
    std::vector<Layer16> l16;
    std::size_t  pf_cap = 0;           // rows the prefill buffers can hold
    id<MTLBuffer> pf_x, pf_xb, pf_q, pf_kt, pf_vt, pf_attn;      // fp32
    id<MTLBuffer> pf_xb16, pf_a16, pf_o16, pf_h16a, pf_h16b, pf_h16c; // fp16
};
DecodePlan g_plan;

// MPS GEMM cache keyed by (rows, n, k).
std::map<std::tuple<std::size_t, std::size_t, std::size_t>,
         MPSMatrixMultiplication*> g_mps;

id<MTLBuffer> ensure(id<MTLBuffer> buf, std::size_t& cap, std::size_t need) {
    if (buf && need <= cap) return buf;
    cap = need + need / 2;
    return [g_dev newBufferWithLength:cap options:MTLResourceStorageModeShared];
}

id<MTLComputePipelineState> pso_for(std::uint32_t t) {
    if (t == Q4K) return g_q4k;
    if (t == Q6K) return g_q6k;
    return nil;
}

id<MTLComputePipelineState> make_pso(id<MTLLibrary> lib, NSString* name) {
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (fn == nil) return nil;
    return [g_dev newComputePipelineStateWithFunction:fn error:&err];
}

std::uint64_t off_of(const WeightView& w) {
    return static_cast<std::uint64_t>(w.data - g_blob_base);
}

void enc_gemv(id<MTLComputeCommandEncoder> enc, std::uint32_t wtype,
              std::uint64_t w_off, id<MTLBuffer> x, id<MTLBuffer> y,
              std::uint32_t n, std::uint32_t k, std::uint32_t mode = 0,
              std::uint32_t rows = 1) {
    MMParams p{n, k, rows, mode, w_off};
    const bool batched = rows > 1;
    const bool q4 = wtype == Q4K;
    [enc setComputePipelineState:batched ? g_prompt[q4][rows > 8] : pso_for(wtype)];
    [enc setBuffer:g_blob offset:0 atIndex:0];
    [enc setBuffer:x offset:0 atIndex:1];
    [enc setBuffer:y offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(batched ? (n + 7) / 8 : (n + 3) / 4, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

void enc_rmsnorm(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x,
                 id<MTLBuffer> y, std::uint64_t w_off, std::uint32_t d,
                 float eps, std::uint32_t rows = 1, std::size_t x_offset = 0) {
    NormParams p{d, eps, w_off};
    [enc setComputePipelineState:g_rms];
    [enc setBuffer:x offset:x_offset atIndex:0];
    [enc setBuffer:y offset:0 atIndex:1];
    [enc setBuffer:g_blob offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(1, rows, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

id<MTLBuffer> find_kv(const float* p) {
    for (auto& [ptr, buf] : g_kv_bufs)
        if (ptr == p) return buf;
    return nil;
}

}  // namespace

// ---- public API -------------------------------------------------------------

static id<MTLBuffer> weight_buffer(const std::byte* blob, std::size_t size, const MappedFile* mapping) {
    id<MTLBuffer> buffer = nil;
    // mmap covers the final partial page. Keep the mapping alive until
    // Metal releases the buffer, including after the model is destroyed.
    if (mapping) {
        const MappedFile owner = *mapping;
        const auto page = static_cast<std::size_t>(getpagesize());
        const auto aligned = ((size - 1) / page + 1) * page;
        buffer = [g_dev newBufferWithBytesNoCopy:const_cast<std::byte*>(blob)
                                          length:aligned options:MTLResourceStorageModeShared
                                     deallocator:^(void*, NSUInteger) { (void)owner; }];
    }
    if (!buffer) {
        buffer = [g_dev newBufferWithBytes:blob length:size
                                   options:MTLResourceStorageModeShared];
    }
    if (buffer == nil) return nil;

#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
    if (@available(macOS 15.0, *)) {
        MTLResidencySetDescriptor* desc = [MTLResidencySetDescriptor new];
        desc.initialCapacity = 1;
        id<MTLResidencySet> weights = [g_dev newResidencySetWithDescriptor:desc error:nullptr];
        if (weights) {
            [weights addAllocation:buffer];
            [weights commit];
            [weights requestResidency];
            [g_queue addResidencySet:weights];
        }
    }
#endif

    return buffer;
}

static bool init_blob(const std::byte* blob, std::size_t size, const MappedFile* mapping) {
    if (!blob || size == 0) return false;
    if (g_tried) return g_ok && blob == g_blob_base;
    g_tried = true;

    const char* env = std::getenv("DAMASCUS_GPU");
    if (env != nullptr && std::strcmp(env, "0") == 0) return false;

    g_dev = MTLCreateSystemDefaultDevice();
    if (g_dev == nil) return false;
    g_queue = [g_dev newCommandQueue];

    g_blob = weight_buffer(blob, size, mapping);
    if (!g_blob) return false;

    NSError* err = nil;
    // The prompt kernel's register layout is specific to Apple GPU families.
    // Other Metal devices can still compile and use the existing MPS path.
    const bool apple_matrix = [g_dev supportsFamily:MTLGPUFamilyApple7];
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.preprocessorMacros = @{@"DAMASCUS_APPLE_MATRIX": @(apple_matrix)};
    id<MTLLibrary> lib =
        [g_dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                            options:options
                              error:&err];
    if (lib == nil) {
        std::fprintf(stderr, "gpu: shader compile failed: %s\n",
                     err ? [[err localizedDescription] UTF8String] : "?");
        return false;
    }
    if (apple_matrix) {
        for (bool q4 : {false, true}) {
            for (bool second : {false, true}) {
                MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
                [constants setConstantValue:&q4 type:MTLDataTypeBool atIndex:0];
                [constants setConstantValue:&second type:MTLDataTypeBool atIndex:1];
                id<MTLFunction> fn = [lib newFunctionWithName:@"prompt_gemv"
                                              constantValues:constants error:&err];
                if (fn)
                    g_prompt[q4][second] = [g_dev newComputePipelineStateWithFunction:fn error:&err];
            }
        }
    }
    g_q4k    = make_pso(lib, @"gemv_q4k");
    g_q6k    = make_pso(lib, @"gemv_q6k");
    g_rms    = make_pso(lib, @"rmsnorm_g");
    g_ropeq  = make_pso(lib, @"rope_q");
    g_ropek  = make_pso(lib, @"rope_k_store");
    g_copy   = make_pso(lib, @"copy_off");
    g_attn_part = make_pso(lib, @"decode_attention_128");
    g_attn_merge = make_pso(lib, @"decode_attention_merge");
    if (g_attn_part.threadExecutionWidth != 32 || g_attn_merge.threadExecutionWidth != 32)
        g_attn_part = g_attn_merge = nil;
    g_attn   = make_pso(lib, @"attn_head");
    g_swiglu = make_pso(lib, @"swiglu_g");
    g_add    = make_pso(lib, @"add_g");
    g_embed  = make_pso(lib, @"embed_q6k");
    g_argmax = make_pso(lib, @"argmax_g");
    g_ropekv  = make_pso(lib, @"rope_kv");
    g_dq4h    = make_pso(lib, @"dq_q4k_h");
    g_dq6h    = make_pso(lib, @"dq_q6k_h");
    g_cvt16   = make_pso(lib, @"cvt_f16");
    g_cvt32   = make_pso(lib, @"cvt_f32");
    g_cvtadd  = make_pso(lib, @"cvt_add_f16");
    g_swigluh = make_pso(lib, @"swiglu_h");
    g_attnf   = make_pso(lib, @"attn_flash");
    g_rmsh    = make_pso(lib, @"rmsnorm_h");
    g_ropekvh = make_pso(lib, @"rope_kv_h");
    g_swigluc = make_pso(lib, @"swiglu_cat");
    for (id p : @[g_q4k, g_q6k, g_rms, g_ropeq, g_ropek, g_copy, g_attn,
                  g_swiglu, g_add, g_embed, g_argmax, g_ropekv, g_dq4h,
                  g_dq6h, g_cvt16, g_cvt32, g_cvtadd, g_swigluh, g_attnf,
                  g_rmsh, g_ropekvh, g_swigluc])
        if (p == nil) return false;

    g_blob_base = blob;
    g_blob_size = size;
    g_ok = true;
    return true;
}

// Generation submits one model at a time and waits before switching. Metal
// pipelines and token scratch are shared. Weight offsets and activations are not.
static DecodePlan alternate_plan;
static id<MTLBuffer> alternate_blob;
static const std::byte* alternate_base = nullptr;
static std::size_t alternate_size = 0;

bool register_helper(const MappedFile& mapping) {
    if (!g_ok || alternate_base || mapping.data() == g_blob_base || !mapping.size()) return false;
    alternate_blob = weight_buffer(mapping.data(), mapping.size(), &mapping);
    if (!alternate_blob) return false;
    alternate_base = mapping.data();
    alternate_size = mapping.size();
    return true;
}

bool init(const MappedFile& mapping) {
    if (alternate_base && mapping.data() == alternate_base) {
        std::swap(g_plan, alternate_plan);
        std::swap(g_blob, alternate_blob);
        std::swap(g_blob_base, alternate_base);
        std::swap(g_blob_size, alternate_size);
    }
    return init_blob(mapping.data(), mapping.size(), &mapping);
}

bool init(const std::byte* blob, std::size_t size) {
    return init_blob(blob, size, nullptr);
}

bool available() { return g_ok; }

float* kv_alloc(std::size_t n_floats) {
    if (!g_ok) return nullptr;
    id<MTLBuffer> buf =
        [g_dev newBufferWithLength:n_floats * sizeof(float)
                           options:MTLResourceStorageModeShared];
    if (buf == nil) return nullptr;
    std::memset([buf contents], 0, n_floats * sizeof(float));
    auto* p = static_cast<float*>([buf contents]);
    g_kv_bufs.emplace_back(p, buf);
    return p;
}

bool matmul(std::span<float> out, std::span<const float> x, std::uint32_t wtype,
            const std::byte* w, std::size_t rows, std::size_t n, std::size_t k) {
    if (!g_ok) return false;
    id<MTLComputePipelineState> pso = pso_for(wtype);
    if (pso == nil) return false;
    if (w < g_blob_base || w >= g_blob_base + g_blob_size) return false;

    g_x = ensure(g_x, g_x_cap, x.size() * sizeof(float));
    g_y = ensure(g_y, g_y_cap, out.size() * sizeof(float));
    std::memcpy([g_x contents], x.data(), x.size() * sizeof(float));

    MMParams p{static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(k),
               static_cast<std::uint32_t>(rows), 0,
               static_cast<std::uint64_t>(w - g_blob_base)};

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:g_blob offset:0 atIndex:0];
    [enc setBuffer:g_x offset:0 atIndex:1];
    [enc setBuffer:g_y offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((n + 3) / 4, rows, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    std::memcpy(out.data(), [g_y contents], out.size() * sizeof(float));
    return true;
}

namespace {

bool prepare(const Model& model) {
    if (model.mapping().data() != g_blob_base && !init(model.mapping())) return false;
    if (g_plan.ready) return true;

    const Config& cfg = model.config();
    g_plan.d        = cfg.embedding_length;
    g_plan.n_head   = cfg.attention_head_count;
    g_plan.n_kv     = cfg.attention_head_count_kv;
    g_plan.head_dim = g_plan.d / g_plan.n_head;
    g_plan.kv_dim   = g_plan.n_kv * g_plan.head_dim;
    g_plan.ff       = cfg.feed_forward_length;
    g_plan.eps      = cfg.rms_norm_epsilon;
    g_plan.scale    = 1.0f / std::sqrt(static_cast<float>(g_plan.head_dim));

    const WeightView embd = model.view("token_embd.weight");
    g_plan.embd      = off_of(embd);
    g_plan.embd_type = embd.type;
    g_plan.vocab     = static_cast<std::uint32_t>(embd.dims.at(1));
    g_plan.out_norm  = off_of(model.view("output_norm.weight"));
    if (pso_for(embd.type) == nil) return false;

    g_plan.layers.resize(cfg.block_count);
    for (std::uint32_t L = 0; L < cfg.block_count; ++L) {
        const std::string p = "blk." + std::to_string(L) + ".";
        LayerOffs& o = g_plan.layers[L];
        o.attn_norm = off_of(model.view(p + "attn_norm.weight"));
        o.ffn_norm  = off_of(model.view(p + "ffn_norm.weight"));
        WeightView wq = model.view(p + "attn_q.weight");
        WeightView wk = model.view(p + "attn_k.weight");
        WeightView wv = model.view(p + "attn_v.weight");
        WeightView wo = model.view(p + "attn_output.weight");
        WeightView wg = model.view(p + "ffn_gate.weight");
        WeightView wu = model.view(p + "ffn_up.weight");
        WeightView wd = model.view(p + "ffn_down.weight");
        o.wq = off_of(wq); o.tq = wq.type;
        o.wk = off_of(wk); o.tk = wk.type;
        o.wv = off_of(wv); o.tv = wv.type;
        o.wo = off_of(wo); o.to = wo.type;
        o.wgate = off_of(wg); o.tgate = wg.type;
        o.wup   = off_of(wu); o.tup   = wu.type;
        o.wdown = off_of(wd); o.tdown = wd.type;
        for (std::uint32_t t : {wq.type, wk.type, wv.type, wo.type,
                                wg.type, wu.type, wd.type})
            if (pso_for(t) == nil) return false;
    }

    // inv_freq like the CPU path: theta^(-2i/hd) / freq_factor[i], in double.
    std::vector<float> ff;
    if (model.has("rope_freqs.weight")) ff = model.get("rope_freqs.weight").data;
    const std::size_t half = g_plan.head_dim / 2;
    std::vector<float> inv(half);
    for (std::size_t i = 0; i < half; ++i) {
        double f = std::pow(static_cast<double>(cfg.rope_freq_base),
                            -2.0 * static_cast<double>(i) /
                                static_cast<double>(g_plan.head_dim));
        if (!ff.empty()) f /= static_cast<double>(ff[i]);
        inv[i] = static_cast<float>(f);
    }
    g_plan.inv_freq = [g_dev newBufferWithBytes:inv.data()
                                         length:half * sizeof(float)
                                        options:MTLResourceStorageModeShared];

    auto fbuf = [&](std::size_t n) {
        return [g_dev newBufferWithLength:n * sizeof(float)
                                  options:MTLResourceStorageModeShared];
    };
    g_plan.x      = fbuf(g_plan.d * SHORT_PREFILL_ROWS);
    g_plan.xb     = fbuf(g_plan.d * SHORT_PREFILL_ROWS);
    g_plan.xb2    = fbuf(g_plan.d);
    g_plan.q      = fbuf(g_plan.d * SHORT_PREFILL_ROWS);
    g_plan.kt     = fbuf(g_plan.kv_dim * SHORT_PREFILL_ROWS);
    g_plan.vt     = fbuf(g_plan.kv_dim * SHORT_PREFILL_ROWS);
    g_plan.attn   = fbuf(g_plan.d * SHORT_PREFILL_ROWS);
    g_plan.h1     = fbuf(g_plan.ff * SHORT_PREFILL_ROWS);
    g_plan.h2     = fbuf(g_plan.ff);
    g_plan.logits = fbuf(g_plan.vocab);

    if (!g_plan.x || !g_plan.xb || !g_plan.q || !g_plan.kt || !g_plan.vt ||
        !g_plan.attn || !g_plan.h1 || !g_plan.logits || !g_plan.inv_freq)
        return false;
    g_plan.ready = true;
    return true;
}

}  // namespace

namespace {

// Encode consecutive rows starting at pos. Project only the final row
// to vocabulary logits. Decode uses this same path with a single row.
void encode_quantized_rows(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> kbuf,
                  id<MTLBuffer> vbuf, std::size_t max_seq, std::uint32_t pos,
                  std::uint32_t rows = 1, bool verify = false) {
    const std::uint32_t d = g_plan.d, kv_dim = g_plan.kv_dim, ffd = g_plan.ff;
    const std::uint32_t half_hd = g_plan.head_dim / 2;

    // Each partition stores 128 unnormalized output values plus its
    // maximum and sum, padded to a float4 boundary.
    const uint32_t parts = (pos + rows + 31) / 32;
    bool split = g_plan.head_dim == 128 && g_attn_part && g_attn_merge && (rows == 1 || pos >= 32);
    if (split) {
        const std::size_t bytes = std::size_t(rows) * g_plan.n_head * parts * 132 * sizeof(float);
        g_plan.attn_parts = ensure(g_plan.attn_parts, g_plan.parts_cap, bytes);
        split = g_plan.attn_parts != nil;
    }
    for (std::size_t L = 0; L < g_plan.layers.size(); ++L) {
        const LayerOffs& o = g_plan.layers[L];
        const std::uint64_t layer_off =
            (static_cast<std::uint64_t>(L) * max_seq) * kv_dim;
        const std::uint64_t row_off = layer_off +
            static_cast<std::uint64_t>(pos) * kv_dim;

        enc_rmsnorm(enc, g_plan.x, g_plan.xb, o.attn_norm, d, g_plan.eps, rows);
        enc_gemv(enc, o.tq, o.wq, g_plan.xb, g_plan.q,  d,      d, 0, rows);
        enc_gemv(enc, o.tk, o.wk, g_plan.xb, g_plan.kt, kv_dim, d, 0, rows);
        enc_gemv(enc, o.tv, o.wv, g_plan.xb, g_plan.vt, kv_dim, d, 0, rows);

        RopeFusedParams rf{g_plan.n_head, g_plan.n_kv, g_plan.head_dim, pos,
                           kv_dim, 0, row_off};
        const std::uint32_t rope_threads =
            (g_plan.n_head + g_plan.n_kv) * half_hd + kv_dim;
        [enc setComputePipelineState:g_ropekv];
        [enc setBuffer:g_plan.q offset:0 atIndex:0];
        [enc setBuffer:g_plan.kt offset:0 atIndex:1];
        [enc setBuffer:g_plan.vt offset:0 atIndex:2];
        [enc setBuffer:kbuf offset:0 atIndex:3];
        [enc setBuffer:vbuf offset:0 atIndex:4];
        [enc setBuffer:g_plan.inv_freq offset:0 atIndex:5];
        [enc setBytes:&rf length:sizeof(rf) atIndex:6];
        [enc dispatchThreads:MTLSizeMake(rope_threads, rows, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        AttnParams ap{g_plan.n_head / g_plan.n_kv, g_plan.head_dim, pos,
                      kv_dim, layer_off, g_plan.scale, rows};
        [enc setComputePipelineState:split ? g_attn_part : g_attn];
        [enc setBuffer:g_plan.q offset:0 atIndex:0];
        [enc setBuffer:kbuf offset:0 atIndex:1];
        [enc setBuffer:vbuf offset:0 atIndex:2];
        [enc setBuffer:split ? g_plan.attn_parts : g_plan.attn offset:0 atIndex:3];
        [enc setBytes:&ap length:sizeof(ap) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(g_plan.n_head, split ? parts : rows, split ? rows : 1)
            threadsPerThreadgroup:MTLSizeMake(split ? 32 : 128, 1, 1)];
        if (split) {
            [enc setComputePipelineState:g_attn_merge];
            [enc setBuffer:g_plan.attn_parts offset:0 atIndex:0];
            [enc setBuffer:g_plan.attn offset:0 atIndex:1];
            [enc setBytes:&parts length:sizeof(parts) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(g_plan.n_head * rows, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }

        enc_gemv(enc, o.to, o.wo, g_plan.attn, g_plan.x, d, d, /*mode=*/1, rows);

        enc_rmsnorm(enc, g_plan.x, g_plan.xb, o.ffn_norm, d, g_plan.eps, rows);
        enc_gemv(enc, o.tgate, o.wgate, g_plan.xb, g_plan.h1, ffd, d, 0, rows);
        enc_gemv(enc, o.tup,   o.wup,   g_plan.xb, g_plan.h1, ffd, d, /*mode=*/2, rows);
        enc_gemv(enc, o.tdown, o.wdown, g_plan.h1, g_plan.x, d, ffd, /*mode=*/1, rows);
    }

    const uint32_t outputs = verify ? rows : 1;
    enc_rmsnorm(enc, g_plan.x, g_plan.xb, g_plan.out_norm, d, g_plan.eps,
                outputs, (rows - outputs) * d * sizeof(float));
    enc_gemv(enc, g_plan.embd_type, g_plan.embd, g_plan.xb, verify ? g_plan.verify_logits : g_plan.logits,
             g_plan.vocab, d, 0, outputs);
}

void encode_embedding(id<MTLComputeCommandEncoder> enc, std::size_t s, std::uint32_t rows = 1) {
    EmbedParams ep{g_plan.d, static_cast<std::uint32_t>(s), g_plan.embd};
    [enc setComputePipelineState:g_embed];
    [enc setBuffer:g_blob offset:0 atIndex:0];
    [enc setBuffer:g_tokens offset:0 atIndex:1];
    [enc setBuffer:g_plan.x offset:0 atIndex:2];
    [enc setBytes:&ep length:sizeof(ep) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(g_plan.d, rows, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

}  // namespace

bool decode_step(const Model& model, KVCache& cache, std::int32_t token,
                 std::span<float> logits_out) {
    if (!g_ok) return false;
    if (!prepare(model)) return false;

    const std::uint32_t pos = static_cast<std::uint32_t>(cache.len);
    if (pos + 1 > cache.max_seq || pos + 1 > MAX_GPU_CTX) return false;
    if (logits_out.size() != g_plan.vocab) return false;

    id<MTLBuffer> kbuf = find_kv(cache.k_row(0, 0));
    id<MTLBuffer> vbuf = find_kv(cache.v_row(0, 0));
    if (kbuf == nil || vbuf == nil) return false;  // cache not GPU-backed

    // Embedding row on CPU (one 3072-float row), uploaded as initial x.
    const std::size_t erb = quant::row_bytes(g_plan.embd_type, g_plan.d);
    if (token < 0 || static_cast<std::uint32_t>(token) >= g_plan.vocab)
        return false;
    quant::dequantize_to(g_plan.embd_type,
                         g_blob_base + g_plan.embd +
                             static_cast<std::size_t>(token) * erb,
                         g_plan.d, static_cast<float*>([g_plan.x contents]));

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!enc) throw std::runtime_error("decode step: missing Metal encoder");
    encode_quantized_rows(enc, kbuf, vbuf, cache.max_seq, pos);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error("decode step: Metal command failed");
    std::memcpy(logits_out.data(), [g_plan.logits contents],
                g_plan.vocab * sizeof(float));
    cache.len = pos + 1;
    return true;
}

// Verify a known token followed by draft tokens. Every row predicts its
// successor; only the prefix through the first mismatch advances the cache.
bool verify_draft(const Model& model, KVCache& cache, std::span<const int32_t> inputs,
                  std::span<int32_t> output, std::size_t& accepted) {
    accepted = 0;
    if (inputs.size() < 2 || inputs.size() > 8 || output.size() < inputs.size()) return false;
    const uint32_t rows = static_cast<uint32_t>(inputs.size());
    if (!g_ok || !prepare(model) ||
        g_plan.embd_type != Q6K || !g_prompt[0][0] || !g_prompt[1][0] ||
        cache.len + rows > std::min<std::size_t>(cache.max_seq, MAX_GPU_CTX)) return false;
    for (auto id : inputs)
        if (id < 0 || static_cast<uint32_t>(id) >= g_plan.vocab) return false;
    id<MTLBuffer> k = find_kv(cache.k_row(0, 0)), v = find_kv(cache.v_row(0, 0));
    if (!k || !v) return false;
    if (!g_plan.verify_logits)
        g_plan.verify_logits = [g_dev newBufferWithLength:8ull * g_plan.vocab * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
    g_tokens = ensure(g_tokens, g_tokens_cap, rows * sizeof(int32_t));
    if (!g_tokens || !g_plan.verify_logits) return false;
    auto* tokens = static_cast<int32_t*>([g_tokens contents]);
    std::copy(inputs.begin(), inputs.end(), tokens);

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!enc) throw std::runtime_error("draft verification: missing Metal encoder");
    encode_embedding(enc, 0, rows);
    encode_quantized_rows(enc, k, v, cache.max_seq, cache.len, rows, true);
    [enc setComputePipelineState:g_argmax];
    [enc setBuffer:g_tokens offset:0 atIndex:1];
    for (uint32_t r = 0; r < rows; ++r) {
        ArgmaxParams ap{g_plan.vocab, r};
        [enc setBuffer:g_plan.verify_logits offset:std::size_t(r) * g_plan.vocab * sizeof(float) atIndex:0];
        [enc setBytes:&ap length:sizeof(ap) atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error("draft verification: Metal command failed");

    accepted = 1;
    while (accepted < rows && tokens[accepted - 1] == inputs[accepted]) ++accepted;
    std::copy_n(tokens, accepted, output.begin());
    cache.len += accepted;
    return true;
}

bool decode_chain(const Model& model, KVCache& cache, std::int32_t token,
                  std::span<std::int32_t> out_tokens) {
    if (!g_ok) return false;
    static const bool disabled = [] {
        const char* e = std::getenv("DAMASCUS_CHAIN");
        return e != nullptr && std::strcmp(e, "0") == 0;
    }();
    if (disabled) return false;
    if (!prepare(model)) return false;
    if (g_plan.embd_type != Q6K) return false;  // embed kernel is Q6_K-only

    const std::size_t n = out_tokens.size();
    const std::uint32_t pos0 = static_cast<std::uint32_t>(cache.len);
    if (n == 0) return true;
    if (pos0 + n > cache.max_seq || pos0 + n > MAX_GPU_CTX) return false;
    if (token < 0 || static_cast<std::uint32_t>(token) >= g_plan.vocab)
        return false;

    id<MTLBuffer> kbuf = find_kv(cache.k_row(0, 0));
    id<MTLBuffer> vbuf = find_kv(cache.v_row(0, 0));
    if (kbuf == nil || vbuf == nil) return false;

    g_tokens = ensure(g_tokens, g_tokens_cap, (n + 1) * sizeof(std::int32_t));
    if (!g_tokens) return false;
    static_cast<std::int32_t*>([g_tokens contents])[0] = token;

    // Commit in groups WITHOUT waiting: the token ring lives on GPU, so group
    // N+1 has no CPU dependency on group N — CPU encoding runs ahead of GPU
    // execution (same queue => in-order) and we only wait on the last buffer.
    constexpr std::size_t GROUP = 16;
    std::vector<id<MTLCommandBuffer>> submitted;
    submitted.reserve((n + GROUP - 1) / GROUP);
    for (std::size_t s0 = 0; s0 < n; s0 += GROUP) {
        const std::size_t s1 = std::min(n, s0 + GROUP);
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        if (!enc) throw std::runtime_error("decode chain: missing Metal encoder");
        for (std::size_t s = s0; s < s1; ++s) {
            encode_embedding(enc, s);
            encode_quantized_rows(enc, kbuf, vbuf, cache.max_seq,
                         pos0 + static_cast<std::uint32_t>(s));

            ArgmaxParams ap{g_plan.vocab, static_cast<std::uint32_t>(s + 1)};
            [enc setComputePipelineState:g_argmax];
            [enc setBuffer:g_plan.logits offset:0 atIndex:0];
            [enc setBuffer:g_tokens offset:0 atIndex:1];
            [enc setBytes:&ap length:sizeof(ap) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        [enc endEncoding];
        [cb commit];
        submitted.push_back(cb);
    }
    [submitted.back() waitUntilCompleted];
    for (id<MTLCommandBuffer> cb : submitted)
        if (cb.status != MTLCommandBufferStatusCompleted)
            throw std::runtime_error("decode chain: Metal command failed");

    const auto* toks = static_cast<const std::int32_t*>([g_tokens contents]);
    for (std::size_t s = 0; s < n; ++s) out_tokens[s] = toks[s + 1];
    cache.len = pos0 + n;
    return true;
}

namespace {

// One-time: dequantize every layer matmul weight to half in one GPU arena
// (~5.6 GB for 3B — fine in 24 GB unified memory). Prefill GEMMs then run on
// the fp16 matmul hardware via MPS.
bool prepare16() {
    if (g_plan.w16_ready) return true;
    if (g_plan.w16_failed) return false;

    std::uint64_t total = 0;
    for (const LayerOffs& o : g_plan.layers) {
        (void)o;
        total += 2ull * g_plan.d * g_plan.d;          // q, o
        total += 2ull * g_plan.kv_dim * g_plan.d;     // k, v
        total += 3ull * g_plan.ff * g_plan.d;         // gate, up, down
    }
    g_plan.w16 = [g_dev newBufferWithLength:total * 2
                                    options:MTLResourceStorageModeShared];
    if (g_plan.w16 == nil) { g_plan.w16_failed = true; return false; }

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    std::uint64_t at = 0;
    auto dq = [&](std::uint32_t type, std::uint64_t w_off, std::uint64_t n) {
        DqParams p{w_off, static_cast<std::uint32_t>(n), 0};
        [enc setComputePipelineState:(type == Q4K ? g_dq4h : g_dq6h)];
        [enc setBuffer:g_blob offset:0 atIndex:0];
        [enc setBuffer:g_plan.w16 offset:at * 2 atIndex:1];
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(n, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        const std::uint64_t off = at;
        at += n;
        return off;
    };

    g_plan.l16.resize(g_plan.layers.size());
    for (std::size_t L = 0; L < g_plan.layers.size(); ++L) {
        const LayerOffs& o = g_plan.layers[L];
        Layer16& h = g_plan.l16[L];
        const std::uint64_t dd = std::uint64_t(g_plan.d) * g_plan.d;
        const std::uint64_t kd = std::uint64_t(g_plan.kv_dim) * g_plan.d;
        const std::uint64_t fd = std::uint64_t(g_plan.ff) * g_plan.d;
        h.q    = dq(o.tq, o.wq, dd);
        h.k    = dq(o.tk, o.wk, kd);
        h.v    = dq(o.tv, o.wv, kd);
        h.o    = dq(o.to, o.wo, dd);
        h.gate = dq(o.tgate, o.wgate, fd);
        h.up   = dq(o.tup,   o.wup,   fd);
        h.down = dq(o.tdown, o.wdown, fd);
    }
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    g_plan.w16_ready = true;
    return true;
}

bool ensure_prefill_bufs(std::size_t S) {
    if (S <= g_plan.pf_cap) return true;
    const std::size_t d = g_plan.d, kv = g_plan.kv_dim, ff = g_plan.ff;
    auto f32 = [&](std::size_t n) {
        return [g_dev newBufferWithLength:n * 4
                                  options:MTLResourceStorageModeShared];
    };
    auto f16 = [&](std::size_t n) {
        return [g_dev newBufferWithLength:n * 2
                                  options:MTLResourceStorageModeShared];
    };
    g_plan.pf_x    = f32(S * d);
    g_plan.pf_xb   = f32(S * d);
    g_plan.pf_q    = f32(S * d);
    g_plan.pf_kt   = f32(S * kv);
    g_plan.pf_vt   = f32(S * kv);
    g_plan.pf_attn = f32(S * d);
    g_plan.pf_xb16 = f16(S * d);
    g_plan.pf_a16  = f16(S * d);
    g_plan.pf_o16  = f16(S * d);
    g_plan.pf_h16a = f16(S * ff);
    g_plan.pf_h16b = f16(S * ff);
    g_plan.pf_h16c = f16(S * 2 * ff);
    for (id b : @[g_plan.pf_x, g_plan.pf_xb, g_plan.pf_q, g_plan.pf_kt,
                  g_plan.pf_vt, g_plan.pf_attn, g_plan.pf_xb16, g_plan.pf_a16,
                  g_plan.pf_o16, g_plan.pf_h16a, g_plan.pf_h16b,
                  g_plan.pf_h16c])
        if (b == nil) return false;
    g_plan.pf_cap = S;
    return true;
}

MPSMatrixMultiplication* mps_for(std::size_t rows, std::size_t n,
                                 std::size_t k) {
    auto key = std::make_tuple(rows, n, k);
    auto it = g_mps.find(key);
    if (it != g_mps.end()) return it->second;
    MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
        initWithDevice:g_dev
         transposeLeft:NO
        transposeRight:YES
            resultRows:rows
         resultColumns:n
       interiorColumns:k
                 alpha:1.0
                  beta:0.0];
    g_mps[key] = mm;
    return mm;
}

MPSMatrix* mps_mat(id<MTLBuffer> buf, std::size_t off_bytes, std::size_t rows,
                   std::size_t cols) {
    MPSMatrixDescriptor* dsc =
        [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                              columns:cols
                                             rowBytes:cols * 2
                                             dataType:MPSDataTypeFloat16];
    return [[MPSMatrix alloc] initWithBuffer:buf offset:off_bytes
                                  descriptor:dsc];
}

// y16[S, n] = x16[S, k] · W16[n, k]^T via MPS. Ends/reopens the encoder.
id<MTLComputeCommandEncoder> mps_gemm(id<MTLCommandBuffer> cb,
                                      id<MTLComputeCommandEncoder> enc,
                                      std::size_t S, std::size_t n,
                                      std::size_t k, id<MTLBuffer> x16,
                                      std::uint64_t w_elem_off,
                                      id<MTLBuffer> y16) {
    [enc endEncoding];
    MPSMatrix* A = mps_mat(x16, 0, S, k);
    MPSMatrix* B = mps_mat(g_plan.w16, w_elem_off * 2, n, k);
    MPSMatrix* C = mps_mat(y16, 0, S, n);
    [mps_for(S, n, k) encodeToCommandBuffer:cb leftMatrix:A rightMatrix:B
                               resultMatrix:C];
    return [cb computeCommandEncoder];
}

void enc_1buf(id<MTLComputeCommandEncoder> enc,
              id<MTLComputePipelineState> pso, id<MTLBuffer> a,
              id<MTLBuffer> b, std::uint32_t n) {
    [enc setComputePipelineState:pso];
    [enc setBuffer:a offset:0 atIndex:0];
    [enc setBuffer:b offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

}  // namespace

bool prefill_chunk(const Model& model, KVCache& cache,
                   std::span<const std::int32_t> tokens,
                   std::span<float> logits_out) {
    if (!g_ok) return false;
    if (!prepare(model)) return false;
    if (g_plan.embd_type != Q6K) return false;

    const std::size_t S = tokens.size();
    const std::uint32_t pos0 = static_cast<std::uint32_t>(cache.len);
    if (S == 0 || pos0 + S > cache.max_seq || pos0 + S > MAX_GPU_CTX)
        return false;
    if (logits_out.size() != g_plan.vocab) return false;
    for (auto id : tokens)
        if (id < 0 || static_cast<std::uint32_t>(id) >= g_plan.vocab)
            throw std::runtime_error("prefill: token id out of range");

    id<MTLBuffer> kbuf = find_kv(cache.k_row(0, 0));
    id<MTLBuffer> vbuf = find_kv(cache.v_row(0, 0));
    if (kbuf == nil || vbuf == nil) return false;

    g_tokens = ensure(g_tokens, g_tokens_cap, (S + 1) * sizeof(std::int32_t));
    if (!g_tokens) return false;
    std::memcpy([g_tokens contents], tokens.data(), S * sizeof(std::int32_t));

    // A short first request can finish directly on quantized weights before
    // the one-time full-model fp16 expansion for MPS would pay for itself.
    if (S <= SHORT_PREFILL_ROWS && !g_plan.w16_ready &&
        g_prompt[0][S > 8] && g_prompt[1][S > 8]) {
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        encode_embedding(enc, 0, S);
        encode_quantized_rows(enc, kbuf, vbuf, cache.max_seq, pos0, S);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError)
            throw std::runtime_error("short prefill: Metal command failed");
        std::memcpy(logits_out.data(), [g_plan.logits contents], logits_out.size_bytes());
        cache.len += S;
        if (std::getenv("DAMASCUS_TRACE_PREFILL"))
            std::fprintf(stderr, "[prefill] quantized Metal, %zu rows\n", S);
        return true;
    }
    if (!prepare16() || !ensure_prefill_bufs(S)) return false;

    const std::uint32_t d = g_plan.d, kv_dim = g_plan.kv_dim, ffd = g_plan.ff;
    const std::uint32_t half_hd = g_plan.head_dim / 2;
    const std::uint32_t Sd = static_cast<std::uint32_t>(S) * d;
    const std::uint32_t Sff = static_cast<std::uint32_t>(S) * ffd;

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    // embeddings -> pf_x [S, d]
    EmbedParams ep{d, 0, g_plan.embd};
    [enc setComputePipelineState:g_embed];
    [enc setBuffer:g_blob offset:0 atIndex:0];
    [enc setBuffer:g_tokens offset:0 atIndex:1];
    [enc setBuffer:g_plan.pf_x offset:0 atIndex:2];
    [enc setBytes:&ep length:sizeof(ep) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(d, S, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    for (std::size_t L = 0; L < g_plan.layers.size(); ++L) {
        const LayerOffs& o = g_plan.layers[L];
        const Layer16& h = g_plan.l16[L];
        const std::uint64_t layer_off =
            (static_cast<std::uint64_t>(L) * cache.max_seq) * kv_dim;
        const std::uint64_t row_off = layer_off +
            static_cast<std::uint64_t>(pos0) * kv_dim;

        // attn rmsnorm (rows) straight to half
        NormParams np{d, g_plan.eps, o.attn_norm};
        [enc setComputePipelineState:g_rmsh];
        [enc setBuffer:g_plan.pf_x offset:0 atIndex:0];
        [enc setBuffer:g_plan.pf_xb16 offset:0 atIndex:1];
        [enc setBuffer:g_blob offset:0 atIndex:2];
        [enc setBytes:&np length:sizeof(np) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, S, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        // fused QKV: q,k,v rows are contiguous in the arena -> one GEMM
        const std::uint32_t qkv_n = d + 2 * kv_dim;
        enc = mps_gemm(cb, enc, S, qkv_n, d, g_plan.pf_xb16, h.q,
                       g_plan.pf_h16b);

        RopeFusedParams rf{g_plan.n_head, g_plan.n_kv, g_plan.head_dim, pos0,
                           kv_dim, qkv_n, row_off};  // pad = qkv row stride
        const std::uint32_t rope_threads =
            (g_plan.n_head + g_plan.n_kv) * half_hd + kv_dim;
        [enc setComputePipelineState:g_ropekvh];
        [enc setBuffer:g_plan.pf_h16b offset:0 atIndex:0];
        [enc setBuffer:g_plan.pf_q offset:0 atIndex:1];
        [enc setBuffer:kbuf offset:0 atIndex:2];
        [enc setBuffer:vbuf offset:0 atIndex:3];
        [enc setBuffer:g_plan.inv_freq offset:0 atIndex:4];
        [enc setBytes:&rf length:sizeof(rf) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(rope_threads, S, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        AttnParams ap{g_plan.n_head / g_plan.n_kv, g_plan.head_dim, pos0,
                      kv_dim, layer_off, g_plan.scale,
                      static_cast<std::uint32_t>(S)};
        [enc setComputePipelineState:g_attnf];
        [enc setBuffer:g_plan.pf_q offset:0 atIndex:0];
        [enc setBuffer:kbuf offset:0 atIndex:1];
        [enc setBuffer:vbuf offset:0 atIndex:2];
        [enc setBuffer:g_plan.pf_a16 offset:0 atIndex:3];
        [enc setBytes:&ap length:sizeof(ap) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(g_plan.n_head, (S + 7) / 8, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        // output projection + residual (attn wrote half directly)
        enc = mps_gemm(cb, enc, S, d, d, g_plan.pf_a16, h.o, g_plan.pf_o16);
        enc_1buf(enc, g_cvtadd, g_plan.pf_x, g_plan.pf_o16, Sd);

        // FFN: rmsnorm->half, fused gate|up GEMM, SwiGLU, down, residual
        NormParams np2{d, g_plan.eps, o.ffn_norm};
        [enc setComputePipelineState:g_rmsh];
        [enc setBuffer:g_plan.pf_x offset:0 atIndex:0];
        [enc setBuffer:g_plan.pf_xb16 offset:0 atIndex:1];
        [enc setBuffer:g_blob offset:0 atIndex:2];
        [enc setBytes:&np2 length:sizeof(np2) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(1, S, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        enc = mps_gemm(cb, enc, S, 2 * ffd, d, g_plan.pf_xb16, h.gate,
                       g_plan.pf_h16c);
        const std::uint32_t nf[2] = {Sff, ffd};
        [enc setComputePipelineState:g_swigluc];
        [enc setBuffer:g_plan.pf_h16c offset:0 atIndex:0];
        [enc setBuffer:g_plan.pf_h16a offset:0 atIndex:1];
        [enc setBytes:nf length:sizeof(nf) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(Sff, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        enc = mps_gemm(cb, enc, S, d, ffd, g_plan.pf_h16a, h.down, g_plan.pf_o16);
        enc_1buf(enc, g_cvtadd, g_plan.pf_x, g_plan.pf_o16, Sd);
    }

    // final rmsnorm on the LAST row only, then the quantized fp32 LM head
    NormParams nf{d, g_plan.eps, g_plan.out_norm};
    [enc setComputePipelineState:g_rms];
    [enc setBuffer:g_plan.pf_x offset:(S - 1) * d * 4 atIndex:0];
    [enc setBuffer:g_plan.xb offset:0 atIndex:1];
    [enc setBuffer:g_blob offset:0 atIndex:2];
    [enc setBytes:&nf length:sizeof(nf) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    enc_gemv(enc, g_plan.embd_type, g_plan.embd, g_plan.xb, g_plan.logits,
             g_plan.vocab, d);

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    std::memcpy(logits_out.data(), [g_plan.logits contents],
                g_plan.vocab * sizeof(float));
    cache.len = pos0 + S;
    return true;
}

}  // namespace damascus::gpu
