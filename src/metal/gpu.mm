// Metal fused dequant matmul host side. Obj-C++ isolated here like
// metal_ctx.mm. Shader source is embedded so there is no metallib/build-path
// dance; it is compiled once at init (~100 ms).

#include "gpu.hpp"

#import <Metal/Metal.h>

#include <cstdlib>
#include <cstring>

namespace damascus::gpu {

namespace {

// Mirrors damascus::quant::GgmlType (kernels keyed by ggml type id).
constexpr std::uint32_t GGML_Q4_K = 12;
constexpr std::uint32_t GGML_Q6_K = 14;

struct MMParams {
    std::uint32_t n;
    std::uint32_t k;
    std::uint32_t rows;
    std::uint32_t pad;
    std::uint64_t w_off;  // byte offset of the weight tensor inside the blob
};

// One simdgroup (32 lanes) per output feature j (grid.x) and batch row m
// (grid.y). Lanes stride over 32-element sub-blocks (Q4_K) or 128-element
// halves (Q6_K), then simd_sum. Dequant arithmetic mirrors quant_cpu.cpp
// exactly; only the accumulation is fp32 and reordered.
const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct MMParams {
    uint  n;
    uint  k;
    uint  rows;
    uint  pad;
    ulong w_off;
};

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
                     uint  lane [[thread_index_in_simdgroup]]) {
    const uint j = tg.x;
    const uint m = tg.y;
    const uint nb = p.k >> 8;                       // super-blocks per row
    device const uchar* wrow = blob + p.w_off + ulong(j) * ulong(nb) * 144;
    device const float* xr   = x + ulong(m) * ulong(p.k);

    float acc = 0.0f;
    const uint nsub = p.k >> 5;                     // 32-element sub-blocks
    for (uint c = lane; c < nsub; c += 32) {
        const uint b = c >> 3;                      // super-block index
        const uint s = c & 7;                       // sub-block within it
        device const uchar* blk = wrow + b * 144;
        const float d    = float(*(device const half*)(blk));
        const float dmin = float(*(device const half*)(blk + 2));
        float sd, sm;
        scale_min_k4(s, blk + 4, sd, sm);
        device const uchar* q  = blk + 16 + (s >> 1) * 32;
        device const float* xv = xr + b * 256 + (s >> 1) * 64 + (s & 1) * 32;

        float sumqx = 0.0f, sumx = 0.0f;
        if ((s & 1) == 0) {
            for (uint l = 0; l < 32; ++l) {
                const float xl = xv[l];
                sumqx += xl * float(q[l] & 0xF);
                sumx  += xl;
            }
        } else {
            for (uint l = 0; l < 32; ++l) {
                const float xl = xv[l];
                sumqx += xl * float(q[l] >> 4);
                sumx  += xl;
            }
        }
        acc += (d * sd) * sumqx - (dmin * sm) * sumx;
    }
    acc = simd_sum(acc);
    if (lane == 0) y[ulong(m) * ulong(p.n) + j] = acc;
}

// block_q6_K: ql[128] qh[64] scales[16](int8) d(half @208) = 210 B, 256 w.
kernel void gemv_q6k(device const uchar* blob [[buffer(0)]],
                     device const float* x    [[buffer(1)]],
                     device float*       y    [[buffer(2)]],
                     constant MMParams&  p    [[buffer(3)]],
                     uint2 tg   [[threadgroup_position_in_grid]],
                     uint  lane [[thread_index_in_simdgroup]]) {
    const uint j = tg.x;
    const uint m = tg.y;
    const uint nb = p.k >> 8;
    device const uchar* wrow = blob + p.w_off + ulong(j) * ulong(nb) * 210;
    device const float* xr   = x + ulong(m) * ulong(p.k);

    float acc = 0.0f;
    const uint nhalf = p.k >> 7;                    // 128-element halves
    for (uint c = lane; c < nhalf; c += 32) {
        const uint b = c >> 1;
        const uint h = c & 1;
        device const uchar* blk = wrow + b * 210;
        device const uchar* ql  = blk + h * 64;
        device const uchar* qh  = blk + 128 + h * 32;
        device const char*  sc  = (device const char*)(blk + 192) + h * 8;
        const float d = float(*(device const half*)(blk + 208));
        device const float* xv = xr + b * 256 + h * 128;

        for (uint l = 0; l < 32; ++l) {
            const uint is = l >> 4;
            const float q1 = float(int((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
            const float q2 = float(int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
            const float q3 = float(int((ql[l]      >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32);
            const float q4 = float(int((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            acc += d * (float(sc[is])     * q1 * xv[l]
                      + float(sc[is + 2]) * q2 * xv[l + 32]
                      + float(sc[is + 4]) * q3 * xv[l + 64]
                      + float(sc[is + 6]) * q4 * xv[l + 96]);
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) y[ulong(m) * ulong(p.n) + j] = acc;
}
)MSL";

id<MTLDevice>               g_dev;
id<MTLCommandQueue>         g_queue;
id<MTLComputePipelineState> g_q4k;
id<MTLComputePipelineState> g_q6k;
id<MTLBuffer>               g_blob;
id<MTLBuffer>               g_x;
id<MTLBuffer>               g_y;
std::size_t                 g_x_cap = 0, g_y_cap = 0;
const std::byte*            g_blob_base = nullptr;
std::size_t                 g_blob_size = 0;
bool g_ok = false, g_tried = false;

id<MTLBuffer> ensure(id<MTLBuffer> buf, std::size_t& cap, std::size_t need) {
    if (need <= cap) return buf;
    cap = need + need / 2;
    return [g_dev newBufferWithLength:cap options:MTLResourceStorageModeShared];
}

}  // namespace

bool init(const std::byte* blob, std::size_t size) {
    if (g_tried) return g_ok && blob == g_blob_base;
    g_tried = true;

    const char* env = std::getenv("DAMASCUS_GPU");
    if (env != nullptr && std::strcmp(env, "0") == 0) return false;

    g_dev = MTLCreateSystemDefaultDevice();
    if (g_dev == nil) return false;
    g_queue = [g_dev newCommandQueue];

    NSError* err = nil;
    id<MTLLibrary> lib =
        [g_dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                            options:nil
                              error:&err];
    if (lib == nil) {
        std::fprintf(stderr, "gpu: shader compile failed: %s\n",
                     err ? [[err localizedDescription] UTF8String] : "?");
        return false;
    }
    g_q4k = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gemv_q4k"]
                                                 error:&err];
    g_q6k = [g_dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gemv_q6k"]
                                                 error:&err];
    if (g_q4k == nil || g_q6k == nil) return false;

    g_blob = [g_dev newBufferWithBytes:blob
                                length:size
                               options:MTLResourceStorageModeShared];
    if (g_blob == nil) return false;

    g_blob_base = blob;
    g_blob_size = size;
    g_ok = true;
    return true;
}

bool available() { return g_ok; }

bool matmul(std::span<float> out, std::span<const float> x, std::uint32_t wtype,
            const std::byte* w, std::size_t rows, std::size_t n, std::size_t k) {
    if (!g_ok) return false;

    id<MTLComputePipelineState> pso = nil;
    if (wtype == GGML_Q4_K) pso = g_q4k;
    else if (wtype == GGML_Q6_K) pso = g_q6k;
    if (pso == nil) return false;

    if (w < g_blob_base || w >= g_blob_base + g_blob_size) {
        return false;  // pointer not inside the uploaded blob
    }

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
    [enc dispatchThreadgroups:MTLSizeMake(n, rows, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    std::memcpy(out.data(), [g_y contents], out.size() * sizeof(float));
    return true;
}

}  // namespace damascus::gpu
