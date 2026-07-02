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
#include "forward.hpp"
#include "quant.hpp"

#import <Metal/Metal.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace damascus::gpu {

namespace {

constexpr std::uint32_t Q4K = 12;
constexpr std::uint32_t Q6K = 14;
constexpr std::uint32_t MAX_GPU_CTX = 2048;  // threadgroup scores[] size

struct MMParams {
    std::uint32_t n;
    std::uint32_t k;
    std::uint32_t rows;
    std::uint32_t pad;
    std::uint64_t w_off;
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
};

const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct MMParams  { uint n; uint k; uint rows; uint pad; ulong w_off; };
struct NormParams{ uint d; float eps; ulong w_off; };
struct RopeParams{ uint n_head; uint head_dim; uint pos; uint pad; ulong dst_off; };
struct AttnParams{ uint n_rep; uint head_dim; uint pos; uint kv_dim; ulong layer_off; float scale; };

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
                     uint  lane [[thread_index_in_simdgroup]]) {
    const uint j = tg.x;
    const uint m = tg.y;
    const uint nb = p.k >> 8;
    device const uchar* wrow = blob + p.w_off + ulong(j) * ulong(nb) * 144;
    device const float* xr   = x + ulong(m) * ulong(p.k);

    float acc = 0.0f;
    const uint nsub = p.k >> 5;
    for (uint c = lane; c < nsub; c += 32) {
        const uint b = c >> 3;
        const uint s = c & 7;
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

// ---- decode-step kernels -------------------------------------------------

// One threadgroup; norm weight is an F32 tensor read straight from the blob.
kernel void rmsnorm_g(device const float* x    [[buffer(0)]],
                      device float*       y    [[buffer(1)]],
                      device const uchar* blob [[buffer(2)]],
                      constant NormParams& p   [[buffer(3)]],
                      uint tid  [[thread_position_in_threadgroup]],
                      uint tsz  [[threads_per_threadgroup]],
                      uint sid  [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    device const float* w = (device const float*)(blob + p.w_off);
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
                      uint h   [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]],
                      uint tsz [[threads_per_threadgroup]]) {
    threadgroup float sc[2048];
    threadgroup float red[128];
    const uint kvh = h / p.n_rep;
    device const float* qh = q + h * p.head_dim;
    const uint ctx = p.pos + 1;

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
        out[h * p.head_dim + e] = acc * inv;
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
id<MTLComputePipelineState> g_q4k, g_q6k, g_rms, g_ropeq, g_ropek, g_copy,
                            g_attn, g_swiglu, g_add;
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
    id<MTLBuffer> x, xb, xb2, q, kt, vt, attn, h1, h2, logits;
};
DecodePlan g_plan;

id<MTLBuffer> ensure(id<MTLBuffer> buf, std::size_t& cap, std::size_t need) {
    if (need <= cap) return buf;
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
              std::uint32_t n, std::uint32_t k) {
    MMParams p{n, k, 1, 0, w_off};
    [enc setComputePipelineState:pso_for(wtype)];
    [enc setBuffer:g_blob offset:0 atIndex:0];
    [enc setBuffer:x offset:0 atIndex:1];
    [enc setBuffer:y offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

void enc_elemwise(id<MTLComputeCommandEncoder> enc,
                  id<MTLComputePipelineState> pso, id<MTLBuffer> a,
                  id<MTLBuffer> b, std::uint32_t n) {
    [enc setComputePipelineState:pso];
    [enc setBuffer:a offset:0 atIndex:0];
    [enc setBuffer:b offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void enc_rmsnorm(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x,
                 id<MTLBuffer> y, std::uint64_t w_off, std::uint32_t d,
                 float eps) {
    NormParams p{d, eps, w_off};
    [enc setComputePipelineState:g_rms];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:y offset:0 atIndex:1];
    [enc setBuffer:g_blob offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

id<MTLBuffer> find_kv(const float* p) {
    for (auto& [ptr, buf] : g_kv_bufs)
        if (ptr == p) return buf;
    return nil;
}

}  // namespace

// ---- public API -------------------------------------------------------------

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
    g_q4k    = make_pso(lib, @"gemv_q4k");
    g_q6k    = make_pso(lib, @"gemv_q6k");
    g_rms    = make_pso(lib, @"rmsnorm_g");
    g_ropeq  = make_pso(lib, @"rope_q");
    g_ropek  = make_pso(lib, @"rope_k_store");
    g_copy   = make_pso(lib, @"copy_off");
    g_attn   = make_pso(lib, @"attn_head");
    g_swiglu = make_pso(lib, @"swiglu_g");
    g_add    = make_pso(lib, @"add_g");
    for (id p : @[g_q4k, g_q6k, g_rms, g_ropeq, g_ropek, g_copy, g_attn,
                  g_swiglu, g_add])
        if (p == nil) return false;

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
    [enc dispatchThreadgroups:MTLSizeMake(n, rows, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    std::memcpy(out.data(), [g_y contents], out.size() * sizeof(float));
    return true;
}

namespace {

bool prepare(const Model& model) {
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
    g_plan.x      = fbuf(g_plan.d);
    g_plan.xb     = fbuf(g_plan.d);
    g_plan.xb2    = fbuf(g_plan.d);
    g_plan.q      = fbuf(g_plan.d);
    g_plan.kt     = fbuf(g_plan.kv_dim);
    g_plan.vt     = fbuf(g_plan.kv_dim);
    g_plan.attn   = fbuf(g_plan.d);
    g_plan.h1     = fbuf(g_plan.ff);
    g_plan.h2     = fbuf(g_plan.ff);
    g_plan.logits = fbuf(g_plan.vocab);

    g_plan.ready = true;
    return true;
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

    const std::uint32_t d = g_plan.d, kv_dim = g_plan.kv_dim, ffd = g_plan.ff;
    const std::uint32_t half_hd = g_plan.head_dim / 2;

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    for (std::size_t L = 0; L < g_plan.layers.size(); ++L) {
        const LayerOffs& o = g_plan.layers[L];
        const std::uint64_t layer_off =
            (static_cast<std::uint64_t>(L) * cache.max_seq) * kv_dim;
        const std::uint64_t row_off = layer_off +
            static_cast<std::uint64_t>(pos) * kv_dim;

        enc_rmsnorm(enc, g_plan.x, g_plan.xb, o.attn_norm, d, g_plan.eps);
        enc_gemv(enc, o.tq, o.wq, g_plan.xb, g_plan.q,  d,      d);
        enc_gemv(enc, o.tk, o.wk, g_plan.xb, g_plan.kt, kv_dim, d);
        enc_gemv(enc, o.tv, o.wv, g_plan.xb, g_plan.vt, kv_dim, d);

        RopeParams rq{g_plan.n_head, g_plan.head_dim, pos, 0, 0};
        [enc setComputePipelineState:g_ropeq];
        [enc setBuffer:g_plan.q offset:0 atIndex:0];
        [enc setBuffer:g_plan.inv_freq offset:0 atIndex:1];
        [enc setBytes:&rq length:sizeof(rq) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(g_plan.n_head * half_hd, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

        RopeParams rk{g_plan.n_kv, g_plan.head_dim, pos, 0, row_off};
        [enc setComputePipelineState:g_ropek];
        [enc setBuffer:g_plan.kt offset:0 atIndex:0];
        [enc setBuffer:kbuf offset:0 atIndex:1];
        [enc setBuffer:g_plan.inv_freq offset:0 atIndex:2];
        [enc setBytes:&rk length:sizeof(rk) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(g_plan.n_kv * half_hd, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

        [enc setComputePipelineState:g_copy];
        [enc setBuffer:g_plan.vt offset:0 atIndex:0];
        [enc setBuffer:vbuf offset:0 atIndex:1];
        [enc setBytes:&row_off length:sizeof(row_off) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(kv_dim, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        AttnParams ap{g_plan.n_head / g_plan.n_kv, g_plan.head_dim, pos,
                      kv_dim, layer_off, g_plan.scale};
        [enc setComputePipelineState:g_attn];
        [enc setBuffer:g_plan.q offset:0 atIndex:0];
        [enc setBuffer:kbuf offset:0 atIndex:1];
        [enc setBuffer:vbuf offset:0 atIndex:2];
        [enc setBuffer:g_plan.attn offset:0 atIndex:3];
        [enc setBytes:&ap length:sizeof(ap) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(g_plan.n_head, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        enc_gemv(enc, o.to, o.wo, g_plan.attn, g_plan.xb2, d, d);
        enc_elemwise(enc, g_add, g_plan.x, g_plan.xb2, d);

        enc_rmsnorm(enc, g_plan.x, g_plan.xb, o.ffn_norm, d, g_plan.eps);
        enc_gemv(enc, o.tgate, o.wgate, g_plan.xb, g_plan.h1, ffd, d);
        enc_gemv(enc, o.tup,   o.wup,   g_plan.xb, g_plan.h2, ffd, d);
        enc_elemwise(enc, g_swiglu, g_plan.h1, g_plan.h2, ffd);
        enc_gemv(enc, o.tdown, o.wdown, g_plan.h1, g_plan.xb2, d, ffd);
        enc_elemwise(enc, g_add, g_plan.x, g_plan.xb2, d);
    }

    enc_rmsnorm(enc, g_plan.x, g_plan.xb, g_plan.out_norm, d, g_plan.eps);
    enc_gemv(enc, g_plan.embd_type, g_plan.embd, g_plan.xb, g_plan.logits,
             g_plan.vocab, d);

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    std::memcpy(logits_out.data(), [g_plan.logits contents],
                g_plan.vocab * sizeof(float));
    cache.len = pos + 1;
    return true;
}

}  // namespace damascus::gpu
