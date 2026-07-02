// Raw MPS fp16 GEMM throughput at damascus's prefill shapes.
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <chrono>
#include <cstdio>

static double bench(id<MTLDevice> dev, id<MTLCommandQueue> q,
                    int M, int N, int K, int reps) {
    id<MTLBuffer> a = [dev newBufferWithLength:(size_t)M * K * 2
                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> b = [dev newBufferWithLength:(size_t)N * K * 2
                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> c = [dev newBufferWithLength:(size_t)M * N * 2
                                       options:MTLResourceStorageModeShared];
    MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
        initWithDevice:dev transposeLeft:NO transposeRight:YES
            resultRows:M resultColumns:N interiorColumns:K alpha:1 beta:0];
    auto mat = [&](id<MTLBuffer> buf, int r, int cc) {
        return [[MPSMatrix alloc]
            initWithBuffer:buf
                descriptor:[MPSMatrixDescriptor
                               matrixDescriptorWithRows:r columns:cc
                                                rowBytes:(size_t)cc * 2
                                                dataType:MPSDataTypeFloat16]];
    };
    // warmup
    for (int i = 0; i < 3; ++i) {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        [mm encodeToCommandBuffer:cb leftMatrix:mat(a, M, K)
                      rightMatrix:mat(b, N, K) resultMatrix:mat(c, M, N)];
        [cb commit];
        [cb waitUntilCompleted];
    }
    auto t0 = std::chrono::steady_clock::now();
    id<MTLCommandBuffer> cb = [q commandBuffer];
    for (int i = 0; i < reps; ++i) {
        [mm encodeToCommandBuffer:cb leftMatrix:mat(a, M, K)
                      rightMatrix:mat(b, N, K) resultMatrix:mat(c, M, N)];
    }
    [cb commit];
    [cb waitUntilCompleted];
    const double s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    return 2.0 * M * N * K * reps / s / 1e12;  // TFLOP/s
}

int main() {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    struct { const char* name; int M, N, K; } shapes[] = {
        {"qo   512x3072x3072",  512, 3072, 3072},
        {"kv   512x1024x3072",  512, 1024, 3072},
        {"ffn  512x8192x3072",  512, 8192, 3072},
        {"down 512x3072x8192",  512, 3072, 8192},
        {"ffn 1024x8192x3072", 1024, 8192, 3072},
    };
    for (auto& s : shapes)
        std::printf("%s : %6.2f TFLOP/s\n", s.name,
                    bench(dev, q, s.M, s.N, s.K, 30));
    return 0;
}
