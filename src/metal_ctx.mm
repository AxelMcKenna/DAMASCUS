// Metal context implementation. Objective-C++ so we can talk to Cocoa/Metal
// directly. Keeps all Obj-C bleed isolated to this file.

#include "metal_ctx_internal.hpp"

#import <Foundation/Foundation.h>

#include <cstdio>

namespace damascus {

struct MetalContext {
    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
};

id<MTLDevice> metal_ctx_device(const MetalContext* ctx) {
    if (ctx == nullptr) return nil;
    return ctx->device;
}

MetalContext* metal_ctx_create() {
    auto* ctx = new MetalContext{};
    ctx->device = MTLCreateSystemDefaultDevice();
    if (ctx->device == nil) {
        delete ctx;
        return nullptr;
    }
    ctx->queue = [ctx->device newCommandQueue];
    if (ctx->queue == nil) {
        delete ctx;
        return nullptr;
    }
    return ctx;
}

void metal_ctx_destroy(MetalContext* ctx) {
    if (ctx == nullptr) return;
    ctx->queue  = nil;
    ctx->device = nil;
    delete ctx;
}

DeviceInfo metal_ctx_device_info(const MetalContext* ctx) {
    DeviceInfo info{};
    if (ctx == nullptr || ctx->device == nil) return info;

    info.name = [[ctx->device name] UTF8String];
    info.recommended_max_working_set_size =
        [ctx->device recommendedMaxWorkingSetSize];
    info.registry_id        = [ctx->device registryID];
    info.has_unified_memory = [ctx->device hasUnifiedMemory];
    info.max_threadgroup_memory_length =
        static_cast<std::uint32_t>([ctx->device maxThreadgroupMemoryLength]);
    info.max_threads_per_threadgroup_x =
        static_cast<std::uint32_t>([ctx->device maxThreadsPerThreadgroup].width);

    info.supports_family_apple7 =
        [ctx->device supportsFamily:MTLGPUFamilyApple7];
    info.supports_family_apple8 =
        [ctx->device supportsFamily:MTLGPUFamilyApple8];
    info.supports_family_apple9 =
        [ctx->device supportsFamily:MTLGPUFamilyApple9];

    return info;
}

void metal_ctx_print_info(const MetalContext* ctx) {
    const auto info = metal_ctx_device_info(ctx);

    std::printf("device                : %s\n", info.name.c_str());
    std::printf("registry id           : %llu\n",
                static_cast<unsigned long long>(info.registry_id));
    std::printf("unified memory        : %s\n",
                info.has_unified_memory ? "yes" : "no");
    std::printf("recommended max wset  : %.2f GB\n",
                static_cast<double>(info.recommended_max_working_set_size) /
                    (1024.0 * 1024.0 * 1024.0));
    std::printf("max threadgroup mem   : %u bytes\n",
                info.max_threadgroup_memory_length);
    std::printf("max threads/group (x) : %u\n",
                info.max_threads_per_threadgroup_x);
    std::printf("GPU family            : %s%s%s\n",
                info.supports_family_apple7 ? "Apple7 " : "",
                info.supports_family_apple8 ? "Apple8 " : "",
                info.supports_family_apple9 ? "Apple9 " : "");
    // M1: Apple7. M2: Apple8. M3 / M4: Apple9.
}

}  // namespace damascus
