#pragma once

// Thin C++ wrapper around MTLDevice + MTLCommandQueue, implemented in Obj-C++
// (metal_ctx.mm) so the rest of the codebase stays pure C++. Eventually this
// gets replaced by metal-cpp direct use, but this insulates main from Cocoa
// for now and is easier to mock for tests.

#include <cstdint>
#include <string>

namespace damascus {

// Snapshot of the device's capabilities. Captured once at startup.
struct DeviceInfo {
    std::string name;
    std::uint64_t recommended_max_working_set_size;  // bytes
    std::uint64_t registry_id;
    std::uint32_t max_threadgroup_memory_length;     // bytes per threadgroup
    std::uint32_t max_threads_per_threadgroup_x;
    bool has_unified_memory;
    bool supports_family_apple7;  // M1
    bool supports_family_apple8;  // M2
    bool supports_family_apple9;  // M3 / M4
};

// Opaque — definition lives in metal_ctx.mm.
struct MetalContext;

// Returns nullptr on failure. Caller owns; pair with metal_ctx_destroy.
MetalContext* metal_ctx_create();
void          metal_ctx_destroy(MetalContext* ctx);

DeviceInfo metal_ctx_device_info(const MetalContext* ctx);
void       metal_ctx_print_info(const MetalContext* ctx);

}  // namespace damascus
