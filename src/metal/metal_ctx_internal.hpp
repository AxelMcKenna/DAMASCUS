#pragma once

// Internal Obj-C++ interface to metal context
// Only include this from .mm files. Pure C++ must use metal_ctx.hpp

#include "metal_ctx.hpp"

#import <Metal/Metal.h>

namespace damascus {

id<MTLDevice> metal_ctx_device(const MetalContext* ctx);

}   // namespace damascus

