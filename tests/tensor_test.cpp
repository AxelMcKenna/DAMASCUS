#include <cassert>
#include <cmath>
#include <iostream>
#include <utility>

#include "metal_ctx.hpp"
#include "tensor.hpp"

using namespace damascus;

static bool nearly_equal(float a, float b) {
    return std::fabs(a - b) < 1e-6f;
}

int main() {
    std::cout << "--- Tensor Metal Smoke Test ---\n";

    // 1. Create Metal context
    std::cout << "[TEST] Creating MetalContext...\n";
    MetalContext* ctx = metal_ctx_create();
    assert(ctx != nullptr);
    std::cout << "[PASS] MetalContext created.\n\n";

    // 2. Allocate f32 Tensor {4}
    std::cout << "[TEST] Allocating Tensor shape {4}, dtype f32...\n";
    Tensor t(ctx, {4}, Dtype::f32);

    const std::size_t expected_bytes = 4 * 4;
    const std::size_t actual_bytes = t.size_in_bytes();

    std::cout << "  Expected size: " << expected_bytes << " bytes\n";
    std::cout << "  Actual size:   " << actual_bytes << " bytes\n";

    assert(actual_bytes == expected_bytes);
    assert(actual_bytes == 16);

    std::cout << "[PASS] Tensor size is 16 bytes.\n\n";

    // 3. Write known values
    std::cout << "[TEST] Writing values through as_f32()...\n";
    float* p = t.as_f32();
    assert(p != nullptr);

    p[0] = 0.0f;
    p[1] = 1.0f;
    p[2] = 2.0f;
    p[3] = 3.0f;

    std::cout << "[PASS] Values written.\n\n";

    // 4. Read them back
    std::cout << "[TEST] Reading values back...\n";
    assert(nearly_equal(p[0], 0.0f));
    assert(nearly_equal(p[1], 1.0f));
    assert(nearly_equal(p[2], 2.0f));
    assert(nearly_equal(p[3], 3.0f));

    std::cout << "  p[0] = " << p[0] << "\n";
    std::cout << "  p[1] = " << p[1] << "\n";
    std::cout << "  p[2] = " << p[2] << "\n";
    std::cout << "  p[3] = " << p[3] << "\n";

    std::cout << "[PASS] Values survived round-trip.\n\n";

    // 5. Move construct and verify moved-to tensor still owns the buffer
    std::cout << "[TEST] Move-constructing Tensor...\n";
    Tensor moved(std::move(t));

    float* moved_p = moved.as_f32();
    assert(moved_p != nullptr);

    assert(nearly_equal(moved_p[0], 0.0f));
    assert(nearly_equal(moved_p[1], 1.0f));
    assert(nearly_equal(moved_p[2], 2.0f));
    assert(nearly_equal(moved_p[3], 3.0f));

    std::cout << "[PASS] Move construction preserved buffer contents.\n\n";

    // 6. Move assignment and verify again
    std::cout << "[TEST] Move-assigning Tensor...\n";
    Tensor moved_again(ctx, {1}, Dtype::f32);
    moved_again = std::move(moved);

    float* moved_again_p = moved_again.as_f32();
    assert(moved_again_p != nullptr);

    assert(nearly_equal(moved_again_p[0], 0.0f));
    assert(nearly_equal(moved_again_p[1], 1.0f));
    assert(nearly_equal(moved_again_p[2], 2.0f));
    assert(nearly_equal(moved_again_p[3], 3.0f));

    std::cout << "[PASS] Move assignment preserved buffer contents.\n\n";

    // 7. Destroy Metal context
    metal_ctx_destroy(ctx);

    std::cout << "--- All Tensor smoke tests passed ---\n";
    return 0;
}