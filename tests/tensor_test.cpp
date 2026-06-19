#include <iostream>
#include <cstdio>
#include <cassert>
#include "tensor.hpp"

using namespace damascus;

int main() {
    std::cout << "--- Tensor Basic Test ---\n";

    // 1. Construct & Size Math
    std::cout << "[TEST] Constructing Tensor with shape {2, 3}, Dtype::f32...\n";
    Tensor t({2, 3}, Dtype::f32);

    std::size_t expected_bytes = 2 * 3 * 4;
    std::size_t actual_bytes = t.size_in_bytes();

    std::cout << "  Expected size: " << expected_bytes << " bytes\n";
    std::cout << "  Actual size:   " << actual_bytes << " bytes\n";
    assert(actual_bytes == expected_bytes);
    std::cout << "[PASS] size_in_bytes() calculation is correct.\n\n";

    // 2. User's Iteration & Write Test
    std::cout << "[TEST] Writing and reading through as_f32() using flat iteration...\n";
    float* float_view = t.as_f32();
    float_view[0] = 1.1f;
    float_view[1] = 2.2f;
    float_view[2] = 3.3f;
    float_view[3] = 4.4f;
    float_view[4] = 5.5f;
    float_view[5] = 6.6f;

    std::cout << "Tensor values (row-major flat print):\n";
    for (int i = 0; i < 6; ++i) {
        std::cout << "  Element [" << i << "]: " << float_view[i] << "\n";
    }
    std::cout << "[PASS] Successfully read and wrote f32 elements!\n\n";

    // 3. User's Guardrail Test
    std::cout << "[TEST] Allocating i8 {2, 2} tensor to test safety guard...\n";
    Tensor b({2, 2}, Dtype::i8);

    std::cout << "Attempting as_f32() on an i8 tensor (expect abort):\n";

    // This will trigger the assert(dtype == Dtype::f32) in tensor.hpp
    b.as_f32();

    // We should never reach this line
    std::cout << "FAIL: The assert did not trip!\n";
    return 0;
}