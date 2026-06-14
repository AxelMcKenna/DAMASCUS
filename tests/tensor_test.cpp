#include "tensor.hpp"
#include <cstdio>
#include <iostream>

using namespace rigel;


int main() {
    Tensor a({2, 3}, Dtype::f32);

    float* float_view = a.as_f32();
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
    std::cout << "Successfully read and wrote f32 elements!\n\n";

    std::cout << "Allocating i8 {2, 2} tensor to test safety guard...\n";
    Tensor b({2, 2}, Dtype::i8);

    std::printf("Attempting as_f32() on an i8 tensor (expect abort):\n");
    b.as_f32();   // dtype != f32 → assert fails → program aborts HERE
}
