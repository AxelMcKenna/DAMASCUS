// Reads a named tensor's bytes through its TensorInfo offset and prints the
// first 8 f32 values + the absolute file offset — to cross-check the C++
// loader's offset math against an independent reader (python gguf lib).
// Metal-free, runs on Linux/CPU in a container.
//   build: g++ -std=c++20 -Iinclude src/gguf/gguf.cpp tools/oracle/gguf_tensor_check.cpp -o chk
//   run:   ./chk <model.gguf> <tensor-name>
#include "gguf.hpp"

#include <cstdio>
#include <cstring>
#include <exception>
#include <string>

using namespace damascus;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <tensor-name>\n", argv[0]);
        return 2;
    }
    const std::string want = argv[2];
    try {
        GGUFModel model = load_gguf(argv[1]);

        const TensorInfo* t = nullptr;
        for (const auto& ti : model.tensors)
            if (ti.name == want) { t = &ti; break; }
        if (t == nullptr) {
            std::fprintf(stderr, "error: tensor %s not found\n", want.c_str());
            return 1;
        }

        // The exact pointer expression from lesson-010:
        //   file buffer base + tensor blob start + GGUF-relative offset
        const std::byte* first = model.data.data() + model.data_blob_start + t->offset;
        const unsigned long long abs_off = model.data_blob_start + t->offset;

        std::printf("tensor:        %s\n", t->name.c_str());
        std::printf("ggml type:     %u\n", t->type);
        std::printf("dims:          ");
        for (auto d : t->dims) std::printf("%lld ", static_cast<long long>(d));
        std::printf("\nrel offset:    %llu\n", static_cast<unsigned long long>(t->offset));
        std::printf("abs offset:    %llu\n", abs_off);

        // type 0 == GGML_TYPE_F32 -> safe to read as floats
        if (t->type == 0) {
            float v[8];
            std::memcpy(v, first, sizeof(v));
            std::printf("first8(f32):  ");
            for (float x : v) std::printf("%.12g ", x);
            std::printf("\n");
        } else {
            std::printf("(non-F32 type; printing first 8 raw bytes)\nfirst8(hex):  ");
            for (int i = 0; i < 8; i++)
                std::printf("%02x ", static_cast<unsigned>(std::to_integer<unsigned char>(first[i])));
            std::printf("\n");
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
