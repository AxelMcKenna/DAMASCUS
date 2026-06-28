// Standalone driver to exercise the C++ GGUF loader (Compartment 3) WITHOUT
// Metal — so it compiles and runs on Linux/CPU inside the oracle container,
// escaping the host EDR block. Mirrors main.cpp's --load-gguf printout.
//   build: g++ -std=c++20 -Iinclude src/gguf/gguf.cpp tools/oracle/gguf_smoke.cpp -o gguf_smoke
//   run:   ./gguf_smoke <model.gguf>
#include "gguf.hpp"

#include <cstdio>
#include <exception>

using namespace damascus;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    try {
        GGUFModel model = load_gguf(argv[1]);
        const Config& cfg = model.config;
        std::printf("--- Loaded GGUF ---\n");
        std::printf("Architecture:     %s\n", cfg.architecture.c_str());
        std::printf("Vocab size:       %u\n", cfg.vocab_size);
        std::printf("Layers:           %u\n", cfg.block_count);
        std::printf("Embedding length: %u\n", cfg.embedding_length);
        std::printf("Query heads:      %u\n", cfg.attention_head_count);
        std::printf("KV heads:         %u\n", cfg.attention_head_count_kv);
        std::printf("FFN length:       %u\n", cfg.feed_forward_length);
        std::printf("RMS epsilon:      %g\n", cfg.rms_norm_epsilon);
        std::printf("RoPE freq base:   %g\n", cfg.rope_freq_base);
        std::printf("Context length:   %u\n", cfg.context_length);
        std::printf("Tensor count:     %zu\n", model.tensors.size());
        std::printf("Data blob start:  %llu\n",
                    static_cast<unsigned long long>(model.data_blob_start));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: failed to load GGUF: %s\n", e.what());
        return 1;
    }
}
