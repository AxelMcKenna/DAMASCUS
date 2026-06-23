// damascus entry point. The week-1 deliverable: prove the toolchain works,
// Metal initializes, and we can read device capabilities. Everything else
// gets layered on after this returns 0 reliably.

#include "metal_ctx.hpp"
#include "gguf.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

using namespace damascus;

static void print_usage(const char* prog) {
    std::fprintf(stderr,
                 "usage: %s <command> [args]\n"
                 "\n"
                 "commands:\n"
                 "  --device-info          print Metal device info and exit\n"
                 "  --load-gguf <path>     load GGUF model and print parsed config\n"
                 "  --version              print version and exit\n",
                 prog);
}

static int run_device_info() {
    auto* ctx = metal_ctx_create();

    if (ctx == nullptr) {
        std::fprintf(stderr,
                     "error: failed to create Metal context. "
                     "Is this running on Apple Silicon?\n");
        return EXIT_FAILURE;
    }

    metal_ctx_print_device_info(ctx);
    metal_ctx_destroy(ctx);

    return EXIT_SUCCESS;
}

static int run_load_gguf(const char* path) {
    try {
        GGUFModel model = load_gguf(path);

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

        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: failed to load GGUF: %s\n", e.what());
        return EXIT_FAILURE;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (std::strcmp(argv[1], "--version") == 0) {
        std::printf("damascus 0.0.1\n");
        return EXIT_SUCCESS;
    }

    if (std::strcmp(argv[1], "--device-info") == 0) {
        return run_device_info();
    }

    if (std::strcmp(argv[1], "--load-gguf") == 0) {
        if (argc < 3) {
            std::fprintf(stderr, "error: --load-gguf requires a path\n\n");
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }

        return run_load_gguf(argv[2]);
    }

    std::fprintf(stderr, "error: unknown command: %s\n\n", argv[1]);
    print_usage(argv[0]);
    return EXIT_FAILURE;
}