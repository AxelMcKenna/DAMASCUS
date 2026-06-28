// damascus entry point. The week-1 deliverable: prove the toolchain works,
// Metal initializes, and we can read device capabilities. Everything else
// gets layered on after this returns 0 reliably.

#include "metal_ctx.hpp"
#include "gguf.hpp"
#include "tokenizer.hpp"
#include "model.hpp"
#include "forward.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

using namespace damascus;

static void print_usage(const char* prog) {
    std::fprintf(stderr,
                 "usage: %s <command> [args]\n"
                 "\n"
                 "commands:\n"
                 "  --device-info            print Metal device info and exit\n"
                 "  --load-gguf <path>       load GGUF model and print parsed config\n"
                 "  --tokenize <path> <text> encode text -> ids, then round-trip decode\n"
                 "  --predict <path> <text>  run the forward pass; print next-token top-5\n"
                 "  --version                print version and exit\n",
                 prog);
}

// Join argv[start..argc) into a single space-separated string (the prompt).
static std::string join_args(int argc, char** argv, int start) {
    std::string text;
    for (int i = start; i < argc; ++i) {
        if (i > start) text += ' ';
        text += argv[i];
    }
    return text;
}

static int run_device_info() {
    auto* ctx = metal_ctx_create();

    if (ctx == nullptr) {
        std::fprintf(stderr,
                     "error: failed to create Metal context. "
                     "Is this running on Apple Silicon?\n");
        return EXIT_FAILURE;
    }

    metal_ctx_device_info(ctx);
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

static int run_tokenize(const char* path, const std::string& text) {
    try {
        GGUFModel gguf = load_gguf(path);
        Tokenizer tok(gguf.tokenizer);

        std::vector<int32_t> ids = tok.encode(text, /*add_bos=*/false);
        std::string back = tok.decode(ids);

        std::printf("text:  \"%s\"\n", text.c_str());
        std::printf("ids:   [");
        for (std::size_t i = 0; i < ids.size(); ++i)
            std::printf("%s%d", i ? ", " : "", ids[i]);
        std::printf("]  (%zu tokens)\n", ids.size());
        std::printf("decode: \"%s\"\n", back.c_str());
        std::printf("round-trip: %s\n", back == text ? "OK" : "MISMATCH");
        return back == text ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: tokenize failed: %s\n", e.what());
        return EXIT_FAILURE;
    }
}

static int run_predict(const char* path, const std::string& text) {
    try {
        GGUFModel gguf = load_gguf(path);
        Tokenizer tok(gguf.tokenizer);
        Model model(gguf);

        std::vector<int32_t> ids = tok.encode(text, /*add_bos=*/false);
        std::printf("prompt: \"%s\"  (%zu tokens)\n", text.c_str(), ids.size());
        std::printf("running forward pass over %u layers...\n", gguf.config.block_count);

        ForwardResult r = forward(model, ids);

        // softmax over the last-position logits, report the top 5.
        const float* row = r.logits.data() + (r.seq - 1) * r.vocab;
        std::vector<int> idx(r.vocab);
        for (std::size_t v = 0; v < r.vocab; ++v) idx[v] = static_cast<int>(v);
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                          [&](int a, int b) { return row[a] > row[b]; });

        float max_logit = row[idx[0]];
        double denom = 0.0;
        for (std::size_t v = 0; v < r.vocab; ++v) denom += std::exp(row[v] - max_logit);

        std::printf("next-token top-5:\n");
        for (int rank = 0; rank < 5; ++rank) {
            int id = idx[rank];
            double p = std::exp(row[id] - max_logit) / denom;
            std::printf("  %d  %6.2f%%  \"%s\"\n", id, 100.0 * p,
                        tok.decode({id}).c_str());
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: predict failed: %s\n", e.what());
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

    if (std::strcmp(argv[1], "--tokenize") == 0) {
        if (argc < 4) {
            std::fprintf(stderr, "error: --tokenize requires <path> <text>\n\n");
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        return run_tokenize(argv[2], join_args(argc, argv, 3));
    }

    if (std::strcmp(argv[1], "--predict") == 0) {
        if (argc < 4) {
            std::fprintf(stderr, "error: --predict requires <path> <text>\n\n");
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        return run_predict(argv[2], join_args(argc, argv, 3));
    }

    std::fprintf(stderr, "error: unknown command: %s\n\n", argv[1]);
    print_usage(argv[0]);
    return EXIT_FAILURE;
}