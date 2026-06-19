// damascus entry point. The week-1 deliverable: prove the toolchain works,
// Metal initializes, and we can read device capabilities. Everything else
// gets layered on after this returns 0 reliably.

#include "metal_ctx.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace damascus;

static void print_usage(const char* prog) {
    std::fprintf(stderr,
                 "usage: %s <command>\n"
                 "\n"
                 "commands:\n"
                 "  --device-info       print Metal device info and exit\n"
                 "  --version           print version and exit\n",
                 prog);
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
        auto* ctx = metal_ctx_create();
        if (ctx == nullptr) {
            std::fprintf(stderr,
                         "error: failed to create Metal context. "
                         "Is this running on Apple Silicon?\n");
            return EXIT_FAILURE;
        }
        metal_ctx_print_info(ctx);
        metal_ctx_destroy(ctx);
        return EXIT_SUCCESS;
    }

    print_usage(argv[0]);
    return EXIT_FAILURE;
}
