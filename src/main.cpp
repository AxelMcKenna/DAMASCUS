// damascus entry point. The week-1 deliverable: prove the toolchain works,
// Metal initializes, and we can read device capabilities. Everything else
// gets layered on after this returns 0 reliably.

#include "metal_ctx.hpp"
#include "gguf.hpp"
#include "gpu.hpp"
#include "kernels.hpp"
#include "tokenizer.hpp"
#include "model.hpp"
#include "forward.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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
                 "  --logits-fnv <path> <text>   hash logits/hidden (perf-spike bit-identity gate)\n"
                 "  --generate <path> <n> <text>     greedy-decode n tokens (KV cache)\n"
                 "  --generate-ref <path> <n> <text> greedy-decode n tokens (full recompute; reference)\n"
                 "  --bench <path> [pp] [tg] [reps]  prefill/decode throughput (default 64 32 3)\n"
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

// Wall-clock milliseconds since an earlier time point.
static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}

// FNV-1a over raw bytes; a change in any logit bit changes the hash.
static uint64_t fnv1a(const void* data, std::size_t len) {
    const auto* p = static_cast<const unsigned char*>(data);
    uint64_t h = 1469598103934665603ULL;
    for (std::size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static int run_predict(const char* path, const std::string& text) {
    try {
        auto t0 = std::chrono::steady_clock::now();
        GGUFModel gguf = load_gguf(path);
        std::fprintf(stderr, "[timing] load_gguf: %.1f ms\n", ms_since(t0));

        Tokenizer tok(gguf.tokenizer);
        Model model(gguf);

        std::vector<int32_t> ids = tok.encode(text, /*add_bos=*/false);
        std::printf("prompt: \"%s\"  (%zu tokens)\n", text.c_str(), ids.size());
        std::printf("running forward pass over %u layers...\n", gguf.config.block_count);

        auto t1 = std::chrono::steady_clock::now();
        ForwardResult r = forward(model, ids);
        std::fprintf(stderr, "[timing] forward (%zu tokens): %.1f ms\n",
                     ids.size(), ms_since(t1));

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

// Greedy sampling: first index of the maximum logit.
static int32_t argmax(const float* row, std::size_t n) {
    std::size_t best = 0;
    for (std::size_t v = 1; v < n; ++v)
        if (row[v] > row[best]) best = v;
    return static_cast<int32_t>(best);
}

static KVCache make_cache(const Config& cfg, std::size_t max_seq) {
    const std::size_t head_dim = cfg.embedding_length / cfg.attention_head_count;
    return KVCache(max_seq, cfg.block_count,
                   cfg.attention_head_count_kv * head_dim);
}

static void print_generation(const Tokenizer& tok,
                             const std::vector<int32_t>& prompt,
                             const std::vector<int32_t>& generated) {
    std::printf("prompt ids: [");
    for (std::size_t i = 0; i < prompt.size(); ++i)
        std::printf("%s%d", i ? ", " : "", prompt[i]);
    std::printf("]\ngenerated ids: [");
    for (std::size_t i = 0; i < generated.size(); ++i)
        std::printf("%s%d", i ? ", " : "", generated[i]);
    std::printf("]\ntext: \"%s\"\n", tok.decode(generated).c_str());
}

// Greedy generation. use_kv=true: prefill once, then one forward_chunk per
// token. use_kv=false: the correctness reference — full recompute per token.
// Both print identical stdout, so `diff` gates KV-cache correctness.
static int run_generate(const char* path, int n_gen, const std::string& text,
                        bool use_kv) {
    try {
        auto t0 = std::chrono::steady_clock::now();
        GGUFModel gguf = load_gguf(path);
        Tokenizer tok(gguf.tokenizer);
        Model model(gguf);
        std::vector<int32_t> ids = tok.encode(text, /*add_bos=*/false);
        const double load_ms = ms_since(t0);

        std::vector<int32_t> generated;
        double prefill_ms = 0.0, decode_ms = 0.0;

        if (use_kv) {
            KVCache cache = make_cache(gguf.config, ids.size() + n_gen);

            auto tp = std::chrono::steady_clock::now();
            std::vector<float> logits = forward_chunk(model, cache, ids);
            prefill_ms = ms_since(tp);

            auto td = std::chrono::steady_clock::now();
            for (int i = 0; i < n_gen; ++i) {
                const int32_t next = argmax(logits.data(), logits.size());
                generated.push_back(next);
                if (i + 1 < n_gen)
                    logits = forward_chunk(model, cache,
                                           std::span<const int32_t>(&next, 1));
            }
            decode_ms = ms_since(td);
        } else {
            std::vector<int32_t> all = ids;
            auto td = std::chrono::steady_clock::now();
            for (int i = 0; i < n_gen; ++i) {
                ForwardResult r = forward(model, all);
                const float* row = r.logits.data() + (r.seq - 1) * r.vocab;
                const int32_t next = argmax(row, r.vocab);
                generated.push_back(next);
                all.push_back(next);
            }
            decode_ms = ms_since(td);
        }

        print_generation(tok, ids, generated);

        std::fprintf(stderr, "[timing] load+tokenize: %.1f ms\n", load_ms);
        if (use_kv) {
            std::fprintf(stderr,
                         "[timing] prefill: %zu tok in %.1f ms  (%.2f tok/s)\n",
                         ids.size(), prefill_ms,
                         1000.0 * static_cast<double>(ids.size()) / prefill_ms);
            std::fprintf(stderr, "[timing] ttft: %.1f ms (load + prefill + first sample)\n",
                         load_ms + prefill_ms);
        }
        if (n_gen > 0) {
            const double per_tok = decode_ms / n_gen;
            std::fprintf(stderr,
                         "[timing] decode: %d tok in %.1f ms  (%.1f ms/tok, %.2f tok/s)\n",
                         n_gen, decode_ms, per_tok, 1000.0 / per_tok);
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: generate failed: %s\n", e.what());
        return EXIT_FAILURE;
    }
}

// Prefill/decode throughput, measured separately (different regimes: prefill
// is batched GEMM work, decode is per-token GEMV work).
static int run_bench(const char* path, std::size_t pp, int tg, int reps) {
    try {
        GGUFModel gguf = load_gguf(path);
        Tokenizer tok(gguf.tokenizer);
        Model model(gguf);

        // Deterministic synthetic prompt of exactly pp tokens.
        std::vector<int32_t> base =
            tok.encode("The quick brown fox jumps over the lazy dog.", false);
        std::vector<int32_t> prompt(pp);
        for (std::size_t i = 0; i < pp; ++i) prompt[i] = base[i % base.size()];

        std::vector<double> pf_tps, dc_tps;
        for (int r = 0; r < reps; ++r) {
            KVCache cache = make_cache(gguf.config, pp + tg);

            auto tp = std::chrono::steady_clock::now();
            std::vector<float> logits = forward_chunk(model, cache, prompt);
            const double pf_ms = ms_since(tp);

            auto td = std::chrono::steady_clock::now();
            for (int i = 0; i < tg; ++i) {
                const int32_t next = argmax(logits.data(), logits.size());
                logits = forward_chunk(model, cache,
                                       std::span<const int32_t>(&next, 1));
            }
            const double dc_ms = ms_since(td);

            pf_tps.push_back(1000.0 * static_cast<double>(pp) / pf_ms);
            dc_tps.push_back(1000.0 * tg / dc_ms);
            std::printf("rep %d: prefill %7.2f tok/s (pp%zu, %.0f ms)   "
                        "decode %6.2f tok/s (tg%d, %.0f ms)\n",
                        r, pf_tps.back(), pp, pf_ms, dc_tps.back(), tg, dc_ms);
        }

        auto median = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
        };
        std::printf("median: prefill %.2f tok/s   decode %.2f tok/s\n",
                    median(pf_tps), median(dc_tps));
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: bench failed: %s\n", e.what());
        return EXIT_FAILURE;
    }
}

// GPU kernel gate (C5 discipline): every quantized matmul weight, GPU vs CPU
// on a deterministic pseudo-random x, max |diff| / max |ref| must be < 1e-4.
static int run_gpu_check(const char* path) {
    try {
        GGUFModel gguf = load_gguf(path);
        Model model(gguf);
        if (!gpu::init(model.blob_base(), model.blob_size())) {
            std::fprintf(stderr, "error: GPU unavailable\n");
            return EXIT_FAILURE;
        }

        std::vector<std::string> names = {"token_embd.weight"};
        for (int L : {0, 13, 27}) {
            const std::string p = "blk." + std::to_string(L) + ".";
            for (const char* s : {"attn_q.weight", "attn_k.weight", "attn_v.weight",
                                  "attn_output.weight", "ffn_gate.weight",
                                  "ffn_up.weight", "ffn_down.weight"})
                names.push_back(p + s);
        }

        int failures = 0;
        uint32_t rng = 0x12345678;
        for (const std::string& name : names) {
            WeightView w = model.view(name);
            const std::size_t k = static_cast<std::size_t>(w.dims.at(0));
            const std::size_t n = static_cast<std::size_t>(w.dims.at(1));

            std::vector<float> x(k);
            for (auto& v : x) {
                rng = rng * 1664525u + 1013904223u;
                v = static_cast<float>(static_cast<int32_t>(rng >> 8)) /
                    static_cast<float>(1 << 23);  // ~[-1, 1]
            }

            std::vector<float> ref(n), got(n);
            cpu::matmul_q(ref, x, w.type, w.data, 1, n, k);
            if (!gpu::matmul(got, x, w.type, w.data, 1, n, k)) {
                std::printf("  %-28s SKIP (no GPU kernel for type %u)\n",
                            name.c_str(), w.type);
                continue;
            }

            float max_ref = 0.0f, max_diff = 0.0f;
            for (std::size_t i = 0; i < n; ++i) {
                max_ref  = std::max(max_ref, std::fabs(ref[i]));
                max_diff = std::max(max_diff, std::fabs(ref[i] - got[i]));
            }
            const float rel = max_diff / (max_ref > 0.0f ? max_ref : 1.0f);
            const bool ok = rel < 1e-4f;
            if (!ok) ++failures;
            std::printf("  %-28s type=%2u [%zux%zu]  max_rel=%.2e  %s\n",
                        name.c_str(), w.type, n, k, rel, ok ? "PASS" : "FAIL");
        }
        std::printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
        return failures ? EXIT_FAILURE : EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: gpu-check failed: %s\n", e.what());
        return EXIT_FAILURE;
    }
}

// Correctness gate for the perf spike: hash every logit and hidden-state byte
// so any numerical drift from an optimization is detectable.
static int run_logits_fnv(const char* path, const std::string& text) {
    try {
        GGUFModel gguf = load_gguf(path);
        Tokenizer tok(gguf.tokenizer);
        Model model(gguf);

        std::vector<int32_t> ids = tok.encode(text, /*add_bos=*/false);
        ForwardResult r = forward(model, ids);

        std::printf("logits fnv1a: %016llx  (%zu floats)\n",
                    static_cast<unsigned long long>(
                        fnv1a(r.logits.data(), r.logits.size() * sizeof(float))),
                    r.logits.size());
        std::printf("hidden fnv1a: %016llx  (%zu floats)\n",
                    static_cast<unsigned long long>(
                        fnv1a(r.hidden.data(), r.hidden.size() * sizeof(float))),
                    r.hidden.size());
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: logits-fnv failed: %s\n", e.what());
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

    if (std::strcmp(argv[1], "--gpu-check") == 0) {
        if (argc < 3) {
            std::fprintf(stderr, "error: --gpu-check requires <path>\n\n");
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        return run_gpu_check(argv[2]);
    }

    if (std::strcmp(argv[1], "--generate") == 0 ||
        std::strcmp(argv[1], "--generate-ref") == 0) {
        if (argc < 5) {
            std::fprintf(stderr, "error: %s requires <path> <n> <text>\n\n", argv[1]);
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        return run_generate(argv[2], std::atoi(argv[3]), join_args(argc, argv, 4),
                            std::strcmp(argv[1], "--generate") == 0);
    }

    if (std::strcmp(argv[1], "--bench") == 0) {
        if (argc < 3) {
            std::fprintf(stderr, "error: --bench requires <path>\n\n");
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        const std::size_t pp = argc > 3 ? static_cast<std::size_t>(std::atoi(argv[3])) : 64;
        const int tg   = argc > 4 ? std::atoi(argv[4]) : 32;
        const int reps = argc > 5 ? std::atoi(argv[5]) : 3;
        return run_bench(argv[2], pp, tg, reps);
    }

    if (std::strcmp(argv[1], "--logits-fnv") == 0) {
        if (argc < 4) {
            std::fprintf(stderr, "error: --logits-fnv requires <path> <text>\n\n");
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        return run_logits_fnv(argv[2], join_args(argc, argv, 3));
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