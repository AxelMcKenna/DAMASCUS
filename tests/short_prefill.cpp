// End-to-end regression for the M5 short-prefill path. Compare every final
// logit with sequential quantized decode, then consume a cached continuation.
#include "forward.hpp"
#include "gpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

using namespace damascus;

static KVCache gpu_cache(const Model& model, std::size_t capacity) {
    const auto& cfg = model.config();
    const auto width = cfg.attention_head_count_kv *
                       (cfg.embedding_length / cfg.attention_head_count);
    const auto count = cfg.block_count * capacity * width;
    auto* k = gpu::kv_alloc(count);
    auto* v = gpu::kv_alloc(count);
    if (!k || !v) throw std::runtime_error("GPU cache unavailable");
    return KVCache(capacity, cfg.block_count, width, k, v);
}

static void compare(const std::vector<float>& actual, const std::vector<float>& expected) {
    float max_error = 0, max_ref = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]))
            throw std::runtime_error("nonfinite logits");
        max_error = std::max(max_error, std::abs(actual[i] - expected[i]));
        max_ref = std::max(max_ref, std::abs(expected[i]));
    }
    if (max_error > 1e-4f * max_ref) throw std::runtime_error("logits differ");
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    auto gguf = load_gguf(argv[1]);
    Model model(gguf);
    if (!gpu::init(model.mapping())) throw std::runtime_error("Metal unavailable");
    for (std::size_t n : {2, 3, 4, 5, 8, 15, 16}) {
        auto together = gpu_cache(model, n + 2);
        auto serial = gpu_cache(model, n + 2);
        std::vector<int32_t> prompt(n);
        for (std::size_t i = 0; i < n; ++i) prompt[i] = 100 + i * 317;
        const int32_t bos = gguf.tokenizer.bos_token_id;
        forward_chunk(model, together, {&bos, 1});
        forward_chunk(model, serial, {&bos, 1});
        auto result = forward_chunk(model, together, prompt);
        std::vector<float> ref;
        for (auto id : prompt) ref = forward_chunk(model, serial, {&id, 1});
        compare(result, ref);
        if (together.len != n + 1) throw std::runtime_error("cache position differs");
        compare(forward_chunk(model, together, {&bos, 1}),
                forward_chunk(model, serial, {&bos, 1}));
        std::printf("PASS %zu prompt rows at nonzero cache offset + continuation\n", n);
    }
}
