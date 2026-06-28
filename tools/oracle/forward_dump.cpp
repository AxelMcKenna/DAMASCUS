// Compartment 6 driver (CPU, Metal-free — runs under gcc/Linux in Docker).
// Loads the GGUF, dequantizes weights, runs the full forward pass over the
// SAME token ids the oracle used (read from <ref>/tokens.npy to isolate C6 from
// the tokenizer), and writes hidden/logits .npy in the oracle layout. Then
// compare.py gates Milestone A against the same-weights oracle.
//
//   build line in the Makefile `forward-check` target.

#include "forward.hpp"
#include "gguf.hpp"
#include "model.hpp"
#include "npy.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace damascus;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <ref_dir> <out_dir>\n", argv[0]);
        return 2;
    }
    const std::string gguf_path = argv[1];
    const std::string ref_dir = argv[2];
    const std::string out_dir = argv[3];

    std::vector<int32_t> tokens = npy::read_i32(ref_dir + "/tokens.npy");
    std::printf("tokens (%zu): [", tokens.size());
    for (std::size_t i = 0; i < tokens.size(); ++i)
        std::printf("%s%d", i ? ", " : "", tokens[i]);
    std::printf("]\n");

    std::printf("loading + dequantizing weights...\n");
    GGUFModel gguf = load_gguf(gguf_path);
    Model model(gguf);

    std::printf("running forward pass (%u layers)...\n", gguf.config.block_count);
    ForwardResult r = forward(model, tokens);

    npy::write_f32(out_dir + "/tokens.npy",
                   std::vector<float>(tokens.begin(), tokens.end()), {tokens.size()});
    npy::write_f32(out_dir + "/hidden.npy", r.hidden, {r.n_layers + 1, r.seq, r.d});
    npy::write_f32(out_dir + "/logits.npy", r.logits, {r.seq, r.vocab});

    // quick local readout: argmax of the last-position logits
    std::size_t best = 0;
    for (std::size_t v = 1; v < r.vocab; ++v)
        if (r.logits[(r.seq - 1) * r.vocab + v] > r.logits[(r.seq - 1) * r.vocab + best]) best = v;
    std::printf("wrote %s/{tokens,hidden,logits}.npy  hidden[%zu,%zu,%zu] logits[%zu,%zu]\n",
                out_dir.c_str(), r.n_layers + 1, r.seq, r.d, r.seq, r.vocab);
    std::printf("last-token argmax id = %zu\n", best);
    return 0;
}
