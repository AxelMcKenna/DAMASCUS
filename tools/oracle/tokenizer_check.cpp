// Compartment 4 verification driver (CPU, Metal-free — runs under gcc/Linux in
// the Docker escape). Loads the real GGUF, builds the byte-level BPE tokenizer,
// and checks it against the HuggingFace oracle ids plus round-trip invariants.
//
//   (build line in the Makefile `tok-check` target)

#include "gguf.hpp"
#include "tokenizer.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace damascus;

static int failures = 0;

static void check_ids(const Tokenizer& tok, const std::string& text,
                      const std::vector<int32_t>& want) {
    std::vector<int32_t> got = tok.encode(text, /*add_bos=*/false);
    bool ok = got == want;
    std::printf("  encode(%-28s) -> [", ('"' + text + '"').c_str());
    for (std::size_t i = 0; i < got.size(); ++i)
        std::printf("%s%d", i ? ", " : "", got[i]);
    std::printf("]  %s\n", ok ? "PASS" : "FAIL");
    if (!ok) {
        ++failures;
        std::printf("        expected [");
        for (std::size_t i = 0; i < want.size(); ++i)
            std::printf("%s%d", i ? ", " : "", want[i]);
        std::printf("]\n");
    }
}

static void check_roundtrip(const Tokenizer& tok, const std::string& text) {
    std::string back = tok.decode(tok.encode(text, /*add_bos=*/false));
    bool ok = back == text;
    std::printf("  roundtrip(%-28s) -> %-30s %s\n",
                ('"' + text + '"').c_str(), ('"' + back + '"').c_str(),
                ok ? "PASS" : "FAIL");
    if (!ok) ++failures;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }

    GGUFModel model = load_gguf(argv[1]);
    std::printf("tokenizer: model=%s pre=%s vocab=%zu merges=%zu bos=%u eos=%u\n\n",
                model.tokenizer.model.c_str(), model.tokenizer.pre.c_str(),
                model.tokenizer.tokens.size(), model.tokenizer.merges.size(),
                model.tokenizer.bos_token_id, model.tokenizer.eos_token_id);

    Tokenizer tok(model.tokenizer);

    // 1) Exact match against the HF oracle (tools/oracle/reference/manifest.json
    //    token_ids for this prompt). This is the Compartment-4 acceptance test.
    std::printf("[HF-oracle id match]\n");
    check_ids(tok, "The capital of France is", {791, 6864, 315, 9822, 374});

    // add_bos prepends begin_of_text (128000 for Llama 3).
    {
        std::vector<int32_t> got = tok.encode("The capital of France is", true);
        bool ok = !got.empty() &&
                  got.front() == static_cast<int32_t>(model.tokenizer.bos_token_id) &&
                  got.size() == 6;
        std::printf("  add_bos prepends %u: %s\n",
                    model.tokenizer.bos_token_id, ok ? "PASS" : "FAIL");
        if (!ok) ++failures;
    }

    // 2) Round-trip invariant on a spread of inputs (byte-level BPE is lossless).
    std::printf("\n[round-trip decode(encode(x)) == x]\n");
    check_roundtrip(tok, "The capital of France is");
    check_roundtrip(tok, "Hello, world!");
    check_roundtrip(tok, "don't  stop  believin'");      // contractions + 2x spaces
    check_roundtrip(tok, "numbers 1234567 and CAPS");
    check_roundtrip(tok, "tabs\tand\nnewlines\n");
    check_roundtrip(tok, "unicode: café — naïve — \xe6\x97\xa5\xe6\x9c\xac");

    std::printf("\n%s  (%d failure%s)\n", failures ? "FAIL" : "PASS",
                failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
