// Compartment 5 verification driver (CPU, Metal-free — runs under gcc/Linux).
// Each CPU reference kernel is checked against values computed independently by
// hand / numpy. When the Metal ports land in C5, they get validated against
// THESE same kernels within 1e-4 (the rule from ARCHITECTURE.md).
//
//   (build line in the Makefile `kernels-check` target)

#include "kernels.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace damascus;

static int failures = 0;

static void approx(const char* name, const std::vector<float>& got,
                   const std::vector<float>& want, float tol = 1e-5f) {
    float max_err = 0.0f;
    bool size_ok = got.size() == want.size();
    if (size_ok) {
        for (std::size_t i = 0; i < got.size(); ++i)
            max_err = std::fmax(max_err, std::fabs(got[i] - want[i]));
    }
    bool ok = size_ok && max_err <= tol;
    std::printf("  %-16s max_err=%.2e  %s\n", name, max_err, ok ? "PASS" : "FAIL");
    if (!ok) {
        ++failures;
        if (!size_ok) std::printf("        size %zu != %zu\n", got.size(), want.size());
    }
}

int main() {
    std::printf("[Compartment 5 — CPU reference kernels]\n");

    // rmsnorm: x=[1,2,3,4], eps=0, weight=1 -> x / sqrt(7.5)
    {
        std::vector<float> x{1, 2, 3, 4}, w{1, 1, 1, 1}, out(4);
        cpu::rmsnorm(out, x, w, 0.0f);
        approx("rmsnorm", out, {0.3651484f, 0.7302967f, 1.0954451f, 1.4605935f});
    }

    // matmul: x[1x3] · Wᵀ, W is [2x3]
    {
        std::vector<float> x{1, 2, 3}, w{1, 0, -1, 2, 2, 2}, out(2);
        cpu::matmul(out, x, w, /*rows=*/1, /*n=*/2, /*k=*/3);
        approx("matmul", out, {-2.0f, 12.0f});
    }

    // embedding_lookup: row 2 of a 3x2 table
    {
        std::vector<float> table{10, 11, 20, 21, 30, 31}, out(2);
        cpu::embedding_lookup(out, table, /*token_id=*/2, /*dim=*/2);
        approx("embedding", out, {30.0f, 31.0f});
    }

    // rope: head_dim=4, pos=1, theta=10000, x=[1,0,0,1] (half-split pairs)
    {
        std::vector<float> x{1, 0, 0, 1};
        cpu::rope(x, /*head_dim=*/4, /*pos=*/1, /*theta=*/10000.0f);
        approx("rope", x, {0.5403023f, -0.0099998f, 0.8414710f, 0.9999500f});
    }

    // softmax over [1,2,3]
    {
        std::vector<float> x{1, 2, 3};
        cpu::softmax(x);
        approx("softmax", x, {0.0900306f, 0.2447285f, 0.6652409f});
    }

    // swiglu: silu(gate) * up
    {
        std::vector<float> gate{0, 1, -1}, up{2, 2, 2}, out(3);
        cpu::swiglu(out, gate, up);
        approx("swiglu", out, {0.0f, 1.4621172f, -0.5378828f});
    }

    // residual_add: in place
    {
        std::vector<float> out{1, 2, 3}, x{10, 20, 30};
        cpu::residual_add(out, x);
        approx("residual_add", out, {11.0f, 22.0f, 33.0f});
    }

    std::printf("\n%s  (%d failure%s)\n", failures ? "FAIL" : "PASS",
                failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
