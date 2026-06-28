# Lesson 012 — Compartment 5: Compute kernels (CPU reference)

Date: 2026-06-

## What I set out to do
<!-- The ARCHITECTURE.md rule: write a scalar CPU reference FIRST, then port each
     kernel to a Metal shader validated against this reference within 1e-4. Why
     this order — what does the reference buy you when a shader is wrong? DoD:
     each CPU kernel matches hand-computed values to f32 epsilon. -->


## Why accumulate in double
<!-- rmsnorm, matmul, and softmax all sum in `double` then cast to float at the
     end. "The reference is the oracle" — explain why the reference deliberately
     uses higher-precision accumulation than the Metal port will, and why that's
     correct rather than a mismatch waiting to happen. -->


## The matmul convention: y = x · Wᵀ
<!-- Weights are row-major [out_features, in_features] and the kernel computes
     out[m,n] = sum_k x[m,k]*w[n,k]. Why does the llama/GGUF weight layout mean a
     *transpose* falls out naturally from row-major iteration (w row j is one
     output's full weight vector)? -->


## RoPE, three ways — and which one the model actually needs
<!-- There are three entry points: rope (half-split / NEOX-HF), rope_interleaved
     (NORM / adjacent pairs), and rope_scaled (per-frequency divisors).
     1. What is the difference between half-split (j, j+d/2) and interleaved
        (2i, 2i+1) rotation?
     2. The crucial fact: llama.cpp's converter *permutes* the q/k weights so
        that interleaved rotation reproduces HF's rotate_half. So which variant
        does the forward pass call on raw GGUF weights, and why would calling the
        half-split one give wrong logits? -->


## llama3 RoPE frequency scaling (freq_factors)
<!-- rope_scaled divides each pair's frequency by freq_factors[j]. What does an
     empty freq_factors span reduce to, and where do the actual factors come from
     for Llama 3.2? -->


## Numerically-stable softmax
<!-- Why subtract max(x) before exp? What overflow does it prevent, and why is the
     result mathematically identical to naive softmax? -->


## SwiGLU and aliasing
<!-- silu(v) = v*sigmoid(v); out[i] = silu(gate[i]) * up[i]. The contract says
     `out may alias gate`. Why is that safe given the loop reads gate[i] and up[i]
     before writing out[i]? -->


## What's verified vs. not
<!-- Which kernels were checked against hand-computed values, by what tool
     (kernels_check / the Docker CPU path)? What is explicitly NOT done yet — the
     Metal shader ports and their 1e-4 validation against these references. -->


## What I'd revisit
<!-- The std::pow per-pair angle cost, fusing RoPE into QKV, anything that will
     matter once these become hot Metal kernels. -->
