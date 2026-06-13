# rigel — architecture

Technical source of truth: the design behind the nine compartments in `README.md`, the data
flow through a forward pass, the milestone acceptance tests, and the key decisions (with
rationale, so they can be revisited deliberately). Pairs with `benchmarks/baseline.md`
(measurement discipline) and `research/amx_prefill.md` (the AMX thesis in depth).

It is a *plan*. It will be wrong in places — correcting it as understanding improves is part
of the work.

---

## 1. Scope and non-goals

**In scope:** single-model, single-stream, single-machine inference of Llama 3.2 3B on one
Apple M1, fast enough to **beat `llama.cpp` on prefill** via a hand-rolled AMX path.

**Headline result lives here:** the fused dequant→AMX prefill kernel and GPU+AMX hybrid
scheduling (Compartment 9). This is *not* a deferral — it is the project's reason to exist.

**Out of scope (for now):** training/fine-tuning, batching multiple requests, multi-GPU,
server/HTTP layer, the ANE. Speculative decoding (1B draft on E-cores) and ANE are *stretch*
ideas noted in `research/amx_prefill.md`, pursued only after Milestone E holds.

---

## 2. Target model and reference oracle

Default target: **Llama 3.2 3B Instruct**, Q4_K_M GGUF (`bartowski/Llama-3.2-3B-Instruct-GGUF`).

| Property | Value (verify against the actual GGUF metadata — do not hardcode) |
|---|---|
| Layers | 28 |
| Hidden dim | 3072 |
| FFN (intermediate) dim | 8192 |
| Attention heads | 24 |
| KV heads (GQA) | 8 |
| Head dim | 128 |
| Vocab | 128256 |
| Embeddings | tied (no separate LM head weight) |
| RoPE theta | 500000 |
| Norm | RMSNorm |
| FFN | SwiGLU (gate / up / down) |

> Read every number from the GGUF in Compartment 3 and assert it. Hardcoding these is how
> silent Milestone-A failures happen.

**Reference oracle:** `llama.cpp` (in `third_party/`) is the correctness reference (dump its
logits / intermediate activations and diff) *and* the benchmark target for C–E. A Python+HF
script is the second, independent correctness oracle for Milestone A.

---

## 3. The nine compartments in detail

### 1. Platform
Metal device acquisition (`MTLCreateSystemDefaultDevice`), command queue, the Objective-C++
bridge (`.mm`), unified-memory `MTLBuffer` allocation (shared storage). Exposes
`--device-info`. The existing scaffold; must compile and run first.

*Key idea:* on Apple Silicon CPU and GPU share physical memory, so a shared-storage
`MTLBuffer` needs no copy — and the *same* memory is what AMX reads. This is the basis of the
zero-copy GPU+AMX hybrid later.

### 2. Tensor & memory
A `Tensor` owning shape, dtype, strides, and a backing `MTLBuffer` (or a non-owning view).
RAII for lifetime; CPU-side fill/readback for tests. Where modern-C++ habits get set (rule of
five, `std::span`, no raw `new`/`delete` in engine code).

### 3. Model loading
GGUF parser: header, KV metadata, tensor directory, then load tensor data into `Tensor`s.
Produces a `Model` (config + named, possibly-quantized weights). Byte-accurate on the format.

### 4. Tokenizer
Llama's byte-level BPE. Encode prompt → ids, decode ids → text; vocab + merges from the GGUF.
Correctness = exact round-trip vs HF.

### 5. Compute kernels
`.metal` shaders + host dispatch: GEMV (decode hot path), GEMM (prefill), RMSNorm, RoPE,
softmax, SwiGLU activation, residual add, embedding lookup. Each validated against a scalar
CPU reference within 1e-4 before use. Where threadgroups, SIMD-groups, coalescing get learned.

### 6. Transformer block
Compose kernels into one decoder layer:
`x → RMSNorm → QKV → RoPE → attn(KV cache, GQA) → Wo → +res → RMSNorm → SwiGLU → +res`.
Stack 28 layers; embed at front, final RMSNorm + (tied) LM head at back → logits.
**Milestone A passes here.**

### 7. Sampling
Logits → next token. Greedy first (deterministic, validates B), then temperature/top-k/top-p.
CPU is fine initially.

### 8. Runtime
The generation loop, two phases:
- **Prefill** — whole prompt at once, GEMM-heavy, compute-bound → the AMX target.
- **Decode** — one token at a time, GEMV-heavy, bandwidth-bound, reads/writes KV cache.
Owns KV-cache allocation/growth and batches GPU work into command buffers to avoid stalls.

### 9. Quantization & AMX  *(the headline)*
- Q4_K_M dequant + quantized matmul paths (block format with super-blocks of scales).
- **Fused dequant→AMX prefill:** unpack Q4_K_M blocks to FP16 *inside* the AMX outer-product
  loop, never materializing FP16 weights in memory. Per `research/amx_prefill.md`, this is the
  kernel that does not exist in any public M1 runtime.
- **GPU+AMX hybrid:** attention on the GPU (SIMD-friendly), FFN on AMX (outer-product-dense),
  coordinated through shared unified memory with no copies.
- AMX is driven via raw `.word`-encoded instructions (Corsix's reverse-engineered table);
  start from `cblas_sgemm` (Accelerate already uses AMX) to get the "through-the-framework"
  number, then hand-roll to beat it.

---

## 4. Data flow (one decode step)

```
token id ──▶ embedding lookup ──▶ x (hidden, dim 3072)
   for each of 28 layers:
      h     = RMSNorm(x)
      q,k,v = h·Wq, h·Wk, h·Wv          (GQA: 24 q-heads, 8 kv-heads, head_dim 128)
      q,k   = RoPE(q,k, pos; theta=5e5)
      append k,v to KV cache at `pos`
      a     = softmax(q·Kᵀ / √128)·V    (over cached positions)
      x     = x + a·Wo                   (residual)
      h     = RMSNorm(x)
      x     = x + (SiLU(h·Wgate) ⊙ (h·Wup))·Wdown   (SwiGLU + residual)
   logits = (RMSNorm(x))·Wembedᵀ          (tied embeddings — reuse embedding matrix)
next token = sample(logits)
```

Prefill runs the same graph over all prompt positions at once (matrices, not vectors),
filling the KV cache so decode can start. **Prefill is where AMX earns its keep.**

---

## 5. Milestone acceptance tests

| Milestone | Definition | Verification |
|---|---|---|
| **0** | Scaffold compiles, `--device-info` runs | `./setup.sh && make && ./build/rigel --device-info` prints device |
| **A** | Forward pass matches HF within 1e-3 | Diff our logits vs HF/`llama.cpp` for a fixed prompt; max abs err < 1e-3 |
| **B** | Coherent text generation | Greedy decode reads as fluent, sane English |
| **C** | Within 50% of `llama.cpp` decode | tok/s on same model+prompt+machine ≥ 0.5× baseline |
| **D** | Within 10% of `llama.cpp` decode | ≥ 0.9× baseline, reproducibly |
| **E** | **+30% prefill via AMX** | 3B Q4_K_M prefill tok/s ≥ 1.30× `llama.cpp`, 10-run medians, non-overlapping p95 |

Every speed number: same machine, model, quant, prompt, context length, stable thermal state,
reported with variance — not a single lucky run. Methodology is fixed in
`benchmarks/baseline.md`; baselines are measured **before the first kernel**.

---

## 6. Build order

1. **Compartment 1** — scaffold compiling + `--device-info` (Milestone 0).
2. **Baselines** — populate `benchmarks/baseline.md` from `llama.cpp` (the forever-numbers).
3. **Compartment 2** — tensors + memory, validated under ASan.
4. **Compartments 3 & 4** — load weights, tokenize (verify config + token round-trip; no compute).
5. **Compartment 5** — kernels one at a time, each vs CPU reference.
6. **Compartment 6** — assemble forward pass → **Milestone A**.
7. **Compartments 7 & 8** — sampling + generation loop → **Milestone B**.
8. **Compartment 9** — decode tuning (**C → D**), then the AMX prefill path → **Milestone E**.

Correctness before speed, always. No optimizing a kernel that hasn't first matched its
reference. No AMX work before a clean baseline exists to beat.

---

## 7. Key decisions log

- **3B over 1B.** Prefill matmuls must be large enough for AMX to win; 1B's are too small to
  showcase the thesis, while 3B still fits 16 GB.
- **Prefill is the win, not decode.** Decode is bandwidth-bound (~35 tok/s ceiling on M1 3B
  Q4); prefill is compute-bound with idle AMX next to the GPU. C/D target decode parity;
  **E** targets prefill, where the headroom actually is.
- **GGUF over safetensors.** Direct diffing against `llama.cpp` + reuse of its tooling.
- **Two correctness oracles (`llama.cpp` + HF).** Reduces risk of matching a shared bug.
- **AMX via raw opcodes, bootstrapped from Accelerate.** Get the BLAS-path number first
  (Accelerate already uses AMX), then hand-roll to skip dispatch overhead and fuse dequant.

_(Append new decisions below as the project proceeds.)_
