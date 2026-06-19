# Damascus

A from-scratch LLM inference engine for Apple Silicon (M1), written in C++20 + Metal,
with a hand-rolled **AMX prefill path** as its headline result.

> **Status:** Milestone 0 — scaffold written, not yet compiling clean. See `ARCHITECTURE.md`
> for the detailed design, `benchmarks/baseline.md` for the measurement discipline,
> `research/amx_prefill.md` for the differentiation thesis, and `CLAUDE.md` for how this
> project is run (it's a learning project: the author writes the code, Claude tutors).

## Goal

Run **Llama 3.2 3B Instruct (Q4_K_M GGUF)** end-to-end on an M1 — load weights, tokenize,
run the forward pass on the GPU via Metal compute shaders, generate coherent text — and then
beat `llama.cpp` where M1 actually has headroom: **prefill**, by driving the Apple Matrix
coprocessor (AMX) directly. Target headline: **+30% prefill throughput** over a freshly
measured `llama.cpp` baseline, proven with 10-run medians and non-overlapping p95 ranges.

The point is what gets learned building it, not the binary. Every line is written by hand
and understood before moving on.

## The thesis (why this can win)

Decode on M1 is **bandwidth-bound** — ~35 tok/s on 3B Q4 is roughly the ceiling no matter how
clever the kernel. Prefill is **compute-bound** and underused: `llama.cpp` prefill leaves M1's
compute on the table because the GPU and the AMX coprocessor can run *in parallel*, and AMX's
opcodes are undocumented so nobody's runtime exploits them. Damascus's bet is a fused
dequant→AMX prefill kernel (and eventually GPU+AMX hybrid layer scheduling) that no public M1
runtime has. Full reasoning in `research/amx_prefill.md`.

## Why these choices (changeable, but deliberate)

- **Target: Llama 3.2 3B Instruct, Q4_K_M.** Big enough that prefill matmuls are real work
  (AMX has something to chew on), small enough to iterate on a 16 GB M1. Standard modern
  architecture (RoPE, RMSNorm, SwiGLU, grouped-query attention), tied embeddings.
- **Format: GGUF.** Same format `llama.cpp` uses → direct logit/activation diffing and a
  shared benchmark target. Cost: we implement a binary GGUF parser.
- **M1 specifically.** AMX (not ARM SME) is the M1/M2 coprocessor; kernels are M1-specific by
  definition. That non-portability is *why* the performance gap exists and is exploitable.
- **Correctness oracles: `llama.cpp` + HuggingFace.** Two independent references so we don't
  accidentally reproduce a shared bug.

## The nine compartments

| # | Compartment | Responsibility | Success criterion |
|---|-------------|----------------|-------------------|
| 1 | **Platform** | Metal device/queue/buffer setup, unified memory, `--device-info` | Scaffold compiles; prints device, GPU family, memory |
| 2 | **Tensor & memory** | Tensor type (shape, dtype, stride, `MTLBuffer` backing), views, lifetime | Allocate/fill/read-back a tensor; clean under ASan |
| 3 | **Model loading** | Parse GGUF, map weights to tensors, expose config | Loads 3B; reported config matches HF |
| 4 | **Tokenizer** | Byte-level BPE encode/decode against the model vocab | Byte-identical round-trip vs HF tokenizer |
| 5 | **Compute kernels** | Metal shaders: GEMV/GEMM, RMSNorm, RoPE, softmax, SwiGLU, residual | Each kernel matches a CPU reference within 1e-4 |
| 6 | **Transformer block** | One decoder layer (GQA + KV cache, SwiGLU FFN), stacked | Full forward pass logits match HF within 1e-3 (**Milestone A**) |
| 7 | **Sampling** | Logits → token: greedy, temperature, top-k, top-p | Greedy is deterministic; output is coherent (**Milestone B**) |
| 8 | **Runtime** | Generation loop: prefill vs decode, KV-cache mgmt, command-buffer scheduling | Sustained generation without recompute blowup |
| 9 | **Quantization & AMX** | Q4_K_M dequant; quantized matmul; **fused dequant→AMX prefill**; GPU+AMX hybrid | Quant within accuracy budget; AMX prefill beats baseline (**→ E**) |

## Milestones

- **A** — forward pass matches HuggingFace within 1e-3
- **B** — coherent text generation
- **C** — within 50% of `llama.cpp` decode throughput
- **D** — within 10% of `llama.cpp` decode throughput
- **E** — **+30% prefill** over `llama.cpp` on 3B Q4_K_M via AMX, reproducibly (10-run
  medians, non-overlapping p95 — see `benchmarks/baseline.md`)

## Build & run (target interface — not yet working)

```sh
./setup.sh                                # fetch metal-cpp + build llama.cpp baseline
make                                      # build ./build/damascus
./build/damascus --device-info            # Milestone 0: print GPU info
./build/damascus -m models/...Q4_K_M.gguf -p "Hello"   # later: generate
```

## Layout (planned)

```
src/            engine source (C++20 + .mm Metal bridge)
shaders/        .metal compute kernels
benchmarks/     baseline.md — the forever-baselines; fill before the first kernel
research/       amx_prefill.md — the AMX differentiation thesis & opcode notes
third_party/    llama.cpp (oracle + benchmark), metal-cpp
docs/progress/  lesson_NNN.md — author's own-words writeups, one per session
models/         GGUF weights (gitignored)
```
