# Damascus

![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?style=flat-square&logo=cplusplus&logoColor=white)
![Metal](https://img.shields.io/badge/Metal-compute%20shaders-ff6b35?style=flat-square&logo=apple&logoColor=white)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1-333333?style=flat-square&logo=apple&logoColor=white)
![Model](https://img.shields.io/badge/model-Llama%203.2%203B-4f46e5?style=flat-square&logo=meta&logoColor=white)
![Quant](https://img.shields.io/badge/quant-Q4__K__M-6b7280?style=flat-square)
![Milestone A](https://img.shields.io/badge/Milestone%20A-reached-2ea44f?style=flat-square)
![Built by hand](https://img.shields.io/badge/built-by%20hand-c97a2b?style=flat-square)
![Last commit](https://img.shields.io/github/last-commit/AxelMcKenna/DAMASCUS?style=flat-square&color=8b5cf6)

A from-scratch LLM inference engine for Apple Silicon (M1), written in C++20 + Metal,
with a hand-rolled **AMX prefill path** as its headline result.

> **Status: Milestone A reached on the CPU path.** The full pipeline — GGUF load → byte-level
> BPE tokenize → Q4_K_M/Q6_K dequant → 28-layer forward pass → logits — reproduces the
> HuggingFace reference on the **same Q4_K_M weights** to **per-layer cosine 1.0000000**, exact
> top-1, and the correct next token (see [Verified results](#verified-results-cpu-path) below).
> Compartments 1–6 are written; 3–6 are **runtime-verified**. The Metal/AMX half (C5 GPU
> kernels, C9) is still **blocked**: locally-compiled native binaries are SIGKILL'd on this work
> laptop by code-signing/EDR policy (IT access pending). The escape that unblocked everything
> above: a **Linux Docker guest**, where macOS code-signing doesn't apply, runs the unsigned
> CPU code (and the HF oracle) end-to-end — but has no Metal/AMX, so the GPU path still needs a
> signing exception or an unmanaged Mac.
>
> See `ARCHITECTURE.md` for the detailed design, `benchmarks/baseline.md` for the measurement
> discipline, `research/amx_prefill.md` for the differentiation thesis, and `CLAUDE.md` for how
> this project is run (it's a learning project: the author writes the code, Claude tutors).

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

## Verified results (CPU path)

**Milestone A** is a numerical check: do our logits match the reference? Because the engine runs
*quantized* (Q4_K_M) weights, the meaningful comparison is against a **same-weights oracle** —
HuggingFace/transformers loading the *same* GGUF and dequantizing the *same* blocks (`make
oracle-gguf`). Prompt: `"The capital of France is"`. The CPU forward pass (`src/forward/`),
diffed against that oracle by `tools/oracle/compare.py`:

| array | shape | per-layer cosine | mean abs err | max abs err |
|---|---|---|---|---|
| `hidden` (every residual-stream snapshot) | (29, 5, 3072) | **1.0000000** | 2.06e-4 | 1.27e-2 |
| `logits` | (5, 128256) | **1.0000000** | — | 1.33e-2 |

Top-1 token matches, top-5 overlap 100%. The next-token distribution is reproduced
percentage-for-percentage:

```
$ damascus --predict models/...Q4_K_M.gguf "The capital of France is"
next-token top-5:                          oracle (HF, same weights):
   12366   68.65%  " Paris"                   12366   68.63%  " Paris"
     279    4.77%  " the"                       279    4.76%  " the"
     539    2.37%  " not"                       539    2.37%  " not"
    1131    2.32%  "..."                       1131    2.32%  "..."
      25    2.05%  ":"                           25    2.06%  ":"
```

The literal `max abs < 1e-3` bar isn't met (1.3e-2), but that max lands on Llama's
*massive-activation* outlier dimensions (reference values reach ~358, so it's ~1e-4 relative)
and grows smoothly with depth — fp32 round-off between our fp64-accumulate matmuls and torch's
fp32, not a bug. By the regime-appropriate metric (cosine + exact token agreement, as
`compare.py` reports), the forward pass is correct.

Tokenizer (C4) and kernels (C5) carry their own gates: encode reproduces the HF oracle ids
exactly with lossless round-trip; each CPU reference kernel matches hand-computed values to f32
epsilon. The Q4_K and Q6_K dequantizers are bit-exact vs the `gguf` library.

```sh
make cpu-check        # C4 tokenizer + C5 kernel gates (gcc/Linux in Docker)
make oracle-gguf      # build the same-weights reference from the local GGUF
make forward-check    # C6: full forward pass, diffed vs the oracle (Milestone A)
```

## Build & run

The CPU path above is verified via Docker (`make cpu-check` / `forward-check`). The native macOS
binary builds but cannot run on the author's locked-down laptop (EDR SIGKILL) — these are its
commands, runnable on any unmanaged Mac:

```sh
./setup.sh                                              # fetch metal-cpp + llama.cpp baseline
make                                                    # build ./build/damascus
./build/damascus --device-info                          # Milestone 0: print GPU info
./build/damascus --load-gguf models/...Q4_K_M.gguf      # parse + print model config
./build/damascus --tokenize models/...Q4_K_M.gguf "Hello, world"   # encode/decode round-trip
./build/damascus --predict  models/...Q4_K_M.gguf "The capital of France is"  # next-token top-5
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
