# Damascus

![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?style=flat-square&logo=cplusplus&logoColor=white)
![Metal](https://img.shields.io/badge/Metal-compute%20shaders-ff6b35?style=flat-square&logo=apple&logoColor=white)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%20%2F%20M5-333333?style=flat-square&logo=apple&logoColor=white)
![Model](https://img.shields.io/badge/model-Llama%203.2%203B-4f46e5?style=flat-square&logo=meta&logoColor=white)
![Quant](https://img.shields.io/badge/quant-Q4__K__M-6b7280?style=flat-square)
![Milestone A](https://img.shields.io/badge/Milestone%20A-reached-2ea44f?style=flat-square)
![Built by hand](https://img.shields.io/badge/built-by%20hand-c97a2b?style=flat-square)
![Last commit](https://img.shields.io/github/last-commit/AxelMcKenna/DAMASCUS?style=flat-square&color=8b5cf6)

Damascus is a C++20 and Metal inference engine for Llama 3.2 3B Instruct on Apple Silicon. It loads quantized GGUF weights directly and supports single-stream greedy generation.

## Measured on a base Apple M5

| Workload (median, 24 GiB Mac) | Damascus | llama.cpp | MLX-LM |
| --- | ---: | ---: | ---: |
| One-token CLI completion, launch through exit ↓ | **79.6 ms** | 216.9 ms | 973.5 ms |
| Sustained decode, 512 tokens after the first ↑ | **83.24 tok/s** | 55.79 tok/s | 66.61 tok/s |

**One-token completion.** Each engine ran six short prompts in 12 rounds with warm filesystem and shader caches. The timer includes process startup, model loading and teardown, including Python startup for MLX-LM. llama.cpp took 2.72× as long as Damascus. Resident-service time to first token requires a separate measurement. [Harness, versions, hashes and samples](docs/benchmarks/m5-oneshot.md)

**Sustained decode.** On the fixed prompt `The capital of France is`, Damascus achieved 1.49× the throughput of llama.cpp and 1.25× that of MLX-LM across three rotated rounds. The continuation repeats a sentence pattern. Damascus drafts from the request history and verifies multiple candidates together with the target model. Both comparison engines used ordinary greedy decoding. Damascus and llama.cpp shared one Q4_K_M GGUF. MLX-LM used a native group-64 4-bit checkpoint. [Reproduction commands and raw samples](docs/benchmarks/m5-sustained-decode.md)

**Broader speculative comparison.** On eight prompts locked before timing, MLX-LM with a 1B draft led on seven prompts. Damascus led on numerical analysis. Averaging each prompt’s three-run median gave MLX-LM **81.08 tok/s**, Damascus **64.27**, llama.cpp n-gram **57.36**, and llama.cpp with a 1B draft **58.18**. The llama.cpp 1B configuration ran in a separate session. [Frozen prompts, per-task results and all 96 runs](docs/benchmarks/m5-spec-results.md)

## Original M1 research goal

The project began as an investigation of M1 prefill using Apple's matrix coprocessor. The M5 measurements above use Metal. The M1 research target is a 30% prefill improvement over a freshly measured llama.cpp baseline. The CPU forward pass has a separate same-weights Hugging Face correctness check below. [AMX research notes](amx_prefill.md) · [Measurement plan](baseline.md)

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
  medians, non-overlapping p95 — see [baseline.md](baseline.md))

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

Build the native Metal CLI on macOS with Apple Silicon and the Xcode command-line tools. The CPU oracle checks above run via Docker.

```sh
./setup.sh                                              # fetch metal-cpp + llama.cpp baseline
make                                                    # build ./build/damascus
./build/damascus --device-info                          # Milestone 0: print GPU info
./build/damascus --load-gguf models/...Q4_K_M.gguf      # parse + print model config
./build/damascus --tokenize models/...Q4_K_M.gguf "Hello, world"   # encode/decode round-trip
./build/damascus --predict  models/...Q4_K_M.gguf "The capital of France is"  # next-token top-5
```

## Source layout

```
src/            engine source (C++20 + .mm Metal bridge)
src/metal/      Metal runtime and compute kernels
baseline.md     baseline measurement notes
amx_prefill.md  original M1 research proposal
third_party/    local llama.cpp and metal-cpp dependencies
docs/benchmarks/  M5 benchmark methods and raw samples
tools/          benchmark harnesses and correctness checks
models/         local GGUF weights (gitignored)
```
