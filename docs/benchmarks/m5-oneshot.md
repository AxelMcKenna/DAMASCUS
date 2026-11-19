# M5 one-shot greedy completion

A local command starts, loads Llama 3.2 3B Instruct, tokenizes a short raw text prompt including BOS, computes the next greedy token, writes it to stdout, and exits. Elapsed time is measured in the parent process. It includes launching the executable or Python interpreter, imports, tokenizer and GPU setup, model access, inference, and teardown. Model downloads, builds, and the first warmup of each engine happen before measurement.

The workload represents a shell tool that releases its model between requests. A resident service, a machine reading weights from cold storage, and a long generation each need their own measurement. The six checked-in prompts contain 6–11 tokens including BOS. Each engine runs all six prompts in each round. Every block of six rounds uses all six engine orders, shuffled with a fixed seed.

Damascus and llama.cpp use the same Q4_K_M GGUF. The MLX-LM adapter uses the project's Python API, lazy loading, a single forward pass and an evaluated argmax, with a native group-quantized 4-bit checkpoint. The harness verifies input IDs and the sampled token for Damascus and MLX. The timed harness requires the quantized Metal prefill diagnostic from Damascus and the GPU device in MLX. For llama-simple it verifies the BOS-inclusive prompt and completion emitted by the binary, plus the full GPU-offload diagnostic.

For generation after the first token, see the [512-token sustained decode benchmark](m5-sustained-decode.md).

## Short-prompt matrix kernel

The follow-up run at `a3df660` used the same M5, weights, binaries for llama.cpp and MLX-LM, six prompts, 12 rounds and balanced execution order. The engine now starts Metal setup while the CPU parses the mapped GGUF and tokenizes. A SIMD-group matrix kernel shares each decoded weight across up to 16 prompt rows, with FP32 inputs and accumulators. All 216 requests passed the original GPU-path and output checks. [Raw samples](m5-oneshot-matrix-2026-11-19.json)

| Engine | Median process latency | p95 |
| --- | ---: | ---: |
| damascus | 79.6 ms | 90.4 ms |
| llama.cpp | 216.9 ms | 235.4 ms |
| mlx-lm | 973.5 ms | 991.4 ms |

Damascus's median fell from 142.7 ms to 79.6 ms. The ratio of llama.cpp's median latency to Damascus's was **2.72×**. Of 72 paired requests, 71 finished in under 100 ms and 72 were at least twice as fast as llama.cpp. The p95 uses the nearest-rank sample percentile.

| Prompt | Damascus median | llama.cpp median |
| --- | ---: | ---: |
| In a standard deck of cards, there are four | 87.7 ms | 228.0 ms |
| The capital of France is | 78.3 ms | 211.5 ms |
| The capital of Japan is | 78.6 ms | 211.4 ms |
| The chemical symbol for oxygen is | 78.3 ms | 213.6 ms |
| The largest planet in our solar system is | 88.2 ms | 228.7 ms |
| The opposite of hot is | 78.0 ms | 211.1 ms |

## Previous run on November 19, 2026

Apple M5, 24 GiB unified memory, macOS 26.6.2 (25G83). Each engine completed 72 timed requests and returned the expected token every time. The p95 column uses the nearest-rank sample percentile. [Raw samples](m5-oneshot-2026-11-19.json)

| Engine | Median process latency | p95 |
| --- | ---: | ---: |
| damascus | 142.7 ms | 180.9 ms |
| llama.cpp | 219.1 ms | 242.5 ms |
| mlx-lm | 989.6 ms | 1079.0 ms |

Damascus finished first against llama.cpp in 72/72 paired requests and against MLX-LM in 72/72. The ratios of pooled median latency were 1.54× and 6.94× respectively. The MLX time includes Python and MLX-LM startup. The measurements are from one machine with warm filesystem and shader caches.

| Prompt | Damascus median (ms) | llama.cpp median (ms) | MLX-LM median (ms) |
| --- | ---: | ---: | ---: |
| In a standard deck of cards, there are four | 173.9 | 231.1 | 996.2 |
| The capital of France is | 134.8 | 220.3 | 989.0 |
| The capital of Japan is | 137.7 | 213.6 | 988.5 |
| The chemical symbol for oxygen is | 141.0 | 214.2 | 988.5 |
| The largest planet in our solar system is | 166.5 | 228.0 | 994.7 |
| The opposite of hot is | 138.1 | 218.0 | 994.2 |

- llama.cpp was built from commit `76f186df71a41abb3ee20fffedee31ca5dbc7d9f` with Release settings, native CPU optimizations and Metal enabled. The harness used its standalone `llama-simple` binary.
- The Llama 3.2 3B Instruct Q4_K_M GGUF SHA-256 was `6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff`.
- MLX 0.32.4, MLX-LM 0.32.2 and Python 3.11.15 used the [4-bit MLX checkpoint](https://huggingface.co/mlx-community/Llama-3.2-3B-Instruct-4bit/tree/7f0dc925e0d0afb0322d96f9255cfddf2ba5636e), group size 64. Its safetensors SHA-256 was `d75e1ee0ea653cc5b76191ec934c7c0d568e94d4e47846619f1f4bc715b7b265`.

## Reproduce

Build Damascus with `make -B -j`, build the upstream llama.cpp `llama-simple` target in Release mode with Metal enabled, and install the measured MLX versions in a separate Python environment. Download both model files before benchmarking. From the Damascus checkout, run

```sh
python3 tools/bench_oneshot.py \
  --damascus build/damascus --gguf "$GGUF" \
  --llama-simple "$LLAMA_BUILD/bin/llama-simple" \
  --mlx-python "$MLX_VENV/bin/python" --mlx-model "$MLX_MODEL" \
  > m5-oneshot.json
```

The default 12 rounds produce 72 samples per engine. The JSON includes each prompt, execution order, wall time in milliseconds and pooled median. Any completion mismatch or process failure makes the harness exit with an error.

## Correctness checks

```sh
make -B test
make build/short_prefill_test
build/short_prefill_test "$GGUF"
build/damascus --gpu-check "$GGUF"
```

The short-prefill test compares all final logits against one-token-at-a-time GPU decode, at a nonzero KV offset and again after a continuation. It covers partial and complete eight-row tiles, including a nine-row prompt and the 16-row limit. The maximum absolute difference must stay within 0.0001 times the largest absolute reference logit.
