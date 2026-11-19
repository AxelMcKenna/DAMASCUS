# M5 one-shot greedy completion

A local command starts, loads Llama 3.2 3B Instruct, tokenizes a short raw text prompt including BOS, computes the next greedy token, writes it to stdout, and exits. Elapsed time is measured in the parent process. It includes launching the executable or Python interpreter, imports, tokenizer and GPU setup, model access, inference, and teardown. Model downloads, builds, and the first warmup of each engine happen before measurement.

The workload represents a shell tool that releases its model between requests. A resident service, a machine reading weights from cold storage, and a long generation each need their own measurement. The six checked-in prompts contain 6–11 tokens including BOS. Each engine runs all six prompts in each round. Every block of six rounds uses all six engine orders, shuffled with a fixed seed.

Damascus and llama.cpp use the same Q4_K_M GGUF. The MLX-LM adapter uses the project's Python API, lazy loading, a single forward pass and an evaluated argmax, with a native group-quantized 4-bit checkpoint. The harness verifies input IDs and the sampled token for Damascus and MLX. For llama-simple it verifies the BOS-inclusive prompt and completion emitted by the binary, plus the full GPU-offload diagnostic.

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

The short-prefill test compares all final logits against one-token-at-a-time GPU decode, at a nonzero KV offset and again after a continuation. It covers partial four-row tiles and the 16-row limit. The maximum absolute difference must stay within 0.0001 times the largest absolute reference logit.
