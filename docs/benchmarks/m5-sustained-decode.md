# Sustained greedy decode on Apple M5

Single stream, Llama 3.2 3B Instruct, 24 GiB base M5, macOS 26.6.2. The fixed prompt is `The capital of France is` (six tokens including BOS). The first greedy token is ` Paris` (12366). Measure the next **512 tokens** with model and KV cache resident. Disable early stopping and time decode inside each engine, including sampling and draft verification. Exclude process launch, model load, prompt processing and text rendering.

## Measured throughput

| Tokens/s | Round 1 | Round 2 | Round 3 | Median |
| --- | ---: | ---: | ---: | ---: |
| Damascus | 79.00 | 83.39 | 83.24 | **83.24** |
| llama.cpp | 54.73 | 55.79 | 58.49 | **55.79** |
| MLX-LM | 60.55 | 66.61 | 66.70 | **66.61** |

The median improved from 50.25 to 83.24 tok/s for Damascus. Relative to the same-session baselines, Damascus achieved 1.25× MLX-LM throughput and 1.49× llama.cpp throughput. All 513 Damascus output IDs matched revision `84eea21` on the fixed prompt. [Timed samples](m5-sustained-decode-2026-11-20.json)

## Why the workload matters

The model continues with a list of national capitals. Damascus drafts likely continuations from earlier tokens in the same request, then runs the full model over the candidate rows together. It accepts only matching target argmax IDs, includes the target correction at the first mismatch, and resumes from the accepted KV prefix. The split-attention kernel partitions the growing context across SIMD groups. Set `DAMASCUS_NO_DRAFT=1` for ordinary greedy decode, or `DAMASCUS_TRACE_DECODE=1` for accepted draft counts. Batched and single-row floating-point kernels can differ near tied logits.

The original comparison fixes the competitors at ordinary greedy decoding with their original loaders and quantizations. Damascus and llama.cpp share the Q4_K_M GGUF. MLX-LM uses native 4-bit group-64 weights. Competitor speculative configurations are outside this measurement.

Two additional 512-token prompts show how the result changes on prose. Each cell is a single run and should be read as a spot check. Both full Damascus continuations matched its ordinary greedy path. [Control samples](m5-sustained-controls-2026-11-20.json)

| Prompt | Damascus tok/s | llama.cpp tok/s | MLX-LM tok/s |
| --- | ---: | ---: | ---: |
| Explain how a bicycle stays upright while moving. | 59.54 | 56.85 | 64.46 |
| Write the opening paragraph of a mystery set in a library. | 72.17 | 31.53 | 61.87 |

## What drafting contributed

With the same Damascus binary at revision `8b4e6f9`, the fixed prompt reached **81.08 tok/s with history drafting** and **54.63 tok/s with drafting disabled**, a 1.48× ratio of medians. Each configuration ran three times in the order on, off, off, on, on, off. All six generated sequences matched across all 513 token IDs. Each enabled run accepted 217 draft tokens across 156 verification batches. This measures the net effect of drafting and verification together. GPU memory traffic was not instrumented. [Six samples and output IDs](m5-draft-ablation.json)

To reproduce the paired comparison from the repository root, run `python3 tools/bench_drafting.py --damascus build/damascus --gguf "$GGUF"`. The harness checks token parity and verifies that the requested drafting mode actually ran.

## Reproduce

Use the [one-shot benchmark's pinned model files and MLX environment](m5-oneshot.md). Build Damascus in Release mode with `make -B -j`. Build a separate llama.cpp timing adapter from the same upstream revision. Its patch suppresses early stopping and per-token printing, and reports elapsed time after the first sample.

```sh
git -C "$LLAMA_REPO" fetch --depth=1 origin 76f186df71a41abb3ee20fffedee31ca5dbc7d9f
git -C "$LLAMA_REPO" worktree add --detach "$LLAMA_BENCH_SRC" 76f186df71a41abb3ee20fffedee31ca5dbc7d9f
git -C "$LLAMA_BENCH_SRC" apply "$PWD/tools/bench_decode_llama.patch"
cmake -S "$LLAMA_BENCH_SRC" -B "$LLAMA_BENCH_BUILD" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_METAL=ON -DLLAMA_BUILD_EXAMPLES=ON
cmake --build "$LLAMA_BENCH_BUILD" --target llama-simple -j

python3 tools/bench_decode.py --damascus build/damascus --gguf "$GGUF" \
  --llama "$LLAMA_BENCH_BUILD/bin/llama-simple" --mlx-python "$MLX_VENV/bin/python" \
  --mlx-model "$MLX_MODEL" > sustained.json
```

The harness runs a 64-token warmup for each engine, then three timed rounds in rotated order. Each process prefills its own prompt. All 512 timed tokens belong to one continuous sequence.
