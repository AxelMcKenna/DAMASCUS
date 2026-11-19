# Speculative decode on eight everyday tasks

On one base Apple M5, **MLX-LM led at 81.08 tokens/s** on the locked score, followed by Damascus at 64.27. Damascus ranked 2 of four configurations. Each score averages eight per-prompt medians, giving each task equal weight. The eight hand-authored prompts model everyday requests. They were selected before timing and are a small convenience sample. [Locked prompts and protocol](m5-spec-panel.json) · [All 96 timed runs](m5-spec-results.json)

| Tokens/s | Damascus | llama.cpp n-gram | MLX-LM 1B draft | llama.cpp 1B draft |
| --- | ---: | ---: | ---: | ---: |
| debugging | 58.64 | 55.86 | 88.89 | 58.74 |
| email | 61.78 | 67.00 | 75.28 | 57.07 |
| analysis | 92.35 | 54.77 | 87.72 | 64.67 |
| planning | 67.31 | 57.61 | 79.92 | 59.98 |
| explanation | 59.01 | 52.57 | 81.02 | 56.77 |
| coding | 70.09 | 52.54 | 90.93 | 61.26 |
| editing | 51.97 | 63.49 | 77.28 | 56.80 |
| fiction | 53.05 | 55.02 | 67.62 | 50.12 |
| **Mean of prompt medians** | 64.27 | 57.36 | 81.08 | 58.18 |
| Pooled tokens / decode seconds | 62.28 | 56.64 | 80.14 | 57.69 |

## Workload and controls

The primary run uses the [published 3B M5 benchmark's hardware, target weights and engine versions](m5-oneshot.md). Each engine prefills identical raw prompt IDs including BOS, samples the first token, and then generates 512 more with greedy target sampling and early stopping disabled. The single-stream decode timer excludes loading, prompt processing and text rendering. Fixed-length raw completions can continue past a natural stopping point. Different quantizations and kernels can select different continuations.

The three primary engines ran serially with engine and prompt order rotated across three rounds. Each engine received one untimed 64-token warmup. Damascus uses its shipped adaptive history drafter. llama.cpp uses upstream `ngram-simple` with the release defaults. MLX-LM uses `speculative_generate_step` with the 4-bit 1B Instruct draft and its default two-token lookahead. The supplemental llama.cpp configuration uses the same draft-model family and two-token lookahead, in a separate three-round session after the primary comparison. The scores describe these exact configurations. [Supplemental configuration](m5-spec-supplement.md)

Raw data includes the decode duration, input IDs, first token, generated-sequence hash, and accepted draft tokens for each sample. Counts are reported by the engine and include the first output if accepted where applicable. All four configurations accepted draft tokens on the panel. A short verbose llama.cpp diagnostic confirmed full GPU offload and the n-gram implementation. The timed runs used the normal log level. During supplemental validation the upstream example interpreted a zero remaining draft budget as unlimited and emitted 515 outputs on the debugging request. The harness rejected it. The supplemental adapter skips drafting on the last target-only step, and the harness still requires exactly 513 outputs for each timed row. The primary run used the [original adapter at its measured revision](https://github.com/AxelMcKenna/DAMASCUS/blob/3ebab38/tools/bench_spec_llama.patch) and passed the same length check in all 72 runs.

## Reproduce

Use the [existing target checkpoint hashes and MLX environment](m5-oneshot.md). Build Damascus and use the [llama.cpp benchmark build instructions](m5-sustained-decode.md) in a fresh checkout of the pinned upstream revision. Apply `tools/bench_spec_llama.patch` to the checkout and build the target `llama-speculative-simple`. The patch records elapsed time after the first output, suppresses token printing during decode, and permits completion past EOS.

The MLX 1B draft uses revision `08231374eeacb049a0eade7922910865b8fce912`, safetensors SHA-256 `35e396644bca888eec399f9c0f843ec7fa78b8f8c5e06841661be62b4edf96dd`. The GGUF draft uses revision `067b946cf014b7c697f3654f621d577a3e3afd1c`, file `Llama-3.2-1B-Instruct-Q4_K_M.gguf`, SHA-256 `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83`. [MLX draft](https://huggingface.co/mlx-community/Llama-3.2-1B-Instruct-4bit/tree/08231374eeacb049a0eade7922910865b8fce912) · [GGUF draft](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/tree/067b946cf014b7c697f3654f621d577a3e3afd1c)

```sh
python3 tools/bench_spec_panel.py --damascus build/damascus --gguf "$GGUF" \
  --llama "$LLAMA_BENCH_BUILD/bin/llama-speculative-simple" \
  --mlx-python "$MLX_VENV/bin/python" --target "$MLX_MODEL" --draft "$MLX_DRAFT" > primary.jsonl
```

For the supplemental run, repeat with `--llama-draft "$GGUF_DRAFT"` and a separate output file. The harness verifies prompt IDs across primary engines and checks token counts. Each row contains throughput in tok/s. Take the median of the three rows for each engine and prompt, then average the eight medians.
