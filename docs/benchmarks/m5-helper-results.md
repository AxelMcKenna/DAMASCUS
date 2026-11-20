# Damascus with a 1B draft model on the M5

Damascus reached **47.83 tok/s** with a 1B GGUF helper on the frozen eight-prompt panel. MLX-LM with a 1B helper reached **76.89 tok/s**. Damascus led on 0 of eight prompts. Each score is the arithmetic mean of eight per-prompt medians. [Prompt lock](m5-spec-panel.json) · [All 72 timed samples and target-only checks](m5-helper-results.json)

| Median tok/s | Damascus 1B | MLX-LM 1B | llama.cpp n-gram |
| --- | ---: | ---: | ---: |
| debugging | 47.55 | 74.54 | 50.37 |
| email | 45.60 | 68.00 | 59.13 |
| analysis | 53.47 | 82.85 | 53.02 |
| planning | 48.98 | 81.21 | 55.95 |
| explanation | 50.75 | 82.37 | 57.21 |
| coding | 52.22 | 91.50 | 54.34 |
| editing | 44.87 | 77.35 | 62.06 |
| fiction | 39.23 | 57.33 | 51.27 |

All engines ran sequentially on the same 24 GiB base M5 under macOS 26.6.2, with the previous panel’s warmup, raw input tokens, rotation and fixed length of 512 output tokens after the first. Each engine generated its own greedy continuation with early stopping disabled. Both model drafters use Llama 3.2 1B Instruct and a two-token lookahead. Damascus uses the Q4_K_M GGUF draft from the earlier llama.cpp supplemental benchmark and the same Q4_K_M 3B target as llama.cpp. MLX uses its pinned native 4-bit models. The llama.cpp control uses the primary panel’s n-gram adapter. [Versions, weight hashes and builds](m5-spec-results.md)

Damascus accepted 313.0 draft tokens per run on average. MLX-LM accepted 306.9. Similar acceptance can yield different throughput because verification, draft inference and cache updates take time on each engine. The experiment measures their combined cost. The earlier Damascus history score was 64.27 tok/s in a different session.

## Verification and scope

The target verifies each proposed prefix with its own logits. At a mismatch it emits the target correction and rewinds the helper cache. After full acceptance it catches the helper up on the last accepted input before drafting again. Each timed row checked the canonical prompt IDs and 513 output IDs including the first token. In separate greedy target-only checks, all 513 IDs matched every repeat of helper generation on 8 of eight prompts. Batched and single-row floating-point evaluation can differ at close logits. The raw file records the output hashes. This comparison covers one model pair and one machine.

## Run

From the repository root, use the [panel environment and timing adapter](m5-spec-results.md) and build Damascus with `make -B -j`.

```sh
build/damascus --generate-helper "$GGUF" "$GGUF_DRAFT" 65 "The capital of France is"

python3 tools/bench_spec_panel.py --damascus build/damascus --damascus-helper "$GGUF_DRAFT" \
  --gguf "$GGUF" --llama "$LLAMA_PRIMARY_BUILD/bin/llama-speculative-simple" \
  --mlx-python "$MLX_VENV/bin/python" --target "$MLX_MODEL" --draft "$MLX_DRAFT" > helper.jsonl

python3 tests/helper_decode.py build/damascus "$GGUF" "$GGUF_DRAFT"
```

Helper generation is opt-in and uses BOS. It requires Metal, tied Llama GGUFs with matching token IDs and supported quantization, and at most 2048 input positions including generated context. Enable `DAMASCUS_TRACE_DECODE=1` for verification counts. The decode timer excludes loading and prefill for both models.
