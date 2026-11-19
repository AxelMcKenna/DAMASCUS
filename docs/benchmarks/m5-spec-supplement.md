# Matched draft-model comparison

The primary panel and scoring rule are frozen in [the original manifest](m5-spec-panel.json). As an additional configuration, run llama.cpp with its native `draft-simple` implementation, Llama 3.2 1B Instruct Q4_K_M as the draft, and a maximum of two draft tokens per step, matching the MLX draft family and lookahead. Use the same target, prompts, token count, three prompt rotations, and one untimed warmup. Run this configuration separately after the primary panel. Report the separate timing session and both llama.cpp scores.
