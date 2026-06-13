# Baseline benchmarks

The numbers in this file are the entire point of the project. If they are not measured rigorously and reproducibly, the project has no claim to make. Fill these in **before** writing the first kernel.

## Hardware

- Chip: M1 (8-core CPU, 7- or 8-core GPU)
- RAM: 16 GB unified
- macOS: <version, e.g. 14.5>
- Thermal state: charger plugged in, lid open, idle for 60s before each run
- All other apps closed except Terminal

## Model

- `Llama-3.2-3B-Instruct`, Q4_K_M GGUF
- SHA256: <fill after download>
- Source: bartowski/Llama-3.2-3B-Instruct-GGUF on Hugging Face

Pin this exactly. If the source rehosts a different quant under the same name, your numbers become meaningless.

## Methodology

### Decode throughput

```
./third_party/llama.cpp/build/bin/llama-bench \
  -m models/llama-3.2-3b-instruct.Q4_K_M.gguf \
  -p 0 -n 256 -r 10 --output md
```

- `-p 0 -n 256`: skip prompt processing, generate 256 tokens
- `-r 10`: 10 repeats
- Report: median, p95, std-dev across runs
- Warmup: discard first run

### Prefill throughput

```
./third_party/llama.cpp/build/bin/llama-bench \
  -m models/llama-3.2-3b-instruct.Q4_K_M.gguf \
  -p 1024 -n 0 -r 10 --output md
```

- `-p 1024 -n 0`: prefill 1024 tokens, no generation
- Same warmup + repeat discipline

### Cold start to first token

```
time ./third_party/llama.cpp/build/bin/llama-cli \
  -m models/llama-3.2-3b-instruct.Q4_K_M.gguf \
  -p "Hello" -n 1 --no-display-prompt 2>&1 | tail -20
```

Measure wall-clock from process start to first token emitted. Run from cold (fresh terminal, no page cache — `sudo purge` between runs).

### Resident memory

```
/usr/bin/time -l ./third_party/llama.cpp/build/bin/llama-cli \
  -m models/llama-3.2-3b-instruct.Q4_K_M.gguf \
  -p "Hello" -n 256 2>&1 | grep "maximum resident"
```

### Thermal steady-state

After 60s of continuous decode, measure tok/s. Compare to cold tok/s. The drop is the throttling cost.

## Recorded baselines

Fill these in. Re-measure if macOS version changes or model file changes.

| Metric | Median | p95 | Std dev | Notes |
|---|---|---|---|---|
| 3B Q4_K_M decode tok/s (256 tok gen) | — | — | — | |
| 3B Q4_K_M prefill tok/s (1024 tok) | — | — | — | |
| 3B Q4_K_M cold start to first token (ms) | — | — | — | sudo purge between runs |
| 3B Q4_K_M peak resident (MB) | — | — | — | |
| 3B Q4_K_M tok/s after 60s sustained | — | — | — | thermal steady-state |

## Why this discipline matters

If your kernel "feels faster," you have nothing. If the benchmark shows +12% median with non-overlapping p95 ranges across 10 runs on a stable thermal state, you have something. Without this file populated honestly, every later claim in the project is rhetoric.

Re-run baselines whenever:
- macOS major version changes
- `llama.cpp` is updated (`git -C third_party/llama.cpp pull`)
- Model file changes
- A hardware change (battery health degradation can affect M1 thermal behavior)
