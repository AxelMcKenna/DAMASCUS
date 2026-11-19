# Spike report: `spike/decode-perf` (2026-07-02, Apple M5 Pro, 24 GB)

Goal: push decode throughput and TTFT toward the milestones without changing
outputs. All numbers measured on this machine, warm, interleaved with a fresh
`llama-bench` run of the same GGUF (llama.cpp build 277a105).

## Where it landed

| Metric | Before spike | After spike | llama.cpp (same run) | Milestone |
|---|---|---|---|---|
| Forward pass, 5 tok | 8 609 ms | 202 ms | — | — |
| Decode tg256 (median) | n/a (no decode path) | **49.0 tok/s** | 58.7 ± 0.5 | C ✅ (≥25.5) · D ❌ (needs ≥52.9, at 83.5%) |
| Prefill pp512 | n/a | **1 107 tok/s** | 1 531 ± 28 | — |
| Prefill pp1024 | n/a | **856 tok/s** | 1 513 ± 53 | E ❌ (needs ≥1 967, at 57%) |
| TTFT (short prompt, warm) | ~9.7 s | **~0.6 s** | — | — |

## Change log (each gated before the next)

1. **Fused dequant matmul** (`cpu::matmul_q`) — bit-identical (logits FNV
   `0c5857fb214f3640` unchanged). Kills ~12.8 GB f32 materialization/forward.
2. **Threaded matmul** over output rows — bit-identical. 6.0×.
3. **8-way fp64 accumulators** — reordered summation, NOT guaranteed
   bit-identical; verified byte-identical `--predict` and generations. 2.65×.
4. **KV cache + greedy decode** (`--generate`, `--bench`) — KV path verified
   token-identical vs full-recompute reference at tg24/tg32.
5. **Metal fused dequant-GEMV** (Q4_K/Q6_K in registers, fp32 accum) —
   `--gpu-check` max rel err ~1e-6 vs CPU (gate 1e-4); generation identical.
6. **Full-GPU decode step** — one command buffer per token; KV cache in shared
   MTLBuffer written directly by CPU prefill (unified memory).
7. **Vectorized kernels + fusions** — float4/uchar4 loads, RoPE+KV-store in
   one dispatch, residual-add and SwiGLU in gemv epilogues, 4 rows per
   threadgroup, on-GPU argmax + embedding so N tokens chain in command
   buffers with zero CPU round-trips.
8. **fp16 MPS prefill** — one-time on-GPU dequant of all layer weights to a
   5.6 GB half arena; prefill GEMMs via MPSMatrixMultiplication; norms/RoPE/
   attention fp32; KV cache stays fp32 so decode is untouched. Gate:
   long-prompt generation identical to the pure-CPU fp64 path.

## Correctness ledger

- CPU path (steps 1–4): byte-identical `--predict`; steps 1–2 bit-identical.
- GPU decode (5–7): greedy generations token-identical to CPU path; per-kernel
  1e-4 gate passed. fp32 accumulation ⇒ not bit-identical by construction.
- fp16 prefill (8): greedy generations identical on tested prompts. fp16
  GEMMs deviate most; a per-layer numeric diff vs the oracle has NOT been run.
- `tools/oracle` gates (C4 tokenizer, C5 kernels) pass natively on this Mac.

## Honest caveats / where the remaining gaps are

- **Milestone D (83.5%)**: the gap to llama.cpp decode is GEMV tuning (their
  q4_K kernel shapes, occupancy) — grind, not architecture.
- **Milestone E (57% at pp1024)**: the ARCHITECTURE.md thesis (idle AMX next
  to a weak M1 GPU) does not transfer to M5: there is no AMX (M3+ uses SME),
  and the M5 GPU with neural accelerators is already the strongest matmul
  unit — which llama.cpp drives well. Reaching +30% here would need a fused
  flash-attention prefill (our O(S²) attention kernel is what decays 1107 →
  856 between pp512 and pp1024), an all-half activation pipeline, and/or
  genuine GPU+SME overlap (FFN on SME while attention runs on GPU).
- Thermal/clock discipline dominates measurement: cold-start GPU clock ramp
  understates short runs by 2×; sustained load saturates the SoC. Every
  number above is from warm, interleaved, multi-rep runs (medians).
- mmap'ing the GGUF (load is ~0.4 s of TTFT) was deliberately not done — it
  restructures the loading path, which this spike was told not to touch.
