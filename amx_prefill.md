# AMX-for-prefill: research notes

The differentiation lever for this project. Decode is bandwidth-bound and your ceiling on M1 is ~35 tok/s on 3B Q4 no matter how clever the kernel. Prefill is compute-bound and has real headroom — that's where the +30% number on the README comes from.

## What AMX is (and is not)

Apple Matrix coprocessor, sitting on the P-core cluster. Reverse-engineered by Corsix, dougallj, and others. Apple does not publish the ISA. Accelerate / vDSP / BLAS use it under the hood, but going through those frameworks costs you in dispatch overhead and prevents kernel fusion.

It is **not** Intel AMX (totally different ISA, confusingly same name) and it is **not** ARM SME (which is what M3/M4 use — SME is the public standardized successor). On M1 and M2, AMX is the proprietary version. **This is why M1 is a good target**: SME is the future, AMX is the now, and AMX kernels are not portable forward — but they're also not portable to anyone else's runtime, which is part of why the gap exists.

## Capabilities (M1, AMX1 generation)

- 32 × 64-byte X registers, 32 × 64-byte Y registers
- 64 × 64-byte Z accumulator (effectively a 32×32 FP32 or 64×64 INT8 tile)
- Outer-product instructions: `Z += X · Y^T` for various dtypes
- Supported types on M1: FP32, FP16, INT8/INT16, with various accumulator widths
- One AMX unit shared per P-core cluster (M1 has 1 cluster of 4 P-cores → one AMX)
- Theoretical throughput: ~1 TFLOPS FP32 outer products, ~2 TFLOPS FP16

For reference, M1 GPU is ~2.6 TFLOPS FP32. So AMX is comparable on FP16, **and** GPU and AMX can run in parallel. That's the unexploited win.

## Why prefill specifically

- Prefill processes N tokens at once → big matmuls (e.g., [N=1024, hidden=3072] × [hidden=3072, hidden=3072])
- These saturate compute, not bandwidth
- llama.cpp prefill on M1 runs ~150–200 tok/s for 3B; Apple SoCs have more compute available than that uses
- Hybrid scheduling — GPU does some FFN layers, AMX does others in parallel — could give 1.4–1.6× over GPU-only

## How to actually use it

Direct AMX instructions are encoded as 32-bit words in the `.word` directive (no real assembler support):

```cpp
// Enable AMX for this thread.
#define AMX_SET()  __asm__ volatile(".word 0x00201000" ::: "memory")
// Disable.
#define AMX_CLR()  __asm__ volatile(".word 0x00201001" ::: "memory")
```

Operations (load X, load Y, FMA into Z, store Z) follow a pattern of:

```
__asm__ volatile(".word (0x00201000 | (op << 5) | (operand))" ::: "memory")
```

Use Corsix's reference (link below) for the full opcode table. Wrap each operation in a typed C++ inline function so the call sites read normally.

State is per-thread but the AMX unit is shared per cluster. Multithreading needs care — pin to specific P-cores, don't oversubscribe.

## Practical work plan

1. **Reference matmul via Accelerate.** First, make `cblas_sgemm` your prefill matmul. Measure tok/s. This already uses AMX through BLAS — gives you the "what's possible going through the framework" number.
2. **Direct AMX FP16 matmul.** Hand-rolled outer-product loops for one specific shape (e.g., the FFN gate projection). Compare to Accelerate. You should be faster by 10–25% due to skipping BLAS dispatch overhead.
3. **Fused dequant + AMX.** The big win. Q4_K_M blocks unpack to FP16; do it inside the AMX loop so you skip writing FP16 weights to memory entirely. This is the kernel that doesn't exist anywhere public for M1.
4. **GPU + AMX hybrid.** Split the model layers across GPU (Metal) and AMX. Attention on GPU (favors GPU's SIMD), FFN on AMX (favors compute-dense outer products). Coordinate via shared unified memory — no copies.
5. **E-core spec decode.** Run a 1B draft model on E-cores via AMX while the 3B verifies on GPU. Energy + latency win.

## Required reading

- Corsix, "Apple AMX": https://github.com/corsix/amx — the canonical reverse-engineered reference
- dougallj's notes on M1 microarchitecture
- Phil Trottier's "Apple AMX Instruction Set" writeup
- Justine Tunney's tinyBLAS commits on the `whisper.cpp` and `llamafile` repos — shows AMX used in production
- Apple's Accelerate framework documentation — for the "high-level" path that already uses AMX

## Risks

- AMX is undocumented. Apple has changed opcodes between generations (M2 added new ops, M3 moved to SME). Your kernel is M1/M2-specific by definition.
- Multithreading AMX is fiddly. Easy to corrupt state across threads.
- Accelerate's BLAS is already quite good. Beating it by 10% takes real work; beating it by 30% requires the fused-dequant trick.
- Profiling AMX is harder than profiling GPU (no Instruments frame capture). Use `mach_absolute_time` micro-benchmarks of single shapes.

## Definition of success

If, by the end of Phase 2 (~weeks 9–16), the README's "+30% prefill" target on Llama-3.2-3B Q4_K_M holds up across 10-run medians with non-overlapping p95 ranges against the freshly-measured `llama.cpp` baseline, the project has its headline result. Everything else — decode tuning, cold start, ANE — is bonus material that makes the writeup stronger.
