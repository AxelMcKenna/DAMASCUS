# Road to the prefill ceiling (Milestone E on M5 Pro)

Written at the end of the `spike/decode-perf` branch (2026-07-02). This is the
engineering plan from the current ~1080 tok/s (pp512) to the measured ~2100
tok/s GEMM ceiling, which sits above the Milestone E bar (1.3 × llama.cpp ≈
1950 tok/s on this machine).

## 1. The ceiling, measured

Raw `MPSMatrixMultiplication` fp16 throughput at damascus's exact prefill
shapes (isolated, 30 reps, scratchpad `gemm_bench.mm`):

| shape (M×N×K) | role | TFLOP/s |
|---|---|---|
| 512×3072×3072 | q / o projection | 7.24 |
| 512×1024×3072 | k / v projection | 10.03 |
| 512×8192×3072 | gate / up | 14.71 |
| 512×3072×8192 | down | 14.56 |
| 1024×8192×3072 | gate / up | 15.49 |

Per-layer GEMM time at these rates (S=512): qo 2.67 ms + kv 0.64 ms + ffn
5.26 ms ≈ 8.6 ms → 28 layers ≈ 240 ms → **~2100 tok/s if GEMMs were the whole
cost**. Milestone E needs ~1950. So E is reachable on this GPU alone — no SME
hybrid required — iff everything that is not a GEMM shrinks to ≲10 % of the
pass.

Caveat to re-verify first: these are *independent* back-to-back GEMMs. Our
real chain is dependent (each GEMM reads the previous kernel's output), so
hazard barriers may erode the ceiling. **Measure a dependent-chain variant of
gemm_bench before investing** — if the dependent rate drops below ~11
TFLOP/s on the FFN shapes, the GPU-only ceiling falls under the E bar and the
fallbacks in §5 become the plan.

## 2. Where the current ~50 % is lost

Measured now (tight alternation): pp512 1080 vs llama.cpp 1343–1383 (0.79×);
pp1024 761 vs 1332 (0.57×). Two sinks, in order:

1. **Encoder churn around MPS.** Every `MPSMatrixMultiplication` encode ends
   our compute encoder and opens a new one — 4 GEMMs × 28 layers ≈ 112 splits
   per pass. Each split is a serialization point; the GPU drains between
   dependent stages instead of overlapping small kernels with GEMM tails.
2. **Flash attention at long S.** `attn_flash` (8-row query tiles, 32-position
   fp32 K/V chunks, per-lane dot products) is why pp1024 (0.57×) is so much
   worse than pp512 (0.79×). Attention FLOPs are only ~7 % of the pass at
   S=1024; the kernel spends its time in threadgroup-memory scalar traffic
   and per-chunk barriers, not math.

Eliminated already (measured, did *not* move throughput — do not revisit):
fp32↔fp16 conversion dispatches, separate q/k/v and gate/up GEMMs (now fused
via arena contiguity), CPU round-trips (whole pass is one command buffer).

## 3. Work item A — the fused dequant→simdgroup GEMM kernel

Replace MPS with one custom kernel so the entire 28-layer pass lives in a
single compute encoder. This is the Compartment-9 headline kernel
(ARCHITECTURE.md calls it "fused dequant→AMX prefill"; on M5 the matrix unit
is the GPU's simdgroup/neural-accelerator path, not AMX).

Spec:
- Inputs: raw Q4_K / Q6_K blocks straight from the GGUF blob buffer (this
  also deletes the 5.6 GB fp16 arena and its ~1 s startup dequant).
- Tile plan (start from `third_party/llama.cpp` `ggml/src/ggml-metal/`
  `kernel_mul_mm` and its templates — read that kernel first):
  - threadgroup computes a BM×BN output tile (theirs: 32×64), iterating BK=32
    slices of K;
  - dequantize the weight slice into threadgroup half tiles (reuse the
    per-block logic already written in `dq_q4k_h` / `dq_q6k_h`);
  - `simdgroup_half8x8` loads + `simdgroup_multiply_accumulate` into
    `simdgroup_float8x8` accumulators (fp32 accum — keeps numerics close to
    the current MPS path, same gates apply);
  - epilogue writes half; fold the residual add (`mode==1`) and SwiGLU
    (`mode==2`) epilogues in, like the decode GEMV already does.
- Acceptance: per-tensor max rel err < 1e-4 vs `cpu::matmul_q` (extend
  `--gpu-check` with a rows>1 case), then the standing generation-identity
  gates (short + long prompt vs `DAMASCUS_GPU=0`).
- Success bar: ≥ MPS's isolated TFLOP/s on the FFN shapes AND ≥ its qo-shape
  rate (7.2 TFLOP/s is MPS's weakest shape — a custom kernel tuned for M=512,
  N=3072 should beat it; that alone is worth ~1.5 ms/layer).

Expected recovery: the ~112 serialization points → ~0, plus small-shape GEMM
gains. This is the single largest remaining lever.

## 4. Work item B — flash attention v2

Target: attention ≤ 25 ms of a pp1024 pass (currently the dominant decay).

- K/V tiles in **half** inside threadgroup memory (fits CK=64 positions ×
  head_dim 128 in ~16 KB with the same 132-stride bank padding — the
  fp32/stride-128 bank-conflict lesson is already encoded in the kernel).
- 16-row query tiles (more K/V reuse per load; watch the 32 KB budget).
- QKᵀ and P·V products via `simdgroup_matrix` instead of per-lane loops —
  same instruction path as work item A, so build A first and reuse.
- Keep online softmax stats (m, l) fp32.
- Gate: unchanged generation-identity gates; if half K/V tiles flip a token
  on the long-prompt gate, fall back to fp32 tiles at CK=32 and take the
  smaller win.

## 5. If the ceiling doesn't hold

Re-measure the dependent-chain GEMM rate (§1 caveat) after A lands. If the
end-to-end number stalls below ~1600:

- **GPU+SME overlap** (ARCHITECTURE.md's hybrid, AMX→SME on M5): run FFN
  GEMMs via Accelerate `cblas_hgemm` on the CPU matrix unit for sequence
  sub-chunk N−1 while the GPU does attention/QKV for sub-chunk N. Get the
  "through-the-framework" SME number first, exactly as the architecture doc
  prescribes for AMX. Worth +25–45 % only if overlap is real; scheduling is
  research-grade work.
- **Measure E on an M1** — the machine the thesis was written for, where
  llama.cpp's GPU prefill ceiling is far lower and the CPU matrix unit
  (actual AMX) is genuinely idle headroom.

## 6. Claiming the milestone

No number counts until it survives `benchmarks/baseline.md` discipline — this
session proved why: the same binary measured 25→55 tok/s decode and llama.cpp
itself swung 31→59 with thermal state.

- Fresh boot, charger, lid open, 60 s idle before each run, nothing else
  running.
- Re-measure llama.cpp *in the same session*, alternated A/B/A, `-r 10`.
- Claim requires 10-run medians with non-overlapping p95 vs 1.3× the
  llama.cpp median. Same treatment to certify Milestone D (decode point
  estimate is ~0.93×, one window measured 1.05× — it likely passes, but has
  not been shown *reproducibly*).
