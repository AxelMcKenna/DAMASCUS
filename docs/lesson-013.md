# Lesson 013 — Q4_K / Q6_K dequantization (pulled forward from Compartment 9)

Date: 2026-06-

## Why this was pulled forward
<!-- Quant is officially Compartment 9, but Milestone A (Compartment 6) needs f32
     weights while the GGUF stores them Q4_K_M-quantized. Explain the dependency:
     what would the forward pass have nothing to multiply with if this didn't
     exist yet? What is the relationship of THIS scalar reference to the
     project's headline fused dequant→AMX path (which validates against it)? -->


## Super-blocks: QK_K = 256
<!-- Both K-quants work in super-blocks of 256 weights. Write the two byte
     layouts from memory:
       block_q4_K = d(f16) dmin(f16) scales[12] qs[128]  -> 144 B
       block_q6_K = ql[128] qh[64] scales[16](int8) d(f16) -> 210 B
     Why must n_elements be a multiple of 256 to dequantize? -->


## The 6-bit packed scale/min (get_scale_min_k4)
<!-- Q4_K stores a 6-bit scale AND a 6-bit min per 32-weight sub-block — 8 pairs
     packed into just 12 bytes. Walk the bit-twiddling: why the j<4 branch reads
     bytes directly but j>=4 has to splice the high 2 bits from an earlier byte.
     What goes visibly wrong in the output if these are unpacked incorrectly? -->


## The dequant formulas
<!-- Q4_K:  w = d * sc * (q & 0xF) - dmin * m   (low then high nibble).
     Q6_K:  combine ql (4 bits) + qh (2 bits) -> 6-bit value, subtract the 32
            zero-point, scale by d * sc[is] (int8 scales).
     Explain the role of each term — the block scale d, the sub-block scale sc,
     and the min/zero-point. Why does Q6_K use a signed zero-point (-32) and
     Q4_K an additive min instead? -->


## half_to_float by hand
<!-- The f16 -> f32 conversion handles three cases: zero/subnormal, inf/nan
     (exp==0x1F), and normal. Why does the subnormal branch have to normalize the
     mantissa in a loop? And why does the final bit-pattern reach `float` via
     memcpy rather than a reinterpret_cast / union (strict aliasing)? -->


## Why this matches ggml exactly
<!-- The layouts follow ggml's block_q4_K / block_q6_K byte-for-byte. Why does
     "exactly" matter — what is the consequence of a single off-by-one in a
     stride when the same bytes will later be read by the AMX path? -->


## What's verified vs. not
<!-- How was correctness established — e.g. dequantizing a real GGUF tensor and
     diffing against a reference (the oracle / a known-good dequant), on the
     Docker CPU path? What is NOT done: the fused dequant→AMX kernel itself. -->


## What I'd revisit
<!-- Other quant types (Q4_0, Q8_0...), Q5_K, the eventual fused path, and whether
     to dequantize lazily vs. cache. -->
