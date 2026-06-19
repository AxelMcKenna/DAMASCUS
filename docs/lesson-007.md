# Lesson 007 — Compartment 3: quantization & the type-general byte_size()

**Date:** 2026-06-17

## What quantization actually is
Quantization stores weights as a 4-bit index, relying on the affine reconstruction formula $w \approx q \cdot scale + min$ to recover the approximate value. The `min` offset is crucial because a scale alone would only yield non-negative ranges, whereas real weights span both sides of zero. Both the scale and minimum require higher precision (FP16) because a 4-bit range is too coarse to capture their exact bounds.

## The granularity tradeoff (why blocks exist)
Assigning a single scale and minimum to an entire tensor allows outliers to crush the resolution of all other weights, while assigning them per weight bloats the metadata and defeats compression entirely. Block quantization strikes the necessary middle ground by grouping weights into manageable chunks. For K-quants, this means a fine-grained level of 32 weights grouped into a 256-weight super-block.

## Q4_K's two-level super-block (the load-bearing idea)
A `Q4_K` super-block contains eight 32-weight sub-blocks, each with its own scale and minimum quantized down to 6 bits instead of full FP16. To maintain precision, two FP16 values (`d` and `dmin`) scale these 6-bit values back up during reconstruction. This two-level approach slashes the metadata from a naive 32 bytes down to just 16 bytes for every 256 weights, literally halving the overhead.

## Counting sizeof(block_q4_K) by hand
Mapping the `block_q4_K` C struct field by field reveals 4 bytes for the FP16 union, 12 bytes for the packed 6-bit scales, and 128 bytes for the 4-bit weights. The union creatively overlaps `d` and `dmin` to avoid extra padding, bringing the total to exactly 144 bytes per 256 weights. This represents roughly a 7× compression compared to the 1024 bytes required for the same weights in FP32.

## The universal size formula
Every `ggml` tensor size can be calculated using a single, universal formula: `(n / block_size) * block_bytes`. This abstraction unifies unquantized types like F32 (block size 1, 4 bytes) with complex blocks like Q4_K (block size 256, 144 bytes) and Q6_K (block size 256, 210 bytes). By isolating these two constants into a lookup table rather than inline magic numbers, adding future types like F16 becomes a trivial one-line update.

## The divisibility guard
Because the universal formula relies on integer division, any tensor size `n` that isn't a perfect multiple of the block size will silently truncate and corrupt all subsequent byte offsets. While the quantizer strictly enforces this divisibility at creation time, defensive parsing still demands an explicit `n % block_size == 0` check. Throwing an exception on violation protects the parser against malicious or corrupted files.

## What the close-the-loop check still can't see
Although the byte size function integrated perfectly into the close-the-loop check, the final tensor tested was F32, meaning only the unquantized code path was actually executed. The 144-byte and 210-byte constants for Q4_K and Q6_K remain unverified by this test, leaving a gap where incorrect constants would still register as a pass. True verification will require dequantizing an actual Q4_K tensor and diffing the resulting values against a reference implementation.

## What tripped me up
A typo in the struct declaration defined a member as two tokens (`uint64_t block type;`) but attempted to use it as `traits.type_size`, resulting in a "no member named" compilation error. This mirrors a previous lesson's issue with nameless members, reinforcing that struct member names must exactly match between declaration and their call sites.

## Open / next session
The next immediate step is to load the actual tensor bytes from the calculated offsets directly into memory. Once loaded, the focus will shift to dequantizing the Q4_K blocks to definitively prove out the 144-byte layout and two-level math in practice. This will successfully bridge the raw buffer loading of Compartment 2 with the eventual matrix multiplication operations of Compartment 9.