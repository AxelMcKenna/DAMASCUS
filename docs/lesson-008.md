# Lesson 008 — Compartment 2 design + Compartment 3 full-tiling verification

**Date:** 2026-06-18

## Why the Tensor type comes next
Moving to compartment 2 requires the bytes to be usable. Because the GGUF parser is what locates
and sizes the weights, completing this was necessary before moving to math/GPU programming.


## What a Tensor stores
Tensor stores data, shape and data type - all 3 are needed to provide a useable object. Eventually we will bring in stries
(non-contiguous layers/views)

## Why `data` is an MTLBuffer (the Apple-Silicon part)
Desigen choice here - either could have lets the tensor own its buffer and copy its slice in OR make the tensor a non-owning
view into the slurp.

What broke the tie was that the GPU must eventually read the bytes and a plain std::vector / std::byte* is CPU heap the metal GPU can't see.
So data = MTLBugger. Shared storage mode -> unified memory - CPU gets a pointer ([buffer contents]) to memcpy the slice in, GPU reads the same
physical memory, no copy.

## The lifetime contract — move-only (the load-bearing idea)
Back to the Ro5, for a unique-resource owner the policy is delete the two copies, write the two moves, destructor frees the memory.
Crucial as if copy ran, two Tensors hold same MTLBuffer handle -> both destructors release it leads to a double free.
Move only makes the two owners code uncompilable.

## The full-tiling check (the real close-the-loop for Compartment 3)
Tensors should tile the blob end to end, for each (sorted by offset asc) we compare to next offset for 3 cases: 

- equal is tight
- next greater - gap (padding)
- next lesser - overlap in bytes (FATAL)

btte

## Zero padding — why this model tiles perfectly
- Padding only appears when a tensor's BYTE SIZE isn't a multiple of the 32-byte alignment
  (NOT when a dim isn't) — offsets are already 32-aligned, so a non-32 size forces a gap.
- Counterexample that WOULD pad: a single Q4_K block, n=256 → 144 bytes, not a multiple of
  32 → needs 16 bytes padding. So "dim is a multiple of 32" does not guarantee no padding.
- Why THIS model has none: dims are large multiples of 256 with many factors of 2
  (3072/256=12, 8192/256=32, 128256/32=4008), so (n/256)*block_bytes lands on a multiple of
  32 every time. No remainder → no padding.

## What tripped me up
- current[i].type — current is a reference (one tensor), not subscriptable → current.type.
- Left the OLD single-tensor close-the-loop block in → redefinition of last_tensor /
  computed_file_end. New tiling check supersedes it; had to DELETE the old block.
- Then over-deleted: removed the sorted_tensors copy + std::sort with it → undeclared.
- last.name typo (× a few) → last_tensor.name. Lesson: rebuild & trust the COMPILER line,
  not a stale binary (a failed build left ./gguf_parser running old code that "passed").

## Open / next session
- Start Compartment 2 for real: the .hpp/.mm split (Tensor needs Metal → can't live in the
  standalone .cpp), construct a Tensor with a shared MTLBuffer, fill it, read it back.
- Then wire the GGUF loader to build a Tensor per directory entry.
- Eventually: dequantize a Q4_K tensor and diff values vs llama.cpp — the VALUE proof the
  tiling check can't give.