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

The pass proves that byte_size ran on all tensors (255) and cumulative offset arithmetic landed on the EOF.
A wrong 144/210 would compound over hundreds of tensors and miss EOF by a lot. So the quant constants are byte-verified, not assumed.
Tiling proves byte consistency, not that the reconstructed values are right - still needs a dequant-diff v. llama.cpp


## Zero padding — why this model tiles perfectly
Padding only appears when a tensor's byte size isn't a multiple of the 32 byte alignment.

## What tripped me up
References and subscription, forgot to delete old "close-the-loop" check.

## Open / next session
Compartment 2 - .hpp/.mm split - fun