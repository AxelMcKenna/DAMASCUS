# Lesson 005 — Compartment 3: GGUF metadata parser (part 2)

**Date:** 2026-06-16

## From one hardcoded KV to the full loop
- The fixed per-iteration prefix: read key (string) → read type
  tag (u32) → dispatch.
- Why the key needs no tag but the value does (the asymmetry, now
  generalized to a loop).

The parser moved from manually reading one kv pair to looping over metadata_kv_count.
Each iteration has same fixed prefix, read the key as a string, read the type-tag, based on the type-tag
parse and read the value.


## The type-tag switch (gguf_type, 0–12)
- Where you found the authoritative enum (the file path).
- Why several scalar cases collapse to read_primitive<T> + print.
- The default case: throw on an unknown tag — corrupt file, not
  programmer bug.

Found the authoritative enum in the llama.cpp/ggml/include/gguf.h file. Most scalar cases are simple read_primitive<T>
print the value, and let the cursor advance, an unknown tag should throw because it is either corrupted or unsupported.

## The array case (tag 9)
- Exact layout: element-type tag + count (u64) + count elements.
- Why it's variable-length — can't read a fixed N bytes like a
  u32.
- Not truly recursive (arrays don't nest), just a loop reusing
  the dispatcher.

An array value is laid out as element type-tag, uint64_t count, then count elements. It cannot be read as fixed size 
primitive because the element type and count are only known after reading the array header.

## Print is cosmetic; the read is the work
- The `print` flag gates std::cout only;
  read_primitive/read_string advance the
  cursor regardless. Why that kept 280k silent elements correct.

Print flag only controls std::cout. The parser still calls read_primitive or read_string, so even silent array elements are
consumed correctly and the cursor stays aligned
 
## Cursor == file size as a correctness test
- Why it held on the vocab file (metadata IS the whole file,
  tensor_count = 0).
- Why it does NOT hold on the 3B model (255 tensors + data blob
  still ahead:
  ~7.8 MB cursor vs ~2 GB file).

On the vocab GGUF, cursor == filesize worked because the file had tensor count of 0, so metadata was the whole file.
On the target 3B model, this will not hold, because there is tensor info after the metadata.

## The hardcoding trap
- Vocab file said 4096/32; real 3B said 3072/28. Same code,
  different file.
- Why a wrong hardcoded constant fails SILENTLY (wrong logits,
  not a crash) —
  the "silent Milestone-A failure" the docs warn about.

Because the model still loads, it just didn't load the target model because path was pointing to another model
so silently failing.

## What tripped me up
- (the path typo: 3.b vs 3b, wrong directory; whatever else was
  real)
Function ordering, typos - grrr 

## Open / next session
- Tensor-info section: name → n_dims → dims → ggml_type tag →
  offset.
- The SECOND enum (ggml_type ≠ gguf_type) — the trap.
- Alignment math (general.alignment) for the data-blob offsets.

Tensor info, second enum, alignment math for data-blob offsets.