# Lesson 006 — Compartment 3: GGUF tensor-info section + data
blob

**Date:** 2026-06-16

## The tensor-info loop
- Runs tensor_count (255) times, starting right where the
  metadata loop left off.
- Per-entry field order + widths: name (string), n_dims (u32),
  dims (n_dims × int64), type (ggml_type u32), offset (u64).
- Why dims is the variable-count field (gated by n_dims) → inner
  loop, push_back.
- Storage design: struct TensorInfo + std::vector<TensorInfo>
  declared outside the loop.

After the metadata loop, the cursor is positioned at the start of the tensor info section. The parser then runs the loop
tensor_count times, which is 255 for the target model.

Each tensor info entry has this field order:

- name GGUF string: u64 length + bytes
- n_dims uint32t
- dims - n_dims x  int64t
- type - ggml_type stored as uint32t
- offset - uint64_t


## The two-enum trap (the load-bearing idea)
- type here is ggml_type, NOT gguf_type. Two separate enums in
  two files
  (gguf.h vs ggml.h).
- The concrete collision: tag 12 = FLOAT64 in gguf_type, but Q4_K
  in ggml_type.
  Piping it through type_name() prints a plausible lie with no
  error.
- Why I used a separate path (raw number / ggml_type_name), never
  crossed them.

The load-bearing idea here is that tensor-info type is not a gguf type.

Metadata values use gguf_type, defined in gguf.h. Tensor storage uses ggml_type, defined separately in ggml.h. They are different enums with different meanings.
The concrete footgun is tag 12:
gguf_type 12 = FLOAT64
ggml_type 12 = Q4_K
So if I reused my metadata type_name() function for tensor types, it would print a plausible lie. A Q4_K tensor would appear as FLOAT64, and the parser would not necessarily crash.


## What the directory revealed about the model
- dims confirm the architecture as real tensor shapes (token_embd
  [3072,128256],
  ffn_gate [3072,8192], 28× blk.N).
- The type mix IS "Q4_K_M": 12=Q4_K (most matmuls), 14=Q6_K
  (embeddings, ffn_down),
  0=F32 (norms, rope). The "_M" = medium mix.
- No output.weight in 255 entries → tied embeddings (LM head
  reuses token_embd),
  matching ARCHITECTURE.md.

The tensor shape confirmed the real model architecture: 3072 embedding size, 28 blocks, and tensors like token_embd.weight[3072, 128256]
The type mix also confirmed Q4_K_M, some Q6_K_M, and F32 for small precision-sensitive tensors

## offset is relative to the data blob, not the file
- gguf.h section 6.5: "offset in the tensor data binary blob."
- absolute position of tensor i = data_blob_start + offset_i.

Offset is relative to the blob not the file, meaning the absolute file position is data_blob_start + tensor.offset

## Alignment + align_up
- No general.alignment key present → GGUF default 32.
- Cursor after tensor-info isn't the data start; the blob is
  padded to a 32-byte
  boundary. data_blob_start = align_up(cursor_after_tensor_info,
  32).
- The formula: align_up(x,a) = ((x + a - 1) / a) * a, and WHY
  adding a-1 rounds up
  (naive (x/a)*a rounds down).

Because there is no general.alignment metadata key, the parser uses the GGUF default alignment value - 32. The data blob
starts at align_up(cursor_after_tensor_infos, 32), using ((x + a - 1) / a) * a; because the raw cursos may be followed by padding

## The close-the-loop check
- data_blob_start + last.offset + last.byte_size == file_size,
  exactly.
- last tensor (output_norm.weight, [3072], F32) → byte_size =
  3072*4 = 12288, by hand.
- What it PROVES about all 255 entries: a wrong field width
  anywhere desyncs the
  cursor → wrong data_blob_start → check fails. So landing on
  file_size confirms
  every entry was *read* with correct widths.
- What it's SILENT on: (answer your recall question here — a bug
  that doesn't move
  the cursor, e.g. mis-storing dims into the struct, would pass
  unnoticed).


Metadata values use gguf_type, defined in gguf.h. Tensor storage uses ggml_type, defined separately in ggml.h. They are different enums with different meanings.
The concrete footgun is tag 12:
gguf_type 12 = FLOAT64
ggml_type 12 = Q4_K
So if I reused my metadata type_name() function for tensor types, it would print a plausible lie. A Q4_K tensor would appear as FLOAT64, and the parser would not necessarily crash.


## What tripped me up
- struct member declared with a type but no name
  (std::vector<int64_t>; declares
  nothing) → "no member named 'dims'" downstream. (+ whatever
  else was real.)

The check is data_blob_start + last.offset + last.byte_size == file_size. For output_norm.weight [3072] as F32, last.byte_size = 3072 * 4 = 12288, and passing the check proves the tensor-info byte walk used the correct field widths.

## Open / next session
- General byte_size(): Q4_K/Q6_K block layout (super-blocks of
  scales), not n*4.
- Load tensor bytes into Tensor objects (links to Compartment 2 /
  MTLBuffer).

One bug was declaring a struct member type without a name, like std::vector<int64_t>;, which meant the struct had no dims member. The bigger lesson was that some bugs still pass the cursor check if they do not move the cursor, such as storing fields incorrectly after reading them.