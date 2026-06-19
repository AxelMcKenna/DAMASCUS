# Lesson 006 — Compartment 3: GGUF tensor-info section + data blob

**Date:** 2026-06-16

## The tensor-info loop

After the metadata loop, the cursor is positioned at the start of the tensor-info section. The parser then runs a loop `tensor_count` times, which is 255 for the target model.

Each tensor-info entry has this field order:

```text
name    -> GGUF string: u64 length + bytes
n_dims  -> uint32_t
dims    -> n_dims × int64_t
type    -> ggml_type, stored as uint32_t
offset  -> uint64_t
```

The variable-count field is `dims`, because the parser only knows how many dimension sizes to read after it reads `n_dims`. That means tensor-info cannot be read as one fixed-size struct; the parser needs an inner loop that reads exactly `n_dims` `int64_t` values and pushes them into `TensorInfo::dims`.

The storage design is:

```cpp
struct TensorInfo {
    std::string name;
    std::vector<int64_t> dims;
    uint32_t type;
    uint64_t offset;
};
```

All 255 entries are stored in a `std::vector<TensorInfo>` declared outside the loop, so the tensor directory survives after parsing and can later be used to load the actual weights.

## The two-enum trap

The load-bearing idea here is that tensor-info `type` is **not** a `gguf_type`.

Metadata values use `gguf_type`, defined in `gguf.h`. Tensor storage uses `ggml_type`, defined separately in `ggml.h`. They are different enums with different meanings.

The concrete footgun is tag `12`:

```text
gguf_type 12 = FLOAT64
ggml_type 12 = Q4_K
```

So if I reused my metadata `type_name()` function for tensor types, it would print a plausible lie. A Q4_K tensor would appear as FLOAT64, and the parser would not necessarily crash.

The fix is to keep the enum paths separate: either print the raw `ggml_type` number for now, or write a separate `ggml_type_name()` function later. Never pipe tensor storage types through the metadata type-name function.

## What the directory revealed about the model

The tensor shapes confirmed the real model architecture: embedding size 3072, 28 transformer blocks, and tensors such as:

```text
token_embd.weight    [3072, 128256]
ffn_gate tensors     [3072, 8192]
blk.N.* tensors      repeated across 28 blocks
```

The tensor type mix also confirmed Q4_K_M: most matrix weights are Q4_K, some tensors such as embeddings and `ffn_down` are Q6_K, and small precision-sensitive tensors such as norms and RoPE-related tensors are F32.

There was no separate `output.weight` among the 255 entries. That implies tied embeddings: the LM head reuses `token_embd.weight`, matching the architecture notes.

## Offset is relative to the data blob, not the file

The tensor-info `offset` is relative to the tensor data blob, not the start of the GGUF file.

So the absolute file position of a tensor is:

```text
absolute_tensor_position = data_blob_start + tensor.offset
```

This matters because metadata, tensor-info, and possible padding all come before the tensor data blob.

## Alignment and `align_up`

Because there is no `general.alignment` metadata key, the parser uses the GGUF default alignment value: 32 bytes.

The cursor after tensor-info is not necessarily the start of tensor data. The data blob is padded to a 32-byte boundary, so the parser computes:

```text
data_blob_start = align_up(cursor_after_tensor_infos, 32)
```

The formula is:

```text
align_up(x, a) = ((x + a - 1) / a) * a
```

The naive formula `(x / a) * a` rounds down. Adding `a - 1` before division pushes any non-aligned value into the next alignment bucket, while leaving already-aligned values unchanged.

## The close-the-loop check

The tensor-section close-the-loop check is:

```text
data_blob_start + last.offset + last.byte_size == file_size
```

For the last tensor, `output_norm.weight [3072]` as F32:

```text
last.byte_size = 3072 * 4 = 12288 bytes
```

Passing this check proves that all 255 tensor-info entries were walked with the correct byte widths. If any field had been read with the wrong size, the cursor would desync, `data_blob_start` would be wrong, and the final equality would fail.

But the check is silent on bugs that do not move the cursor. For example, reading the right number of dimensions but storing them incorrectly would still advance the cursor correctly, so the close-the-loop check could pass while the in-memory `TensorInfo` objects were semantically wrong.

## What tripped me up

One bug was declaring a struct member type without giving it a name:

```cpp
std::vector<int64_t>;
```

That declares nothing useful, so later code failed with errors like “no member named `dims`.” The correct declaration is:

```cpp
std::vector<int64_t> dims;
```

The bigger lesson is that a parser can consume bytes correctly while still interpreting or storing them incorrectly. Cursor checks prove the byte walk, not every semantic detail.

## Open / next session

Next is a general tensor `byte_size()` function.

This cannot just be:

```text
num_elements * sizeof(scalar)
```

because quantized formats like Q4_K and Q6_K are block-encoded. Their byte size depends on GGML’s quantization block layout, including packed weights and scale data.

After that, the parser can start loading tensor bytes into tensor objects, connecting this GGUF parser to the runtime tensor representation and eventually the Metal buffer upload path.
