# Lesson 003 — Compartment 3: the GGUF format

**Date:** 2026-06-15

## Binary parsing foundation
  <!-- file = flat array of bytes; a "format" is a layout contract that gives bytes meaning;
       same bytes mean different things under different contracts -->
To read these files we have to interpret as bytes, the GGUF format has a specific contract that model files must
abide to which allows us to write a universal parser for GGUF files and therefore LLMs

## Endianness & reading hex
  <!-- little-endian = least-significant byte first; why the zeros in 03 00 00 00 didn't change the
  value -->
  <!-- hex->decimal: 0x51 = 5*16 + 1 = 81 (the slip I made) -->
Little endian means LSB first and is why the zeros in 03 00 00 00 don't change the value as they are
placeholders.

## The four sections of a GGUF file
  <!-- in order, and what each is for -->
Header, KV Metadata Block, Tensor Information Block, Tensor Data

### 1. Header
  <!-- magic "GGUF", version (u32), tensor_count (u64), metadata_kv_count (u64) — FIXED layout -->

Magic number - ASCII byte matching GGUF
Version - A 32-bit unsigned integer representing the GGUF spec - 3 for modern files
Tensor Count - 64-bit unsigned integer stating total number of tensors in section 3 and 4 of file
Metadata_KV_count - 64-bit unsigned integer stating exact total number of key-value pairs contained in section 2

### 2. Metadata key-value pairs
  <!-- self-describing loop: key-string, type-tag, value; how one pair is read byte by byte;
       the recursive array case (element type, then count, then elements) -->

Key length - 64 bit integer stating size of the key name
Key Name - A UTF8 string
Value Type - A 32-bit integer indicating the data type
Value Data - The actual configuration data matching that type


### 3. Tensor info (directory)
  <!-- per entry: name, n_dims, dims, dtype, offset; it's a table of contents, NOT the data;
       offset = a "page number" into the data blob -->
Name
Dimensions count
Dimensions
Type - F32, F16, I8, I4
Offset - byte offset relative to start of Tensor Data section, where tensor's raw binary begins

### 4. Tensor data
  <!-- the weight blob; offsets are RELATIVE to the data-section start, not the file; why relative -->
Relative offsets provide resilience to metadata changes, streamlined virtual memory mapping - since inference engines don't 
typically read the entire GGUF file into RAM relative offsets are needed


## The two primitives (write this most carefully)
  <!-- type tag: include when the type is NOT fixed by context (could be anything) -> metadata value -->
  <!-- count:    include when the length is variable -> string bytes, shape dims -->
  <!-- "fixed by context" = when you OMIT the type tag. string elems always bytes; shape dims always 
  u64;
       array elems are NOT fixed -> array needs BOTH. this slipped 3x, so spell it out -->
Primitive 1 - Count
Primitive 2 - Type

Strings (count only)

The context dictates that a string is a sequence of bytes. Therefore, the type is fixed by context but length is variable

Tensor shapes (count only)

A shape is always a sequence of dimension sizes. Therefore, the type of each dimension size is fixed by context, but number of dims is better

An array has a variable length and its content could be any valid GGUF data type. Because the element
type is not fixed by context, an array requires both primitives.

## Alignment
  <!-- tensors padded to 32-byte boundaries; why SIMD/AMX wide loads need it (aligned = one fetch;
       unaligned = split/fault); the format pre-arranges bytes for the math hardware -->
SIMD and AMX read wide loads - 16, 32, 64 bytes in a single instruction, sometimes only valid if memory address is a multiple
of the load width. An aligned load is one clean fetch.

## What tripped me up
  <!-- type-vs-count confusion (3x); hex; whatever else was real -->
Type v count confusion

## Open / next session
  <!-- download the 3B GGUF (setup.sh banner has the command); then write the actual parser in C++ -->
Download the 3B GGUF, write parser!