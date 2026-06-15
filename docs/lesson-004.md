# Lesson 004 — Compartment 3: writing the GGUF parser (part 1)

**Date:** 2026-06-15

## Getting the file into memory: slurp vs stream
  <!-- chose A (slurp into std::vector<std::byte> + integer cursor) over B (stream/seek);
       WHY: in-memory buffer = free random access, IS the "flat array of bytes" model;
       caveat: wouldn't slurp the 2GB model — that's what mmap is for, later -->
The first design choice was getting the GGUF file into the parser.

There were two obvious options:

1) Stream the file with std::ifstream, reading and seeking as needed 
2) Slurping the file with std::vector<std::byte> and parse it with an integer cursor

I chose to slurp the file and read the whole GGUF file as a stream of bytes. This makes it easier to reason about
memory and treat it as one flat array of bytes.

## Opening the file
  <!-- std::ifstream + std::ios::binary (why binary matters); std::ios::ate to get size from tellg;
       seekg back to 0 before reading; check if(!file) for open AND read failures -->

The parser opens the file with std::ifstream in binary mode.

std::ifstream file(path, std::ios::binary | std::ios::ate);

The std::ios::binary flag matters because this isn't a text file and therefore I didn't want C++ doing any translations,
I wanted the raw bytes.

The std::ios::ate flag opens the file with the cursor positioned at the end. That makes it easy to get file size with tellg()

auto size = file.tellg()

After getting the size i needed to set the cursor back before reading:

file.seekg(0)

Then the buffer can be resized and filled:

std::vector<std::byte> buffer(size);
file.read(reinterpret_cast<char*>(buffer.data()), size);

## read_primitive<T> — the safe door for fixed-size reads
  <!-- template over T; bounds check (throw, not assert — corrupt file is a runtime/IO error, not a
       programmer bug); memcpy &buffer[cursor] -> value; advance cursor by sizeof(T) -->
  <!-- WHY cursor is size_t& (by reference): by value -> local copy -> caller never advances -> reads
       the same bytes forever -->
  <!-- latent assumption: memcpy reads host byte order; fine because host + GGUF both little-endian -->

The parser needed a reusable way to read fixed-size values from the byte buffer - integers, floats, booleans, type tags.

This is what read_primitive<T> is for:

It does 4 things:

1) Check there are enough bytes left for the buffer
2) Copy sizeof(T) bytes from the buffer into the value
3) Advance cursor by sizeof(T)
4) Return the value

The bounds check should throw an error instead of an assert, as a corrupted GGUF file is not a programmer bug and should still compile the program

The cursor is passed by reference:

size_& cursor

That matters because if the cursor were parsed be value it would only advance a local copy. Therefore the parser would keep
reading the same bytes.

## read_string — length-prefixed strings
  <!-- read u64 length via read_primitive, bounds-check, build std::string(ptr, length), advance by length
  -->
  <!-- the std::byte -> char* cast (reinterpret_cast): read()/string ctor speak char*, buffer is std::byte;
       cast reinterprets the type, not the bytes — safe -->

GGUF strings lengths are pre-fixed, meaning they store as something like [uint64_t length][raw string bytes]

So reading as string is a two step process:

1) Read the length of the type
2) Read the underlying string

The cast - reinterpret_cast<const char*>(&buffer[cursor]) only tells c++ how to view the same memory as char data because 
std::string and ifstream::read work with char but not raw std::bytes which the parser stores.

## Walking the header and first pair (cursor trace)
  <!-- 0 -> 24 (magic, version, tensor_count, kv_count)
       24 -> 52 (key: 8-byte len + 20-byte "general.architecture")
       52 -> 56 (u32 value-type tag = 8 = STRING)
       56 -> 69 (value: 8-byte len + "llama")
       the cursor position IS the parser's entire state -->
0 -> 24 header
24 -> 52 key.string "general.architecture"
52 -> 56 value-type tag - e.g 8 = string
56 -> 69 value string - e.g "llama"

The key itself consumed 28 bytes - 8 bytes length, 20 bytes string contents. Then the type tag was 8 meaning STRING, then
the parser called read_string again and got: "llama"


## The type-tag asymmetry (answer the recall question here)
  <!-- why the KEY needed no type tag but the VALUE did, in terms of "fixed by context":
       a key is always a string (type fixed) -> read_string directly;
       a value could be any type (not fixed) -> must read the tag first to know how to read it -->
Key's type is fixed by context. In GGUF metadata, the key is always a string, so we can call read_string directly.

The value is different, A value could be a string, integer, float, boolean, array, etc. The parser can't know how many bytes
to read until it first reads the type-tag

## What tripped me up
  <!-- compiler-caught typos (ifsteam/ccer/const:); memcpy needing &buffer[cursor] not buffer[cursor];
       forgetting to actually slurp (declared nothing, read nothing); whatever else was real -->
typos, slurping

## Open / next session
  <!-- generalize to the 20-iteration metadata loop: switch on type tag, all scalar types,
       the recursive ARRAY case (type-9: element-type + count + elements; vocab = 128256 strings);
       then tensor-info section; then point it at the real 3B GGUF (download via setup.sh) -->x
generalise parser, switch on type tag, all scalar types, recursive array case (own dtype and length) and then
download a real 3B GGUF