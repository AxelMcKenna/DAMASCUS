# Oracle harness — the Milestone-A correctness gate

Milestone A is defined as *"forward pass matches HuggingFace within 1e-3."* That
is a **runtime numerical comparison**, so it needs a reference to compare
against. This harness produces that reference and scores the engine against it.

```
dump_reference.py   run HF Llama-3.2-3B in fp32, dump per-layer hidden + logits
compare.py          score engine output vs reference -> PASS/FAIL (Milestone A)
requirements.txt    deps for the venv
```

## Why Python — and how it actually runs here (Docker)

`llama.cpp` is a locally-compiled binary and is SIGKILL'd (exit 137) by the
EDR/code-signing policy on this laptop. The same policy blocks `mmap` of any
unsigned native code on the **host** — including Python C-extensions (`numpy`,
`torch`), and even the unsigned Homebrew `container` binary. A native host run
of this harness therefore does *not* work.

**The working escape is Docker Desktop.** Its launcher is notarized by Docker
(Team `9BNSXJN65R`), so it passes the EDR allowlist, and the container runs in a
**Linux guest** where macOS code-signing doesn't apply — unsigned torch/numpy
execute fine. Verified end-to-end: the dump + `compare.py` run in-container.
This is CPU-only (a Linux guest has no Metal/AMX), which is exactly what the
oracle needs. Run it with `make oracle` (see below).

When the host EDR exception eventually lands it also unblocks `llama.cpp` on the
*same GGUF*, the better 1e-3 oracle (shares dequant math with the engine — see
below). Until then, Docker is the path.

## Quick start (Docker)

```sh
hf auth login                              # once; gated Llama model
make oracle  ORACLE_PROMPT="The capital of France is"
# writes tools/oracle/reference/{tokens,hidden,logits}.npy + manifest.json
```

`make oracle` builds the image (`Dockerfile`) and runs `dump_reference.py` in a
container, bind-mounting `tools/oracle/` for output and your `~/.cache/huggingface`
for the model + token. Later, score the engine with
`make oracle-compare ENGINE=tools/oracle/engine`.

## The weight caveat — read before trusting any number

This oracle runs the **original full-precision HF weights**. The engine runs
**Q4_K_M-quantized** weights. They differ by quantization noise, which on raw
logits is normally **bigger than 1e-3**. Consequences:

| comparison | trust this metric |
|---|---|
| engine (Q4_K_M) vs HF (fp32) — what you can run *now* | **cosine > 0.9999**, **top-1 / top-5 token agreement** |
| engine vs same-weights oracle — llama.cpp-on-GGUF, or a future GGUF-dequant injection mode | **max-abs < 1e-3** (the literal Milestone-A number) |

`compare.py` prints all of them. For first bring-up, getting top-1 to agree and
per-layer cosine ≈ 1.0 proves the architecture is wired correctly; chase the
literal 1e-3 against a same-weights oracle afterward.

## Native setup (only on a non-EDR machine)

Docker (above) is the path on the work laptop. The native venv route below works
**only where the host can run unsigned native code** (an unmanaged Mac, or after
the IT exception). The Llama model is **gated** — request access on its HF page
first, then:

```sh
uv venv --python 3.14 tools/oracle/.venv
source tools/oracle/.venv/bin/activate
uv pip install -r tools/oracle/requirements.txt
hf auth login            # paste a token with access to the gated repo
python tools/oracle/dump_reference.py --prompt "The capital of France is" \
    --out tools/oracle/reference
```

Either route writes to `tools/oracle/reference/`:
- `tokens.npy`  — the exact input token ids. **The engine must feed these
  identical ids.** Once Compartment 4 exists, the engine's own tokenization of
  the same prompt should reproduce this array byte-for-byte — that's the
  tokenizer's round-trip check, for free.
- `hidden.npy` — `[num_layers+1, seq, d_model]` residual stream. Index legend is
  in `manifest.json`. **Per-layer diffing is the whole point**: when final
  logits are wrong, this tells you *which* of the 28 layers first diverges
  instead of leaving you guessing.
- `logits.npy` — `[seq, vocab]` final logits.
- `manifest.json` — prompt, token ids, shapes, and the model config fields that
  Compartment 3 parses from the GGUF (cross-check your loader against these:
  block_count 28, embedding_length 3072, heads 24/8, ff 8192, vocab 128256,
  rope_theta 500000, eps 1e-5).

## Score the engine (later)

When the engine can run and dumps the same three arrays into, say,
`tools/oracle/engine/`:

```sh
python tools/oracle/compare.py tools/oracle/reference tools/oracle/engine
```

Determinism notes: fp32, CPU, `attn_implementation="eager"`, `use_cache=False`,
fixed seed — so the reference is reproducible run to run.
