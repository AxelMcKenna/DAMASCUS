#!/usr/bin/env python3
"""Milestone-A oracle: dump HuggingFace reference activations for Llama 3.2 3B.

This is the *reference* side of the correctness check. It runs the model in
full fp32 on CPU, deterministically, and writes the per-layer residual stream
plus final logits to disk. The Damascus engine will later dump the same arrays
(same token ids) and `compare.py` declares Milestone A when they agree.

Why this and not llama.cpp: llama.cpp is a locally-built binary and is killed
by the same EDR/code-signing policy that SIGKILLs the engine (exit 137). The
system Python interpreter is signed and runs, so the HF route is the only
oracle that executes on this machine today.

A caveat you must keep in mind (see README): this oracle runs the *original*
HF weights, while the engine runs *Q4_K_M*-quantized weights. The two differ by
quantization noise, which on raw logits is usually larger than 1e-3. So:
  - absolute 1e-3 is only meaningful against a SAME-WEIGHTS oracle
    (llama.cpp on the same GGUF once EDR is unblocked, or a GGUF-dequant
    weight-injection mode added later);
  - against this HF oracle, judge correctness by per-layer cosine (>0.9999)
    and top-1 / top-5 token agreement, which are robust to quantization.
`compare.py` reports all of these so the harness is useful in both regimes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct",
                   help="HF repo id or local path (gated; only used when --gguf is unset)")
    p.add_argument("--gguf", default=None,
                   help="path to a local .gguf file. Loads weights+tokenizer from it "
                        "directly (NO gated HF access), and — because it dequantizes the "
                        "SAME Q4_K_M blocks the engine uses — makes the 1e-3 absolute "
                        "comparison meaningful (a same-weights oracle).")
    p.add_argument("--prompt", default="The capital of France is",
                   help="prompt text")
    p.add_argument("--chat", action="store_true",
                   help="wrap the prompt with the instruct chat template "
                        "(default: raw prompt + BOS, simpler for first bring-up)")
    p.add_argument("--out", default="tools/oracle/reference",
                   help="output directory for .npy arrays + manifest.json")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"],
                   help="compute dtype. fp32 is the truest reference (~12GB for 3B); "
                        "bf16/fp16 (~6GB) if the container OOMs")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Heavy imports deferred so `--help` works without torch installed.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dt = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    # transformers v5 renamed `torch_dtype` -> `dtype`; support both.
    # low_cpu_mem_usage avoids the 2x peak (load shard-by-shard).
    load_kw = dict(attn_implementation="eager", low_cpu_mem_usage=True)
    if args.gguf:
        # ungated same-weights oracle: dir + filename, weights/tokenizer from the GGUF
        gguf = Path(args.gguf)
        src, load_kw["gguf_file"] = str(gguf.parent), gguf.name
        print(f"==> loading from GGUF {gguf} (fp32, cpu, eager attn) — ungated, same-weights", file=sys.stderr)
    else:
        src = args.model
        print(f"==> loading tokenizer + model: {src} (fp32, cpu, eager attn)", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(src, **{k: v for k, v in load_kw.items() if k == "gguf_file"})
    try:
        model = AutoModelForCausalLM.from_pretrained(src, dtype=dt, **load_kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=dt, **load_kw)
    model.eval()

    # --- tokenize -------------------------------------------------------
    if args.chat:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True, return_tensors="pt",
        )
    else:
        ids = tok(args.prompt, return_tensors="pt").input_ids  # adds BOS for llama
    ids = ids.to(torch.long)
    token_ids = ids[0].tolist()
    print(f"==> {len(token_ids)} tokens: {token_ids}", file=sys.stderr)

    # --- forward --------------------------------------------------------
    with torch.no_grad():
        out_obj = model(ids, output_hidden_states=True, use_cache=False)

    # hidden_states: tuple of (num_layers + 1) tensors, each [1, seq, d_model].
    #   index 0  -> token embeddings (input to layer 0)
    #   index k  -> residual stream AFTER decoder layer k-1
    #   index -1 -> residual stream after the final RMSNorm (input to lm_head)
    hs = [h[0].to(torch.float32).cpu().numpy() for h in out_obj.hidden_states]
    hidden = np.stack(hs, axis=0)  # [num_hs, seq, d_model]
    logits = out_obj.logits[0].to(torch.float32).cpu().numpy()  # [seq, vocab]

    np.save(out / "tokens.npy", np.asarray(token_ids, dtype=np.int32))
    np.save(out / "hidden.npy", hidden)
    np.save(out / "logits.npy", logits)

    cfg = model.config
    rope_theta = getattr(cfg, "rope_theta", None)
    if rope_theta is None:  # GGUF-loaded configs sometimes nest it
        rope_theta = (getattr(cfg, "rope_scaling", None) or {}).get("rope_theta")
    manifest = {
        "model": args.gguf or args.model,
        "dtype": args.dtype,
        "prompt": args.prompt,
        "chat_template": bool(args.chat),
        "seed": args.seed,
        "token_ids": token_ids,
        "shapes": {"hidden": list(hidden.shape), "logits": list(logits.shape)},
        "hidden_index_legend": {
            "0": "token embeddings (layer-0 input)",
            "k": "residual stream after decoder layer k-1",
            "-1": "after final RMSNorm (lm_head input)",
        },
        # the same fields Compartment 3 parses out of the GGUF; cross-check these
        "config": {
            "vocab_size": cfg.vocab_size,
            "block_count": cfg.num_hidden_layers,
            "embedding_length": cfg.hidden_size,
            "attention_head_count": cfg.num_attention_heads,
            "attention_head_count_kv": cfg.num_key_value_heads,
            "feed_forward_length": cfg.intermediate_size,
            "rms_norm_epsilon": cfg.rms_norm_eps,
            "rope_freq_base": rope_theta,
            "context_length": cfg.max_position_embeddings,
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"==> wrote {out}/{{tokens,hidden,logits}}.npy + manifest.json", file=sys.stderr)
    print(f"    hidden {hidden.shape}  logits {logits.shape}", file=sys.stderr)
    argmax = int(logits[-1].argmax())
    print(f"    next-token argmax id={argmax} ({tok.decode([argmax])!r})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
