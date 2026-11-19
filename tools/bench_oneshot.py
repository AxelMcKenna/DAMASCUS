"""Measure warm-file-cache, fresh-process, single-token greedy completion."""
import argparse
import itertools
import json
import os
import random
import re
import statistics
import subprocess
import time

# Each case supplies a raw prompt, its BOS-inclusive Llama 3 IDs, and one
# expected visible token. Validate every invocation, including warmups.
CASES = [
    ("The capital of France is", [128000, 791, 6864, 315, 9822, 374], 12366, " Paris"),
    ("The capital of Japan is", [128000, 791, 6864, 315, 6457, 374], 27286, " Tokyo"),
    ("The opposite of hot is", [128000, 791, 14329, 315, 4106, 374], 9439, " cold"),
    ("The largest planet in our solar system is",
     [128000, 791, 7928, 11841, 304, 1057, 13238, 1887, 374], 50789, " Jupiter"),
    ("The chemical symbol for oxygen is",
     [128000, 791, 11742, 7891, 369, 24463, 374], 507, " O"),
    ("In a standard deck of cards, there are four",
     [128000, 644, 264, 5410, 9722, 315, 7563, 11, 1070, 527, 3116], 29076, " suits"),
]

# Use MLX-LM directly so CLI chat formatting and sampling cannot vary.
# Lazy loading lets MLX schedule materialization with the forward pass.
MLX = """
import json, sys
import mlx.core as mx
from mlx_lm import load
model, tokenizer = load(sys.argv[1], lazy=True)
if mx.default_device() != mx.Device(mx.gpu):
    raise RuntimeError("MLX Metal GPU required")
ids = tokenizer.encode(sys.argv[2], add_special_tokens=True)
token = mx.argmax(model(mx.array([ids]))[0, -1]).item()
print(json.dumps([ids, token, tokenizer.decode([token])]))
"""


def measure(args, engine, case):
    prompt, ids, token, piece = case
    commands = {
        "damascus": [args.damascus, "--generate-bos", args.gguf, "1", prompt],
        "llama.cpp": [args.llama_simple, "-m", args.gguf, "-n", "1", prompt],
        "mlx-lm": [args.mlx_python, "-c", MLX, args.mlx_model, prompt],
    }
    env = dict(os.environ, DAMASCUS_TRACE_PREFILL="1") if engine == "damascus" else None
    start = time.perf_counter_ns()
    p = subprocess.run(commands[engine], capture_output=True, text=True, timeout=120, env=env)
    ms = (time.perf_counter_ns() - start) / 1e6
    if p.returncode:
        raise RuntimeError(f"{engine} failed ({p.returncode})\n{p.stderr}")
    if engine == "damascus":
        expected = f'prompt ids: {ids}\ngenerated ids: [{token}]\ntext: "{piece}"\n'
        valid = p.stdout == expected and f"[prefill] quantized Metal, {len(ids)} rows" in p.stderr
    elif engine == "llama.cpp":
        valid = (p.stdout == "<|begin_of_text|>" + prompt + piece + "\n"
                 and re.search(r"offloaded (\d+)/\1 layers to GPU", p.stderr))
    else:
        valid = json.loads(p.stdout) == [ids, token, piece]
    if not valid:
        raise RuntimeError(f"{engine} failed parity or offload check for {prompt!r}: {p.stdout!r}")
    return ms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("damascus", "gguf", "llama-simple", "mlx-python", "mlx-model"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--rounds", type=int, default=12)
    args = parser.parse_args()
    if args.rounds <= 0 or args.rounds % 6:
        parser.error("rounds must be a positive multiple of six to balance engine order")
    engines = ("damascus", "llama.cpp", "mlx-lm")
    for engine in engines:
        measure(args, engine, CASES[0])
    rng = random.Random(42)
    samples = []
    for block in range(args.rounds // 6):
        orders = list(itertools.permutations(engines))
        rng.shuffle(orders)
        for i, order in enumerate(orders):
            cases = list(CASES)
            rng.shuffle(cases)
            for case in cases:
                times = {engine: measure(args, engine, case) for engine in order}
                samples.append(dict(round=6 * block + i + 1, prompt=case[0], order=order, ms=times))
    medians = {e: statistics.median(s["ms"][e] for s in samples) for e in engines}
    print(json.dumps(dict(median_ms=medians, samples=samples), indent=2))


if __name__ == "__main__":
    main()
