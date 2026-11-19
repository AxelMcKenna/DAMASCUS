"""Time 512 greedy tokens after prefill, one sequence per Metal engine."""
import argparse
import json
import os
import re
import statistics
import subprocess

PROMPT = "The capital of France is"
MLX = """
from time import perf_counter
import sys
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
model, tok = load(sys.argv[1])
ids = tok.encode(sys.argv[2], add_special_tokens=True)
if ids != [128000, 791, 6864, 315, 9822, 374] or mx.default_device() != mx.Device(mx.gpu):
    raise RuntimeError("expected Llama 3 tokens and a Metal device")
n = int(sys.argv[3])
steps = generate_step(mx.array(ids), model, max_tokens=n + 1)
first, _ = next(steps)
if first != 12366:
    raise RuntimeError("expected first greedy token for Paris")
start = perf_counter()
count = sum(1 for _ in steps)
print('decode-rate', count, count / (perf_counter() - start))
"""


def measure(args, engine, n):
    commands = {
        "Damascus": [args.damascus, "--generate-bos", args.gguf, str(n + 1), PROMPT],
        "llama.cpp": [args.llama, "-m", args.gguf, "-n", str(n + 1), PROMPT],
        "MLX-LM": [args.mlx_python, "-c", MLX, args.mlx_model, PROMPT, str(n)],
    }
    proc = subprocess.run(commands[engine], capture_output=True, text=True, check=True,
                          timeout=120, env=dict(os.environ, HF_HUB_OFFLINE="1", DAMASCUS_TRACE_PREFILL="1"))
    if engine == "Damascus":
        ok = "[prefill] quantized Metal, 6 rows" in proc.stderr and "generated ids: [12366," in proc.stdout
        match = re.search(r"decode: (\d+) tok in .*?, ([\d.]+) tok/s", proc.stderr)
    else:
        ok = engine == "MLX-LM" or ("offloaded 29/29 layers to GPU" in proc.stderr and "first 12366" in proc.stderr)
        match = re.search(r"decode-rate (\d+) ([\d.]+)", proc.stderr + proc.stdout)
    if not ok or not match or int(match[1]) != n:
        raise RuntimeError(f"invalid {engine} token count, first token or GPU execution")
    return float(match[2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ["damascus", "gguf", "llama", "mlx-python", "mlx-model"]:
        parser.add_argument("--" + flag, required=True)
    args = parser.parse_args()
    engines = ["Damascus", "llama.cpp", "MLX-LM"]
    samples = []
    for i in range(-1, 3):
        start = i % len(engines)
        for engine in engines[start:] + engines[:start]:
            rate = measure(args, engine, 64 if i < 0 else 512)
            if i >= 0:
                samples.append(dict(round=i + 1, engine=engine, tokens_per_second=rate))
    medians = {e: statistics.median(s["tokens_per_second"] for s in samples if s["engine"] == e)
               for e in engines}
    print(json.dumps(dict(prompt=PROMPT, prompt_tokens=6, concurrent_sequences=1, decode_tokens_per_run=512,
                          warmup_tokens_per_engine=64, early_stopping=False, samples=samples, median=medians), indent=2))


if __name__ == "__main__":
    main()
