"""Run the frozen M5 single-stream benchmark. Write one JSON object per timed run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

PANEL = Path(__file__).resolve().parents[1] / "docs/benchmarks/m5-spec-panel.json"
LOCK = "77d36ef14a9333c1b4016fe0f630e60c207016b1628e20da476e629cd8eb7173"
ENGINES = ["Damascus", "llama.cpp", "MLX-LM"]


def mlx_run(target, draft, prompt, count):
    import time
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import speculative_generate_step

    model, tok = load(target)
    drafter, draft_tok = load(draft)
    ids = tok.encode(prompt, add_special_tokens=True)
    if ids != draft_tok.encode(prompt, add_special_tokens=True) or mx.default_device() != mx.Device(mx.gpu):
        raise ValueError("incompatible draft tokenizer or missing GPU")
    steps = speculative_generate_step(mx.array(ids), model, drafter, max_tokens=count + 1)
    first, _, accepted = next(steps)
    start = time.perf_counter()
    output = [first]
    hits = int(accepted)
    for token, _, accepted in steps:
        output.append(token)
        hits += accepted
    seconds = time.perf_counter() - start
    print(json.dumps([ids, output, seconds, hits]))


def measure(args, engine, prompt, count):
    env = dict(os.environ, HF_HUB_OFFLINE="1", DAMASCUS_TRACE_DECODE="1")
    for key in ("DAMASCUS_NO_DRAFT", "DAMASCUS_CHAIN"):
        env.pop(key, None)
    commands = {
        "Damascus": [args.damascus, "--generate-bos", args.gguf, str(count + 1), prompt],
        "llama.cpp": [args.llama, "--spec-type", "ngram-simple", "--temp", "0", "-ngl", "99",
                      "-c", "2048", "-m", args.gguf, "-n", str(count + 1), "-p", prompt],
        "MLX-LM": [args.mlx_python, __file__, "_mlx", args.target, args.draft, prompt, str(count)],
    }
    if engine == "llama.cpp 1B":
        commands[engine] = commands["llama.cpp"].copy()
        commands[engine][2:3] = ["draft-simple", "-md", args.llama_draft, "--spec-draft-n-max", "2"]
    p = subprocess.run(commands[engine], env=env, capture_output=True, timeout=180)
    out, err = (x.decode(errors="replace") for x in (p.stdout, p.stderr))
    if p.returncode:
        raise RuntimeError(f"{engine} exited {p.returncode}: {err[-4000:]}")
    stats = {}
    if engine == "Damascus":
        ids, generated = map(json.loads, re.search(r"prompt ids: (\[.*?\])\ngenerated ids: (\[.*?\])", out).groups())
        n, ms = re.search(r"decode: (\d+) tok in ([\d.]+) ms", err).groups()
        seconds = float(ms) / 1000
        batches, hits = map(int, re.search(r"(\d+) verification batches, (\d+) matched draft tokens", err).groups())
        stats["verification_batches"] = batches
    elif engine.startswith("llama.cpp"):
        n, elapsed, hits, proposed, inp, gen = re.search(
            r"panel (\d+) ([\d.]+) (\d+) (\d+)\n([\d ]+)\n([\d ]+)\n", out).groups()
        ids, generated = [list(map(int, line.split())) for line in (inp, gen)]
        stats["proposed_tokens"] = int(proposed)
        seconds = float(elapsed)
    else:
        ids, generated, seconds, hits = json.loads(out)
        n = len(generated) - 1
    if len(generated) != count + 1 or int(n) != count or not ids or ids[0] != 128000 or seconds <= 0:
        raise ValueError(f"{engine}: invalid token count, BOS or timer")
    digest = hashlib.sha256(json.dumps(generated).encode()).hexdigest()
    return dict(seconds=seconds, tok_s=count/seconds, input_ids=ids, first=generated[0],
                output_sha256=digest, accepted=int(hits), **stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("damascus", "gguf", "llama", "mlx-python", "target", "draft"):
        parser.add_argument("--" + arg, required=True)
    parser.add_argument("--llama-draft", help="Run only the supplemental two-token 1B llama.cpp configuration")
    args = parser.parse_args()
    if hashlib.sha256(PANEL.read_bytes()).hexdigest() != LOCK:
        raise ValueError("benchmark lock changed")
    prompts = list(json.loads(PANEL.read_text())["prompts"].items())
    engines = ["llama.cpp 1B"] if args.llama_draft else ENGINES
    for engine in engines:
        measure(args, engine, "The capital of France is", 64)
    references = {}
    for r in range(3):
        for i, (key, prompt) in enumerate(prompts[r:] + prompts[:r]):
            offset = (i + 2*r) % len(engines)
            for engine in engines[offset:] + engines[:offset]:
                row = measure(args, engine, prompt, 512)
                if row["input_ids"] != references.setdefault(key, row["input_ids"]):
                    raise ValueError(f"input IDs differ on {key}")
                print(json.dumps(dict(round=r+1, prompt=key, engine=engine, **row)), flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["_mlx"]:
        mlx_run(*sys.argv[2:5], int(sys.argv[5]))
    else:
        main()
