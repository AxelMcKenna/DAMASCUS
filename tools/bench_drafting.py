"""Measure history drafting on the same request and check all output IDs."""
import argparse
import json
import os
import re
import subprocess

from bench_decode import PROMPT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--damascus", required=True)
    parser.add_argument("--gguf", required=True)
    args = parser.parse_args()
    samples = []
    reference = None
    for enabled in (True, False, False, True, True, False):
        env = dict(os.environ, DAMASCUS_TRACE_DECODE="1", DAMASCUS_CHAIN="1")
        env.pop("DAMASCUS_NO_DRAFT", None)
        if not enabled:
            env["DAMASCUS_NO_DRAFT"] = "1"
        run = subprocess.run(
            [args.damascus, "--generate-bos", args.gguf, "513", PROMPT],
            env=env, capture_output=True, text=True, check=True, timeout=120,
        )
        ids = json.loads(re.search(r"generated ids: (\[.*?\])", run.stdout)[1])
        if len(ids) != 513 or ids[0] != 12366 or (reference is not None and ids != reference):
            raise RuntimeError("generated sequence failed the parity check")
        reference = ids
        batch, matched = map(int, re.search(
            r"(\d+) verification batches, (\d+) matched draft tokens", run.stderr
        ).groups())
        if enabled != (batch > 0) or enabled != (matched > 0):
            raise RuntimeError("unexpected draft verification counts")
        rate = float(re.search(r"decode: 512 tok in .*?, ([\d.]+) tok/s", run.stderr)[1])
        samples.append(dict(drafting=enabled, tokens_per_second=rate,
                            verification_batches=batch, accepted_draft_tokens=matched))
    print(json.dumps(dict(prompt=PROMPT, generated_ids=reference, samples=samples), indent=2))


if __name__ == "__main__":
    main()
