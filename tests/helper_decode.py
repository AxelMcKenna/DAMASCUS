"""Model-backed CLI checks. Pass the Damascus binary, target GGUF and helper GGUF."""
import json
import os
import re
import subprocess
import sys


def generate(command):
    run = subprocess.run(command, capture_output=True, check=True,
                         env=dict(os.environ, DAMASCUS_NO_DRAFT="1", DAMASCUS_TRACE_DECODE="1"))
    ids = json.loads(re.search(rb"generated ids: (\[.*?\])", run.stdout)[1])
    return ids, run.stderr


def main(binary, target, helper):
    prompt = "The capital of France is"
    reference, _ = generate([binary, "--generate-bos", target, "65", prompt])
    for size in (1, 2, 3, 4, 5, 65):
        ids, trace = generate([binary, "--generate-helper", target, helper, str(size), prompt])
        if ids != reference[:size]:
            raise ValueError(f"target token mismatch at length {size}")
        if size == 65:
            batches, matches = map(int, re.search(rb"(\d+) verification batches, (\d+) matched", trace).groups())
            if not 0 < matches < 2 * batches:
                raise ValueError("fixture must exercise draft acceptance and rejection")
        print(f"PASS {size} tokens")


if __name__ == "__main__":
    main(*sys.argv[1:])
