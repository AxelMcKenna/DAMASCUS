#!/usr/bin/env python3
"""Compare engine activations against the HF oracle — the Milestone-A gate.

Usage:
    python compare.py REF_DIR ENGINE_DIR [--threshold 1e-3]

Both dirs hold .npy arrays written with matching names (tokens/hidden/logits).
The engine side is produced once the Damascus binary can run and dump its own
activations in the same format.

Two regimes (see dump_reference.py header):
  * SAME-WEIGHTS oracle  -> absolute max-abs diff < threshold is meaningful.
  * HF full-precision oracle vs Q4_K_M engine -> absolute diff is dominated by
    quantization noise; trust COSINE and TOP-K agreement instead.
The table reports all metrics so you can read the right one for your setup.

Exit code is nonzero if the gate fails, so this can sit in CI / a Make target.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def metrics(ref: np.ndarray, test: np.ndarray) -> dict:
    ref = ref.astype(np.float64).ravel()
    test = test.astype(np.float64).ravel()
    diff = np.abs(ref - test)
    denom = np.maximum(np.abs(ref), 1e-9)
    cos = float(np.dot(ref, test) / (np.linalg.norm(ref) * np.linalg.norm(test) + 1e-30))
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel": float((diff / denom).max()),
        "cosine": cos,
    }


def topk_agreement(ref: np.ndarray, test: np.ndarray, k: int = 5) -> tuple[bool, float]:
    """Last-position next-token agreement: top-1 match and top-k overlap."""
    r, t = ref[-1], test[-1]
    rk = set(np.argsort(r)[-k:].tolist())
    tk = set(np.argsort(t)[-k:].tolist())
    top1 = int(r.argmax()) == int(t.argmax())
    overlap = len(rk & tk) / k
    return top1, overlap


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref_dir")
    ap.add_argument("engine_dir")
    ap.add_argument("--threshold", type=float, default=1e-3,
                    help="max-abs gate for same-weights comparisons (default 1e-3)")
    args = ap.parse_args()

    ref_dir, eng_dir = Path(args.ref_dir), Path(args.engine_dir)
    names = sorted(p.name for p in ref_dir.glob("*.npy"))
    if not names:
        print(f"no .npy arrays in {ref_dir}", file=sys.stderr)
        return 2

    print(f"{'array':<14}{'shape':<18}{'max_abs':>12}{'max_rel':>12}{'cosine':>12}")
    print("-" * 68)

    ok = True
    logits_top1 = None
    for name in names:
        if name == "tokens.npy":
            continue
        ref = np.load(ref_dir / name)
        eng_path = eng_dir / name
        if not eng_path.exists():
            print(f"{name:<14}{'(missing in engine dir)':<18}")
            ok = False
            continue
        eng = np.load(eng_path)
        if ref.shape != eng.shape:
            print(f"{name:<14}SHAPE MISMATCH ref{ref.shape} eng{eng.shape}")
            ok = False
            continue
        m = metrics(ref, eng)
        print(f"{name:<14}{str(ref.shape):<18}{m['max_abs']:>12.2e}{m['max_rel']:>12.2e}{m['cosine']:>12.6f}")
        if name == "logits.npy":
            top1, overlap = topk_agreement(ref, eng)
            logits_top1 = top1
            print(f"{'  logits':<14}top1_match={top1}  top5_overlap={overlap:.0%}  "
                  f"(cosine is the quant-robust metric)")
        if m["max_abs"] > args.threshold:
            ok = False

    # tokens: if both present, mismatch means the tokenizer (Compartment 4) is wrong
    if (ref_dir / "tokens.npy").exists() and (eng_dir / "tokens.npy").exists():
        rt = np.load(ref_dir / "tokens.npy")
        et = np.load(eng_dir / "tokens.npy")
        same = rt.shape == et.shape and bool((rt == et).all())
        print(f"\ntokens identical: {same}" + ("" if same else "  <- tokenizer mismatch"))
        if not same:
            ok = False

    print()
    if ok:
        print(f"PASS  (all arrays within {args.threshold:g} abs)")
    elif logits_top1:
        print("ABS GATE NOT MET, but top-1 token agrees — expected when comparing "
              "Q4_K_M engine vs fp32 HF oracle.\nJudge by cosine/top-k above, or "
              "switch to a same-weights oracle for a true 1e-3 check.")
    else:
        print("FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
