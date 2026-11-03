#!/usr/bin/env python3
"""Runs a boring-llm binary on the per-format synthetic models test/fixtures/fmt-*.gguf and compares
its logits with the float64 numpy reference of tools/make_arch_test_ggufs.py (--formats).

`--tokens` prints the top 5 (token, logit) per position; each printed logit is compared with the
reference's value for that token. Every format has a native GPU kernel and matches to rounding; the
tolerance is relative to the largest reference logit of the model (default 0.005).

    .venv/bin/python tools/formats_vs_reference.py --binary ./main_metal [--tolerance 0.005] [NAME ...]
"""
import argparse
import glob
import os
import re
import subprocess
import sys

import numpy as np

TOKENS = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4]
LINE = re.compile(r"pos (\d+) \(token \d+\): top5 = \[(.*)\]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--tolerance", type=float, default=0.005)
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()
    paths = sorted(glob.glob("test/fixtures/fmt-*.gguf"))
    if args.names:
        paths = [p for p in paths if os.path.basename(p)[4:-5] in args.names]
    failed = 0
    for path in paths:
        stem = os.path.basename(path)[:-5]
        ref = np.fromfile(f"test/fixtures/{stem}.logits.f32", dtype="<f4").reshape(len(TOKENS), 10)
        run = subprocess.run([args.binary, "--gguf", path, "--quantized", "--tokens", ",".join(map(str, TOKENS))],
                             capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=600)
        worst, top1, rows = 0.0, 0, 0
        for line in run.stdout.splitlines():
            m = LINE.match(line)
            if not m:
                continue
            pos = int(m.group(1))
            pairs = re.findall(r"\((\d+), (-?[0-9.e+-]+)\)", m.group(2))
            rows += 1
            for tok, logit in pairs:
                worst = max(worst, abs(float(logit) - float(ref[pos][int(tok)])))
            top1 += int(pairs[0][0]) == int(np.argmax(ref[pos]))
        scale = float(np.abs(ref).max())
        ok = rows == len(TOKENS) and worst <= args.tolerance * scale
        failed += 0 if ok else 1
        print(f"{stem:14s} {rows} positions, top-1 {top1}/{rows}, max |logit - reference| = {worst:.5f} ({100 * worst / scale:.2f}% of {scale:.3f}) {'ok' if ok else 'MISMATCH'}")
        if rows != len(TOKENS):
            print((run.stdout + run.stderr)[-400:])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
