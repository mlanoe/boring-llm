#!/usr/bin/env python3
"""Checks the numpy references of tools/make_arch_test_ggufs.py against real llama.cpp.

The synthetic models test/fixtures/arch-*.gguf (and fmt-*.gguf with --formats) carry a minimal
vocabulary, so llama.cpp loads them; its logits for the fixture prompt must match the float64
numpy transcription of its graph that test/test_arch.br and the Metal checks compare boring-llm
against. This is what makes those references trustworthy (YaRN, Phi-3 factors, Gemma 2/3
soft-capping and sliding windows, ...). Tolerance: 0.01 for F32 files; Q8_0 files differ by
llama.cpp's own activation quantization (up to ~0.1) and are only reported.

    .venv/bin/python tools/arch_vs_llamacpp.py [--formats]
"""
import glob
import os
import sys

import numpy as np
from llama_cpp import Llama

TOKENS = [3, 1, 4, 1, 5, 9, 2, 6]
FMT_TOKENS = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4]


def context_for(stem):
    """Phi-3 picks its RoPE factor set from n_ctx (original context 1024): short below it, long above."""
    if stem.endswith("phi3-long"):
        return 2048
    if stem.endswith("phi3-short"):
        return 64
    return 512


def run(gguf, expected, n_ctx, tokens):
    llm = Llama(model_path=gguf, n_gpu_layers=0, n_ctx=n_ctx, logits_all=True, verbose=False)
    llm.eval(tokens)
    got = np.array(llm.scores[: len(tokens)])[:, :10]
    ref = np.fromfile(expected, dtype="<f4").reshape(len(tokens), 10)
    return float(np.abs(got - ref).max())


def main():
    formats = "--formats" in sys.argv
    pattern = "test/fixtures/fmt-*.gguf" if formats else "test/fixtures/arch-*.gguf"
    failed = 0
    for path in sorted(glob.glob(pattern)):
        stem = os.path.basename(path)[:-5]
        err = run(path, f"test/fixtures/{stem}.logits.f32", context_for(stem), FMT_TOKENS if formats else TOKENS)
        informational = "-q8-" in stem or formats  # llama.cpp's own activation quantization / mixed formats
        ok = informational or err < 0.01
        print(f"{stem:22s} max |llama.cpp - numpy reference| = {err:.4f} {'(quantized: informational)' if informational else ('ok' if ok else 'MISMATCH')}")
        failed += 0 if ok else 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
