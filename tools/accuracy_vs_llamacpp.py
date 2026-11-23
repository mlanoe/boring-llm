#!/usr/bin/env python3
"""Model-level accuracy check of a boring-llm binary against llama.cpp.

Feeds the same token ids to both and compares the logits at every position:
how often boring-llm's top token is llama.cpp's (top-1 agreement) and how far
llama.cpp's top logit is from boring-llm's value for that token (mean absolute
difference). llama.cpp runs on the CPU, so it is an independent oracle of the
GPU kernels.

Why it exists: kernel unit tests passed while models regressed three times (a
half-precision tile staging, a race in the biased Q8_0 prefill, misaligned loads
of Q6_K blocks). A unit test compares a kernel with another implementation of
the same idea; this compares the whole forward pass with someone else's. Run it
for every compiler (`boring`) update that touches the GPU kernels, on at least
one Q8_0 and one Q4_K_M model (Q4_K_M files also contain Q6_K tensors).

Healthy values on the default text (tools/accuracy_text.txt, a frozen copy of the
project README: do not edit it, the values below depend on it), Apple M3: Qwen2.5-0.5B
Q8_0 594/613 and mean 0.06; 0.5B Q4_K_M 601/613 and 0.06; 7B Q4_K_M 298/313 and
0.16 (top-1 misses on natural text are near-ties between two tokens; the mean
difference is the sharp signal). A broken Q6_K prefill gave 28/613 and 6.1 on
the 0.5B Q4_K_M, 0/313 and 22.9 on the 7B; earlier regressions gave 2 to 3.

Usage:
    tools/accuracy_vs_llamacpp.py --gguf weights/qwen2.5-0.5b-instruct-q8_0.gguf \\
        --binary target/release/main                     # 613 positions of the default text
    tools/accuracy_vs_llamacpp.py --gguf weights/qwen2.5-7b-instruct-q4_k_m.gguf \\
        --binary ./main_metal --positions 313
    tools/accuracy_vs_llamacpp.py --model qwen2.5-0.5b-instruct-q8_0 --fetch --binary ./main_metal
    tools/accuracy_vs_llamacpp.py --selftest

Needs the project's .venv (llama-cpp-python, numpy). The reference logits are
cached next to the binary's working directory in `.accuracy-cache/` (delete it
to recompute). Exit status 0 when both thresholds hold, 1 otherwise.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

DEFAULT_TEXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accuracy_text.txt")
POS_RE = re.compile(r"^pos (\d+) \(token (\d+)\): top5 = \[(.*)\]")
PAIR_RE = re.compile(r"\((\d+), ([-\d.e+]+)\)")


def parse_positions(output):
    """boring-llm's `--tokens` output: {position: [(token, logit), ...]} (top 5)."""
    rows = {}
    for line in output.splitlines():
        m = POS_RE.match(line)
        if m:
            rows[int(m.group(1))] = [(int(a), float(v)) for a, v in PAIR_RE.findall(m.group(3))]
    return rows


def compare(reference, ours):
    """reference/ours: {position: [(token, logit), ...]}, best first.
    Returns (positions compared, top-1 agreements, mean |logit diff| of the
    reference's top token, positions where that token is not in our top 5)."""
    n = agree = missing = 0
    diffs = []
    for pos, ref in reference.items():
        if pos not in ours or not ours[pos]:
            continue
        n += 1
        agree += ours[pos][0][0] == ref[0][0]
        mine = dict(ours[pos])
        if ref[0][0] in mine:
            diffs.append(abs(mine[ref[0][0]] - ref[0][1]))
        else:
            missing += 1
    return n, agree, (sum(diffs) / len(diffs) if diffs else float("inf")), missing


def selftest():
    ref = {0: [(5, 10.0), (6, 9.0)], 1: [(7, 8.0), (1, 7.0)], 2: [(3, 4.0), (2, 3.0)]}
    ours = {0: [(5, 10.5), (6, 9.0)], 1: [(1, 8.5), (7, 8.0)], 2: [(9, 4.0), (8, 3.0)]}
    n, agree, mean, missing = compare(ref, ours)
    assert (n, agree, missing) == (3, 1, 1), (n, agree, missing)
    assert abs(mean - 0.25) < 1e-9, mean  # |10.5-10| and |8.0-8| over the two positions where the token is present
    text = "pos 0 (token 8420): top5 = [(594, 15.630145), (374, 14.847761)]\nnoise\npos 1 (token 374): top5 = [(264, -2.5e-1)]"
    parsed = parse_positions(text)
    assert parsed == {0: [(594, 15.630145), (374, 14.847761)], 1: [(264, -0.25)]}, parsed
    print("selftest ok")


def reference_logits(gguf, token_ids, cache_dir):
    key = hashlib.sha256(
        (os.path.basename(gguf) + str(os.path.getsize(gguf)) + ",".join(map(str, token_ids))).encode()
    ).hexdigest()[:16]
    path = os.path.join(cache_dir, key + ".json")
    if os.path.exists(path):
        with open(path) as f:
            return {int(p): [tuple(t) for t in v] for p, v in json.load(f).items()}
    import numpy as np
    from llama_cpp import Llama

    llm = Llama(model_path=gguf, n_gpu_layers=0, n_ctx=len(token_ids) + 8, logits_all=True, verbose=False)
    llm.eval(token_ids)
    scores = np.array(llm.scores[: len(token_ids)])
    rows = {}
    for pos in range(len(token_ids)):
        top = np.argsort(scores[pos])[::-1][:5]
        rows[pos] = [(int(t), float(scores[pos][t])) for t in top]
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f)
    return rows


def tokenize(gguf, text, count):
    from llama_cpp import Llama

    llm = Llama(model_path=gguf, vocab_only=True, verbose=False)
    ids = llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)
    if len(ids) < count:
        sys.exit(f"the text has {len(ids)} tokens, {count} requested; pass a longer --text")
    return ids[:count]


def resolve_model(args):
    """--model NAME (a tools/fetch_models.py catalog name) -> args.gguf, downloading it when --fetch
    (or BORING_LLM_FETCH=1) is given and the file is missing."""
    if getattr(args, "model", None):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import fetch_models
        args.gguf = fetch_models.ensure(args.model, fetch=fetch_models.want_fetch(args.fetch))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf")
    ap.add_argument("--model", help="a model of tools/fetch_models.py instead of --gguf (e.g. qwen2.5-0.5b-instruct-q8_0)")
    ap.add_argument("--fetch", action="store_true", help="with --model: download it when missing")
    ap.add_argument("--binary", help="a boring-llm executable (Metal or wgpu build)")
    ap.add_argument("--text", default=DEFAULT_TEXT, help="text to tokenize (default: tools/accuracy_text.txt, frozen so that results stay comparable)")
    ap.add_argument("--positions", type=int, default=613)
    ap.add_argument("--min-top1", type=float, default=0.95, help="required fraction of top-1 agreement")
    ap.add_argument("--max-mean-diff", type=float, default=0.5, help="allowed mean |logit diff|")
    ap.add_argument("--cache-dir", default=".accuracy-cache")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    resolve_model(args)
    if not args.gguf or not args.binary:
        ap.error("--gguf (or --model) and --binary are required")
    with open(args.text, encoding="utf-8") as f:
        ids = tokenize(args.gguf, f.read(), args.positions)
    reference = reference_logits(args.gguf, ids, args.cache_dir)
    run = subprocess.run(
        [args.binary, "--gguf", args.gguf, "--quantized", "--tokens", ",".join(map(str, ids))],
        capture_output=True, text=True, timeout=args.timeout, stdin=subprocess.DEVNULL,
    )
    ours = parse_positions(run.stdout)
    if len(ours) < len(ids):
        sys.exit(f"{args.binary} printed {len(ours)} of {len(ids)} positions (exit {run.returncode}):\n{run.stderr[-800:]}")
    n, agree, mean, missing = compare(reference, ours)
    ok = agree / n >= args.min_top1 and mean <= args.max_mean_diff
    print(f"{os.path.basename(args.gguf)}: top-1 {agree}/{n} ({100 * agree / n:.1f}%), "
          f"mean |logit diff| {mean:.3f}, reference token outside our top 5 at {missing} positions "
          f"-> {'ok' if ok else 'FAILED'} (needs >= {args.min_top1:.0%} and <= {args.max_mean_diff})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
