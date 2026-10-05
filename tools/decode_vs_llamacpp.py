#!/usr/bin/env python3
"""Decode-path check of a boring-llm binary against llama.cpp.

Feeds the same prompt token ids to both, greedy-decodes N tokens, and compares the
generated ids. llama.cpp runs on the CPU, so it is an independent oracle of the GPU kernels.

Why it exists: tools/accuracy_vs_llamacpp.py compares logits from `--tokens`, which runs
one batched forward pass. Generation uses a different path (prefill in chunks, then one
`forward_step` per token with the KV cache and the single-row kernels). A commit that
broke only that path (a fused residual add that updated a copy of the residual stream,
so every layer lost its attention output) passed the logits check, the unit tests and
the benchmarks, and shipped a model that printed random text. Run this one for every
change that touches the forward pass or the GPU kernels, next to the logits check.

How a prompt is judged: the two continuations are compared token by token. When they
agree to the end, it passes. At the first token where they differ, the comparison stops
(the contexts differ from there on) and the gap decides: how much higher llama.cpp's own
logit is for its token than for ours, at that position. A near-tie (gap below
--max-gap, default 1.0; healthy kernels differ from llama.cpp by about 0.06 to 0.16 on
average) is rounding noise and passes; a token llama.cpp finds clearly worse, or outside
its top five, fails. A broken decode path differs at the first or second token with a gap
of several units.

Usage:
    tools/decode_vs_llamacpp.py --gguf weights/qwen2.5-0.5b-instruct-q8_0.gguf --binary ./main_metal
    tools/decode_vs_llamacpp.py --model qwen2.5-0.5b-instruct-q4_k_m --fetch --binary ./main_metal
    tools/decode_vs_llamacpp.py --selftest

The default prompts use the ChatML turn markers of the Qwen2.5 family; for another chat
format pass --prompts-file (a JSON list of strings, tokenized with special tokens on).
Needs the project's .venv (llama-cpp-python, numpy). The reference continuations are
cached in `.accuracy-cache/`. Exit status 0 when every prompt passes, 1 otherwise.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

CHATML = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
DEFAULT_PROMPTS = [
    CHATML.format("Write a short poem about the sea."),
    CHATML.format("List three prime numbers and explain what a prime number is."),
    CHATML.format("What is the capital of France, and what is it known for?"),
]
GENERATED_RE = re.compile(r"^generated: ?([-\d,]*)\s*$", re.M)


def parse_generated(output):
    """boring-llm's `--generate-tokens` output: the list of generated ids, or None when absent."""
    m = GENERATED_RE.search(output)
    if not m:
        return None
    return [int(t) for t in m.group(1).split(",") if t]


def until_eos(tokens, eos_ids):
    for i, t in enumerate(tokens):
        if t in eos_ids:
            return list(tokens[: i + 1])
    return list(tokens)


def verdict(reference, ours, tops, max_gap, eos_ids):
    """(tokens that agree, gap at the first difference or None, ok).

    `tops[i]` is llama.cpp's top logits at generation step i, as [(token, logit), ...], best
    first. The gap is llama.cpp's top logit minus its logit for our token; when our token is
    not among the listed ones it is at least the distance to the last listed one."""
    ref = until_eos(reference, eos_ids)
    mine = until_eos(ours, eos_ids)
    for i in range(min(len(ref), len(mine))):
        if ref[i] != mine[i]:
            row = tops[i]
            logits = dict((t, v) for t, v in row)
            top = row[0][1]
            gap = top - logits[mine[i]] if mine[i] in logits else top - row[-1][1]
            return i, gap, gap < max_gap
    return min(len(ref), len(mine)), None, len(ref) == len(mine)


def selftest():
    assert parse_generated("[diag] x\ngenerated: 5,6,7\n") == [5, 6, 7]
    assert parse_generated("generated: \n") == []
    assert parse_generated("nothing") is None
    eos = {99}
    tops = [[(1, 10.0), (2, 9.9)], [(2, 8.0), (3, 7.5), (4, 7.0)], [(3, 5.0), (7, 1.0)]]
    assert verdict([1, 2, 3], [1, 2, 3], tops, 1.0, eos) == (3, None, True)
    assert verdict([1, 2, 3, 99, 5], [1, 2, 3, 99, 6], tops, 1.0, eos) == (4, None, True)  # stop at end of turn
    assert verdict([1, 2, 3], [2, 2, 3], tops, 1.0, eos)[2] is True  # near-tie at the first token
    assert verdict([1, 2, 3], [1, 4, 3], tops, 1.0, eos) == (1, 1.0, False)  # a worse token by exactly the gap
    assert verdict([1, 2, 3], [1, 9, 3], tops, 1.0, eos) == (1, 1.0, False)  # outside the list: >= distance to the last
    assert verdict([1, 2, 3], [1, 2, 7], tops, 1.0, eos) == (2, 4.0, False)
    assert verdict([1, 2, 3], [1, 2], tops, 1.0, eos) == (2, None, False)  # ours stopped early
    print("selftest ok")


def reference_continuations(gguf, prompts, new_tokens, cache_dir):
    """[(prompt ids, greedy continuation ids, eos ids, top-5 logits per step)] from llama.cpp on the CPU."""
    key = hashlib.sha256((os.path.basename(gguf) + str(os.path.getsize(gguf)) + str(new_tokens)
                          + json.dumps(prompts) + "v2").encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, "decode-" + key + ".json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    import numpy as np
    from llama_cpp import Llama

    rows = []
    for text in prompts:
        llm = Llama(model_path=gguf, n_gpu_layers=0, n_ctx=2048, verbose=False)
        ids = llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)
        eos = {int(llm.token_eos())}
        out = []
        for t in llm.generate(ids, top_k=1, top_p=1.0, temp=0.0, repeat_penalty=1.0, reset=True):
            out.append(int(t))
            if t in eos or len(out) >= new_tokens:
                break
        del llm
        # the logits behind each step, from one teacher-forced pass over prompt + continuation
        llm = Llama(model_path=gguf, n_gpu_layers=0, n_ctx=len(ids) + len(out) + 8, logits_all=True, verbose=False)
        llm.eval(ids + out)
        tops = []
        for step in range(len(out)):
            row = np.array(llm.scores[len(ids) - 1 + step])
            best = np.argsort(row)[::-1][:5]
            tops.append([[int(t), float(row[t])] for t in best])
        del llm
        rows.append([ids, out, sorted(eos), tops])
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f)
    return rows


def resolve_model(args):
    if getattr(args, "model", None):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import fetch_models
        args.gguf = fetch_models.ensure(args.model, fetch=fetch_models.want_fetch(args.fetch))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf")
    ap.add_argument("--model", help="a model of tools/fetch_models.py instead of --gguf")
    ap.add_argument("--fetch", action="store_true", help="with --model: download it when missing")
    ap.add_argument("--binary", help="a boring-llm executable (Metal or wgpu build)")
    ap.add_argument("--prompts-file", help="JSON list of prompt strings (default: three ChatML prompts)")
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--max-gap", type=float, default=1.0,
                    help="largest logit gap, in llama.cpp's own logits, that still counts as a near-tie")
    ap.add_argument("--cache-dir", default=".accuracy-cache")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    resolve_model(args)
    if not args.gguf or not args.binary:
        ap.error("--gguf (or --model) and --binary are required")
    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as f:
            prompts = json.load(f)
    rows = reference_continuations(args.gguf, prompts, args.new_tokens, args.cache_dir)
    failed = 0
    for text, (ids, ref, eos, tops) in zip(prompts, rows):
        run = subprocess.run(
            [args.binary, "--gguf", args.gguf, "--generate-tokens", ",".join(map(str, ids)),
             "--max-tokens", str(args.new_tokens)],
            capture_output=True, text=True, timeout=args.timeout, stdin=subprocess.DEVNULL,
        )
        ours = parse_generated(run.stdout)
        name = os.path.basename(args.gguf)
        label = repr(text[-50:].replace("\n", " "))
        if ours is None:
            print(f"{name}: {label}: no `generated:` line (exit {run.returncode}):\n{run.stderr[-600:]}")
            failed += 1
            continue
        agreed, gap, ok = verdict(ref, ours, tops, args.max_gap, set(eos))
        what = (f"agrees on all {agreed} tokens" if gap is None and ok
                else f"differs at token {agreed}, llama.cpp's logit gap {gap:.2f}" if gap is not None
                else f"stopped after {agreed} tokens, llama.cpp continues")
        print(f"{name}: {label}: {what} -> {'ok' if ok else 'FAILED'}")
        failed += not ok
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
