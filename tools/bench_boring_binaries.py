#!/usr/bin/env python3
"""Paired A/B decode benchmark for two boring-llm binaries."""
import argparse
import json
import os
import statistics
from types import SimpleNamespace

import bench_vs_llamacpp as bench


def order(rep):
    return ("baseline", "candidate") if rep % 2 == 0 else ("candidate", "baseline")


def summary(values):
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def selftest():
    assert order(0) == ("baseline", "candidate")
    assert order(1) == ("candidate", "baseline")
    assert summary([3.0, 1.0, 2.0]) == {"min": 1.0, "median": 2.0, "max": 3.0}
    print("selftest: ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf")
    ap.add_argument("--baseline")
    ap.add_argument("--candidate")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--max-load", type=float, default=4.0)
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.gguf or not args.baseline or not args.candidate:
        ap.error("--gguf, --baseline, and --candidate are required")
    if args.reps < 1 or args.gen < 2:
        ap.error("--reps must be positive and --gen must be at least 2")
    if args.prompt_tokens + args.gen + 64 > args.ctx:
        ap.error("--ctx must hold the prompt plus reply")

    binaries = {"baseline": os.path.abspath(args.baseline),
                "candidate": os.path.abspath(args.candidate)}
    samples = {name: [] for name in binaries}
    paired = []
    loads = []
    salt = 0
    for rep in range(args.reps):
        current = bench.load_average()
        loads.append(current)
        if current > args.max_load:
            raise RuntimeError(f"load average is {current:.1f} (> {args.max_load}) before pair {rep + 1}")
        pair = {}
        for name in order(rep):
            current = bench.load_average()
            loads.append(current)
            if current > args.max_load:
                raise RuntimeError(f"load average rose to {current:.1f} (> {args.max_load})")
            server_args = SimpleNamespace(binary=binaries[name], gguf=os.path.abspath(args.gguf),
                                          port=args.port, ctx=args.ctx, ours_arg=[])
            engine = bench.Ours(server_args)
            try:
                salt += 1
                engine.run(bench.build_prompt(64, salt), 8)
                salt += 1
                result = engine.run(bench.build_prompt(args.prompt_tokens, salt), args.gen)
            finally:
                engine.close()
            ms = 1000.0 / result["decode_tps"]
            samples[name].append({**result, "ms_per_token": ms})
            pair[name] = ms
            print(f"rep {rep + 1}/{args.reps} {name:<9} {ms:7.3f} ms/token "
                  f"({result['decode_tps']:6.2f} tok/s)", flush=True)
        delta = pair["candidate"] - pair["baseline"]
        paired.append(delta)
        print(f"  paired candidate - baseline: {delta:+.3f} ms/token", flush=True)

    rows = {name: summary([sample["ms_per_token"] for sample in values])
            for name, values in samples.items()}
    median_delta = statistics.median(paired)
    median_speedup = rows["baseline"]["median"] / rows["candidate"]["median"]
    print(f"\nload range: {min(loads):.1f}-{max(loads):.1f}")
    for name, row in rows.items():
        print(f"{name:<9} {row['min']:.3f} / {row['median']:.3f} / {row['max']:.3f} ms/token")
    print(f"paired median delta: {median_delta:+.3f} ms/token; median speedup: {median_speedup:.3f}x")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"schema": 1, "machine": bench.machine(), "load_samples": loads,
                       "configuration": {"gguf": os.path.abspath(args.gguf), **binaries,
                                         "prompt_tokens": args.prompt_tokens,
                                         "generated_tokens": args.gen, "repetitions": args.reps,
                                         "context_size": args.ctx, "max_load": args.max_load},
                       "summary_ms_per_token": rows, "paired_deltas_ms_per_token": paired,
                       "paired_median_delta_ms_per_token": median_delta,
                       "median_speedup": median_speedup, "raw": samples}, f, indent=1)


if __name__ == "__main__":
    main()
