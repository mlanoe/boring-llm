#!/usr/bin/env python3
"""Measure marginal decode cost by duplicating one resident pipeline stage.

The binary's diagnostic ``--decode-profile-stage`` mode queues an extra copy of
the selected stage at every layer (or once for ``output-head``). This driver
alternates/rotates baseline and stage variants, warms every fresh server, and
reports the increase in milliseconds per generated token. No stage-local clock
or synchronization is introduced. On a machine whose performance drifts during
the run, ``--paired-baseline`` measures an adjacent baseline for every stage and
alternates A/B with B/A ordering; the load threshold is checked before every
sample, not only at startup.

Example (quiet machine, after rebuilding the Metal binary):
  .venv/bin/python tools/profile_decode_marginal.py \
    --gguf weights/qwen2.5-0.5b-instruct-q8_0.gguf \
    --binary src/main_metal/target/release/main --layers 24 --reps 5 \
    --json results/q8-decode-marginal.json
"""
import argparse
import json
import os
import statistics
from types import SimpleNamespace

import bench_vs_llamacpp as bench


STAGES = ("baseline", "qkv", "attention", "attn-proj", "residual-norm", "ffn",
          "ffn-gate", "ffn-up", "swiglu", "ffn-down", "output-head")


def run_order(rep, stages):
    """Rotate, then reverse every other repetition to spread thermal/order bias."""
    n = len(stages)
    order = list(stages[rep % n:]) + list(stages[:rep % n])
    return list(reversed(order)) if rep % 2 else order


def summarize(samples, layers):
    base = statistics.median(samples["baseline"])
    rows = {}
    for stage, values in samples.items():
        med = statistics.median(values)
        delta = med - base
        rows[stage] = {
            "min_ms_per_token": min(values), "median_ms_per_token": med,
            "max_ms_per_token": max(values), "delta_ms_per_token": delta,
            "delta_us_per_layer": None if stage in ("baseline", "output-head") else delta * 1000.0 / layers,
        }
    return rows


def summarize_paired(samples, paired_deltas, layers):
    """Summarize stage samples using their adjacent baseline, not a global one."""
    rows = summarize(samples, layers)
    for stage, deltas in paired_deltas.items():
        delta = statistics.median(deltas)
        rows[stage]["delta_ms_per_token"] = delta
        rows[stage]["delta_us_per_layer"] = (
            None if stage == "output-head" else delta * 1000.0 / layers
        )
    return rows


def selftest():
    assert run_order(0, ("a", "b", "c")) == ["a", "b", "c"]
    assert run_order(1, ("a", "b", "c")) == ["a", "c", "b"]
    rows = summarize({"baseline": [10.0, 11.0, 12.0], "ffn": [13.0, 14.0, 15.0],
                      "output-head": [12.0, 13.0, 14.0]}, 24)
    assert rows["ffn"]["delta_ms_per_token"] == 3.0
    assert rows["ffn"]["delta_us_per_layer"] == 125.0
    assert rows["output-head"]["delta_us_per_layer"] is None
    paired = summarize_paired(
        {"baseline": [10.0, 20.0], "ffn": [13.0, 24.0], "output-head": [12.0, 23.0]},
        {"ffn": [3.0, 4.0], "output-head": [2.0, 3.0]}, 24)
    assert paired["ffn"]["delta_ms_per_token"] == 3.5
    assert paired["ffn"]["delta_us_per_layer"] == 3.5 * 1000.0 / 24
    assert paired["output-head"]["delta_ms_per_token"] == 2.5
    print("selftest: ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf")
    ap.add_argument("--binary")
    ap.add_argument("--layers", type=int, default=24, help="model layer count (for the per-layer report)")
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--stages", default=",".join(STAGES), help="comma-separated subset including baseline")
    ap.add_argument("--max-load", type=float, default=4.0)
    ap.add_argument("--ours-arg", action="append", default=[], help="extra argument for boring-llm (repeatable), e.g. --ours-arg=-ctk --ours-arg=q8_0")
    ap.add_argument("--paired-baseline", action="store_true",
                    help="measure an adjacent baseline for every stage (better under drifting load)")
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.gguf or not args.binary:
        ap.error("--gguf and --binary are required")
    if args.layers < 1 or args.reps < 1:
        ap.error("--layers and --reps must be positive")
    stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
    unknown = set(stages) - set(STAGES)
    if unknown or "baseline" not in stages:
        ap.error(f"--stages must include baseline and use only: {', '.join(STAGES)}")
    if args.paired_baseline and len(stages) == 1:
        ap.error("--paired-baseline needs at least one non-baseline stage")
    if args.prompt_tokens + args.gen + 64 > args.ctx:
        ap.error("--ctx must hold the prompt plus reply")
    load_before = bench.load_average()
    if load_before > args.max_load:
        raise SystemExit(f"load average is {load_before:.1f} (> {args.max_load}); wait for a quiet machine")

    samples = {stage: [] for stage in stages}
    raw = {stage: [] for stage in stages}
    paired_deltas = {stage: [] for stage in stages if stage != "baseline"}
    paired_baselines = {stage: [] for stage in stages if stage != "baseline"}
    salt = 0
    common = dict(binary=os.path.abspath(args.binary), gguf=os.path.abspath(args.gguf),
                  port=args.port, ctx=args.ctx)

    def measure(stage):
        nonlocal salt
        current_load = bench.load_average()
        if current_load > args.max_load:
            raise RuntimeError(
                f"load average rose to {current_load:.1f} (> {args.max_load}) during the run")
        ours_arg = list(args.ours_arg) + ([] if stage == "baseline" else ["--decode-profile-stage", stage])
        server_args = SimpleNamespace(**common, ours_arg=ours_arg)
        engine = bench.Ours(server_args)
        try:
            salt += 1
            engine.run(bench.build_prompt(64, salt), 8)
            salt += 1
            result = engine.run(bench.build_prompt(args.prompt_tokens, salt), args.gen)
        finally:
            engine.close()
        ms = 1000.0 / result["decode_tps"]
        samples[stage].append(ms)
        raw[stage].append(result)
        print(f"rep {rep + 1}/{args.reps} {stage:<14} {ms:7.3f} ms/token "
              f"({result['decode_tps']:7.2f} tok/s, {result['generated']} tokens)", flush=True)
        return ms

    for rep in range(args.reps):
        order = run_order(rep, stages)
        if args.paired_baseline:
            order = [stage for stage in order if stage != "baseline"]
            for pair_index, stage in enumerate(order):
                if (rep + pair_index) % 2 == 0:
                    base_ms, stage_ms = measure("baseline"), measure(stage)
                else:
                    stage_ms, base_ms = measure(stage), measure("baseline")
                paired_baselines[stage].append(base_ms)
                paired_deltas[stage].append(stage_ms - base_ms)
        else:
            for stage in order:
                measure(stage)

    rows = (summarize_paired(samples, paired_deltas, args.layers)
            if args.paired_baseline else summarize(samples, args.layers))
    load_after = bench.load_average()
    load_window_valid = load_before <= args.max_load and load_after <= args.max_load
    print(f"\nload average before/after: {load_before:.1f} / {load_after:.1f}")
    if not load_window_valid:
        print("WARNING: load exceeded the configured threshold; do not publish these timings")
    print(f"{'stage':<14} {'ms/token min / median / max':<34} {'delta':>10} {'per layer':>12}")
    for stage in stages:
        row = rows[stage]
        per_layer = "--" if row["delta_us_per_layer"] is None else f"{row['delta_us_per_layer']:+.2f} us"
        print(f"{stage:<14} {row['min_ms_per_token']:7.3f} / {row['median_ms_per_token']:7.3f} / "
              f"{row['max_ms_per_token']:7.3f} {row['delta_ms_per_token']:+9.3f} ms {per_layer:>12}")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({
                "schema": 1, "machine": bench.machine(),
                "load_average": {"before": load_before, "after": load_after,
                                 "window_valid": load_window_valid},
                "configuration": {"gguf": common["gguf"], "binary": common["binary"],
                                  "layers": args.layers, "prompt_tokens": args.prompt_tokens,
                                  "generated_tokens": args.gen, "repetitions": args.reps,
                                  "context_size": args.ctx, "stages": stages,
                                  "paired_baseline": args.paired_baseline,
                                  "max_load": args.max_load},
                "summary": rows, "raw": raw,
                "paired_baselines_ms_per_token": paired_baselines if args.paired_baseline else None,
                "paired_deltas_ms_per_token": paired_deltas if args.paired_baseline else None,
            }, f, indent=1)


if __name__ == "__main__":
    main()
