#!/usr/bin/env python3
"""Speed comparison of boring-llm and llama.cpp (llama-cpp-python) on the same GGUF file.

Both engines are driven the same way: a chat request with a fixed prompt of about N
tokens, greedy, streamed, and timed on the client side:

  prefill tok/s = prompt tokens / time to the first content token
  decode  tok/s = (generated tokens - 1) / time between the first and the last content token

Each repetition loads one engine, measures every prompt size, frees it, then does the
same for the other one, and the whole thing is repeated `--reps` times (alternating
A/B, like every timing on a busy machine should be). Models are never resident in both
engines at once (a 7B model does not fit twice in 16 GB).

What is not equal, and is reported rather than hidden: llama.cpp keeps its KV cache in
float16 by default (`--llama-kv f32` changes that), boring-llm in float32; the two
tokenize the same text with the same GGUF vocabulary, so prompt sizes agree to a few
tokens, and the real counts are printed; llama-cpp-python adds a little Python around
llama.cpp (a few percent on decode at most).

Usage (repo root, a quiet machine; stop other heavy work first):
    tools/bench_vs_llamacpp.py --gguf weights/qwen2.5-7b-instruct-q4_k_m.gguf \\
        --binary src/main_metal/target/release/main --prompt-tokens 512,4096 --reps 3
    tools/bench_vs_llamacpp.py --selftest        # checks the parsing and statistics, no model
Needs llama-cpp-python (the project's .venv has it): run it with .venv/bin/python.

With several prompt sizes, the report also converts decode throughput to milliseconds
per token and prints the endpoint slope. This is the next-context-cost diagnostic: the
short endpoint contains mostly fixed projection/launch work, while its growth towards
4k/8k tokens points at attention and KV-cache work.
"""
import argparse
import http.client
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

SENTENCE = "The quick brown fox jumps over the lazy dog. "   # ~10 tokens in the Qwen/Llama vocabularies
ASK = "\nNow write a numbered list of 100 different animals, one per line, each with a short comment."


def build_prompt(target_tokens, salt):
    """A user message of about `target_tokens` tokens. `salt` makes each request's text
    differ from the very first words, so no engine can reuse a previous request's cache."""
    n = max(1, target_tokens // 10)
    return f"Session {salt}. " + SENTENCE * n + ASK


def summarize(values):
    values = sorted(values)
    return {"min": values[0], "median": statistics.median(values), "max": values[-1], "n": len(values)}


def median_decode_ms(results, engine, size):
    runs = results.get((engine, size), [])
    if not runs:
        return None
    return statistics.median(1000.0 / r["decode_tps"] for r in runs)


def decode_slope(results, engine, sizes):
    """Extra decode milliseconds per 1,000 cached prompt tokens, using endpoints."""
    lo, hi = min(sizes), max(sizes)
    if lo == hi:
        return None
    lo_ms = median_decode_ms(results, engine, lo)
    hi_ms = median_decode_ms(results, engine, hi)
    if lo_ms is None or hi_ms is None:
        return None
    return {
        "low_size": lo, "high_size": hi, "low_ms": lo_ms, "high_ms": hi_ms,
        "delta_ms": hi_ms - lo_ms,
        "ms_per_1k_tokens": (hi_ms - lo_ms) * 1000.0 / (hi - lo),
    }


def rates(t_start, content_times, prompt_tokens):
    """Prefill and decode rates from the instants (seconds) of each content chunk."""
    if len(content_times) < 3:
        raise RuntimeError(f"only {len(content_times)} content chunks: the reply is too short to time")
    ttft = content_times[0] - t_start
    span = content_times[-1] - content_times[0]
    return {
        "prompt_tokens": prompt_tokens,
        "generated": len(content_times),
        "ttft_s": ttft,
        "prefill_tps": prompt_tokens / ttft if ttft > 0 else float("inf"),
        "decode_tps": (len(content_times) - 1) / span if span > 0 else float("inf"),
    }


def sse_content_times(lines, clock):
    """Timestamps of the chunks that carry text, and the usage block if any. `lines` yields
    decoded SSE lines as they arrive; comments (keep-alive) and role chunks are ignored."""
    times, usage = [], None
    for raw in lines:
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        obj = json.loads(payload)
        if obj.get("usage"):
            usage = obj["usage"]
        for ch in obj.get("choices", []):
            delta = ch.get("delta", {})
            # a reasoning model (Qwen3) streams `reasoning_content` before its first `content`
            if delta.get("content") or delta.get("reasoning_content"):
                times.append(clock())
    return times, usage


# ── boring-llm ───────────────────────────────────────────────────────────

class Ours:
    name = "boring-llm"

    def __init__(self, args):
        self.args = args
        self.log = tempfile.NamedTemporaryFile("w+", prefix="bench-boring-llm-", suffix=".log", delete=False)
        cmd = [args.binary, "--gguf", args.gguf, "--serve", "--port", str(args.port),
               "--ctx-size", str(args.ctx), "--cache-slots", "1"] + args.ours_arg
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.time() + 600
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"boring-llm exited early, see {self.log.name}")
            try:
                c = http.client.HTTPConnection("127.0.0.1", args.port, timeout=2)
                c.request("GET", "/health")
                if c.getresponse().status == 200:
                    return
            except OSError:
                time.sleep(1)
        raise RuntimeError("boring-llm did not become ready")

    def run(self, prompt, gen):
        body = json.dumps({"model": "x", "stream": True, "max_tokens": gen, "temperature": 0,
                           "stream_options": {"include_usage": True},
                           "messages": [{"role": "user", "content": prompt}]})
        c = http.client.HTTPConnection("127.0.0.1", self.args.port, timeout=3600)
        t0 = time.perf_counter()
        c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
        resp = c.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:200]!r}")
        times, usage = sse_content_times((l.decode("utf-8", "replace") for l in resp), time.perf_counter)
        c.close()
        return rates(t0, times, (usage or {}).get("prompt_tokens", 0))

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()
        os.unlink(self.log.name)


# ── llama.cpp ────────────────────────────────────────────────────────────

class Llama:
    name = "llama.cpp"

    def __init__(self, args):
        from llama_cpp import Llama as L
        kv = {"f16": 1, "f32": 0}[args.llama_kv]
        self.llm = L(model_path=args.gguf, n_ctx=args.ctx, n_gpu_layers=-1, n_batch=512,
                     type_k=kv, type_v=kv, flash_attn=args.llama_flash_attn, verbose=False)

    def run(self, prompt, gen):
        self.llm.reset()                       # no prefix reuse between requests
        t0 = time.perf_counter()
        times = []
        for chunk in self.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}], max_tokens=gen,
                temperature=0, stream=True):
            if chunk["choices"][0].get("delta", {}).get("content"):
                times.append(time.perf_counter())
        # the last sampled token is never fed back, so n_tokens = prompt + generated - 1 (+-1 is irrelevant here)
        prompt_tokens = self.llm.n_tokens - len(times) + 1
        return rates(t0, times, prompt_tokens)

    def close(self):
        del self.llm


# ── driver ───────────────────────────────────────────────────────────────

def load_average():
    return os.getloadavg()[0]


def machine():
    out = []
    for key in ("machdep.cpu.brand_string", "hw.memsize"):
        try:
            out.append(subprocess.check_output(["sysctl", "-n", key], text=True).strip())
        except Exception:
            pass
    return " / ".join(out)


def report(results, sizes):
    print()
    print(f"machine: {machine()}")
    print(f"{'prompt':>8}  {'engine':<11} {'prefill tok/s (min / median / max)':<40} {'decode tok/s (min / median / max)':<36}")
    for size in sizes:
        row = {}
        for eng in ("boring-llm", "llama.cpp"):
            runs = results.get((eng, size), [])
            if not runs:
                continue
            pre = summarize([r["prefill_tps"] for r in runs])
            dec = summarize([r["decode_tps"] for r in runs])
            toks = int(statistics.median([r["prompt_tokens"] for r in runs]))
            row[eng] = (pre["median"], dec["median"])
            print(f"{toks:>8}  {eng:<11} {pre['min']:>9.1f} / {pre['median']:>9.1f} / {pre['max']:>9.1f}      "
                  f"{dec['min']:>8.2f} / {dec['median']:>8.2f} / {dec['max']:>8.2f}   (n={pre['n']})")
        if len(row) == 2:
            b, l = row["boring-llm"], row["llama.cpp"]
            print(f"{'':>8}  boring-llm is {b[0] / l[0]:.2f}x llama.cpp on prefill, {b[1] / l[1]:.2f}x on decode (medians)")
    if len(sizes) > 1:
        print("\ndecode context cost (endpoint medians; descriptive, not a fitted model):")
        for eng in ("boring-llm", "llama.cpp"):
            slope = decode_slope(results, eng, sizes)
            if slope:
                print(f"  {eng:<11} {slope['low_ms']:.2f} -> {slope['high_ms']:.2f} ms/token "
                      f"({slope['delta_ms']:+.2f} ms; {slope['ms_per_1k_tokens']:+.2f} ms per 1k prompt tokens)")


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
    ap.add_argument("--model", help="a model of tools/fetch_models.py instead of --gguf")
    ap.add_argument("--fetch", action="store_true", help="with --model: download it when missing")
    ap.add_argument("--binary", help="boring-llm binary built with --target metal (or wgpu)")
    ap.add_argument("--prompt-tokens", default="512,4096", help="comma-separated prompt sizes")
    ap.add_argument("--gen", type=int, default=64, help="tokens to generate per request")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--engines", default="ours,llama")
    ap.add_argument("--llama-kv", choices=["f16", "f32"], default="f16")
    ap.add_argument("--llama-flash-attn", action="store_true")
    ap.add_argument("--ours-arg", action="append", default=[], help="extra argument for boring-llm (repeatable)")
    ap.add_argument("--max-load", type=float, default=4.0, help="refuse to start above this 1-minute load average")
    ap.add_argument("--json", help="also write the raw results here")
    ap.add_argument("--label", default="", help="free-form run label stored in --json output")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    resolve_model(args)
    if not args.gguf:
        ap.error("--gguf (or --model) is required")
    engines = args.engines.split(",")
    if "ours" in engines and not args.binary:
        ap.error("--binary is required to measure boring-llm")
    sizes = [int(s) for s in args.prompt_tokens.split(",")]
    if max(sizes) + args.gen + 64 > args.ctx:
        ap.error("--ctx must hold the largest prompt plus the reply")
    load = load_average()
    if load > args.max_load:
        sys.exit(f"load average is {load:.1f} (> {args.max_load}): the machine is busy, the numbers would be noise. "
                 f"Wait, or pass --max-load.")

    results, salt = {}, 0
    for rep in range(args.reps):
        for key in engines:
            eng = Ours(args) if key == "ours" else Llama(args)
            try:
                # one throwaway request: shader/pipeline compilation is not what is being timed
                salt += 1
                eng.run(build_prompt(64, salt), 8)
                for size in sizes:
                    salt += 1
                    r = eng.run(build_prompt(size, salt), args.gen)
                    results.setdefault((eng.name, size), []).append(r)
                    print(f"rep {rep + 1}/{args.reps} {eng.name:<11} prompt {r['prompt_tokens']:>5}: "
                          f"prefill {r['prefill_tps']:8.1f} tok/s, decode {r['decode_tps']:6.2f} tok/s "
                          f"({r['generated']} tokens)", flush=True)
            finally:
                eng.close()
    final_load = load_average()
    print(f"\nload average before/after: {load:.1f} / {final_load:.1f}")
    report(results, sizes)
    if args.json:
        parent = os.path.dirname(os.path.abspath(args.json))
        os.makedirs(parent, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({
                "schema": 2, "label": args.label, "machine": machine(),
                "load_average": {"before": load, "after": final_load},
                "configuration": {
                    "gguf": os.path.abspath(args.gguf),
                    "binary": os.path.abspath(args.binary) if args.binary else None,
                    "prompt_tokens": sizes, "generated_tokens": args.gen, "repetitions": args.reps,
                    "context_size": args.ctx, "engines": engines, "llama_kv": args.llama_kv,
                    "llama_flash_attn": args.llama_flash_attn, "ours_arg": args.ours_arg,
                },
                "results": {f"{e}|{s}": v for (e, s), v in results.items()},
                "decode_slopes": {e: decode_slope(results, e, sizes)
                                  for e in ("boring-llm", "llama.cpp")},
            }, f, indent=1)


def selftest():
    # build_prompt: size and salt
    p = build_prompt(512, 7)
    assert p.startswith("Session 7. ") and p.count(SENTENCE) == 51 and p.endswith("short comment.")
    assert build_prompt(512, 8) != p
    # SSE parsing: role chunk, keep-alive comments and the usage chunk carry no text
    clock = iter([1.0, 1.1, 1.2, 1.3, 1.4]).__next__
    sse = [
        'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}', "",
        ": prefill 512/900", "",
        'data: {"choices":[{"delta":{"reasoning_content":"r"}}]}', "",
        'data: {"choices":[{"delta":{"content":"A"}}]}', "",
        'data: {"choices":[{"delta":{"content":" b"}}]}', "",
        'data: {"choices":[{"delta":{"content":" c"}}]}', "",
        'data: {"choices":[{"delta":{"content":" d"}}]}', "",
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}', "",
        'data: {"choices":[],"usage":{"prompt_tokens":900,"completion_tokens":4}}', "",
        "data: [DONE]", ""]
    times, usage = sse_content_times(iter(sse), clock)
    assert times == [1.0, 1.1, 1.2, 1.3, 1.4] and usage["prompt_tokens"] == 900
    r = rates(0.0, [2.0, 2.5, 3.0, 3.5], 900)
    assert abs(r["prefill_tps"] - 450.0) < 1e-9 and abs(r["decode_tps"] - 2.0) < 1e-9, r
    try:
        rates(0.0, [1.0], 10)
        raise AssertionError("a one-token reply must be rejected")
    except RuntimeError:
        pass
    s = summarize([3.0, 1.0, 2.0])
    assert s == {"min": 1.0, "median": 2.0, "max": 3.0, "n": 3}
    fake = {
        ("boring-llm", 500): [{"decode_tps": 100.0}],
        ("boring-llm", 4500): [{"decode_tps": 50.0}],
    }
    slope = decode_slope(fake, "boring-llm", [500, 4500])
    assert slope == {"low_size": 500, "high_size": 4500, "low_ms": 10.0, "high_ms": 20.0,
                     "delta_ms": 10.0, "ms_per_1k_tokens": 2.5}, slope
    print("selftest: ok")


if __name__ == "__main__":
    main()
