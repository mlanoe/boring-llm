#!/usr/bin/env python3
"""Compare boring-llm's deterministic HTTP plumbing with llama-server.

The checks do not generate text: for every request fixture they compare the
rendered chat prompt, token ids and raw token pieces, detokenization, and the
chat input-token count. This isolates template/tokenizer/API drift from
floating-point model differences.

Usage:
    tools/compare_servers.py http://127.0.0.1:8080 http://127.0.0.1:8081
    tools/compare_servers.py BORING_URL LLAMA_URL --fixtures test/fixtures/requests/simple.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.error
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES = sorted((ROOT / "test/fixtures/requests").glob("*.json"))


def post(base: str, path: str, body: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        try:
            payload = json.load(error)
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {"raw_error": error.read().decode(errors="replace")}
        return error.code, payload


def first_difference(left, right, path: str = "$") -> str | None:
    """Return a compact path/value description of the first JSON difference."""
    if type(left) is not type(right):
        return f"{path}: {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return f"{path}: keys {list(left)} != {list(right)}"
        for key in left:
            difference = first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: lengths {len(left)} != {len(right)}"
        for index, (a, b) in enumerate(zip(left, right)):
            difference = first_difference(a, b, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if left != right:
        return f"{path}: {repr(left)[:160]} != {repr(right)[:160]}"
    return None


def compare_endpoint(name: str, boring_url: str, llama_url: str, path: str, body: dict) -> tuple[bool, dict, dict]:
    boring_status, boring = post(boring_url, path, body)
    llama_status, llama = post(llama_url, path, body)
    if boring_status != llama_status:
        print(f"FAIL  {name}: HTTP {boring_status} != {llama_status}")
        return False, boring, llama
    difference = first_difference(boring, llama)
    if difference:
        print(f"FAIL  {name}: {difference}")
        return False, boring, llama
    print(f"PASS  {name}")
    return True, boring, llama


def compare_fixture(path: pathlib.Path, boring_url: str, llama_url: str) -> int:
    body = json.loads(path.read_text())
    prefix = path.name
    failures = 0
    ok, boring_template, llama_template = compare_endpoint(
        f"{prefix} apply-template", boring_url, llama_url, "/apply-template", body
    )
    failures += not ok

    # Tokenize each server's own prompt. If prompts differ, this still reveals
    # whether tokenization agrees independently of template rendering.
    if "prompt" in boring_template and "prompt" in llama_template:
        bs, boring_tokens = post(boring_url, "/tokenize", {"content": boring_template["prompt"], "parse_special": True})
        ls, llama_tokens = post(llama_url, "/tokenize", {"content": llama_template["prompt"], "parse_special": True})
        difference = None if bs == ls else f"HTTP {bs} != {ls}"
        if difference is None:
            difference = first_difference(boring_tokens, llama_tokens)
        if difference:
            print(f"FAIL  {prefix} tokenize: {difference}")
            failures += 1
        else:
            print(f"PASS  {prefix} tokenize")

        # Grammar-constrained decoding operates on token pieces, not decoded
        # Unicode strings: a single token may contain only part of a UTF-8
        # sequence. Keep this comparison beside the ordinary token-id oracle so
        # future output constraints cannot accidentally use lossy pieces.
        bs, boring_pieces = post(
            boring_url, "/tokenize",
            {"content": boring_template["prompt"], "parse_special": True, "with_pieces": True},
        )
        ls, llama_pieces = post(
            llama_url, "/tokenize",
            {"content": llama_template["prompt"], "parse_special": True, "with_pieces": True},
        )
        difference = None if bs == ls else f"HTTP {bs} != {ls}"
        if difference is None:
            difference = first_difference(boring_pieces, llama_pieces)
        if difference:
            print(f"FAIL  {prefix} tokenize pieces: {difference}")
            failures += 1
        else:
            print(f"PASS  {prefix} tokenize pieces")

        # Detokenize the common token sequence only. If tokenization already
        # differs, feeding each server a different list would make this check
        # look useful while testing two different inputs.
        boring_ids = boring_tokens.get("tokens") if isinstance(boring_tokens, dict) else None
        llama_ids = llama_tokens.get("tokens") if isinstance(llama_tokens, dict) else None
        if bs == ls == 200 and boring_ids == llama_ids and isinstance(boring_ids, list):
            ok, _, _ = compare_endpoint(
                f"{prefix} detokenize", boring_url, llama_url,
                "/detokenize", {"tokens": boring_ids},
            )
            failures += not ok
        else:
            print(f"SKIP  {prefix} detokenize: token ids differ")

    ok, _, _ = compare_endpoint(
        f"{prefix} input_tokens", boring_url, llama_url,
        "/v1/chat/completions/input_tokens", body,
    )
    failures += not ok
    return failures


def selftest() -> None:
    assert first_difference({"a": [1, 2]}, {"a": [1, 2]}) is None
    assert first_difference({"a": [1, 2]}, {"a": [1, 3]}) == "$.a[1]: 2 != 3"
    assert "lengths" in first_difference([1], [1, 2])
    assert "keys" in first_difference({"a": 1}, {"b": 1})
    assert first_difference({"tokens": [{"id": 1, "piece": "a"}]},
                            {"tokens": [{"id": 1, "piece": "a"}]}) is None
    assert "piece" in first_difference({"tokens": [{"id": 1, "piece": "a"}]},
                                       {"tokens": [{"id": 1, "piece": "b"}]})
    print("compare_servers: selftest ok")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("boring_url", nargs="?")
    parser.add_argument("llama_url", nargs="?")
    parser.add_argument("--fixtures", nargs="+", type=pathlib.Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return 0
    if not args.boring_url or not args.llama_url:
        parser.error("BORING_URL and LLAMA_URL are required unless --selftest is used")

    failures = 0
    for fixture in args.fixtures:
        try:
            failures += compare_fixture(fixture, args.boring_url, args.llama_url)
        except (OSError, json.JSONDecodeError) as error:
            print(f"FAIL  {fixture.name}: {error}")
            failures += 1
    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
