#!/usr/bin/env python3
"""Download the GGUF models the tests, tools and benchmarks use, on demand.

Models are not part of the repository (`weights/` is git-ignored: they are
hundreds of MB to several GB). This script knows where each one comes from,
downloads it into `weights/` only when it is missing, and checks it against the
SHA-256 Hugging Face publishes. Nothing is ever downloaded unless you ask:
`ensure` asks, and `path` asks only with --fetch or BORING_LLM_FETCH=1.

Usage:
    tools/fetch_models.py list                      # catalog, what is present, sizes
    tools/fetch_models.py ensure qwen3-4b-q4_k_m    # download if missing (resumes a partial file)
    tools/fetch_models.py path qwen2.5-0.5b-instruct-q8_0          # print the path; exit 2 if missing
    tools/fetch_models.py path qwen2.5-0.5b-instruct-q8_0 --fetch  # ... downloading it first if missing
    tools/fetch_models.py --selftest

In scripts and tests:
    MODEL=$(tools/fetch_models.py path qwen2.5-0.5b-instruct-q4_k_m --fetch)
    ./test_quant_kernels_real "$MODEL"
    tools/accuracy_vs_llamacpp.py --model qwen2.5-0.5b-instruct-q8_0 --fetch --binary ...

Standard library only. The download resumes from `<file>.part`, refuses to start
when the disk would be left with less than 2 GB, and verifies the SHA-256
before moving the file into place.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEFAULT_DIR = os.path.join(ROOT, "weights")
HF = "https://huggingface.co"
MIN_FREE_AFTER = 2 * 1024**3

# name -> (Hugging Face repository, file name in that repository, file name in weights/, note)
CATALOG = {
    "qwen2.5-0.5b-instruct-q8_0": ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf", "qwen2.5-0.5b-instruct-q8_0.gguf", "accuracy checks, most tests"),
    "qwen2.5-0.5b-instruct-q4_k_m": ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf", "qwen2.5-0.5b-instruct-q4_k_m.gguf", "K-quant tests (Q4_K and Q6_K tensors)"),
    "qwen2.5-0.5b-instruct-q5_k_m": ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q5_k_m.gguf", "qwen2.5-0.5b-instruct-q5_k_m.gguf", "Q5_K and Q6_K tensors"),
    "qwen2.5-0.5b-instruct-iq4_xs": ("bartowski/Qwen2.5-0.5B-Instruct-GGUF", "Qwen2.5-0.5B-Instruct-IQ4_XS.gguf", "qwen2.5-0.5b-instruct-iq4_xs.gguf", "IQ4_XS tensors"),
    "tinymixtral-4x248m-q8_0": ("mradermacher/TinyMixtral-4x248M-MoE-GGUF", "TinyMixtral-4x248M-MoE.Q8_0.gguf", "tinymixtral-4x248m-q8_0.gguf", "mixtral architecture (4 experts, 2 used), Q8_0"),
    "tinymixtral-4x248m-q4_k_m": ("mradermacher/TinyMixtral-4x248M-MoE-GGUF", "TinyMixtral-4x248M-MoE.Q4_K_M.gguf", "tinymixtral-4x248m-q4_k_m.gguf", "mixtral architecture, K-quant experts"),
    "granite-3.3-2b-instruct-q4_k_m": ("ibm-granite/granite-3.3-2b-instruct-GGUF", "granite-3.3-2b-instruct-Q4_K_M.gguf", "granite-3.3-2b-instruct-q4_k_m.gguf", "granite architecture (embedding / attention / residual / logit scales)"),
    "smollm3-3b-q4_k_m": ("ggml-org/SmolLM3-3B-GGUF", "SmolLM3-Q4_K_M.gguf", "smollm3-3b-q4_k_m.gguf", "smollm3 architecture (NoPE layers), reasoning template"),
    "qwen2.5-0.5b-instruct-q4_0": ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_0.gguf", "qwen2.5-0.5b-instruct-q4_0.gguf", "test_quant_kernels_real"),
    "qwen2.5-0.5b-instruct-q2_k": ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q2_k.gguf", "qwen2.5-0.5b-instruct-q2_k.gguf", "test_quant_kernels_real"),
    "qwen2.5-0.5b-instruct-fp16": ("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-fp16.gguf", "qwen2.5-0.5b-instruct-fp16.gguf", "float reference"),
    "gemma-3-270m-it-q8_0": ("ggml-org/gemma-3-270m-it-GGUF", "gemma-3-270m-it-Q8_0.gguf", "gemma-3-270m-it-Q8_0.gguf", "gemma3 architecture (sliding window, post-norms, 262k vocabulary)"),
    "qwen2.5-3b-instruct-q4_k_m": ("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf", "qwen2.5-3b-instruct-q4_k_m.gguf", "OpenCode trials, server checks"),
    "qwen2.5-7b-instruct-q4_k_m": ("bartowski/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5-7B-Instruct-Q4_K_M.gguf", "qwen2.5-7b-instruct-q4_k_m.gguf", "speed and accuracy references (the official repository splits this file in two)"),
    "tinyllama-1.1b-chat-v1.0-q8_0": ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "tinyllama-1.1b-chat-v1.0.Q8_0.gguf", "tinyllama-1.1b-chat-v1.0-q8_0.gguf", "Llama architecture, SentencePiece tokenizer"),
    "qwen3-4b-q4_k_m": ("Qwen/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf", "qwen3-4b-q4_k_m.gguf", "qwen3 architecture"),
}


def local_path(name, directory=DEFAULT_DIR):
    return os.path.join(directory, CATALOG[name][2])


def pick_entry(tree, remote_name):
    """The (size, sha256) of `remote_name` in a Hugging Face `tree` listing, or None."""
    for e in tree:
        if e.get("path") == remote_name:
            lfs = e.get("lfs") or {}
            return int(lfs.get("size") or e.get("size") or 0), lfs.get("oid")
    return None


def enough_space(free_bytes, needed_bytes, margin=MIN_FREE_AFTER):
    return free_bytes - needed_bytes >= margin


def sha256_of(path, progress=None):
    h = hashlib.sha256()
    done = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8 << 20)
            if not chunk:
                break
            h.update(chunk)
            done += len(chunk)
            if progress:
                progress(done)
    return h.hexdigest()


def remote_info(repo, remote_name):
    url = f"{HF}/api/models/{repo}/tree/main"
    with urllib.request.urlopen(url, timeout=60) as r:
        info = pick_entry(json.load(r), remote_name)
    if info is None:
        sys.exit(f"{remote_name} is not in {repo} (the repository may have changed; update CATALOG)")
    return info


def download(name, directory=DEFAULT_DIR):
    repo, remote_name, local_name, _ = CATALOG[name]
    os.makedirs(directory, exist_ok=True)
    dest = os.path.join(directory, local_name)
    size, sha = remote_info(repo, remote_name)
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    free = shutil.disk_usage(directory).free
    if not enough_space(free, size - have):
        sys.exit(f"not enough disk space for {name}: needs {(size - have) / 1e9:.1f} GB, {free / 1e9:.1f} GB free "
                 f"(a margin of {MIN_FREE_AFTER / 1e9:.0f} GB is kept)")
    print(f"downloading {name}: {repo}/{remote_name}, {size / 1e9:.2f} GB"
          + (f", resuming at {have / 1e9:.2f} GB" if have else ""), file=sys.stderr)
    req = urllib.request.Request(f"{HF}/{repo}/resolve/main/{remote_name}")
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have == size:  # already complete
            resp = None
        else:
            raise
    if resp is not None:
        with resp, open(part, "ab" if have and resp.status == 206 else "wb") as out:
            done = have if resp.status == 206 else 0
            last = -1
            while True:
                chunk = resp.read(4 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                pct = int(100 * done / size) if size else 0
                if pct != last and pct % 5 == 0:
                    print(f"  {pct}%", file=sys.stderr)
                    last = pct
    if os.path.getsize(part) != size:
        sys.exit(f"{part} has {os.path.getsize(part)} bytes, expected {size}: run the command again to resume")
    if sha:
        print("verifying SHA-256...", file=sys.stderr)
        got = sha256_of(part)
        if got != sha:
            os.remove(part)
            sys.exit(f"SHA-256 mismatch for {name}: got {got}, Hugging Face publishes {sha}; the partial file was removed")
    os.replace(part, dest)
    print(f"{name}: {dest}", file=sys.stderr)
    return dest


def ensure(name, directory=DEFAULT_DIR, fetch=True):
    """Path of model `name`, downloading it first when `fetch` and it is missing."""
    if name not in CATALOG:
        sys.exit(f"unknown model '{name}'; known: {', '.join(sorted(CATALOG))}")
    path = local_path(name, directory)
    if os.path.exists(path):
        return path
    if not fetch:
        sys.exit(f"{path} is missing. Download it with: tools/fetch_models.py ensure {name}  "
                 f"(or pass --fetch, or set BORING_LLM_FETCH=1)")
    return download(name, directory)


def want_fetch(flag):
    return flag or os.environ.get("BORING_LLM_FETCH", "") in ("1", "true", "yes")


def selftest():
    assert len({v[2] for v in CATALOG.values()}) == len(CATALOG), "local file names must be unique"
    for name, (repo, remote, local, _) in CATALOG.items():
        assert "/" in repo and remote.endswith(".gguf") and local.endswith(".gguf"), name
        assert local == local.lower(), name
    tree = [{"path": "a.gguf", "size": 5, "lfs": {"oid": "ab" * 32, "size": 100}}, {"path": "README.md", "size": 3}]
    assert pick_entry(tree, "a.gguf") == (100, "ab" * 32)
    assert pick_entry(tree, "README.md") == (3, None)
    assert pick_entry(tree, "missing.gguf") is None
    assert enough_space(10 * 1024**3, 5 * 1024**3) and not enough_space(6 * 1024**3, 5 * 1024**3)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x")
        with open(p, "wb") as f:
            f.write(b"abc")
        assert sha256_of(p) == hashlib.sha256(b"abc").hexdigest()
        name = "qwen2.5-0.5b-instruct-q8_0"
        try:
            ensure(name, d, fetch=False)
            raise AssertionError("ensure(fetch=False) must not succeed on a missing file")
        except SystemExit as e:
            assert "tools/fetch_models.py ensure" in str(e)
        open(local_path(name, d), "wb").close()
        assert ensure(name, d, fetch=False) == local_path(name, d)
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=["list", "ensure", "path"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--dir", default=DEFAULT_DIR, help="where models live (default: weights/)")
    ap.add_argument("--fetch", action="store_true", help="path: download a missing model")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.command is None or args.command == "list":
        for name, (repo, remote, local, note) in CATALOG.items():
            p = local_path(name, args.dir)
            state = f"{os.path.getsize(p) / 1e9:5.2f} GB" if os.path.exists(p) else "missing "
            print(f"{state}  {name:34s} {repo}  ({note})")
        return 0
    if not args.names:
        ap.error(f"{args.command} needs at least one model name")
    for name in args.names:
        path = ensure(name, args.dir, fetch=True if args.command == "ensure" else want_fetch(args.fetch))
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
