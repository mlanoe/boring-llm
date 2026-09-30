# boring-llm

A real, GPU-accelerated LLM inference engine written in [Boring](https://github.com/mlanoe/boring), loading and running actual quantized [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) model files — the format used by [llama.cpp](https://github.com/ggerganov/llama.cpp) and the wider `ggml` ecosystem — with no conversion step.

## Goal

**Be a small, fast, portable LLM inference engine in the same spirit as llama.cpp** — load a `.gguf` file, run it directly on the GPU, get text out — but written entirely in Boring instead of C++, and used as a real-world proving ground for the language: a decoder-only transformer, real quantization formats, and Boring's own GPU kernel qualifiers (`'unified`/`'actor`/`'global`) driving genuine, not simulated, GPU compute across multiple backends.

## Features

- **Real GGUF loading, no conversion step.** Parses the format directly — header, every KV-metadata value kind (including the embedded tokenizer's vocab/merges as string arrays), tensor-info list — and reads the same files llama.cpp does.
- **Architecture-generic model/weight loading**: every config value is read from GGUF's own per-architecture metadata keys (not hardcoded to one model), and both RoPE conventions the Llama/Mistral/Qwen2 GGUF family needs (NEOX "rotate-half" vs. the original NORM "interleaved" style) and Qwen2's attention Q/K/V bias tensors (absent in Llama/Mistral) are handled generically. Verified against real Qwen2.5-0.5B-Instruct *and* real TinyLlama-1.1B-Chat (`llama` architecture) weights.
- **Multiple quantization formats**: F32, F16, Q4_0, Q5_0, Q8_0, Q2_K, Q3_K, Q4_K, Q6_K, IQ4_NL, and real-world files that mix several of these across different tensors (e.g. a typical "Q4_K_M" model, or a "Q2_K"-named preset that actually turns out to be an IQ4_NL/Q3_K/Q5_0/Q8_0/F32 mix at this model's size). Every format is ported bit-for-bit from ggml's own reference dequantization source and verified against real downloaded GGUF weights.
- **Fused-dequantization GPU kernels**: every quantization format above except F32/F16 (Q4_0, Q5_0, Q8_0, Q2_K, Q3_K, Q4_K, Q6_K, IQ4_NL) is dequantized on the fly inside the tiled-GEMM shared-memory load — including real mixed-format files — so the full model never needs a separate dequantized copy in memory. Verified against real downloaded weights end-to-end (`--quantized --gpu`) for Q8_0, Q4_K_M-mixed, Q4_0, and the IQ4_NL/Q3_K/Q5_0/Q8_0-mixed "Q2_K" preset; Q2_K itself has no real-world file to test against (see "Not yet done").
- **Two independently GPU-hardware-verified backends**: Metal and wgpu (cross-platform: Vulkan/DirectX12/Metal), producing bit-identical real-model output on both. CUDA and ROCm share the same qualified source but are untested on this machine (no hardware available).
- **GGUF-embedded tokenizer, both families**: byte-level BPE (Qwen2's own `tokenizer.ggml.model = "gpt2"`) and SentencePiece BPE (Llama/Mistral's `"llama"`, auto-detected), built directly from the file's own `tokenizer.ggml.*` metadata — no separate `vocab.json`/`merges.txt`/`tokenizer.model` export step for either.
- **Any architecture's own GGUF-embedded chat template, actually wired into `--prompt`/`--chat`.** A real Jinja2-subset interpreter (`src/jinja.br`) — `{% if/elif/else/for/set %}`, `{{ expr }}` output, `and`/`or`/`not`, attribute/index access, `is defined`, the `trim`/`tojson` filters, and full `{%-`/`-%}` whitespace-trim handling, verified byte-for-byte against real `jinja2` on both Qwen2.5's own tool-calling-capable template and TinyLlama's simpler one — renders each turn's conversation into prompt text, which `tokenizer.br`'s `render_chat_prompt`/`encode_with_specials` then tokenizes, recognizing the template's own literal special-token strings (`<|im_start|>`, `<s>`, ...) atomically instead of running them through ordinary BPE/SentencePiece merging (llama.cpp's own `tokenizer_st_partition` concept). Replaces the project's earlier Qwen2.5-only hardcoded ChatML construction entirely — verified byte-for-byte identical to it against real `llama-cpp-python` output (`test/test_tokenizer_real.br`), including the KV-cache multi-turn case, where each turn's incremental tokens are now derived generically (`full_ids[all_tokens.length..]`, re-rendering the whole conversation and slicing off what the cache already has) instead of a per-architecture hardcoded shape.
- **KV-cache incremental decoding**: O(1) GEMM work per new token per layer instead of re-forwarding the whole sequence every step.
- **Sampling**: greedy, or temperature/top-k/top-p, plus a llama.cpp-style repeat penalty.
- **Interactive multi-turn chat** (`--chat`, like llama.cpp's own `-cnv`): keeps the whole conversation, and with `--kv-cache` reuses each earlier turn's KV-cache instead of re-prefilling the whole conversation from scratch every turn — only the new turn's own tokens are processed.
- **Verified against llama.cpp**: token-for-token identical tokenization, and matching (often bit-identical) generation output against `llama-cpp-python`/`llama-cli`, on real downloaded model weights.

## What is Boring?

Boring is a high-level language that transpiles to idiomatic Rust. It borrows its syntax from Python and Swift, maps every construct directly onto Rust, and produces auditable output you can read, modify, and ship.

```boring
string greet(string? name):
    let who = name else "stranger"
    "Hello, {who}!"
```

See the [Boring repository](https://github.com/mlanoe/boring) for the full language reference.

## Goal model

Verified against [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) in GGUF format: small, Apache-2.0 licensed (no gated download), standard modern architecture (RMSNorm, RoPE, grouped-query attention, SwiGLU MLP) representative of the whole Llama/Mistral/Qwen GGUF family.

## Prerequisites

- [Rust toolchain](https://rustup.rs/) (edition 2021)
- [boring](https://github.com/mlanoe/boring) installed

```sh
cargo install --git https://github.com/mlanoe/boring
```

## Repository layout

```
boring-llm/
├── src/
│   ├── io.br             # LE binary readers (u16/u32/u64/f32, f16-bit-decode)
│   ├── gguf.br           # GGUF parser: header, typed KV-metadata, tensor-info list
│   ├── quant.br          # f16 bit-decode + Q8_0/Q5_0/Q6_K/Q4_K dequant (CPU reference)
│   ├── tokenizer.br      # BPE encode/decode, fed GGUF-embedded tokens/merges, ChatML
│   ├── model.br          # Config/weight loading, forward pass (fp32 + fused-Q8_0),
│   │                     # KV-cache (GenCache/forward_step/prefill), generate()/generate_kv()
│   ├── math.br           # CPU-only reference: RMSNorm, RoPE, naive GQA, SwiGLU
│   ├── math_gpu.br       # GPU kernels: tiled GEMM (+ warp-shuffle variant), RmsNorm/
│   │                     # RopeApply/SwigluCombine, GQA attention, fused-dequant Q8_0 GEMM
│   ├── sampling.br       # xorshift64 RNG, greedy/temperature/top-k/top-p sampling
│   └── main.br           # CLI entry point
├── tools/
│   ├── inspect_gguf.py      # Python `gguf`-package cross-check oracle (metadata dump)
│   ├── make_test_gguf.py    # synthetic small-fixture GGUF file generator
│   └── reference_logits.py  # llama-cpp-python logit oracle for math-correctness checks
├── weights/              # .gguf files, gitignored (download separately, see below)
├── test/
│   ├── fixtures/tiny.gguf   # small synthetic fixture for interpreter-speed tests
│   └── test_*.br            # per-module unit tests, plus *_real.br compiled-only tests
│                             # needing a real --target build + real downloaded weights
├── docs/development-log.md  # full phase-by-phase build history and bug hunt notes
└── .github/workflows/ci.yml # `boring run` over test/*.br (interpreter path only)
```

## Run

**Quick smoke test** (interpreter, no build, no download — uses the tiny synthetic fixture):

```sh
# The `--` is required — `boring run` otherwise treats `--gguf` as one of its
# own flags (e.g. `--gpu`/`--locked`/`--offline`) and rejects it.
boring run src/main.br -- --gguf test/fixtures/tiny.gguf --dump-metadata
```

**Real model**, any real command (`--dump-metadata`, `--tokens`, `--prompt`, `--quantized`, `--kv-cache`, ...): `boring run`'s interpreter OOMs on a file this size (a documented interpreter limitation), and the plain `boring build` (no `--target`) can't emit this project's GPU-kernel code at all, so a real GGUF always needs a compiled `--target metal` (or `--target wgpu`, also fully verified) build first:

```sh
boring build --target metal src/main.br
cd src/main_metal && cargo build --release
./target/release/main --gguf ../../weights/qwen2.5-0.5b-instruct-q8_0.gguf --dump-metadata
```

Model weights aren't tracked by git — download them from Hugging Face, e.g.:

```sh
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q8_0.gguf --local-dir weights
```

Then, for real chat generation:

```sh
./target/release/main --gguf ../../weights/qwen2.5-0.5b-instruct-q8_0.gguf --quantized --kv-cache \
    --prompt "What is the capital of France?" --max-tokens 30
```

Or for an interactive, multi-turn conversation (llama.cpp's own `-cnv` mode) — type a message and press enter, an empty line or Ctrl-D quits:

```sh
./target/release/main --gguf ../../weights/qwen2.5-0.5b-instruct-q8_0.gguf --quantized --kv-cache --chat \
    --max-tokens 60 --repeat-penalty 1.1
```

## Status

The core pipeline — GGUF parsing, tokenizer, forward pass, sampling, generation — is complete and verified end-to-end against `llama-cpp-python`/`llama-cli` on real Qwen2.5-0.5B-Instruct weights, across every quantization format listed above. GPU-accelerated inference is verified correct and bit-identical to the CPU reference on both the Metal and wgpu backends; CUDA and ROCm share the same source but remain untested (no hardware available). See [`docs/development-log.md`](docs/development-log.md) for the full phase-by-phase history, including the dozen real Boring compiler bugs found and fixed along the way.

**Not yet done**: CUDA/ROCm hardware verification (no hardware available); real-world end-to-end verification of the fused Q2_K GPU kernel specifically (no real downloaded GGUF for this model contains an actual Q2_K tensor — see the quantization-format entries in `docs/development-log.md` — verified instead via a from-scratch independent Python oracle plus bit-for-bit cross-checking against the CPU reference dequantizer on real Metal hardware). Everything else is functionally done — every ggml quantization format this project's usual small-model source actually produces (including IQ4_NL, the non-linear-codebook format llama.cpp's own heuristics substitute for Q2_K/Q3_K at small model sizes) is implemented, fused into the GPU path, and verified against real downloaded weights. Real end-to-end `--prompt`/`--chat` generation is verified on both GGUF families this project targets: Qwen2.5 (byte-for-byte identical prompt tokens against real `llama-cpp-python`, across every code path) and TinyLlama-1.1B-Chat (`architecture = "llama"`, SentencePiece tokenizer, its own non-ChatML Jinja template, GQA 32/4 heads) — single-shot `--prompt` and multi-turn `--chat` with and without `--kv-cache` all produce correct, coherent output on real downloaded weights for both.

**Perf opportunity investigated, doesn't pan out**: the fp32 warp-shuffle GEMM kernel (`math_gpu.br`'s `linear_warp_gpu`) beats the tiled GEMM by ~1.4-1.8x at the single-token decode-step shape (`seq=1`, the dominant real workload once `--kv-cache` is in use) — but real generation runs on quantized weights, and the fused-dequant Q8_0 counterpart built to test whether this carries over (`q8_linear_warp_gpu`) turned out SLOWER than the tiled `q8_linear_gpu` at every shape, including seq=1 (by ~10-12%, and ~6.6x slower at the largest prefill shape) — the extra per-lane byte-granular dequant work outweighs the barrier-removal win that worked in fp32. Not adopted; `model.br`'s real decode path is unaffected. See `docs/development-log.md` for the full benchmark matrix.

## Contributing

Contributions are welcome.

Before your first Pull Request can be merged, you must sign the [Contributor License Agreement](CLA.md) by adding your name to [`CLA-signatories.md`](CLA-signatories.md) as part of your PR.

## License

Copyright (C) 2026 Mickaël LANOË

This program is free software: you can redistribute it and/or modify it under the terms of the
[GNU General Public License v3.0](LICENSE) as published by the Free Software Foundation.

This program is distributed in the hope that it will be useful, but **without any warranty**.
See the LICENSE file for the full terms.

---

*Mickaël LANOË*
