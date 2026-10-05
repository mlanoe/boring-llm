# boring-llm

A real, GPU-accelerated LLM inference engine written in [Boring](https://github.com/mlanoe/boring), loading and running actual quantized [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) model files — the format used by [llama.cpp](https://github.com/ggerganov/llama.cpp) and the wider `ggml` ecosystem — with no conversion step.

## Goal

**Be a small, fast, portable LLM inference engine in the same spirit as llama.cpp** — load a `.gguf` file, run it directly on the GPU, get text out — but written entirely in Boring instead of C++, and used as a real-world proving ground for the language: a decoder-only transformer, real quantization formats, and Boring's own GPU kernel qualifiers (`'unified`/`'actor`/`'global`) driving genuine, not simulated, GPU compute across multiple backends.

## Features

- **Real GGUF loading, no conversion step.** Parses the format directly — header, every KV-metadata value kind (including the embedded tokenizer's vocab/merges as string arrays), tensor-info list — and reads the same files llama.cpp does.
- **Architecture-generic model/weight loading**: every config value is read from GGUF's own per-architecture metadata keys (not hardcoded to one model), and both RoPE conventions the Llama/Mistral/Qwen2 GGUF family needs (NEOX "rotate-half" vs. the original NORM "interleaved" style) and Qwen2's attention Q/K/V bias tensors (absent in Llama/Mistral) are handled generically. Verified against real Qwen2.5-0.5B-Instruct *and* real TinyLlama-1.1B-Chat (`llama` architecture) weights.
- **Multiple quantization formats**: F32, F16, Q4_0, Q5_0, Q8_0, Q2_K, Q3_K, Q4_K, Q6_K, IQ4_NL, and real-world files that mix several of these across different tensors (e.g. a typical "Q4_K_M" model, or a "Q2_K"-named preset that actually turns out to be an IQ4_NL/Q3_K/Q5_0/Q8_0/F32 mix at this model's size). Every format is ported bit-for-bit from ggml's own reference dequantization source and verified against real downloaded GGUF weights.
- **Portable quantized tensor kernels**: `gpu.tensor.linear` consumes packed Q4_0, Q5_0, Q8_0, Q2_K, Q3_K, Q4_K, Q6_K, and IQ4_NL weights directly, so the full model never needs a separate dequantized copy in memory. All quantized inference paths use the same tensor API, including full-sequence forward, batched KV-cache prefill, cache append, and single-token decode. The earlier hand-written fused kernels remain as correctness and performance baselines. The tensor path is verified against real downloaded weights end-to-end on Metal and WGPU, including Q4_K_M and IQ4_NL/Q3_K/Q5_0/Q8_0 mixed files.
- **Two independently GPU-hardware-verified backends**: Metal and wgpu (cross-platform: Vulkan/DirectX12/Metal), producing bit-identical real-model output on both, across every quantization format and every fused/warp-shuffle kernel in `math_gpu.br`, no exceptions — re-swept on wgpu whenever new kernel code lands, not just once. CUDA and ROCm share the same qualified source but are untested on this machine (no hardware available).
- **GGUF-embedded tokenizer, both families**: byte-level BPE (Qwen2's own `tokenizer.ggml.model = "gpt2"`) and SentencePiece BPE (Llama/Mistral's `"llama"`, auto-detected), built directly from the file's own `tokenizer.ggml.*` metadata — no separate `vocab.json`/`merges.txt`/`tokenizer.model` export step for either.
- **Any architecture's own GGUF-embedded chat template, actually wired into `--prompt`/`--chat`.** A real Jinja2-subset interpreter (`src/jinja.br`) — `{% if/elif/else/for/set %}`, `{{ expr }}` output, `and`/`or`/`not`, attribute/index access, `is defined`, the `trim`/`tojson` filters, and full `{%-`/`-%}` whitespace-trim handling, verified byte-for-byte against real `jinja2` on both Qwen2.5's own tool-calling-capable template and TinyLlama's simpler one — renders each turn's conversation into prompt text, which `tokenizer.br`'s `render_chat_prompt`/`encode_with_specials` then tokenizes, recognizing the template's own literal special-token strings (`<|im_start|>`, `<s>`, ...) atomically instead of running them through ordinary BPE/SentencePiece merging (llama.cpp's own `tokenizer_st_partition` concept). Replaces the project's earlier Qwen2.5-only hardcoded ChatML construction entirely — verified byte-for-byte identical to it against real `llama-cpp-python` output (`test/test_tokenizer_real.br`), including the KV-cache multi-turn case, where each turn's incremental tokens are now derived generically (`full_ids[all_tokens.length..]`, re-rendering the whole conversation and slicing off what the cache already has) instead of a per-architecture hardcoded shape.
- **KV-cache incremental decoding**: O(1) GEMM work per new token per layer instead of re-forwarding the whole sequence every step.
- **OpenAI-compatible HTTP server** (`--serve`): `/v1/models` + `/v1/chat/completions` (JSON and SSE), tool calling through the GGUF's own chat template, written in Boring over plain `std::net`. Its request rendering is verified token-for-token against `jinja2` + `llama.cpp` (`tools/oracle_render_request.py`), including multi-turn tool calls and non-ASCII text.
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
│   │                     # RopeApply/SwigluCombine, GQA attention, quantized tensor GEMM
│   ├── sampling.br       # xorshift64 RNG, greedy/temperature/top-k/top-p sampling
│   ├── json.br           # JSON parser into jinja.br's JValue (ordered keys, floats)
│   ├── openai.br         # OpenAI-API plumbing: message normalization, <tool_call> parsing,
│   │                     # JSON/SSE response bodies (pure, interpreter-testable)
│   ├── server.br         # `--serve`: OpenAI-compatible HTTP server over std::net
│   └── main.br           # CLI entry point
├── tools/
│   ├── inspect_gguf.py      # Python `gguf`-package cross-check oracle (metadata dump)
│   ├── make_test_gguf.py    # synthetic small-fixture GGUF file generator
│   ├── reference_logits.py  # llama-cpp-python logit oracle for math-correctness checks
│   ├── oracle_encode.py          # llama.cpp tokenization oracle for `--encode-file`
│   ├── oracle_render_request.py  # jinja2 + llama.cpp oracle for `--render-request`
│   └── smoke_server.py           # stdlib-only wire-format smoke test for `--serve`
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

### OpenAI-compatible server (`--serve`)

`--serve` loads the model once (packed quantized weights, GPU) and exposes it as an OpenAI-compatible HTTP API on `127.0.0.1`: `GET /v1/models` and `POST /v1/chat/completions` (JSON, or SSE with `"stream": true`, including `stream_options.include_usage`). The conversation goes through the GGUF's own chat template, so `tools`, `role: "tool"` results and assistant `tool_calls` all work for models whose template supports them (e.g. Qwen2.5); Hermes-style `<tool_call>{...}</tool_call>` output is parsed back into structured `tool_calls` with `finish_reason: "tool_calls"`. One request is served at a time (a single GPU needs that anyway).

```sh
./target/release/main --gguf ../../weights/qwen2.5-0.5b-instruct-q8_0.gguf --serve --port 8080 \
    --ctx-size 4096 --max-tokens 512   # --max-tokens = default when a request sets none
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":32}'
../../tools/smoke_server.py http://127.0.0.1:8080   # wire-format checks
```

A client config for [OpenCode](https://opencode.ai) (`opencode.json`) would look like the following — **written from OpenCode's documented config shape, not yet run against OpenCode itself**:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "boring-llm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "boring-llm (local)",
      "options": { "baseURL": "http://127.0.0.1:8080/v1" },
      "models": {
        "qwen2.5-0.5b-instruct": { "name": "Qwen2.5 0.5B Instruct", "tool_call": true,
                                    "limit": { "context": 4096, "output": 1024 } }
      }
    }
  }
}
```

Known limits today (see `docs/development-log.md`): the whole reply is generated before it is sent (the SSE framing is real, the streaming is not incremental yet), every request re-prefills its whole prompt (no cross-request KV-cache reuse), and a prompt longer than `--ctx-size` is rejected with `context_length_exceeded` — an agent like OpenCode sends prompts of many thousands of tokens, which needs a larger context plus chunked prefill (the attention kernel materializes a `seq x seq` score matrix per head). A small model (0.5B) answers tool requests but is not a reliable agent.

## Status

The core pipeline — GGUF parsing, tokenizer, forward pass, sampling, generation — is complete and verified end-to-end against `llama-cpp-python`/`llama-cli` on real Qwen2.5-0.5B-Instruct weights, across every quantization format listed above. GPU-accelerated inference is verified correct and bit-identical to the CPU reference on both the Metal and wgpu backends; CUDA and ROCm share the same source but remain untested (no hardware available). See [`docs/development-log.md`](docs/development-log.md) for the full phase-by-phase history, including the dozen real Boring compiler bugs found and fixed along the way.

**Not yet done**: CUDA/ROCm hardware verification (no hardware available). Everything else is functionally done — every ggml quantization format this project targets (including IQ4_NL and Q2_K, both of which needed a larger real model to find genuine real-world tensors of that type in, since small Qwen2.5-0.5B files never get them from llama.cpp's own per-tensor quantization heuristics) is implemented in the packed-weight tensor path and verified against real downloaded weights on both the Qwen2.5 and Llama GGUF families. Real end-to-end `--prompt`/`--chat` generation is verified on both GGUF families this project targets: Qwen2.5 (byte-for-byte identical prompt tokens against real `llama-cpp-python`, across every code path) and TinyLlama-1.1B-Chat (`architecture = "llama"`, SentencePiece tokenizer, its own non-ChatML Jinja template, GQA 32/4 heads) — single-shot `--prompt` and multi-turn `--chat` with and without `--kv-cache` all produce correct, coherent output on real downloaded weights for both. (A real Llama-2-7B-Chat GGUF, used specifically for Q2_K real-weight verification, has no embedded chat template to drive `--prompt`/`--chat` with — verified instead via `--tokens` against real token sequences and real `llama-cpp-python` top-token predictions, matching this project's original Phase 1 verification bar.)

**Perf opportunity investigated — retained as a baseline.** The fp32 warp-shuffle GEMM kernel (`math_gpu.br`'s `linear_warp_gpu`) beats the tiled GEMM by ~1.4-1.8x at the single-token decode-step shape (`seq=1`, the dominant real workload once `--kv-cache` is in use) — but the first fused-dequant Q8_0 counterpart (`q8_linear_warp_gpu`) was slower than the tiled `q8_linear_gpu`. A follow-up (`q8_linear_warp_broadcast_gpu`) broadcasts the shared scale header with `gpu.warp.shuffle` and wins by ~20-33% at `seq=1`; equivalent Q5_0, Q4_0, and IQ4_NL variants win by roughly 30-40%. The legacy `fused_linear_gpu` dispatcher still selects these variants by shape and remains available for benchmarking. Production inference uses `gpu.tensor.linear` for single-row decode (`seq == 1`), but **prefill (`seq > 1`) is routed to the hand-written tiled kernels behind `fused_linear_gpu`**: the compiler's tensor schedule only has a scalar prefill (one thread per output cell, no weight reuse across rows), which measured ~0.5 s of FFN matmul per layer at 28 tokens on a 0.5B model (~1-2 GFLOP/s) — a 38-token prefill took ~32-36 s, ~3 s with the tiled kernels. Output is unchanged (same greedy text on every checked prompt, same top-5 logits to ~5 digits); the split goes away once `boring` grows a tiled tensor schedule that wins at prefill shapes. Decode went from ~0.9 to ~0.40 s/token (Qwen2.5-0.5B Q8_0) after `boring` stopped re-creating the Metal device on every dispatch and the model weights became **GPU-resident** (uploaded once at load, `upload_bytes_gpu` in `math_gpu.br`, instead of cloned and re-uploaded per matmul per token). What remains is mostly GPU->host synchronization: each layer still reads K/V back and re-uploads the growing cache on the host, so the next step is a GPU-resident KV cache. See `docs/development-log.md` for the numbers.

Tensor scheduling is selected in `boring.toml`. The project requests the Q8_0
warp-broadcast schedule for single-row decode; the `[tensor.linear.prefill]`
setting is now only the fallback, since `math_gpu.br`'s `tensor_linear_gpu`
sends `seq > 1` to the tiled kernels itself. Changing this policy requires a new
`boring build --target ...` followed by `cargo build`; it does not require
rebuilding or reinstalling the Boring compiler.

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
