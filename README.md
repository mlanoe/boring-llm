# boring-llm

A local LLM inference engine written entirely in [Boring](https://github.com/mlanoe/boring): it loads quantized [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) model files (the llama.cpp format, no conversion step), runs them on the GPU (Metal or wgpu), and serves them through an **OpenAI-compatible HTTP API** that coding agents such as [OpenCode](https://opencode.ai) can use.

```sh
main --gguf qwen2.5-3b-instruct-q4_k_m.gguf --serve --port 8080 --ctx-size 16384 --prompt-cache ~/.cache/boring-llm
```

Full documentation: [the book](docs/book.md) ([HTML](docs/book.html)): command line, HTTP API, caches, architecture, verification and a development guide. This README is the short version.

## What it does

- **Runs real models.** Qwen2.5 (0.5B, 3B, 7B tested), Qwen3 (4B tested) and Llama-architecture models (TinyLlama 1.1B tested), in F32, F16, Q4_0, Q5_0, Q8_0, Q2_K, Q3_K, Q4_K, Q6_K and IQ4_NL, including the mixed-format files llama.cpp produces (`*-Q4_K_M.gguf`). Weights stay quantized on the GPU; there is no dequantized copy.
- **Needs little memory.** Tensors are read from disk one at a time, then live on the GPU: a 3B Q4_K_M server starts in ~5 s and holds ~3.1 GB.
- **Serves the OpenAI API.** `GET /v1/models` and `POST /v1/chat/completions`, JSON or streamed (SSE, token by token, with `stream_options.include_usage`), `stop` strings, `max_tokens`, sampling parameters, `top_k`, `repeat_penalty`, `frequency_penalty` and `presence_penalty`, and **tool calling** through the model's own chat template (Hermes-style `<tool_call>` output is parsed back into `tool_calls`; `tool_choice: "none"` leaves the tools out). Reasoning models (Qwen3) have their `<think>` text returned as `reasoning_content`, streamed too, and `chat_template_kwargs` (e.g. `{"enable_thinking": false}`) reaches the chat template.
- **Is fast enough for agents.** An agent resends its whole history every turn, so the server keeps the KV caches of its recent requests in memory (`--cache-slots`) and, with `--prompt-cache DIR`, on disk across restarts: only the tokens after the shared prefix are processed.
- **Has two GPU backends**, Metal and wgpu (Vulkan, DirectX 12, Metal), producing the same output. CUDA and ROCm share the source but have never been run (no hardware).
- **Is checked against llama.cpp.** Tokenization and chat-template rendering are compared token for token with `llama-cpp-python`/`jinja2`, and greedy generation matches llama.cpp's on real weights (identical text on Qwen2.5 0.5B Q8_0 and Q4_K_M and on 7B Q4_K_M for the prompts checked; the 3B was checked for consistency across backends and compilers, not against llama.cpp).

## Quick start

Requirements: the [Rust toolchain](https://rustup.rs/), [Boring](https://github.com/mlanoe/boring) (`cargo install --git https://github.com/mlanoe/boring`), and a GPU supported by Metal or wgpu. Use a `boring` built from `51d7260e` or later. Earlier builds cannot select launch batching only during decode (before `51d7260e`), leave about 1.8 ms/token on the table for the large Q8_0 decode projections (before `a08df631`), decode Q8_0 weights ~2x slower on Metal (before `1faf71e2`), do not compile the disk prompt cache or the native Metal K-quant kernels on macOS 27 (before `a057dd8c`), decode Q4_K weights ~2x slower (before `6fb92947`), prefill ~3x slower on Metal, stage the prefill tiles in half precision (`6acc7d44` to `316c7695`: logits of the 7B drift) or, between `316c7695` and `ff79f091`, return wrong values for the biased Q8_0 projections of small Qwen models, or, between `8bdac11d` and `d38813c8`, return wrong values for every odd output column of Q6_K matrices (Q4_K_M models then produce garbage), or, before `46c2473f`, fail to transpile a `.split()` used as a call argument.

```sh
# 1. build (Metal; use --target wgpu on other platforms)
boring build --target metal src/main.br
(cd src/main_metal && cargo build --release)
BIN=src/main_metal/target/release/main

# 2. get a model
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_k_m.gguf --local-dir weights
# (or: tools/fetch_models.py ensure qwen2.5-3b-instruct-q4_k_m   -- the models the tests and tools use,
#  downloaded on demand into weights/, resumable and checked against Hugging Face's SHA-256)

# 3. chat from the terminal
$BIN --gguf weights/qwen2.5-3b-instruct-q4_k_m.gguf --quantized --kv-cache --chat

# or serve it
$BIN --gguf weights/qwen2.5-3b-instruct-q4_k_m.gguf --serve --port 8080 --ctx-size 16384
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":32}'
```

A real model always needs the compiled build: the interpreter (`boring run`) cannot hold a file this size and a plain `boring build` cannot emit GPU kernels. Without a build, `boring run src/main.br -- --gguf test/fixtures/tiny.gguf --dump-metadata` exercises the parser on a tiny synthetic file.

## Using it with OpenCode

Point an OpenAI-compatible provider at the server (`opencode.json`). Start the server with `--ctx-size 16384` at least: OpenCode's system prompt plus its ten tool definitions is ~7.6k tokens.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "boring-llm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "boring-llm (local)",
      "options": { "baseURL": "http://127.0.0.1:8080/v1" },
      "models": {
        "Qwen2.5 3B Instruct": { "name": "Qwen2.5 3B Instruct", "tool_call": true,
                                  "limit": { "context": 16384, "output": 1024 } }
      }
    }
  }
}
```

The model id is whatever `GET /v1/models` returns (the GGUF's `general.name`, or `--model-name`). `tools/opencode_trial.sh <main-binary> <model.gguf> "<prompt>"` runs a real OpenCode against a fresh server with its own configuration, data and cache in a temporary directory (yours is never touched), a throwaway project and OpenCode's default permissions.

What to expect, from the trials run so far (OpenCode 1.18.32, five small tasks in a throwaway project):

| Model | Result |
|---|---|
| Qwen2.5-0.5B Q8_0 | reads a file and explains it; other tasks fail (invented paths, loops calling the `task` tool) |
| Qwen2.5-3B Instruct Q4_K_M | read a file, read the README, list the directory and create a file: correct; an edit went to a wrong new file |
| Qwen2.5-7B Instruct Q4_K_M | the same four correct; the edit changed the function signature but not its body, and was reported as finished |

The server never failed on the plumbing. The limit is the model: small models are not reliable agents.

The first request of a session is the slow one, because the ~7.5k-token prompt has to be processed once (about 2 minutes on a 7B model: 115 s of prefill measured through OpenCode on a 16 GB Apple-silicon machine). OpenCode gives up on a response that is silent for 5 minutes, so the server opens the stream immediately and sends an SSE comment after every 512 prompt tokens. Later turns reuse ~99% of the prompt from the cache (a few seconds), and with `--prompt-cache` the first request after a restart does too, as long as the project directory is the same (OpenCode writes its path into the system prompt).

## Server options

| Option | Default | |
|---|---|---|
| `--gguf PATH` | | model file (required) |
| `--serve`, `--port N` | `8080` | HTTP server on `127.0.0.1` |
| `--ctx-size N` | `4096` | prompt + reply tokens a request may use; a longer prompt gets `context_length_exceeded` |
| `--max-tokens N` | `40` | reply length when a request sets none |
| `--model-name NAME` | GGUF `general.name` | id returned by `/v1/models` |
| `--cache-slots N` | `4` | KV caches kept in memory (LRU) |
| `--prompt-cache DIR` | off | also keep KV caches on disk, reused after a restart |
| `--prompt-cache-files N` | `2` | snapshots kept on disk |
| `--prompt-cache-max-mb N` | `4096` | cap on all snapshots together (`0` = none) |
| `--kquant-prefill handwritten` | `tensor` | wgpu/CUDA/ROCm: keep the hand-written Q4_K/Q6_K prefill kernels (faster there today) |
| `--temperature`, `--top-k`, `--top-p`, `--seed`, `--repeat-penalty`, `--repeat-last-n` | greedy | sampling defaults; a request can override temperature, top_p, top_k, seed and repeat_penalty, and add frequency_penalty and presence_penalty |

One request is served at a time (one GPU). Every response is `Connection: close`.

**Prompt cache on disk.** After a request, the cache it left is written to `DIR/prompt-cache-<n>.bin` when it adds at least 1024 tokens to what the disk already holds; the write happens after the response has been sent. A later request whose prompt starts like a snapshot reads those positions back (`N reused from disk` in the log) instead of recomputing them. A snapshot replaces the one it extends, else takes a free file, else the oldest; the oldest are also dropped to stay under `--prompt-cache-max-mb`. A snapshot is used only by the exact model that wrote it, a truncated or half-written file is ignored, and floats are stored as the GPU holds them, so a restored cache gives the same answers as a cold one. The files contain the prompts' token ids and activations: keep the directory private. On the 0.5B model a 6.4k-token prompt takes ~13 s cold and 0.5-1.5 s restored. On the 7B model, OpenCode's first request (7.6k tokens) spent 321 s in prefill cold and 1.9 s restored from an ~890 MB snapshot (7,636 tokens read back), after a server restart.

## Performance

Measured on one 16 GB Apple-silicon machine (Apple M3, Metal). The 0.5B rows use `boring` 8ed96591; the other rows are the earlier clean run with 46c2473f. Timings on it still move by tens of percent from one repetition to the next (min-max is in `docs/development-log.md`), so read the ratios as indications, not as constants.

Against llama.cpp (llama-cpp-python, Metal, same GGUF, same machine, 3 alternating runs, medians, `tools/bench_vs_llamacpp.py`), tokens per second:

| Model | Prompt | boring-llm prefill | llama.cpp prefill | | boring-llm decode | llama.cpp decode | |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B Q4_K_M | ~570 | 114 | 125 | 0.91x | 7.7 | 13.0 | 0.59x |
| Qwen2.5-7B Q4_K_M | ~4150 | 87 | 122 | 0.71x | 6.1 | 10.0 | 0.61x |
| Qwen3-4B Q4_K_M | ~550 | 181 | 277 | 0.66x | 10.8 | 28.3 | 0.38x |
| Qwen3-4B Q4_K_M | ~4130 | 130 | 221 | 0.59x | 9.0 | 15.6 | 0.58x |
| Qwen2.5-3B Q4_K_M | ~570 | 222 | 314 | 0.71x | 11.7 | 28.8 | 0.41x |
| Qwen2.5-3B Q4_K_M | ~4150 | 158 | 271 | 0.58x | 10.3 | 20.9 | 0.49x |
| Qwen2.5-0.5B Q8_0 | ~630 | 1,874 | 2,938 | 0.64x | 52.2 | 124.3 | 0.42x |
| Qwen2.5-0.5B Q8_0 | ~4200 | 1,149 | 2,169 | 0.53x | 44.0 | 82.5 | 0.53x |

The server needs 3.1 GB (peak 3.3 GB) for the 3B model and 6.4 GB (peak 6.8 GB) for the 7B, and is ready in about 4 s on the 7B (disk-streaming loader). Through OpenCode, the 7B model's first request (7,539 tokens, ten tools) spent 115 s in prefill cold and 2.5 s on the follow-up that reused 7,559 of 7,638 tokens; a cold start used to take 321-453 s.

Where the gap to llama.cpp is: prefill is close on short prompts on the 7B and remains more variable at long context; a newer run no longer reproduced the earlier 0.91x-to-0.71x fall, and making llama.cpp's KV cache float32 did not explain the remaining gap. Decode is about 0.6x on the 7B. On the 0.5B, `boring` 1faf71e2's four-row Q8_0 Metal schedule nearly doubled short-context decode and distributing RMSNorm rows across SIMD groups raised it by another 44% (21.2 to 14.7 ms per token in the CLI). `boring` a08df631 then fully unrolls the four private rows for large Q8_0 projections: gate/up rise from 47.2 to 61.3 GB/s, down from 49.7 to 66.0 GB/s and the tied head from 59.8 to 71.3 GB/s, predicting about 1.8 ms/token saved. Launches are now batched by eight automatically during KV-cache decode, without applying the policy to prefill; an adjacent 200-token run fell from 5.07 to 4.49 seconds including model loading. Fusing the small attention projections is the clearest remaining short-context target. The detail is in `docs/development-log.md`. Earlier gains came from `boring`'s Q4_K single-token decode, the fragment-blocked quantized prefill schedule and float32 staging, plus this project's fused attention and two-buffer KV cache.

On wgpu the portable K-quant prefill is still ~3x slower than the hand-written kernels (hence `--kquant-prefill handwritten` there).

## Verifying it

```sh
for t in test/test_*.br; do case $t in *_real.br) ;; *) boring run $t;; esac; done   # interpreter tests, no model needed
tools/smoke_server.py http://127.0.0.1:8080                                         # wire-format checks against a running server
```

The `test/*_real.br` programs need a compiled build and real weights (each lists the files it expects). The models are not in the repository (`weights/` is git-ignored): `tools/fetch_models.py list` shows what the tests and tools use and what is present, `ensure NAME` downloads one, and `tools/accuracy_vs_llamacpp.py --model NAME --fetch` fetches a missing one first. `tools/` holds the oracles that compare against llama.cpp: `accuracy_vs_llamacpp.py` (the whole forward pass, position by position: run it on a Q8_0 and a Q4_K_M model after any change to the GPU kernels or to `boring`), `decode_vs_llamacpp.py` (generation through the KV-cache decode path, which the logits check never runs: run it with the accuracy check), `reference_logits.py` (logits), `oracle_encode.py` (tokenization), `oracle_render_request.py` (chat-template rendering), `inspect_gguf.py` (metadata).

## Repository layout

```
src/
  gguf.br, gguf_file.br   GGUF parser; reads tensors from disk on demand
  io.br, json.br, jinja.br  binary readers, JSON, the Jinja2 subset chat templates use
  tokenizer.br            byte-level BPE and SentencePiece, from the GGUF's own metadata
  quant.br                CPU reference dequantization for every supported format
  model.br                config and weight loading, forward pass, KV cache, prefill/decode
  math.br, math_gpu.br    CPU reference math; GPU kernels (attention, norms, RoPE, copies, quantized GEMM)
  sampling.br             greedy, temperature, top-k, top-p, repeat penalty
  openai.br               message normalization, tool-call parsing, JSON/SSE bodies (pure)
  cache_policy.br         what the disk prompt cache keeps and when it is used (pure)
  prompt_cache.br         reading and writing the disk snapshots
  server.br               the HTTP server
  main.br                 command line
tools/                    oracles, accuracy and speed comparisons with llama.cpp, model downloader, smoke test, OpenCode trial script, Metal prefill lab
test/                     interpreter tests (test_*.br) and compiled real-model tests (*_real.br)
docs/book.md              the book: using, operating, understanding and extending the engine
docs/build.py             generates the HTML versions of the documentation (python3 docs/build.py)
docs/development-log.md   how it was built, with the measurements and the bugs found on the way
boring.toml               kernel scheduling policy (native Metal paths for Q8_0, Q4_K, Q6_K)
```

## Limits

- Requests are served one at a time; there is no batching.
- Context is limited by `--ctx-size` and by memory (the KV cache is float32: ~115 KB per token for the 7B model).
- Only Llama-style GGUF layouts are supported (Qwen2 and Llama are the ones tested); no MoE, no vision.
- Chat templates use a Jinja2 subset (enough for Qwen2.5 and TinyLlama); an unsupported construct is reported, not ignored.
- Snapshots on disk and the disk-streaming loader use Unix file APIs; GGUF files split in several parts are not supported.
- CUDA and ROCm have not been run.

## Boring

Boring is a language that transpiles to Rust, with Python-like indentation and Swift-like optionals. This project is also its real-world test: every bug found in the compiler while building it was reported and fixed upstream (see `docs/development-log.md`). See the [Boring repository](https://github.com/mlanoe/boring) for the language reference.

## Contributing

Contributions are welcome. Before your first Pull Request can be merged, sign the [Contributor License Agreement](CLA.md) by adding your name to [`CLA-signatories.md`](CLA-signatories.md) as part of your PR.

## License

Copyright (C) 2026 Mickaël LANOË

This program is free software: you can redistribute it and/or modify it under the terms of the [GNU General Public License v3.0](LICENSE) as published by the Free Software Foundation. It is distributed in the hope that it will be useful, but **without any warranty**. See the LICENSE file for the full terms.
