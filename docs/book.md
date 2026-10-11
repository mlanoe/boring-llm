# The boring-llm Book

boring-llm is a local LLM inference engine written entirely in [Boring](https://github.com/mlanoe/boring), a language that transpiles to Rust. It reads quantized [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) model files (the llama.cpp format, with no conversion step), runs them on the GPU through Boring's own kernel language, and serves them over an OpenAI-compatible HTTP API that coding agents such as [OpenCode](https://opencode.ai) can use.

This book is the reference for using, operating, understanding and extending it. The [README](../README.md) is the short version; the [development log](development-log.md) is the history, with every measurement and every bug found along the way. Where this book gives a number, the development log says how it was measured.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Getting Started](#2-getting-started)
3. [Command-Line Reference](#3-command-line-reference)
4. [The HTTP Server](#4-the-http-server)
5. [Using It With Coding Agents](#5-using-it-with-coding-agents)
6. [Models and Formats](#6-models-and-formats)
7. [Caches](#7-caches)
8. [Performance](#8-performance)
9. [Architecture](#9-architecture)
10. [GPU Backends and Kernel Scheduling](#10-gpu-backends-and-kernel-scheduling)
11. [Verification](#11-verification)
12. [Development Guide](#12-development-guide)
13. [Limits and Known Issues](#13-limits-and-known-issues)
14. [Troubleshooting](#14-troubleshooting)
15. [Appendix: Reference Tables](#15-appendix-reference-tables)

---

## 1. Introduction

### What it is

- A **GGUF loader and inference engine**: tokenizer, chat-template renderer, transformer forward pass, KV caches, sampling. All of it is Boring source (about 9,000 lines under `src/`), including the GPU kernels.
- An **OpenAI-compatible server** (`--serve`) with streaming, stop strings and tool calling, tuned for agent clients that resend a long, mostly identical prompt every turn.
- A **real-world test of the Boring language and its GPU targets**: every compiler bug found while building it was reduced to a minimal case and reported upstream.

### What it is not

- Not a training or fine-tuning framework.
- Not a multi-user server: one GPU, one request at a time, no batching.
- Not a replacement for llama.cpp on speed. On the 7B model it runs at roughly half to two thirds of llama.cpp's speed on the same Apple-silicon machine (see [chapter 8](#8-performance)). Its value is that the whole stack is readable Boring code, checked against llama.cpp at every level.

### Support matrix

| | Metal | wgpu | CUDA | ROCm |
|---|---|---|---|---|
| Status | primary, all tests | all tests, same output as Metal | source shared, never run | source shared, never run |
| Quantized prefill | native Metal schedules | portable schedule, or hand-written K-quant kernels | not run | not run |

| Architecture | Tested models |
|---|---|
| `qwen2` | Qwen2.5 0.5B, 3B, 7B Instruct |
| `qwen3` | Qwen3 4B (thinking and non-thinking) |
| `llama` | TinyLlama 1.1B Chat |

Any GGUF of those two architectures should load. Other architectures are rejected with an explicit error rather than guessed (see [section 6.2](#62-architectures)).

### How the book is organized

Chapters 2 to 5 are for people who want to run it. Chapters 6 to 8 explain models, caches and speed. Chapters 9 to 12 are for people who want to read or change the code. Chapters 13 to 15 collect limits, fixes and tables.

---

## 2. Getting Started

### Requirements

- The [Rust toolchain](https://rustup.rs/).
- The `boring` compiler: `cargo install --git https://github.com/mlanoe/boring`. The README states the minimum commit; older builds miss kernel fixes or have known accuracy problems (see [section 10.4](#104-compiler-versions)).
- A GPU supported by Metal (macOS) or wgpu (Vulkan, DirectX 12, Metal).
- A GGUF model file, for example from Hugging Face.

### Build

A real model always needs the compiled build. The interpreter (`boring run`) cannot hold a file this size, and a plain `boring build` cannot emit GPU kernels, so a GPU target is required.

```sh
boring build --target metal src/main.br          # or: --target wgpu
(cd src/main_metal && cargo build --release)
BIN=src/main_metal/target/release/main
```

The first command transpiles `src/main.br` and every module it uses into a Cargo project (`src/main_metal/`, or `src/main_wgpu/`); the second compiles it. `boring.toml` at the repository root sets the kernel scheduling policy (see [section 10.2](#102-boringtoml)).

### Get a model

```sh
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_k_m.gguf --local-dir weights
tools/fetch_models.py list                        # or: the models the tests and tools use, with their sources
tools/fetch_models.py ensure qwen3-4b-q4_k_m      # download one into weights/ if it is missing (resumes, checks the SHA-256)
```

The `weights/` directory is git-ignored. Q4_K_M files are a good default: about 2 GB for a 3B model, 4.7 GB for a 7B.

### Chat in the terminal

```sh
$BIN --gguf weights/qwen2.5-3b-instruct-q4_k_m.gguf --quantized --kv-cache --chat
```

Type a message and press enter; an empty line or Ctrl-D ends the session. `--kv-cache` reuses the cache across turns, so each turn only processes the new tokens.

### Serve it

```sh
$BIN --gguf weights/qwen2.5-3b-instruct-q4_k_m.gguf --serve --port 8080 --ctx-size 16384 --prompt-cache ~/.cache/boring-llm
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":32}'
```

The server prints one line per request on its standard output and listens on `127.0.0.1` only (see [chapter 4](#4-the-http-server)).

### Try the parser without a GPU

```sh
boring run src/main.br -- --gguf test/fixtures/tiny.gguf --dump-metadata
```

This runs the GGUF parser in the interpreter on a tiny synthetic file. It is the only mode that works without a GPU build.

---

## 3. Command-Line Reference

The executable has one required flag, `--gguf`, and several modes selected by the other flags. Unknown flags are ignored.

### Modes

| Mode | Flags | What it does |
|---|---|---|
| Serve | `--serve` | OpenAI-compatible HTTP server; always uses the quantized GPU path |
| Chat | `--chat` | interactive conversation in the terminal |
| One prompt | `--prompt "..."` | one chat turn, prints the reply |
| Metadata | `--dump-metadata` | prints every GGUF metadata key and tensor, and whether embeddings are tied |
| Logits | `--tokens 1,2,3` | forward pass over given token ids, prints the top 5 logits per position |
| Generate from ids | `--generate-tokens 1,2,3 --max-tokens N` | greedy (or sampled) generation through the KV-cache decode path from token ids; prints `generated: id,id,...` |
| Encode | `--encode-file F` | encodes each string of a JSON array and prints the token ids |
| Render | `--render-request F` | renders a request body through the chat template and prints the prompt |

`--tokens` is the debugging entry point used by the accuracy tools: it bypasses the tokenizer, so a tokenizer bug cannot be confused with a transformer bug. Add `--quantized` to run the packed-weight GPU path (what the server uses) or `--gpu` for the dequantize-at-load float32 GPU path; without either it runs the CPU reference. `--tokens` runs one batched forward pass; it never runs the single-token decode path that generation uses, which is what `--generate-tokens` and `tools/decode_vs_llamacpp.py` are for.

### Common flags

| Flag | Default | Meaning |
|---|---|---|
| `--gguf PATH` | | model file (required) |
| `--quantized` | off | keep weights quantized on the GPU (always on for `--serve`) |
| `--kv-cache` | off | incremental decode with a KV cache (`--chat`, `--prompt`) |
| `--system TEXT` | `You are a helpful assistant.` | system message for `--chat` and `--prompt` |
| `--max-tokens N` | `40` | reply length; for `--serve` the default when a request sets none |

Without `--quantized`, `--chat` and `--prompt` dequantize every tensor to float32 at load time, which needs several times the file size in memory and is only sensible for small models.

### Sampling flags

| Flag | Default | Meaning |
|---|---|---|
| `--temperature T` | `0` (greedy) | softmax temperature; 0 or less means greedy argmax |
| `--top-k K` | `0` (off) | keep the K most likely tokens |
| `--top-p P` | `1.0` | nucleus sampling threshold |
| `--seed S` | `0` | PRNG seed (0 is replaced by a fixed constant) |
| `--repeat-penalty R` | `1.0` (off) | llama.cpp-style repeat penalty |
| `--repeat-last-n N` | `64` | window for the penalty; 0 or less means the whole history |

For `--serve`, these set the server defaults; a request may override `temperature`, `top_p`, `top_k`, `seed` and `repeat_penalty`, and add `frequency_penalty` and `presence_penalty` (see [chapter 4](#4-the-http-server)).

### Server flags

| Flag | Default | Meaning |
|---|---|---|
| `--port N` | `8080` | listening port |
| `--ctx-size N` | `4096` | prompt plus reply tokens one request may use |
| `--model-name NAME` | GGUF `general.name` | id returned by `/v1/models` |
| `--cache-slots N` | `4` | KV caches kept in memory (least recently used evicted) |
| `--prompt-cache DIR` | off | also keep KV caches on disk, reused after a restart |
| `--prompt-cache-files N` | `2` | snapshot files kept on disk |
| `--prompt-cache-max-mb N` | `4096` | size cap on all snapshots together; 0 means none |
| `--kquant-prefill MODE` | `tensor` | `tensor` or `handwritten`; wgpu, CUDA and ROCm only (see [chapter 10](#10-gpu-backends-and-kernel-scheduling)) |

### Examples

```sh
# which tensors and formats does this file hold?
$BIN --gguf weights/model.gguf --dump-metadata

# top-5 logits for a hand-picked token sequence, on the packed GPU path
$BIN --gguf weights/model.gguf --quantized --tokens 8420,374,264,1467

# creative sampling, 200 tokens
$BIN --gguf weights/model.gguf --quantized --kv-cache --prompt "Write a haiku about rust." \
     --max-tokens 200 --temperature 0.8 --top-k 40 --top-p 0.95 --seed 7

# print the exact prompt a request would produce, without running the model
$BIN --gguf weights/model.gguf --render-request test/fixtures/requests/tools.json
```

---

## 4. The HTTP Server

### Overview

`--serve` loads the model once, keeps it resident on the GPU and runs one blocking accept loop with plain `std::net`, with no extra crate and no thread pool. One request is processed at a time, which is exactly the serialization a single GPU needs. Every response carries `Connection: close`. CORS headers allow any origin.

The server listens on `127.0.0.1` only and has no authentication. To reach it from another machine use an SSH tunnel or a port forward; do not expose the port.

### Endpoints

| Method and path | Purpose |
|---|---|
| `GET /v1/models` (also `/models`) | one model, its id is the GGUF `general.name` or `--model-name` |
| `POST /v1/chat/completions` (also `/chat/completions`) | chat completion, JSON or SSE |
| `GET /health` (also `/`) | `{"status": "ok"}` |
| `OPTIONS` anything | `204`, for browser preflight |

Any other route returns `404` with an OpenAI-shaped error body.

### Chat completion request

| Field | Behavior |
|---|---|
| `messages` | required, non-empty; roles `system`, `user`, `assistant`, `tool`; `content` may be a string or an array of typed parts (the text parts are concatenated) |
| `tools` | handed to the model's own chat template as its `tools` variable |
| `chat_template_kwargs` | an object whose keys become extra variables of the chat template; Qwen3's template reads `enable_thinking` (`false` starts the reply with an empty thinking block, so the model answers directly) |
| `max_tokens`, `max_completion_tokens` | reply length; the smaller of this and the room left in the context |
| `temperature`, `top_p`, `top_k`, `seed` | override the server defaults |
| `repeat_penalty` | llama.cpp-style repeat penalty over the prompt and the reply so far (an extension of the OpenAI API) |
| `frequency_penalty`, `presence_penalty` | OpenAI-style penalties over the reply so far: the logit of a token drops by its count times the frequency penalty plus the presence penalty (default 0, may be negative) |
| `tool_choice` | `"none"` leaves the tool definitions out of the prompt and never returns tool calls; any other value is accepted but not enforced (the model decides, as with `"auto"`) |
| `stop` | a string or an array of strings; the reply is cut at the earliest match |
| `stream` | `true` for server-sent events |
| `stream_options.include_usage` | adds a final usage chunk to a stream |

Other fields (`n`, `logprobs`, `response_format`, ...) are ignored. A `null` or missing `content` (an assistant message that only carries tool calls) is read as an empty string, because templates do string operations on it. An assistant message's `tool_calls[i].function.arguments` may be a JSON string (as OpenAI sends it) or an object; it is normalized for the template.

### Responses

A non-streaming reply is a normal `chat.completion` object with `usage` filled in. When the model reasons (`<think>...</think>`, Qwen3's default), the reasoning is returned apart as `message.reasoning_content` (as llama.cpp and DeepSeek's API do) and `content` holds only the answer; the markers never appear in either. `finish_reason` is `tool_calls` when the reply contains tool calls, `stop` when the model ended or a stop string matched, and `length` when `max_tokens` or the context ran out.

### Context overflow

The server never drops part of a conversation to make it fit. A prompt that fills the whole context (`--ctx-size`) is refused with the 400 `context_length_exceeded` error below, and a reply that reaches the end of the context stops there with `finish_reason` `length`. llama.cpp behaves the same way when its context shift is off. Shifting the context would mean discarding old positions and re-rotating the cached keys (or recomputing the cache), and the in-memory and on-disk caches rely on exact token prefixes. Agent clients such as OpenCode shorten the history themselves when they see this error.

### Streaming

With `"stream": true` the reply is sent as server-sent events, token by token as the model produces them:

1. the head and a role chunk are sent **immediately**, before the prompt is processed;
2. while a long prompt is processed, an SSE comment line `: prefill N/M` follows every 512 prompt tokens, so the connection is never silent;
3. deltas, one per decoded piece of text (a character spanning several byte-level tokens is held back until it is complete): `reasoning_content` deltas while the model is inside `<think>`, then `content` deltas;
4. tool calls, if any, as one chunk;
5. the finish chunk, an optional usage chunk, and `data: [DONE]`.

Steps 1 and 2 matter for agent clients: OpenCode gives up on a response whose headers or body stay silent for five minutes, and the first request of a session on a 7B model can spend over two minutes in prefill. A client that disconnects mid-prefill is noticed by the failing write, and the prefill stops.

### Tool calling

The tool definitions are rendered by the model's own chat template, so each model sees the format it was trained on. For Qwen2.5 and many other instruct models that format is Hermes-style:

```
<tool_call>
{"name": "read", "arguments": {"filePath": "src/main.br"}}
</tool_call>
```

The server parses these blocks back into OpenAI `tool_calls` (id, type `function`, name, arguments as a JSON string). A block that is not valid JSON with a `name` stays in `content` verbatim rather than being dropped.

Small models, especially under a long prompt, often get the closing braces wrong, because Qwen2.5's template shows its own example with doubled braces. When a strict parse fails, the parser takes the balanced object that starts at each of the first three `{` and keeps the first that is an object with a `name`. A legitimate nested `}}` inside an object is never touched, and trailing junk is ignored.

### Errors

Errors use the OpenAI shape `{"error": {"message", "type", "code"}}`.

| Status | Code | Cause |
|---|---|---|
| 400 | `invalid_json` | the body is not valid JSON |
| 400 | `missing_messages` | `messages` absent or empty |
| 400 | `context_length_exceeded` | the prompt is at least as long as `--ctx-size`; the error also carries `n_prompt_tokens` and `n_ctx`, and its message contains the phrases OpenAI-compatible clients and llama.cpp use ("maximum context length", "exceeds the available context size") |
| 404 | `not_found` | unknown route |
| 500 | `internal_error` | a failure inside the request (also logged as `request N failed`) |

A failure after a stream has opened cannot change the status; an SSE `error` event followed by `[DONE]` ends the stream instead. In both cases the server keeps running.

### Log lines

The server writes to standard output:

| Line | Meaning |
|---|---|
| `[diag] ...` | startup progress: parsing, loading, model dimensions |
| `[serve] model '...' ready — listening on ...` | ready, with context size and slot count |
| `[serve] POST /v1/chat/completions` | a request arrived |
| `[serve] chat: N messages, T tools, prompt P tokens (R reused from cache or disk), generated G, finish=..., ...` | one finished chat request with timings |
| `[serve] prompt cache: ...` | disk snapshots found, saved, dropped or failed |

### Checking a server

`tools/smoke_server.py` checks the wire format against a running server: models, a plain completion, an SSE stream that really arrives token by token, the error shapes, and the JSON shape of a tool-calling round trip.

```sh
tools/smoke_server.py http://127.0.0.1:8080
```

---

## 5. Using It With Coding Agents

### OpenCode

Point an OpenAI-compatible provider at the server in `opencode.json`:

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

The model id is whatever `GET /v1/models` returns. Start the server with `--ctx-size 16384` at least: OpenCode's system prompt plus its ten tool definitions is about 7.6k tokens.

### Why the first request is slow, and the rest are not

An agent resends its whole history every turn. The first request of a session therefore has to process the ~7.5k-token prompt once (about two minutes on a 7B model). Later turns share almost all of it with the previous turn, and the server reuses about 99 percent of the prompt from a cache (a few seconds). With `--prompt-cache` the first request after a restart reuses it too, provided the project directory is the same, because OpenCode writes its path into the system prompt.

Agents also interleave small unrelated requests, such as title generation. Several in-memory slots (`--cache-slots`) exist so such a request does not evict the main conversation's cache.

### The trial script

`tools/opencode_trial.sh` runs a real OpenCode against a fresh server, with its own configuration, data and cache in a temporary directory (yours is never touched), a throwaway project and OpenCode's default permissions.

```sh
tools/opencode_trial.sh src/main_metal/target/release/main weights/qwen2.5-7b-instruct-q4_k_m.gguf "Read math_utils.py and explain it"
```

Environment variables: `OPENCODE_TRIAL_PROMPT_CACHE` (a cache directory to test the disk cache), `OPENCODE_TRIAL_WORK` (the work directory) and `OPENCODE_TRIAL_TIMEOUT`. The script is one braced block, so editing it while a trial runs cannot change what the running copy executes.

### What to expect from the model

The server never failed on the plumbing in the trials run so far. The limit is the model: small models are not reliable agents. Results from five small tasks in a throwaway project:

| Model | Result |
|---|---|
| Qwen2.5-0.5B Q8_0 | reads a file and explains it; the other tasks fail |
| Qwen2.5-3B Instruct Q4_K_M | read a file, read the README, list a directory, create a file: correct; an edit went to a wrong new file |
| Qwen2.5-7B Instruct Q4_K_M | the same four correct; the edit changed a function signature but not its body, and was reported as finished |

### Other clients

Anything that speaks the OpenAI chat-completions protocol works: the Vercel AI SDK, the OpenAI client libraries (set the base URL), `curl`. Clients that send `content` as an array of text parts are handled.

---

## 6. Models and Formats

### 6.1 Formats

Weights stay in their quantized form on the GPU and are decoded inside the matrix-multiply kernels; there is no dequantized copy. Supported tensor types:

| ggml type | Block | Bytes per block | Notes |
|---|---|---|---|
| F32, F16 | | | norms, biases, some embeddings |
| Q4_0 | 32 values | 18 | |
| Q5_0 | 32 values | 22 | |
| Q8_0 | 32 values | 34 | symmetric int8 with an f16 scale |
| Q2_K | 256 values | 84 | |
| Q3_K | 256 values | 110 | |
| Q4_K | 256 values | 144 | the bulk of `Q4_K_M` files |
| Q6_K | 256 values | 210 | `Q4_K_M` files keep some tensors in Q6_K |
| IQ4_NL | 32 values | 18 | |

Files produced by llama.cpp mix formats per tensor (`*-Q4_K_M.gguf` holds Q4_K and Q6_K tensors, F32 norms and biases); each tensor is dispatched on its own type. The K-quant formats need the row length (`k`) to be a multiple of 256.

### 6.2 Architectures

Every hyperparameter is read from the GGUF's own `{arch}.*` metadata keys, never hard-coded to one model. The head size is `{arch}.attention.key_length` when the file gives one (Qwen3 4B: 128, although 2560 / 32 heads would say 80) and `d_model / n_head` otherwise. Two things GGUF does not carry are hard-coded per architecture, mirroring llama.cpp's own table:

- **RoPE rotation convention**: `qwen2` and `qwen3` use the NEOX "rotate-half" pairing, `llama` (which also covers Mistral) the interleaved one. Any other architecture string is rejected with an error naming the llama.cpp function to check.
- **Attention biases**: Qwen2 has Q, K and V biases, Llama and Qwen3 do not; a missing bias tensor becomes an all-zero vector.
- **Query and key norms** (`qwen3`): an RMS norm over each head's query and key vectors, with its own `attn_q_norm` and `attn_k_norm` weights of one head's size, applied after the projection and before RoPE. It is switched on when the file has those tensors.

Grouped-query attention is handled with the usual head mapping, and tied embeddings (no separate `output.weight`) are detected from the tensor list. There is no mixture-of-experts and no vision support.

### 6.3 Tokenizers

Both tokenizer families are built directly from the metadata embedded in the GGUF; no separate vocabulary files are needed.

- **Byte-level BPE** (`tokenizer.ggml.model = "gpt2"`, Qwen2 and friends): a hand-written scanner for the Qwen2 pre-tokenization regular expression, then BPE merges by rank. Digits split one per token, punctuation absorbs the newlines after it, and a whitespace run gives its last character to the next token. It was checked token for token against llama.cpp on a tricky corpus (`tools/oracle_encode.py`).
- **SentencePiece BPE** (`"llama"`, Llama and Mistral): greedy highest-score-first bigram merging over the file's scores, with byte fallback through the `<0xXX>` tokens.
- **Special tokens** (control and user-defined types, for example `<|im_start|>` or `<tool_call>`) are matched atomically in the rendered text instead of going through ordinary merging, as llama.cpp does.

### 6.4 Chat templates

The model's own `tokenizer.chat_template` is rendered by a Jinja2 subset interpreter written for this project (`src/jinja.br`), then encoded with special tokens recognized. The subset covers what real templates need:

- `{% if %}`, `{% elif %}`, `{% else %}`, `{% for %}` with `loop.first`, `loop.last`, `loop.index`, `loop.index0`, `{% set %}`;
- `{{ expression }}` output, `and`, `or`, `not`, `==`, `!=`, string concatenation with `+`, attribute and subscript access, `is defined`;
- the `trim` and `tojson` filters, and the `{%-` / `-%}` whitespace-trim markers.

Qwen3's template adds `namespace(...)` objects with `{% set ns.attr = ... %}`, `range(...)`, the string methods `split`, `strip`, `lstrip`, `rstrip`, `startswith`, `endswith`, `replace` and `lower`, `in` and `not in`, the comparison operators, negative list indices, the tests `is string`, `is none`, `is false` and friends, the `length` filter, `raise_exception(...)`, and the Jinja rule that a `for` body has its own scope (a namespace is the way to carry state out of it). There are no macros, includes, or dict or list literals. An unsupported construct is reported as an error, not ignored. Rendering was compared token for token with `jinja2` on the Qwen2.5, Qwen3 and TinyLlama templates (`tools/oracle_render_request.py`, which also covers the multi-turn tool-call, reasoning and `enable_thinking` request fixtures).

### 6.5 Memory

- **Loading**: only the GGUF header (metadata and tensor table, about 7 MB for Qwen2.5) is parsed from memory. Each tensor is then read straight from the file when needed and uploaded, so the peak is about one tensor rather than the whole model. A 3B Q4_K_M server starts in about 5 seconds and holds about 3.1 GB; a 7B holds about 6.4 GB (peak 6.8 GB).
- **KV cache**: float32, `n_layer x 2 x n_kv_head x d_head x 4` bytes per token. That is about 115 KB per token for the 7B model, so a 16k context costs about 1.9 GB per cached conversation, and each in-memory slot holds its own cache.
- **Disk snapshots** hold the same data per token plus the token id (see [chapter 7](#7-caches)).

---

## 7. Caches

### 7.1 Why caches matter

Processing a prompt (prefill) is faster per token than generating (about 10 to 15 ms against 120 to 150 ms on the 7B model), but an agent's prompt is thousands of tokens: 7,500 of them take about two minutes. The next turn resends a prompt that is 99 percent identical, so every cache below exists to avoid recomputing the keys and values of a shared prefix.

### 7.2 In-memory slots

After each request the KV cache it left is kept in a slot together with the token ids it covers. A new request starts from the slot sharing the longest token prefix with its prompt. Placement rules:

1. a slot that is a prefix of the new conversation is extended in place;
2. a slot that already covers the new conversation is kept as it is;
3. a diverging prompt gets its own slot (the old branch may come back), evicting the least recently used when all `--cache-slots` are taken.

The last prompt token always goes through the model, because its logits pick the first reply token, so at most `length - 1` cached positions are reusable.

### 7.3 The KV cache layout

Each layer's cache is two GPU buffers: a **main** buffer holding the oldest rows and a short **tail** of at most 63 rows. A decode step appends its new key and value row to the tail (a few dozen rows copied at most), and the tail is folded into the main buffer every 64 tokens, so the full copy is amortized over 64 steps. Before this split, every decode token copied the entire cache for every layer; at 7.6k tokens of context the tail alone was worth about 21 percent of decode speed.

Decode attention is a split-K kernel that reads both buffers. Prefill and disk snapshots fold the tail first and work on one buffer. Truncating a cache to its first N positions (the reuse case) only sets a position counter, because the buffers are only read up to it.

### 7.4 The disk prompt cache

`--prompt-cache DIR` writes the cache a request leaves to `DIR/prompt-cache-<n>.bin`, and a later request whose prompt starts like a snapshot reads those positions back instead of recomputing them. On the 7B model, OpenCode's first request after a restart went from 321 s of prefill to 1.9 s (7,636 positions read back from an ~890 MB snapshot).

When it saves: after a request has finished and its response has been sent, a snapshot is written if it adds at least **1,024** tokens to what the disk already holds. When it restores: only when it covers at least **256** more usable tokens of the prompt than the best in-memory slot.

Which file it writes to: the snapshot replaces the one it extends, else takes a free file (up to `--prompt-cache-files`), else replaces the oldest. When the total would exceed `--prompt-cache-max-mb`, the oldest other snapshots are dropped, and a snapshot that alone exceeds the cap is not written. These decisions are a pure function (`src/cache_policy.br`) with its own interpreter tests.

Safety properties:

- a snapshot is valid only for the exact model that wrote it: the header carries a fingerprint (file size, shape, a hash of the first and last 64 KB of tensor data), and a mismatch is ignored;
- the header is written last, so an interrupted save leaves a file without the magic number, which a later scan ignores and the next save overwrites;
- floats are stored exactly as the GPU holds them, so a restored cache gives the same answers as a cold one;
- the files contain the prompts' token ids and activations. Keep the directory private.

The file layout is in [the appendix](#152-prompt-cache-file-layout).

---

## 8. Performance

### 8.1 Measured numbers

Measured on one 16 GB Apple-silicon machine (M3, Metal) with `boring` 46c2473f, with nothing else running but the benchmark. Timings still move by tens of percent between repetitions, so read the ratios as indications; the per-run minimum and maximum are in the development log.

| Model | Prompt | Prefill tok/s | Decode tok/s | Decode ms/token |
|---|---|---|---|---|
| Qwen2.5-7B Q4_K_M | about 570 | 114 | 7.7 | 130 |
| Qwen2.5-7B Q4_K_M | about 4,150 | 87 | 6.1 | 164 |
| Qwen3-4B Q4_K_M | about 550 | 181 | 10.8 | 93 |
| Qwen3-4B Q4_K_M | about 4,130 | 130 | 9.0 | 111 |
| Qwen2.5-3B Q4_K_M | about 570 | 222 | 11.7 | 85 |
| Qwen2.5-3B Q4_K_M | about 4,150 | 158 | 10.3 | 97 |
| Qwen2.5-0.5B Q8_0 | about 570 | 1,574 | 22.2 | 45 |
| Qwen2.5-0.5B Q8_0 | about 4,150 | 848 | 16.5 | 61 |

Server memory: 3.1 GB (peak 3.3 GB) for the 3B and 6.4 GB (peak 6.8 GB) for the 7B, ready in about 4 s on the 7B. Through OpenCode, the 7B model's first request (7,539 tokens, ten tools) spent 115 s in prefill when cold, and 2.5 s on the next request, which reused 7,559 of 7,638 tokens.

### 8.2 Against llama.cpp

`tools/bench_vs_llamacpp.py` runs both engines on the same GGUF, alternating them, with streamed requests timed on the client side. It refuses to start on a busy machine. Reasoning chunks (Qwen3) count as generated text.

```sh
.venv/bin/python tools/bench_vs_llamacpp.py --model qwen2.5-7b-instruct-q4_k_m --fetch \
    --binary src/main_metal/target/release/main --prompt-tokens 512,4096 --gen 64 --reps 3 --ctx 8192
```

Medians, boring-llm over llama.cpp: prefill 0.91x on the 7B at a 570-token prompt and 0.71x at 4,150; 0.66x and 0.59x on Qwen3-4B; 0.71x and 0.58x on the 3B; 0.62x and 0.56x on the 0.5B. Decode: 0.59x and 0.61x on the 7B, 0.38x and 0.58x on Qwen3-4B, 0.41x and 0.49x on the 3B, 0.20x and 0.28x on the 0.5B (the README has the table). At the start of this work these ratios were 0.14 to 0.20 on prefill and 0.24 to 0.29 on decode for the 7B.

### 8.3 Where the time goes

- **Prefill is matrix-multiply bound.** A marginal-cost profile (extra copies of one stage class per layer) showed the feed-forward projections take about 70 percent of a prefill chunk and everything else is within noise. The attention and normalization are fused or small.
- **Decode is bandwidth bound on large models.** The Q4_K decode kernels move 19 to 64 GB/s depending on the matrix shape; the projections with k = 3,584 are the slowest.
- **A decode token is dominated by the weight matrices**, not by attention, until the context reaches several thousand tokens.
- **Small models are overhead-bound in decode.** The 0.5B model takes 45 ms per token, far more than reading its 0.7 GB of weights needs, so the per-token cost of kernel launches and synchronization dominates; llama.cpp takes 9 ms for it. Long-context prefill (0.6 to 0.7 times llama.cpp's speed at 4,000 tokens, against 0.9 at 570) has not been profiled; attention over the whole cache and the float32 KV cache are the suspects.

A stage-by-stage profile of the prefill matrix multiply lives in `tools/metal_prefill_lab/`: stand-alone Metal kernels plus a timing harness, used to measure what each stage of a schedule costs and to specify faster ones.

### 8.4 Tuning

- Use a model quantized as Q4_K_M or Q8_0; both have native Metal kernels.
- Keep `--ctx-size` as small as the work needs; it bounds the KV cache.
- Use `--prompt-cache` for agent workloads. It is the largest single saving.
- On wgpu, CUDA and ROCm try `--kquant-prefill handwritten`, which is faster there today.
- Close other GPU users before measuring. A load average above 3 distorts timings.
- `BORING_METAL_BATCH=N` (a `boring` environment variable for Metal builds, `boring` ecc15864 or later) batches kernel launches into shared command buffers. It cut the CPU cost of a launch from 22 to 1.5 us but gave no measurable decode speed-up here (the GPU is the bottleneck) and made prefill 7 to 13 percent slower, so leave it unset.

---

## 9. Architecture

### 9.1 Module map

| File | Role |
|---|---|
| `src/gguf.br`, `src/gguf_file.br` | GGUF parser; reads tensors from disk on demand |
| `src/io.br`, `src/json.br`, `src/jinja.br` | binary readers, JSON, the Jinja2 subset |
| `src/tokenizer.br` | byte-level BPE and SentencePiece from the GGUF's metadata, chat-prompt rendering |
| `src/quant.br` | CPU reference dequantization for every format |
| `src/model.br` | configuration, weight loading, forward passes, KV cache, prefill and decode |
| `src/math.br`, `src/math_gpu.br` | CPU reference math; GPU kernels and their host wrappers |
| `src/sampling.br` | greedy, temperature, top-k, top-p, repeat penalty, PRNG |
| `src/openai.br` | message normalization, tool-call parsing, JSON and SSE bodies (pure) |
| `src/cache_policy.br` | the disk-cache decisions (pure) |
| `src/prompt_cache.br` | reading and writing the disk snapshots |
| `src/server.br` | the HTTP server |
| `src/main.br` | the command line |

Modules marked pure touch neither files nor the GPU, so the interpreter tests exercise them. Modules that call Rust's `std::fs` (`gguf_file.br`, `prompt_cache.br`) are kept apart from the ones the interpreter must run (`gguf.br`, `io.br`).

### 9.2 A request, end to end

1. **Parse**: `server.br` reads the HTTP request and parses the JSON body.
2. **Normalize**: `openai.br` turns content parts into text and tool-call arguments into objects.
3. **Render**: the chat template (`jinja.br`) turns the messages and tools into prompt text; the tokenizer encodes it with special tokens atomic.
4. **Find a prefix**: the longest common token prefix against the in-memory slots and the disk snapshots decides what can be reused (`cache_policy.br` decides between memory and disk).
5. **Prefill**: the remaining prompt tokens go through the model in chunks of 512, each attending over the cached keys plus its own; for a stream, a keep-alive comment is written after each chunk.
6. **Decode**: for each token, sample from the logits, advance the cache by one position, and (when streaming) send the text that has become complete.
7. **Finish**: parse tool calls from the final text, apply stop strings, write the response or the last chunks.
8. **Store**: the cache the request left is placed in a slot; if it adds enough tokens, it is written to disk after the socket is closed.

### 9.3 The forward pass

One layer, for one position (decode) or a chunk of positions (prefill):

1. RMS norm of the residual stream;
2. Q, K and V projections (with biases for Qwen2), each a quantized matrix multiply;
3. RoPE on Q and K at the right absolute positions;
4. K and V appended to the cache (the tail in decode);
5. grouped-query attention: a fused flash-style kernel for prefill (head size 64 or 128), a split-K kernel for decode, with a three-kernel fallback for other head sizes;
6. output projection and residual add;
7. RMS norm, gate and up projections, SwiGLU, down projection, residual add.

After the last layer: final RMS norm and the output projection to vocabulary logits (only the last row in prefill; the full logits matrix of a long prompt would be most of the prefill's cost for nothing).

The residual stream is uploaded once and stays on the GPU: every add consumes and produces a resident buffer. Weights are uploaded once at load time and referenced by every call.

### 9.4 The GPU kernels

| Kernel family | Purpose |
|---|---|
| fused-dequantization linear (one per format) | decode the packed weights and multiply, in one pass |
| `tensor_linear_gpu` | dispatch by tensor type to the native tensor schedules, or to the hand-written kernels |
| `RmsNormKernel`, `RopeApplyKernel` | normalization and rotary embedding (NEOX or interleaved) |
| `SwigluCombineKernel`, `AddKernel` | feed-forward activation and residual add |
| `KvAppendKernel`, `kv_tail_push` | cache appends |
| `FlashAttn64/128Kernel` | fused prefill attention |
| `FlashDecode64/128Kernel` and combine kernels | split-K decode attention |
| `FlashDecodeTail64/128Kernel` | the same over a cache split in a main buffer and a tail |
| `CopyBytesKernel`, `CopyFloatsKernel` | host-to-device upload (two-dimensional grid for tensors beyond 65,535 workgroups) |

Kernels are Boring `kernel` structs with `'global`, `'unified` and `'actor` qualifiers, transpiled to Metal Shading Language or WGSL. `src/math_gpu.br` also keeps older schedules (tiled, warp-shuffle) that the dispatcher selects for the formats without a native schedule.

### 9.5 State as values

A `mut` kernel field built from a resident argument is a copy of it (the generated wrapper copies the buffer), so a kernel cannot update the caller's buffer in place, and writing to such a field is not an error: the write goes to the copy and is lost without any message. The engine therefore passes state as values: `forward_step` takes a cache and returns `(logits, new_cache)`, the sampler takes an RNG and returns `(token, new_rng)`, and the Jinja evaluator threads its environment through return values. This is a deliberate idiom, not a workaround, and it is why a cache can be shared between slots, truncated or snapshotted without copying. The silence is the dangerous part: a fused residual-add + RMSNorm kernel written as an in-place update compiled, passed its unit test and made every model print random text (see [section 11.5](#115-the-decode-check)); `tools/decode_vs_llamacpp.py` is what catches this class of bug today, and a compile-time warning is requested from `boring` (`task_e0c5ff89`).

---

## 10. GPU Backends and Kernel Scheduling

### 10.1 Targets

| Target | Command | Notes |
|---|---|---|
| Metal | `boring build --target metal src/main.br` | primary; native tensor schedules for Q8_0, Q4_K and Q6_K |
| wgpu | `boring build --target wgpu src/main.br` | portable (Vulkan, DirectX 12, Metal); same output as Metal |
| CUDA, ROCm | `--target cuda`, `--target rocm` | shared source, never run for lack of hardware |

The generated Cargo projects (`src/main_metal/`, `src/main_wgpu/`) are fully regenerable and git-ignored.

### 10.2 boring.toml

```toml
[tensor.linear]
algorithm = "auto"

[tensor.linear.decode]
q8_0 = "warp-broadcast"

[tensor.linear.prefill]
q8_0 = "native"
q4_k = "native"
q6_k = "native"
algorithm = "scalar"
```

`kernel.tensors.linear` is Boring's built-in quantized matrix-multiply. The file chooses its schedule per phase and format: single-token decode, and batched prefill. Here the Metal native schedules are enabled for Q8_0, Q4_K and Q6_K prefill, and the default for any other format stays the scalar schedule. The decode path for Q4_K and Q6_K uses `boring`'s native decode kernels under `auto`.

A prefill schedule has a staging precision: `float` (the default, exact against the scalar schedule up to summation order) or `half` (an opt-in, faster, with a half-precision rounding error of about 3e-4 of the largest output). Half was measured to move the 7B model's logits, so it is off by default.

### 10.3 K-quant prefill on other backends

On wgpu, CUDA and ROCm the portable tensor schedule for K-quant prefill is about three times slower than the hand-written Q4_K and Q6_K kernels in `math_gpu.br`. `--kquant-prefill handwritten` selects the hand-written ones. On Metal the native schedules are faster than both, so the default is `tensor`.

### 10.4 Compiler versions

The project is also a regression suite for `boring`. Several compiler versions are known bad for it:

- before `a057dd8c`: the disk prompt cache and the native Metal K-quant kernels do not compile on macOS 27;
- before `6fb92947`: Q4_K decode is about twice as slow;
- `6acc7d44` to `316c7695`: prefill tiles are staged in half precision, so the 7B's logits drift;
- `316c7695` to `ff79f091`: biased Q8_0 projections return wrong values on small Qwen models.
- `8bdac11d` to before `d38813c8`: Q6_K prefill returns wrong values for every odd output column (the 8-byte loads of 210-byte blocks were misaligned), so Q4_K_M models produce garbage.
- before `46c2473f`: a `.split(...)` written directly as a call argument stopped a later `.split(...)` of the same function from transpiling (this project's `think_parts` in `openai.br` relies on the fix).

The README states the minimum commit to use. After any compiler update that touches the GPU kernels, run the accuracy check in [section 11.4](#114-the-accuracy-check).

---

## 11. Verification

### 11.1 What is checked, against what

| Layer | Oracle | Tool |
|---|---|---|
| GGUF metadata and tensors | the Python `gguf` package | `tools/inspect_gguf.py` |
| Tokenization | `llama-cpp-python` | `tools/oracle_encode.py`, `test/test_tokenizer_real.br` |
| Chat-template rendering | `jinja2` | `tools/oracle_render_request.py` (every fixture of `test/fixtures/requests/` against a build), `test/test_jinja_real.br` |
| Dequantization | the CPU reference in `quant.br` | `test/test_quant*.br` |
| GPU kernels | the CPU reference math | `test/test_math_gpu_real.br`, `test/test_quant_kernels_real.br` |
| Forward-pass logits | llama.cpp | `tools/reference_logits.py`, `tools/accuracy_vs_llamacpp.py` |
| Generation (decode path) | llama.cpp | `tools/decode_vs_llamacpp.py` |
| Server wire format | the OpenAI protocol | `tools/smoke_server.py` |

### 11.2 Interpreter tests

These need no model and no GPU:

```sh
for t in test/test_*.br; do case $t in *_real.br) ;; *) boring run $t;; esac; done
```

They cover the GGUF parser (on a synthetic fixture), binary readers, JSON, the Jinja subset, the OpenAI plumbing, sampling, the cache policy and the CPU math.

### 11.3 Compiled tests

The `test/*_real.br` programs need a compiled build and real weights, and each lists the files it expects in its header. The models are not in the repository; `tools/fetch_models.py` downloads the ones a test needs (`MODEL=$(tools/fetch_models.py path qwen2.5-0.5b-instruct-q4_k_m --fetch)` prints the path, downloading first when it is missing), and `tools/accuracy_vs_llamacpp.py --model NAME --fetch` does the same for the accuracy check:

```sh
boring build --target metal test/test_math_gpu_real.br
(cd test/test_math_gpu_real_metal && cargo build --release && ./target/release/test_math_gpu_real)
```

`test_math_gpu_real.br` checks every GPU kernel against the CPU reference, including the split-cache attention against a single-buffer one, on both Metal and wgpu.

### 11.4 The accuracy check

`tools/accuracy_vs_llamacpp.py` is the most important check. It feeds the same token ids to llama.cpp on the CPU and to the engine, and compares the logits at every position: top-1 agreement, and the mean absolute difference of the reference's top logit.

```sh
.venv/bin/python tools/accuracy_vs_llamacpp.py --gguf weights/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --binary src/main_metal/target/release/main
```

Healthy values on the default text (`tools/accuracy_text.txt`, a frozen copy of the project README that must not be edited): Qwen2.5-0.5B Q8_0 594 of 613 and mean 0.06; 0.5B Q4_K_M 601 of 613 and 0.06; 7B Q4_K_M 298 of 313 and 0.16. Top-1 misses on natural text are near-ties between two tokens, so the mean difference is the sharp signal: broken kernels gave 2 to 23.

It exists because kernel unit tests passed while the models regressed three times: half-precision staging, a race in the biased Q8_0 prefill, and misaligned loads of Q6_K blocks. A unit test compares a kernel with another implementation of the same idea; this compares the whole forward pass with someone else's. Run it on a Q8_0 and a Q4_K_M model (Q4_K_M files also contain Q6_K tensors) after every compiler update that touches the GPU kernels.

### 11.5 The decode check

`tools/decode_vs_llamacpp.py` covers what the accuracy check cannot: the generation path. Generation prefills in chunks and then runs one `forward_step` per token with the KV cache and the single-row kernels, while `--tokens` runs one batched forward pass. A commit once fused the attention residual add into the following RMSNorm with a kernel that updated a copy of the residual stream: every layer lost its attention output and the model printed random text, yet the unit tests, the logits check and the benchmarks all passed.

```sh
.venv/bin/python tools/decode_vs_llamacpp.py --gguf weights/qwen2.5-0.5b-instruct-q8_0.gguf \
    --binary src/main_metal/target/release/main
```

It feeds three chat prompts (as token ids, so tokenization is out of the picture) to llama.cpp on the CPU and to `--generate-tokens`, greedy-decodes 32 tokens in each, and compares the ids. If they agree to the end the prompt passes. At the first difference the comparison stops, and llama.cpp's own logit gap between its token and ours decides: below 1.0 is a near-tie (healthy kernels differ from llama.cpp by 0.06 to 0.16 on average) and passes. Healthy values: Qwen2.5-0.5B Q8_0 and Q4_K_M, 3B Q4_K_M and 7B Q4_K_M all pass, with gaps of 0.01 to 0.33 where the continuations differ; the broken fusion failed every prompt at the first token with gaps of 3 to 6. Run it with the accuracy check after any change to the forward pass or the GPU kernels. The prompts are in the ChatML format of the Qwen2.5 family; use `--prompts-file` for another format.

### 11.6 End-to-end checks

- `tools/smoke_server.py` against a running server;
- `tools/opencode_trial.sh` for a real agent session;
- greedy generation compared with llama.cpp's on real weights (identical text on Qwen2.5 0.5B Q8_0 and Q4_K_M and on 7B Q4_K_M for the prompts checked).

---

## 12. Development Guide

### 12.1 The build and test loop

1. Edit `src/*.br`.
2. `boring run test/test_<module>.br` for anything the interpreter can run.
3. `boring build --target metal src/main.br` and `cargo build --release` in `src/main_metal/` for anything on the GPU.
4. For kernels, extend `test/test_math_gpu_real.br` with a case against the CPU reference, with data that is not exactly representable in half precision (an exactly representable test hid a half-precision error once) and with a bias case where one exists.
5. For anything that changes the forward pass or a compiler update, run `tools/accuracy_vs_llamacpp.py` and `tools/decode_vs_llamacpp.py`.
6. For anything that changes the server, run `tools/smoke_server.py`.

Never edit a script while a background job runs it: shell scripts are read incrementally, and a run can pick up the edited text. The scripts here are braced blocks for that reason.

### 12.2 Adding a quantization format

1. `src/quant.br`: add the block constants, a `dequantize_<fmt>` function, and the cases in `quantized_row_bytes` and `dequantize_row` (this is the CPU reference and the embedding-row decoder).
2. `src/math_gpu.br`: add a fused-dequantization linear kernel and wrapper, and the cases in `fused_linear_gpu` and `fused_linear_nobias_gpu`.
3. `boring.toml`: if `boring` has a native schedule for the format, enable it.
4. Tests: `test/test_quant*.br` against the reference, `test/test_quant_kernels_real.br` against real weights, then the accuracy check on a model that uses the format.

### 12.3 Adding an architecture

Qwen3 is the worked example: three small changes in `model.br` (the head size from `key_length`, the NEOX rotation for the `qwen3` name, and the optional query/key norms through the existing row-wise norm kernel with one row per head), the Jinja constructs its template needs in `jinja.br`, and reasoning output in the server. Its logits matched llama.cpp on the first try because each piece was checked on its own first.

`src/model.br` reads every hyperparameter from `{arch}.*` metadata. Adding an architecture means: confirming the tensor names, adding its RoPE convention to `rope_style_for_arch` (check llama.cpp's `llama_model_rope_type`; GGUF does not say), handling any structural difference in the layer (norm placement, biases, sliding windows), and checking the logits against llama.cpp position by position. Add a tokenizer case only if its `tokenizer.ggml.model` is neither `gpt2` nor `llama`.

### 12.4 Working with GPU kernels in Boring

- A kernel is a `kernel` struct with `'global` inputs and `'unified` outputs; a host wrapper (`req ... 'kernel'unified`) launches it and returns a resident buffer.
- Resident values passed to a non-`req` helper may be rejected; keep GPU orchestration in `req` functions.
- Values built from resident arguments are copied, never mutated in place, and a write to such a copy is silently lost (no compiler diagnostic yet); return new state from functions, and run `tools/decode_vs_llamacpp.py` after touching a kernel.
- A matrix with more than 65,535 workgroups needs a two-dimensional grid (see the copy kernels).
- Profile with the marginal-cost method: queue extra copies of one stage class and measure the delta, because the GPU queue is asynchronous and per-stage synchronization inflates timings.

### 12.5 Compiler bugs

Boring is developed alongside this project. When a real compiler bug or gap turns up, reduce it to a minimal program, report it with the program and the expected behavior, and keep the workaround out of shared code once it is fixed. The development log records each one: what was seen, the reduction, and which commit fixed it.

### 12.6 Documentation

This book is `docs/book.md`. The HTML version is generated, with no dependency beyond Python 3:

```sh
python3 docs/build.py
```

This writes `docs/book.html`, `README.html` at the repository root, and one HTML file per other Markdown file in `docs/`.

---

## 13. Limits and Known Issues

- **One request at a time.** There is no batching and no concurrent decoding.
- **Context** is limited by `--ctx-size` and memory. The KV cache is float32, twice the size llama.cpp's float16 cache needs.
- **Architectures**: Llama-style GGUF layouts only (`qwen2` and `llama` are the tested ones); no mixture-of-experts, no vision.
- **Chat templates** use a Jinja2 subset; an unsupported construct is reported.
- **Single-file GGUF only.** Split models are not supported.
- **Unix only for the disk features**: the streaming loader and snapshots use Unix file APIs.
- **wgpu, CUDA, ROCm**: K-quant prefill through the portable schedule is about three times slower than the hand-written kernels; CUDA and ROCm have never been run.
- **Speed** is about half to two thirds of llama.cpp's on the 7B model (see [chapter 8](#8-performance)); the remaining gap is in the Metal kernels and the float32 KV cache.
- **Loopback only, no authentication.**

---

## 14. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `context_length_exceeded` | the prompt is longer than `--ctx-size`; raise it (OpenCode needs at least 16384) |
| a client times out on the first request | it gave up on a silent response; check that it reads SSE comment lines, or use `--prompt-cache` so only the first request ever is slow |
| the build fails on macOS 27 with Metal K-quant kernels | the `boring` compiler is older than the README's minimum commit |
| `quantized tensor dimension k must be a multiple of 256` | a K-quant tensor whose row length is not a multiple of 256 |
| random text from a model whose logits check passes | a decode-path bug (`forward_step`, the KV cache, a single-row kernel, a residual update on a copy); run `tools/decode_vs_llamacpp.py` |
| garbage output on a Q4_K_M model after a compiler update | a kernel regression; run `tools/accuracy_vs_llamacpp.py` and compare the mean logit difference with the healthy values in [section 11.4](#114-the-accuracy-check) |
| `config_from_gguf: unknown RoPE convention for architecture` | an architecture other than `qwen2` or `llama`; see [section 12.3](#123-adding-an-architecture) |
| the machine swaps or freezes at load | use `--quantized` (and `--serve`, which always does); the float32 path needs several times the file size |
| a tool call arrives as text | the model wrote an invalid block; small models do this under long prompts; the parser repairs unbalanced braces but cannot fix a missing name |
| `weights/....gguf is missing` | the models are not in git; `tools/fetch_models.py ensure NAME` downloads one (`tools/fetch_models.py list` shows the names and what is present) |
| a Qwen3 reply is long before the answer comes | it is thinking; the reasoning arrives as `reasoning_content`. Send `chat_template_kwargs: {"enable_thinking": false}` for a direct answer |
| the disk cache is never used | the prompt must start like a snapshot; OpenCode writes the project path into its system prompt, so a different directory means a different prefix; check the `[serve] prompt cache:` lines |
| timings vary by 2x between runs | other work on the GPU or memory pressure; measure on a quiet machine, alternating engines |

---

## 15. Appendix: Reference Tables

### 15.1 Source file sizes and roles

| File | Lines (about) | Contents |
|---|---|---|
| `src/math_gpu.br` | 3,200 | every GPU kernel and its host wrapper |
| `src/model.br` | 1,000 | configuration, loading, forward, caches |
| `src/jinja.br` | 830 | the Jinja2 subset |
| `src/tokenizer.br` | 700 | both tokenizers, chat-prompt rendering |
| `src/main.br` | 560 | command line, modes |
| `src/quant.br` | 520 | dequantization reference |
| `src/server.br` | 450 | HTTP server |
| `src/gguf.br` | 370 | GGUF parser |
| `src/openai.br` | 350 | API plumbing |
| `src/json.br` | 220 | JSON |
| `src/prompt_cache.br` | 200 | disk snapshots |
| `src/sampling.br` | 190 | sampling |
| `src/math.br` | 180 | CPU reference math |
| `src/io.br` | 165 | binary readers |
| `src/cache_policy.br` | 105 | cache decisions |

### 15.2 Prompt cache file layout

Little-endian. One file per snapshot, `prompt-cache-<n>.bin`:

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | magic `BCGK` (1263681346) |
| 4 | 4 | version (1) |
| 8 | 4 | model fingerprint |
| 12 | 4 | number of layers |
| 16 | 4 | K/V dimension (`n_kv_head x d_head`) |
| 20 | 4 | number of tokens N |
| 24 | 4 | sequence number (larger means saved later) |
| 28 | 4 | reserved |
| 32 | 4 x N | token ids |
| after | per layer | K rows, then V rows, `N x kv_dim` float32 each |

The header is written last. A file whose size does not equal header plus tokens plus `n_layer x 2 x N x kv_dim x 4` is treated as incomplete and ignored. Bytes per token in a snapshot: `4 + n_layer x 2 x kv_dim x 4`.

### 15.3 Defaults at a glance

| Setting | Default |
|---|---|
| port | 8080 |
| context size | 4,096 tokens |
| reply length | 40 tokens |
| in-memory cache slots | 4 |
| prefill chunk | 512 tokens |
| KV tail | folded every 64 tokens |
| disk snapshot files | 2 |
| disk cache cap | 4,096 MB |
| disk save threshold | 1,024 new tokens |
| disk restore threshold | 256 more tokens than memory |
| sampling | greedy, repeat penalty off |

### 15.4 Tools

| Tool | Purpose |
|---|---|
| `tools/accuracy_vs_llamacpp.py` | whole forward pass against llama.cpp, position by position |
| `tools/decode_vs_llamacpp.py` | generation (the decode path) against llama.cpp, token by token |
| `tools/bench_vs_llamacpp.py` | prefill and decode speed against llama.cpp |
| `tools/reference_logits.py` | llama.cpp's own logits for a fixed prompt |
| `tools/oracle_encode.py` | tokenization against llama.cpp |
| `tools/oracle_render_request.py` | chat-template rendering against `jinja2`; with `--binary` it compares a build on every request file |
| `tools/fetch_models.py` | downloads the test and benchmark models into `weights/` on demand, with resume and SHA-256 verification |
| `tools/inspect_gguf.py` | GGUF metadata through the Python `gguf` package |
| `tools/make_test_gguf.py` | builds the small synthetic GGUF used by the interpreter tests |
| `tools/smoke_server.py` | wire-format checks against a running server |
| `tools/opencode_trial.sh` | a real OpenCode session in an isolated configuration |
| `tools/metal_prefill_lab/` | stand-alone Metal kernels and timing harness for the prefill matrix multiply |

### 15.5 Glossary

| Term | Meaning |
|---|---|
| prefill | processing the prompt tokens before the first reply token |
| decode | generating one reply token at a time |
| KV cache | the keys and values of every processed position, per layer |
| GQA | grouped-query attention: several query heads share one key and value head |
| RoPE | rotary position embedding |
| GGUF | the model file format of llama.cpp |
| K-quant | the 256-value super-block formats (Q2_K to Q6_K) |
| SSE | server-sent events, the streaming response format |
| slot | an in-memory KV cache with the token ids it covers |
| snapshot | a KV cache written to disk |
