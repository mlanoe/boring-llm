# Development log

This is the phase-by-phase build history of `boring-llm`: what was implemented, how it was verified, and every real [Boring](https://github.com/mlanoe/boring) compiler bug/gap found and fixed along the way. It used to live in the main `README.md`; moved here so the README can focus on presenting the project rather than its construction. See `README.md` for what the project is and how to run it.

---

**Phase 0 done and verified against a real model file.** `src/gguf.br` parses a GGUF file's header, all metadata value kinds (scalars, strings, arrays), and tensor-info list in pure Boring. Verified two ways against the real `Qwen/Qwen2.5-0.5B-Instruct-GGUF` q8_0 file (676MB, 291 tensors, 26 metadata keys, real production weights — not a toy fixture): cross-checked field-for-field against `tools/inspect_gguf.py` (the Python `gguf` package as ground truth) with an exact match, and independently verified by reading raw tensor bytes back out of a synthetic fixture (`test/test_gguf.br`) and confirming they match the source values bit-for-bit. Resolves this model's tied-vs-untied embedding question directly: it has a separate `output.weight` tensor (untied).

Note: `boring run`'s interpreter OOMs on a file this size (a real, filed interpreter limitation — packed byte buffers are stored as one boxed `Value` per byte internally) — real-scale parsing goes through `boring build` + `cargo build`, same workaround `whisper-boring` already uses for its own large-array limitation. `boring run` is still used for fast dev-loop testing against the small synthetic fixture in `test/fixtures/`.

**Phase 1 done and verified against the real model too.** `src/math.br` (RMSNorm, NEOX-style RoPE, naive causal GQA attention, SwiGLU) + `src/quant.br` (Q8_0 dequantization) + `src/model.br` (config/weight loading, full forward pass) — pure Boring, no GPU. Verified against real `llama-cpp-python` logits (`tools/reference_logits.py`) for the prompt "The capital of France is" on **both** the fp16 and Q8_0 real GGUF files: every position's top-5 token ranking matches exactly, logit values agree to within float32-accumulation-order noise (a few hundredths on fp16, a bit more on Q8_0 near close ties — both expected, documented gaps, not bugs). Final-position top prediction is token 12095 ("Paris") on both files, matching llama.cpp exactly.

```sh
boring build src/main.br
cd src/main_rust && cargo build --release
./target/release/main --gguf ../../weights/qwen2.5-0.5b-instruct-fp16.gguf --tokens "785,6722,315,9625,374"
```

**Phase 2a done and verified against the real model, on real Metal GPU hardware.** `src/math_gpu.br` — tiled-GEMM `LinearKernel`, `RmsNormKernel`, `RopeApplyKernel`, `SwigluCombineKernel`, and GQA-aware `MatMulBTHeadsKernel`/`SoftmaxRowsKernel`/`MatMulHeadsKernel` — all fp32, still no fused dequantization (Q8_0 tensors are dequantized on the CPU at load time, same as Phase 1; Phase 2b fuses that into the GEMM kernel itself). Still no KV-cache — full re-forward every position. Verified against Phase 1's own CPU output on both real GGUFs: matches to 5-6 significant figures (tiny reduction-order rounding differences, not a bug), same top-5 rankings, same top-1 prediction (token 12095, "Paris"). Runs in ~11-33s including model load and weight upload.

```sh
boring build --target metal src/main.br
cd src/main_metal && cargo build --release
./target/release/main --gguf ../../weights/qwen2.5-0.5b-instruct-fp16.gguf --tokens "785,6722,315,9625,374" --gpu
```

**Phase 2b done and verified against the real Q8_0 model, on real Metal GPU hardware.** `Q8LinearKernel` (`src/math_gpu.br`) dequantizes each weight element on the fly inside the tiled GEMM's shared-memory tile-load step, reading straight from GGUF's raw Q8_0-packed bytes — no full fp32 weight copy is ever materialized (for the LM head alone, 151936×896, that's the difference between ~146MB of packed bytes and a ~544MB dequantized copy). `model.br`'s `load_model_quantized`/`forward_gpu_quantized` load every big matmul weight (Q/K/V/O projections, FFN gate/up/down, LM head) as raw bytes; only norms/biases (always F32 in real GGUF files) and the token-embedding lookup (dequantized one row at a time, not the whole table) touch the CPU. Verified: produces **bit-identical output** to Phase 2a's CPU-dequant-then-GPU-matmul path on the real model — same top-5 rankings, same logit values to the last printed digit.

```sh
./target/release/main --gguf ../../weights/qwen2.5-0.5b-instruct-q8_0.gguf --tokens "785,6722,315,9625,374" --quantized
```

**Phase 3 done and verified — full end-to-end chat generation, matching llama.cpp exactly.** `src/tokenizer.br` (byte-level BPE, built directly from GGUF-embedded `tokenizer.ggml.*` metadata — no separate vocab.json/merges.txt export step) + Qwen2.5 ChatML wrapping + a greedy-decode loop (`model.br`'s `generate()`, still no KV-cache — full re-forward every step) + detokenization. Verified against `llama-cpp-python`:
- `encode()`/`encode_chat()` produce **token-for-token identical** IDs to the reference tokenizer, including a full 26-token ChatML-wrapped prompt (`test/test_tokenizer_real.br`).
- End-to-end generation for "What is the capital of France?" produces **the exact same text** as `llama-cpp-python`'s own greedy chat completion: `"The capital of France is Paris."`, stopping correctly at `<|im_end|>`.
- A second, longer prompt ("What is the tallest mountain in the world?") produces coherent, factually correct output over 40 generated tokens.

```sh
./target/release/main --gguf ../../weights/qwen2.5-0.5b-instruct-q8_0.gguf --quantized \
    --prompt "What is the capital of France?" --max-tokens 30
```

This closes out the plan's core roadmap (Phases 0-3).

**Phase 4 (stretch): KV-cache incremental decode, done and verified.** `model.br`'s `generate_kv()`/`forward_step()`/`prefill_gpu_quantized()` replace "re-forward the whole sequence every step" with a real growing per-layer K/V cache: a single batched forward pass over the prompt (same kernels/dispatch count as the no-cache path, regardless of prompt length) seeds the cache, then each new token costs O(1) GEMM work per layer instead of O(current length) — only attention's own cost still scales with the cache size, which no KV-cache design avoids. Verified **bit-identical output** to the already-verified no-cache path on two different prompts (short and longer generations), and measurably faster once generation length is long enough to amortize per-call overhead (28 generated tokens: ~48s with `--kv-cache` vs. ~72s without, on the same prompt). For very short generations the no-cache path can still be marginally faster (fixed per-step dispatch overhead dominates before the O(seq)-vs-O(seq²) FLOP savings catch up) — both paths are kept, selectable via `--kv-cache`.

```sh
./target/release/main --gguf ../../weights/qwen2.5-0.5b-instruct-q8_0.gguf --quantized \
    --prompt "What is the capital of France?" --max-tokens 30 --kv-cache
```

**Phase 4 (stretch): temperature/top-k/top-p sampling, done and verified.** `src/sampling.br` — a self-seeded xorshift64 PRNG (Boring has no builtin RNG; same precedent as scratch-boring's own `Rng`, kept purely functional here — `rng_next_u64`/`rng_next_f32` return `(value, new_Rng)` rather than mutating in place) plus temperature scaling, a top-k pre-filter (bounded selection, no full 150k-entry sort), and top-p/nucleus filtering over the (already-sorted) top-k candidates. `temperature <= 0` is still plain greedy argmax (the default, and what every earlier phase's verification used). Verified: `--seed` gives reproducible output for the same seed; different seeds diverge into genuinely different, coherent, on-topic completions at a higher temperature (three different short cat stories from three seeds); a highly-confident completion ("The capital of France is Paris.") stays the same across seeds/greedy, as expected when one continuation dominates the probability mass regardless of sampling settings.

```sh
./target/release/main --gguf ../../weights/qwen2.5-0.5b-instruct-q8_0.gguf --quantized --kv-cache \
    --prompt "Tell me a short story about a cat." --max-tokens 40 \
    --temperature 1.3 --top-k 60 --top-p 0.95 --seed 3
```

**Phase 4 (stretch): k-quant formats beyond Q8_0, done and verified.** `src/quant.br` adds `dequantize_q5_0`/`dequantize_q6_k`/`dequantize_q4_k` — a legacy 5-bit format and two real "superblock" k-quant formats (256-element blocks with quantized per-sub-block scales, Q4_K's own quantized per-sub-block *minimums* too), ported bit-for-bit from ggml's own `dequantize_row_q5_0`/`q6_K`/`q4_K`. `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` (the model this name usually refers to) turns out to be a genuine **mix** of five tensor formats depending on layer/role (F32 norms/biases, Q5_0 for token embeddings and FFN gate/up, Q8_0 for some attention/output tensors, Q6_K and Q4_K split across different `ffn_down` layers) — all three new dequantizers, plus the already-existing F32/F16/Q8_0 ones, now load correctly and produce real end-to-end generation. Verified two ways: the exact real superblock bytes from this file, dequantized in Boring and cross-checked element-for-element against Python's `gguf` package's own `dequantize()` (`test/test_quant_kquant.br`); and full text generation ("What is the capital of France?" → "The capital of France is Paris.", matching every earlier phase's verified output) on the real, fully-mixed-format file.

```sh
./target/release/main --gguf ../../weights/qwen2.5-0.5b-instruct-q4_k_m.gguf \
    --prompt "Tell me a short story about a cat." --max-tokens 40 --temperature 1.0 --top-k 40 --top-p 0.9 --seed 5
```

Scope note: these three new formats go through the same "dequantize once at load, run ordinary fp32 GPU kernels" path Phase 2a already proved (`generate_f32`/`forward_gpu`) — none of them got their own **fused**-dequant-inside-the-GEMM-kernel treatment the way Q8_0 did in Phase 2b (`--quantized` stays Q8_0-only, and has no KV-cache-free variant for these formats yet). A real follow-up, not attempted here, given how much more involved Q4_K/Q6_K's bit-unpacking already is just on the CPU side — and every lesson from Phase 2b's `Q8LinearKernel` (inline everything into the kernel's own `def()` body; no kernel-method or free-function-with-array-parameter factoring works on Metal) would still apply, just with considerably more logic to inline per element.

**Phase 4 (stretch): warp-shuffle GEMM reduction — implemented, verified correct, honestly *not* a performance win.** `math_gpu.br`'s `LinearWarpKernel`/`linear_warp_gpu` uses `gpu.warp.*` (real, working primitives — `warp.lane`/`warp.size`/`shuffle_xor`, verified first on an isolated micro-kernel, then on this real GEMM shape) for a one-warp-per-output-cell reduction with zero shared memory and zero block-wide barriers, the exact follow-up `boring`'s own `warp-level-primitives.md` names for this project's tiled kernels. Verified numerically correct against `linear_gpu` (small hand-picked values and a `d_in=100` case exercising multiple warp-stride rounds). **Benchmarked honestly** (`test/bench_linear_warp.br`, real Metal hardware, this model's actual dimensions) — and it's **slower**, not faster: ~1.5x slower at attention/FFN-projection scale (d_in=896, d_out=896), ~2.4x slower at LM-head scale (d_in=896, d_out=8192). Removing the barrier was never the bottleneck here — `linear_gpu`'s shared-memory tile is a real data-reuse mechanism (every input loaded from global memory once, reused by a whole 16x16 block of threads); this kernel has none of that, so far more redundant global-memory bandwidth is spent than the barriers it removes were ever costing. Same lesson as the KV-cache work: measure wall-clock at this project's real sizes, don't assume "fewer synchronization points" means "faster." Kept in the codebase (correct, independently verified, real production dispatch/grid-overflow handling) alongside `linear_gpu`, not swapped in anywhere — a properly warp-aligned tiled+shuffle hybrid (keep the shared-memory reuse, use shuffles only for the final in-warp reduction step) is the real follow-up this result points to, not attempted here.

```sh
boring build --target metal test/bench_linear_warp.br
cd test/bench_linear_warp_metal && cargo build --release
./target/release/bench_linear_warp tiled-big   # linear_gpu
./target/release/bench_linear_warp warp-big    # linear_warp_gpu
```

**Phase 4 (stretch): wgpu backend verification — done completely, and it drove 12 real `boring` compiler fixes.** Rebuilt `test/test_math_gpu_real.br` under `--target wgpu` (the only alternative GPU backend with real testable hardware on this machine, via Metal-backed wgpu — CUDA/ROCm still have no available hardware) and worked through failures one at a time, same source verified correct on `--target metal` throughout. Found and filed 12 distinct, real `boring` wgpu-backend bugs/gaps along the way, in three broad clusters:

- **Codegen correctness**: a literal-division-by-zero Inf/NaN construction rejected by WGSL's compile-time-constant rules; `.powf()`/`.ln()`/`.signum()`/`.cbrt()`/`.log10()` transpiling to nonexistent WGSL identifiers instead of `pow`/`log`/`sign`/etc.; a non-literal block-size dispatch dimension silently becoming workgroup size 1 (fixed twice — once for top-level `let` constants, once more for constants local to the enclosing function); a workgroup-shared field-name collision between two different kernel structs; a stray `&` breaking host-function-to-host-function chaining.
- **Kernel init()-setup dropped in host wrapper functions**: a cluster of related-looking cases (an `init()`-derived scalar field staying 0, a `.reshape()`'d input's axis-dimension fields never assigned) that shared one systemic root cause in the wgpu host-codegen path.
- **Missing capability**: `[uint8]`-element storage buffers weren't supported at all (WGSL has no 8-bit storage type) — first landed as a correct compile-time rejection instead of silent data corruption, then as full support via transparent `array<u32>` packing; and the wgpu device-creation code requested `wgpu::Limits::default()` (a conservative 128 MiB max buffer-binding size) instead of the adapter's real capability (confirmed ~9.5 GB really available on this machine), which blocked any buffer bigger than that — including this model's ~144 MB LM-head weight.

**All 12 are now fixed upstream in `boring` itself and reverified with zero regression.** Every source-level workaround added while chasing them — the flat-array/manual-indexing rewrites in `TransposeKernel`/`MatMulBTHeadsKernel`/`SoftmaxRowsKernel`/`MatMulHeadsKernel`, `RopeApplyKernel`'s recomputed `half_dim` and `log2`/`exp`-based `.powf()` replacement, the `q8_tile_x`/`q8_tile_w` field rename, `io.br`'s runtime-derived-zero Inf/NaN construction, `linear_warp_gpu`'s literal-`8` block-size dispatch — was reverted back to clean, idiomatic Boring source. Also cleaned up in the process, once the host-function-chaining bug was confirmed fixed: `model.br`'s `passthrough()` round-trip, previously mandatory at *every* GPU-host-function-to-GPU-host-function boundary throughout `forward_gpu`/`forward_gpu_quantized`/`prefill_gpu_quantized`/`forward_step` (~150 call sites), is now only needed where a value is actually iterated (`for..in`) or placed into a struct constructor/tuple directly.

**Final result**: the full compiled kernel suite (`test_math_gpu_real.br`, every kernel, no exceptions) passes on `--target wgpu` with zero source-level workarounds, and the real CLI's end-to-end Q8_0 chat generation — GGUF parsing, tokenizer, the full 24-layer forward pass, sampling, all chained together against the actual downloaded model weights — produces **bit-identical output to `--target metal`** (`"The capital of France is Paris."`, with and without `--kv-cache`). Both backends are now equally real, equally verified. CUDA/ROCm remain untested (no hardware available on this machine for either).

Remaining Phase 4 ideas, not started: CUDA/ROCm verification (no hardware available to test either).

**Phase 5 (post-MVP shared-library extraction with whisper-boring) — considered, decided against.** Diffed this project's `math_gpu.br`/`io.br` against whisper-boring's once both were stable: the genuinely identical surface turned out to be small (`TransposeKernel`, `SoftmaxRowsKernel`, and `io.br`'s LE binary readers — a few hundred lines) — `LinearKernel` (pre-transpose contract vs. GGUF's native layout) and the head-mapping kernels (GQA vs. no GQA) have real, load-bearing differences that aren't safely unifiable without genuine design work, not just a mechanical extraction. Decision: not worth a new shared repo's overhead for that little truly-common code — keep both projects' kernels independent. (Also surfaced, in passing, unrelated to this decision: whisper-boring's `io.br` still has the naive per-byte UTF-8 decoder this project's own `bytes_to_str` used to have — currently harmless there since it's only ever applied to ASCII tensor names, but worth knowing if that ever changes.)
