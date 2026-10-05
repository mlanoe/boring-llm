# Metal prefill lab

Stand-alone Metal kernels and a timing harness for the quantized prefill matmul
(Q4_K weights, float32 activations, m = 512): it answers "where does the time of
the native prefill schedule go, and what does a different schedule give",
without going through `boring`'s transpiler.

- `harness.swift`: compiles a `.metal` file at run time, builds deterministic Q4_K
  weights and activations, times back-to-back dispatches with the GPU's own clock
  (median of 5), and can check the result against a saved reference (`--ref`) or
  against a CPU dequantization of 48 sampled outputs (`--cpu`).
- `base.metal`: a transcription of `boring`'s Q4_K prefill schedule (64x32 tile,
  8 simdgroups, k step 32). `STUB` removes one stage at a time (1 dequant
  arithmetic, 2 weight tile fill, 3 both tile fills, 4 everything but the matrix
  multiply) to measure the cost of each.
- `tiled.metal`: the same computation with a parametrised tile (`BM`, `BN`,
  `SG_M`, `SG_N`, `BK`), fragment-blocked threadgroup tiles and one scale decode
  per 8-16 weights. Shapes must be multiples of the tile (no edge code).
- `prefetch.metal`: `tiled.metal` with the global loads of the next step issued
  before the matrix multiply of the current one (it does not help).

```bash
swiftc -O harness.swift -o harness
./harness base.metal k_base 256 64 32 --D ACT=float --D WGT=float --out ref.bin --cpu
./harness tiled.metal k_t 256 64 64 --D BM=64 --D BN=64 --D SG_M=2 --D SG_N=4 --D BK=32 \
          --D ACT=float --D WGT=float --D SWIZZLE=1 --ref ref.bin
```

Arguments: file, kernel, threads per group, tile rows (m), tile columns (n); then
`--D NAME=VALUE` macros and `--m/--n/--k/--iters/--reps`. Results are in
`docs/development-log.md` (2026-10-10).
