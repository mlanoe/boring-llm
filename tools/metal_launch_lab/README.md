# Metal launch-cost lab

How much one kernel launch costs when launches are chained on resident data (what
a decode token does: 435 kernels on the 0.5B model, 18 per layer).

- `launch.br`: a Boring program (`gpu.thread`, a `'unified` output, a `req` wrapper) that
  runs N trivial kernels in a chain on a resident buffer and prints the time per launch.
  Build with `boring build --target metal launch.br` in a directory with a `boring.toml`
  (`[project] name = "launchcost"`, `main = "launch.br"`), then `cargo build --release` in
  `launch_metal` and run `launch 435 1024`.
- `lc.swift`: the same chain in plain Metal, three ways: one command buffer per launch,
  one command buffer with one encoder per launch, one command buffer and one encoder.
  `swiftc -O lc.swift -o lc && ./lc 435 1024`.

Results (Apple M3, `boring` 46c2473f, 435 launches, a 1,024-element kernel): Boring 21-27 us
per launch; plain Metal with one command buffer per launch 22.3 us, with one command buffer
and one encoder 1.4 us. At 1,048,576 elements the kernel itself dominates (65-73 us per launch
in plain Metal in every mode).
