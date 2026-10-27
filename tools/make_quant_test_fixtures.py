#!/usr/bin/env python3
"""Raw blocks of the formats added after Q8_0/K-quants (Q4_1, Q5_1, Q5_K, IQ4_XS, IQ4_NL, F16, BF16)
plus the values the Python `gguf` package dequantizes them to, for test/test_quant_formats.br.

    .venv/bin/python tools/make_quant_test_fixtures.py
"""
import numpy as np
import gguf
from gguf import GGMLQuantizationType as T

OUT = "test/fixtures"
rng = np.random.default_rng(11)


def random_blocks(qtype, n_elements):
    block, size = gguf.GGML_QUANT_SIZES[qtype]
    n_blocks = n_elements // block
    raw = rng.integers(0, 256, size=(n_blocks, size), dtype=np.uint8)
    if qtype in (T.Q4_1, T.Q5_1, T.Q5_K, T.IQ4_XS, T.IQ4_NL):  # an f16 scale up front (and a second f16 field: minimum, dmin or scales_h/l bytes, left random)
        raw[:, 0:2] = rng.normal(0, 0.05, n_blocks).astype(np.float16).view(np.uint8).reshape(-1, 2)
        raw[:, 2:4] = rng.normal(0, 0.05, n_blocks).astype(np.float16).view(np.uint8).reshape(-1, 2)
    return raw


for qtype, name, n in ((T.Q4_1, "q4_1", 96), (T.Q5_1, "q5_1", 96), (T.Q5_K, "q5_k", 768), (T.IQ4_XS, "iq4_xs", 768), (T.IQ4_NL, "iq4_nl", 96)):
    raw = random_blocks(qtype, n)
    raw.tofile(f"{OUT}/quant-{name}.bin")
    gguf.quants.dequantize(raw, qtype).astype("<f4").reshape(-1).tofile(f"{OUT}/quant-{name}.f32")

vals = rng.normal(0, 3, 64).astype(np.float32)
vals.astype(np.float16).view(np.uint8).tofile(f"{OUT}/quant-f16.bin")
vals.astype(np.float16).astype("<f4").tofile(f"{OUT}/quant-f16.f32")
bf = (vals.view(np.uint32) >> 16).astype(np.uint16)
bf.view(np.uint8).tofile(f"{OUT}/quant-bf16.bin")
(bf.astype(np.uint32) << 16).view(np.float32).astype("<f4").tofile(f"{OUT}/quant-bf16.f32")
print("wrote quant fixtures")
