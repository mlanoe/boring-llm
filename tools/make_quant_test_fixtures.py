#!/usr/bin/env python3
"""Raw blocks of the formats added after Q8_0/K-quants (Q4_1, Q5_1, Q5_K, IQ4_XS, IQ4_NL, the IQ1/IQ2/IQ3 family, F16, BF16)
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
    if qtype in (T.IQ2_XXS, T.IQ2_XS, T.IQ2_S, T.IQ3_XXS, T.IQ3_S, T.IQ1_S):  # f16 d up front, the rest random
        raw[:, 0:2] = rng.normal(0, 0.05, n_blocks).astype(np.float16).view(np.uint8).reshape(-1, 2)
    if qtype == T.IQ1_M:  # no d field: its f16 bits are the top nibbles of the four u16 scale words at bytes 48..55
        d_bits = rng.normal(0, 0.05, n_blocks).astype(np.float16).view(np.uint16).astype(np.uint32)
        for k in range(4):
            word = raw[:, 48 + 2 * k].astype(np.uint32) | (raw[:, 49 + 2 * k].astype(np.uint32) << 8)
            word = (word & 0x0FFF) | (((d_bits >> (4 * k)) & 0xF) << 12)
            raw[:, 48 + 2 * k] = (word & 0xFF).astype(np.uint8)
            raw[:, 49 + 2 * k] = (word >> 8).astype(np.uint8)
    return raw


QUANTS = (
    (T.Q4_1, "q4_1", 96), (T.Q5_1, "q5_1", 96), (T.Q5_K, "q5_k", 768), (T.IQ4_XS, "iq4_xs", 768), (T.IQ4_NL, "iq4_nl", 96),
    (T.IQ2_XXS, "iq2_xxs", 768), (T.IQ2_XS, "iq2_xs", 768), (T.IQ2_S, "iq2_s", 768),
    (T.IQ3_XXS, "iq3_xxs", 768), (T.IQ3_S, "iq3_s", 768), (T.IQ1_S, "iq1_s", 768), (T.IQ1_M, "iq1_m", 768),
)
for qtype, name, n in QUANTS:
    raw = random_blocks(qtype, n)
    raw.tofile(f"{OUT}/quant-{name}.bin")
    gguf.quants.dequantize(raw, qtype).astype("<f4").reshape(-1).tofile(f"{OUT}/quant-{name}.f32")

vals = rng.normal(0, 3, 64).astype(np.float32)
vals.astype(np.float16).view(np.uint8).tofile(f"{OUT}/quant-f16.bin")
vals.astype(np.float16).astype("<f4").tofile(f"{OUT}/quant-f16.f32")
bf = (vals.view(np.uint32) >> 16).astype(np.uint16)
bf.view(np.uint8).tofile(f"{OUT}/quant-bf16.bin")
(bf.astype(np.uint32) << 16).view(np.float32).astype("<f4").tofile(f"{OUT}/quant-bf16.f32")
# float32 -> f16 bit patterns (including subnormals, ties and overflow) for io.br's f32_to_f16_bits
tof16 = np.concatenate([
    rng.normal(0, 1, 200),
    rng.normal(0, 1e-5, 200),  # subnormal range of binary16
    rng.uniform(-70000, 70000, 40),  # up to and beyond the overflow threshold
    [0.0, 1.0, -1.0, 65504.0, 65519.0, 65520.0, 6.1035156e-05, 5.9604645e-08, 2.9802322e-08],
]).astype(np.float32)
tof16.astype("<f4").tofile(f"{OUT}/quant-tof16.in.f32")
tof16.astype(np.float16).view(np.uint16).astype("<u2").tofile(f"{OUT}/quant-tof16.out.u16")

# Q8_0 re-quantization of an IQ tensor's values (what the loader does with IQ1/IQ2/IQ3 weights)
iq_vals = np.fromfile(f"{OUT}/quant-iq2_xxs.f32", dtype="<f4")
gguf.quants.quantize(iq_vals, T.Q8_0).tofile(f"{OUT}/quant-q8req.out.bin")
print("wrote quant fixtures")
