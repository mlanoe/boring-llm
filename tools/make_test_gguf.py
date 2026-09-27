#!/usr/bin/env python3
"""Builds a tiny synthetic GGUF file for test/test_gguf.br to parse.

Not part of the real model-loading pipeline (that's src/gguf.br, pure
Boring) — this is a one-off fixture generator using the real Python `gguf`
package as ground truth, so the fixture's byte layout is guaranteed correct
GGUF, independent of anything this project's own parser does.
"""
import numpy as np
import gguf

OUT = "test/fixtures/tiny.gguf"

w = gguf.GGUFWriter(OUT, arch="tinytest")

w.add_uint32("tinytest.block_count", 3)
w.add_int8("tinytest.neg_i8", -5)
w.add_int16("tinytest.neg_i16", -1000)
w.add_float32("tinytest.attention.layer_norm_rms_epsilon", 1.0e-6)
w.add_bool("tinytest.flag", True)
w.add_string("tinytest.description", "hello gguf")
w.add_array("tinytest.token_list", ["<pad>", "hi", "there"])
w.add_array("tinytest.token_type", [1, 2, 2])

tensor = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
w.add_tensor("weight.0", tensor)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()

print(f"wrote {OUT}")
