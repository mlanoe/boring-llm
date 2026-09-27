#!/usr/bin/env python3
"""Cross-check oracle for src/gguf.br's parser — NOT a conversion step.

GGUF needs no Python pre-conversion (unlike whisper-boring's .wbrf format,
which exists only because Whisper's source is a bare PyTorch checkpoint).
This script exists purely so Phase 0's boring-authored parser has an
independent ground truth to diff against, using the real `gguf` package.

Usage:
    tools/inspect_gguf.py weights/qwen2.5-0.5b-instruct-q8_0.gguf
"""
import sys
import gguf

path = sys.argv[1]
r = gguf.GGUFReader(path)

print(f"GGUF version: {r.fields['GGUF.version'].parts[-1][0]}")
print(f"tensor_count: {len(r.tensors)}")
print(f"kv_count: {len([f for f in r.fields if not f.startswith('GGUF.')])}")
print()

for name, field in r.fields.items():
    if name.startswith("GGUF."):
        continue
    types = field.types
    if types and types[0] == gguf.GGUFValueType.ARRAY:
        print(f"{name} = [array of {len(field.data)} items]")
    elif types and types[0] == gguf.GGUFValueType.STRING:
        s = str(bytes(field.parts[-1]), encoding="utf-8", errors="replace")
        print(f'{name} = "{s}"')
    else:
        vals = [field.parts[idx][0] for idx in field.data]
        print(f"{name} = {vals[0] if len(vals) == 1 else vals}")

print()
print(f"-- tensors ({len(r.tensors)}) --")
has_output_weight = False
for t in r.tensors:
    dims = list(t.shape)
    print(f"  {t.name}: dims={dims} type={t.tensor_type.name} offset={t.data_offset}")
    if t.name == "output.weight":
        has_output_weight = True

tok_field = r.fields.get("tokenizer.ggml.tokens")
vocab_size = len(tok_field.data) if tok_field else 0
print()
print(f"vocab size: {vocab_size}")
print(f"has separate output.weight (untied embeddings): {has_output_weight}")
