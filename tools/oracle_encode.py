#!/usr/bin/env python3
"""Independent oracle for tokenizer.br's plain-text encoding.

Reads a JSON array of strings and prints one `ids: 1,2,3` line per string, in
llama.cpp's own tokenization (special-token strings parsed atomically, no BOS)
— the exact format `main --encode-file <file>` prints, so the outputs diff.

Usage (repo root, project venv):
    .venv/bin/python tools/oracle_encode.py weights/model.gguf test/fixtures/pretok_corpus.json
"""
import json
import sys

from llama_cpp import Llama

llm = Llama(model_path=sys.argv[1], vocab_only=True, verbose=False)
for text in json.load(open(sys.argv[2])):
    ids = llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)
    print("ids: " + ",".join(str(i) for i in ids))
