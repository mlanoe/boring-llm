#!/usr/bin/env python3
"""Reference oracle for Phase 1: dumps llama.cpp's own logits for a fixed
token-id sequence, on both the F16/F32 and Q8_0 GGUFs, so boring-llm's own
CPU-only forward pass (src/math.br + src/model.br) can be diffed against
real ground truth. Token IDs are hardcoded on the Boring side too, so a
tokenizer bug can't be confused with a transformer-math bug (see the
project's README "Phase 1" note).

Usage:
    tools/reference_logits.py
"""
import numpy as np
from llama_cpp import Llama

PROMPT = "The capital of France is"


def dump(gguf_path, label):
    llm = Llama(model_path=gguf_path, n_gpu_layers=0, n_ctx=64, verbose=False, logits_all=True)
    tokens = llm.tokenize(PROMPT.encode("utf-8"), add_bos=True)
    print(f"-- {label} --")
    print(f"tokens: {tokens}")
    llm.reset()
    llm.eval(tokens)
    logits = np.array(llm.eval_logits)  # (n_tokens, n_vocab)
    for pos in range(len(tokens)):
        row = logits[pos]
        top = np.argsort(row)[::-1][:5]
        print(f"  pos {pos} (token {tokens[pos]}): top5 = {[(int(t), float(row[t])) for t in top]}")
    np.save(f"/tmp/ref_logits_{label}.npy", logits)
    llm.close()
    return tokens, logits


if __name__ == "__main__":
    dump("weights/qwen2.5-0.5b-instruct-fp16.gguf", "fp16")
    dump("weights/qwen2.5-0.5b-instruct-q8_0.gguf", "q8_0")
