#!/usr/bin/env python3
"""Independent oracle for src/openai.br + tokenizer.br's render_chat_prompt_messages.

Renders an OpenAI-style request body (messages + tools) with the GGUF's OWN
embedded chat template through real `jinja2`, then tokenizes the text with
llama.cpp's tokenizer (special tokens parsed atomically, no BOS added) and
prints `ids: 1,2,3` — the exact line `main --render-request <file>` prints, so
the two can be diffed. Normalization (content parts -> string, tool_call
`arguments` JSON string -> object) is re-implemented here from the OpenAI
wire format, independently of openai.br.

Usage (from the repo root, project venv):
    .venv/bin/python tools/oracle_render_request.py weights/model.gguf request.json [--text]
"""
import json
import sys

import jinja2
from gguf import GGUFReader
from llama_cpp import Llama


def field_str(reader, key):
    f = reader.fields[key]
    return bytes(f.parts[f.data[0]]).decode("utf-8")


def normalize(messages):
    out = []
    for m in messages:
        m = dict(m)
        c = m.get("content")
        if isinstance(c, list):
            m["content"] = "".join(p.get("text", "") for p in c)
        if isinstance(m.get("tool_calls"), list):
            calls = []
            for call in m["tool_calls"]:
                call = dict(call)
                fn = dict(call.get("function", {}))
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"])
                    except ValueError:
                        pass
                call["function"] = fn
                calls.append(call)
            m["tool_calls"] = calls
        out.append(m)
    return out


def main():
    gguf_path, req_path = sys.argv[1], sys.argv[2]
    reader = GGUFReader(gguf_path)
    template = field_str(reader, "tokenizer.chat_template")
    tokens_field = reader.fields["tokenizer.ggml.tokens"]
    tokens = [bytes(tokens_field.parts[i]).decode("utf-8") for i in tokens_field.data]
    bos_id = int(reader.fields["tokenizer.ggml.bos_token_id"].parts[-1][0]) if "tokenizer.ggml.bos_token_id" in reader.fields else None
    eos_id = int(reader.fields["tokenizer.ggml.eos_token_id"].parts[-1][0])
    body = json.load(open(req_path))

    env = jinja2.Environment()
    # HF's own `tojson` (json.dumps, ensure_ascii=False, default separators) —
    # NOT jinja2's built-in one, which sorts keys and HTML-escapes.
    env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)
    text = env.from_string(template).render(
        messages=normalize(body["messages"]),
        tools=body.get("tools", []),
        add_generation_prompt=True,
        bos_token=tokens[bos_id] if bos_id is not None else "",
        eos_token=tokens[eos_id],
    )
    if "--text" in sys.argv:
        print(text)
        return
    llm = Llama(model_path=gguf_path, vocab_only=True, verbose=False)
    ids = llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)
    print("ids: " + ",".join(str(i) for i in ids))


if __name__ == "__main__":
    main()
