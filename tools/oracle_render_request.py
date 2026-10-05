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
    # compare every request of a directory with a build, per model (exit 1 on any difference):
    .venv/bin/python tools/oracle_render_request.py weights/model.gguf test/fixtures/requests/*.json --binary ./main_metal

A request may carry `chat_template_kwargs` (e.g. {"enable_thinking": false} for Qwen3).
"""
import json
import os
import subprocess
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
        elif not isinstance(c, str):
            m["content"] = ""  # content: null or absent (assistant messages that only carry tool calls)
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
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    binary = None
    if "--binary" in sys.argv:
        binary = sys.argv[sys.argv.index("--binary") + 1]
        args.remove(binary)
    gguf_path, req_paths = args[0], args[1:]
    reader = GGUFReader(gguf_path)
    template = field_str(reader, "tokenizer.chat_template")
    tokens_field = reader.fields["tokenizer.ggml.tokens"]
    tokens = [bytes(tokens_field.parts[i]).decode("utf-8") for i in tokens_field.data]
    bos_id = int(reader.fields["tokenizer.ggml.bos_token_id"].parts[-1][0]) if "tokenizer.ggml.bos_token_id" in reader.fields else None
    eos_id = int(reader.fields["tokenizer.ggml.eos_token_id"].parts[-1][0])

    env = jinja2.Environment()
    # HF's own `tojson` (json.dumps, ensure_ascii=False, default separators) —
    # NOT jinja2's built-in one, which sorts keys and HTML-escapes.
    env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)
    llm = None
    failed = 0
    for req_path in req_paths:
        body = json.load(open(req_path))
        text = env.from_string(template).render(
            messages=normalize(body["messages"]),
            tools=body.get("tools", []),
            add_generation_prompt=True,
            bos_token=tokens[bos_id] if bos_id is not None else "",
            eos_token=tokens[eos_id],
            **body.get("chat_template_kwargs", {}),
        )
        if "--text" in sys.argv:
            print(text)
            continue
        if llm is None:
            llm = Llama(model_path=gguf_path, vocab_only=True, verbose=False)
        ids = "ids: " + ",".join(str(i) for i in llm.tokenize(text.encode("utf-8"), add_bos=False, special=True))
        if binary is None:
            print(ids)
            continue
        run = subprocess.run([binary, "--gguf", gguf_path, "--render-request", req_path],
                             capture_output=True, text=True, stdin=subprocess.DEVNULL)
        ours = next((l for l in run.stdout.splitlines() if l.startswith("ids: ")), "")
        same = ours == ids
        failed += not same
        n = ids.count(",") + 1
        print(f"{os.path.basename(gguf_path)} {os.path.basename(req_path)}: " + (f"identical ({n} tokens)" if same else "DIFFERENT"))
        if not same:
            print("  oracle: " + ids[:160])
            print("  binary: " + (ours[:160] or (run.stdout + run.stderr)[-300:]))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
