#!/usr/bin/env python3
"""Protocol smoke test for `main --serve` (stdlib only, no model-quality claims).

Checks the OpenAI wire format an OpenAI-compatible client (OpenCode, the Vercel
AI SDK, curl) relies on: /v1/models, a plain chat completion, an SSE stream
(role chunk .. finish chunk .. usage chunk .. [DONE]) that really arrives token by token, error shapes (404, bad
JSON, missing messages, context exceeded), and a tool-calling round trip's
JSON shape. Generation is real, so it is slow (seconds per request); keep
`max_tokens` small.

Usage:
    tools/smoke_server.py [http://127.0.0.1:8080 [ctx_size]]   # ctx_size: the server's --ctx-size (default 4096)
"""
import http.client
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080").rstrip("/")
CTX = int(sys.argv[2]) if len(sys.argv) > 2 else 4096  # the server's --ctx-size (sizes the overflow check)
failures = []


def call(path, body=None, raw=False):
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    req = urllib.request.Request(BASE + path, data=data, method="GET" if data is None else "POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, r.headers, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read().decode()


def raw_http(request):
    """Send an intentionally malformed request that urllib would normalize."""
    u = urllib.parse.urlparse(BASE)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
    conn.sock = None
    conn.connect()
    conn.sock.sendall(request)
    response = conn.response_class(conn.sock, method="POST")
    response.begin()
    result = response.status, response.read().decode()
    conn.close()
    return result


def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else f"  -- {detail}"))
    if not cond:
        failures.append(name)


NO_THINK = {"enable_thinking": False}  # for the generic checks: a reasoning model would spend a small token budget on <think>; ignored by templates that do not read it
TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}

s, h, b = call("/v1/models")
m = json.loads(b)
check("GET /v1/models lists one model", s == 200 and m["object"] == "list" and len(m["data"]) == 1, b)
model_id = m["data"][0]["id"]
meta = m["data"][0].get("meta", {})
check("GET /v1/models exposes llama-server metadata",
      meta.get("vocab_type") in (1, 2) and meta.get("n_vocab", 0) > 0
      and meta.get("n_ctx") == CTX and meta.get("n_ctx_train", 0) > 0
      and meta.get("n_embd", 0) > 0 and meta.get("n_params", 0) > 0
      and meta.get("size", 0) > 0, b)

s, h, b = call("/v1/health")
check("GET /v1/health", s == 200 and json.loads(b).get("status") == "ok", b)
s, h, b = call("/props")
props = json.loads(b)
check("GET /props exposes llama-server diagnostics", s == 200 and props.get("model_path")
      and props.get("chat_template") and props.get("default_generation_settings", {}).get("n_ctx") == CTX, b)

count_body = {"model": model_id, "messages": [{"role": "user", "content": "Say hi."}], "chat_template_kwargs": NO_THINK}
s, h, b = call("/v1/chat/completions/input_tokens", count_body)
counted = json.loads(b)
check("chat input token counting endpoint", s == 200 and counted["object"] == "response.input_tokens" and counted["input_tokens"] > 0, b)
s, h, b = call("/v1/chat/completions/input_tokens", {
    "model": model_id,
    "messages": [{"role": "developer", "content": "Answer briefly."},
                 {"role": "user", "content": "Say hi."}],
})
check("developer role is accepted", s == 200 and json.loads(b).get("input_tokens", 0) > 0, b)

s, h, b = call("/apply-template", count_body)
applied = json.loads(b)
check("apply-template returns the rendered prompt", s == 200 and isinstance(applied.get("prompt"), str) and applied["prompt"], b)
s, h, b = call("/tokenize", {"content": applied["prompt"], "parse_special": True})
prompt_ids = json.loads(b).get("tokens", [])
check("tokenize agrees with chat input token count", s == 200 and len(prompt_ids) == counted["input_tokens"], b)
s, h, b = call("/tokenize", {"content": applied["prompt"], "parse_special": True, "with_pieces": True})
pieces = json.loads(b).get("tokens", [])
check("tokenize with_pieces returns llama-server objects", s == 200 and len(pieces) == len(prompt_ids)
      and [piece.get("id") for piece in pieces] == prompt_ids
      and all(isinstance(piece.get("piece"), (str, list)) for piece in pieces), b)
s, h, b = call("/detokenize", {"tokens": prompt_ids})
check("detokenize accepts the prompt tokens", s == 200 and isinstance(json.loads(b).get("content"), str), b)

s, h, b = call("/v1/chat/completions", {"model": model_id, "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8, "temperature": 0, "chat_template_kwargs": NO_THINK})
r = json.loads(b)
ch = r["choices"][0]
check("chat completion shape", s == 200 and r["object"] == "chat.completion" and ch["message"]["role"] == "assistant"
      and ch["finish_reason"] in ("stop", "length") and r["usage"]["total_tokens"] == r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"], b)
check("content-type is JSON", h.get("Content-Type", "").startswith("application/json"))

completion_body = {"model": model_id, "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0}
s, h, b = call("/v1/completions", completion_body)
classic = json.loads(b)
check("legacy text completion shape", s == 200 and classic.get("object") == "text_completion"
      and isinstance(classic.get("choices", [])[0].get("text"), str)
      and classic["choices"][0].get("finish_reason") in ("stop", "length"), b)
s, h, b = call("/tokenize", {"content": completion_body["prompt"], "parse_special": True})
completion_prompt_ids = json.loads(b).get("tokens", [])
s, h, b = call("/v1/completions", dict(completion_body, prompt=completion_prompt_ids))
token_classic = json.loads(b)
check("legacy text completion accepts prompt token ids", s == 200
      and token_classic["choices"][0]["text"] == classic["choices"][0]["text"], b)
s, h, b = call("/v1/completions", dict(completion_body, stream=True, stream_options={"include_usage": True}))
classic_events = [e for e in b.split("\n\n") if e.strip()]
classic_chunks = [json.loads(e[len("data: "):]) for e in classic_events[:-1]]
classic_choice_chunks = [c for c in classic_chunks if c.get("choices")]
classic_usage_chunks = [c for c in classic_chunks if not c.get("choices")]
check("legacy text completion streams", s == 200 and classic_events[-1] == "data: [DONE]"
      and all(c.get("object") == "text_completion" for c in classic_chunks)
      and "".join(c["choices"][0]["text"] for c in classic_choice_chunks) == classic["choices"][0]["text"], b)
check("legacy text completion stream includes usage", len(classic_usage_chunks) == 1
      and classic_usage_chunks[0]["usage"] == classic["usage"], b)

native_body = {"prompt": completion_body["prompt"], "n_predict": 8, "temperature": 0}
s, h, b = call("/completion", native_body)
native = json.loads(b)
check("llama native completion shape", s == 200 and native.get("content") == classic["choices"][0]["text"]
      and native.get("tokens_predicted") == 8 and len(native.get("tokens", [])) == 8
      and native.get("stop_type") in ("eos", "word", "limit"), b)
s, h, b = call("/completion", {"prompt": "cache this prompt", "n_predict": 0})
native_zero = json.loads(b)
check("llama native zero-token completion", s == 200 and native_zero.get("content") == ""
      and native_zero.get("tokens") == [] and native_zero.get("tokens_predicted") == 0, b)
s, h, b = call("/completion", dict(native_body, prompt=completion_prompt_ids[:2] + [" of France is"]))
mixed_native = json.loads(b)
check("llama native completion accepts mixed token and text prompts", s == 200
      and isinstance(mixed_native.get("content"), str) and mixed_native.get("tokens_predicted") == 8, b)
s, h, b = call("/completion", dict(native_body, prompt={"prompt_string": completion_body["prompt"]}))
object_native = json.loads(b)
check("llama native completion accepts prompt_string objects", s == 200
      and object_native.get("content") == native.get("content"), b)
s, h, b = call("/completions", dict(native_body, stream=True))
native_events = [json.loads(e[len("data: "):]) for e in b.split("\n\n") if e.strip()]
check("llama native completion streams", s == 200 and native_events[-1].get("stop") is True
      and "".join(e.get("content", "") for e in native_events) == native["content"]
      and sum(len(e.get("tokens", [])) for e in native_events) == len(native["tokens"]), b)

s, h, b = call("/v1/chat/completions", {"model": model_id, "stream": True, "stream_options": {"include_usage": True},
                                          "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8, "temperature": 0, "chat_template_kwargs": NO_THINK})
events = [e for e in b.split("\n\n") if e.strip()]
check("SSE content-type", h.get("Content-Type", "").startswith("text/event-stream"), str(h.get("Content-Type")))
check("SSE ends with [DONE]", events[-1] == "data: [DONE]", events[-1])
chunks = [json.loads(e[len("data: "):]) for e in events[:-1]]
check("SSE first chunk carries role", chunks[0]["choices"][0]["delta"].get("role") == "assistant")
fin = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
check("SSE has exactly one finish chunk", len(fin) == 1)
check("SSE usage chunk has empty choices", chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] > 0)
text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
check("SSE deltas rebuild the non-stream text", text == ch["message"]["content"], f"{text!r} vs {ch['message']['content']!r}")

s, h, b = call("/v1/chat/completions", {"model": model_id, "temperature": 0, "max_tokens": 200, "chat_template_kwargs": NO_THINK,
        "messages": [{"role": "system", "content": "You are a helpful assistant. Always use the provided tools when they apply."},
                     {"role": "user", "content": "Call get_weather for the city Paris."}], "tools": [TOOL]})
r = json.loads(b)
msg = r["choices"][0]["message"]
if msg.get("tool_calls"):
    tc = msg["tool_calls"][0]
    args = json.loads(tc["function"]["arguments"])
    check("tool call shape", r["choices"][0]["finish_reason"] == "tool_calls" and tc["type"] == "function"
          and tc["function"]["name"] == "get_weather" and isinstance(args, dict) and tc["id"], b)
else:
    print("SKIP  tool call shape (this model answered in plain text; not a protocol failure)")

# A tool-result turn must be accepted (assistant tool_calls with string arguments + tool message).
s, h, b = call("/v1/chat/completions", {"model": model_id, "temperature": 0, "max_tokens": 16, "tools": [TOOL],
        "messages": [{"role": "user", "content": "Weather in Paris?"},
                     {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
                     {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 21C"}]})
check("tool-result turn accepted", s == 200 and json.loads(b)["choices"][0]["message"]["role"] == "assistant", b)

s, h, b = call("/nope")
check("404 JSON error", s == 404 and json.loads(b)["error"]["code"] == "not_found", b)
s, h, b = call("/v1/chat/completions", b"{not json")
check("400 on invalid JSON", s == 400 and json.loads(b)["error"]["code"] == "invalid_json", b)
s, h, b = call("/v1/chat/completions", b'{"messages":[{"role":"user","content":"bad\\q"}]}')
check("400 on invalid JSON escape", s == 400 and json.loads(b)["error"]["code"] == "invalid_json", b)
s, h, b = call("/v1/chat/completions", b'{"messages":[{"role":"user","content":"\\ud800"}]}')
check("400 on invalid JSON surrogate", s == 400 and json.loads(b)["error"]["code"] == "invalid_json", b)
s, h, b = call("/v1/chat/completions", {"model": "x"})
check("400 on missing messages", s == 400 and json.loads(b)["error"]["code"] == "missing_messages", b)
s, h, b = call("/v1/chat/completions", {"model": model_id, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]})
check("400 on unsupported message content", s == 400 and json.loads(b)["error"]["code"] == "invalid_request", b)
s, h, b = call("/v1/chat/completions", {"model": model_id, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 0})
check("400 on invalid generation parameter", s == 400 and json.loads(b)["error"]["code"] == "invalid_parameter", b)
s, h, b = call("/v1/chat/completions", {"model": model_id, "messages": [{"role": "user", "content": "hi"}], "n": 2})
check("400 on unsupported multiple choices", s == 400 and json.loads(b)["error"]["code"] == "invalid_parameter", b)
s, h, b = call("/v1/chat/completions", {"model": model_id, "messages": [{"role": "user", "content": "json"}],
                                         "response_format": {"type": "json_object"}, "max_tokens": 1})
check("400 on unsupported structured response format", s == 400 and json.loads(b)["error"]["code"] == "invalid_parameter"
      and "constrained decoding" in json.loads(b)["error"]["message"], b)
s, h, b = call("/v1/chat/completions", {"model": model_id, "messages": [{"role": "user", "content": "hi"}],
                                         "logprobs": True, "top_logprobs": 3, "max_tokens": 1})
check("400 on unsupported requested logprobs", s == 400 and json.loads(b)["error"]["code"] == "invalid_parameter"
      and "logprobs" in json.loads(b)["error"]["message"], b)
s, h, b = call("/v1/completions", {"model": model_id, "prompt": [0, "bad"], "max_tokens": 1})
check("400 on invalid completion prompt tokens", s == 400 and json.loads(b)["error"]["code"] == "invalid_prompt", b)
s, h, b = call("/v1/completions", {"model": model_id, "prompt": [1 << 30], "max_tokens": 1})
check("400 on out-of-vocabulary completion token", s == 400 and json.loads(b)["error"]["code"] == "invalid_prompt_token", b)
s, h, b = call("/completion", {"prompt": {"prompt_string": "look", "multimodal_data": ["AA=="]}, "n_predict": 1})
check("400 on unsupported multimodal native prompt", s == 400 and json.loads(b)["error"]["code"] == "unsupported_multimodal_prompt", b)
status, b = raw_http(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: nope\r\n\r\n")
check("400 on invalid Content-Length", status == 400 and json.loads(b)["error"]["code"] == "malformed_http_request", b)
status, b = raw_http(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n")
check("400 on unsupported chunked request", status == 400 and json.loads(b)["error"]["code"] == "malformed_http_request", b)
s, h, b = call("/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": "word " * (CTX * 3 // 2)}]})
check("400 context_length_exceeded", s == 400 and json.loads(b)["error"]["code"] == "context_length_exceeded", b)
e = json.loads(b).get("error", {})
check("context error names the sizes and how to fix it", "maximum context length" in e["message"] and "--ctx-size" in e["message"]
      and e.get("n_prompt_tokens", 0) >= e.get("n_ctx", 1 << 60) > 0, b)

# ── Incremental streaming ────────────────────────────────────────────────

def stream(body):
    """POST a streaming request; returns (headers, [(seconds since send, parsed event or '[DONE]')])."""
    u = urllib.parse.urlparse(BASE)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
    t0 = time.time()
    conn.request("POST", "/v1/chat/completions", json.dumps(dict(body, stream=True)).encode(), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    out = []
    while True:
        line = resp.fp.readline()
        if not line:
            break
        line = line.decode().strip()
        if line.startswith("data: "):
            payload = line[len("data: "):]
            out.append((time.time() - t0, "[DONE]" if payload == "[DONE]" else json.loads(payload)))
            if payload == "[DONE]":
                break
    headers = resp.headers
    conn.close()
    return headers, out


def content_of(events):
    return "".join(e["choices"][0]["delta"].get("content", "") for _, e in events if e != "[DONE]" and e["choices"])


STORY = {"model": model_id, "temperature": 0, "max_tokens": 48, "chat_template_kwargs": NO_THINK,
         "messages": [{"role": "user", "content": "Count from 1 to 40, separated by commas."}]}
hdr, ev = stream(STORY)
contents = [(t, e) for t, e in ev if e != "[DONE]" and e["choices"] and e["choices"][0]["delta"].get("content")]
t_done = ev[-1][0]
check("stream has no Content-Length (body ends at close)", hdr.get("Content-Length") is None, str(hdr.get("Content-Length")))
check("stream sends many content events, not one", len(contents) >= 5, f"{len(contents)} content events")
check("first content arrives well before the end", bool(contents) and contents[0][0] < 0.6 * t_done,
      f"first {contents[0][0]:.2f}s of {t_done:.2f}s" if contents else "no content")
s2, h2, b2 = call("/v1/chat/completions", STORY)
base = json.loads(b2)["choices"][0]
check("streamed text equals the non-stream text", content_of(ev) == base["message"]["content"], f"{content_of(ev)!r} vs {base['message']['content']!r}")
fin = [e for _, e in ev if e != "[DONE]" and e["choices"] and e["choices"][0].get("finish_reason")]
check("streamed finish_reason equals the non-stream one", len(fin) == 1 and fin[0]["choices"][0]["finish_reason"] == base["finish_reason"],
      f"{[f['choices'][0]['finish_reason'] for f in fin]} vs {base['finish_reason']}")

# A stop string ends generation early (no wasted tokens) and is never sent.
full = base["message"]["content"]
if len(full) >= 12:
    stop = full[len(full) // 2: len(full) // 2 + 3]
    if stop.strip():
        s3, h3, b3 = call("/v1/chat/completions", dict(STORY, stop=[stop]))
        r3 = json.loads(b3)
        want = full.split(stop)[0].strip()
        check("stop string cuts the reply", r3["choices"][0]["message"]["content"] == want and r3["choices"][0]["finish_reason"] == "stop",
              f"{r3['choices'][0]['message']['content']!r} vs {want!r}")
        check("stop string ends generation early", r3["usage"]["completion_tokens"] < json.loads(b2)["usage"]["completion_tokens"],
              f"{r3['usage']['completion_tokens']} vs {json.loads(b2)['usage']['completion_tokens']}")
        hdr4, ev4 = stream(dict(STORY, stop=[stop]))
        txt4 = content_of(ev4)
        check("streamed reply with a stop string never contains it", stop not in txt4 and txt4 == want, f"{txt4!r} vs {want!r}")

# A tool call streams as the same tool_calls the non-stream reply carries.
TQ = {"model": model_id, "temperature": 0, "max_tokens": 400, "chat_template_kwargs": NO_THINK, "tools": [TOOL],
      "messages": [{"role": "user", "content": "What's the weather in Paris?"}]}
s5, h5, b5 = call("/v1/chat/completions", TQ)
msg5 = json.loads(b5)["choices"][0]["message"]
if msg5.get("tool_calls"):
    hdr6, ev6 = stream(TQ)
    deltas = [e["choices"][0]["delta"] for _, e in ev6 if e != "[DONE]" and e["choices"]]
    tcs = [tc for d in deltas for tc in d.get("tool_calls", [])]
    check("streamed tool call equals the non-stream one",
          len(tcs) == len(msg5["tool_calls"]) and tcs[0]["function"]["name"] == msg5["tool_calls"][0]["function"]["name"]
          and tcs[0]["function"]["arguments"] == msg5["tool_calls"][0]["function"]["arguments"], f"{tcs} vs {msg5['tool_calls']}")
    check("streamed tool call finishes with tool_calls", ev6[-2][1]["choices"][0].get("finish_reason") == "tool_calls" or
          any(e != "[DONE]" and e["choices"] and e["choices"][0].get("finish_reason") == "tool_calls" for _, e in ev6))
    check("tool call syntax never leaks into streamed content", "<tool_call>" not in content_of(ev6), content_of(ev6))
else:
    print("SKIP  streamed tool call (this model answered in plain text; not a protocol failure)")

# tool_choice "none": the tool definitions stay out of the prompt, so the reply carries no tool call.
none_body = dict(TQ, tool_choice="none")
s7, h7, b7 = call("/v1/chat/completions", none_body)
r7 = json.loads(b7)
check("tool_choice none: no tool call", s7 == 200 and not r7["choices"][0]["message"].get("tool_calls")
      and r7["choices"][0]["finish_reason"] != "tool_calls", b7)
if msg5.get("tool_calls"):
    check("tool_choice none: the prompt is shorter (no tool definitions)",
          r7["usage"]["prompt_tokens"] < json.loads(b5)["usage"]["prompt_tokens"])

# Sampling fields: accepted, and a strong frequency penalty changes a repetitive greedy reply.
REP = {"model": model_id, "temperature": 0, "max_tokens": 40, "chat_template_kwargs": NO_THINK,
       "messages": [{"role": "user", "content": "Repeat the word 'echo' twenty times, separated by spaces."}]}
s8, h8, b8 = call("/v1/chat/completions", REP)
base_text = json.loads(b8)["choices"][0]["message"]["content"]
s9, h9, b9 = call("/v1/chat/completions", dict(REP, frequency_penalty=2.0, presence_penalty=1.0,
                                                 top_k=5, min_p=0.05, repeat_penalty=1.1, logit_bias={"0": 0}))
check("sampling parameters, penalties and logit_bias are accepted", s9 == 200, b9)
if s9 == 200:
    check("a strong frequency penalty changes a repetitive greedy reply",
          json.loads(b9)["choices"][0]["message"]["content"] != base_text, base_text)
s10, h10, b10 = call("/v1/chat/completions", dict(REP, frequency_penalty=0, presence_penalty=0))
check("zero penalties leave the greedy reply unchanged", s10 == 200 and json.loads(b10)["choices"][0]["message"]["content"] == base_text, b10)

# Reasoning models (Qwen3): <think>...</think> goes to reasoning_content, never into content, streamed or not.
RQ = {"model": model_id, "temperature": 0, "max_tokens": 300,
      "messages": [{"role": "user", "content": "What is 17 + 25? Answer with just the number."}]}
s11, h11, b11 = call("/v1/chat/completions", RQ)
m11 = json.loads(b11)["choices"][0]["message"]
check("no <think> markers in content", "<think>" not in m11["content"] and "</think>" not in m11["content"], m11["content"])
if m11.get("reasoning_content"):
    check("reasoning_content is the text between the markers", "<think>" not in m11["reasoning_content"]
          and "</think>" not in m11["reasoning_content"] and not m11["reasoning_content"].startswith("\n"), m11["reasoning_content"][:80])
    hdr12, ev12 = stream(RQ)
    rdeltas = [e["choices"][0]["delta"].get("reasoning_content", "") for _, e in ev12 if e != "[DONE]" and e["choices"]]
    check("streamed reasoning_content equals the non-stream one", "".join(rdeltas) == m11["reasoning_content"],
          f"{''.join(rdeltas)[:60]!r} vs {m11['reasoning_content'][:60]!r}")
    check("streamed content equals the non-stream one and has no markers", content_of(ev12) == m11["content"], f"{content_of(ev12)!r} vs {m11['content']!r}")
    check("reasoning arrives before the answer", next((i for i, d in enumerate(rdeltas) if d), 99) < 99)
    s13, h13, b13 = call("/v1/chat/completions", dict(RQ, chat_template_kwargs={"enable_thinking": False}))
    m13 = json.loads(b13)["choices"][0]["message"]
    check("enable_thinking false: no reasoning, a direct answer", not m13.get("reasoning_content") and m13["content"].strip() != "", b13)
else:
    print("SKIP  reasoning_content (this model does not think; not a protocol failure)")

print("\n%d failure(s)" % len(failures))
sys.exit(1 if failures else 0)
