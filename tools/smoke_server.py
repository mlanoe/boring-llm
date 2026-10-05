#!/usr/bin/env python3
"""Protocol smoke test for `main --serve` (stdlib only, no model-quality claims).

Checks the OpenAI wire format an OpenAI-compatible client (OpenCode, the Vercel
AI SDK, curl) relies on: /v1/models, a plain chat completion, an SSE stream
(role chunk .. finish chunk .. usage chunk .. [DONE]), error shapes (404, bad
JSON, missing messages, context exceeded), and a tool-calling round trip's
JSON shape. Generation is real, so it is slow (seconds per request); keep
`max_tokens` small.

Usage:
    tools/smoke_server.py [http://127.0.0.1:8080]
"""
import json
import sys
import urllib.error
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080").rstrip("/")
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


def check(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + ("" if cond else f"  -- {detail}"))
    if not cond:
        failures.append(name)


TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}

s, h, b = call("/v1/models")
m = json.loads(b)
check("GET /v1/models lists one model", s == 200 and m["object"] == "list" and len(m["data"]) == 1, b)
model_id = m["data"][0]["id"]

s, h, b = call("/v1/chat/completions", {"model": model_id, "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8, "temperature": 0})
r = json.loads(b)
ch = r["choices"][0]
check("chat completion shape", s == 200 and r["object"] == "chat.completion" and ch["message"]["role"] == "assistant"
      and ch["finish_reason"] in ("stop", "length") and r["usage"]["total_tokens"] == r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"], b)
check("content-type is JSON", h.get("Content-Type", "").startswith("application/json"))

s, h, b = call("/v1/chat/completions", {"model": model_id, "stream": True, "stream_options": {"include_usage": True},
                                          "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8, "temperature": 0})
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

s, h, b = call("/v1/chat/completions", {"model": model_id, "temperature": 0, "max_tokens": 60,
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
s, h, b = call("/v1/chat/completions", {"model": "x"})
check("400 on missing messages", s == 400 and json.loads(b)["error"]["code"] == "missing_messages", b)
s, h, b = call("/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": "word " * 6000}]})
check("400 context_length_exceeded", s == 400 and json.loads(b)["error"]["code"] == "context_length_exceeded", b)

print("\n%d failure(s)" % len(failures))
sys.exit(1 if failures else 0)
