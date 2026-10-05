#!/bin/bash
# Runs a real OpenCode (https://opencode.ai) against `main --serve`, fully isolated:
# OpenCode's config, data and cache live in a temporary directory (your own
# ~/.config/opencode, ~/.local/share/opencode and ~/.cache/opencode are never read or
# written), the project it works in is a throwaway directory, and permissions are left
# at OpenCode's defaults (never `--auto`: a small model calls tools at random).
#
# Set OPENCODE_TRIAL_PROMPT_CACHE=<dir> to start the server with `--prompt-cache <dir>`: a second
# trial against the same directory restores the first one's KV cache from disk instead of
# prefilling OpenCode's ~7.5k-token system prompt again.
#
# First use downloads OpenCode's provider package `@ai-sdk/openai-compatible` (~400 KB,
# from the npm registry) into that temporary cache.
#
# Usage (repo root, after building a `main` binary and with a GGUF in weights/):
#     tools/opencode_trial.sh <main-binary> <model.gguf> "<prompt>" [port]
# e.g. tools/opencode_trial.sh src/main_metal/target/release/main \
#          weights/qwen2.5-0.5b-instruct-q8_0.gguf "Read math_utils.py and explain it."
set -e
# The whole script is one braced block that bash parses before running anything: editing this
# file while a trial is running cannot make the running copy execute the wrong lines.
{
BIN=$1; GGUF=$2; PROMPT=$3; PORT=${4:-8765}
[ -n "$BIN" ] && [ -n "$GGUF" ] && [ -n "$PROMPT" ] || { sed -n '2,15p' "$0"; exit 2; }
command -v opencode >/dev/null || { echo "opencode not found in PATH"; exit 1; }
if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then echo "port $PORT is already in use; pass another port"; exit 1; fi
T=$(mktemp -d /tmp/boring-llm-opencode.XXXXXX)
mkdir -p "$T/config/opencode" "$T/data" "$T/cache"
# OpenCode puts the working directory's path in its system prompt, so a prompt cache can only
# match a later trial if the directory has the same path: OPENCODE_TRIAL_WORK=<dir> pins it
# (its two demo files are rewritten; nothing is deleted).
if [ -n "$OPENCODE_TRIAL_WORK" ]; then W=$OPENCODE_TRIAL_WORK; else W=$T/work; fi
mkdir -p "$W"
printf 'def add(a, b):\n    return a + b\n' > "$W/math_utils.py"
printf '# demo project\n' > "$W/README.md"
(cd "$W" && git init -q .)
EXTRA=()
[ -n "$OPENCODE_TRIAL_PROMPT_CACHE" ] && EXTRA=(--prompt-cache "$OPENCODE_TRIAL_PROMPT_CACHE")
"$BIN" --gguf "$GGUF" --serve --port "$PORT" --ctx-size 16384 --max-tokens 1024 "${EXTRA[@]}" > "$T/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null; echo "server log: $T/server.log"' EXIT
for i in $(seq 1 240); do curl -s "localhost:$PORT/health" 2>/dev/null | grep -q ok && break; sleep 1; done
# The served model id (the GGUF's general.name unless --model-name is given) goes into the config.
MODEL_ID=$(curl -s "localhost:$PORT/v1/models" | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')
[ -n "$MODEL_ID" ] || { echo "could not read the served model id"; exit 1; }
cat > "$T/config/opencode/opencode.json" <<JSON
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "boring-llm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "boring-llm (local)",
      "options": { "baseURL": "http://127.0.0.1:$PORT/v1" },
      "models": { "$MODEL_ID": { "name": "$MODEL_ID", "tool_call": true,
                                   "limit": { "context": 16384, "output": 1024 } } }
    }
  },
  "autoupdate": false,
  "share": "disabled"
}
JSON
# stdin is closed on purpose: without a terminal `opencode run` waits for its standard
# input to end before it sends the message.
(cd "$W" && XDG_CONFIG_HOME="$T/config" XDG_DATA_HOME="$T/data" XDG_CACHE_HOME="$T/cache" \
    timeout "${OPENCODE_TRIAL_TIMEOUT:-300}" opencode run -m "boring-llm/$MODEL_ID" "$PROMPT" < /dev/null) || echo "(opencode exited with status $?)"
echo "--- requests the server handled:"; grep "chat:" "$T/server.log" | sed 's/\[serve\] chat: //' | sed 's/, prefill.*//'
exit $?
}
