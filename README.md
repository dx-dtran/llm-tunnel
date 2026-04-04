# LLM Tunnel

A minimal Anthropic-API-compatible LLM inference server (`server.py`) meant to run on a remote GPU host (e.g. Vast.ai).

It streams tokens privately to your local machine. No conversation content is ever logged — request bodies are never captured by FastAPI, and `--no-access-log` suppresses uvicorn's HTTP request logs entirely.

It exposes `/v1/messages` with streaming SSE so that local tools using the Anthropic SDK (Claude Code, aider, etc.) can point at it via an SSH tunnel.

## Server setup (on the GPU host)

```bash
pip install -r requirements.txt
MODEL_ID=google/gemma-3-4b-it uvicorn server:app --host 127.0.0.1 --port 8080 --no-access-log
```

The server binds to localhost only — access is exclusively through the SSH tunnel.

## SSH tunnel (from local machine)

```bash
# Vast.ai example — adjust port/IP from the instance dashboard
ssh -L 8080:localhost:8080 -p 12345 root@123.45.67.89
```

## Local client config

```bash
export ANTHROPIC_API_KEY=dummy   # SDK requires a value; server ignores it
export ANTHROPIC_BASE_URL=http://localhost:8080
```

Then use Claude Code, aider, or any Anthropic SDK client normally — they'll hit the local model.

## Models

Set `MODEL_ID` to any HuggingFace causal LM. Models with a `chat_template` in their tokenizer config work best (Gemma, Llama, Qwen, Mistral, etc.). Base models without a chat template fall back to a plain `Role: content` prompt format.

## Architecture

Everything lives in `server.py`:

- **`load_model`** — loads tokenizer + model at startup into bfloat16 on CUDA
- **`build_prompt`** — converts Anthropic `messages` + `system` to the model's chat template (or fallback)
- **`POST /v1/messages`** — handles both streaming and non-streaming; streaming uses `TextIteratorStreamer` in a daemon thread with `await asyncio.sleep(0)` between tokens to keep the event loop unblocked
- **`GET /v1/models`** — returns the loaded model ID; some harnesses call this on startup