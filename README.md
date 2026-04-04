# LLM Tunnel

A minimal LLM inference server (`server.py`) that exposes both an Anthropic-compatible (`/v1/messages`) and OpenAI-compatible (`/v1/chat/completions`) API. Designed to run on a remote GPU host (e.g. Vast.ai) and stream tokens privately to your local machine via an SSH tunnel.

No conversation content is ever logged — request bodies are never captured by FastAPI, and `--no-access-log` suppresses uvicorn's HTTP request logs entirely.

## Quick start (local)

```bash
pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8080 --no-access-log
```

Default model: `google/gemma-3-270m-it`. Downloaded automatically from HuggingFace on first run.

Test it:

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 64}' \
  | python3 -m json.tool
```

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

## Connecting a chat UI

The server speaks both OpenAI and Anthropic API formats, so most frontends work.

**LibreChat (recommended)** — create `librechat.yaml` in your LibreChat project folder:

```yaml
version: 1.3.5
endpoints:
  custom:
    - name: "Local Model"
      apiKey: "dummy"
      baseURL: "http://host.docker.internal:8080/v1"
      models:
        default: ["google/gemma-3-270m-it"]
        fetch: false
      titleConvo: false
```

Then mount it via `docker-compose.override.yml`:

```yaml
services:
  api:
    volumes:
      - type: bind
        source: ./librechat.yaml
        target: /app/librechat.yaml
```

Restart: `docker compose down && docker compose up -d`

**Chatbox** — Settings → AI Provider → OpenAI API → API Host: `http://localhost:8080`, API Key: `dummy`

**Claude Code / aider (Anthropic SDK clients)**:

```bash
export ANTHROPIC_API_KEY=dummy
export ANTHROPIC_BASE_URL=http://localhost:8080
```

## Models

Set `MODEL_ID` to any HuggingFace causal LM. Models with a `chat_template` in their tokenizer config work best (Gemma, Llama, Qwen, Mistral, etc.). Base models without a chat template fall back to a plain `Role: content` prompt format.

## Architecture

Everything lives in `server.py`:

- **`load_model`** — loads tokenizer + model at startup into bfloat16; auto-selects CUDA, MPS (Apple Silicon), or CPU
- **`_build_prompt`** — applies the model's chat template (or plain fallback) to the message list
- **`POST /v1/messages`** — Anthropic-compatible endpoint, streaming and non-streaming
- **`POST /v1/chat/completions`** — OpenAI-compatible endpoint, streaming and non-streaming
- **`GET /v1/models`** — returns the loaded model ID
