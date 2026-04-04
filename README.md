# LLM Tunnel

Chat with open-weight LLMs on your own remote GPUs.

This is a small LLM inference server that exposes both Anthropic and OpenAI compatible APIs. It's designed to run on a remote GPU host (e.g. Vast.ai) and stream tokens privately to your local machine via an SSH tunnel.

You can chat with your remote LLM via a GUI that supports custom servers such as LibreChat.

No conversation content is ever logged. Request bodies are never captured by FastAPI, and `--no-access-log` suppresses uvicorn's HTTP request logs entirely.

## Quick start (local)

```bash
pip install -r requirements.txt
```

```bash
uvicorn server:app --host 127.0.0.1 --port 8080 --no-access-log
```

Default model: `google/gemma-3-270m-it`. Downloaded automatically from HuggingFace on first run.

Test it:

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Write a short story."}], "max_tokens": 256}' \
  | python3 -m json.tool
```

---

## Vast.ai setup

### 1. Rent a GPU instance

On the [Vast.ai console](https://vast.ai), rent any CUDA instance. Note the **SSH port** and **IP address** from the instance dashboard.

### 2. SSH into the instance

```bash
ssh -p <PORT> root@<IP>
```

### 3. Install dependencies and start the server

```bash
pip install -r requirements.txt
MODEL_ID=google/gemma-3-4b-it uvicorn server:app --host 127.0.0.1 --port 8080 --no-access-log
```

The server binds to `127.0.0.1` only — it is not reachable from the public internet, only via the SSH tunnel.

Swap `MODEL_ID` for any HuggingFace model ID. The model is downloaded automatically on first run.

### 4. Open the SSH tunnel (on your local machine)

In a separate terminal on your local machine:

```bash
ssh -L 8080:localhost:8080 -p <PORT> root@<IP>
```

This forwards `localhost:8080` on your machine to the server running on the Vast.ai instance. Keep this terminal open while you use the server.

---

## LibreChat setup

LibreChat runs locally in Docker and connects to the server through the SSH tunnel. From its perspective the server is always at `host.docker.internal:8080` regardless of whether it's running locally or on Vast.ai — the tunnel makes them identical.

### 1. Get LibreChat

```bash
git clone https://github.com/danny-avila/LibreChat.git
cd LibreChat
cp .env.example .env
```

### 2. Create `librechat.yaml`

In the LibreChat project folder, create `librechat.yaml`:

```yaml
version: 1.3.5
endpoints:
  custom:
    - name: "Local Model"
      apiKey: "dummy"
      baseURL: "http://host.docker.internal:8080/v1"
      models:
        default: ["google/gemma-3-4b-it"]
        fetch: false
      titleConvo: false
```

Update the model name in `default` to match whatever `MODEL_ID` you started the server with.

### 3. Create `docker-compose.override.yml`

In the same folder, create `docker-compose.override.yml` to mount the config:

```yaml
services:
  api:
    volumes:
      - type: bind
        source: ./librechat.yaml
        target: /app/librechat.yaml
```

### 4. Start LibreChat

```bash
docker compose up -d
```

Open [http://localhost:3080](http://localhost:3080), create an account, and the model will appear in the model selector under "Local Model".

### Switching between local and Vast.ai

No changes needed in LibreChat or its config. Just make sure the SSH tunnel is running when using Vast.ai. When testing locally, start `server.py` on your machine and the same `host.docker.internal:8080` URL works automatically.

---

## Other clients

**Chatbox** — Settings → AI Provider → OpenAI API → API Host: `http://localhost:8080`, API Key: `dummy`

**Claude Code / aider (Anthropic SDK)**:

```bash
export ANTHROPIC_API_KEY=dummy
export ANTHROPIC_BASE_URL=http://localhost:8080
```

---

## Models

Set `MODEL_ID` to any HuggingFace causal LM. Models with a `chat_template` in their tokenizer config work best (Gemma, Llama, Qwen, Mistral, etc.). Base models without a chat template fall back to a plain `Role: content` prompt format.

## Architecture

Everything lives in `server.py`:

- **`load_model`** — loads tokenizer + model at startup into bfloat16; auto-selects CUDA, MPS (Apple Silicon), or CPU
- **`_build_prompt`** — applies the model's chat template (or plain fallback) to the message list
- **`POST /v1/messages`** — Anthropic-compatible endpoint, streaming and non-streaming
- **`POST /v1/chat/completions`** — OpenAI-compatible endpoint, streaming and non-streaming
- **`GET /v1/models`** — returns the loaded model ID