# LLM Tunnel

Privately chat with open-weight LLMs on remote GPUs.

This is a small LLM inference server powered by [vLLM](https://github.com/vllm-project/vllm) that exposes both Anthropic and OpenAI compatible APIs. It's designed to run on a remote GPU host (e.g. Vast.ai) and stream tokens privately to your local machine via an SSH tunnel.

You can chat with your remote LLM via a GUI that supports custom servers such as LibreChat.

No conversation content is ever logged. vLLM request/stats logging is disabled, and `--no-access-log` suppresses uvicorn's HTTP request logs entirely.

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

On the Vast.ai console, rent any CUDA instance. Note the **SSH port** and **IP address** from the instance dashboard.

### 2. SSH into the instance

```bash
ssh -p <PORT> root@<IP>
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

```bash
chmod +x run.sh
```

### 4. Start the server

```bash
./run.sh gpt-oss-20b
```

The server binds to `127.0.0.1` only — it is not reachable from the public internet, only via the SSH tunnel.

Models are downloaded automatically on first run and cached for all subsequent runs. Run `./run.sh` with no arguments to see all available models.

**Available models:**

| Alias | Model | VRAM (approx) | Notes |
|-------|-------|---------------|-------|
| `gemma3-270m` | google/gemma-3-270m-it | ~1 GB | tiny, for local testing |
| `gemma4-e2b` | google/gemma-4-E2B-it | ~4 GB | 2B multimodal, bf16 |
| `gemma4-e4b` | google/gemma-4-E4B-it | ~8 GB | 4B multimodal, bf16 |
| `gemma4-26b` | google/gemma-4-26B-A4B-it | ~52 GB | 26B MoE (4B active), bf16 |
| `gemma4-26b-fp8` | protoLabsAI/gemma-4-26B-A4B-it-FP8 | ~26 GB | FP8 pre-quantized, no quality loss |
| `gemma4-31b` | google/gemma-4-31B-it | ~62 GB | 31B dense, bf16 |
| `gemma4-31b-nvfp4` | nvidia/Gemma-4-31B-IT-NVFP4 | ~16 GB | NVFP4, Hopper/Blackwell GPU required |
| `gpt-oss-20b` | openai/gpt-oss-20b | ~16 GB | natively MXFP4 quantized |
| `gpt-oss-120b` | openai/gpt-oss-120b | ~60 GB | natively MXFP4 quantized |
| `gpt-oss-120b-awq` | twhitworth/gpt-oss-120b-awq-w4a16 | ~34 GB | AWQ W4A16, works on non-Hopper GPUs |
| `qwq-32b` | Qwen/QwQ-32B | ~64 GB | thinking model, bf16 |
| `qwq-32b-fp8` | modelscope/QwQ-32B-FP8 | ~32 GB | thinking model, FP8 pre-quantized |
| `qwq-32b-awq` | Qwen/QwQ-32B-AWQ | ~20 GB | thinking model, AWQ W4A16 |
| `deepseek-r1-qwen-32b` | deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | ~64 GB | thinking model, 32B distill, bf16 |
| `qwen3-32b` | Qwen/Qwen3-32B | ~64 GB | thinking model, bf16 |

Pre-quantized models (FP8, NVFP4, MXFP4, AWQ) are detected automatically by vLLM — no extra flags needed. The gpt-oss models ship as MXFP4 natively; that's already their fast format.

**Single RTX 3090 (24 GB):** `gemma4-e2b`, `gemma4-e4b`, `gpt-oss-20b`, `qwq-32b-awq` fit comfortably. For 40–80 GB GPUs (A100/H100): `gemma4-26b-fp8`, `gpt-oss-120b`, `gpt-oss-120b-awq`, `qwq-32b-fp8`. `gemma4-31b-nvfp4` requires a Hopper (H100) or Blackwell GPU. Full-precision 32B thinking models need ~64 GB.

### 5. Open the SSH tunnel (on your local machine)

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

## Thinking models

Reasoning models (QwQ-32B, DeepSeek-R1, Qwen3) emit `<think>...</think>` blocks before their response. The server automatically detects and separates this content, exposing it in the correct format for each API:

**Anthropic API** — thinking appears as a separate content block before the text:

```json
{
  "content": [
    {"type": "thinking", "thinking": "Let me work through this..."},
    {"type": "text", "text": "The answer is 42."}
  ]
}
```

During streaming, thinking arrives via `thinking_delta` events, then regular `text_delta` events after `</think>`.

**OpenAI API** — thinking appears in a `reasoning_content` field on the message (DeepSeek-compatible format):

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "reasoning_content": "Let me work through this...",
      "content": "The answer is 42."
    }
  }]
}
```

During streaming, `delta.reasoning_content` carries the thinking tokens; `delta.content` carries the response.

The `thinking` request parameter (used by the Anthropic SDK to request extended thinking) is accepted and ignored — the model decides whether to think based on its own architecture. Non-thinking models are completely unaffected.

---

## Coding agents

The server works with any coding agent that supports a custom OpenAI-compatible or Anthropic-compatible base URL. All agents below connect via `http://localhost:8080` through the SSH tunnel.

| Agent | API | Recommended |
|-------|-----|-------------|
| [Aider](https://aider.chat) | OpenAI | CLI-based, excellent model support, handles `reasoning_content` |
| [Cline](https://github.com/cline/cline) (VS Code) | OpenAI or Anthropic | IDE-integrated, supports both formats, active development |
| [Goose](https://github.com/block/goose) | OpenAI or Anthropic | Block's open-source agent, tool-use focused |
| [Continue](https://continue.dev) (VS Code/JetBrains) | OpenAI | IDE-integrated autocomplete + chat |
| [Open Interpreter](https://openinterpreter.com) | OpenAI | Runs code locally, general-purpose |
| [Claude Code](https://claude.ai/code) | Anthropic | Anthropic's own CLI; works via `ANTHROPIC_BASE_URL` |
| [LibreChat](https://librechat.ai) | OpenAI | GUI chat, see setup below |
| Chatbox | OpenAI | Lightweight GUI chat |

### Aider

```bash
aider \
  --openai-api-base http://localhost:8080/v1 \
  --openai-api-key dummy \
  --model openai/qwq-32b   # use whatever MODEL_ID you started the server with
```

### Cline (VS Code)

Settings → Provider: **OpenAI Compatible** → Base URL: `http://localhost:8080/v1`, API Key: `dummy`, Model: your model ID.

Or use the Anthropic provider: Settings → Provider: **Anthropic** → then set `ANTHROPIC_BASE_URL=http://localhost:8080` in your shell before launching VS Code.

### Goose

```bash
export GOOSE_PROVIDER=openai
export OPENAI_HOST=http://localhost:8080
export OPENAI_API_KEY=dummy
goose session
```

### Continue

In `~/.continue/config.yaml`:

```yaml
models:
  - name: Local Model
    provider: openai
    model: your-model-id
    apiBase: http://localhost:8080/v1
    apiKey: dummy
```

### Claude Code

```bash
export ANTHROPIC_API_KEY=dummy
export ANTHROPIC_BASE_URL=http://localhost:8080
claude
```

Note: Claude Code is optimized for Claude's specific capabilities. It works, but local models may struggle with the tool-use patterns Claude Code relies on. QwQ-32B and Qwen3-32B handle tool use best among the thinking models.

### Chatbox

Settings → AI Provider → OpenAI API → API Host: `http://localhost:8080`, API Key: `dummy`

---

## Models

Set `MODEL_ID` to any HuggingFace causal LM. Models with a `chat_template` in their tokenizer config work best (Gemma, Llama, Qwen, Mistral, etc.). Base models without a chat template fall back to a plain `Role: content` prompt format.

## Architecture

Everything lives in `server.py`:

- **`load_model`** — initializes vLLM's `AsyncLLMEngine` at startup; uses bf16, with quantization auto-detected from the model's config (AWQ, FP8, NVFP4, MXFP4 all work out of the box)
- **`_build_prompt`** — applies the model's chat template (or plain fallback) to the message list
- **`POST /v1/messages`** — Anthropic-compatible endpoint, streaming and non-streaming
- **`POST /v1/chat/completions`** — OpenAI-compatible endpoint, streaming and non-streaming
- **`GET /v1/models`** — returns the loaded model ID

### Privacy

- vLLM telemetry disabled (`VLLM_NO_USAGE_STATS=1`)
- Request and stats logging disabled in the engine (`disable_log_requests`, `disable_log_stats`)
- Uvicorn access logs suppressed (`--no-access-log`)
- Server binds to `127.0.0.1` only — accessible exclusively through the SSH tunnel
- No conversation content is stored or logged anywhere
