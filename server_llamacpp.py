import os
import json
import uuid
import time
from dataclasses import dataclass

# Disable telemetry before importing anything else
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

import logging
logging.getLogger("uvicorn.access").disabled = True

MODEL_ID = os.environ.get("MODEL_ID", "ggml-org/gemma-4-31B-it-GGUF")
MODEL_FILE = os.environ.get("MODEL_FILE", "gemma-4-31B-it-Q4_K_M.gguf")
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "-1"))  # -1 = offload all
N_CTX = int(os.environ.get("N_CTX", "8192"))

app = FastAPI()


@app.exception_handler(Exception)
async def _suppress_traceback(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": {"message": "Internal server error", "type": type(exc).__name__}})


llm: Llama = None


# ---------- model-specific thinking format ----------

@dataclass(frozen=True)
class _ThinkFmt:
    """Describes how a model encodes its reasoning block in raw output."""
    open: str         # text that opens the thinking block
    close: str        # text that closes the thinking block
    resp: str | None  # text that introduces the response channel (harmony only); None = response follows directly after close
    think_token: str | None  # token to prepend to the system message to activate thinking (Gemma 4 only)


# <think>...</think> — QwQ, DeepSeek-R1, Qwen3, etc.
_THINK_TAG = _ThinkFmt(
    open="<think>",
    close="</think>",
    resp=None,
    think_token=None,
)

# Gemma 4 channel format: <|channel>thought\n...<channel|>
# Thinking is opt-in: prepend <|think|> (token 98) to the system message content.
_GEMMA4_CH = _ThinkFmt(
    open="<|channel>thought\n",
    close="<channel|>",
    resp=None,
    think_token="<|think|>",
)

# GPT-OSS harmony format: multi-channel output with dedicated analysis and final channels.
# All markers are special tokens in the o200k_harmony vocabulary.
_HARMONY = _ThinkFmt(
    open="<|channel|>analysis<|message|>",
    close="<|end|>",
    resp="<|channel|>final<|message|>",
    think_token=None,
)


def _get_think_fmt(model_id: str) -> _ThinkFmt:
    lower = model_id.lower()
    if "gpt-oss" in lower:
        return _HARMONY
    if "gemma-4" in lower or "gemma4" in lower:
        return _GEMMA4_CH
    return _THINK_TAG


_think_fmt: _ThinkFmt = _THINK_TAG  # replaced at startup


@app.on_event("startup")
async def load_model():
    global llm, _think_fmt

    _think_fmt = _get_think_fmt(MODEL_ID)

    print(f"Downloading {MODEL_FILE} from {MODEL_ID} ...")
    model_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=MODEL_FILE,
    )
    print(f"Model cached at: {model_path}")

    print("Loading model into GPU ...")
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        verbose=False,
    )
    print("Model loaded.")


# ---------- request / response types ----------

class Message(BaseModel):
    role: str
    content: str | list


class MessagesRequest(BaseModel):
    model: str = ""
    messages: list[Message]
    max_tokens: int = 8192
    stream: bool = False
    system: str | None = None
    temperature: float = 1.0
    top_p: float | None = None
    stop_sequences: list[str] | None = None
    thinking: dict | None = None


class ChatRequest(BaseModel):
    model: str = ""
    messages: list[Message]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False
    temperature: float = 1.0
    top_p: float | None = None
    stop: str | list[str] | None = None


# ---------- shared helpers ----------

def _extract_text(content: str | list) -> str:
    if isinstance(content, str):
        return content
    return "".join(
        b["text"] for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    )


def _merge_consecutive(chat: list[dict]) -> list[dict]:
    merged = []
    for msg in chat:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n" + msg["content"]
        else:
            merged.append(dict(msg))
    return merged


def _build_messages(chat: list[dict]) -> list[dict]:
    """Build a messages list for llama-cpp-python's chat completion."""
    messages = _merge_consecutive(chat)

    if _think_fmt.think_token is None:
        return messages

    # Gemma 4: activate thinking mode by prepending <|think|> to the system
    # message content. llama.cpp encodes this as the actual special token (ID 98),
    # which instructs the model to emit a <|channel>thought\n...<channel|> block.
    messages = list(messages)  # shallow copy before mutating
    for i, msg in enumerate(messages):
        if msg["role"] == "system":
            messages[i] = {**msg, "content": _think_fmt.think_token + msg["content"]}
            return messages
    # No system message present — add a minimal one with just the think token.
    return [{"role": "system", "content": _think_fmt.think_token}] + messages


def _parse_thinking(text: str) -> tuple[str, str, bool]:
    """Split raw model output into (thinking, response, is_still_thinking).

    Handles three distinct formats determined by the loaded model:

    • <think>…</think>  — QwQ, DeepSeek-R1, Qwen3 (plain text markers)
    • <|channel>thought\\n…<channel|>  — Gemma 4 (special vocab tokens 100/101)
    • <|channel|>analysis<|message|>…<|end|> … <|channel|>final<|message|>…
                        — GPT-OSS harmony format (o200k_harmony special tokens)
    """
    fmt = _think_fmt

    if not text.startswith(fmt.open):
        if fmt.resp is not None:
            # Harmony: response lives in the 'final' channel.
            resp_idx = text.find(fmt.resp)
            if resp_idx == -1:
                return ("", "", False)
            return ("", text[resp_idx + len(fmt.resp):], False)
        return ("", text, False)

    close_idx = text.find(fmt.close)
    if close_idx == -1:
        return (text[len(fmt.open):], "", True)

    thinking = text[len(fmt.open):close_idx]
    after_close = text[close_idx + len(fmt.close):]

    if fmt.resp is not None:
        resp_idx = after_close.find(fmt.resp)
        if resp_idx == -1:
            # Analysis finished; final channel header not yet in the stream.
            return (thinking, "", False)
        response = after_close[resp_idx + len(fmt.resp):]
    else:
        response = after_close
        if response.startswith("\n"):
            response = response[1:]

    return (thinking, response, False)


def _generate(messages: list[dict], max_tokens: int, temperature: float, top_p: float | None):
    """Non-streaming generation. Returns (text, input_tokens, output_tokens)."""
    kwargs: dict = dict(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if top_p is not None:
        kwargs["top_p"] = top_p

    result = llm.create_chat_completion(**kwargs)
    text = result["choices"][0]["message"]["content"] or ""
    input_tokens = result["usage"]["prompt_tokens"]
    output_tokens = result["usage"]["completion_tokens"]
    return text, input_tokens, output_tokens


def _generate_stream(messages: list[dict], max_tokens: int, temperature: float, top_p: float | None):
    """Streaming generation. Yields delta text strings."""
    kwargs: dict = dict(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    if top_p is not None:
        kwargs["top_p"] = top_p

    for chunk in llm.create_chat_completion(**kwargs):
        delta = chunk["choices"][0].get("delta", {})
        content = delta.get("content")
        if content:
            yield content


# ---------- endpoints ----------

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "created": int(time.time()), "owned_by": "local"}],
    }


# --- Anthropic ---

@app.post("/v1/messages")
async def messages(req: MessagesRequest):
    chat = []
    if req.system:
        chat.append({"role": "system", "content": req.system})
    for msg in req.messages:
        chat.append({"role": msg.role, "content": _extract_text(msg.content)})

    messages_list = _build_messages(chat)
    model_name = req.model or MODEL_ID

    if req.stream:
        return StreamingResponse(
            _anthropic_stream(messages_list, req.max_tokens, req.temperature, req.top_p, model_name),
            media_type="text/event-stream",
        )

    text, input_tokens, output_tokens = _generate(messages_list, req.max_tokens, req.temperature, req.top_p)
    thinking, response, _ = _parse_thinking(text)

    content = []
    if thinking:
        content.append({"type": "thinking", "thinking": thinking})
    content.append({"type": "text", "text": response})

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model_name,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


async def _anthropic_stream(messages_list: list[dict], max_tokens: int, temperature: float, top_p: float | None, model_name: str):
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    open_len = len(_think_fmt.open)

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    yield sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant", "content": [],
            "model": model_name, "stop_reason": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })
    yield sse("ping", {"type": "ping"})

    phase = "buffering"
    block_index = 0
    thinking_emitted = 0
    response_emitted = 0
    output_tokens = 0
    full_text = ""

    for delta in _generate_stream(messages_list, max_tokens, temperature, top_p):
        full_text += delta
        output_tokens += 1  # approximate

        if phase == "buffering":
            if len(full_text) < open_len:
                continue
            if full_text.startswith(_think_fmt.open):
                phase = "thinking"
                yield sse("content_block_start", {"type": "content_block_start", "index": 0,
                           "content_block": {"type": "thinking", "thinking": ""}})
            else:
                phase = "responding"
                yield sse("content_block_start", {"type": "content_block_start", "index": 0,
                           "content_block": {"type": "text", "text": ""}})

        if phase == "thinking":
            thinking, response, still = _parse_thinking(full_text)
            delta_t = thinking[thinking_emitted:]
            thinking_emitted = len(thinking)
            if delta_t:
                yield sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                           "delta": {"type": "thinking_delta", "thinking": delta_t}})
            if not still:
                phase = "responding"
                yield sse("content_block_stop", {"type": "content_block_stop", "index": 0})
                block_index = 1
                yield sse("content_block_start", {"type": "content_block_start", "index": 1,
                           "content_block": {"type": "text", "text": ""}})

        if phase == "responding":
            _, response, _ = _parse_thinking(full_text)
            delta_r = response[response_emitted:]
            response_emitted = len(response)
            if delta_r:
                yield sse("content_block_delta", {"type": "content_block_delta", "index": block_index,
                           "delta": {"type": "text_delta", "text": delta_r}})

    if phase == "buffering":
        yield sse("content_block_start", {"type": "content_block_start", "index": 0,
                   "content_block": {"type": "text", "text": ""}})
        if full_text:
            yield sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                       "delta": {"type": "text_delta", "text": full_text}})

    yield sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
    yield sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })
    yield sse("message_stop", {"type": "message_stop"})


# --- OpenAI ---

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    chat = [{"role": msg.role, "content": _extract_text(msg.content)} for msg in req.messages]

    messages_list = _build_messages(chat)
    max_tokens = req.max_completion_tokens or req.max_tokens or 8192
    model_name = req.model or MODEL_ID

    if req.stream:
        return StreamingResponse(
            _oai_stream(messages_list, max_tokens, req.temperature, req.top_p, model_name),
            media_type="text/event-stream",
        )

    text, input_tokens, output_tokens = _generate(messages_list, max_tokens, req.temperature, req.top_p)
    thinking, response, _ = _parse_thinking(text)

    message = {"role": "assistant", "content": response}
    if thinking:
        message["reasoning_content"] = thinking

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens, "total_tokens": input_tokens + output_tokens},
    }


async def _oai_stream(messages_list: list[dict], max_tokens: int, temperature: float, top_p: float | None, model_name: str):
    msg_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    open_len = len(_think_fmt.open)

    def sse(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    def chunk(delta: dict, finish_reason=None) -> dict:
        return {"id": msg_id, "object": "chat.completion.chunk", "created": created,
                "model": model_name, "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}

    yield sse(chunk({"role": "assistant", "content": ""}))

    phase = "buffering"
    thinking_emitted = 0
    response_emitted = 0
    full_text = ""

    for delta in _generate_stream(messages_list, max_tokens, temperature, top_p):
        full_text += delta

        if phase == "buffering":
            if len(full_text) < open_len:
                continue
            phase = "thinking" if full_text.startswith(_think_fmt.open) else "responding"

        if phase == "thinking":
            thinking, response, still = _parse_thinking(full_text)
            delta_t = thinking[thinking_emitted:]
            thinking_emitted = len(thinking)
            if delta_t:
                yield sse(chunk({"reasoning_content": delta_t}))
            if not still:
                phase = "responding"

        if phase == "responding":
            _, response, _ = _parse_thinking(full_text)
            delta_r = response[response_emitted:]
            response_emitted = len(response)
            if delta_r:
                yield sse(chunk({"content": delta_r}))

    if phase == "buffering" and full_text:
        yield sse(chunk({"content": full_text}))

    yield sse(chunk({}, finish_reason="stop"))
    yield "data: [DONE]\n\n"
