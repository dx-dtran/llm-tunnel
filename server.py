import os
import json
import uuid
import time

# Disable telemetry before importing anything else
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

import logging
logging.getLogger("uvicorn.access").disabled = True

MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-3-270m-it")

app = FastAPI()


@app.exception_handler(Exception)
async def _suppress_traceback(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": {"message": "Internal server error", "type": type(exc).__name__}})

engine: AsyncLLMEngine = None
tokenizer = None


@app.on_event("startup")
async def load_model():
    global engine, tokenizer

    # Pre-quantized models (AWQ, FP8, NVFP4, MXFP4) are auto-detected by vLLM
    # via each model's quantization_config in config.json — no extra flags needed.
    engine_args = AsyncEngineArgs(
        model=MODEL_ID,
        dtype="bfloat16",
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = engine.tokenizer


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


def _build_prompt(chat: list[dict]) -> str:
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(_merge_consecutive(chat), tokenize=False, add_generation_prompt=True)
    parts = []
    for m in chat:
        prefix = "System" if m["role"] == "system" else m["role"].capitalize()
        parts.append(f"{prefix}: {m['content']}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def _make_sampling_params(max_tokens: int, temperature: float, top_p: float | None) -> SamplingParams:
    kwargs = dict(max_tokens=max_tokens)
    if temperature == 0:
        kwargs["temperature"] = 0
    else:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    return SamplingParams(**kwargs)


def _count_input_tokens(prompt: str) -> int:
    return len(tokenizer.encode(prompt))


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_OPEN_LEN = len(_THINK_OPEN)


def _parse_thinking(text: str) -> tuple[str, str, bool]:
    """Split cumulative model output into (thinking, response, is_still_thinking).

    Only detects <think> at position 0 (how all reasoning models behave).
    """
    if not text.startswith(_THINK_OPEN):
        return ("", text, False)

    end_idx = text.find(_THINK_CLOSE)
    if end_idx == -1:
        return (text[_THINK_OPEN_LEN:], "", True)

    thinking = text[_THINK_OPEN_LEN:end_idx]
    response = text[end_idx + len(_THINK_CLOSE):]
    if response.startswith("\n"):
        response = response[1:]
    return (thinking, response, False)


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

    prompt = _build_prompt(chat)
    input_tokens = _count_input_tokens(prompt)
    params = _make_sampling_params(req.max_tokens, req.temperature, req.top_p)
    model_name = req.model or MODEL_ID

    if req.stream:
        return StreamingResponse(
            _anthropic_stream(prompt, input_tokens, params, model_name),
            media_type="text/event-stream",
        )

    request_id = f"msg-{uuid.uuid4().hex[:12]}"
    final = None
    async for output in engine.generate(prompt, params, request_id):
        final = output
    text = final.outputs[0].text
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
        "usage": {"input_tokens": input_tokens, "output_tokens": len(final.outputs[0].token_ids)},
    }


async def _anthropic_stream(prompt: str, input_tokens: int, params: SamplingParams, model_name: str):
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    yield sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant", "content": [],
            "model": model_name, "stop_reason": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    })
    yield sse("ping", {"type": "ping"})

    request_id = f"msg-{uuid.uuid4().hex[:12]}"
    phase = "buffering"  # "buffering" | "thinking" | "responding"
    block_index = 0
    thinking_emitted = 0
    response_emitted = 0
    output_tokens = 0
    full_text = ""

    async for output in engine.generate(prompt, params, request_id):
        full_text = output.outputs[0].text
        output_tokens = len(output.outputs[0].token_ids)

        if phase == "buffering":
            if len(full_text) < _THINK_OPEN_LEN:
                continue
            if full_text.startswith(_THINK_OPEN):
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

    # Generation ended while still buffering (very short output, no <think>)
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

    prompt = _build_prompt(chat)
    input_tokens = _count_input_tokens(prompt)
    max_tokens = req.max_completion_tokens or req.max_tokens or 8192
    params = _make_sampling_params(max_tokens, req.temperature, req.top_p)
    model_name = req.model or MODEL_ID

    if req.stream:
        return StreamingResponse(
            _oai_stream(prompt, input_tokens, params, model_name),
            media_type="text/event-stream",
        )

    request_id = f"oai-{uuid.uuid4().hex[:12]}"
    final = None
    async for output in engine.generate(prompt, params, request_id):
        final = output
    text = final.outputs[0].text
    output_tokens = len(final.outputs[0].token_ids)
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


async def _oai_stream(prompt: str, input_tokens: int, params: SamplingParams, model_name: str):
    msg_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def sse(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    def chunk(delta: dict, finish_reason=None) -> dict:
        return {"id": msg_id, "object": "chat.completion.chunk", "created": created,
                "model": model_name, "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}

    yield sse(chunk({"role": "assistant", "content": ""}))

    request_id = f"oai-{uuid.uuid4().hex[:12]}"
    phase = "buffering"
    thinking_emitted = 0
    response_emitted = 0
    full_text = ""

    async for output in engine.generate(prompt, params, request_id):
        full_text = output.outputs[0].text

        if phase == "buffering":
            if len(full_text) < _THINK_OPEN_LEN:
                continue
            phase = "thinking" if full_text.startswith(_THINK_OPEN) else "responding"

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

    # Generation ended while still buffering
    if phase == "buffering" and full_text:
        yield sse(chunk({"content": full_text}))

    yield sse(chunk({}, finish_reason="stop"))
    yield "data: [DONE]\n\n"
