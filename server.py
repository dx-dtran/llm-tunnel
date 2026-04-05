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
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "0") == "1"

app = FastAPI()


@app.exception_handler(Exception)
async def _suppress_traceback(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": {"message": "Internal server error", "type": type(exc).__name__}})

engine: AsyncLLMEngine = None
tokenizer = None


@app.on_event("startup")
async def load_model():
    global engine, tokenizer

    engine_kwargs = dict(
        model=MODEL_ID,
        dtype="bfloat16",
        device="auto",
        disable_log_requests=True,
        disable_log_stats=True,
    )
    if LOAD_IN_4BIT:
        engine_kwargs["quantization"] = "bitsandbytes"
        engine_kwargs["load_format"] = "bitsandbytes"

    engine_args = AsyncEngineArgs(**engine_kwargs)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = await engine.get_tokenizer()


# ---------- request / response types ----------

class Message(BaseModel):
    role: str
    content: str | list


class MessagesRequest(BaseModel):
    model: str = ""
    messages: list[Message]
    max_tokens: int = 2048
    stream: bool = False
    system: str | None = None
    temperature: float = 1.0
    top_p: float | None = None
    stop_sequences: list[str] | None = None


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

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
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
    yield sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
    yield sse("ping", {"type": "ping"})

    request_id = f"msg-{uuid.uuid4().hex[:12]}"
    prev_text = ""
    output_tokens = 0
    async for output in engine.generate(prompt, params, request_id):
        new_text = output.outputs[0].text
        delta = new_text[len(prev_text):]
        prev_text = new_text
        if delta:
            output_tokens = len(output.outputs[0].token_ids)
            yield sse("content_block_delta", {
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": delta},
            })

    yield sse("content_block_stop", {"type": "content_block_stop", "index": 0})
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
    max_tokens = req.max_completion_tokens or req.max_tokens or 2048
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

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens, "total_tokens": input_tokens + output_tokens},
    }


async def _oai_stream(prompt: str, input_tokens: int, params: SamplingParams, model_name: str):
    msg_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def sse(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    yield sse({
        "id": msg_id, "object": "chat.completion.chunk", "created": created, "model": model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    })

    request_id = f"oai-{uuid.uuid4().hex[:12]}"
    prev_text = ""
    async for output in engine.generate(prompt, params, request_id):
        new_text = output.outputs[0].text
        delta = new_text[len(prev_text):]
        prev_text = new_text
        if delta:
            yield sse({
                "id": msg_id, "object": "chat.completion.chunk", "created": created, "model": model_name,
                "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
            })

    yield sse({
        "id": msg_id, "object": "chat.completion.chunk", "created": created, "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    })
    yield "data: [DONE]\n\n"
