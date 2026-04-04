import os
import json
import uuid
import time
import asyncio
import threading

# Disable Hugging Face telemetry before importing transformers/huggingface_hub
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import torch
import transformers
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig

transformers.logging.set_verbosity_error()

import logging
logging.getLogger("uvicorn.access").disabled = True

MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-3-270m-it")
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "0") == "1"

app = FastAPI()


@app.exception_handler(Exception)
async def _suppress_traceback(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"error": {"message": "Internal server error", "type": type(exc).__name__}})
model: AutoModelForCausalLM = None
tokenizer: AutoTokenizer = None
input_device: torch.device = None


@app.on_event("startup")
async def load_model():
    global model, tokenizer, input_device
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    quantization_config = BitsAndBytesConfig(load_in_4bit=True) if LOAD_IN_4BIT else None
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto", quantization_config=quantization_config)
    model.eval()
    input_device = next(model.parameters()).device


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


def _prepare_inputs(chat: list[dict]):
    prompt = _build_prompt(chat)
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(input_device)
    attention_mask = encoded.attention_mask.to(input_device)
    return input_ids, attention_mask


def _make_gen_kwargs(max_tokens: int, temperature: float, top_p: float | None, attention_mask) -> dict:
    kwargs = dict(
        max_new_tokens=max_tokens,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
    )
    if temperature == 0:
        kwargs["do_sample"] = False
    else:
        kwargs["do_sample"] = True
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    return kwargs


async def _iter_streamer(input_ids, kwargs: dict):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = {**kwargs, "streamer": streamer}

    thread = threading.Thread(target=lambda: model.generate(input_ids, **kwargs), daemon=True)
    thread.start()

    loop = asyncio.get_running_loop()
    it = iter(streamer)

    def next_chunk():
        try:
            return next(it)
        except StopIteration:
            return None

    while True:
        chunk = await loop.run_in_executor(None, next_chunk)
        if chunk is None:
            break
        if chunk:
            yield chunk

    thread.join()


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

    input_ids, attention_mask = _prepare_inputs(chat)
    input_tokens = input_ids.shape[-1]
    kwargs = _make_gen_kwargs(req.max_tokens, req.temperature, req.top_p, attention_mask)
    model_name = req.model or MODEL_ID

    if req.stream:
        return StreamingResponse(
            _anthropic_stream(input_ids, input_tokens, kwargs, model_name),
            media_type="text/event-stream",
        )

    with torch.inference_mode():
        output = model.generate(input_ids, **kwargs)

    new_ids = output[0][input_tokens:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": model_name,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": len(new_ids)},
    }


async def _anthropic_stream(input_ids, input_tokens: int, kwargs: dict, model_name: str):
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

    output_tokens = 0
    async for chunk in _iter_streamer(input_ids, kwargs):
        output_tokens += 1
        yield sse("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": chunk},
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

    input_ids, attention_mask = _prepare_inputs(chat)
    input_tokens = input_ids.shape[-1]
    max_tokens = req.max_completion_tokens or req.max_tokens or 2048
    kwargs = _make_gen_kwargs(max_tokens, req.temperature, req.top_p, attention_mask)
    model_name = req.model or MODEL_ID

    if req.stream:
        return StreamingResponse(
            _oai_stream(input_ids, input_tokens, kwargs, model_name),
            media_type="text/event-stream",
        )

    with torch.inference_mode():
        output = model.generate(input_ids, **kwargs)

    new_ids = output[0][input_tokens:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": input_tokens, "completion_tokens": len(new_ids), "total_tokens": input_tokens + len(new_ids)},
    }


async def _oai_stream(input_ids, input_tokens: int, kwargs: dict, model_name: str):
    msg_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def sse(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    # opening chunk with role
    yield sse({
        "id": msg_id, "object": "chat.completion.chunk", "created": created, "model": model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    })

    async for chunk in _iter_streamer(input_ids, kwargs):
        yield sse({
            "id": msg_id, "object": "chat.completion.chunk", "created": created, "model": model_name,
            "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
        })

    yield sse({
        "id": msg_id, "object": "chat.completion.chunk", "created": created, "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    })
    yield "data: [DONE]\n\n"
