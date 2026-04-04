import os
import json
import uuid
import time
import asyncio
import threading

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-3-4b-it")

app = FastAPI()
model: AutoModelForCausalLM = None
tokenizer: AutoTokenizer = None


@app.on_event("startup")
async def load_model():
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()


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


# ---------- helpers ----------

def build_prompt(req: MessagesRequest) -> str:
    chat = []
    if req.system:
        chat.append({"role": "system", "content": req.system})
    for msg in req.messages:
        if isinstance(msg.content, str):
            text = msg.content
        else:
            text = "".join(
                b["text"] for b in msg.content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        chat.append({"role": msg.role, "content": text})

    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

    # fallback for base models without a chat template
    parts = []
    for m in chat:
        prefix = "System" if m["role"] == "system" else m["role"].capitalize()
        parts.append(f"{prefix}: {m['content']}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def gen_kwargs(req: MessagesRequest) -> dict:
    kwargs = dict(max_new_tokens=req.max_tokens)
    if req.temperature == 0:
        kwargs["do_sample"] = False
    else:
        kwargs["do_sample"] = True
        kwargs["temperature"] = req.temperature
    if req.top_p is not None:
        kwargs["top_p"] = req.top_p
    return kwargs


# ---------- endpoints ----------

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/messages")
async def messages(req: MessagesRequest):
    prompt = build_prompt(req)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    input_tokens = input_ids.shape[-1]
    kwargs = gen_kwargs(req)

    if req.stream:
        return StreamingResponse(
            _stream(input_ids, input_tokens, kwargs, req.model or MODEL_ID),
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
        "model": req.model or MODEL_ID,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": len(new_ids),
        },
    }


async def _stream(input_ids, input_tokens: int, kwargs: dict, model_name: str):
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    def sse(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    yield sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model_name,
            "stop_reason": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    })
    yield sse("content_block_start", {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    })
    yield sse("ping", {"type": "ping"})

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    kwargs["streamer"] = streamer

    thread = threading.Thread(
        target=lambda: model.generate(input_ids, **kwargs),
        daemon=True,
    )
    thread.start()

    output_tokens = 0
    for chunk in streamer:
        if not chunk:
            continue
        output_tokens += 1
        yield sse("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": chunk},
        })
        await asyncio.sleep(0)  # yield control to the event loop between tokens

    thread.join()

    yield sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })
    yield sse("message_stop", {"type": "message_stop"})