#!/bin/bash

# model alias → HF_MODEL_ID
declare -A MODELS=(
    # Local testing
    ["gemma3-270m"]="google/gemma-3-270m-it"
    # Gemma 4 — full precision (small enough to run as-is)
    ["gemma4-e2b"]="google/gemma-4-E2B-it"
    ["gemma4-e4b"]="google/gemma-4-E4B-it"
    # Gemma 4 — full precision
    ["gemma4-26b"]="google/gemma-4-26B-A4B-it"
    ["gemma4-31b"]="google/gemma-4-31B-it"
    # Gemma 4 — pre-quantized (fastest inference)
    # FP8: no quality loss, 175 tok/s on H100
    ["gemma4-26b-fp8"]="protoLabsAI/gemma-4-26B-A4B-it-FP8"
    # NVFP4: NVIDIA Model Optimizer quant, requires Hopper/Blackwell GPU
    ["gemma4-31b-nvfp4"]="nvidia/Gemma-4-31B-IT-NVFP4"
    # GPT-OSS — natively MXFP4 quantized (vLLM auto-detects, no extra flags needed)
    ["gpt-oss-20b"]="openai/gpt-oss-20b"
    ["gpt-oss-120b"]="openai/gpt-oss-120b"
    # GPT-OSS — AWQ W4A16 (community quant, ~7x smaller, works on non-Hopper GPUs)
    ["gpt-oss-120b-awq"]="twhitworth/gpt-oss-120b-awq-w4a16"
    # Thinking / reasoning models (output <think>...</think> blocks)
    ["qwq-32b"]="Qwen/QwQ-32B"
    ["qwq-32b-fp8"]="modelscope/QwQ-32B-FP8"
    ["qwq-32b-awq"]="Qwen/QwQ-32B-AWQ"
    ["deepseek-r1-qwen-32b"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    ["qwen3-32b"]="Qwen/Qwen3-32B"
)

if [ -z "$1" ] || [ -z "${MODELS[$1]+x}" ]; then
    echo "Usage: ./run.sh <model>"
    echo ""
    echo "Available models:"
    for key in $(echo "${!MODELS[@]}" | tr ' ' '\n' | sort); do
        printf "  %-25s %s\n" "$key" "${MODELS[$key]}"
    done
    echo ""
    echo "Models download automatically on first run and are cached for subsequent runs."
    exit 1
fi

MODEL_ID="${MODELS[$1]}"
export MODEL_ID

fuser -k 8080/tcp 2>/dev/null; sleep 1

echo "Model: $MODEL_ID"
uvicorn server:app --host 127.0.0.1 --port 8080 --no-access-log
