#!/bin/bash

# model alias → "HF_MODEL_ID [trust_remote_code]"
declare -A MODELS=(
    # Local testing
    ["gemma3-270m"]="google/gemma-3-270m-it"
    # Gemma 4 — full precision (small enough to run as-is)
    ["gemma4-e2b"]="google/gemma-4-E2B-it"
    ["gemma4-e4b"]="google/gemma-4-E4B-it"
    # Gemma 4 — full precision
    ["gemma4-26b"]="google/gemma-4-26B-A4B-it"
    ["gemma4-31b"]="google/gemma-4-31B-it"
    # Gemma 4 — 4-bit AWQ (RTX recommended)
    ["gemma4-26b-4bit"]="cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"
    ["gemma4-31b-4bit"]="cyankiwi/gemma-4-31B-it-AWQ-4bit"
    # GPT-OSS — full precision
    ["gpt-oss-20b"]="openai/gpt-oss-20b"
    ["gpt-oss-120b"]="openai/gpt-oss-120b"
    # GPT-OSS — 4-bit quantized
    ["gpt-oss-20b-4bit"]="unsloth/gpt-oss-20b-bnb-4bit"
    ["gpt-oss-120b-4bit"]="twhitworth/gpt-oss-120b-awq-w4a16"
)

if [ -z "$1" ] || [ -z "${MODELS[$1]+x}" ]; then
    echo "Usage: ./run.sh <model>"
    echo ""
    echo "Available models:"
    for key in $(echo "${!MODELS[@]}" | tr ' ' '\n' | sort); do
        hf_id=$(echo "${MODELS[$key]}" | awk '{print $1}')
        printf "  %-25s %s\n" "$key" "$hf_id"
    done
    echo ""
    echo "Models download automatically on first run and are cached for subsequent runs."
    exit 1
fi

entry="${MODELS[$1]}"
MODEL_ID=$(echo "$entry" | awk '{print $1}')
TRUST=$(echo "$entry" | grep -q "trust_remote_code" && echo "1" || echo "0")

export MODEL_ID
export TRUST_REMOTE_CODE="$TRUST"

echo "Model: $MODEL_ID"
uvicorn server:app --host 127.0.0.1 --port 8080 --no-access-log
