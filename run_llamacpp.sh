#!/bin/bash

# llama.cpp backend — uses pre-quantized GGUF files via llama-cpp-python.
# Downloads the GGUF file on first run via huggingface_hub (cached for subsequent runs).
# Minimal deps: pip install -r requirements_llamacpp.txt

declare -A MODELS=(
    # Gemma 4 31B — Q4_K_M quantization (~18.7 GB, fits on 24 GB GPU)
    ["gemma4-31b"]="ggml-org/gemma-4-31B-it-GGUF|gemma-4-31B-it-Q4_K_M.gguf"
    # Gemma 4 26B — Q4_K_M quantization
    ["gemma4-26b"]="ggml-org/gemma-4-26B-A4B-it-GGUF|gemma-4-26B-A4B-it-Q4_K_M.gguf"
    # Gemma 4 E4B — Q4_K_M quantization (small, fast)
    ["gemma4-e4b"]="ggml-org/gemma-4-E4B-it-GGUF|gemma-4-E4B-it-Q4_K_M.gguf"
    # Gemma 4 E2B — Q4_K_M quantization (tiny, testing)
    ["gemma4-e2b"]="ggml-org/gemma-4-E2B-it-GGUF|gemma-4-E2B-it-Q4_K_M.gguf"
    # GPT-OSS 20B — Q4_K_M quantization (~11.7 GB)
    # Uses harmony format: reasoning in analysis channel, response in final channel.
    # The native format is MXFP4; K-quants are community re-quantizations.
    ["gpt-oss-20b"]="bartowski/openai_gpt-oss-20b-GGUF|openai_gpt-oss-20b-Q4_K_M.gguf"
    # GPT-OSS 20B — Q5_K_M quantization (~11.7 GB, slightly higher quality)
    ["gpt-oss-20b-q5"]="bartowski/openai_gpt-oss-20b-GGUF|openai_gpt-oss-20b-Q5_K_M.gguf"
)

if [ -z "$1" ] || [ -z "${MODELS[$1]+x}" ]; then
    echo "Usage: ./run_llamacpp.sh <model>"
    echo ""
    echo "Available models:"
    for key in $(echo "${!MODELS[@]}" | tr ' ' '\n' | sort); do
        IFS='|' read -r repo file <<< "${MODELS[$key]}"
        printf "  %-20s %s (%s)\n" "$key" "$repo" "$file"
    done
    echo ""
    echo "Environment variables:"
    echo "  N_GPU_LAYERS=-1  GPU layers to offload (-1 = all, default)"
    echo "  N_CTX=8192       Context length (default: 8192)"
    echo ""
    echo "Setup: pip install -r requirements_llamacpp.txt"
    echo "GGUF files download automatically on first run and are cached."
    exit 1
fi

IFS='|' read -r MODEL_ID MODEL_FILE <<< "${MODELS[$1]}"
export MODEL_ID
export MODEL_FILE
export N_GPU_LAYERS="${N_GPU_LAYERS:--1}"
export N_CTX="${N_CTX:-8192}"
export HF_HUB_DISABLE_XET=1

fuser -k 8080/tcp 2>/dev/null; sleep 1

echo "Model:      $MODEL_ID"
echo "GGUF file:  $MODEL_FILE"
echo "GPU layers: $N_GPU_LAYERS"
echo "Context:    $N_CTX"
uvicorn server_llamacpp:app --host 127.0.0.1 --port 8080 --no-access-log
