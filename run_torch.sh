#!/bin/bash

# Pure PyTorch backend — no vLLM dependency.
# Downloads BF16 weights and applies int4 quantization at load time (~30-60s).
# Set QUANTIZE=0 to skip quantization (e.g. for small models or FP8 GPUs).

declare -A MODELS=(
    # Gemma 4 — full precision (quantized at load time by default)
    ["gemma4-e2b"]="google/gemma-4-E2B-it"
    ["gemma4-e4b"]="google/gemma-4-E4B-it"
    ["gemma4-26b"]="google/gemma-4-26B-A4B-it"
    ["gemma4-31b"]="google/gemma-4-31B-it"
    # Local testing
    ["gemma3-270m"]="google/gemma-3-270m-it"
)

if [ -z "$1" ] || [ -z "${MODELS[$1]+x}" ]; then
    echo "Usage: ./run_torch.sh <model>"
    echo ""
    echo "Available models:"
    for key in $(echo "${!MODELS[@]}" | tr ' ' '\n' | sort); do
        printf "  %-20s %s\n" "$key" "${MODELS[$key]}"
    done
    echo ""
    echo "Environment variables:"
    echo "  QUANTIZE=0     Skip int4 quantization (default: 1)"
    echo "  MAX_SEQ_LEN=N  Max sequence length (default: 8192)"
    echo ""
    echo "Models download automatically on first run and are cached."
    exit 1
fi

export MODEL_ID="${MODELS[$1]}"
export QUANTIZE="${QUANTIZE:-1}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
export HF_HUB_DISABLE_XET=1

fuser -k 8080/tcp 2>/dev/null; sleep 1

echo "Model:    $MODEL_ID"
echo "Quantize: $([ "$QUANTIZE" = "1" ] && echo "int4 weight-only" || echo "none (BF16)")"
echo "Max seq:  $MAX_SEQ_LEN"
uvicorn server_torch:app --host 127.0.0.1 --port 8080 --no-access-log
