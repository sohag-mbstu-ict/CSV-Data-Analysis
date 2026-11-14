#!/usr/bin/env bash
# Start the vLLM server that the CSV app uses when LLM_BACKEND=vllm.
#
#   ./serve_vllm.sh                 start (stops leftover vLLM processes first)
#   ./serve_vllm.sh --stop          stop every vLLM process and free the GPU
#
# Settings (environment variables):
#   VLLM_MODEL_DIR   model weights folder        (default: base_model/Qwen3.5-4B)
#   VLLM_PORT        port to listen on            (default: 8001; Django uses 8000)
#   VLLM_GPU_UTIL    share of GPU memory to use   (default: 0.85)
set -euo pipefail
cd "$(dirname "$0")"

MODEL_DIR="${VLLM_MODEL_DIR:-base_model/Qwen3.5-4B}"
PORT="${VLLM_PORT:-8001}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
SERVED_NAME="qwen3.5-4b"

vllm_pids() {
    # Leftover `vllm serve` processes and their engine workers, excluding this script.
    pgrep -f "vllm serve|VLLM::EngineCore" | grep -vx "$$" || true
}

stop_vllm() {
    local pids
    pids="$(vllm_pids)"
    [ -z "$pids" ] && return 0
    echo "Stopping vLLM processes:"
    ps -o pid=,args= -p $pids | sed 's/^/  /'
    kill $pids 2>/dev/null || true
    for _ in $(seq 1 20); do
        [ -z "$(vllm_pids)" ] && return 0
        sleep 0.5
    done
    echo "  still running after 10 s; forcing them to stop"
    kill -9 $(vllm_pids) 2>/dev/null || true
    sleep 1
}

if [ "${1:-}" = "--stop" ]; then
    stop_vllm
    echo "vLLM stopped."
    exit 0
fi

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "Model weights not found in $MODEL_DIR (no config.json). Download them with: uv run python download_weight.py" >&2
    exit 1
fi

# Already running and healthy: nothing to do.
if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
    echo "vLLM is already serving on port $PORT:"
    curl -s "http://localhost:$PORT/v1/models"; echo
    exit 0
fi

# A half-started or crashed vLLM keeps GPU memory and makes the next start fail.
stop_vllm

if ss -ltn "sport = :$PORT" | grep -q LISTEN; then
    echo "Port $PORT is used by another program:" >&2
    ss -ltnp "sport = :$PORT" >&2
    echo "Stop it, or pick another port: VLLM_PORT=8002 ./serve_vllm.sh (and run Django with VLLM_URL=http://localhost:8002/v1)" >&2
    exit 1
fi

# Ollama keeps a model in GPU memory for 30 minutes after its last use.
# `ollama ps` fails when the Ollama service is stopped; that just means nothing is loaded.
if command -v ollama >/dev/null 2>&1; then
    loaded="$(ollama ps 2>/dev/null | awk 'NR>1 {print $1}' || true)"
    for model in $loaded; do
        echo "Unloading Ollama model $model from the GPU"
        ollama stop "$model" || true
    done
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    read -r used total < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1 | tr -d ',')
    echo "GPU memory in use before start: ${used} / ${total} MiB (vLLM will take ${GPU_UTIL} of the total)"
    others="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true)"
    if [ -n "$others" ]; then
        echo "Other programs on the GPU (vLLM may not fit; lower VLLM_GPU_UTIL if it fails):"
        echo "$others" | sed 's/^/  /'
    fi
fi

echo "Starting vLLM: $MODEL_DIR on port $PORT as \"$SERVED_NAME\". Ready when it prints 'Application startup complete' (first start ~2 min while it compiles)."
# fp8_per_tensor: weights stored in 8 bits (~4.7 GB instead of ~9.3 GB bf16) so a 4B model fits on 8 GB.
# --language-model-only: skip the vision encoder; the app sends text only.
# CUDA graphs: the GPU replays each decoding step as one recorded graph instead of Python
# launching ~1,000 small kernels per token (3x faster: ~72 instead of ~24 tok/s on an RTX 3070).
# --max-num-seqs 4 + --max-cudagraph-capture-size 4: record graphs only for batches up to 4.
# The default (up to 512) needs ~4 GB more and runs out of memory on an 8 GB GPU.
# The first start compiles the model (~2 min); later starts reuse ~/.cache/vllm.
exec uv run vllm serve "$MODEL_DIR" \
    --port "$PORT" \
    --served-model-name "$SERVED_NAME" \
    --quantization fp8_per_tensor \
    --language-model-only \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --max-cudagraph-capture-size 4 \
    --gpu-memory-utilization "$GPU_UTIL"
