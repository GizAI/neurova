#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/home/user/models/Qwen3.6-27B}"
QB_DIR="${QB_DIR:-/home/user/models/Qwen3.6-27B-langburst-q3}"
MODEL_NAME="${MODEL_NAME:-langburst-qwen3.6-27b-q3}"
SERVER_PORT="${SERVER_PORT:-8008}"
OPENWEBUI_COMPAT_PORT="${OPENWEBUI_COMPAT_PORT:-5000}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-${LANGBURST_CONTEXT_WINDOW:-${RECENT_WINDOW:-16384}}}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-${LANGBURST_MAX_PROMPT_TOKENS:-$CONTEXT_WINDOW}}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-${LANGBURST_KV_CACHE_DTYPE:-fp8_e4m3}}"
KV_BLOCK_SIZE="${KV_BLOCK_SIZE:-${LANGBURST_KV_BLOCK_SIZE:-16}}"
MAX_ACTIVE_REQUESTS="${MAX_ACTIVE_REQUESTS:-${LANGBURST_MAX_ACTIVE_REQUESTS:-1}}"
KV_BLOCKS="${KV_BLOCKS:-$(( ((CONTEXT_WINDOW + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE) * MAX_ACTIVE_REQUESTS ))}"
MAX_QUEUED_REQUESTS="${MAX_QUEUED_REQUESTS:-8}"
MAX_STATE_POOL_SIZE="${MAX_STATE_POOL_SIZE:-${LANGBURST_MAX_STATE_POOL_SIZE:-1}}"
LOG_DIR="${LOG_DIR:-/tmp}"

cd /home/user/workspace/neurova/langburst
source ~/miniconda3/etc/profile.d/conda.sh
conda activate langburst
source /home/user/workspace/neurova/langburst/scripts/langburst_cuda_env.sh

if ! ss -ltn | grep -q ":${SERVER_PORT} "; then
  nohup env \
    CUDA_HOME="$CUDA_HOME" \
    CUDACXX="$CUDACXX" \
    PATH="$PATH" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    LANGBURST_REQUIRE_CUDA_EXT="$LANGBURST_REQUIRE_CUDA_EXT" \
    LANGBURST_SERVE_BATCH="${LANGBURST_SERVE_BATCH:-1}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m langburst.server \
      --adapter qwen36 \
      --hf-model "$MODEL_DIR" \
      --qb-model "$QB_DIR" \
      --model-name "$MODEL_NAME" \
      --host 0.0.0.0 \
      --port "$SERVER_PORT" \
      --recent-window "$CONTEXT_WINDOW" \
      --kv-cache-dtype "$KV_CACHE_DTYPE" \
      --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
      --kv-block-size "$KV_BLOCK_SIZE" \
      --kv-blocks "$KV_BLOCKS" \
      --max-active-requests "$MAX_ACTIVE_REQUESTS" \
      --max-queued-requests "$MAX_QUEUED_REQUESTS" \
      --max-state-pool-size "$MAX_STATE_POOL_SIZE" \
    > "${LOG_DIR}/langburst_server.log" 2>&1 &
  echo $! > "${LOG_DIR}/langburst_server.pid"
fi

for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${SERVER_PORT}/v1/models" >/dev/null; then
    break
  fi
  sleep 1
done

if ! curl -fsS "http://127.0.0.1:${SERVER_PORT}/v1/models" >/dev/null; then
  echo "langburst server did not become ready; see ${LOG_DIR}/langburst_server.log" >&2
  exit 1
fi

if ! ss -ltn | grep -q ":${OPENWEBUI_COMPAT_PORT} "; then
  nohup socat \
    "TCP-LISTEN:${OPENWEBUI_COMPAT_PORT},fork,reuseaddr" \
    "TCP:127.0.0.1:${SERVER_PORT}" \
    > "${LOG_DIR}/langburst_openwebui_proxy.log" 2>&1 &
  echo $! > "${LOG_DIR}/langburst_openwebui_proxy.pid"
fi

echo "LangBurst: http://127.0.0.1:${SERVER_PORT}/v1"
echo "OpenWebUI-compatible URL: http://host.docker.internal:${OPENWEBUI_COMPAT_PORT}/v1"
curl -fsS "http://127.0.0.1:${OPENWEBUI_COMPAT_PORT}/v1/models"
echo
