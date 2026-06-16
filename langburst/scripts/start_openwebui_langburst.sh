#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/home/user/models/Qwen3.6-27B}"
QB_DIR="${QB_DIR:-/home/user/models/Qwen3.6-27B-qb4-marlin-fused}"
MODEL_NAME="${MODEL_NAME:-langburst-qwen3.6-27b-q4-marlin}"
SERVER_PORT="${SERVER_PORT:-8008}"
OPENWEBUI_COMPAT_PORT="${OPENWEBUI_COMPAT_PORT:-5000}"
RECENT_WINDOW="${RECENT_WINDOW:-16384}"
LOG_DIR="${LOG_DIR:-/tmp}"

cd /home/user/workspace/neurova/langburst
source ~/miniconda3/etc/profile.d/conda.sh
conda activate langburst

if ! ss -ltn | grep -q ":${SERVER_PORT} "; then
  nohup env \
    LANGBURST_REQUIRE_CUDA_EXT=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m langburst.server \
      --adapter qwen36 \
      --hf-model "$MODEL_DIR" \
      --qb-model "$QB_DIR" \
      --model-name "$MODEL_NAME" \
      --host 0.0.0.0 \
      --port "$SERVER_PORT" \
      --recent-window "$RECENT_WINDOW" \
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
