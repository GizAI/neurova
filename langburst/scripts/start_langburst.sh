#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="${LANGBURST_CONFIG:-$REPO_DIR/configs/ml-dmc8-q4.yaml}"

cd "$REPO_DIR"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate langburst
source "$REPO_DIR/scripts/langburst_cuda_env.sh"
eval "$(python -m langburst.config export-shell --config "$CONFIG_FILE")"

_is_true() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|on|yes) return 0 ;;
    *) return 1 ;;
  esac
}

CPU_EMBED_ARG=()
if _is_true "$CPU_EMBED"; then
  CPU_EMBED_ARG=(--cpu-embed)
fi

MTP_ARG=(--disable-mtp)
if _is_true "$ENABLE_MTP"; then
  MTP_ARG=(--enable-mtp)
fi

OVERFLOW_ARG=(--no-allow-context-overflow)
if _is_true "$ALLOW_CONTEXT_OVERFLOW"; then
  OVERFLOW_ARG=(--allow-context-overflow)
fi

if _is_true "$RESTART_LANGBURST"; then
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${SERVER_PORT}/tcp" >/dev/null 2>&1 || true
    for _ in $(seq 1 30); do
      if ! ss -ltn | grep -q ":${SERVER_PORT} "; then
        break
      fi
      sleep 0.2
    done
  fi

  mapfile -t EXISTING_PIDS < <(
    ps -eo pid=,args= \
      | awk -v port="$SERVER_PORT" '
          /python -m langburst.server/ && $0 ~ "--port " port "([^0-9]|$)" {print $1}
        '
  )
  if [ "${#EXISTING_PIDS[@]}" -gt 0 ]; then
    kill "${EXISTING_PIDS[@]}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      if ! ss -ltn | grep -q ":${SERVER_PORT} "; then
        break
      fi
      sleep 1
    done
    if ss -ltn | grep -q ":${SERVER_PORT} "; then
      kill -9 "${EXISTING_PIDS[@]}" 2>/dev/null || true
      sleep 1
    fi
  fi
fi

if ! ss -ltn | grep -q ":${SERVER_PORT} "; then
  rm -f "${LOG_DIR}/langburst_server.pid"

  SERVER_ENV=(
    "CUDA_HOME=$CUDA_HOME"
    "CUDACXX=$CUDACXX"
    "PATH=$PATH"
    "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
  )
  for key in $LANGBURST_SERVER_ENV_KEYS; do
    SERVER_ENV+=("$key=${!key-}")
  done

  nohup env "${SERVER_ENV[@]}" \
    python -m langburst.server \
      --engine "$LANGBURST_ENGINE" \
      --adapter "$LANGBURST_ADAPTER" \
      --hf-model "$MODEL_DIR" \
      --qb-model "$QB_DIR" \
      --model-name "$MODEL_NAME" \
      --host "$SERVER_HOST" \
      --port "$SERVER_PORT" \
      --device "$LANGBURST_DEVICE" \
      --weight-device "$LANGBURST_WEIGHT_DEVICE" \
      "${MTP_ARG[@]}" \
      --mtp-speculative-tokens "$MTP_SPECULATIVE_TOKENS" \
      --recent-window "$CONTEXT_WINDOW" \
      --max-active-requests "$MAX_ACTIVE_REQUESTS" \
      --max-queued-requests "$MAX_QUEUED_REQUESTS" \
      --max-state-pool-size "$MAX_STATE_POOL_SIZE" \
      --max-generation-tokens "$MAX_GENERATION_TOKENS" \
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
      --batch-prefill-chunk-size "$PREFILL_CHUNK_SIZE" \
      --max-prefill-rows-per-batch "$MAX_PREFILL_ROWS_PER_BATCH" \
      --decode-prefill-interleave-steps "$DECODE_PREFILL_INTERLEAVE_STEPS" \
      --exclusive-prefill-tokens "$EXCLUSIVE_PREFILL_TOKENS" \
      --reserve-free-vram-mib "$RESERVE_FREE_VRAM_MIB" \
      --context-tiers "$CONTEXT_TIERS" \
      --context-tier-slots "$CONTEXT_TIER_SLOTS" \
      "${OVERFLOW_ARG[@]}" \
      "${CPU_EMBED_ARG[@]}" \
      --kv-cache-dtype "$KV_CACHE_DTYPE" \
      --prefix-cache "$PREFIX_CACHE" \
    > "${LOG_DIR}/langburst_server.log" 2>&1 < /dev/null &

  SERVER_PID="$!"
  echo "$SERVER_PID" > "${LOG_DIR}/langburst_server.pid"
  sleep 1
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "langburst server exited during startup; see ${LOG_DIR}/langburst_server.log" >&2
    tail -n 120 "${LOG_DIR}/langburst_server.log" >&2 || true
    exit 1
  fi
fi

for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${SERVER_PORT}/v1/models" >/dev/null 2>&1; then
    break
  fi
  if [ -f "${LOG_DIR}/langburst_server.pid" ]; then
    SERVER_PID="$(cat "${LOG_DIR}/langburst_server.pid" 2>/dev/null || true)"
    if [ -n "$SERVER_PID" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "langburst server exited before readiness; see ${LOG_DIR}/langburst_server.log" >&2
      tail -n 120 "${LOG_DIR}/langburst_server.log" >&2 || true
      exit 1
    fi
  fi
  sleep 1
done

if ! curl -fsS "http://127.0.0.1:${SERVER_PORT}/v1/models" >/dev/null; then
  echo "langburst server did not become ready; see ${LOG_DIR}/langburst_server.log" >&2
  exit 1
fi

echo "LangBurst: http://127.0.0.1:${SERVER_PORT}/v1"
curl -fsS "http://127.0.0.1:${SERVER_PORT}/v1/models"
echo
