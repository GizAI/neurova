#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/home/user/models/Qwen3.6-27B}"
QB_DIR="${QB_DIR:-/home/user/models/Qwen3.6-27B-langburst-q3}"
MODEL_NAME="${MODEL_NAME:-langburst-qwen3.6-27b-q3}"
LANGBURST_ENGINE="${LANGBURST_ENGINE:-native}"
SERVER_PORT="${SERVER_PORT:-8008}"
OPENWEBUI_COMPAT_PORT="${OPENWEBUI_COMPAT_PORT:-5000}"
ENABLE_MTP="${ENABLE_MTP:-${LANGBURST_ENABLE_MTP:-1}}"
MTP_SPECULATIVE_TOKENS="${MTP_SPECULATIVE_TOKENS:-${LANGBURST_MTP_SPECULATIVE_TOKENS:-1}}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-${LANGBURST_CONTEXT_WINDOW:-${RECENT_WINDOW:-12288}}}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-${LANGBURST_MAX_PROMPT_TOKENS:-$CONTEXT_WINDOW}}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-${LANGBURST_KV_CACHE_DTYPE:-int4_bdr}}"
KV_BLOCK_SIZE="${KV_BLOCK_SIZE:-${LANGBURST_KV_BLOCK_SIZE:-16}}"
PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-${LANGBURST_PREFILL_CHUNK_SIZE:-64}}"
RAW_PREFILL_BLOCK_TOKENS="${RAW_PREFILL_BLOCK_TOKENS:-${LANGBURST_RAW_PREFILL_BLOCK_TOKENS:-$PREFILL_CHUNK_SIZE}}"
CPU_EMBED="${CPU_EMBED:-${LANGBURST_CPU_EMBED:-0}}"
MAX_ACTIVE_REQUESTS="${MAX_ACTIVE_REQUESTS:-${LANGBURST_MAX_ACTIVE_REQUESTS:-1}}"
KV_BLOCKS="${KV_BLOCKS:-$(( ((CONTEXT_WINDOW + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE) * MAX_ACTIVE_REQUESTS ))}"
MAX_QUEUED_REQUESTS="${MAX_QUEUED_REQUESTS:-8}"
MAX_STATE_POOL_SIZE="${MAX_STATE_POOL_SIZE:-${LANGBURST_MAX_STATE_POOL_SIZE:-1}}"
PREFIX_CACHE="${PREFIX_CACHE:-${LANGBURST_PREFIX_CACHE:-off}}"
LOG_DIR="${LOG_DIR:-/tmp}"
CPU_EMBED_ARG=()
case "$(printf '%s' "$CPU_EMBED" | tr '[:upper:]' '[:lower:]')" in
  1|true|on|yes) CPU_EMBED_ARG=(--cpu-embed) ;;
esac
MTP_ARG=(--disable-mtp)
case "$(printf '%s' "$ENABLE_MTP" | tr '[:upper:]' '[:lower:]')" in
  1|true|on|yes) MTP_ARG=(--enable-mtp) ;;
esac

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
    LANGBURST_BATCH_STATE_ARENA="${LANGBURST_BATCH_STATE_ARENA:-1}" \
    LANGBURST_PAGED_KV="${LANGBURST_PAGED_KV:-1}" \
    LANGBURST_PAGED_KV_MIRROR="${LANGBURST_PAGED_KV_MIRROR:-0}" \
    LANGBURST_PAGED_KV_SHADOW="${LANGBURST_PAGED_KV_SHADOW:-1}" \
    LANGBURST_PAGED_ATTENTION_KERNELS="${LANGBURST_PAGED_ATTENTION_KERNELS:-1}" \
    LANGBURST_PAGED_PREFILL_BLOCK="${LANGBURST_PAGED_PREFILL_BLOCK:-1}" \
    LANGBURST_SHORT_PREFILL_SDPA_TOKENS="${LANGBURST_SHORT_PREFILL_SDPA_TOKENS:-0}" \
    LANGBURST_SHORT_PREFILL_SDPA_MIN_FREE_MIB="${LANGBURST_SHORT_PREFILL_SDPA_MIN_FREE_MIB:-384}" \
    LANGBURST_ATTENTION_RECENT_TOKENS="${LANGBURST_ATTENTION_RECENT_TOKENS:-128}" \
    LANGBURST_PAGED_ATTENTION_BACKEND="${LANGBURST_PAGED_ATTENTION_BACKEND:-auto}" \
    LANGBURST_INT4_KV_LAYOUT="${LANGBURST_INT4_KV_LAYOUT:-tiled}" \
    LANGBURST_MARLIN_DIRECT_MAX_BATCH="${LANGBURST_MARLIN_DIRECT_MAX_BATCH:-64}" \
    LANGBURST_MTP_MAX_DRAFT="${LANGBURST_MTP_MAX_DRAFT:-$MTP_SPECULATIVE_TOKENS}" \
    LANGBURST_MTP_ADAPTIVE="${LANGBURST_MTP_ADAPTIVE:-1}" \
    LANGBURST_REQUEST_TIMEOUT_S="${LANGBURST_REQUEST_TIMEOUT_S:-300}" \
    LANGBURST_DEFAULT_MAX_TOKENS="${LANGBURST_DEFAULT_MAX_TOKENS:-1024}" \
    LANGBURST_TRIM_CACHE_AFTER_REQUEST="${LANGBURST_TRIM_CACHE_AFTER_REQUEST:-${LANGBURST_TRIM_CACHE_DURING_PREFILL:-1}}" \
    LANGBURST_TRIM_CACHE_FREE_BELOW_MIB="${LANGBURST_TRIM_CACHE_FREE_BELOW_MIB:-768}" \
    LANGBURST_RAW_PREFILL_BLOCK_TOKENS="$RAW_PREFILL_BLOCK_TOKENS" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m langburst.server \
      --engine "$LANGBURST_ENGINE" \
      --adapter qwen36 \
      --hf-model "$MODEL_DIR" \
      --qb-model "$QB_DIR" \
      --model-name "$MODEL_NAME" \
      --host 0.0.0.0 \
      --port "$SERVER_PORT" \
      "${MTP_ARG[@]}" \
      --mtp-speculative-tokens "$MTP_SPECULATIVE_TOKENS" \
      --recent-window "$CONTEXT_WINDOW" \
      "${CPU_EMBED_ARG[@]}" \
      --kv-cache-dtype "$KV_CACHE_DTYPE" \
      --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
      --kv-block-size "$KV_BLOCK_SIZE" \
      --kv-blocks "$KV_BLOCKS" \
      --max-active-requests "$MAX_ACTIVE_REQUESTS" \
      --max-queued-requests "$MAX_QUEUED_REQUESTS" \
      --admission-timeout-s "${LANGBURST_ADMISSION_TIMEOUT_S:-300}" \
      --reserve-free-vram-mib "${LANGBURST_RESERVE_FREE_VRAM_MIB:-64}" \
      --max-state-pool-size "$MAX_STATE_POOL_SIZE" \
      --batch-prefill-chunk-size "$PREFILL_CHUNK_SIZE" \
      --prefix-cache "$PREFIX_CACHE" \
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
