#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PROMPT=""
STREAM=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt)
      PROMPT="${2:-}"
      shift 2
      ;;
    --no-stream)
      STREAM=0
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$PROMPT" ]]; then
  echo "missing --prompt" >&2
  exit 2
fi

scripts/mamba3_chat_serverctl.sh start >/dev/null

STREAM_FLAG=()
if [[ "$STREAM" == "1" ]]; then
  STREAM_FLAG=(--stream)
fi

python scripts/mamba3_chat_client.py \
  --host "${NEUROVA_MAMBA3_SERVER_HOST:-127.0.0.1}" \
  --port "${NEUROVA_MAMBA3_SERVER_PORT:-8765}" \
  --prompt "$PROMPT" \
  --max-new "${NEUROVA_MAMBA3_MAX_NEW:-24}" \
  "${STREAM_FLAG[@]}"
