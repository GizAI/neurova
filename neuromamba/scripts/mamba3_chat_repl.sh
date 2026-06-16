#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

neuromamba/scripts/mamba3_infer_guard.sh run neuromamba/scripts/mamba3_chat_serverctl.sh start >/dev/null

echo "Neurova chat ready. Type /q to quit."
while true; do
  printf "you> "
  if ! IFS= read -r PROMPT; then
    echo
    break
  fi
  [[ -z "$PROMPT" ]] && continue
  case "$PROMPT" in
    /q|/quit|/exit|quit|exit)
      break
      ;;
  esac
  printf "neurova> "
  neuromamba/scripts/mamba3_infer_guard.sh run python neuromamba/scripts/mamba3_chat_client.py \
    --host "${NEUROVA_MAMBA3_SERVER_HOST:-127.0.0.1}" \
    --port "${NEUROVA_MAMBA3_SERVER_PORT:-8765}" \
    --prompt "$PROMPT" \
    --max-new "${NEUROVA_MAMBA3_MAX_NEW:-24}" \
    --stream
done
