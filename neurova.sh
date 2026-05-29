#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
MODEL_DIR="${NEUROVA_MODEL_DIR:-.neurova_tiny_llm}"
TTT_MEMORY="${NEUROVA_TTT_MEMORY:-./.neurova_ttt_memory.json}"
EMBEDDING_URL="${EMBEDDING_URL:-http://ml-dmc8:8081/v1/embeddings}"

case "${1:-ttt}" in
  ttt|chat|ttt-chat)
    shift 2>/dev/null || true
    EMBEDDING_URL="$EMBEDDING_URL" \
    NEUROVA_TTT_MEMORY="$TTT_MEMORY" \
    PYTHONPATH=. python -m neurova.ttt_chat_cli "$@"
    ;;
  llm|tiny-llm)
    shift
    PYTHONPATH=. python -m neurova.tiny_llm_cli chat --model-dir "$MODEL_DIR" "$@"
    ;;
  train)
    shift
    PYTHONPATH=. python -m neurova.tiny_llm_cli train --model-dir "$MODEL_DIR" "$@"
    ;;
  *)
    echo "Usage: $0 {ttt|llm|train} [args...]"
    echo "  ttt       TTT Chat with qwen3-embedding memory (default)"
    echo "  llm       Tiny LLM local chat"
    echo "  train     Train Tiny LLM"
    exit 1
    ;;
esac
