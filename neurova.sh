#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
MODEL_DIR="${NEUROVA_MODEL_DIR:-.neurova_tiny_llm}"
TTT_MEMORY="${NEUROVA_TTT_MEMORY:-./.neurova_ttt_memory.json}"
VLLM_URL="${VLLM_URL:-http://ml-dmc8:8081}"

case "${1:-ttt-qwen}" in
  ttt-qwen|qwen|chat)
    shift 2>/dev/null || true
    VLLM_URL="$VLLM_URL" PYTHONPATH=. python -m neurova.ttt_qwen_cli "$@"
    ;;
  ttt|ttt-chat)
    shift 2>/dev/null || true
    EMBEDDING_URL="${EMBEDDING_URL:-http://ml-dmc8:8081/v1/embeddings}" \
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
    echo "Usage: $0 {ttt-qwen|ttt|llm|train} [args...]"
    echo "  ttt-qwen  Qwen3.5-4B + TTT (default)"
    echo "  ttt       TTT Chat with qwen3-embedding memory"
    echo "  llm       Tiny LLM local chat"
    echo "  train     Train Tiny LLM"
    exit 1
    ;;
esac
