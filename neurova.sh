#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

TTT_MEMORY="${NEUROVA_TTT_MEMORY:-./.neurova_ttt_memory.json}"
VLLM_URL="${VLLM_URL:-http://ml-dmc8:8081}"

case "${1:-inplace}" in
  inplace|ttt|inplace-ttt)
    shift 2>/dev/null || true
    PYTHONPATH=. python -m neurova.neurova_cli "$@"
    ;;
  qwen|ttt-qwen|chat)
    shift 2>/dev/null || true
    VLLM_URL="$VLLM_URL" PYTHONPATH=. python -m neurova.ttt_qwen_cli "$@"
    ;;
  ttt-chat)
    shift 2>/dev/null || true
    EMBEDDING_URL="${EMBEDDING_URL:-http://ml-dmc8:8081/v1/embeddings}" \
    NEUROVA_TTT_MEMORY="$TTT_MEMORY" \
    PYTHONPATH=. python -m neurova.ttt_chat "$@"
    ;;
  status)
    echo "=== In-Place TTT (Qwen3.5-4B native) ==="
    echo "  ./neurova.sh inplace   (default)"
    echo ""
    echo "=== vLLM TTT-Qwen (Qwen3.5-4B via vLLM) ==="
    echo "  ./neurova.sh qwen      (requires vLLM on ml-dmc8:8081)"
    echo ""
    echo "=== TTT-Chat (embedding memory) ==="
    echo "  ./neurova.sh ttt-chat"
    ;;
  *)
    echo "Usage: $0 {inplace|qwen|ttt-chat|status} [args...]"
    echo "  inplace   In-Place TTT — Qwen3.5-4B native TTT (default)"
    echo "  qwen      TTT-Qwen — vLLM + LoRA TTT"
    echo "  ttt-chat  TTT-Chat — embedding memory"
    echo "  status    Show available modes"
    exit 1
    ;;
esac
