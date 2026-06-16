#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

if [[ $# -gt 0 ]]; then
  case "${1:-}" in
    bf16|4bit)
      export QWEN_MEMORY_MODE="$1"
      shift || true
      ;;
    help|-h|--help)
      cat <<'EOF'
Qwen Memory runner

Usage:
  ./neurova.sh qwen-memory [bf16|4bit]

This is a legacy Qwen embed_tokens + USearch personal-memory layer.
EOF
      exit 0
      ;;
  esac
fi

exec python3 qwen_memory/main.py "$@"
