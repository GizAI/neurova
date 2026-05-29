#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
EMBEDDING_URL=http://ml-dmc8:8081/v1/embeddings \
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B \
NEUROVA_TTT_MEMORY=./.neurova_ttt_memory.json \
PYTHONPATH=. python -m neurova.ttt_chat_cli "$@"
