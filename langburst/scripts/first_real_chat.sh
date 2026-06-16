#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   HF_MODEL=/models/Qwen3.6-27B QB_MODEL=/models/Qwen3.6-27B-qb4 ./scripts/first_real_chat.sh

cd "$(dirname "$0")/.."
: "${HF_MODEL:?set HF_MODEL=/path/to/original/HF/Qwen3.6-27B}"
: "${QB_MODEL:?set QB_MODEL=/path/to/converted/langburst-lowbit}"
DEVICE="${DEVICE:-cuda}"
PROMPT="${PROMPT:-안녕. 너는 누구야? 한 문단으로 대답해.}"

langburst-audit "$QB_MODEL" --hf-model "$HF_MODEL"
langburst-chat \
  --hf-model "$HF_MODEL" \
  --qb-model "$QB_MODEL" \
  --device "$DEVICE" \
  --recent-window "${RECENT_WINDOW:-${LANGBURST_CONTEXT_WINDOW:-32768}}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-96}" \
  --temperature "${TEMPERATURE:-0}" \
  --prompt "$PROMPT" \
  --stream
