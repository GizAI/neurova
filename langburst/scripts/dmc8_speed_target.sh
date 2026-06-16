#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/home/user/models/Qwen3.6-27B}"
QB_DIR="${QB_DIR:-/home/user/models/Qwen3.6-27B-qb4-marlin-fused}"
RECENT_WINDOW="${RECENT_WINDOW:-${LANGBURST_CONTEXT_WINDOW:-32768}}"
PROMPT="${PROMPT:-Say hello.}"

echo "[langburst target-only]"
LANGBURST_LOWBIT_ROWS_PER_CTA="${LANGBURST_LOWBIT_ROWS_PER_CTA:-4}" \
python -m langburst.generate \
  --hf-model "$MODEL_DIR" \
  --qb-model "$QB_DIR" \
  --device cuda \
  --recent-window "$RECENT_WINDOW" \
  --max-new-tokens "${MAX_NEW_TOKENS:-64}" \
  --temperature 0 \
  --prompt "$PROMPT" \
  --stats
