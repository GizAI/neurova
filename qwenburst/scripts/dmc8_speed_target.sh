#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/home/user/models/Qwen3.6-27B}"
QB_DIR="${QB_DIR:-/home/user/models/Qwen3.6-27B-qb3}"
RECENT_WINDOW="${RECENT_WINDOW:-256}"
PROMPT="${PROMPT:-Say hello.}"

echo "[qwenburst target-only]"
QWENBURST_LOWBIT_ROWS_PER_CTA="${QWENBURST_LOWBIT_ROWS_PER_CTA:-4}" \
python -m qwenburst.generate \
  --hf-model "$MODEL_DIR" \
  --qb-model "$QB_DIR" \
  --device cuda \
  --recent-window "$RECENT_WINDOW" \
  --max-new-tokens "${MAX_NEW_TOKENS:-64}" \
  --temperature 0 \
  --prompt "$PROMPT" \
  --stats

cat <<'EOF'

[speculative path]
DFlash is allowed only as a qwenburst draft adapter. Do not switch the target
runtime to vLLM/SGLang for qwenburst speed claims.
EOF
