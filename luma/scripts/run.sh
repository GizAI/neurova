#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ "${NEUROVA_ALLOW_LUMA:-0}" != "1" ]]; then
  echo "LUMA is stopped and archived; it is not the active Neurova path." >&2
  echo "Use NEUROVA_ALLOW_LUMA=1 only for explicit archive/debug runs." >&2
  exit 2
fi
LUMA_CHECKPOINT="${NEUROVA_LUMA_CHECKPOINT:-luma/runs/luma_current/model.pt}"
if [[ ! -f "$LUMA_CHECKPOINT" ]]; then
  echo "LUMA checkpoint not found." >&2
  echo "Expected: ${NEUROVA_LUMA_CHECKPOINT:-luma/runs/luma_current/model.pt}" >&2
  echo "Promote a current strict-tokenizer checkpoint after training finishes." >&2
  exit 2
fi
DEVICE="${NEUROVA_LUMA_DEVICE:-$(python3 - <<'PY'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
PY
)}"
MAX_NEW="${NEUROVA_LUMA_MAX_NEW:-160}"
CONTEXT="${NEUROVA_LUMA_CONTEXT:-512}"
TEMP="${NEUROVA_LUMA_TEMP:-0.75}"
TOP_K="${NEUROVA_LUMA_TOP_K:-40}"
TOP_P="${NEUROVA_LUMA_TOP_P:-0.9}"
REP="${NEUROVA_LUMA_REPETITION_PENALTY:-1.08}"
NO_REPEAT="${NEUROVA_LUMA_NO_REPEAT_NGRAM:-4}"
DTYPE="${NEUROVA_LUMA_DTYPE:-auto}"
if [[ $# -gt 0 ]]; then
  exec python3 luma/scripts/luma_chat.py \
    --ckpt "$LUMA_CHECKPOINT" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --max-new "$MAX_NEW" \
    --context "$CONTEXT" \
    --temperature "$TEMP" \
    --top-k "$TOP_K" \
    --top-p "$TOP_P" \
    --repetition-penalty "$REP" \
    --no-repeat-ngram "$NO_REPEAT" \
    --prompt "$*"
fi
exec python3 luma/scripts/luma_chat.py \
  --ckpt "$LUMA_CHECKPOINT" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --max-new "$MAX_NEW" \
  --context "$CONTEXT" \
  --temperature "$TEMP" \
  --top-k "$TOP_K" \
  --top-p "$TOP_P" \
  --repetition-penalty "$REP" \
  --no-repeat-ngram "$NO_REPEAT"
