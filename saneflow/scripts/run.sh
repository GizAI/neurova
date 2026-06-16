#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/lib/neurova_shell.sh

if [[ $# -eq 0 ]]; then
  ACTION="chat"
else
  ACTION="${1:-chat}"
  case "$ACTION" in
    chat|generate|eval|status|train|help|-h|--help)
      shift || true
      ;;
    *)
      ACTION="generate"
      ;;
  esac
fi

usage() {
  cat <<'EOF'
SaneFlow runner

Usage:
  ./neurova.sh saneflow help
  ./neurova.sh saneflow chat [prompt]
  ./neurova.sh saneflow generate [prompt]
  ./neurova.sh saneflow eval
  ./neurova.sh saneflow status
  ./neurova.sh saneflow train
EOF
}

HOST="${NEUROVA_SANEFLOW_HOST:-ml-dmc8}"
ROOT="${NEUROVA_SANEFLOW_ROOT:-/home/user/workspace/neurova}"
ENV_NAME="${NEUROVA_SANEFLOW_ENV:-mamba3_siso}"
CHECKPOINT="${NEUROVA_SANEFLOW_CHECKPOINT:-saneflow/runs/saneflow_current/model.pt}"
FALLBACK_CHECKPOINT="${NEUROVA_SANEFLOW_FALLBACK_CHECKPOINT:-saneflow/runs/saneflow_chatml_sft_v9_assistant/model.pt}"
TRAINING_CHECKPOINT="${NEUROVA_SANEFLOW_TRAINING_CHECKPOINT:-saneflow/runs/saneflow_fineweb_edu_base_v3_100m_muon_mem/latest.pt}"
PROMPT="${*:-Explain what a computer is in simple words:}"
MAX_NEW="${NEUROVA_SANEFLOW_MAX_NEW:-160}"
CONTEXT="${NEUROVA_SANEFLOW_CONTEXT:-256}"
TEMP="${NEUROVA_SANEFLOW_TEMP:-0.75}"
TOP_K="${NEUROVA_SANEFLOW_TOP_K:-40}"
TOP_P="${NEUROVA_SANEFLOW_TOP_P:-0.9}"
REP="${NEUROVA_SANEFLOW_REPETITION_PENALTY:-1.08}"
NO_REPEAT="${NEUROVA_SANEFLOW_NO_REPEAT_NGRAM_SIZE:-4}"
DTYPE="${NEUROVA_SANEFLOW_DTYPE:-bf16}"
DECODE_MODE="${NEUROVA_SANEFLOW_DECODE_MODE:-cache}"
CHATML_FLAG=""
if [[ "${NEUROVA_SANEFLOW_CHATML:-1}" == "1" ]]; then
  CHATML_FLAG="--chatml"
fi

CHAT_PROMPT_FLAG=""
if [[ "$ACTION" == "chat" && $# -gt 0 ]]; then
  CHAT_PROMPT_FLAG="--prompt $(neurova_q "$PROMPT")"
fi

remote_exec() { neurova_remote_exec "$HOST" "$ROOT" "$ENV_NAME" "$@"; }
PICK_CHECKPOINT="$(neurova_pick_existing_file_script CKPT "$CHECKPOINT" "$FALLBACK_CHECKPOINT" "$TRAINING_CHECKPOINT")"
COMMON_GENERATE_ARGS="--max-new $(neurova_q "$MAX_NEW") --context $(neurova_q "$CONTEXT") --temperature $(neurova_q "$TEMP") --top-k $(neurova_q "$TOP_K") --top-p $(neurova_q "$TOP_P") --repetition-penalty $(neurova_q "$REP") --no-repeat-ngram-size $(neurova_q "$NO_REPEAT") --decode $(neurova_q "$DECODE_MODE") --device cuda --dtype $(neurova_q "$DTYPE")"

case "$ACTION" in
  help|-h|--help)
    usage
    ;;
  chat)
    remote_exec "$PICK_CHECKPOINT; python saneflow/scripts/saneflow_chat.py --ckpt \"\$CKPT\" $COMMON_GENERATE_ARGS $CHATML_FLAG $CHAT_PROMPT_FLAG"
    ;;
  generate)
    remote_exec "$PICK_CHECKPOINT; python saneflow/scripts/saneflow_generate.py --ckpt \"\$CKPT\" --prompt $(neurova_q "$PROMPT") $COMMON_GENERATE_ARGS $CHATML_FLAG"
    ;;
  eval)
    remote_exec "$PICK_CHECKPOINT; python saneflow/scripts/saneflow_eval_prompts.py --ckpt \"\$CKPT\" --out saneflow/runs/saneflow_latest_prompt_eval.json --max-new 80 --context $(neurova_q "$CONTEXT") --temperature $(neurova_q "$TEMP") --top-k $(neurova_q "$TOP_K") --top-p $(neurova_q "$TOP_P") --decode $(neurova_q "$DECODE_MODE") --device cuda --dtype $(neurova_q "$DTYPE")"
    ;;
  status)
    remote_exec "pgrep -af 'saneflow_train.py|saneflow_autoresearch_loop.sh' || true; bash saneflow/scripts/saneflow_researchctl_dmc8.sh status 2>/dev/null || true"
    ;;
  train)
    remote_exec "chmod +x saneflow/scripts/saneflow_researchctl_dmc8.sh saneflow/scripts/saneflow_autoresearch_loop.sh; saneflow/scripts/saneflow_researchctl_dmc8.sh start-auto"
    ;;
esac
exit $?
