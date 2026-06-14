#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"

TOKENIZER_BACKEND="${TOKENIZER_BACKEND:-qwen}"
MODEL_D="${MODEL_D:-768}"
MODEL_LAYERS="${MODEL_LAYERS:-10}"
MODEL_SLOTS="${MODEL_SLOTS:-256}"
SEQ_LEN="${SEQ_LEN:-256}"
BATCH_SIZE="${BATCH_SIZE:-6}"
TOPK="${TOPK:-8}"
CHUNK="${CHUNK:-64}"
LOCAL_HEADS="${LOCAL_HEADS:-0}"
USE_LOCAL_ATTENTION="${USE_LOCAL_ATTENTION:-0}"
COPY_WINDOW="${COPY_WINDOW:-0}"
SAVE_EVERY="${SAVE_EVERY:-200}"

STAGE1="${STAGE1:-runs/luma_stage1_qwen_natural_pre_v1}"
STAGE2="${STAGE2:-runs/luma_stage2_qwen_chat_sft_v1}"
STAGE3="${STAGE3:-runs/luma_stage3_qwen_slotproof_v1}"
STAGE1_RAW_DATA="${STAGE1_RAW_DATA:-data/luma_stage_doc_cont_v1.jsonl}"
if [[ ! -f "${STAGE1_RAW_DATA}" ]]; then
  STAGE1_RAW_DATA=data/luma_stage_raw_cont_v2.jsonl
fi
STAGE1_MIN_PASS="${STAGE1_MIN_PASS:-0.60}"
STAGE2_MIN_PASS="${STAGE2_MIN_PASS:-0.80}"

wait_for_model() {
  local path="$1"
  local label="$2"
  while [[ ! -f "${path}" ]]; do
    echo "waiting for ${label}: ${path}"
    sleep 60
  done
}

wait_for_run_idle() {
  local run="$1"
  local label="$2"
  wait_for_model "${run}/model.pt" "${label}"
  while pgrep -af "luma\\.(train|generate|eval_memory).*${run}" >/dev/null; do
    echo "waiting for ${label} post-train tasks to finish: ${run}"
    sleep 30
  done
}

require_pass_rate() {
  local path="$1"
  local min_rate="$2"
  local label="$3"
  python - "$path" "$min_rate" "$label" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
min_rate = float(sys.argv[2])
label = sys.argv[3]
if not path.exists():
    raise SystemExit(f"{label} gate missing: {path}")
payload = json.loads(path.read_text(encoding="utf-8"))
rate = float(payload.get("pass_rate", 0.0))
print(f"{label}_pass_rate={rate:.4f} min={min_rate:.4f}")
if rate < min_rate:
    raise SystemExit(f"{label} gate failed: pass_rate={rate:.4f} < {min_rate:.4f}")
PY
}

run_stage2() {
  if [[ -f "${STAGE2}/model.pt" ]]; then
    echo "stage2 already complete: ${STAGE2}/model.pt"
    return
  fi
  mkdir -p "${STAGE2}"
  RUN_DIR="${STAGE2}" \
  INIT_FROM="${STAGE1}/model.pt" \
  RECIPE=custom \
  RAW_DATA="${STAGE1_RAW_DATA}" \
  QA_DATA=data/luma_stage_natural_speak_raw_v2.jsonl \
  RAW_DATASET_MODE=packed \
  RAW_ANSWER_ONLY=0 \
  CHAT_DATA=data/luma_stage_chatml_dialogue_v2.jsonl \
  MEMORY_DATA= \
  RAW_WEIGHT=0.15 \
  QA_WEIGHT=0.15 \
  CHAT_WEIGHT=0.70 \
  MEMORY_WEIGHT=0.0 \
  SLOT_PROOF_WEIGHT=0.0 \
  TOKENIZER_BACKEND="${TOKENIZER_BACKEND}" \
  USE_SLOTS=0 \
  USE_LOCAL_ATTENTION="${USE_LOCAL_ATTENTION}" \
  LOCAL_HEADS="${LOCAL_HEADS}" \
  STEPS="${STAGE2_STEPS:-1000}" \
  SEQ_LEN="${SEQ_LEN}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  D_MODEL="${MODEL_D}" \
  LAYERS="${MODEL_LAYERS}" \
  SLOTS="${MODEL_SLOTS}" \
  TOPK="${TOPK}" \
  CHUNK="${CHUNK}" \
  COPY_WINDOW="${COPY_WINDOW}" \
  SAVE_EVERY="${SAVE_EVERY}" \
  RUN_GENERATE=0 \
  RUN_MEMORY_EVAL=0 \
  ./scripts/luma_train_dmc8.sh
  python -m luma.eval_chat_sanity \
    --ckpt "${STAGE2}/model.pt" \
    --out "${STAGE2}/chat_sanity.json" \
    --device cuda \
    --dtype bf16 || true
  require_pass_rate "${STAGE2}/chat_sanity.json" "${STAGE2_MIN_PASS}" "stage2_chat"
}

run_stage3() {
  if [[ -f "${STAGE3}/model.pt" ]]; then
    echo "stage3 already complete: ${STAGE3}/model.pt"
    return
  fi
  mkdir -p "${STAGE3}"
  RUN_DIR="${STAGE3}" \
  INIT_FROM="${STAGE2}/model.pt" \
  RECIPE=custom \
  RAW_DATA="${STAGE1_RAW_DATA}" \
  QA_DATA=data/luma_stage_natural_speak_raw_v2.jsonl \
  RAW_DATASET_MODE=packed \
  RAW_ANSWER_ONLY=0 \
  CHAT_DATA=data/luma_stage_chatml_dialogue_v2.jsonl \
  MEMORY_DATA=data/luma_stage_chatml_slotproof_v2.jsonl \
  RAW_WEIGHT=0.15 \
  QA_WEIGHT=0.10 \
  CHAT_WEIGHT=0.35 \
  MEMORY_WEIGHT=0.25 \
  SLOT_PROOF_WEIGHT=0.10 \
  AB_MARGIN_WEIGHT=0.25 \
  MEMORY_LOGIT_WEIGHT=0.0 \
  TOKENIZER_BACKEND="${TOKENIZER_BACKEND}" \
  USE_SLOTS=1 \
  USE_LOCAL_ATTENTION="${USE_LOCAL_ATTENTION}" \
  LOCAL_HEADS="${LOCAL_HEADS}" \
  STEPS="${STAGE3_STEPS:-3000}" \
  SEQ_LEN="${SEQ_LEN}" \
  BATCH_SIZE="${STAGE3_BATCH_SIZE:-4}" \
  D_MODEL="${MODEL_D}" \
  LAYERS="${MODEL_LAYERS}" \
  SLOTS="${MODEL_SLOTS}" \
  TOPK="${TOPK}" \
  CHUNK="${CHUNK}" \
  COPY_WINDOW="${COPY_WINDOW}" \
  SAVE_EVERY="${SAVE_EVERY}" \
  EVAL_CASES="${EVAL_CASES:-80}" \
  RUN_GENERATE=0 \
  RUN_MEMORY_EVAL=1 \
  ./scripts/luma_train_dmc8.sh
  python -m luma.eval_chat_sanity \
    --ckpt "${STAGE3}/model.pt" \
    --out "${STAGE3}/chat_sanity.json" \
    --device cuda \
    --dtype bf16 || true
  python -m luma.eval_gate \
    --chat "${STAGE3}/chat_sanity.json" \
    --memory "${STAGE3}/memory_ablation_eval.json" \
    --out "${STAGE3}/gate_summary.json" || true
}

if [[ ! -f "${STAGE1}/model.pt" ]]; then
  mkdir -p "${STAGE1}"
  RUN_DIR="${STAGE1}" \
  RECIPE=custom \
  RAW_DATA="${STAGE1_RAW_DATA}" \
  QA_DATA=data/luma_stage_natural_speak_raw_v2.jsonl \
  CHAT_DATA= \
  MEMORY_DATA= \
  RAW_WEIGHT=0.75 \
  QA_WEIGHT=0.25 \
  RAW_DATASET_MODE=packed \
  RAW_ANSWER_ONLY=0 \
  CHAT_WEIGHT=0.0 \
  MEMORY_WEIGHT=0.0 \
  SLOT_PROOF_WEIGHT=0.0 \
  TOKENIZER_BACKEND="${TOKENIZER_BACKEND}" \
  USE_SLOTS=0 \
  USE_LOCAL_ATTENTION="${USE_LOCAL_ATTENTION}" \
  LOCAL_HEADS="${LOCAL_HEADS}" \
  SLOTS=0 \
  TOPK=1 \
  STEPS="${STAGE1_STEPS:-1500}" \
  SEQ_LEN="${SEQ_LEN}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  D_MODEL="${MODEL_D}" \
  LAYERS="${MODEL_LAYERS}" \
  COPY_WINDOW="${COPY_WINDOW}" \
  SAVE_EVERY="${SAVE_EVERY}" \
  RUN_GENERATE=0 \
  RUN_MEMORY_EVAL=0 \
  ./scripts/luma_train_dmc8.sh
  python -m luma.eval_natural_sanity \
    --ckpt "${STAGE1}/model.pt" \
    --out "${STAGE1}/natural_sanity.json" \
    --device cuda \
    --dtype bf16 || true
else
  echo "stage1 already complete: ${STAGE1}/model.pt"
fi
if [[ ! -f "${STAGE1}/natural_sanity.json" ]]; then
  python -m luma.eval_natural_sanity \
    --ckpt "${STAGE1}/model.pt" \
    --out "${STAGE1}/natural_sanity.json" \
    --device cuda \
    --dtype bf16 || true
fi
require_pass_rate "${STAGE1}/natural_sanity.json" "${STAGE1_MIN_PASS}" "stage1_natural"
wait_for_run_idle "${STAGE1}" "stage1"
run_stage2
if [[ ! -f "${STAGE2}/chat_sanity.json" ]]; then
  python -m luma.eval_chat_sanity \
    --ckpt "${STAGE2}/model.pt" \
    --out "${STAGE2}/chat_sanity.json" \
    --device cuda \
    --dtype bf16 || true
fi
require_pass_rate "${STAGE2}/chat_sanity.json" "${STAGE2_MIN_PASS}" "stage2_chat"
wait_for_run_idle "${STAGE2}" "stage2"
run_stage3

echo "pipeline_done stage1=${STAGE1} stage2=${STAGE2} stage3=${STAGE3}"
