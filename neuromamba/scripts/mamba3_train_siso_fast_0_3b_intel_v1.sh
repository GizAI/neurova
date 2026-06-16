#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"

MODE="${MODE:-mamba3-siso-fast-0.3b-ds128}"
TOKENIZER="${TOKENIZER:-llama31}"
SEQ_LEN="${SEQ_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-12000}"
LR="${LR:-8e-6}"
BASE_ACCUM_STEPS="${BASE_ACCUM_STEPS:-7}"
ANSWER_ACCUM_STEPS="${ANSWER_ACCUM_STEPS:-1}"
ANSWER_LOSS_WEIGHT="${ANSWER_LOSS_WEIGHT:-0.35}"
OPTIMIZER="${OPTIMIZER:-adamw8bit}"
SAVE_EVERY="${SAVE_EVERY:-500}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
RUN_DIR="${RUN_DIR:-neuromamba/runs/mamba3_siso_fast_0_3b_ds128_intel_v1}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/model.pt}"
START_CHECKPOINT="${START_CHECKPOINT:-neuromamba/runs/mamba3_siso_fast_0_3b_ds128_v3/model.pt}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/train.log}"
BASE_DATA="${BASE_DATA:-neuromamba/data/splits/base_doc_cont_v3_train.jsonl}"
FALLBACK_BASE_DATA="${FALLBACK_BASE_DATA:-luma/data/english_completion_bootstrap.txt}"
CHAT_RECORDS="${CHAT_RECORDS:-80000}"
STATE_MEMORY_RECORDS="${STATE_MEMORY_RECORDS:-40000}"
MMLU_REDUX_LIMIT="${MMLU_REDUX_LIMIT:-100}"

mkdir -p "${RUN_DIR}" neuromamba/data/splits
exec > >(tee "${LOG_FILE}") 2>&1

if [[ ! -f "${BASE_DATA}" ]]; then
  BASE_DATA="${FALLBACK_BASE_DATA}"
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  if [[ ! -f "${START_CHECKPOINT}" ]]; then
    echo "missing START_CHECKPOINT=${START_CHECKPOINT}" >&2
    exit 1
  fi
  echo "seed checkpoint ${CHECKPOINT} from ${START_CHECKPOINT}"
  cp "${START_CHECKPOINT}" "${CHECKPOINT}"
fi

echo "== siso fast 0.3b intelligence v1 metadata =="
python - <<'PY'
import json, os, platform, time
payload = {
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "hostname": platform.node(),
    "goal": "Continue 0.3B fast SISO from chat-capable v3 toward real knowledge without benchmark contamination.",
    "mode": os.environ.get("MODE"),
    "tokenizer": os.environ.get("TOKENIZER"),
    "checkpoint": os.environ.get("CHECKPOINT"),
    "start_checkpoint": os.environ.get("START_CHECKPOINT"),
    "base_data": os.environ.get("BASE_DATA"),
    "seq_len": int(os.environ.get("SEQ_LEN", "512")),
    "batch_size": int(os.environ.get("BATCH_SIZE", "8")),
    "steps": int(os.environ.get("STEPS", "12000")),
    "lr": float(os.environ.get("LR", "8e-6")),
    "base_accum_steps": int(os.environ.get("BASE_ACCUM_STEPS", "7")),
    "answer_accum_steps": int(os.environ.get("ANSWER_ACCUM_STEPS", "1")),
    "answer_loss_weight": float(os.environ.get("ANSWER_LOSS_WEIGHT", "0.35")),
    "optimizer": os.environ.get("OPTIMIZER"),
    "save_every": int(os.environ.get("SAVE_EVERY", "500")),
}
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

python -m neuromamba.cli check-contract \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --device "${DEVICE}"

python neuromamba/scripts/mamba3_generate_neurova_chat_sft.py \
  --out neuromamba/data/neurova_siso_fast_intel_chat_v1.jsonl \
  --records "${CHAT_RECORDS}" \
  --seed 20260613

python neuromamba/scripts/mamba3_generate_state_memory_curriculum.py \
  --out neuromamba/data/neurova_siso_fast_intel_state_v1.jsonl \
  --records "${STATE_MEMORY_RECORDS}" \
  --seed 20260614

python neuromamba/scripts/mamba3_make_splits.py \
  --inputs neuromamba/data/neurova_siso_fast_intel_chat_v1.jsonl neuromamba/data/neurova_siso_fast_intel_state_v1.jsonl \
  --train-out neuromamba/data/splits/neurova_siso_fast_intel_v1_train.jsonl \
  --valid-out neuromamba/data/splits/neurova_siso_fast_intel_v1_valid.jsonl \
  --valid-ratio 0.05 \
  --seed 20260613

python -m neuromamba.cli train-multitask \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --steps "${STEPS}" \
  --lr "${LR}" \
  --base-accum-steps "${BASE_ACCUM_STEPS}" \
  --answer-accum-steps "${ANSWER_ACCUM_STEPS}" \
  --answer-loss-weight "${ANSWER_LOSS_WEIGHT}" \
  --optimizer "${OPTIMIZER}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --base-data "${BASE_DATA}" \
  --answer-data neuromamba/data/splits/neurova_siso_fast_intel_v1_train.jsonl \
  --checkpoint "${CHECKPOINT}" \
  --save-every "${SAVE_EVERY}"

python -m neuromamba.cli eval-loss \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --data "${BASE_DATA}" \
  --checkpoint "${CHECKPOINT}" \
  --batches 64 | tee "${RUN_DIR}/eval_base_loss.json"

python -m neuromamba.cli eval-answer-loss \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --data neuromamba/data/splits/neurova_siso_fast_intel_v1_valid.jsonl \
  --checkpoint "${CHECKPOINT}" \
  --batches 64 | tee "${RUN_DIR}/eval_answer_loss.json"

python neuromamba/scripts/mamba3_chat_quality_gate.py \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --seq-len 128 \
  --max-new 48 \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --min-pass-rate 0.70 | tee "${RUN_DIR}/chat_quality_gate.json" || true

python neuromamba/scripts/mamba3_eval_mcq_bench.py \
  --suite smoke \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --seq-len 128 \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --out "${RUN_DIR}/mcq_smoke.json" | tee "${RUN_DIR}/mcq_smoke.stdout" || true

python neuromamba/scripts/mamba3_eval_mcq_bench.py \
  --suite mmlu_redux \
  --mmlu-subject all \
  --redux-filter ok \
  --limit "${MMLU_REDUX_LIMIT}" \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --seq-len 128 \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --out "${RUN_DIR}/mmlu_redux_sample.json" | tee "${RUN_DIR}/mmlu_redux_sample.stdout" || true

echo "checkpoint=${CHECKPOINT}"
