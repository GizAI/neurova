#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODE="${MODE:-mimo-r4-tiny}"
TOKENIZER="${TOKENIZER:-llama31}"
SEQ_LEN="${SEQ_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-9000}"
LR="${LR:-2e-5}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
OPTIMIZER="${OPTIMIZER:-adamw8bit}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
RECORDS="${RECORDS:-60000}"
RUN_DIR="${RUN_DIR:-runs/mamba3_neurova_chat_v1}"
START_CHECKPOINT="${START_CHECKPOINT:-runs/mamba3_neurova_speak_v1/sft.pt}"
FALLBACK_CHECKPOINT="${FALLBACK_CHECKPOINT:-runs/mamba3_kr_tiny/model_mimo_r4_speak.pt}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/chat.pt}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/train.log}"
CONTINUE_EXISTING="${CONTINUE_EXISTING:-0}"

cd "${ROOT}"
mkdir -p "${RUN_DIR}/checkpoints" data/splits
exec > >(tee "${LOG_FILE}") 2>&1

if [[ ! -f "${START_CHECKPOINT}" ]]; then
  START_CHECKPOINT="${FALLBACK_CHECKPOINT}"
fi
if [[ ! -f "${START_CHECKPOINT}" ]]; then
  echo "missing START_CHECKPOINT=${START_CHECKPOINT}" >&2
  exit 2
fi

echo "== neurova chat v1 metadata =="
python - <<'PY'
import json, os, platform, time
payload = {
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "hostname": platform.node(),
    "goal": "24h usable Neurova-chat v1, not 2B SOTA pretraining",
    "mode": os.environ.get("MODE", "mimo-r4-tiny"),
    "tokenizer": os.environ.get("TOKENIZER", "llama31"),
    "start_checkpoint": os.environ.get("START_CHECKPOINT", ""),
    "checkpoint": os.environ.get("CHECKPOINT", ""),
    "records": int(os.environ.get("RECORDS", "60000")),
    "steps": int(os.environ.get("STEPS", "9000")),
    "lr": float(os.environ.get("LR", "2e-5")),
    "seq_len": int(os.environ.get("SEQ_LEN", "128")),
    "batch_size": int(os.environ.get("BATCH_SIZE", "32")),
    "grad_accum_steps": int(os.environ.get("GRAD_ACCUM_STEPS", "1")),
    "optimizer": os.environ.get("OPTIMIZER", "adamw8bit"),
}
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

echo "== build chat SFT corpus =="
python scripts/mamba3_generate_neurova_chat_sft.py \
  --out data/neurova_chat_sft_v1.jsonl \
  --records "${RECORDS}" \
  --seed 20260613

python scripts/mamba3_make_splits.py \
  --inputs data/neurova_chat_sft_v1.jsonl \
  --train-out data/splits/neurova_chat_v1_train.jsonl \
  --valid-out data/splits/neurova_chat_v1_valid.jsonl \
  --valid-ratio 0.05 \
  --seed 20260613

if [[ -f "${CHECKPOINT}" && "${CONTINUE_EXISTING}" == "1" ]]; then
  echo "continuing existing checkpoint=${CHECKPOINT}"
else
  cp "${START_CHECKPOINT}" "${CHECKPOINT}"
  cp "${START_CHECKPOINT}" "${RUN_DIR}/checkpoints/seed.pt"
fi

echo "== answer-only chat SFT =="
python -m mamba3_kr.cli train-answer \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --steps "${STEPS}" \
  --lr "${LR}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --optimizer "${OPTIMIZER}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --data data/splits/neurova_chat_v1_train.jsonl \
  --checkpoint "${CHECKPOINT}" \
  --save-every 500 \
  --no-resume

echo "== validation loss =="
python -m mamba3_kr.cli eval-answer-loss \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --data data/splits/neurova_chat_v1_valid.jsonl \
  --checkpoint "${CHECKPOINT}" \
  --batches 64 | tee "${RUN_DIR}/eval_answer_loss.json"

echo "== chat quality gate =="
timeout "${CHAT_GATE_TIMEOUT_SECONDS:-900}" python scripts/mamba3_chat_quality_gate.py \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --seq-len "${SEQ_LEN}" \
  --max-new 48 \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --min-pass-rate "${MIN_PASS_RATE:-0.70}" | tee "${RUN_DIR}/chat_quality_gate.json" || true

echo "== decode tune =="
timeout "${DECODE_TUNE_TIMEOUT_SECONDS:-900}" python scripts/mamba3_decode_tune.py \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --seq-len "${SEQ_LEN}" \
  --max-new 32 \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --out "${RUN_DIR}/decode_tune/latest.json" | tee "${RUN_DIR}/decode_tune.stdout.json"

echo "checkpoint=${CHECKPOINT}"
