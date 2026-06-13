#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODE="${MODE:-mamba3-recall-r2-tiny}"
TOKENIZER="${TOKENIZER:-llama31}"
SEQ_LEN="${SEQ_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EXTRA_STEPS="${EXTRA_STEPS:-2400}"
LR="${LR:-1e-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
OPTIMIZER="${OPTIMIZER:-adamw8bit}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
BASE_DATA="${BASE_DATA:-data/splits/base_first_train.txt}"
VALID_DATA="${VALID_DATA:-data/splits/base_first_valid.txt}"
START_CHECKPOINT="${START_CHECKPOINT:-runs/mamba3_recall_r2_base_first_v1/base.pt}"
CHECKPOINT="${CHECKPOINT:-runs/mamba3_recall_r2_base_first_v1/base_long.pt}"
RUN_DIR="${RUN_DIR:-$(dirname "${CHECKPOINT}")}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/base_long_train.log}"

cd "${ROOT}"
mkdir -p "${RUN_DIR}"
exec > >(tee "${LOG_FILE}") 2>&1

echo "== base continuation metadata =="
python - <<'PY'
import json, os, platform, subprocess, time
payload = {
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "hostname": platform.node(),
    "mode": os.environ.get("MODE", "mamba3-recall-r2-tiny"),
    "tokenizer": os.environ.get("TOKENIZER", "llama31"),
    "seq_len": os.environ.get("SEQ_LEN", "128"),
    "batch_size": os.environ.get("BATCH_SIZE", "1"),
    "extra_steps": os.environ.get("EXTRA_STEPS", "2400"),
    "lr": os.environ.get("LR", "1e-4"),
    "grad_accum_steps": os.environ.get("GRAD_ACCUM_STEPS", "4"),
    "optimizer": os.environ.get("OPTIMIZER", "adamw8bit"),
    "base_data": os.environ.get("BASE_DATA", "data/splits/base_first_train.txt"),
    "valid_data": os.environ.get("VALID_DATA", "data/splits/base_first_valid.txt"),
    "start_checkpoint": os.environ.get("START_CHECKPOINT", "runs/mamba3_recall_r2_base_first_v1/base.pt"),
    "checkpoint": os.environ.get("CHECKPOINT", "runs/mamba3_recall_r2_base_first_v1/base_long.pt"),
}
try:
    payload["git_head"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    payload["git_head"] = "unavailable"
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

if [[ ! -f "${CHECKPOINT}" ]]; then
  if [[ ! -f "${START_CHECKPOINT}" ]]; then
    echo "missing START_CHECKPOINT=${START_CHECKPOINT}" >&2
    exit 2
  fi
  cp "${START_CHECKPOINT}" "${CHECKPOINT}"
fi

echo "== base continuation training =="
python -m mamba3_kr.cli train-packed \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --steps "${EXTRA_STEPS}" \
  --lr "${LR}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --optimizer "${OPTIMIZER}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --data "${BASE_DATA}" \
  --checkpoint "${CHECKPOINT}" \
  --save-every 200

echo "== base validation loss =="
python -m mamba3_kr.cli eval-loss \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --data "${VALID_DATA}" \
  --checkpoint "${CHECKPOINT}" \
  --batches 32

echo "== base English quality probe =="
python -m mamba3_kr.cli quality-gate \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --max-new 32 \
  --seq-len "${SEQ_LEN}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --top-k 1 \
  --top-p 0 \
  --temperature 1.0 \
  --repetition-penalty 1.0 || true
