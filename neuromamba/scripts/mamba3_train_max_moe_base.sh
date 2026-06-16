#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODE="${MODE:-mimo-r4-moe-2.4b}"
TOKENIZER="${TOKENIZER:-llama31}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"

TRAIN_DATA="${TRAIN_DATA:-neuromamba/data/splits/base_doc_cont_v3_train.jsonl}"
VALID_DATA="${VALID_DATA:-neuromamba/data/splits/base_doc_cont_v3_valid.jsonl}"
RUN_DIR="${RUN_DIR:-neuromamba/runs/mamba3_clean_doc_base_moe24_v1}"
BASE_CKPT="${BASE_CKPT:-${RUN_DIR}/base.pt}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/train.log}"

SEQ_LEN="${SEQ_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LR="${LR:-3e-5}"
OPTIMIZER="${OPTIMIZER:-adamw8bit}"
BASE_STEPS="${BASE_STEPS:-1000}"
SAVE_EVERY="${SAVE_EVERY:-250}"
EVAL_BATCHES="${EVAL_BATCHES:-32}"
SHUFFLE_TEXTS="${SHUFFLE_TEXTS:-1}"
DATA_SEED="${DATA_SEED:-424242}"

cd "${ROOT}"
mkdir -p "${RUN_DIR}"
exec > >(tee "${LOG_FILE}") 2>&1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "== max MoE base metadata =="
python - <<'PY'
import json, os, platform, subprocess, time
payload = {
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "hostname": platform.node(),
    "mode": os.environ.get("MODE", "mimo-r4-moe-2.4b"),
    "tokenizer": os.environ.get("TOKENIZER", "llama31"),
    "train_data": os.environ.get("TRAIN_DATA", "neuromamba/data/splits/base_doc_cont_v3_train.jsonl"),
    "valid_data": os.environ.get("VALID_DATA", "neuromamba/data/splits/base_doc_cont_v3_valid.jsonl"),
    "seq_len": os.environ.get("SEQ_LEN", "2048"),
    "batch_size": os.environ.get("BATCH_SIZE", "1"),
    "grad_accum_steps": os.environ.get("GRAD_ACCUM_STEPS", "1"),
    "lr": os.environ.get("LR", "3e-5"),
    "optimizer": os.environ.get("OPTIMIZER", "adamw8bit"),
    "base_steps": os.environ.get("BASE_STEPS", "1000"),
    "save_every": os.environ.get("SAVE_EVERY", "250"),
    "shuffle_texts": os.environ.get("SHUFFLE_TEXTS", "1"),
    "data_seed": os.environ.get("DATA_SEED", "424242"),
    "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
}
try:
    payload["git_head"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    payload["git_head"] = "unavailable"
print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
PY

python neuromamba/scripts/mamba3_validate_doc_continuation_split.py "${TRAIN_DATA}" "${VALID_DATA}"

rm -f "${BASE_CKPT}"

echo "== model info =="
python -m neuromamba.cli model-info \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --device "${DEVICE}"

echo "== raw-document CLM from scratch: max MoE =="
python -m neuromamba.cli train-packed \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${BASE_CKPT}" \
  --data "${TRAIN_DATA}" \
  --steps "${BASE_STEPS}" \
  --lr "${LR}" \
  --save-every "${SAVE_EVERY}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --optimizer "${OPTIMIZER}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --no-resume \
  --no-save-optimizer \
  $( [[ "${SHUFFLE_TEXTS}" == "1" ]] && printf '%s ' --shuffle-texts ) \
  --data-seed "${DATA_SEED}"

echo "== held-out doc continuation loss =="
python -m neuromamba.cli eval-loss \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${BASE_CKPT}" \
  --data "${VALID_DATA}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --batches "${EVAL_BATCHES}"

echo "== raw continuation decode probe =="
python -m neuromamba.cli fast-generate \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${BASE_CKPT}" \
  --prompt '<doc source="probe" domain="science"> The main idea is' \
  --max-new 96 \
  --seq-len "${SEQ_LEN}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --top-k 1 \
  --top-p 0 \
  --temperature 1.0 \
  --repetition-penalty 1.0 \
  --safe-decode || true

echo "max_moe_base_run=${RUN_DIR}"
