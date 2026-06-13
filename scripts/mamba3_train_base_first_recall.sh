#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODE="${MODE:-mamba3-recall-r2-tiny}"
TOKENIZER="${TOKENIZER:-llama31}"
SEQ_LEN="${SEQ_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
BASE_STEPS="${BASE_STEPS:-1000}"
CURRICULUM_STEPS="${CURRICULUM_STEPS:-200}"
CURRICULUM_LOSS="${CURRICULUM_LOSS:-lm}"
BASE_LR="${BASE_LR:-2e-4}"
CURRICULUM_LR="${CURRICULUM_LR:-8e-5}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
MULTITASK_BASE_ACCUM_STEPS="${MULTITASK_BASE_ACCUM_STEPS:-3}"
MULTITASK_ANSWER_ACCUM_STEPS="${MULTITASK_ANSWER_ACCUM_STEPS:-1}"
MULTITASK_ANSWER_LOSS_WEIGHT="${MULTITASK_ANSWER_LOSS_WEIGHT:-1.0}"
OPTIMIZER="${OPTIMIZER:-adamw8bit}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
RUN_DIR="${RUN_DIR:-runs/mamba3_recall_r2_base_first}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/train.log}"
PROGRAMMATIC_DATA="${PROGRAMMATIC_DATA:-data/mamba3_programmatic_curriculum_base_first.jsonl}"
PROGRAMMATIC_RECORDS="${PROGRAMMATIC_RECORDS:-4000}"
CURRICULUM_TASKS="${CURRICULUM_TASKS:-}"
CURRICULUM_DIFFICULTY="${CURRICULUM_DIFFICULTY:-normal}"
CURRICULUM_MAX_RECORDS="${CURRICULUM_MAX_RECORDS:-1200}"

cd "${ROOT}"
mkdir -p data/splits "${RUN_DIR}"
exec > >(tee "${LOG_FILE}") 2>&1

echo "== base-first metadata =="
python - <<'PY'
import json, os, platform, subprocess, time
payload = {
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "hostname": platform.node(),
    "mode": os.environ.get("MODE", "mamba3-recall-r2-tiny"),
    "tokenizer": os.environ.get("TOKENIZER", "llama31"),
    "seq_len": os.environ.get("SEQ_LEN", "128"),
    "batch_size": os.environ.get("BATCH_SIZE", "1"),
    "base_steps": os.environ.get("BASE_STEPS", "1000"),
    "curriculum_steps": os.environ.get("CURRICULUM_STEPS", "200"),
    "curriculum_loss": os.environ.get("CURRICULUM_LOSS", "lm"),
    "base_lr": os.environ.get("BASE_LR", "2e-4"),
    "curriculum_lr": os.environ.get("CURRICULUM_LR", "8e-5"),
    "grad_accum_steps": os.environ.get("GRAD_ACCUM_STEPS", "4"),
    "multitask_base_accum_steps": os.environ.get("MULTITASK_BASE_ACCUM_STEPS", "3"),
    "multitask_answer_accum_steps": os.environ.get("MULTITASK_ANSWER_ACCUM_STEPS", "1"),
    "multitask_answer_loss_weight": os.environ.get("MULTITASK_ANSWER_LOSS_WEIGHT", "1.0"),
    "optimizer": os.environ.get("OPTIMIZER", "adamw8bit"),
    "programmatic_data": os.environ.get("PROGRAMMATIC_DATA", "data/mamba3_programmatic_curriculum_base_first.jsonl"),
    "curriculum_tasks": os.environ.get("CURRICULUM_TASKS", ""),
    "curriculum_difficulty": os.environ.get("CURRICULUM_DIFFICULTY", "normal"),
    "curriculum_max_records": os.environ.get("CURRICULUM_MAX_RECORDS", "1200"),
}
try:
    payload["git_head"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    payload["git_head"] = "unavailable"
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

echo "== split governed base data =="
if [[ ! -f data/clean_english_sft_bootstrap_v1.jsonl ]]; then
  python scripts/mamba3_generate_clean_english_sft.py \
    --out data/clean_english_sft_bootstrap_v1.jsonl \
    --records 1600 \
    --seed 20260613
fi
python scripts/mamba3_make_splits.py \
  --inputs data/governed_fineweb_edu_sample.jsonl data/governed_dclm_sample.jsonl data/clean_english_sft_bootstrap_v1.jsonl \
  --train-out data/splits/base_first_train.txt \
  --valid-out data/splits/base_first_valid.txt \
  --valid-ratio 0.02 \
  --seed 1337

echo "== generate/split separated recall curriculum =="
GEN_CURRICULUM_ARGS=(
  --out "${PROGRAMMATIC_DATA}"
  --records "${PROGRAMMATIC_RECORDS}"
  --seed 20260613
  --difficulty "${CURRICULUM_DIFFICULTY}"
)
if [[ -n "${CURRICULUM_TASKS}" ]]; then
  GEN_CURRICULUM_ARGS+=(--tasks "${CURRICULUM_TASKS}")
fi
python scripts/mamba3_generate_programmatic_curriculum.py "${GEN_CURRICULUM_ARGS[@]}"
python scripts/mamba3_make_splits.py \
  --inputs "${PROGRAMMATIC_DATA}" \
  --train-out data/splits/base_first_curriculum_train.txt \
  --valid-out data/splits/base_first_curriculum_valid.txt \
  --valid-ratio 0.10 \
  --max-records "${CURRICULUM_MAX_RECORDS}" \
  --seed 2027

BASE_CKPT="${RUN_DIR}/base.pt"
CURRICULUM_CKPT="${RUN_DIR}/curriculum.pt"

echo "== long governed base training =="
python -m mamba3_kr.cli train-packed \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --steps "${BASE_STEPS}" \
  --lr "${BASE_LR}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --optimizer "${OPTIMIZER}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --data data/splits/base_first_train.txt \
  --checkpoint "${BASE_CKPT}" \
  --save-every 100 \
  --no-resume

echo "== base validation loss =="
python -m mamba3_kr.cli eval-loss \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --data data/splits/base_first_valid.txt \
  --checkpoint "${BASE_CKPT}" \
  --batches 16

echo "== base English quality probe =="
python -m mamba3_kr.cli quality-gate \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${BASE_CKPT}" \
  --max-new 32 \
  --seq-len "${SEQ_LEN}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --top-k 1 \
  --top-p 0 \
  --temperature 1.0 \
  --repetition-penalty 1.0 || true

if [[ "${CURRICULUM_STEPS}" != "0" ]]; then
  echo "== recall curriculum training from base =="
  cp "${BASE_CKPT}" "${CURRICULUM_CKPT}"
  if [[ "${CURRICULUM_LOSS}" == "answer" ]]; then
    python -m mamba3_kr.cli train-answer \
      --mode "${MODE}" \
      --tokenizer "${TOKENIZER}" \
      --seq-len "${SEQ_LEN}" \
      --batch-size "${BATCH_SIZE}" \
      --steps "${CURRICULUM_STEPS}" \
      --lr "${CURRICULUM_LR}" \
      --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
      --optimizer "${OPTIMIZER}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}" \
      --data data/splits/base_first_curriculum_train.txt \
      --checkpoint "${CURRICULUM_CKPT}" \
      --save-every 100 \
      --no-resume
  elif [[ "${CURRICULUM_LOSS}" == "multitask" ]]; then
    python -m mamba3_kr.cli train-multitask \
      --mode "${MODE}" \
      --tokenizer "${TOKENIZER}" \
      --seq-len "${SEQ_LEN}" \
      --batch-size "${BATCH_SIZE}" \
      --steps "${CURRICULUM_STEPS}" \
      --lr "${CURRICULUM_LR}" \
      --base-accum-steps "${MULTITASK_BASE_ACCUM_STEPS}" \
      --answer-accum-steps "${MULTITASK_ANSWER_ACCUM_STEPS}" \
      --answer-loss-weight "${MULTITASK_ANSWER_LOSS_WEIGHT}" \
      --optimizer "${OPTIMIZER}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}" \
      --base-data data/splits/base_first_train.txt \
      --answer-data data/splits/base_first_curriculum_train.txt \
      --checkpoint "${CURRICULUM_CKPT}" \
      --save-every 100 \
      --no-resume
  elif [[ "${CURRICULUM_LOSS}" == "lm" ]]; then
    python -m mamba3_kr.cli train-packed \
      --mode "${MODE}" \
      --tokenizer "${TOKENIZER}" \
      --seq-len "${SEQ_LEN}" \
      --batch-size "${BATCH_SIZE}" \
      --steps "${CURRICULUM_STEPS}" \
      --lr "${CURRICULUM_LR}" \
      --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
      --optimizer "${OPTIMIZER}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}" \
      --data data/splits/base_first_curriculum_train.txt \
      --checkpoint "${CURRICULUM_CKPT}" \
      --save-every 100 \
      --no-resume
  else
    echo "Unsupported CURRICULUM_LOSS=${CURRICULUM_LOSS}; expected lm, answer, or multitask" >&2
    exit 2
  fi

  echo "== curriculum validation loss =="
  if [[ "${CURRICULUM_LOSS}" == "answer" || "${CURRICULUM_LOSS}" == "multitask" ]]; then
    python -m mamba3_kr.cli eval-answer-loss \
      --mode "${MODE}" \
      --tokenizer "${TOKENIZER}" \
      --seq-len "${SEQ_LEN}" \
      --batch-size "${BATCH_SIZE}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}" \
      --data data/splits/base_first_curriculum_valid.txt \
      --checkpoint "${CURRICULUM_CKPT}" \
      --batches 16
    if [[ "${CURRICULUM_LOSS}" == "multitask" ]]; then
      python -m mamba3_kr.cli eval-loss \
        --mode "${MODE}" \
        --tokenizer "${TOKENIZER}" \
        --seq-len "${SEQ_LEN}" \
        --batch-size "${BATCH_SIZE}" \
        --device "${DEVICE}" \
        --dtype "${DTYPE}" \
        --data data/splits/base_first_valid.txt \
        --checkpoint "${CURRICULUM_CKPT}" \
        --batches 16
    fi
  else
    python -m mamba3_kr.cli eval-loss \
      --mode "${MODE}" \
      --tokenizer "${TOKENIZER}" \
      --seq-len "${SEQ_LEN}" \
      --batch-size "${BATCH_SIZE}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}" \
      --data data/splits/base_first_curriculum_valid.txt \
      --checkpoint "${CURRICULUM_CKPT}" \
      --batches 16
  fi

  echo "== curriculum exact-match =="
  python scripts/mamba3_eval_programmatic.py \
    --mode "${MODE}" \
    --tokenizer "${TOKENIZER}" \
    --checkpoint "${CURRICULUM_CKPT}" \
    --data "${PROGRAMMATIC_DATA}" \
    --limit 64 \
    --seq-len "${SEQ_LEN}" \
    --max-new 16 \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --top-k 1 \
    --top-p 0 \
    --temperature 1.0 \
    --repetition-penalty 1.0 || true

  echo "== post-curriculum English quality probe =="
  python -m mamba3_kr.cli quality-gate \
    --mode "${MODE}" \
    --tokenizer "${TOKENIZER}" \
    --checkpoint "${CURRICULUM_CKPT}" \
    --max-new 32 \
    --seq-len "${SEQ_LEN}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --top-k 1 \
    --top-p 0 \
    --temperature 1.0 \
    --repetition-penalty 1.0 || true
fi
