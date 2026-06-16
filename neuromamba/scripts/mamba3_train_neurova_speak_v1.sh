#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODE="${MODE:-mimo-r4-tiny}"
TOKENIZER="${TOKENIZER:-llama31}"
SEQ_LEN="${SEQ_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-3000}"
LR="${LR:-4e-5}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
OPTIMIZER="${OPTIMIZER:-adamw8bit}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
RUN_DIR="${RUN_DIR:-neuromamba/runs/mamba3_neurova_speak_v1}"
START_CHECKPOINT="${START_CHECKPOINT:-neuromamba/runs/mamba3_tiny/model_mimo_r4_speak.pt}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/sft.pt}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/train.log}"
CONTINUE_EXISTING="${CONTINUE_EXISTING:-0}"

cd "${ROOT}"
mkdir -p "${RUN_DIR}" neuromamba/data/splits
exec > >(tee "${LOG_FILE}") 2>&1

echo "== neurova speak v1 metadata =="
python - <<'PY'
import json, os, platform, time
payload = {
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "hostname": platform.node(),
    "goal": "highest-quality same-day speaking Neurova v1, not SOTA pretraining",
    "mode": os.environ.get("MODE", "mimo-r4-tiny"),
    "tokenizer": os.environ.get("TOKENIZER", "llama31"),
    "start_checkpoint": os.environ.get("START_CHECKPOINT", "neuromamba/runs/mamba3_tiny/model_mimo_r4_speak.pt"),
    "checkpoint": os.environ.get("CHECKPOINT", "neuromamba/runs/mamba3_neurova_speak_v1/sft.pt"),
    "steps": int(os.environ.get("STEPS", "3000")),
    "lr": float(os.environ.get("LR", "4e-5")),
    "seq_len": int(os.environ.get("SEQ_LEN", "128")),
    "batch_size": int(os.environ.get("BATCH_SIZE", "4")),
    "grad_accum_steps": int(os.environ.get("GRAD_ACCUM_STEPS", "2")),
    "optimizer": os.environ.get("OPTIMIZER", "adamw8bit"),
    "continue_existing": os.environ.get("CONTINUE_EXISTING", "0"),
}
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

if [[ ! -f "${START_CHECKPOINT}" ]]; then
  echo "missing START_CHECKPOINT=${START_CHECKPOINT}" >&2
  exit 2
fi

python neuromamba/scripts/mamba3_generate_neurova_speak_sft.py \
  --out neuromamba/data/neurova_speak_sft_v1.jsonl \
  --repeats "${SPEAK_REPEATS:-100}" \
  --seed 20260613

python neuromamba/scripts/mamba3_make_splits.py \
  --inputs neuromamba/data/neurova_speak_sft_v1.jsonl neuromamba/data/basic_english_qa_v1.jsonl neuromamba/data/clean_english_sft_bootstrap_v1.jsonl \
  --train-out neuromamba/data/splits/neurova_speak_v1_train.txt \
  --valid-out neuromamba/data/splits/neurova_speak_v1_valid.txt \
  --valid-ratio 0.08 \
  --seed 20260613

mkdir -p "${RUN_DIR}/checkpoints"
if [[ -f "${CHECKPOINT}" && "${CONTINUE_EXISTING}" == "1" ]]; then
  echo "continuing existing checkpoint=${CHECKPOINT}"
else
  cp "${START_CHECKPOINT}" "${CHECKPOINT}"
  cp "${START_CHECKPOINT}" "${RUN_DIR}/checkpoints/seed.pt"
fi

echo "== train speaking SFT =="
python -m neuromamba.cli train-answer \
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
  --data neuromamba/data/splits/neurova_speak_v1_train.txt \
  --checkpoint "${CHECKPOINT}" \
  --save-every 300 \
  --no-resume

echo "== validation =="
python -m neuromamba.cli eval-answer-loss \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --data neuromamba/data/splits/neurova_speak_v1_valid.txt \
  --checkpoint "${CHECKPOINT}" \
  --batches 32 | tee "${RUN_DIR}/eval_loss.json"

echo "== sample generations =="
for prompt in \
  "Question: Who are you? Answer:" \
  "Instruction: Where is Korea? Answer:" \
  "Instruction: What is machine learning inference? Answer:" \
  "Instruction: 너는 누구야? Answer:"
do
  echo "--- prompt=${prompt}"
  python -m neuromamba.cli fast-generate \
    --mode "${MODE}" \
    --tokenizer "${TOKENIZER}" \
    --checkpoint "${CHECKPOINT}" \
    --prompt "${prompt}" \
    --max-new 48 \
    --seq-len "${SEQ_LEN}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --top-k 40 \
    --top-p 0.9 \
    --temperature 0.8 \
    --repetition-penalty 1.15 \
    --safe-decode | tee -a "${RUN_DIR}/samples.txt"
done

echo "== quality gate =="
python -m neuromamba.cli quality-gate \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --max-new 32 \
  --seq-len "${SEQ_LEN}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --top-k 40 \
  --top-p 0.9 \
  --temperature 0.8 \
  --repetition-penalty 1.15 | tee "${RUN_DIR}/quality_gate.json" || true

echo "checkpoint=${CHECKPOINT}"
