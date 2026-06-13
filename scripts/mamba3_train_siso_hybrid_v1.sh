#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODE="${MODE:-mamba3-siso-fast-0.3b}"
TOKENIZER="${TOKENIZER:-llama31}"
SEQ_LEN="${SEQ_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-4}"
STEPS="${STEPS:-3000}"
LR="${LR:-2e-5}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
OPTIMIZER="${OPTIMIZER:-adamw8bit}"
SAVE_EVERY="${SAVE_EVERY:-500}"
RESUME="${RESUME:-1}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
RECORDS="${RECORDS:-60000}"
RUN_DIR="${RUN_DIR:-runs/mamba3_siso_fast_0_3b_v1}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/model.pt}"
START_CHECKPOINT="${START_CHECKPOINT:-}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/train.log}"
STATE_MEMORY_RECORDS="${STATE_MEMORY_RECORDS:-$((RECORDS / 3))}"
STATE_MEMORY_TASKS="${STATE_MEMORY_TASKS:-}"

cd "${ROOT}"
mkdir -p "${RUN_DIR}" data/splits
exec > >(tee "${LOG_FILE}") 2>&1

if [[ ! -f "${CHECKPOINT}" && -n "${START_CHECKPOINT}" && -f "${START_CHECKPOINT}" ]]; then
  echo "seed checkpoint ${CHECKPOINT} from ${START_CHECKPOINT}"
  cp "${START_CHECKPOINT}" "${CHECKPOINT}"
fi

echo "== siso hybrid v1 metadata =="
python - <<'PY'
import json, os, platform, time
payload = {
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "hostname": platform.node(),
    "goal": "self-contained Mamba-3 SISO + sparse GQA assistant without external LLM teacher",
    "mode": os.environ.get("MODE", "mamba3-siso-hybrid-0.3b"),
    "tokenizer": os.environ.get("TOKENIZER", "llama31"),
    "checkpoint": os.environ.get("CHECKPOINT", ""),
    "start_checkpoint": os.environ.get("START_CHECKPOINT", ""),
    "records": int(os.environ.get("RECORDS", "60000")),
    "state_memory_records": int(os.environ.get("STATE_MEMORY_RECORDS", "0")),
    "state_memory_tasks": os.environ.get("STATE_MEMORY_TASKS", ""),
    "steps": int(os.environ.get("STEPS", "3000")),
    "seq_len": int(os.environ.get("SEQ_LEN", "128")),
    "batch_size": int(os.environ.get("BATCH_SIZE", "4")),
    "grad_accum_steps": int(os.environ.get("GRAD_ACCUM_STEPS", "4")),
    "optimizer": os.environ.get("OPTIMIZER", "adamw8bit"),
    "save_every": int(os.environ.get("SAVE_EVERY", "500")),
    "resume": os.environ.get("RESUME", "1"),
}
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

python -m mamba3_kr.cli check-contract \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --device "${DEVICE}"

python scripts/mamba3_generate_neurova_chat_sft.py \
  --out data/neurova_siso_hybrid_sft_v1.jsonl \
  --records "${RECORDS}" \
  --seed 20260613

python scripts/mamba3_generate_state_memory_curriculum.py \
  --out data/neurova_state_memory_curriculum_v1.jsonl \
  --records "${STATE_MEMORY_RECORDS}" \
  --seed 20260613 \
  $(if [[ -n "${STATE_MEMORY_TASKS}" ]]; then printf '%s %q' '--tasks' "${STATE_MEMORY_TASKS}"; fi)

python scripts/mamba3_make_splits.py \
  --inputs data/neurova_siso_hybrid_sft_v1.jsonl data/neurova_state_memory_curriculum_v1.jsonl \
  --train-out data/splits/neurova_siso_hybrid_v1_train.jsonl \
  --valid-out data/splits/neurova_siso_hybrid_v1_valid.jsonl \
  --valid-ratio 0.05 \
  --seed 20260613

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
  --data data/splits/neurova_siso_hybrid_v1_train.jsonl \
  --checkpoint "${CHECKPOINT}" \
  --save-every "${SAVE_EVERY}" \
  $(if [[ "${RESUME}" != "1" ]]; then printf '%s' '--no-resume'; fi)

python -m mamba3_kr.cli eval-answer-loss \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --data data/splits/neurova_siso_hybrid_v1_valid.jsonl \
  --checkpoint "${CHECKPOINT}" \
  --batches 64 | tee "${RUN_DIR}/eval_answer_loss.json"

python scripts/mamba3_chat_quality_gate.py \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --seq-len "${SEQ_LEN}" \
  --max-new 48 \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --min-pass-rate "${MIN_PASS_RATE:-0.70}" | tee "${RUN_DIR}/chat_quality_gate.json" || true

python scripts/mamba3_recurrent_parity.py \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --seq-len "${SEQ_LEN}" \
  --steps 12 \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" | tee "${RUN_DIR}/recurrent_parity.json" || true

echo "checkpoint=${CHECKPOINT}"
