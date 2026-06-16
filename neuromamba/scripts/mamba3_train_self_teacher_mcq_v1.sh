#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"

MODE="${MODE:-mamba3-siso-fast-0.3b-ds128}"
TOKENIZER="${TOKENIZER:-llama31}"
RUN_DIR="${RUN_DIR:-neuromamba/runs/mamba3_self_teacher_mcq_v1}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/model.pt}"
START_CHECKPOINT="${START_CHECKPOINT:-neuromamba/runs/mamba3_autonomous_hybrid_research/20260613T184309Z/mamba3-siso-fast-0.3b-ds128/model.pt}"
FALLBACK_CHECKPOINT="${FALLBACK_CHECKPOINT:-neuromamba/runs/mamba3_siso_fast_0_3b_ds128_v3/model.pt}"
TEACHER_PROVIDER="${TEACHER_PROVIDER:-deterministic}"
MCQ_DATA="${MCQ_DATA:-neuromamba/data/no_cheat_mcq_sft_v1.jsonl}"
DEEPSEEK_MCQ_DATA="${DEEPSEEK_MCQ_DATA:-neuromamba/data/deepseek_no_cheat_mcq_sft_v1.jsonl}"
BASE_DATA="${BASE_DATA:-neuromamba/data/splits/no_cheat_knowledge_v1_train.jsonl}"
SEQ_LEN="${SEQ_LEN:-256}"
BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-2500}"
LR="${LR:-8e-6}"
BASE_ACCUM_STEPS="${BASE_ACCUM_STEPS:-1}"
ANSWER_ACCUM_STEPS="${ANSWER_ACCUM_STEPS:-3}"
ANSWER_LOSS_WEIGHT="${ANSWER_LOSS_WEIGHT:-1.0}"
OPTIMIZER="${OPTIMIZER:-adamw8bit}"
SAVE_EVERY="${SAVE_EVERY:-500}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
MCQ_RECORDS="${MCQ_RECORDS:-120000}"
DEEPSEEK_RECORDS="${DEEPSEEK_RECORDS:-20000}"
DEEPSEEK_BATCH_SIZE="${DEEPSEEK_BATCH_SIZE:-8}"
DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-pro}"

export MODE TOKENIZER RUN_DIR CHECKPOINT START_CHECKPOINT FALLBACK_CHECKPOINT
export TEACHER_PROVIDER MCQ_DATA DEEPSEEK_MCQ_DATA BASE_DATA SEQ_LEN BATCH_SIZE STEPS LR
export BASE_ACCUM_STEPS ANSWER_ACCUM_STEPS ANSWER_LOSS_WEIGHT OPTIMIZER SAVE_EVERY DEVICE DTYPE
export MCQ_RECORDS DEEPSEEK_RECORDS DEEPSEEK_BATCH_SIZE DEEPSEEK_MODEL

mkdir -p "${RUN_DIR}" neuromamba/data/splits

if [[ ! -f "${START_CHECKPOINT}" ]]; then
  START_CHECKPOINT="${FALLBACK_CHECKPOINT}"
fi
if [[ ! -f "${START_CHECKPOINT}" ]]; then
  echo "missing START_CHECKPOINT=${START_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -f "${BASE_DATA}" ]]; then
  echo "missing BASE_DATA=${BASE_DATA}; build no-cheat knowledge splits first" >&2
  exit 2
fi

if [[ "${TEACHER_PROVIDER}" == "deepseek" ]]; then
  MCQ_DATA="${DEEPSEEK_MCQ_DATA}"
  export MCQ_DATA
  python neuromamba/scripts/mamba3_generate_deepseek_mcq_sft.py \
    --out "${MCQ_DATA}" \
    --records "${DEEPSEEK_RECORDS}" \
    --batch-size "${DEEPSEEK_BATCH_SIZE}" \
    --model "${DEEPSEEK_MODEL}" \
    --env-file "${DEEPSEEK_ENV_FILE:-.env}" \
    --seed 20260614
elif [[ "${TEACHER_PROVIDER}" == "file" ]]; then
  if [[ ! -f "${MCQ_DATA}" ]]; then
    echo "missing MCQ_DATA=${MCQ_DATA} for TEACHER_PROVIDER=file" >&2
    exit 2
  fi
else
  python neuromamba/scripts/mamba3_generate_no_cheat_mcq_sft.py \
    --out "${MCQ_DATA}" \
    --records "${MCQ_RECORDS}" \
    --seed 20260614
fi

python neuromamba/scripts/mamba3_make_splits.py \
  --inputs "${MCQ_DATA}" \
  --train-out neuromamba/data/splits/no_cheat_mcq_sft_v1_train.jsonl \
  --valid-out neuromamba/data/splits/no_cheat_mcq_sft_v1_valid.jsonl \
  --valid-ratio 0.03 \
  --seed 20260614

cp "${START_CHECKPOINT}" "${CHECKPOINT}"

python - <<'PY'
import json, os, time, platform
provider = os.environ.get("TEACHER_PROVIDER")
deepseek_model = os.environ.get("DEEPSEEK_MODEL")
teacher_note = {
    "deepseek": f"{deepseek_model} generated no-cheat MCQ/rationale; no MMLU/MMLU-Redux training examples",
    "file": "prebuilt teacher/deterministic MCQ mix; no MMLU/MMLU-Redux training examples",
    "deterministic": "deterministic programmatic no-cheat MCQ generator; no MMLU/MMLU-Redux training examples",
}.get(provider, "unknown teacher provider; no MMLU/MMLU-Redux training examples")
payload = {
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "hostname": platform.node(),
    "goal": "self-teacher no-cheat MCQ/rationale post-training",
    "mode": os.environ.get("MODE"),
    "start_checkpoint": os.environ.get("START_CHECKPOINT"),
    "checkpoint": os.environ.get("CHECKPOINT"),
    "base_data": os.environ.get("BASE_DATA"),
    "mcq_data": os.environ.get("MCQ_DATA"),
    "teacher_provider": provider,
    "deepseek_model": deepseek_model,
    "steps": int(os.environ.get("STEPS", "0")),
    "seq_len": int(os.environ.get("SEQ_LEN", "0")),
    "batch_size": int(os.environ.get("BATCH_SIZE", "0")),
    "base_accum_steps": int(os.environ.get("BASE_ACCUM_STEPS", "0")),
    "answer_accum_steps": int(os.environ.get("ANSWER_ACCUM_STEPS", "0")),
    "lr": float(os.environ.get("LR", "0")),
    "teacher": teacher_note,
}
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

python -m neuromamba.cli train-multitask \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --base-data "${BASE_DATA}" \
  --answer-data neuromamba/data/splits/no_cheat_mcq_sft_v1_train.jsonl \
  --steps "${STEPS}" \
  --lr "${LR}" \
  --save-every "${SAVE_EVERY}" \
  --no-resume \
  --base-accum-steps "${BASE_ACCUM_STEPS}" \
  --answer-accum-steps "${ANSWER_ACCUM_STEPS}" \
  --answer-loss-weight "${ANSWER_LOSS_WEIGHT}" \
  --optimizer "${OPTIMIZER}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}"

python -m neuromamba.cli eval-answer-loss \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --data neuromamba/data/splits/no_cheat_mcq_sft_v1_valid.jsonl \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --batches 64 | tee "${RUN_DIR}/eval_mcq_answer_loss.json"

python neuromamba/scripts/mamba3_eval_mcq_bench.py \
  --suite mmlu_redux \
  --mmlu-subject all \
  --redux-filter ok \
  --limit "${MMLU_REDUX_LIMIT:-100}" \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --seq-len 128 \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --out "${RUN_DIR}/mmlu_redux_sample.json" | tee "${RUN_DIR}/mmlu_redux_sample.stdout" || true

python neuromamba/scripts/mamba3_chat_quality_gate.py \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --seq-len 128 \
  --max-new 48 \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --min-pass-rate "${MIN_PASS_RATE:-0.70}" | tee "${RUN_DIR}/chat_quality_gate.json" || true

echo "checkpoint=${CHECKPOINT}"
