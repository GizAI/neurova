#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"

MODE="${MODE:-mamba3-siso-fast-0.3b-ds128}"
TOKENIZER="${TOKENIZER:-llama31}"
RUN_DIR="${RUN_DIR:-neuromamba/runs/mamba3_siso_fast_0_3b_ds128_no_cheat_knowledge_v1}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/model.pt}"
START_CHECKPOINT="${START_CHECKPOINT:-neuromamba/runs/mamba3_siso_fast_0_3b_ds128_v3/model.pt}"
CORPUS="${CORPUS:-neuromamba/data/no_cheat_knowledge_v1.jsonl}"
SEQ_LEN="${SEQ_LEN:-1024}"
BATCH_SIZE="${BATCH_SIZE:-4}"
STEPS="${STEPS:-20000}"
LR="${LR:-4e-6}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
OPTIMIZER="${OPTIMIZER:-adamw8bit}"
SAVE_EVERY="${SAVE_EVERY:-500}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
MAX_RECORDS="${MAX_RECORDS:-200000}"

mkdir -p "${RUN_DIR}" neuromamba/data/splits

if [[ ! -f "${CORPUS}" ]]; then
  python neuromamba/scripts/mamba3_build_no_cheat_knowledge_corpus.py \
    --out "${CORPUS}" \
    --max-records "${MAX_RECORDS}"
fi

python neuromamba/scripts/mamba3_make_splits.py \
  --inputs "${CORPUS}" \
  --train-out neuromamba/data/splits/no_cheat_knowledge_v1_train.jsonl \
  --valid-out neuromamba/data/splits/no_cheat_knowledge_v1_valid.jsonl \
  --valid-ratio 0.01 \
  --seed 20260613

if [[ ! -f "${CHECKPOINT}" ]]; then
  cp "${START_CHECKPOINT}" "${CHECKPOINT}"
fi

python -m neuromamba.cli train-packed \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --data neuromamba/data/splits/no_cheat_knowledge_v1_train.jsonl \
  --steps "${STEPS}" \
  --lr "${LR}" \
  --save-every "${SAVE_EVERY}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --optimizer "${OPTIMIZER}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --shuffle-texts \
  --data-seed 20260613

python -m neuromamba.cli eval-loss \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --data neuromamba/data/splits/no_cheat_knowledge_v1_valid.jsonl \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --batches 64 | tee "${RUN_DIR}/eval_loss.json"

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

echo "checkpoint=${CHECKPOINT}"
