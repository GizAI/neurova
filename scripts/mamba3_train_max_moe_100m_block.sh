#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TOKENS_PER_STEP="${TOKENS_PER_STEP:-2048}"
TARGET_TOKENS="${TARGET_TOKENS:-100000000}"
STEPS_PER_ROUND="${STEPS_PER_ROUND:-2000}"
MAX_ROUNDS="${MAX_ROUNDS:-$(((TARGET_TOKENS + TOKENS_PER_STEP * STEPS_PER_ROUND - 1) / (TOKENS_PER_STEP * STEPS_PER_ROUND)))}"

export ROOT
export MODE="${MODE:-mimo-r4-moe-2.4b}"
export RUN_DIR="${RUN_DIR:-${ROOT}/runs/mamba3_clean_doc_base_moe24_v1}"
export CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/base.pt}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export LR="${LR:-8e-6}"
export STEPS_PER_ROUND
export SAVE_EVERY="${SAVE_EVERY:-1000}"
export EVAL_BATCHES="${EVAL_BATCHES:-32}"
export MAX_ROUNDS
export TARGET_LOSS="${TARGET_LOSS:-5.0}"
export MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-16}"
export MAX_REPEATED_WORD_RUN="${MAX_REPEATED_WORD_RUN:-12}"
export MIN_DISTINCT_WORDS="${MIN_DISTINCT_WORDS:-8}"
export SHUFFLE_TEXTS="${SHUFFLE_TEXTS:-1}"
export NO_SAVE_OPTIMIZER="${NO_SAVE_OPTIMIZER:-1}"
export DATA_SEED_BASE="${DATA_SEED_BASE:-997000}"

planned_tokens=$((TOKENS_PER_STEP * STEPS_PER_ROUND * MAX_ROUNDS))
echo "== Mamba-3 max-MoE long base block =="
echo "mode=${MODE}"
echo "run_dir=${RUN_DIR}"
echo "target_tokens=${TARGET_TOKENS}"
echo "planned_tokens=${planned_tokens}"
echo "steps_per_round=${STEPS_PER_ROUND}"
echo "max_rounds=${MAX_ROUNDS}"
echo "lr=${LR}"
echo "save_every=${SAVE_EVERY}"
echo "eval_batches=${EVAL_BATCHES}"
echo "This is still base CLM training. Chat/QA quality is evaluated only after collapse-free continuation is reached."

exec "${ROOT}/scripts/mamba3_train_clean_doc_until_gate.sh"
