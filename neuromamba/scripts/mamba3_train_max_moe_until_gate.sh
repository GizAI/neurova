#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export ROOT
export MODE="${MODE:-mimo-r4-moe-2.4b}"
export RUN_DIR="${RUN_DIR:-${ROOT}/neuromamba/runs/mamba3_clean_doc_base_moe24_v1}"
export CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/base.pt}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export LR="${LR:-1.5e-5}"
export STEPS_PER_ROUND="${STEPS_PER_ROUND:-500}"
export SAVE_EVERY="${SAVE_EVERY:-250}"
export EVAL_BATCHES="${EVAL_BATCHES:-16}"
export MAX_ROUNDS="${MAX_ROUNDS:-4}"
export TARGET_LOSS="${TARGET_LOSS:-5.0}"
export MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-16}"
export SHUFFLE_TEXTS="${SHUFFLE_TEXTS:-1}"
export NO_SAVE_OPTIMIZER="${NO_SAVE_OPTIMIZER:-1}"
export DATA_SEED_BASE="${DATA_SEED_BASE:-992000}"

exec "${ROOT}/neuromamba/scripts/mamba3_train_clean_doc_until_gate.sh"
