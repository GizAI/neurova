#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export ROOT
export MODE="${MODE:-mimo-r4-moe-2.4b}"
export RUN_DIR="${RUN_DIR:-${ROOT}/runs/mamba3_clean_doc_base_moe24_v1}"
export CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/base.pt}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export LR="${LR:-2e-5}"
export STEPS="${STEPS:-250}"
export SAVE_EVERY="${SAVE_EVERY:-250}"
export EVAL_BATCHES="${EVAL_BATCHES:-16}"
export NO_SAVE_OPTIMIZER="${NO_SAVE_OPTIMIZER:-1}"
export SHUFFLE_TEXTS="${SHUFFLE_TEXTS:-1}"

exec "${ROOT}/scripts/mamba3_continue_clean_doc_base.sh"
