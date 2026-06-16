#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_DIR="${RUN_DIR:-${ROOT}/neuromamba/runs/mamba3_siso_hybrid_0_3b_v1}"
LOG_DIR="${RUN_DIR}/logs"
AUTOSTAGE_LOG="${LOG_DIR}/autostage.log"

cd "${ROOT}"
mkdir -p "${LOG_DIR}"

active_train_pid() {
  ps -eo pid,cmd | awk '
    (index($0, "mamba3_train_siso_hybrid_v1.sh") || index($0, "train-answer --mode mamba3-siso-hybrid")) &&
    !index($0, "mamba3_siso_hybrid_0_3b_autostage.sh") &&
    !index($0, "awk") {print $1; exit}'
}

wait_for_idle() {
  while true; do
    pid="$(active_train_pid || true)"
    [[ -z "${pid}" ]] && return 0
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) waiting for current 0.3B stage pid=${pid}" | tee -a "${AUTOSTAGE_LOG}"
    sleep "${WAIT_SECONDS:-60}"
  done
}

run_stage() {
  local name="$1"
  shift
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) start stage=${name}" | tee -a "${AUTOSTAGE_LOG}"
  (
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate mamba3_siso
    export RUN_DIR
    export MODE="mamba3-siso-hybrid-0.3b"
    export CHECKPOINT="${RUN_DIR}/model.pt"
    export RESUME=1
    export SAVE_EVERY="${SAVE_EVERY:-500}"
    "$@"
  ) 2>&1 | tee "${LOG_DIR}/${name}.log"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) end stage=${name}" | tee -a "${AUTOSTAGE_LOG}"
}

wait_for_idle

run_stage "stage2_seq256_paraphrase" env \
  SEQ_LEN="${STAGE2_SEQ_LEN:-256}" \
  BATCH_SIZE="${STAGE2_BATCH_SIZE:-8}" \
  GRAD_ACCUM_STEPS="${STAGE2_GRAD_ACCUM_STEPS:-4}" \
  STEPS="${STAGE2_STEPS:-2000}" \
  RECORDS="${STAGE2_RECORDS:-120000}" \
  LR="${STAGE2_LR:-1.5e-5}" \
  neuromamba/scripts/mamba3_train_siso_hybrid_v1.sh

wait_for_idle

run_stage "stage3_seq512_vram" env \
  SEQ_LEN="${STAGE3_SEQ_LEN:-512}" \
  BATCH_SIZE="${STAGE3_BATCH_SIZE:-4}" \
  GRAD_ACCUM_STEPS="${STAGE3_GRAD_ACCUM_STEPS:-8}" \
  STEPS="${STAGE3_STEPS:-2000}" \
  RECORDS="${STAGE3_RECORDS:-160000}" \
  LR="${STAGE3_LR:-1.0e-5}" \
  neuromamba/scripts/mamba3_train_siso_hybrid_v1.sh

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) autostage complete checkpoint=${RUN_DIR}/model.pt" | tee -a "${AUTOSTAGE_LOG}"
