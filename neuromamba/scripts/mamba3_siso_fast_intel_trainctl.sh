#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"

RUN_DIR="${RUN_DIR:-${ROOT}/neuromamba/runs/mamba3_siso_fast_0_3b_ds128_intel_v1}"
CONTROL_DIR="${RUN_DIR}/control"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${CONTROL_DIR}" "${LOG_DIR}"

active_pid() {
  ps -eo pid=,args= | awk '
    index($0, "mamba3_train_siso_fast_0_3b_intel_v1.sh") &&
    !index($0, "mamba3_siso_fast_intel_trainctl.sh") &&
    !index($0, "awk") {print $1; exit}'
}

latest_log() {
  ls -1t "${LOG_DIR}"/*.log 2>/dev/null | head -n 1 || true
}

start() {
  local pid stamp log cmdfile
  pid="$(active_pid || true)"
  if [[ -n "${pid}" ]]; then
    echo "0.3B intelligence training already running pid=${pid}"
    exit 0
  fi
  stamp="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
  log="${LOG:-${LOG_DIR}/${stamp}.log}"
  cmdfile="${CONTROL_DIR}/${stamp}.cmd"
  cat > "${cmdfile}" <<EOF
cd "${ROOT}"
source "\${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate mamba3_siso
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RUN_DIR="${RUN_DIR}"
export MODE="${MODE:-mamba3-siso-fast-0.3b-ds128}"
export TOKENIZER="${TOKENIZER:-llama31}"
export SEQ_LEN="${SEQ_LEN:-512}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export STEPS="${STEPS:-12000}"
export LR="${LR:-8e-6}"
export BASE_ACCUM_STEPS="${BASE_ACCUM_STEPS:-7}"
export ANSWER_ACCUM_STEPS="${ANSWER_ACCUM_STEPS:-1}"
export ANSWER_LOSS_WEIGHT="${ANSWER_LOSS_WEIGHT:-0.35}"
export OPTIMIZER="${OPTIMIZER:-adamw8bit}"
export SAVE_EVERY="${SAVE_EVERY:-500}"
export START_CHECKPOINT="${START_CHECKPOINT:-neuromamba/runs/mamba3_siso_fast_0_3b_ds128_v3/model.pt}"
export BASE_DATA="${BASE_DATA:-neuromamba/data/splits/base_doc_cont_v3_train.jsonl}"
export CHAT_RECORDS="${CHAT_RECORDS:-80000}"
export STATE_MEMORY_RECORDS="${STATE_MEMORY_RECORDS:-40000}"
export MMLU_REDUX_LIMIT="${MMLU_REDUX_LIMIT:-100}"
neuromamba/scripts/mamba3_exclusive_gpu_guard.sh run neuromamba/scripts/mamba3_train_siso_fast_0_3b_intel_v1.sh
EOF
  setsid bash "${cmdfile}" > "${log}" 2>&1 < /dev/null &
  echo "$!" > "${CONTROL_DIR}/train.pid"
  printf '%s\n' "${log}" > "${CONTROL_DIR}/current.log.path"
  ln -sfn "${log}" "${CONTROL_DIR}/current.log"
  printf '%s\n' "${cmdfile}" > "${CONTROL_DIR}/current.cmd.path"
  echo "started_pid=$!"
  echo "log=${log}"
}

status() {
  local pid log
  echo "== process =="
  pid="$(active_pid || true)"
  if [[ -n "${pid}" ]]; then
    ps -p "${pid}" -o pid,etime,cmd
  else
    echo "not running"
  fi
  echo "== gpu =="
  nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true
  echo "== log =="
  log=""
  [[ -f "${CONTROL_DIR}/current.log.path" ]] && log="$(cat "${CONTROL_DIR}/current.log.path")"
  [[ -z "${log}" || ! -f "${log}" ]] && log="$(latest_log)"
  if [[ -n "${log}" && -f "${log}" ]]; then
    echo "${log}"
    tail -n "${TAIL_LINES:-80}" "${log}"
  else
    echo "no log"
  fi
}

stop() {
  local pid
  pid="$(active_pid || true)"
  if [[ -n "${pid}" ]]; then
    kill "${pid}" || true
  fi
}

case "${1:-status}" in
  start) start ;;
  status) status ;;
  tail)
    log=""
    [[ -f "${CONTROL_DIR}/current.log.path" ]] && log="$(cat "${CONTROL_DIR}/current.log.path")"
    [[ -z "${log}" || ! -f "${log}" ]] && log="$(latest_log)"
    [[ -n "${log}" ]] || { echo "no log" >&2; exit 1; }
    tail -f "${log}"
    ;;
  stop) stop ;;
  *) echo "usage: $0 {start|status|tail|stop}" >&2; exit 2 ;;
esac
