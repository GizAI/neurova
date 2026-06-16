#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_DIR="${RUN_DIR:-${ROOT}/neuromamba/runs/mamba3_siso_fast_0_3b_v1}"
CONTROL_DIR="${CONTROL_DIR:-${RUN_DIR}/control}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"
PATTERN="mamba3_train_siso_hybrid_v1.sh"

mkdir -p "${CONTROL_DIR}" "${LOG_DIR}"

active_pid() {
  ps -eo pid,cmd | awk '
    (index($0, "mamba3_train_siso_hybrid_v1.sh") || index($0, "train-answer --mode mamba3-siso-hybrid")) &&
    !index($0, "mamba3_siso_hybrid_trainctl.sh") &&
    !index($0, "awk") {print $1; exit}'
}

latest_log() {
  ls -1t "${LOG_DIR}"/*.log 2>/dev/null | head -n 1 || true
}

case "${1:-status}" in
  start)
    pid="$(active_pid || true)"
    if [[ -n "${pid}" ]]; then
      echo "siso hybrid training already running pid=${pid}"
      exit 0
    fi
    stamp="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
    log="${LOG:-${LOG_DIR}/${stamp}.log}"
    cmdfile="${CONTROL_DIR}/${stamp}.cmd"
    cat > "${cmdfile}" <<EOF
cd "${ROOT}"
source "\${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate mamba3_siso
export RUN_DIR="${RUN_DIR}"
export MODE="${MODE:-mamba3-siso-fast-0.3b}"
export TOKENIZER="${TOKENIZER:-llama31}"
export SEQ_LEN="${SEQ_LEN:-128}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export STEPS="${STEPS:-3000}"
export LR="${LR:-2e-5}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
export OPTIMIZER="${OPTIMIZER:-adamw8bit}"
export SAVE_EVERY="${SAVE_EVERY:-500}"
export RESUME="${RESUME:-1}"
export RECORDS="${RECORDS:-60000}"
export START_CHECKPOINT="${START_CHECKPOINT:-}"
export STATE_MEMORY_RECORDS="${STATE_MEMORY_RECORDS:-}"
export STATE_MEMORY_TASKS="${STATE_MEMORY_TASKS:-}"
exec neuromamba/scripts/mamba3_train_siso_hybrid_v1.sh
EOF
    setsid bash "${cmdfile}" > "${log}" 2>&1 < /dev/null &
    echo $! > "${CONTROL_DIR}/train.pid"
    printf '%s\n' "${log}" > "${CONTROL_DIR}/current.log.path"
    ln -sfn "${log}" "${CONTROL_DIR}/current.log"
    echo "started_pid=$!"
    echo "log=${log}"
    ;;
  status)
    echo "== process =="
    pid="$(active_pid || true)"
    if [[ -n "${pid}" ]]; then
      ps -p "${pid}" -o pid,etime,cmd
    else
      echo "not running"
    fi
    echo "== gpu =="
    nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true
    log=""
    [[ -f "${CONTROL_DIR}/current.log.path" ]] && log="$(cat "${CONTROL_DIR}/current.log.path")"
    [[ -z "${log}" || ! -f "${log}" ]] && log="$(latest_log)"
    if [[ -n "${log}" && -f "${log}" ]]; then
      echo "== log: ${log} =="
      tail -n "${TAIL_LINES:-80}" "${log}"
    fi
    ;;
  tail)
    log=""
    [[ -f "${CONTROL_DIR}/current.log.path" ]] && log="$(cat "${CONTROL_DIR}/current.log.path")"
    [[ -z "${log}" || ! -f "${log}" ]] && log="$(latest_log)"
    [[ -n "${log}" ]] || { echo "no log" >&2; exit 1; }
    tail -f "${log}"
    ;;
  stop)
    pid="$(active_pid || true)"
    if [[ -n "${pid}" ]]; then
      kill "${pid}" || true
    fi
    ;;
  *)
    echo "usage: $0 {start|status|tail|stop}" >&2
    exit 2
    ;;
esac
