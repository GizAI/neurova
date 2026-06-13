#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"

RUN_DIR="${RUN_DIR:-${ROOT}/runs/mamba3_siso_fast_0_3b_ds128_no_cheat_knowledge_v1}"
CONTROL_DIR="${RUN_DIR}/control"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${CONTROL_DIR}" "${LOG_DIR}"

active_pid() {
  ps -eo pid=,args= | awk 'index($0, "mamba3_train_0_3b_no_cheat_knowledge_v1.sh") && !index($0, "mamba3_no_cheat_knowledge_trainctl.sh") && !index($0, "awk") {print $1; exit}'
}

latest_log() {
  ls -1t "${LOG_DIR}"/*.log 2>/dev/null | head -n 1 || true
}

case "${1:-status}" in
  start)
    pid="$(active_pid || true)"
    if [[ -n "${pid}" ]]; then
      echo "no-cheat knowledge training already running pid=${pid}"
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
export STEPS="${STEPS:-20000}"
export MAX_RECORDS="${MAX_RECORDS:-200000}"
scripts/mamba3_exclusive_gpu_guard.sh run scripts/mamba3_train_0_3b_no_cheat_knowledge_v1.sh
EOF
    setsid bash "${cmdfile}" > "${log}" 2>&1 < /dev/null &
    echo "$!" > "${CONTROL_DIR}/train.pid"
    printf '%s\n' "${log}" > "${CONTROL_DIR}/current.log.path"
    ln -sfn "${log}" "${CONTROL_DIR}/current.log"
    printf '%s\n' "${cmdfile}" > "${CONTROL_DIR}/current.cmd.path"
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
    ;;
  stop)
    pid="$(active_pid || true)"
    [[ -n "${pid}" ]] && kill "${pid}" || true
    ;;
  tail)
    log=""
    [[ -f "${CONTROL_DIR}/current.log.path" ]] && log="$(cat "${CONTROL_DIR}/current.log.path")"
    [[ -z "${log}" || ! -f "${log}" ]] && log="$(latest_log)"
    [[ -n "${log}" ]] || { echo "no log" >&2; exit 1; }
    tail -f "${log}"
    ;;
  *) echo "usage: $0 {start|status|stop|tail}" >&2; exit 2 ;;
esac
