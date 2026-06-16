#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_DIR="${RUN_DIR:-${ROOT}/neuromamba/runs/mamba3_neurova_speak_v1}"
CONTROL_DIR="${CONTROL_DIR:-${RUN_DIR}/control}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"
PATTERN="mamba3_train_neurova_speak_v1.sh"

mkdir -p "${CONTROL_DIR}" "${LOG_DIR}"

usage() {
  cat <<'EOF'
Usage: neuromamba/scripts/mamba3_speak_trainctl.sh <command>

Commands:
  start    Start detached Neurova Speak v1 training if not already running.
  status   Print process, GPU, latest training lines, and latest samples.
  tail     Follow the active log.
  stop     Gracefully terminate the active speaking run.
  logs     List logs and run artifacts.
EOF
}

active_pid() {
  ps -eo pid,cmd | awk -v pat="${PATTERN}" 'index($0, pat) && !index($0, "awk") {print $1; exit}'
}

latest_log() {
  ls -1t "${LOG_DIR}"/*.log 2>/dev/null | head -n 1 || true
}

write_current_log() {
  local log="$1"
  if [[ -n "${log}" && -f "${log}" ]]; then
    printf '%s\n' "${log}" > "${CONTROL_DIR}/current.log.path"
    ln -sfn "${log}" "${CONTROL_DIR}/current.log"
  fi
}

start() {
  local pid log cmdfile stamp
  pid="$(active_pid || true)"
  if [[ -n "${pid}" ]]; then
    echo "Neurova Speak training already running pid=${pid}"
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
export BATCH_SIZE="${BATCH_SIZE:-32}"
export STEPS="${STEPS:-3000}"
export LR="${LR:-4e-5}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export SPEAK_REPEATS="${SPEAK_REPEATS:-120}"
exec neuromamba/scripts/mamba3_train_neurova_speak_v1.sh
EOF
  setsid bash "${cmdfile}" > "${log}" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "${pid}" > "${CONTROL_DIR}/train.pid"
  printf '%s\n' "${cmdfile}" > "${CONTROL_DIR}/current.cmd.path"
  write_current_log "${log}"
  echo "started_pid=${pid}"
  echo "log=${log}"
  echo "cmdfile=${cmdfile}"
}

status() {
  local pid log
  pid="$(active_pid || true)"
  log=""
  if [[ -f "${CONTROL_DIR}/current.log.path" ]]; then
    log="$(cat "${CONTROL_DIR}/current.log.path")"
  fi
  if [[ -z "${log}" || ! -f "${log}" ]]; then
    log="$(latest_log)"
  fi
  echo "== process =="
  if [[ -n "${pid}" ]]; then
    ps -p "${pid}" -o pid,etime,cmd
  else
    echo "not running"
  fi
  echo "== gpu =="
  nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true
  if [[ -n "${log}" && -f "${log}" ]]; then
    write_current_log "${log}"
    echo "== log: ${log} =="
    tail -n "${TAIL_LINES:-50}" "${log}"
  fi
  if [[ -f "${RUN_DIR}/samples.txt" ]]; then
    echo "== latest samples =="
    tail -n 80 "${RUN_DIR}/samples.txt"
  fi
}

tail_log() {
  local log
  log=""
  if [[ -f "${CONTROL_DIR}/current.log.path" ]]; then
    log="$(cat "${CONTROL_DIR}/current.log.path")"
  fi
  if [[ -z "${log}" || ! -f "${log}" ]]; then
    log="$(latest_log)"
  fi
  if [[ -z "${log}" || ! -f "${log}" ]]; then
    echo "No log found." >&2
    exit 1
  fi
  tail -f "${log}"
}

stop() {
  local pid
  pid="$(active_pid || true)"
  if [[ -z "${pid}" ]]; then
    echo "not running"
    return
  fi
  kill -TERM "${pid}" 2>/dev/null || true
  echo "stop requested pid=${pid}"
}

logs() {
  echo "== control =="
  ls -ltr "${CONTROL_DIR}" || true
  echo "== logs =="
  ls -ltr "${LOG_DIR}" || true
  echo "== run dir =="
  find "${RUN_DIR}" -maxdepth 2 -type f | sort || true
}

cmd="${1:-status}"
case "${cmd}" in
  start) start ;;
  status) status ;;
  tail) tail_log ;;
  stop) stop ;;
  logs) logs ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
