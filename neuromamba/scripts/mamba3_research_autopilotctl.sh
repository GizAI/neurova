#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"

AUTOPILOT_DIR="${AUTOPILOT_DIR:-${ROOT}/neuromamba/runs/mamba3_research_autopilot}"
LOG_DIR="${AUTOPILOT_DIR}/logs"
CONTROL_DIR="${AUTOPILOT_DIR}/control"
PID_FILE="${CONTROL_DIR}/autopilot.pid"
mkdir -p "${LOG_DIR}" "${CONTROL_DIR}"

pid_from_file() {
  [[ -f "${PID_FILE}" ]] && cat "${PID_FILE}" || true
}

is_running() {
  local pid
  pid="$(pid_from_file)"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

find_pid() {
  ps -eo pid,cmd | awk '/mamba3_research_autopilot.sh/ && !/mamba3_research_autopilotctl.sh/ && !/awk/ {print $1; exit}'
}

start() {
  local pid stamp log cmdfile
  if is_running; then
    echo "autopilot running pid=$(pid_from_file)"
    exit 0
  fi
  pid="$(find_pid || true)"
  if [[ -n "${pid}" ]]; then
    echo "${pid}" > "${PID_FILE}"
    echo "autopilot already running pid=${pid}"
    exit 0
  fi
  rm -f "${CONTROL_DIR}/stop"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  log="${LOG_DIR}/autopilot_${stamp}.log"
  cmdfile="${CONTROL_DIR}/autopilot_${stamp}.cmd"
  cat > "${cmdfile}" <<EOF
cd "${ROOT}"
export GPU_POLICY="${GPU_POLICY:-train_priority}"
export INTERVAL="${INTERVAL:-120}"
export MAX_LOOPS="${MAX_LOOPS:-0}"
exec neuromamba/scripts/mamba3_research_autopilot.sh
EOF
  setsid bash "${cmdfile}" > "${log}" 2>&1 < /dev/null &
  pid=$!
  echo "${pid}" > "${PID_FILE}"
  ln -sfn "${log}" "${CONTROL_DIR}/current.log"
  printf '%s\n' "${log}" > "${CONTROL_DIR}/current.log.path"
  printf '%s\n' "${cmdfile}" > "${CONTROL_DIR}/current.cmd.path"
  echo "autopilot_pid=${pid}"
  echo "log=${log}"
}

status() {
  echo "== autopilot process =="
  if is_running; then
    ps -p "$(pid_from_file)" -o pid,etime,cmd
  else
    pid="$(find_pid || true)"
    if [[ -n "${pid}" ]]; then
      echo "${pid}" > "${PID_FILE}"
      ps -p "${pid}" -o pid,etime,cmd
    else
      echo "not running"
    fi
  fi
  echo
  echo "== latest snapshot =="
  cat "${CONTROL_DIR}/latest_status.txt" 2>/dev/null || true
  echo
  echo "== latest log =="
  tail -n "${TAIL_LINES:-80}" "${CONTROL_DIR}/current.log" 2>/dev/null || true
}

stop() {
  touch "${CONTROL_DIR}/stop"
  echo "autopilot stop requested"
}

logs() {
  echo "== control =="
  ls -ltr "${CONTROL_DIR}" || true
  echo "== logs =="
  ls -ltr "${LOG_DIR}" | tail -n 30 || true
}

case "${1:-status}" in
  start) start ;;
  status) status ;;
  stop) stop ;;
  logs) logs ;;
  tail) tail -f "${CONTROL_DIR}/current.log" ;;
  *) echo "usage: $0 {start|status|stop|logs|tail}" >&2; exit 2 ;;
esac
