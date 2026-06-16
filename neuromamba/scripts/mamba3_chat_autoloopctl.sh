#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LOOP_DIR="${LOOP_DIR:-${ROOT}/neuromamba/runs/mamba3_neurova_chat_autoloop}"
CONTROL_DIR="${CONTROL_DIR:-${LOOP_DIR}/control}"
LOG_DIR="${LOG_DIR:-${LOOP_DIR}/logs}"
PATTERN="mamba3_chat_autoloop.sh"

mkdir -p "${CONTROL_DIR}" "${LOG_DIR}"

active_pid() {
  ps -eo pid,cmd | awk -v pat="${PATTERN}" 'index($0, pat) && !index($0, "mamba3_chat_autoloopctl.sh") && !index($0, "awk") {print $1; exit}'
}

start() {
  local pid stamp log cmdfile
  pid="$(active_pid || true)"
  if [[ -n "${pid}" ]]; then
    echo "autoloop already running pid=${pid}"
    exit 0
  fi
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  log="${LOG_DIR}/autoloop_${stamp}.log"
  cmdfile="${CONTROL_DIR}/autoloop_${stamp}.cmd"
  cat >"${cmdfile}" <<EOF
cd "${ROOT}"
source "\${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate mamba3_siso
export LOOP_DIR="${LOOP_DIR}"
export MAX_TRIALS="${MAX_TRIALS:-6}"
export DEADLINE_HOURS="${DEADLINE_HOURS:-24}"
export MIN_PASS_RATE="${MIN_PASS_RATE:-0.70}"
exec neuromamba/scripts/mamba3_chat_autoloop.sh
EOF
  setsid bash "${cmdfile}" >"${log}" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "${pid}" > "${CONTROL_DIR}/autoloop.pid"
  printf '%s\n' "${log}" > "${CONTROL_DIR}/current.log.path"
  printf '%s\n' "${cmdfile}" > "${CONTROL_DIR}/current.cmd.path"
  ln -sfn "${log}" "${CONTROL_DIR}/current.log"
  echo "started_pid=${pid}"
  echo "log=${log}"
}

status() {
  local pid log
  pid="$(active_pid || true)"
  log="$(cat "${CONTROL_DIR}/current.log.path" 2>/dev/null || true)"
  echo "== autoloop process =="
  if [[ -n "${pid}" ]]; then
    ps -p "${pid}" -o pid,etime,cmd
  else
    echo "not running"
  fi
  echo "== gpu =="
  nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true
  if [[ -n "${log}" && -f "${log}" ]]; then
    echo "== autoloop log: ${log} =="
    tail -n "${TAIL_LINES:-80}" "${log}"
  fi
  if [[ -f "${LOOP_DIR}/summary.jsonl" ]]; then
    echo "== summary =="
    tail -n 20 "${LOOP_DIR}/summary.jsonl"
  fi
  if [[ -f "${ROOT}/neuromamba/runs/mamba3_current/autoloop_metadata.json" ]]; then
    echo "== current promotion =="
    cat "${ROOT}/neuromamba/runs/mamba3_current/autoloop_metadata.json"
  fi
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

tail_log() {
  local log
  log="$(cat "${CONTROL_DIR}/current.log.path" 2>/dev/null || true)"
  [[ -n "${log}" && -f "${log}" ]] || { echo "No log found." >&2; exit 1; }
  tail -f "${log}"
}

case "${1:-status}" in
  start) start ;;
  status) status ;;
  stop) stop ;;
  tail) tail_log ;;
  *) echo "Usage: $0 {start|status|stop|tail}" >&2; exit 2 ;;
esac
