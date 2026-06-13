#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"

CONTROL_DIR="${CONTROL_DIR:-runs/mamba3_current/apply_watcher}"
LOG_DIR="${LOG_DIR:-${CONTROL_DIR}/logs}"
PID_FILE="${CONTROL_DIR}/watcher.pid"
PATTERN="mamba3_current_apply_watcher.sh"
mkdir -p "${CONTROL_DIR}" "${LOG_DIR}"

active_pid() {
  ps -eo pid,cmd | awk -v pat="${PATTERN}" 'index($0, pat) && !index($0, "mamba3_current_applyctl.sh") && !index($0, "awk") {print $1; exit}'
}

case "${1:-status}" in
  start)
    pid="$(active_pid || true)"
    if [[ -n "${pid}" ]]; then
      echo "current apply watcher already running pid=${pid}"
      exit 0
    fi
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    log="${LOG_DIR}/watcher_${stamp}.log"
    setsid scripts/mamba3_current_apply_watcher.sh > "${log}" 2>&1 < /dev/null &
    pid=$!
    printf '%s\n' "${pid}" > "${PID_FILE}"
    printf '%s\n' "${log}" > "${CONTROL_DIR}/current.log.path"
    echo "started_pid=${pid}"
    echo "log=${log}"
    ;;
  status)
    pid="$(active_pid || true)"
    echo "== current apply watcher =="
    if [[ -n "${pid}" ]]; then
      ps -p "${pid}" -o pid,etime,cmd
    else
      echo "not running"
    fi
    echo "checkpoint_mtime=$(stat -c '%Y' runs/mamba3_current/model.pt 2>/dev/null || echo missing)"
    echo "last_applied_mtime=$(cat "${CONTROL_DIR}/last_applied_mtime" 2>/dev/null || echo none)"
    echo "last_applied_at=$(cat "${CONTROL_DIR}/last_applied_at" 2>/dev/null || echo none)"
    log="$(cat "${CONTROL_DIR}/current.log.path" 2>/dev/null || true)"
    if [[ -n "${log}" && -f "${log}" ]]; then
      echo "== log: ${log} =="
      tail -n "${TAIL_LINES:-40}" "${log}"
    fi
    ;;
  stop)
    pid="$(active_pid || true)"
    [[ -n "${pid}" ]] && kill -TERM "${pid}" 2>/dev/null || true
    echo "stop requested pid=${pid:-none}"
    ;;
  tail)
    log="$(cat "${CONTROL_DIR}/current.log.path" 2>/dev/null || true)"
    [[ -n "${log}" && -f "${log}" ]] || { echo "no log" >&2; exit 1; }
    tail -f "${log}"
    ;;
  *)
    echo "usage: $0 {start|status|stop|tail}" >&2
    exit 2
    ;;
esac
