#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"

RUN_ROOT="${RUN_ROOT:-neuromamba/runs/mamba3_siso_intel_ablation}"
LOG_DIR="${RUN_ROOT}/logs"
PID_FILE="${RUN_ROOT}/ablation.pid"
mkdir -p "${LOG_DIR}"

case "${1:-status}" in
  start)
    if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
      echo "already running pid=$(cat "${PID_FILE}")"
      exit 0
    fi
    log="${LOG_DIR}/ablation_$(date -u +%Y%m%dT%H%M%SZ).log"
    nohup bash -lc "cd '${ROOT}' && exec neuromamba/scripts/mamba3_siso_intel_ablation_loop.sh" >"${log}" 2>&1 &
    echo $! > "${PID_FILE}"
    echo "started pid=$! log=${log}"
    ;;
  stop)
    if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
      kill "$(cat "${PID_FILE}")"
      echo "stopped pid=$(cat "${PID_FILE}")"
    else
      echo "not running"
    fi
    ;;
  status)
    if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
      echo "running pid=$(cat "${PID_FILE}")"
    else
      echo "not running"
    fi
    latest="$(ls -t "${LOG_DIR}"/ablation_*.log 2>/dev/null | head -n 1 || true)"
    [[ -n "${latest}" ]] && { echo "log=${latest}"; tail -n 40 "${latest}" || true; }
    [[ -f "${RUN_ROOT}/summary.jsonl" ]] && { echo "== summary =="; tail -n 20 "${RUN_ROOT}/summary.jsonl"; }
    true
    ;;
  *)
    echo "usage: $0 {start|stop|status}" >&2
    exit 2
    ;;
esac
