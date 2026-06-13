#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"

AUTOPILOT_DIR="${AUTOPILOT_DIR:-${ROOT}/runs/mamba3_research_autopilot}"
LOG_DIR="${AUTOPILOT_DIR}/logs"
CONTROL_DIR="${AUTOPILOT_DIR}/control"
mkdir -p "${LOG_DIR}" "${CONTROL_DIR}"

GPU_POLICY="${GPU_POLICY:-train_priority}"
INTERVAL="${INTERVAL:-120}"
MAX_LOOPS="${MAX_LOOPS:-0}"
MAMBA3_SERVER_PORTS="${MAMBA3_SERVER_PORTS:-8765 8767}"
MOE_RUN_DIR="${MOE_RUN_DIR:-${ROOT}/runs/mamba3_clean_doc_base_moe24_v1}"
MOE_CTL="${MOE_CTL:-${ROOT}/scripts/mamba3_moe24_trainctl.sh}"

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

log_json() {
  local event="$1"
  shift || true
  python - "$event" "$@" <<'PY'
import json
import sys
from datetime import datetime, timezone

event = sys.argv[1]
extra = {}
for item in sys.argv[2:]:
    if "=" in item:
        k, v = item.split("=", 1)
        extra[k] = v
payload = {"time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "event": event, **extra}
print(json.dumps(payload, ensure_ascii=False), flush=True)
PY
}

training_pid() {
  ps -eo pid,cmd | awk '/mamba3_kr.cli train-packed --mode mimo-r4-moe-2.4b/ && !/awk/ {print $1; exit}'
}

watchdog_pid() {
  ps -eo pid,cmd | awk '/mamba3_moe24_trainctl.sh watchdog-loop/ && !/awk/ {print $1; exit}'
}

server_pids() {
  ps -eo pid,cmd | awk '/scripts\/mamba3_chat_server.py/ && !/awk/ {print $1}'
}

stop_servers_for_training() {
  local port
  for port in ${MAMBA3_SERVER_PORTS}; do
    NEUROVA_MAMBA3_SERVER_PORT="${port}" scripts/mamba3_chat_serverctl.sh stop >/dev/null 2>&1 || true
  done
  local pid
  for pid in $(server_pids); do
    kill -TERM "${pid}" >/dev/null 2>&1 || true
  done
}

ensure_watchdog() {
  if [[ -z "$(watchdog_pid || true)" ]]; then
    "${MOE_CTL}" watchdog-start
    return 10
  fi
  return 0
}

ensure_training() {
  local watchdog_started=0
  if [[ "${GPU_POLICY}" == "train_priority" ]]; then
    stop_servers_for_training
  fi
  ensure_watchdog || watchdog_started=$?
  if [[ "${watchdog_started}" == "10" ]]; then
    # The watchdog owns the first resume after startup. Calling resume here too
    # can race and load the 2.4B checkpoint twice on a 16GB GPU.
    return 0
  fi
  if [[ -z "$(training_pid || true)" ]]; then
    "${MOE_CTL}" resume
  fi
}

write_snapshot() {
  local snapshot="${CONTROL_DIR}/latest_status.txt"
  {
    echo "== autopilot =="
    echo "time_utc=$(timestamp)"
    echo "gpu_policy=${GPU_POLICY}"
    echo "interval=${INTERVAL}"
    echo
    echo "== processes =="
    ps -eo pid,etime,cmd | grep -E "mamba3_chat_server|mamba3_moe24_trainctl.sh watchdog-loop|mamba3_kr.cli train-packed --mode mimo-r4-moe-2.4b" | grep -v grep || true
    echo
    echo "== gpu =="
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
    echo
    echo "== moe status =="
    "${MOE_CTL}" status || true
    echo
    echo "== moe decision =="
    "${MOE_CTL}" decision || true
    echo
    echo "== watchdog =="
    "${MOE_CTL}" watchdog-status || true
  } > "${snapshot}.tmp" 2>&1
  mv "${snapshot}.tmp" "${snapshot}"
}

main() {
  log_json "autopilot_started" gpu_policy="${GPU_POLICY}" interval="${INTERVAL}" max_loops="${MAX_LOOPS}" root="${ROOT}"
  local loops=0
  while true; do
    if [[ -f "${CONTROL_DIR}/stop" ]]; then
      rm -f "${CONTROL_DIR}/stop"
      log_json "stop_file_seen"
      break
    fi
    loops=$((loops + 1))
    log_json "loop_start" loop="${loops}"
    ensure_training || log_json "ensure_training_failed" loop="${loops}"
    write_snapshot || true
    log_json "loop_done" loop="${loops}" training_pid="$(training_pid || true)" watchdog_pid="$(watchdog_pid || true)"
    if [[ "${MAX_LOOPS}" != "0" && "${loops}" -ge "${MAX_LOOPS}" ]]; then
      log_json "max_loops_reached" loops="${loops}"
      break
    fi
    sleep "${INTERVAL}"
  done
  log_json "autopilot_finished"
}

main "$@"
