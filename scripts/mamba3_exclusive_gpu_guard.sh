#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"

if [[ "${1:-}" != "run" || $# -lt 2 ]]; then
  echo "usage: $0 run <command> [args...]" >&2
  exit 2
fi
shift

autopilot_running=0
watchdog_running=0
training_running=0

if ps -eo pid=,args= | awk '/mamba3_research_autopilot.sh/ && !/mamba3_research_autopilotctl.sh/ && !/awk/ {found=1} END {exit found ? 0 : 1}'; then
  autopilot_running=1
fi
if ps -eo pid=,args= | awk '/mamba3_moe24_trainctl.sh watchdog-loop/ && !/awk/ {found=1} END {exit found ? 0 : 1}'; then
  watchdog_running=1
fi
if ps -eo pid=,args= | awk '/mamba3_kr.cli train-packed --mode mimo-r4-moe-2.4b/ && !/awk/ {found=1} END {exit found ? 0 : 1}'; then
  training_running=1
fi

restart() {
  if [[ "${autopilot_running}" == "1" ]]; then
    GPU_POLICY="${GPU_POLICY:-train_priority}" INTERVAL="${INTERVAL:-120}" scripts/mamba3_research_autopilotctl.sh start >/dev/null 2>&1 || true
  elif [[ "${watchdog_running}" == "1" ]]; then
    scripts/mamba3_moe24_trainctl.sh watchdog-start >/dev/null 2>&1 || true
  elif [[ "${training_running}" == "1" ]]; then
    scripts/mamba3_moe24_trainctl.sh resume >/dev/null 2>&1 || true
  fi
}
trap restart EXIT INT TERM

if [[ "${autopilot_running}" == "1" ]]; then
  scripts/mamba3_research_autopilotctl.sh stop >/dev/null 2>&1 || true
fi
if [[ "${watchdog_running}" == "1" ]]; then
  scripts/mamba3_moe24_trainctl.sh watchdog-stop >/dev/null 2>&1 || true
fi
if [[ "${training_running}" == "1" ]]; then
  scripts/mamba3_moe24_trainctl.sh stop >/dev/null 2>&1 || true
fi

matching_pids() {
  ps -eo pid=,args= | awk '(index($0, "mamba3_research_autopilot.sh") || index($0, "mamba3_moe24_trainctl.sh watchdog-loop") || index($0, "mamba3_train_clean_doc_until_gate.sh") || index($0, "mamba3_kr.cli train-packed --mode mimo-r4-moe-2.4b")) && !index($0, "mamba3_exclusive_gpu_guard.sh") && !index($0, "awk") {print $1}'
}

for _ in {1..10}; do
  pids="$(matching_pids || true)"
  [[ -z "${pids}" ]] && break
  # shellcheck disable=SC2086
  kill -TERM ${pids} 2>/dev/null || true
  sleep 1
done

pids="$(matching_pids || true)"
if [[ -n "${pids}" ]]; then
  # shellcheck disable=SC2086
  kill -KILL ${pids} 2>/dev/null || true
fi

for _ in {1..20}; do
  if [[ -z "$(matching_pids || true)" ]]; then
    break
  fi
  sleep 1
done

"$@"
