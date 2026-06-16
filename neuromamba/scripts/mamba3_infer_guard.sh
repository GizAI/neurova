#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "run" || $# -lt 2 ]]; then
  echo "usage: $0 run <command> [args...]" >&2
  exit 2
fi
shift

PIDS=""
if [[ "${NEUROVA_MAMBA3_PAUSE_TRAINING:-1}" != "0" ]]; then
  PIDS="$(
    ps -eo pid=,comm=,args= |
      awk '($2 ~ /^python/ || $2 ~ /^python3/) && $0 ~ / -m neuromamba.cli train-(answer|packed|multitask)/ {print $1}' ||
      true
  )"
fi

resume() {
  if [[ -n "$PIDS" ]]; then
    # shellcheck disable=SC2086
    kill -CONT $PIDS 2>/dev/null || true
  fi
}
trap resume EXIT INT TERM

if [[ -n "$PIDS" ]]; then
  # shellcheck disable=SC2086
  kill -STOP $PIDS 2>/dev/null || true
fi

"$@"
