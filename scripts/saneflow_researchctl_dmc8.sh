#!/usr/bin/env bash
set -euo pipefail

cd "${ROOT:-$HOME/workspace/neurova}"
PYTHON="${SANEFLOW_PYTHON:-python}"

cmd="${1:-status}"
case "$cmd" in
  status)
    "$PYTHON" scripts/saneflow_run.py status dmc8-speak-base-v1 dmc8-chatml-sft-v9
    echo
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader || true
    ;;
  start-auto|auto)
    mkdir -p runs/saneflow_autoresearch
    if pgrep -af "bash scripts/saneflow_autoresearch_loop.sh dmc8" | grep -v pgrep >/dev/null; then
      echo "saneflow autoresearch dmc8 already running"
    else
      [[ -s runs/saneflow_autoresearch/dmc8.out ]] && mv runs/saneflow_autoresearch/dmc8.out "runs/saneflow_autoresearch/dmc8.$(date +%Y%m%d_%H%M%S).out" || true
      nohup bash scripts/saneflow_autoresearch_loop.sh dmc8 \
        > runs/saneflow_autoresearch/dmc8.out 2>&1 &
      echo "started saneflow autoresearch dmc8 pid=$!"
    fi
    ;;
  restart-auto)
    pkill -f "bash scripts/saneflow_autoresearch_loop.sh dmc8" || true
    mkdir -p runs/saneflow_autoresearch
    [[ -s runs/saneflow_autoresearch/dmc8.out ]] && mv runs/saneflow_autoresearch/dmc8.out "runs/saneflow_autoresearch/dmc8.$(date +%Y%m%d_%H%M%S).out" || true
    nohup bash scripts/saneflow_autoresearch_loop.sh dmc8 \
      > runs/saneflow_autoresearch/dmc8.out 2>&1 &
    echo "restarted saneflow autoresearch dmc8 pid=$!"
    ;;
  start-speak)
    "$PYTHON" scripts/saneflow_run.py wait-train dmc8-speak-base-v1 --build-speak
    ;;
  start-sft)
    "$PYTHON" scripts/saneflow_run.py wait-train dmc8-chatml-sft-v9 --build-chatml
    ;;
  *)
    echo "usage: $0 {status|start-auto|restart-auto|start-speak|start-sft}" >&2
    exit 2
    ;;
esac
