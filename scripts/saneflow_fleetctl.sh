#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/user/workspace/neurova}"
DMC8="${SANEFLOW_DMC8_HOST:-ml-dmc8}"
DMC9="${SANEFLOW_DMC9_HOST:-ml-dmc9}"
DMC8_PYTHON="${SANEFLOW_DMC8_PYTHON:-${SANEFLOW_PYTHON:-python}}"
DMC9_PYTHON="${SANEFLOW_DMC9_PYTHON:-${SANEFLOW_PYTHON:-/home/user/miniconda3/envs/saneflow/bin/python}}"

remote() {
  local host="$1"
  shift
  ssh "$host" "cd '$ROOT' && $*"
}

cmd="${1:-status}"
case "$cmd" in
  status)
    echo "== dmc8: active line A, speak-base -> gated SFT =="
    remote "$DMC8" "bash scripts/saneflow_researchctl_dmc8.sh status || true; echo; ps -eo pid,etime,pcpu,pmem,args | grep -E 'saneflow_train.py|saneflow_autoresearch_loop' | grep -v grep || true; echo; tail -5 runs/saneflow_autoresearch/dmc8.out 2>/dev/null || true"
    echo
    echo "== dmc9: active lines B/C, practical-base + dense 0.3B =="
    remote "$DMC9" "bash scripts/saneflow_researchctl_dmc9.sh status || true; echo; ps -eo pid,etime,pcpu,pmem,args | grep -E 'saneflow_train.py|saneflow_autoresearch_loop' | grep -v grep || true; echo; tail -8 runs/saneflow_autoresearch/dmc9.out 2>/dev/null || true"
    ;;
  active)
    echo "== dmc8 active =="
    remote "$DMC8" "$DMC8_PYTHON scripts/saneflow_run.py status dmc8-speak-base-v1 dmc8-chatml-sft-v9; echo; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader || true"
    echo
    echo "== dmc9 active =="
    remote "$DMC9" "$DMC9_PYTHON scripts/saneflow_run.py status dmc9-practical-base-100m dmc9-dense-0.3b-v1; echo; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader || true"
    ;;
  start)
    remote "$DMC8" "bash scripts/saneflow_researchctl_dmc8.sh start-auto"
    remote "$DMC9" "bash scripts/saneflow_researchctl_dmc9.sh start-auto"
    ;;
  logs)
    echo "== dmc8 autoresearch =="
    remote "$DMC8" "tail -80 runs/saneflow_autoresearch/dmc8.out 2>/dev/null || true"
    echo "== dmc9 autoresearch =="
    remote "$DMC9" "tail -80 runs/saneflow_autoresearch/dmc9.out 2>/dev/null || true"
    ;;
  audit)
    remote "$DMC8" "$DMC8_PYTHON scripts/saneflow_data_audit.py --out runs/saneflow_data_audit/dmc8_audit.json --sample-rows 5000"
    remote "$DMC9" "$DMC9_PYTHON scripts/saneflow_data_audit.py --out runs/saneflow_data_audit/dmc9_audit.json --sample-rows 5000"
    ;;
  *)
    echo "usage: $0 {status|active|start|logs|audit}" >&2
    exit 2
    ;;
esac
