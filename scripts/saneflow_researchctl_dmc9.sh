#!/usr/bin/env bash
set -euo pipefail

cd "${ROOT:-$HOME/workspace/neurova}"
PYTHON="${SANEFLOW_PYTHON:-/home/user/miniconda3/envs/saneflow/bin/python}"

cmd="${1:-status}"
case "$cmd" in
  status)
    "$PYTHON" scripts/saneflow_run.py status dmc9-practical-base-100m dmc9-r-champion-delta-landmark-long dmc9-r-champion-practical-cont
    echo
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader
    ;;
  status-all)
    "$PYTHON" scripts/saneflow_run.py status dmc9-practical-base-100m dmc9-r-champion-delta-landmark-long dmc9-r-champion-practical-cont dmc9-sparse-chatml-sft dmc9-neurova-r-full
    echo
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader
    ;;
  start-chatml-sft)
    "$PYTHON" scripts/saneflow_run.py start dmc9-sparse-chatml-sft
    ;;
  start-neurova-r-small|start-neurova-r-full)
    "$PYTHON" scripts/saneflow_run.py start dmc9-neurova-r-full
    ;;
  start-auto|auto)
    if pgrep -af "bash scripts/saneflow_autoresearch_loop.sh dmc9" | grep -v pgrep >/dev/null; then
      echo "saneflow autoresearch dmc9 already running"
    else
      mkdir -p runs/saneflow_autoresearch
      [[ -s runs/saneflow_autoresearch/dmc9.out ]] && mv runs/saneflow_autoresearch/dmc9.out "runs/saneflow_autoresearch/dmc9.$(date +%Y%m%d_%H%M%S).out" || true
      nohup bash scripts/saneflow_autoresearch_loop.sh dmc9 \
        > runs/saneflow_autoresearch/dmc9.out 2>&1 &
      echo "started saneflow autoresearch dmc9 pid=$!"
    fi
    ;;
  restart-auto)
    pkill -f "bash scripts/saneflow_autoresearch_loop.sh dmc9" || true
    mkdir -p runs/saneflow_autoresearch
    [[ -s runs/saneflow_autoresearch/dmc9.out ]] && mv runs/saneflow_autoresearch/dmc9.out "runs/saneflow_autoresearch/dmc9.$(date +%Y%m%d_%H%M%S).out" || true
    nohup bash scripts/saneflow_autoresearch_loop.sh dmc9 \
      > runs/saneflow_autoresearch/dmc9.out 2>&1 &
    echo "restarted saneflow autoresearch dmc9 pid=$!"
    ;;
  eval-latest)
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON" scripts/saneflow_reasoning_gate.py \
      --ckpt runs/saneflow_100m_research_v4/v2_fixed_sparse_chatml_masked_sft_v3/latest.pt \
      --out runs/saneflow_100m_research_v4/v2_fixed_sparse_chatml_masked_sft_v3/reasoning_gate_latest.json \
      --device cuda --dtype bf16 --max-new "${MAX_NEW:-48}"
    ;;
  *)
    echo "usage: $0 {status|status-all|start-chatml-sft|start-neurova-r-full|start-auto|restart-auto|eval-latest}" >&2
    exit 2
    ;;
esac
