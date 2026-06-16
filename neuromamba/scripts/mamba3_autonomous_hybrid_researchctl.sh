#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"

CONTROL_ROOT="${CONTROL_ROOT:-${ROOT}/neuromamba/runs/mamba3_autonomous_hybrid_research/control}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/neuromamba/runs/mamba3_autonomous_hybrid_research/logs}"
mkdir -p "${CONTROL_ROOT}" "${LOG_ROOT}"

active_pid() {
  ps -eo pid=,args= | awk '
    index($0, "mamba3_autonomous_hybrid_research_loop.sh") &&
    !index($0, "mamba3_autonomous_hybrid_researchctl.sh") &&
    !index($0, "awk") {print $1; exit}'
}

latest_log() {
  ls -1t "${LOG_ROOT}"/*.log 2>/dev/null | head -n 1 || true
}

case "${1:-status}" in
  start)
    pid="$(active_pid || true)"
    if [[ -n "${pid}" ]]; then
      echo "autonomous hybrid research already running pid=${pid}"
      exit 0
    fi
    stamp="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
    run_root="${RUN_ROOT:-${ROOT}/neuromamba/runs/mamba3_autonomous_hybrid_research/${stamp}}"
    log="${LOG:-${LOG_ROOT}/${stamp}.log}"
    cmdfile="${CONTROL_ROOT}/${stamp}.cmd"
    cat > "${cmdfile}" <<EOF
cd "${ROOT}"
source "\${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate mamba3_siso
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RUN_ROOT="${run_root}"
export MODES_CSV="${MODES_CSV:-mamba3-siso-fast-0.3b-ds128,mamba3-siso-hybrid-0.3b}"
export STEPS="${STEPS:-2500}"
export SEQ_LEN="${SEQ_LEN:-1024}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
export LR="${LR:-4e-6}"
export MMLU_REDUX_LIMIT="${MMLU_REDUX_LIMIT:-100}"
export SEED_POLICY="${SEED_POLICY:-same_mode_only}"
exec neuromamba/scripts/mamba3_exclusive_gpu_guard.sh run neuromamba/scripts/mamba3_autonomous_hybrid_research_loop.sh
EOF
    setsid bash "${cmdfile}" > "${log}" 2>&1 < /dev/null &
    echo "$!" > "${CONTROL_ROOT}/research.pid"
    printf '%s\n' "${log}" > "${CONTROL_ROOT}/current.log.path"
    printf '%s\n' "${cmdfile}" > "${CONTROL_ROOT}/current.cmd.path"
    printf '%s\n' "${run_root}" > "${CONTROL_ROOT}/current.run_root.path"
    ln -sfn "${log}" "${CONTROL_ROOT}/current.log"
    echo "started_pid=$!"
    echo "run_root=${run_root}"
    echo "log=${log}"
    ;;
  status)
    echo "== process =="
    pid="$(active_pid || true)"
    if [[ -n "${pid}" ]]; then
      ps -p "${pid}" -o pid,etime,cmd
    else
      echo "not running"
    fi
    echo "== gpu =="
    nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true
    echo "== run root =="
    [[ -f "${CONTROL_ROOT}/current.run_root.path" ]] && cat "${CONTROL_ROOT}/current.run_root.path" || true
    echo "== log =="
    log=""
    [[ -f "${CONTROL_ROOT}/current.log.path" ]] && log="$(cat "${CONTROL_ROOT}/current.log.path")"
    [[ -z "${log}" || ! -f "${log}" ]] && log="$(latest_log)"
    if [[ -n "${log}" && -f "${log}" ]]; then
      echo "${log}"
      tail -n "${TAIL_LINES:-80}" "${log}"
    else
      echo "no log"
    fi
    run_root=""
    [[ -f "${CONTROL_ROOT}/current.run_root.path" ]] && run_root="$(cat "${CONTROL_ROOT}/current.run_root.path")"
    if [[ -n "${run_root}" && -d "${run_root}" ]]; then
      echo "== candidate train logs =="
      find "${run_root}" -mindepth 2 -maxdepth 2 -name train.log -print | sort | while read -r train_log; do
        echo "-- ${train_log} --"
        tail -n "${TRAIN_TAIL_LINES:-12}" "${train_log}" || true
      done
    fi
    ;;
  tail)
    log=""
    [[ -f "${CONTROL_ROOT}/current.log.path" ]] && log="$(cat "${CONTROL_ROOT}/current.log.path")"
    [[ -z "${log}" || ! -f "${log}" ]] && log="$(latest_log)"
    [[ -n "${log}" ]] || { echo "no log" >&2; exit 1; }
    tail -f "${log}"
    ;;
  stop)
    pid="$(active_pid || true)"
    [[ -n "${pid}" ]] && kill "${pid}" || true
    ;;
  *)
    echo "usage: $0 {start|status|tail|stop}" >&2
    exit 2
    ;;
esac
