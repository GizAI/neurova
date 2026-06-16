#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/neuromamba/runs/mamba3_teacher_research/deepseek_v4_pro_0_3b}"
CONTROL_DIR="${CONTROL_DIR:-${RUN_ROOT}/control}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
PATTERN="mamba3_teacher_research_loop.sh"

mkdir -p "${CONTROL_DIR}" "${LOG_DIR}"

active_pid() {
  ps -eo pid,cmd | awk -v pat="${PATTERN}" 'index($0, pat) && !index($0, "mamba3_teacher_researchctl.sh") && !index($0, "awk") {print $1; exit}'
}

active_legacy_pid() {
  ps -eo pid,cmd | awk '
    index($0, "mamba3_generate_deepseek_mcq_sft.py") &&
    index($0, "neuromamba/data/deepseek_no_cheat_mcq_sft_v1.jsonl") &&
    !index($0, "mamba3_teacher_research_loop.sh") &&
    !index($0, "awk") {print $1; exit}'
}

latest_log() {
  ls -1t "${LOG_DIR}"/*.log 2>/dev/null | head -n 1 || true
}

case "${1:-status}" in
  start)
    pid="$(active_pid || true)"
    if [[ -n "${pid}" ]]; then
      echo "teacher research already running pid=${pid}"
      exit 0
    fi
    legacy="$(active_legacy_pid || true)"
    if [[ -n "${legacy}" && "${STOP_LEGACY:-1}" == "1" ]]; then
      echo "stopping legacy self-teacher run pid=${legacy}"
      kill -TERM "${legacy}" 2>/dev/null || true
      sleep 3
    fi
    stamp="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
    log="${LOG:-${LOG_DIR}/teacher_research_${stamp}.log}"
    cmdfile="${CONTROL_DIR}/teacher_research_${stamp}.cmd"
    cat > "${cmdfile}" <<EOF
cd "${ROOT}"
source "\${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate mamba3_siso
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RUN_ROOT="${RUN_ROOT}"
export MAX_ROUNDS="${MAX_ROUNDS:-3}"
export DEEPSEEK_RECORDS="${DEEPSEEK_RECORDS:-20000}"
export DEEPSEEK_RECORDS_CSV="${DEEPSEEK_RECORDS_CSV:-3000,12000,30000}"
export DEEPSEEK_BATCH_SIZE="${DEEPSEEK_BATCH_SIZE:-24}"
export BOOTSTRAP_EXISTING="${BOOTSTRAP_EXISTING:-1}"
export BOOTSTRAP_DETERMINISTIC_RECORDS="${BOOTSTRAP_DETERMINISTIC_RECORDS:-120000}"
export BOOTSTRAP_MIX_RECORDS="${BOOTSTRAP_MIX_RECORDS:-120000}"
export MCQ_STEPS="${MCQ_STEPS:-2500}"
export MCQ_LR="${MCQ_LR:-8e-6}"
export MCQ_SEQ_LEN="${MCQ_SEQ_LEN:-256}"
export MCQ_BATCH_SIZE="${MCQ_BATCH_SIZE:-8}"
export MMLU_REDUX_LIMIT="${MMLU_REDUX_LIMIT:-200}"
export CHAT_REPAIR="${CHAT_REPAIR:-1}"
export CHAT_STEPS="${CHAT_STEPS:-1200}"
export CHAT_LR="${CHAT_LR:-1e-5}"
export PROMOTE_CURRENT="${PROMOTE_CURRENT:-1}"
exec neuromamba/scripts/mamba3_teacher_research_loop.sh
EOF
    neuromamba/scripts/mamba3_current_applyctl.sh start >/dev/null 2>&1 || true
    setsid bash "${cmdfile}" > "${log}" 2>&1 < /dev/null &
    pid=$!
    printf '%s\n' "${pid}" > "${CONTROL_DIR}/teacher_research.pid"
    printf '%s\n' "${log}" > "${CONTROL_DIR}/current.log.path"
    printf '%s\n' "${cmdfile}" > "${CONTROL_DIR}/current.cmd.path"
    ln -sfn "${log}" "${CONTROL_DIR}/current.log"
    echo "started_pid=${pid}"
    echo "run_root=${RUN_ROOT}"
    echo "log=${log}"
    ;;
  status)
    echo "== teacher research process =="
    pid="$(active_pid || true)"
    if [[ -n "${pid}" ]]; then
      ps -p "${pid}" -o pid,etime,cmd
    else
      echo "not running"
    fi
    legacy="$(active_legacy_pid || true)"
    if [[ -n "${legacy}" ]]; then
      echo "legacy self-teacher still running pid=${legacy}"
      ps -p "${legacy}" -o pid,etime,cmd || true
    fi
    echo "== gpu =="
    nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true
    echo "== data progress =="
    find "${RUN_ROOT}" -name 'deepseek_no_cheat_mcq.jsonl' -o -path "${ROOT}/neuromamba/data/deepseek_no_cheat_mcq_sft_v1.jsonl" 2>/dev/null \
      | while read -r f; do wc -l "${f}" 2>/dev/null || true; done
    echo "== log =="
    log="$(cat "${CONTROL_DIR}/current.log.path" 2>/dev/null || true)"
    [[ -z "${log}" || ! -f "${log}" ]] && log="$(latest_log)"
    if [[ -n "${log}" && -f "${log}" ]]; then
      echo "${log}"
      tail -n "${TAIL_LINES:-100}" "${log}"
    else
      echo "no log"
    fi
    if [[ -f "${RUN_ROOT}/summary.jsonl" ]]; then
      echo "== summary =="
      tail -n 20 "${RUN_ROOT}/summary.jsonl"
    fi
    if [[ -f "${RUN_ROOT}/best/best.json" ]]; then
      echo "== best =="
      cat "${RUN_ROOT}/best/best.json"
    fi
    ;;
  tail)
    log="$(cat "${CONTROL_DIR}/current.log.path" 2>/dev/null || true)"
    [[ -z "${log}" || ! -f "${log}" ]] && log="$(latest_log)"
    [[ -n "${log}" && -f "${log}" ]] || { echo "no log" >&2; exit 1; }
    tail -f "${log}"
    ;;
  stop)
    pid="$(active_pid || true)"
    if [[ -n "${pid}" ]]; then
      pkill -TERM -P "${pid}" 2>/dev/null || true
      kill -TERM "${pid}" 2>/dev/null || true
      sleep 2
      pkill -KILL -P "${pid}" 2>/dev/null || true
      kill -KILL "${pid}" 2>/dev/null || true
    fi
    echo "stop requested pid=${pid:-none}"
    ;;
  *)
    echo "usage: $0 {start|status|tail|stop}" >&2
    exit 2
    ;;
esac
