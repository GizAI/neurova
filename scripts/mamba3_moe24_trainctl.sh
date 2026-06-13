#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/mamba3_clean_doc_base_moe24_v1}"
CONTROL_DIR="${CONTROL_DIR:-${RUN_DIR}/control}"
LONG_DIR="${LONG_DIR:-${RUN_DIR}/long_blocks}"
MODE="${MODE:-mimo-r4-moe-2.4b}"
TRAIN_PATTERN="mamba3_kr.cli train-packed --mode ${MODE}"
WRAPPER_PATTERN="mamba3_train_clean_doc_until_gate.sh"

mkdir -p "${CONTROL_DIR}" "${LONG_DIR}"

usage() {
  cat <<'EOF'
Usage: scripts/mamba3_moe24_trainctl.sh <command>

Commands:
  start    Start a detached managed 100M-token training block if none is running.
  resume   Alias for start; continues from the latest weight checkpoint.
  adopt    Register an already-running training process and latest log under control/.
  status   Print process, GPU, progress, warnings, and gate trend.
  decision Print the 100M-token decision report from the gate summary.
  router-diagnostics
           Run MoE router/expert diagnostics when no training process is active.
  tail     Follow the active log.
  stop     Gracefully terminate active wrapper/train processes.
  logs     List managed and long-block logs.
  watchdog-start   Start detached auto-resume watchdog.
  watchdog-status  Print watchdog process and log.
  watchdog-stop    Stop watchdog loop without stopping training.
EOF
}

train_pid() {
  ps -eo pid,cmd | awk -v pat="${TRAIN_PATTERN}" 'index($0, pat) && !index($0, "awk") {print $1; exit}'
}

wrapper_pid() {
  ps -eo pid,cmd | awk -v pat="${WRAPPER_PATTERN}" 'index($0, pat) && !index($0, "awk") {print $1; exit}'
}

latest_log() {
  ls -1t "${LONG_DIR}"/*_100m.log 2>/dev/null | head -n 1 || true
}

active_training_pid() {
  local tpid wpid
  tpid="$(train_pid || true)"
  wpid="$(wrapper_pid || true)"
  if [[ -n "${tpid}" ]]; then
    echo "${tpid}"
    return
  fi
  if [[ -n "${wpid}" ]]; then
    echo "${wpid}"
  fi
}

remaining_rounds_from_log() {
  local log tokens_per_step default_rounds
  log="${1:-}"
  tokens_per_step="${TOKENS_PER_STEP:-2048}"
  default_rounds="${MAX_ROUNDS:-25}"
  if [[ -z "${log}" || ! -f "${log}" ]]; then
    echo "${default_rounds}"
    return
  fi
  python - "${log}" "${tokens_per_step}" "${default_rounds}" <<'PY'
import math
import re
import sys
from pathlib import Path

log = Path(sys.argv[1])
tokens_per_step = int(sys.argv[2])
default_rounds = int(sys.argv[3])
text = log.read_text(encoding="utf-8", errors="replace")
planned = None
steps_per_round = None
round_now = None
step_now = None
for match in re.finditer(r"planned_tokens=(\d+)", text):
    planned = int(match.group(1))
for match in re.finditer(r"steps_per_round=(\d+)", text):
    steps_per_round = int(match.group(1))
for match in re.finditer(r"== round (\d+)/(\d+)", text):
    round_now = int(match.group(1))
    if planned is None and steps_per_round is not None:
        planned = int(match.group(2)) * steps_per_round * tokens_per_step
for match in re.finditer(r"step=(\d+)", text):
    step_now = int(match.group(1))
if planned is None or steps_per_round is None or round_now is None or step_now is None:
    print(default_rounds)
    raise SystemExit
trained_steps = max(0, (round_now - 1) * steps_per_round + step_now)
trained_tokens = trained_steps * tokens_per_step
remaining_tokens = max(0, planned - trained_tokens)
remaining_rounds = math.ceil(remaining_tokens / (steps_per_round * tokens_per_step)) if remaining_tokens else 0
print(remaining_rounds)
PY
}

completed_rounds_from_log() {
  local log tokens_per_step
  log="${1:-}"
  tokens_per_step="${TOKENS_PER_STEP:-2048}"
  if [[ -z "${log}" || ! -f "${log}" ]]; then
    echo "0"
    return
  fi
  python - "${log}" "${tokens_per_step}" <<'PY'
import re
import sys
from pathlib import Path

log = Path(sys.argv[1])
tokens_per_step = int(sys.argv[2])
text = log.read_text(encoding="utf-8", errors="replace")
steps_per_round = None
round_now = None
step_now = None
for match in re.finditer(r"steps_per_round=(\d+)", text):
    steps_per_round = int(match.group(1))
for match in re.finditer(r"== round (\d+)/(\d+)", text):
    round_now = int(match.group(1))
for match in re.finditer(r"step=(\d+)", text):
    step_now = int(match.group(1))
if steps_per_round is None or round_now is None or step_now is None:
    print(0)
    raise SystemExit
trained_steps = max(0, (round_now - 1) * steps_per_round + step_now)
print(trained_steps // steps_per_round)
PY
}

write_current_log() {
  local log="$1"
  if [[ -n "${log}" && -f "${log}" ]]; then
    printf '%s\n' "${log}" > "${CONTROL_DIR}/current.log.path"
    ln -sfn "${log}" "${CONTROL_DIR}/current.log"
  fi
}

status() {
  local log
  log="${LOG:-}"
  if [[ -z "${log}" && -f "${CONTROL_DIR}/current.log.path" ]]; then
    log="$(cat "${CONTROL_DIR}/current.log.path")"
  fi
  if [[ -z "${log}" || ! -f "${log}" ]]; then
    log="$(latest_log)"
  fi
  if [[ -n "${log}" ]]; then
    write_current_log "${log}"
    LOG="${log}" "${ROOT}/scripts/mamba3_status_max_moe_100m.sh"
  else
    "${ROOT}/scripts/mamba3_status_max_moe_100m.sh"
  fi
}

adopt() {
  local tpid wpid log
  tpid="$(train_pid || true)"
  wpid="$(wrapper_pid || true)"
  log="$(latest_log)"
  if [[ -z "${tpid}" && -z "${wpid}" ]]; then
    echo "No active ${MODE} training process found." >&2
    exit 1
  fi
  [[ -n "${tpid}" ]] && printf '%s\n' "${tpid}" > "${CONTROL_DIR}/train.pid"
  [[ -n "${wpid}" ]] && printf '%s\n' "${wpid}" > "${CONTROL_DIR}/wrapper.pid"
  write_current_log "${log}"
  cat > "${CONTROL_DIR}/state.json" <<EOF
{"mode":"${MODE}","train_pid":"${tpid}","wrapper_pid":"${wpid}","log":"${log}","adopted_at_utc":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF
  echo "adopted_train_pid=${tpid}"
  echo "adopted_wrapper_pid=${wpid}"
  echo "log=${log}"
}

start() {
  local existing log stamp cmdfile launcher_pid tpid remaining_rounds completed_rounds data_seed_base
  existing="$(active_training_pid || true)"
  if [[ -n "${existing}" ]]; then
    echo "Training is already running with pid=${existing}; adopting it." >&2
    adopt
    exit 0
  fi

  stamp="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
  log="${LOG:-${LONG_DIR}/${stamp}_100m.log}"
  cmdfile="${CONTROL_DIR}/${stamp}_100m.cmd"
  remaining_rounds="${MAX_ROUNDS:-$(remaining_rounds_from_log "$(latest_log)")}"
  completed_rounds="$(completed_rounds_from_log "$(latest_log)")"
  data_seed_base="${DATA_SEED_BASE:-$((997000 + completed_rounds))}"
  if [[ "${remaining_rounds}" -le 0 ]]; then
    echo "planned training already appears complete; not starting a new run"
    exit 0
  fi

  cat > "${cmdfile}" <<EOF
cd "${ROOT}"
source "\${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate mamba3_siso
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TARGET_TOKENS="${TARGET_TOKENS:-100000000}"
export LR="${LR:-8e-6}"
export EVAL_BATCHES="${EVAL_BATCHES:-32}"
export DATA_SEED_BASE="${data_seed_base}"
export MAX_ROUNDS="${remaining_rounds}"
export MAX_TEXT_CHARS="${MAX_TEXT_CHARS:-65536}"
export MAX_TEXT_TOKENS="${MAX_TEXT_TOKENS:-120000}"
exec scripts/mamba3_train_max_moe_100m_block.sh
EOF

  setsid bash "${cmdfile}" > "${log}" 2>&1 < /dev/null &
  launcher_pid=$!
  printf '%s\n' "${launcher_pid}" > "${CONTROL_DIR}/launcher.pid"
  write_current_log "${log}"
  printf '%s\n' "${cmdfile}" > "${CONTROL_DIR}/current.cmd.path"

  sleep 5
  tpid="$(train_pid || true)"
  if [[ -z "${tpid}" ]]; then
    echo "Training did not start. Last log lines:" >&2
    tail -n 120 "${log}" >&2 || true
    exit 1
  fi
  printf '%s\n' "${tpid}" > "${CONTROL_DIR}/train.pid"
  cat > "${CONTROL_DIR}/state.json" <<EOF
{"mode":"${MODE}","train_pid":"${tpid}","launcher_pid":"${launcher_pid}","log":"${log}","cmdfile":"${cmdfile}","started_at_utc":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF
  echo "started_train_pid=${tpid}"
  echo "launcher_pid=${launcher_pid}"
  echo "log=${log}"
  echo "cmdfile=${cmdfile}"
}

stop() {
  local tpid wpid
  tpid="$(train_pid || true)"
  wpid="$(wrapper_pid || true)"
  if [[ -z "${tpid}" && -z "${wpid}" ]]; then
    echo "No active training process found."
    return
  fi
  [[ -n "${tpid}" ]] && kill -TERM "${tpid}" 2>/dev/null || true
  [[ -n "${wpid}" ]] && kill -TERM "${wpid}" 2>/dev/null || true
  sleep 5
  tpid="$(train_pid || true)"
  wpid="$(wrapper_pid || true)"
  if [[ -n "${tpid}" || -n "${wpid}" ]]; then
    echo "Processes still alive after TERM; leaving them running for manual inspection." >&2
    ps -eo pid,etime,cmd | grep -E "${TRAIN_PATTERN}|${WRAPPER_PATTERN}" | grep -v grep >&2 || true
    exit 1
  fi
  echo "stopped"
}

tail_log() {
  local log
  log="${LOG:-}"
  if [[ -z "${log}" && -f "${CONTROL_DIR}/current.log.path" ]]; then
    log="$(cat "${CONTROL_DIR}/current.log.path")"
  fi
  if [[ -z "${log}" || ! -f "${log}" ]]; then
    log="$(latest_log)"
  fi
  if [[ -z "${log}" || ! -f "${log}" ]]; then
    echo "No log found." >&2
    exit 1
  fi
  write_current_log "${log}"
  tail -f "${log}"
}

logs() {
  echo "== control =="
  ls -ltr "${CONTROL_DIR}" || true
  echo "== long blocks =="
  ls -ltr "${LONG_DIR}" | tail -n 30 || true
}

decision() {
  python "${ROOT}/scripts/mamba3_post100m_decision.py" \
    --summary "${RUN_DIR}/until_gate/summary.jsonl" \
    --planned-tokens "${PLANNED_TOKENS:-102400000}"
}

router_diagnostics() {
  local existing out
  existing="$(active_training_pid || true)"
  if [[ -n "${existing}" && "${FORCE_ROUTER_DIAGNOSTICS:-0}" != "1" ]]; then
    echo "Training is active with pid=${existing}; refusing router diagnostics to avoid GPU OOM." >&2
    echo "Run after the 100M block finishes, or set FORCE_ROUTER_DIAGNOSTICS=1 on a separate GPU." >&2
    exit 1
  fi
  out="${ROUTER_DIAG_OUT:-${RUN_DIR}/router_diagnostics/$(date -u +%Y%m%dT%H%M%SZ).json}"
  python "${ROOT}/scripts/mamba3_moe_router_diagnostics.py" \
    --mode "${MODE}" \
    --tokenizer "${TOKENIZER:-llama31}" \
    --checkpoint "${CHECKPOINT:-${RUN_DIR}/base.pt}" \
    --data "${VALID_DATA:-data/splits/base_doc_cont_v3_valid.jsonl}" \
    --seq-len "${SEQ_LEN:-2048}" \
    --batch-size "${BATCH_SIZE:-1}" \
    --batches "${ROUTER_DIAG_BATCHES:-8}" \
    --device "${DEVICE:-cuda}" \
    --dtype "${DTYPE:-bf16}" \
    --out "${out}"
}

watchdog_loop() {
  local interval active remaining log
  interval="${WATCHDOG_INTERVAL:-60}"
  echo "watchdog_started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  while true; do
    if [[ -f "${CONTROL_DIR}/watchdog.stop" ]]; then
      echo "watchdog_stop_file_seen_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      rm -f "${CONTROL_DIR}/watchdog.stop"
      exit 0
    fi
    active="$(active_training_pid || true)"
    if [[ -n "${active}" ]]; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) active_pid=${active}"
      adopt >/dev/null 2>&1 || true
    else
      log="$(latest_log)"
      remaining="$(remaining_rounds_from_log "${log}")"
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) inactive remaining_rounds=${remaining}"
      if [[ "${remaining}" -le 0 ]]; then
        echo "watchdog_done_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        exit 0
      fi
      MAX_ROUNDS="${remaining}" start || true
    fi
    sleep "${interval}"
  done
}

watchdog_start() {
  local existing log pid
  existing="$(ps -eo pid,cmd | awk -v pat="mamba3_moe24_trainctl.sh watchdog-loop" 'index($0, pat) && !index($0, "awk") {print $1; exit}')"
  if [[ -n "${existing}" ]]; then
    echo "watchdog already running pid=${existing}"
    exit 0
  fi
  rm -f "${CONTROL_DIR}/watchdog.stop"
  log="${CONTROL_DIR}/watchdog.log"
  setsid "${ROOT}/scripts/mamba3_moe24_trainctl.sh" watchdog-loop > "${log}" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "${pid}" > "${CONTROL_DIR}/watchdog.pid"
  echo "watchdog_pid=${pid}"
  echo "watchdog_log=${log}"
}

watchdog_status() {
  echo "== watchdog process =="
  ps -eo pid,etime,cmd | grep "mamba3_moe24_trainctl.sh watchdog-loop" | grep -v grep || true
  echo "== watchdog log =="
  tail -n 40 "${CONTROL_DIR}/watchdog.log" 2>/dev/null || true
}

watchdog_stop() {
  touch "${CONTROL_DIR}/watchdog.stop"
  echo "watchdog stop requested"
}

cmd="${1:-status}"
case "${cmd}" in
  start|resume) start ;;
  adopt) adopt ;;
  status) status ;;
  decision) decision ;;
  router-diagnostics) router_diagnostics ;;
  tail) tail_log ;;
  stop) stop ;;
  logs) logs ;;
  watchdog-start) watchdog_start ;;
  watchdog-status) watchdog_status ;;
  watchdog-stop) watchdog_stop ;;
  watchdog-loop) watchdog_loop ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
