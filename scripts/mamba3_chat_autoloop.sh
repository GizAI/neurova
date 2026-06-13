#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOOP_DIR="${LOOP_DIR:-${ROOT}/runs/mamba3_neurova_chat_autoloop}"
LOG_DIR="${LOG_DIR:-${LOOP_DIR}/logs}"
SUMMARY="${SUMMARY:-${LOOP_DIR}/summary.jsonl}"
CONTROL_DIR="${CONTROL_DIR:-${LOOP_DIR}/control}"
MAX_TRIALS="${MAX_TRIALS:-6}"
DEADLINE_HOURS="${DEADLINE_HOURS:-24}"
MODE="${MODE:-mimo-r4-tiny}"
TOKENIZER="${TOKENIZER:-llama31}"
SEQ_LEN="${SEQ_LEN:-128}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
MIN_PASS_RATE="${MIN_PASS_RATE:-0.70}"
POLL_SECONDS="${POLL_SECONDS:-60}"
GATE_TIMEOUT_SECONDS="${GATE_TIMEOUT_SECONDS:-900}"
TUNE_TIMEOUT_SECONDS="${TUNE_TIMEOUT_SECONDS:-900}"

cd "${ROOT}"
mkdir -p "${LOG_DIR}" "${CONTROL_DIR}" "${LOOP_DIR}"

busy_pids() {
  pgrep -af "mamba3_train_neurova_chat_v1|mamba3_chat_quality_gate|mamba3_decode_tune|train-answer" \
    | grep -v "mamba3_chat_autoloop" \
    | grep -v "pgrep -af" || true
}

json_get() {
  local path="$1" expr="$2"
  python - "$path" "$expr" <<'PY'
import json, sys
path, expr = sys.argv[1:3]
try:
    data = json.load(open(path))
except Exception:
    print("")
    raise SystemExit(0)
cur = data
for part in expr.split("."):
    if not part:
        continue
    if isinstance(cur, dict):
        cur = cur.get(part)
    else:
        cur = None
        break
print("" if cur is None else cur)
PY
}

promote_if_better() {
  local trial_dir="$1"
  local gate="${trial_dir}/chat_quality_gate.json"
  local model="${trial_dir}/chat.pt"
  local tune="${trial_dir}/decode_tune/latest.json"
  local current_meta="${ROOT}/runs/mamba3_current/autoloop_metadata.json"
  [[ -f "${gate}" && -f "${model}" ]] || return 0
  local ok pass_rate best_rate
  ok="$(json_get "${gate}" ok)"
  pass_rate="$(json_get "${gate}" pass_rate)"
  best_rate="0"
  if [[ -f "${current_meta}" ]]; then
    best_rate="$(json_get "${current_meta}" pass_rate)"
  fi
  python - "$ok" "$pass_rate" "$best_rate" <<'PY' || return 0
import sys
ok = sys.argv[1] == "True"
rate = float(sys.argv[2] or 0)
best = float(sys.argv[3] or 0)
if not ok or rate < best:
    sys.exit(1)
PY
  mkdir -p "${ROOT}/runs/mamba3_current"
  cp "${model}" "${ROOT}/runs/mamba3_current/model.pt"
  cp "${gate}" "${ROOT}/runs/mamba3_current/chat_quality_gate.json"
  [[ -f "${tune}" ]] && cp "${tune}" "${ROOT}/runs/mamba3_current/decode_tune.json"
  python - "${trial_dir}" "${pass_rate}" "${current_meta}" <<'PY'
import json, sys, time
trial, pass_rate, out = sys.argv[1:4]
payload = {
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "trial_dir": trial,
    "pass_rate": float(pass_rate or 0),
    "promoted": "runs/mamba3_current/model.pt",
}
open(out, "w").write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(payload, ensure_ascii=False))
PY
}

wait_for_existing() {
  local waited=0
  while busy_pids >/tmp/neurova_chat_busy.$$; do
    if [[ ! -s /tmp/neurova_chat_busy.$$ ]]; then
      rm -f /tmp/neurova_chat_busy.$$
      return 0
    fi
    echo "existing_chat_job_wait waited=${waited}s"
    cat /tmp/neurova_chat_busy.$$
    if [[ "${waited}" -ge "${GATE_TIMEOUT_SECONDS}" ]]; then
      echo "existing_chat_job_timeout_kill"
      awk '{print $1}' /tmp/neurova_chat_busy.$$ | xargs -r kill -TERM || true
      sleep 5
      awk '{print $1}' /tmp/neurova_chat_busy.$$ | xargs -r kill -KILL || true
      rm -f /tmp/neurova_chat_busy.$$
      return 0
    fi
    sleep "${POLL_SECONDS}"
    waited=$((waited + POLL_SECONDS))
  done
}

trial_params() {
  local trial="$1"
  case "${trial}" in
    1) echo "STEPS=4500 LR=1.5e-5 RECORDS=60000 START_CHECKPOINT=runs/mamba3_neurova_chat_v1/chat.pt" ;;
    2) echo "STEPS=6000 LR=1.0e-5 RECORDS=90000 START_CHECKPOINT=runs/mamba3_current/model.pt" ;;
    3) echo "STEPS=4500 LR=8e-6 RECORDS=120000 START_CHECKPOINT=runs/mamba3_current/model.pt" ;;
    4) echo "STEPS=3000 LR=6e-6 RECORDS=120000 START_CHECKPOINT=runs/mamba3_current/model.pt" ;;
    *) echo "STEPS=3000 LR=5e-6 RECORDS=120000 START_CHECKPOINT=runs/mamba3_current/model.pt" ;;
  esac
}

start_time="$(date +%s)"
deadline=$((start_time + DEADLINE_HOURS * 3600))
echo "autoloop_started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "loop_dir=${LOOP_DIR}"
echo "max_trials=${MAX_TRIALS} deadline_hours=${DEADLINE_HOURS}"

wait_for_existing
if [[ -f runs/mamba3_neurova_chat_v1/chat.pt ]]; then
  promote_if_better "runs/mamba3_neurova_chat_v1" || true
fi

for trial in $(seq 1 "${MAX_TRIALS}"); do
  now="$(date +%s)"
  [[ "${now}" -lt "${deadline}" ]] || break
  wait_for_existing

  trial_dir="${LOOP_DIR}/trial_${trial}"
  mkdir -p "${trial_dir}"
  params="$(trial_params "${trial}")"
  log="${LOG_DIR}/trial_${trial}_$(date -u +%Y%m%dT%H%M%SZ).log"
  echo "trial=${trial} params=${params} log=${log}"
  {
    echo "== trial ${trial} =="
    echo "${params}"
    eval "${params}"
    if [[ ! -f "${START_CHECKPOINT}" ]]; then
      START_CHECKPOINT="runs/mamba3_neurova_speak_v1/sft.pt"
    fi
    RUN_DIR="${trial_dir}" \
    MODE="${MODE}" TOKENIZER="${TOKENIZER}" SEQ_LEN="${SEQ_LEN}" DEVICE="${DEVICE}" DTYPE="${DTYPE}" \
    STEPS="${STEPS}" LR="${LR}" RECORDS="${RECORDS}" START_CHECKPOINT="${START_CHECKPOINT}" \
    MIN_PASS_RATE="${MIN_PASS_RATE}" \
    scripts/mamba3_train_neurova_chat_v1.sh
  } >"${log}" 2>&1 || true

  promote_if_better "${trial_dir}" || true
  python - "${trial}" "${trial_dir}" "${log}" >>"${SUMMARY}" <<'PY'
import json, pathlib, sys, time
trial, trial_dir, log = sys.argv[1:4]
gate_path = pathlib.Path(trial_dir) / "chat_quality_gate.json"
tune_path = pathlib.Path(trial_dir) / "decode_tune" / "latest.json"
def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return {}
gate = load(gate_path)
tune = load(tune_path)
payload = {
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "trial": int(trial),
    "trial_dir": trial_dir,
    "log": log,
    "gate_ok": gate.get("ok"),
    "pass_rate": gate.get("pass_rate"),
    "passed": gate.get("passed"),
    "total": gate.get("total"),
    "decode_best": tune.get("best", {}).get("name"),
    "decode_tok_s": tune.get("best", {}).get("avg_new_tokens_per_sec"),
}
print(json.dumps(payload, ensure_ascii=False))
PY
done

echo "autoloop_finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
