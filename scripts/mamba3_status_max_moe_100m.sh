#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/mamba3_clean_doc_base_moe24_v1}"
LONG_DIR="${LONG_DIR:-${RUN_DIR}/long_blocks}"
LOG="${LOG:-$(ls -1t "${LONG_DIR}"/*_100m.log 2>/dev/null | head -n 1 || true)}"
SUMMARY="${SUMMARY:-${RUN_DIR}/until_gate/summary.jsonl}"

cd "${ROOT}"

echo "== process =="
ps -eo pid,etime,cmd | grep -E "mamba3_train_clean_doc_until_gate|mamba3_kr.cli train-packed --mode mimo-r4-moe-2.4b" | grep -v grep || true
train_pid="$(ps -eo pid,cmd | awk '/mamba3_kr.cli train-packed --mode mimo-r4-moe-2.4b/ && !/awk/ {print $1; exit}')"
if [[ -n "${train_pid}" && -n "${LOG}" ]]; then
  pidfile="${LOG%.log}.pid"
  echo "${train_pid}" > "${pidfile}"
  echo "active_train_pid=${train_pid}"
  echo "pidfile=${pidfile}"
fi

echo "== gpu =="
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true

if [[ -n "${LOG}" && -f "${LOG}" ]]; then
  echo "== progress =="
  current_round="$(grep -oE '== round [0-9]+/[0-9]+' "${LOG}" | tail -n 1 | awk '{print $3}' || true)"
  current_step="$(grep -oE 'step=[0-9]+' "${LOG}" | tail -n 1 | cut -d= -f2 || true)"
  steps_per_round="$(grep -oE 'steps_per_round=[0-9]+' "${LOG}" | head -n 1 | cut -d= -f2 || true)"
  planned_tokens="$(grep -oE 'planned_tokens=[0-9]+' "${LOG}" | head -n 1 | cut -d= -f2 || true)"
  tokens_per_step="${TOKENS_PER_STEP:-2048}"
  if [[ -n "${current_round}" && -n "${current_step}" && -n "${steps_per_round}" && -n "${planned_tokens}" ]]; then
    round_now="${current_round%/*}"
    round_total="${current_round#*/}"
    completed_rounds=$((round_now - 1))
    trained_steps=$((completed_rounds * steps_per_round + current_step))
    trained_tokens=$((trained_steps * tokens_per_step))
    progress_pct="$(python - "${trained_tokens}" "${planned_tokens}" <<'PY'
import sys
trained = int(sys.argv[1])
planned = int(sys.argv[2])
print(f"{trained / planned * 100:.2f}")
PY
)"
    latest_tok_s="$(grep -oE 'tok_s=[0-9]+([.][0-9]+)?' "${LOG}" | tail -n 100 | cut -d= -f2 | python -c 'import sys; xs=[float(x) for x in sys.stdin if x.strip()]; print(f"{sum(xs)/len(xs):.1f}" if xs else "")')"
    echo "round=${current_round}"
    echo "step_in_round=${current_step}/${steps_per_round}"
    echo "trained_tokens=${trained_tokens}"
    echo "planned_tokens=${planned_tokens}"
    echo "progress_pct=${progress_pct}"
    if [[ -n "${latest_tok_s}" ]]; then
      eta_text="$(python - "${planned_tokens}" "${trained_tokens}" "${latest_tok_s}" <<'PY'
import sys
planned = int(sys.argv[1])
trained = int(sys.argv[2])
tok_s = float(sys.argv[3])
remaining = max(planned - trained, 0)
seconds = remaining / tok_s if tok_s > 0 else 0
hours = int(seconds // 3600)
minutes = int((seconds % 3600) // 60)
print(f"{hours}h{minutes:02d}m")
PY
)"
      echo "avg_tok_s_last100=${latest_tok_s}"
      echo "eta_at_current_speed=${eta_text}"
    fi
  else
    echo "Progress fields are not available yet."
  fi

  echo "== data warnings =="
  warning_count="$(grep -c 'Token indices sequence length is longer than the specified maximum sequence length' "${LOG}" || true)"
  echo "tokenizer_length_warnings=${warning_count}"
  if [[ "${warning_count}" != "0" ]]; then
    grep -n 'Token indices sequence length is longer than the specified maximum sequence length' "${LOG}" | tail -n 5
  fi

  echo "== log: ${LOG} =="
  tail -n "${TAIL_LINES:-30}" "${LOG}"
else
  echo "== log =="
  echo "No *_100m.log found under ${LONG_DIR}"
fi

if [[ -f "${SUMMARY}" ]]; then
  echo "== gate trend =="
  python - "${SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
records = []
for line in path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        continue
    if "loss" in payload:
        records.append(payload)

if not records:
    print("No gate records with loss yet.")
    raise SystemExit

recent = records[-8:]
losses = [float(item["loss"]) for item in records]
best = min(losses)
latest = losses[-1]
first = losses[0]
collapsed_count = sum(1 for item in records if item.get("collapsed") is True)
passed_count = sum(1 for item in records if item.get("passed") is True)
print(f"records={len(records)}")
print(f"loss_first={first:.6f}")
print(f"loss_latest={latest:.6f}")
print(f"loss_best={best:.6f}")
print(f"loss_delta_first_to_latest={latest - first:+.6f}")
print(f"collapsed_records={collapsed_count}")
print(f"passed_records={passed_count}")
print("recent_rounds=" + ", ".join(
    f"r{item.get('round')}:{float(item['loss']):.4f}:collapsed={item.get('collapsed')}"
    for item in recent
))
PY

  echo "== gate summary: ${SUMMARY} =="
  tail -n "${SUMMARY_LINES:-10}" "${SUMMARY}"
fi
