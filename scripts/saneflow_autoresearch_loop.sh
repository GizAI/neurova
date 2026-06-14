#!/usr/bin/env bash
set -euo pipefail

cd "${ROOT:-$HOME/workspace/neurova}"
if [[ -n "${SANEFLOW_PYTHON:-}" ]]; then
  PYTHON="$SANEFLOW_PYTHON"
elif [[ -x /home/user/miniconda3/envs/saneflow/bin/python ]]; then
  PYTHON=/home/user/miniconda3/envs/saneflow/bin/python
else
  PYTHON="$(command -v python3 || command -v python)"
fi
LOG_DIR="runs/saneflow_autoresearch"
mkdir -p "$LOG_DIR"

host="${1:-local}"

log_json() {
  printf '{"ts":"%s","host":"%s","event":"%s"}\n' "$(date -Is)" "$host" "$1" | tee -a "$LOG_DIR/loop.jsonl"
}

run_gate() {
  local ckpt="$1"
  local out="$2"
  if [[ -f "$ckpt" ]]; then
    "$PYTHON" scripts/saneflow_quality_gate.py \
      --ckpt "$ckpt" \
      --out "$out" \
      --valid-data data/corpus/sources/fineweb_edu_sample10bt/valid.jsonl \
      --max-new "${MAX_NEW:-96}" \
      --context "${CONTEXT:-384}" \
      --temperature "${TEMPERATURE:-0.7}" \
      --top-k "${TOP_K:-40}" \
      --top-p "${TOP_P:-0.9}" \
      --repetition-penalty "${REPETITION_PENALTY:-1.08}" \
      --no-repeat-ngram-size "${NO_REPEAT_NGRAM_SIZE:-4}" \
      --device cuda --dtype bf16 || true
  fi
}

profile_running() {
  local profile="$1"
  local out
  out="$("$PYTHON" scripts/saneflow_run.py list | "$PYTHON" -c 'import json,sys; data=json.load(sys.stdin); print(data[sys.argv[1]]["out"])' "$profile")"
  pgrep -af "scripts/saneflow_train.py" | grep -F -- "--out ${out}" | grep -v grep >/dev/null
}

profile_done() {
  local profile="$1"
  local out
  out="$("$PYTHON" scripts/saneflow_run.py list | "$PYTHON" -c 'import json,sys; data=json.load(sys.stdin); print(data[sys.argv[1]]["out"])' "$profile")"
  [[ -f "${out}/model.pt" ]]
}

profile_latest_exists() {
  local profile="$1"
  local out
  out="$("$PYTHON" scripts/saneflow_run.py list | "$PYTHON" -c 'import json,sys; data=json.load(sys.stdin); print(data[sys.argv[1]]["out"])' "$profile")"
  [[ -f "${out}/latest.pt" || -f "${out}/model.pt" ]]
}

start_profile_once() {
  local profile="$1"
  if profile_running "$profile"; then
    log_json "profile_already_running:${profile}"
    return 0
  fi
  if profile_done "$profile"; then
    log_json "profile_already_done:${profile}"
    return 0
  fi
  "$PYTHON" scripts/saneflow_run.py start "$profile"
}

gate_passed() {
  local path="$1"
  "$PYTHON" - "$path" <<'PY'
import json, sys
path = sys.argv[1]
try:
    data = json.load(open(path, encoding="utf-8"))
except Exception:
    raise SystemExit(1)
s = data.get("summary", {})
ok = (
    s.get("empty_output_rate", 1.0) <= 0.05
    and s.get("invalid_output_rate", 1.0) <= 0.02
    and s.get("max_repeated_4gram", 999) <= 4
    and s.get("mean_distinct_1", 0.0) >= 0.22
    and s.get("mean_distinct_2", 0.0) >= 0.45
)
raise SystemExit(0 if ok else 1)
PY
}

run_dmc8_once() {
  log_json "dmc8_cycle_start"
  "$PYTHON" scripts/saneflow_build_speak_pretrain_v1.py || true
  "$PYTHON" scripts/saneflow_build_chatml_sft.py || true

  if ! profile_done dmc8-speak-base-v1; then
    log_json "ensure_speak_base"
    start_profile_once dmc8-speak-base-v1 || true
    return 0
  fi

  gate_path="runs/saneflow_speak_base_v1_100m/quality_gate.json"
  if [[ ! -f "$gate_path" || "runs/saneflow_speak_base_v1_100m/model.pt" -nt "$gate_path" ]]; then
    log_json "gate_speak_base"
    run_gate "runs/saneflow_speak_base_v1_100m/model.pt" "$gate_path"
  fi

  if gate_passed "$gate_path"; then
    if ! profile_done dmc8-chatml-sft-v9; then
      log_json "ensure_chatml_sft_v9"
      start_profile_once dmc8-chatml-sft-v9 || true
      return 0
    fi
    log_json "gate_chatml_sft_v9"
    run_gate "runs/saneflow_chatml_sft_v9_assistant/model.pt" "runs/saneflow_chatml_sft_v9_assistant/quality_gate.json"
    mkdir -p runs/saneflow_current
    ln -sfn "$(pwd)/runs/saneflow_chatml_sft_v9_assistant/model.pt" runs/saneflow_current/model.pt
    log_json "promoted_chatml_sft_v9"
  else
    log_json "speak_gate_failed_hold_sft"
  fi
}

run_dmc9_once() {
  log_json "dmc9_cycle_start"
  "$PYTHON" scripts/saneflow_build_practical_pretrain_mix.py || true
  log_json "ensure_practical_base"
  start_profile_once dmc9-practical-base-100m || true

  log_json "ensure_r_champion"
  start_profile_once dmc9-r-champion-delta-landmark-long || true
  if profile_latest_exists dmc9-r-champion-delta-landmark-long; then
    run_gate "runs/saneflow_r_champion/d_delta_landmark_long/latest.pt" "runs/saneflow_r_champion/d_delta_landmark_long/quality_gate_latest.json"
  fi
  if profile_done dmc9-r-champion-delta-landmark-long; then
    log_json "ensure_r_champion_practical_cont"
    start_profile_once dmc9-r-champion-practical-cont || true
    if profile_latest_exists dmc9-r-champion-practical-cont; then
      run_gate "runs/saneflow_r_champion/d_delta_landmark_practical_cont/latest.pt" "runs/saneflow_r_champion/d_delta_landmark_practical_cont/quality_gate_latest.json"
    fi
  fi
}

run_once() {
  case "$host" in
  dmc8)
    run_dmc8_once
    ;;
  dmc9)
    run_dmc9_once
    ;;
  *)
    echo "usage: $0 {dmc8|dmc9}" >&2
    exit 2
    ;;
  esac
}

if [[ "${SANEFLOW_LOOP_FOREVER:-1}" == "1" ]]; then
  log_json "loop_forever_start"
  while true; do
    run_once || log_json "cycle_failed"
    sleep "${SANEFLOW_LOOP_SLEEP_SECONDS:-300}"
  done
else
  run_once
fi
