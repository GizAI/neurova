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
  local valid_data
  valid_data="$("$PYTHON" - <<'PY'
import json
data = json.load(open("configs/saneflow_profiles.json", encoding="utf-8"))
print(data["paths"]["fineweb_valid"])
PY
)"
  if [[ -f "$ckpt" ]]; then
    "$PYTHON" scripts/saneflow_quality_gate.py \
      --ckpt "$ckpt" \
      --out "$out" \
      --valid-data "$valid_data" \
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

run_reasoning_gate() {
  local ckpt="$1"
  local out="$2"
  if [[ -f "$ckpt" ]]; then
    "$PYTHON" scripts/saneflow_reasoning_gate.py \
      --ckpt "$ckpt" \
      --out "$out" \
      --max-new "${REASONING_MAX_NEW:-40}" \
      --context "${REASONING_CONTEXT:-384}" \
      --temperature "${REASONING_TEMPERATURE:-0.2}" \
      --top-k "${REASONING_TOP_K:-20}" \
      --top-p "${REASONING_TOP_P:-0.9}" \
      --device cuda --dtype bf16 || true
  fi
}

profile_running() {
  local profile="$1"
  local out
  out="$(profile_out "$profile")"
  pgrep -af "scripts/saneflow_train.py" | grep -F -- "--out ${out}" | grep -v grep >/dev/null
}

profile_done() {
  local profile="$1"
  local out
  out="$(profile_out "$profile")"
  [[ -f "${out}/model.pt" ]]
}

profile_latest_exists() {
  local profile="$1"
  local out
  out="$(profile_out "$profile")"
  [[ -f "${out}/latest.pt" || -f "${out}/model.pt" ]]
}

profile_out() {
  local profile="$1"
  "$PYTHON" scripts/saneflow_run.py list | "$PYTHON" -c 'import json,sys; data=json.load(sys.stdin); print(data[sys.argv[1]]["out"])' "$profile"
}

profile_latest() {
  local profile="$1"
  local out
  out="$(profile_out "$profile")"
  if [[ -f "${out}/latest.pt" ]]; then
    printf '%s\n' "${out}/latest.pt"
  else
    printf '%s\n' "${out}/model.pt"
  fi
}

profile_model() {
  local profile="$1"
  printf '%s/model.pt\n' "$(profile_out "$profile")"
}

dense_tokenizer_path() {
  "$PYTHON" scripts/saneflow_run.py list | "$PYTHON" -c 'import json,sys; data=json.load(sys.stdin); print(data["dmc9-dense-0.3b-v1"]["tokenizer_path"])'
}

ensure_dense_tokenizer() {
  local tok
  tok="$(dense_tokenizer_path)"
  if [[ -f "${tok}/tokenizer.model" ]]; then
    log_json "dense_tokenizer_exists:${tok}"
    return 0
  fi
  log_json "train_dense_sentencepiece_tokenizer:${tok}"
  "$PYTHON" scripts/saneflow_train_tokenizer.py \
    --kind sentencepiece_unigram \
    --vocab-size "${DENSE_TOKENIZER_VOCAB_SIZE:-65536}" \
    --character-coverage "${DENSE_TOKENIZER_CHARACTER_COVERAGE:-0.99995}" \
    --out "$tok" \
    --input \
      data/corpus/mixes/saneflow_practical_pretrain_v1/train.jsonl \
      data/corpus/mixes/saneflow_practical_pretrain_v1/valid.jsonl
}

ensure_doremi_mix() {
  local ratios="data/corpus/mixes/saneflow_practical_pretrain_v1/doremi_ratios.json"
  if [[ "${DOREMI_AUTO:-1}" != "1" ]]; then
    log_json "doremi_auto_disabled"
    return 0
  fi
  if [[ -f "$ratios" ]]; then
    log_json "doremi_ratios_exists"
    return 0
  fi
  if profile_running dmc9-dense-0.3b-v1; then
    log_json "doremi_wait_dense_running"
    return 0
  fi
  log_json "run_doremi_proxy_pipeline"
  CUDA_VISIBLE_DEVICES="${DOREMI_CUDA_VISIBLE_DEVICES:-1}" "$PYTHON" scripts/saneflow_doremi_pipeline.py \
    --steps "${DOREMI_PROXY_STEPS:-200}" \
    --reference-steps "${DOREMI_REFERENCE_STEPS:-300}" \
    --seq-len "${DOREMI_PROXY_SEQ_LEN:-512}" \
    --batch-size "${DOREMI_PROXY_BATCH_SIZE:-1}" \
    --tokenizer-path "$(dense_tokenizer_path)" \
    --tf32 \
    --activation-checkpointing || log_json "doremi_proxy_failed"
}

ensure_practical_mix() {
  local train="data/corpus/mixes/saneflow_practical_pretrain_v1/train.jsonl"
  local valid="data/corpus/mixes/saneflow_practical_pretrain_v1/valid.jsonl"
  if [[ "${FORCE_REBUILD_PRACTICAL_MIX:-0}" != "1" && -s "$train" && -s "$valid" ]]; then
    log_json "practical_mix_exists"
    return 0
  fi
  "$PYTHON" scripts/saneflow_build_practical_pretrain_mix.py || true
}

ensure_practical_v2_en_mix() {
  local train="data/corpus/mixes/saneflow_practical_pretrain_v2_en/train.jsonl"
  local valid="data/corpus/mixes/saneflow_practical_pretrain_v2_en/valid.jsonl"
  if [[ -s "$train" && -s "$valid" ]]; then
    log_json "practical_v2_en_mix_exists"
    return 0
  fi
  if pgrep -af "scripts/saneflow_prepare_pretrain_v2.py" | grep -v grep >/dev/null; then
    log_json "practical_v2_en_prepare_running"
    return 0
  fi
  log_json "start_practical_v2_en_prepare"
  mkdir -p runs/data_prep
  nohup "$PYTHON" scripts/saneflow_prepare_pretrain_v2.py \
    --recipe configs/saneflow_pretrain_sources_v2.json \
    > runs/data_prep/pretrain_v2_en.out 2>&1 &
}

ensure_doremi_mix_v2_en() {
  local ratios="data/corpus/mixes/saneflow_practical_pretrain_v2_en/doremi_ratios.json"
  local train="data/corpus/mixes/saneflow_practical_pretrain_v2_en/train.jsonl"
  local valid="data/corpus/mixes/saneflow_practical_pretrain_v2_en/valid.jsonl"
  if [[ "${DOREMI_AUTO:-1}" != "1" ]]; then
    log_json "doremi_v2_auto_disabled"
    return 0
  fi
  if [[ -f "$ratios" ]]; then
    log_json "doremi_v2_ratios_exists"
    return 0
  fi
  if [[ ! -s "$train" || ! -s "$valid" ]]; then
    log_json "doremi_v2_wait_mix"
    return 0
  fi
  if profile_running dmc9-dense-0.3b-v2-en-cont || profile_running dmc9-dense-deepthin-0.3b-v2-en-cont; then
    log_json "doremi_v2_wait_training_running"
    return 0
  fi
  log_json "run_doremi_v2_proxy_pipeline"
  CUDA_VISIBLE_DEVICES="${DOREMI_CUDA_VISIBLE_DEVICES:-1}" "$PYTHON" scripts/saneflow_doremi_pipeline.py \
    --recipe configs/saneflow_practical_pretrain_mix_v2_en.json \
    --out runs/doremi_proxy_practical_v2_en \
    --reference-out runs/doremi_reference_practical_v2_en \
    --steps "${DOREMI_PROXY_STEPS:-200}" \
    --reference-steps "${DOREMI_REFERENCE_STEPS:-300}" \
    --seq-len "${DOREMI_PROXY_SEQ_LEN:-512}" \
    --batch-size "${DOREMI_PROXY_BATCH_SIZE:-1}" \
    --tokenizer-path "$(dense_tokenizer_path)" \
    --tf32 \
    --activation-checkpointing || log_json "doremi_v2_proxy_failed"
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

  speak_out="$(profile_out dmc8-speak-base-v1)"
  gate_path="${speak_out}/quality_gate.json"
  speak_model="$(profile_model dmc8-speak-base-v1)"
  if [[ ! -f "$gate_path" || "$speak_model" -nt "$gate_path" ]]; then
    log_json "gate_speak_base"
    run_gate "$speak_model" "$gate_path"
  fi

  if gate_passed "$gate_path"; then
    if ! profile_done dmc8-chatml-sft-v9; then
      log_json "ensure_chatml_sft_v9"
      start_profile_once dmc8-chatml-sft-v9 || true
      return 0
    fi
    log_json "gate_chatml_sft_v9"
    chat_model="$(profile_model dmc8-chatml-sft-v9)"
    chat_out="$(profile_out dmc8-chatml-sft-v9)"
    run_gate "$chat_model" "${chat_out}/quality_gate.json"
    mkdir -p runs/saneflow_current
    ln -sfn "$(pwd)/${chat_model}" runs/saneflow_current/model.pt
    log_json "promoted_chatml_sft_v9"
  else
    log_json "speak_gate_failed_hold_sft"
  fi
}

run_dmc9_once() {
  log_json "dmc9_cycle_start"
  ensure_practical_mix
  ensure_practical_v2_en_mix
  ensure_dense_tokenizer || true
  ensure_doremi_mix || true
  ensure_doremi_mix_v2_en || true
  if [[ data/corpus/mixes/saneflow_practical_pretrain_v1/doremi_ratios.json -nt data/corpus/mixes/saneflow_practical_pretrain_v1/train.jsonl ]]; then
    "$PYTHON" scripts/saneflow_build_practical_pretrain_mix.py || true
  fi
  log_json "ensure_dense_0_3b"
  start_profile_once dmc9-dense-0.3b-v1 || true
  log_json "ensure_dense_deepthin_0_3b"
  start_profile_once dmc9-dense-deepthin-0.3b-v1 || true
  if [[ -s data/corpus/mixes/saneflow_practical_pretrain_v2_en/train.jsonl && -s data/corpus/mixes/saneflow_practical_pretrain_v2_en/valid.jsonl && -f data/corpus/mixes/saneflow_practical_pretrain_v2_en/doremi_ratios.json ]]; then
    if profile_running dmc9-dense-0.3b-v1 || profile_running dmc9-dense-deepthin-0.3b-v1; then
      log_json "hold_v2_en_cont_until_v1_gpu_frees"
      return 0
    fi
    log_json "ensure_dense_v2_en_cont"
    start_profile_once dmc9-dense-0.3b-v2-en-cont || true
    log_json "ensure_dense_deepthin_v2_en_cont"
    start_profile_once dmc9-dense-deepthin-0.3b-v2-en-cont || true
  fi
  for profile in dmc9-dense-0.3b-v1 dmc9-dense-deepthin-0.3b-v1 dmc9-dense-0.3b-v2-en-cont dmc9-dense-deepthin-0.3b-v2-en-cont; do
    if profile_latest_exists "$profile"; then
      dense_latest="$(profile_latest "$profile")"
      dense_out="$(profile_out "$profile")"
      run_gate "$dense_latest" "${dense_out}/quality_gate_latest.json"
      run_reasoning_gate "$dense_latest" "${dense_out}/reasoning_gate_latest.json"
    fi
  done
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
