#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"

RUN_ROOT="${RUN_ROOT:-runs/mamba3_teacher_research/deepseek_v4_pro_0_3b}"
SUMMARY="${SUMMARY:-${RUN_ROOT}/summary.jsonl}"
BEST_DIR="${BEST_DIR:-${RUN_ROOT}/best}"
MODE="${MODE:-mamba3-siso-fast-0.3b-ds128}"
TOKENIZER="${TOKENIZER:-llama31}"
START_CHECKPOINT="${START_CHECKPOINT:-runs/mamba3_autonomous_hybrid_research/20260613T184309Z/mamba3-siso-fast-0.3b-ds128/model.pt}"
FALLBACK_CHECKPOINT="${FALLBACK_CHECKPOINT:-runs/mamba3_siso_fast_0_3b_ds128_v3/model.pt}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
DEEPSEEK_RECORDS="${DEEPSEEK_RECORDS:-20000}"
DEEPSEEK_RECORDS_CSV="${DEEPSEEK_RECORDS_CSV:-3000,12000,30000}"
DEEPSEEK_BATCH_SIZE="${DEEPSEEK_BATCH_SIZE:-8}"
BOOTSTRAP_EXISTING="${BOOTSTRAP_EXISTING:-1}"
BOOTSTRAP_DETERMINISTIC_RECORDS="${BOOTSTRAP_DETERMINISTIC_RECORDS:-120000}"
BOOTSTRAP_MIX_RECORDS="${BOOTSTRAP_MIX_RECORDS:-120000}"
MCQ_STEPS="${MCQ_STEPS:-2500}"
MCQ_LR="${MCQ_LR:-8e-6}"
MCQ_SEQ_LEN="${MCQ_SEQ_LEN:-256}"
MCQ_BATCH_SIZE="${MCQ_BATCH_SIZE:-8}"
BASE_ACCUM_STEPS="${BASE_ACCUM_STEPS:-1}"
ANSWER_ACCUM_STEPS="${ANSWER_ACCUM_STEPS:-3}"
MMLU_REDUX_LIMIT="${MMLU_REDUX_LIMIT:-200}"
CHAT_REPAIR="${CHAT_REPAIR:-1}"
CHAT_STEPS="${CHAT_STEPS:-1200}"
CHAT_LR="${CHAT_LR:-1e-5}"
CHAT_RECORDS="${CHAT_RECORDS:-60000}"
CHAT_BATCH_SIZE="${CHAT_BATCH_SIZE:-32}"
CHAT_SEQ_LEN="${CHAT_SEQ_LEN:-128}"
MIN_CHAT_PASS_RATE="${MIN_CHAT_PASS_RATE:-0.70}"
PROMOTE_CURRENT="${PROMOTE_CURRENT:-1}"
AUTO_RESTART_SERVER="${AUTO_RESTART_SERVER:-1}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"

mkdir -p "${RUN_ROOT}" "${BEST_DIR}"

json_field() {
  local path="$1" expr="$2" default="${3:-}"
  python - "$path" "$expr" "$default" <<'PY'
import json, sys
path, expr, default = sys.argv[1:4]
try:
    data = json.load(open(path))
except Exception:
    print(default)
    raise SystemExit(0)
cur = data
for part in expr.split("."):
    if isinstance(cur, dict):
        cur = cur.get(part)
    else:
        cur = None
        break
print(default if cur is None else cur)
PY
}

score_json() {
  local round="$1" phase="$2" ckpt="$3" mmlu="$4" gate="$5" promoted="$6"
  python - "$round" "$phase" "$ckpt" "$mmlu" "$gate" "$promoted" <<'PY'
import json, pathlib, sys, time
round_id, phase, ckpt, mmlu_path, gate_path, promoted = sys.argv[1:7]
def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return {}
mmlu = load(mmlu_path)
gate = load(gate_path)
payload = {
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "round": int(round_id),
    "phase": phase,
    "checkpoint": ckpt,
    "mmlu_redux_choice_acc": mmlu.get("choice_accuracy") or mmlu.get("choice_acc") or mmlu.get("accuracy") or mmlu.get("acc"),
    "mmlu_redux_letter_acc": mmlu.get("letter_accuracy") or mmlu.get("letter_acc"),
    "mmlu_redux_path": mmlu_path if pathlib.Path(mmlu_path).exists() else None,
    "chat_ok": gate.get("ok"),
    "chat_pass_rate": gate.get("pass_rate"),
    "chat_gate_path": gate_path if pathlib.Path(gate_path).exists() else None,
    "promoted": promoted == "1",
}
print(json.dumps(payload, ensure_ascii=False))
PY
}

better_than_best() {
  local candidate_json="$1"
  python - "$candidate_json" "${BEST_DIR}/best.json" <<'PY'
import json, sys
candidate_path, best_path = sys.argv[1:3]
try:
    cand = json.load(open(candidate_path))
except Exception:
    sys.exit(1)
try:
    best = json.load(open(best_path))
except Exception:
    best = {}
def val(d, k):
    try:
        return float(d.get(k) or 0)
    except Exception:
        return 0.0
c_acc = val(cand, "mmlu_redux_choice_acc")
b_acc = val(best, "mmlu_redux_choice_acc")
c_chat = val(cand, "chat_pass_rate")
b_chat = val(best, "chat_pass_rate")
if c_acc > b_acc or (c_acc == b_acc and c_chat >= b_chat):
    sys.exit(0)
sys.exit(1)
PY
}

write_metadata() {
  python - <<'PY'
import json, os, platform, time
payload = {
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "hostname": platform.node(),
    "goal": "DeepSeek teacher no-cheat MCQ/rationale research loop for 0.3B Mamba-3 SISO",
    "mode": os.environ.get("MODE"),
    "tokenizer": os.environ.get("TOKENIZER"),
    "max_rounds": int(os.environ.get("MAX_ROUNDS", "0")),
    "deepseek_records_per_round": os.environ.get("DEEPSEEK_RECORDS_CSV") or os.environ.get("DEEPSEEK_RECORDS", "0"),
    "mmlu_redux_limit": int(os.environ.get("MMLU_REDUX_LIMIT", "0")),
    "policy": "MMLU/MMLU-Redux are held-out eval only; do not train on benchmark examples.",
    "teacher": os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"),
}
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
}

export MODE TOKENIZER MAX_ROUNDS DEEPSEEK_RECORDS MMLU_REDUX_LIMIT
export DEEPSEEK_RECORDS_CSV
write_metadata | tee "${RUN_ROOT}/metadata.json"

if [[ ! -f "${START_CHECKPOINT}" ]]; then
  START_CHECKPOINT="${FALLBACK_CHECKPOINT}"
fi
if [[ ! -f "${START_CHECKPOINT}" ]]; then
  echo "missing START_CHECKPOINT=${START_CHECKPOINT}" >&2
  exit 2
fi

seed_checkpoint="${START_CHECKPOINT}"
if [[ -f "${BEST_DIR}/model.pt" ]]; then
  seed_checkpoint="${BEST_DIR}/model.pt"
fi

for round in $(seq 1 "${MAX_ROUNDS}"); do
  round_dir="${RUN_ROOT}/round_${round}"
  mkdir -p "${round_dir}"
  round_records="$(python - "${DEEPSEEK_RECORDS_CSV}" "${DEEPSEEK_RECORDS}" "${round}" <<'PY'
import sys
csv, default, round_id = sys.argv[1:4]
vals = [x.strip() for x in csv.split(",") if x.strip()]
idx = max(0, int(round_id) - 1)
print(vals[idx] if idx < len(vals) else default)
PY
)"
  echo "== round ${round} seed=${seed_checkpoint} =="
  teacher_provider="deepseek"
  mcq_data="${round_dir}/deepseek_no_cheat_mcq.jsonl"

  if [[ "${round}" == "1" && "${BOOTSTRAP_EXISTING}" == "1" ]]; then
    deterministic_data="${round_dir}/deterministic_no_cheat_mcq.jsonl"
    mix_data="${round_dir}/teacher_mcq_mix.jsonl"
    python scripts/mamba3_generate_no_cheat_mcq_sft.py \
      --out "${deterministic_data}" \
      --records "${BOOTSTRAP_DETERMINISTIC_RECORDS}" \
      --seed 20260614
    python scripts/mamba3_build_teacher_mcq_mix.py \
      --inputs \
        data/deepseek_no_cheat_mcq_sft_v1.jsonl \
        "${round_dir}/deepseek_no_cheat_mcq.jsonl" \
        "${deterministic_data}" \
      --out "${mix_data}" \
      --max-records "${BOOTSTRAP_MIX_RECORDS}" \
      --seed 20260614
    teacher_provider="file"
    mcq_data="${mix_data}"
  fi

  RUN_DIR="${round_dir}/mcq" \
  CHECKPOINT="${round_dir}/mcq/model.pt" \
  START_CHECKPOINT="${seed_checkpoint}" \
  TEACHER_PROVIDER="${teacher_provider}" \
  DEEPSEEK_ENV_FILE=.env \
  DEEPSEEK_RECORDS="${round_records}" \
  DEEPSEEK_BATCH_SIZE="${DEEPSEEK_BATCH_SIZE}" \
  DEEPSEEK_MCQ_DATA="${round_dir}/deepseek_no_cheat_mcq.jsonl" \
  MCQ_DATA="${mcq_data}" \
  STEPS="${MCQ_STEPS}" \
  LR="${MCQ_LR}" \
  SEQ_LEN="${MCQ_SEQ_LEN}" \
  BATCH_SIZE="${MCQ_BATCH_SIZE}" \
  BASE_ACCUM_STEPS="${BASE_ACCUM_STEPS}" \
  ANSWER_ACCUM_STEPS="${ANSWER_ACCUM_STEPS}" \
  MMLU_REDUX_LIMIT="${MMLU_REDUX_LIMIT}" \
  MODE="${MODE}" TOKENIZER="${TOKENIZER}" DEVICE="${DEVICE}" DTYPE="${DTYPE}" \
  scripts/mamba3_train_self_teacher_mcq_v1.sh

  mcq_ckpt="${round_dir}/mcq/model.pt"
  mcq_mmlu="${round_dir}/mcq/mmlu_redux_sample.json"
  mcq_gate="${round_dir}/mcq/chat_quality_gate.json"
  mcq_summary="${round_dir}/mcq_summary.json"
  score_json "${round}" "mcq" "${mcq_ckpt}" "${mcq_mmlu}" "${mcq_gate}" "0" | tee "${mcq_summary}" | tee -a "${SUMMARY}"

  final_ckpt="${mcq_ckpt}"
  final_mmlu="${mcq_mmlu}"
  final_gate="${mcq_gate}"
  phase="mcq"

  if [[ "${CHAT_REPAIR}" == "1" ]]; then
    chat_dir="${round_dir}/chat_repair"
    RUN_DIR="${chat_dir}" \
    START_CHECKPOINT="${mcq_ckpt}" \
    MODE="${MODE}" TOKENIZER="${TOKENIZER}" \
    SEQ_LEN="${CHAT_SEQ_LEN}" BATCH_SIZE="${CHAT_BATCH_SIZE}" \
    STEPS="${CHAT_STEPS}" LR="${CHAT_LR}" RECORDS="${CHAT_RECORDS}" \
    MIN_PASS_RATE="${MIN_CHAT_PASS_RATE}" \
    DEVICE="${DEVICE}" DTYPE="${DTYPE}" \
    scripts/mamba3_train_neurova_chat_v1.sh

    python scripts/mamba3_eval_mcq_bench.py \
      --suite mmlu_redux \
      --mmlu-subject all \
      --redux-filter ok \
      --limit "${MMLU_REDUX_LIMIT}" \
      --mode "${MODE}" \
      --tokenizer "${TOKENIZER}" \
      --checkpoint "${chat_dir}/chat.pt" \
      --seq-len 128 \
      --device "${DEVICE}" \
      --dtype "${DTYPE}" \
      --out "${chat_dir}/mmlu_redux_sample.json" | tee "${chat_dir}/mmlu_redux_sample.stdout" || true

    final_ckpt="${chat_dir}/chat.pt"
    final_mmlu="${chat_dir}/mmlu_redux_sample.json"
    final_gate="${chat_dir}/chat_quality_gate.json"
    phase="chat_repair"
  fi

  final_summary="${round_dir}/final_summary.json"
  promoted=0
  score_json "${round}" "${phase}" "${final_ckpt}" "${final_mmlu}" "${final_gate}" "0" > "${final_summary}"
  if better_than_best "${final_summary}"; then
    cp "${final_ckpt}" "${BEST_DIR}/model.pt"
    cp "${final_summary}" "${BEST_DIR}/best.json"
    [[ -f "${final_mmlu}" ]] && cp "${final_mmlu}" "${BEST_DIR}/mmlu_redux_sample.json"
    [[ -f "${final_gate}" ]] && cp "${final_gate}" "${BEST_DIR}/chat_quality_gate.json"
    seed_checkpoint="${BEST_DIR}/model.pt"
    if [[ "${PROMOTE_CURRENT}" == "1" ]] && [[ "$(json_field "${final_gate}" ok false)" == "True" ]]; then
      mkdir -p runs/mamba3_current
      cp "${final_ckpt}" runs/mamba3_current/model.pt
      cp "${final_summary}" runs/mamba3_current/teacher_research_metadata.json
      [[ -f "${final_gate}" ]] && cp "${final_gate}" runs/mamba3_current/chat_quality_gate.json
      [[ -f "${final_mmlu}" ]] && cp "${final_mmlu}" runs/mamba3_current/mmlu_redux_sample.json
      if [[ "${AUTO_RESTART_SERVER}" == "1" ]]; then
        NEUROVA_MAMBA3_CHECKPOINT=runs/mamba3_current/model.pt \
        NEUROVA_MAMBA3_MODE="${MODE}" \
        NEUROVA_MAMBA3_TOKENIZER="${TOKENIZER}" \
        NEUROVA_MAMBA3_SEQ=128 \
        NEUROVA_MAMBA3_DECODE_MODE=cache \
        NEUROVA_MAMBA3_CUDA_GRAPH=1 \
        scripts/mamba3_chat_serverctl.sh restart || true
      fi
      promoted=1
    fi
  fi
  score_json "${round}" "${phase}" "${final_ckpt}" "${final_mmlu}" "${final_gate}" "${promoted}" | tee -a "${SUMMARY}"
done

echo "teacher_research_finished_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
