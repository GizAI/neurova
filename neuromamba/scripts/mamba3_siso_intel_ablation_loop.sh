#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV:-mamba3_siso}" ]]; then
  # The Mamba CUDA extensions live in the project training env on ml-dmc8.
  if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-mamba3_siso}"
  fi
fi

RUN_ROOT="${RUN_ROOT:-neuromamba/runs/mamba3_siso_intel_ablation}"
TOKENIZER="${TOKENIZER:-llama31}"
MODES_CSV="${MODES_CSV:-mamba3-siso-fast-0.3b-ds128,mamba3-siso-fast-0.3b-ds128-outnorm,mamba3-siso-fast-0.3b-ds128-outnorm-meta8,mamba3-siso-fast-0.3b-intel-v2,mamba3-siso-fast-0.3b-stateedit-v1,mamba3-siso-fast-0.3b-intel-v3,mamba3-siso-deep-0.35b-intel}"
BASE_DATA="${BASE_DATA:-neuromamba/data/splits/no_cheat_knowledge_v1_train.jsonl}"
ANSWER_DATA="${ANSWER_DATA:-neuromamba/data/splits/no_cheat_mcq_sft_v1_train.jsonl}"
STEPS="${STEPS:-300}"
SEQ_LEN="${SEQ_LEN:-256}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-8e-6}"
BASE_ACCUM_STEPS="${BASE_ACCUM_STEPS:-1}"
ANSWER_ACCUM_STEPS="${ANSWER_ACCUM_STEPS:-1}"
OPTIMIZER="${OPTIMIZER:-adamw8bit}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
SAVE_EVERY="${SAVE_EVERY:-300}"
MMLU_REDUX_LIMIT="${MMLU_REDUX_LIMIT:-100}"
WAIT_FOR_TEACHER="${WAIT_FOR_TEACHER:-1}"
SUMMARY="${SUMMARY:-${RUN_ROOT}/summary.jsonl}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-neuromamba/runs/mamba3_current/model.pt}"

mkdir -p "${RUN_ROOT}"

if [[ "${WAIT_FOR_TEACHER}" == "1" ]]; then
  while pgrep -f "neuromamba/scripts/mamba3_teacher_research_loop.sh|neuromamba.cli train-|mamba3_eval_mcq_bench.py|mamba3_chat_quality_gate.py|mamba3_train_neurova_chat_v1.sh" >/dev/null; do
    echo "waiting_for_active_training_or_eval $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    sleep 120
  done
fi

IFS=',' read -r -a MODES <<< "${MODES_CSV}"
for mode in "${MODES[@]}"; do
  mode="${mode//[[:space:]]/}"
  [[ -n "${mode}" ]] || continue
  run_dir="${RUN_ROOT}/${mode}"
  ckpt="${run_dir}/model.pt"
  mkdir -p "${run_dir}"

  echo "== siso-intel-ablation mode=${mode} =="
  python -m neuromamba.cli model-info \
    --mode "${mode}" \
    --tokenizer "${TOKENIZER}" | tee "${run_dir}/model_info.json"

  if [[ -f "${INIT_CHECKPOINT}" ]]; then
    python neuromamba/scripts/mamba3_transplant_checkpoint.py \
      --source "${INIT_CHECKPOINT}" \
      --out "${ckpt}" \
      --mode "${mode}" \
      --tokenizer "${TOKENIZER}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}" | tee "${run_dir}/transplant.json"
  fi

  python -m neuromamba.cli train-multitask \
    --mode "${mode}" \
    --tokenizer "${TOKENIZER}" \
    --checkpoint "${ckpt}" \
    --base-data "${BASE_DATA}" \
    --answer-data "${ANSWER_DATA}" \
    --steps "${STEPS}" \
    --lr "${LR}" \
    --save-every "${SAVE_EVERY}" \
    --no-resume \
    --base-accum-steps "${BASE_ACCUM_STEPS}" \
    --answer-accum-steps "${ANSWER_ACCUM_STEPS}" \
    --optimizer "${OPTIMIZER}" \
    --seq-len "${SEQ_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" | tee "${run_dir}/train.log"

  python -m neuromamba.cli eval-answer-loss \
    --mode "${mode}" \
    --tokenizer "${TOKENIZER}" \
    --checkpoint "${ckpt}" \
    --data neuromamba/data/splits/no_cheat_mcq_sft_v1_valid.jsonl \
    --seq-len "${SEQ_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --batches 32 | tee "${run_dir}/eval_answer_loss.json"

  python neuromamba/scripts/mamba3_eval_mcq_bench.py \
    --suite mmlu_redux \
    --mmlu-subject all \
    --redux-filter ok \
    --limit "${MMLU_REDUX_LIMIT}" \
    --mode "${mode}" \
    --tokenizer "${TOKENIZER}" \
    --checkpoint "${ckpt}" \
    --seq-len 128 \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --out "${run_dir}/mmlu_redux_sample.json" | tee "${run_dir}/mmlu_redux_sample.stdout" || true

  python - "$mode" "$run_dir" <<'PY' | tee -a "${SUMMARY}"
import json, pathlib, sys, time
mode, run_dir = sys.argv[1:3]
root = pathlib.Path(run_dir)
def load(name):
    try:
        return json.load(open(root / name))
    except Exception:
        return {}
info = load("model_info.json")
eval_loss = load("eval_answer_loss.json")
mmlu = load("mmlu_redux_sample.json")
payload = {
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "mode": mode,
    "estimated_parameters": info.get("estimated_parameters"),
    "eval_answer_loss": eval_loss.get("loss"),
    "mmlu_redux_choice_acc": mmlu.get("choice_accuracy") or mmlu.get("accuracy"),
    "mmlu_redux_letter_acc": mmlu.get("letter_accuracy"),
    "run_dir": run_dir,
}
print(json.dumps(payload, ensure_ascii=False))
PY
done

echo "summary=${SUMMARY}"
