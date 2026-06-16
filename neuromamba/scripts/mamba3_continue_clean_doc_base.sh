#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODE="${MODE:-mimo-r4-moe-900m}"
TOKENIZER="${TOKENIZER:-llama31}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"

TRAIN_DATA="${TRAIN_DATA:-neuromamba/data/splits/base_doc_cont_v3_train.jsonl}"
VALID_DATA="${VALID_DATA:-neuromamba/data/splits/base_doc_cont_v3_valid.jsonl}"
RUN_DIR="${RUN_DIR:-neuromamba/runs/mamba3_clean_doc_base_moe900_v1}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/base.pt}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/continuations}"

SEQ_LEN="${SEQ_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-3}"
STEPS="${STEPS:-500}"
SAVE_EVERY="${SAVE_EVERY:-250}"
LR="${LR:-2e-5}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
OPTIMIZER="${OPTIMIZER:-adamw8bit}"
EVAL_BATCHES="${EVAL_BATCHES:-32}"
PROMPT="${PROMPT:-<doc source=\"probe\" domain=\"science\"> The main idea is}"
MAX_NEW="${MAX_NEW:-96}"
NO_SAVE_OPTIMIZER="${NO_SAVE_OPTIMIZER:-1}"
SHUFFLE_TEXTS="${SHUFFLE_TEXTS:-1}"
DATA_SEED="${DATA_SEED:-$(date -u +%s)}"

cd "${ROOT}"
mkdir -p "${LOG_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
train_log="${LOG_DIR}/${stamp}_train.log"
eval_json="${LOG_DIR}/${stamp}_eval.json"
decode_log="${LOG_DIR}/${stamp}_decode.txt"
meta_json="${LOG_DIR}/${stamp}_meta.json"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "missing CHECKPOINT=${CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -f "${TRAIN_DATA}" || ! -f "${VALID_DATA}" ]]; then
  echo "missing train/valid data: ${TRAIN_DATA} ${VALID_DATA}" >&2
  exit 2
fi

python neuromamba/scripts/mamba3_validate_doc_continuation_split.py "${TRAIN_DATA}" "${VALID_DATA}" >/dev/null

python - <<'PY' > "${meta_json}"
import json, os, platform, subprocess, time
payload = {
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "hostname": platform.node(),
    "mode": os.environ.get("MODE", "mimo-r4-moe-900m"),
    "tokenizer": os.environ.get("TOKENIZER", "llama31"),
    "checkpoint": os.environ.get("CHECKPOINT", "neuromamba/runs/mamba3_clean_doc_base_moe900_v1/base.pt"),
    "train_data": os.environ.get("TRAIN_DATA", "neuromamba/data/splits/base_doc_cont_v3_train.jsonl"),
    "valid_data": os.environ.get("VALID_DATA", "neuromamba/data/splits/base_doc_cont_v3_valid.jsonl"),
    "seq_len": int(os.environ.get("SEQ_LEN", "2048")),
    "batch_size": int(os.environ.get("BATCH_SIZE", "3")),
    "steps": int(os.environ.get("STEPS", "500")),
    "save_every": int(os.environ.get("SAVE_EVERY", "250")),
    "lr": float(os.environ.get("LR", "2e-5")),
    "grad_accum_steps": int(os.environ.get("GRAD_ACCUM_STEPS", "1")),
    "optimizer": os.environ.get("OPTIMIZER", "adamw8bit"),
    "shuffle_texts": os.environ.get("SHUFFLE_TEXTS", "1") not in {"0", "false", "False"},
    "data_seed": int(os.environ.get("DATA_SEED", "0")),
    "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
    "weight_only_restart": True,
    "save_optimizer": os.environ.get("NO_SAVE_OPTIMIZER", "1") != "1",
}
try:
    payload["git_head"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    payload["git_head"] = "unavailable"
print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
PY

echo "== clean doc weight-only continuation metadata =="
cat "${meta_json}"

train_args=(
  python -m neuromamba.cli train-packed
  --mode "${MODE}"
  --tokenizer "${TOKENIZER}"
  --checkpoint "${CHECKPOINT}"
  --data "${TRAIN_DATA}"
  --steps "${STEPS}"
  --lr "${LR}"
  --save-every "${SAVE_EVERY}"
  --grad-accum-steps "${GRAD_ACCUM_STEPS}"
  --optimizer "${OPTIMIZER}"
  --seq-len "${SEQ_LEN}"
  --batch-size "${BATCH_SIZE}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --max-text-chars "${MAX_TEXT_CHARS:-65536}"
  --max-text-tokens "${MAX_TEXT_TOKENS:-120000}"
  --no-resume
)
if [[ "${NO_SAVE_OPTIMIZER}" == "1" ]]; then
  train_args+=(--no-save-optimizer)
fi
if [[ "${SHUFFLE_TEXTS}" != "0" && "${SHUFFLE_TEXTS}" != "false" ]]; then
  train_args+=(--shuffle-texts --data-seed "${DATA_SEED}")
fi

echo "== clean doc weight-only continuation training =="
"${train_args[@]}" | tee "${train_log}"

echo "== held-out doc continuation loss =="
python -m neuromamba.cli eval-loss \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --data "${VALID_DATA}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --batches "${EVAL_BATCHES}" | tee "${eval_json}"

echo "== raw continuation decode probe =="
python -m neuromamba.cli fast-generate \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --prompt "${PROMPT}" \
  --max-new "${MAX_NEW}" \
  --seq-len "${SEQ_LEN}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --top-k 1 \
  --top-p 0 \
  --temperature 1.0 \
  --repetition-penalty 1.0 \
  --safe-decode | tee "${decode_log}" || true

echo "metadata=${meta_json}"
echo "train_log=${train_log}"
echo "eval_json=${eval_json}"
echo "decode_log=${decode_log}"
