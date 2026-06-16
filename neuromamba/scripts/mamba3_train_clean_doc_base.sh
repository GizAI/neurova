#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODE="${MODE:-mimo-r4-moe-900m}"
TOKENIZER="${TOKENIZER:-llama31}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"

DOC_CORPUS="${DOC_CORPUS:-neuromamba/data/base_doc_continuation_v1.jsonl}"
TRAIN_DATA="${TRAIN_DATA:-neuromamba/data/splits/base_doc_cont_train.txt}"
VALID_DATA="${VALID_DATA:-neuromamba/data/splits/base_doc_cont_valid.txt}"
OVERFIT_DATA="${OVERFIT_DATA:-neuromamba/data/splits/base_doc_cont_overfit64.txt}"

RUN_DIR="${RUN_DIR:-neuromamba/runs/mamba3_clean_doc_base_moe900_v1}"
OVERFIT_CKPT="${OVERFIT_CKPT:-${RUN_DIR}/overfit.pt}"
BASE_CKPT="${BASE_CKPT:-${RUN_DIR}/base.pt}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/train.log}"

SEQ_LEN="${SEQ_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-3}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LR="${LR:-5e-5}"
OPTIMIZER="${OPTIMIZER:-adamw8bit}"

OVERFIT_RECORDS="${OVERFIT_RECORDS:-64}"
OVERFIT_STEPS="${OVERFIT_STEPS:-200}"
BASE_STEPS="${BASE_STEPS:-2000}"
SAVE_EVERY="${SAVE_EVERY:-200}"
EVAL_BATCHES="${EVAL_BATCHES:-32}"
VALID_RATIO="${VALID_RATIO:-0.02}"
SEED="${SEED:-4242}"

cd "${ROOT}"
mkdir -p "${RUN_DIR}" "$(dirname "${TRAIN_DATA}")"
exec > >(tee "${LOG_FILE}") 2>&1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "== clean doc base metadata =="
python - <<'PY'
import json, os, platform, subprocess, time
payload = {
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "hostname": platform.node(),
    "mode": os.environ.get("MODE", "mimo-r4-moe-900m"),
    "tokenizer": os.environ.get("TOKENIZER", "llama31"),
    "seq_len": os.environ.get("SEQ_LEN", "2048"),
    "batch_size": os.environ.get("BATCH_SIZE", "3"),
    "grad_accum_steps": os.environ.get("GRAD_ACCUM_STEPS", "1"),
    "lr": os.environ.get("LR", "5e-5"),
    "optimizer": os.environ.get("OPTIMIZER", "adamw8bit"),
    "doc_corpus": os.environ.get("DOC_CORPUS", "neuromamba/data/base_doc_continuation_v1.jsonl"),
    "train_data": os.environ.get("TRAIN_DATA", "neuromamba/data/splits/base_doc_cont_train.txt"),
    "valid_data": os.environ.get("VALID_DATA", "neuromamba/data/splits/base_doc_cont_valid.txt"),
    "overfit_data": os.environ.get("OVERFIT_DATA", "neuromamba/data/splits/base_doc_cont_overfit64.txt"),
    "base_steps": os.environ.get("BASE_STEPS", "2000"),
    "overfit_steps": os.environ.get("OVERFIT_STEPS", "200"),
    "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
}
try:
    payload["git_head"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    payload["git_head"] = "unavailable"
print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
PY

if [[ ! -f "${DOC_CORPUS}" ]]; then
  echo "missing DOC_CORPUS=${DOC_CORPUS}; build it with neuromamba/scripts/mamba3_build_doc_continuation_corpus.py" >&2
  exit 2
fi

if [[ ! -f "${TRAIN_DATA}" || ! -f "${VALID_DATA}" ]]; then
  echo "== build source-stratified doc split =="
  python neuromamba/scripts/mamba3_make_source_stratified_splits.py \
    --inputs "${DOC_CORPUS}" \
    --train-out "${TRAIN_DATA}" \
    --valid-out "${VALID_DATA}" \
    --valid-ratio "${VALID_RATIO}" \
    --seed "${SEED}"
fi

echo "== build overfit sanity subset =="
python - <<'PY'
import os
from pathlib import Path

train = Path(os.environ.get("TRAIN_DATA", "neuromamba/data/splits/base_doc_cont_train.txt"))
out = Path(os.environ.get("OVERFIT_DATA", "neuromamba/data/splits/base_doc_cont_overfit64.txt"))
n = int(os.environ.get("OVERFIT_RECORDS", "64"))
out.parent.mkdir(parents=True, exist_ok=True)
with train.open("r", encoding="utf-8") as src, out.open("w", encoding="utf-8") as dst:
    written = 0
    for line in src:
        if line.strip():
            dst.write(line)
            written += 1
            if written >= n:
                break
print({"overfit_data": str(out), "records": written})
PY

rm -f "${OVERFIT_CKPT}" "${BASE_CKPT}"

echo "== stage 0: small-data overfit sanity =="
python -m neuromamba.cli train-packed \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${OVERFIT_CKPT}" \
  --data "${OVERFIT_DATA}" \
  --steps "${OVERFIT_STEPS}" \
  --lr "${LR}" \
  --save-every "${SAVE_EVERY}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --optimizer "${OPTIMIZER}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --no-resume

echo "== stage 0 eval: overfit subset =="
python -m neuromamba.cli eval-loss \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${OVERFIT_CKPT}" \
  --data "${OVERFIT_DATA}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --batches "${EVAL_BATCHES}"

echo "== stage 1: raw-document CLM from scratch =="
python -m neuromamba.cli train-packed \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${BASE_CKPT}" \
  --data "${TRAIN_DATA}" \
  --steps "${BASE_STEPS}" \
  --lr "${LR}" \
  --save-every "${SAVE_EVERY}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --optimizer "${OPTIMIZER}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --no-resume

echo "== stage 1 eval: held-out doc continuation =="
python -m neuromamba.cli eval-loss \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${BASE_CKPT}" \
  --data "${VALID_DATA}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --batches "${EVAL_BATCHES}"

echo "== stage 1 decode probe: raw continuation only =="
python -m neuromamba.cli fast-generate \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${BASE_CKPT}" \
  --prompt '<doc source="probe" domain="science"> The main idea is' \
  --max-new 96 \
  --seq-len "${SEQ_LEN}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --top-k 1 \
  --top-p 0 \
  --temperature 1.0 \
  --repetition-penalty 1.0 \
  --safe-decode || true

echo "clean_doc_base_run=${RUN_DIR}"
