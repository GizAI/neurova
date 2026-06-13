#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_DIR="${RUN_DIR:-runs/luma_dmc8_proto}"
DOC_CORPUS="${DOC_CORPUS:-data/base_doc_continuation_v1.jsonl}"
TRAIN_DATA="${TRAIN_DATA:-data/splits/base_doc_cont_train.txt}"
VALID_DATA="${VALID_DATA:-data/splits/base_doc_cont_valid.txt}"
SEQ_LEN="${SEQ_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-16}"
STEPS="${STEPS:-2000}"
LR="${LR:-3e-4}"
D_MODEL="${D_MODEL:-256}"
LAYERS="${LAYERS:-6}"
SLOTS="${SLOTS:-128}"
TOPK="${TOPK:-8}"
CHUNK="${CHUNK:-32}"
MAX_RECORDS="${MAX_RECORDS:-0}"

cd "${ROOT}"
mkdir -p "${RUN_DIR}" data/splits
exec > >(tee "${RUN_DIR}/train.log") 2>&1

echo "== LUMA dmc8 metadata =="
python - <<'PY'
import json, os, platform, time
print(json.dumps({
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "hostname": platform.node(),
    "run_dir": os.environ.get("RUN_DIR", "runs/luma_dmc8_proto"),
    "train_data": os.environ.get("TRAIN_DATA", "data/splits/base_doc_cont_train.txt"),
    "seq_len": os.environ.get("SEQ_LEN", "512"),
    "batch_size": os.environ.get("BATCH_SIZE", "16"),
    "steps": os.environ.get("STEPS", "2000"),
    "d_model": os.environ.get("D_MODEL", "256"),
    "layers": os.environ.get("LAYERS", "6"),
    "slots": os.environ.get("SLOTS", "128"),
    "topk": os.environ.get("TOPK", "8"),
}, ensure_ascii=False, indent=2))
PY

if [[ ! -f "${TRAIN_DATA}" || ! -f "${VALID_DATA}" ]]; then
  if [[ ! -f "${DOC_CORPUS}" ]]; then
    echo "missing ${DOC_CORPUS}; build/download Mamba corpus first" >&2
    exit 2
  fi
  python scripts/mamba3_make_source_stratified_splits.py \
    --inputs "${DOC_CORPUS}" \
    --train-out "${TRAIN_DATA}" \
    --valid-out "${VALID_DATA}" \
    --valid-ratio 0.02 \
    --seed 4242
fi

python -m luma.train \
  --out "${RUN_DIR}" \
  --data "${TRAIN_DATA}" \
  --max-records "${MAX_RECORDS}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --seq-len "${SEQ_LEN}" \
  --d-model "${D_MODEL}" \
  --layers "${LAYERS}" \
  --slots "${SLOTS}" \
  --topk "${TOPK}" \
  --chunk "${CHUNK}" \
  --lr "${LR}" \
  --device cuda \
  --dtype bf16

python -m luma.generate \
  --ckpt "${RUN_DIR}/model.pt" \
  --prompt $'Memory page:\nMina owns the blue key.\nMina should go to seoul.\nQuestion: What object belongs to Mina?\nAnswer:' \
  --max-new 96 \
  --temperature 0.8 | tee "${RUN_DIR}/decode_probe.txt" || true

echo "luma_checkpoint=${RUN_DIR}/model.pt"
