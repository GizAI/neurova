#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RECIPE="${RECIPE:-memory_proof}"
RUN_DIR="${RUN_DIR:-luma/runs/luma_memory_proof_v1}"
DOC_CORPUS="${DOC_CORPUS:-luma/data/base_doc_continuation_v1.jsonl}"
TRAIN_DATA="${TRAIN_DATA:-luma/data/splits/base_doc_cont_train.txt}"
VALID_DATA="${VALID_DATA:-luma/data/splits/base_doc_cont_valid.txt}"
RAW_DATA="${RAW_DATA-luma/data/luma_stage_raw_cont_v1.jsonl}"
QA_DATA="${QA_DATA:-}"
CHAT_DATA="${CHAT_DATA-luma/data/luma_stage_chatml_sft_v1.jsonl}"
MEMORY_DATA="${MEMORY_DATA-luma/data/luma_stage_chatml_memory_v1.jsonl}"
SEQ_LEN="${SEQ_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-6}"
STEPS="${STEPS:-2000}"
LR="${LR:-3e-4}"
D_MODEL="${D_MODEL:-384}"
LAYERS="${LAYERS:-8}"
SLOTS="${SLOTS:-192}"
TOPK="${TOPK:-8}"
CHUNK="${CHUNK:-64}"
LOCAL_HEADS="${LOCAL_HEADS:-4}"
USE_SLOTS="${USE_SLOTS:-1}"
USE_LOCAL_ATTENTION="${USE_LOCAL_ATTENTION:-1}"
COPY_WINDOW="${COPY_WINDOW:-128}"
MAX_RECORDS="${MAX_RECORDS:-0}"
MAX_RAW_RECORDS="${MAX_RAW_RECORDS:-50000}"
MAX_QA_RECORDS="${MAX_QA_RECORDS:-0}"
MAX_CHAT_RECORDS="${MAX_CHAT_RECORDS:-0}"
MAX_MEMORY_RECORDS="${MAX_MEMORY_RECORDS:-0}"
RAW_DATASET_MODE="${RAW_DATASET_MODE:-packed}"
RAW_ANSWER_ONLY="${RAW_ANSWER_ONLY:-0}"
RAW_WEIGHT="${RAW_WEIGHT:-0.35}"
QA_WEIGHT="${QA_WEIGHT:-0.0}"
CHAT_WEIGHT="${CHAT_WEIGHT:-0.20}"
MEMORY_WEIGHT="${MEMORY_WEIGHT:-0.25}"
SLOT_PROOF_WEIGHT="${SLOT_PROOF_WEIGHT:-0.20}"
SLOT_PROOF_GAP_LINES="${SLOT_PROOF_GAP_LINES:-8}"
EVAL_CASES="${EVAL_CASES:-30}"
EVAL_MAX_NEW="${EVAL_MAX_NEW:-32}"
SAVE_EVERY="${SAVE_EVERY:-100}"
RUN_GENERATE="${RUN_GENERATE:-1}"
RUN_MEMORY_EVAL="${RUN_MEMORY_EVAL:-1}"
INIT_FROM="${INIT_FROM:-}"
AB_MARGIN_WEIGHT="${AB_MARGIN_WEIGHT:-0.0}"
MEMORY_LOGIT_WEIGHT="${MEMORY_LOGIT_WEIGHT:-0.0}"
MEMORY_READ_BIAS="${MEMORY_READ_BIAS:--6.0}"
TOKENIZER_BACKEND="${TOKENIZER_BACKEND:-bytepatch}"
QWEN_TOKENIZER_PATH="${QWEN_TOKENIZER_PATH:-luma/tokenizers/qwen35}"
BYTEPATCH_VOCAB_PATH="${BYTEPATCH_VOCAB_PATH:-luma/tokenizers/luma_bytepatch/bytepatch_vocab.json}"
export RUN_DIR RECIPE RAW_DATA QA_DATA CHAT_DATA MEMORY_DATA SEQ_LEN BATCH_SIZE STEPS D_MODEL LAYERS SLOTS TOPK TOKENIZER_BACKEND BYTEPATCH_VOCAB_PATH SLOT_PROOF_WEIGHT SLOT_PROOF_GAP_LINES EVAL_CASES EVAL_MAX_NEW INIT_FROM

cd "${ROOT}"
mkdir -p "${RUN_DIR}" luma/data/splits
exec > >(tee "${RUN_DIR}/train.log") 2>&1

echo "== LUMA dmc8 metadata =="
python - <<'PY'
import json, os, platform, time
print(json.dumps({
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "hostname": platform.node(),
    "run_dir": os.environ.get("RUN_DIR", "luma/runs/luma_dmc8_proto"),
    "recipe": os.environ.get("RECIPE", "memory_proof"),
    "raw_data": os.environ.get("RAW_DATA", "luma/data/splits/base_doc_cont_train.txt"),
    "qa_data": os.environ.get("QA_DATA", ""),
    "chat_data": os.environ.get("CHAT_DATA", "luma/data/neurova_chat_sft_v1.jsonl"),
    "memory_data": os.environ.get("MEMORY_DATA", "luma/data/luma_memory_curriculum_v1.jsonl"),
    "seq_len": os.environ.get("SEQ_LEN", "512"),
    "batch_size": os.environ.get("BATCH_SIZE", "16"),
    "steps": os.environ.get("STEPS", "2000"),
    "d_model": os.environ.get("D_MODEL", "256"),
    "layers": os.environ.get("LAYERS", "6"),
    "slots": os.environ.get("SLOTS", "128"),
    "topk": os.environ.get("TOPK", "8"),
    "local_heads": os.environ.get("LOCAL_HEADS", "4"),
    "use_slots": os.environ.get("USE_SLOTS", "1"),
    "use_local_attention": os.environ.get("USE_LOCAL_ATTENTION", "1"),
    "slot_proof_weight": os.environ.get("SLOT_PROOF_WEIGHT", "0.20"),
    "slot_proof_gap_lines": os.environ.get("SLOT_PROOF_GAP_LINES", "8"),
    "init_from": os.environ.get("INIT_FROM", ""),
    "eval_cases": os.environ.get("EVAL_CASES", "30"),
    "tokenizer_backend": os.environ.get("TOKENIZER_BACKEND", "bytepatch"),
    "bytepatch_vocab_path": os.environ.get("BYTEPATCH_VOCAB_PATH", "luma/tokenizers/luma_bytepatch/bytepatch_vocab.json"),
}, ensure_ascii=False, indent=2))
PY

if [[ "${RAW_DATA}" == *"${TRAIN_DATA}"* && ( ! -f "${TRAIN_DATA}" || ! -f "${VALID_DATA}" ) ]]; then
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

read -r -a RAW_DATA_ARGS <<< "${RAW_DATA}"
read -r -a QA_DATA_ARGS <<< "${QA_DATA}"
read -r -a CHAT_DATA_ARGS <<< "${CHAT_DATA}"
read -r -a MEMORY_DATA_ARGS <<< "${MEMORY_DATA}"

TRAIN_ARGS=(
  --out "${RUN_DIR}"
  --recipe "${RECIPE}"
  --max-records "${MAX_RECORDS}"
  --max-raw-records "${MAX_RAW_RECORDS}"
  --max-qa-records "${MAX_QA_RECORDS}"
  --max-chat-records "${MAX_CHAT_RECORDS}"
  --max-memory-records "${MAX_MEMORY_RECORDS}"
  --raw-dataset-mode "${RAW_DATASET_MODE}"
  --raw-weight "${RAW_WEIGHT}"
  --qa-weight "${QA_WEIGHT}"
  --chat-weight "${CHAT_WEIGHT}"
  --memory-weight "${MEMORY_WEIGHT}"
  --slot-proof-weight "${SLOT_PROOF_WEIGHT}"
  --slot-proof-gap-lines "${SLOT_PROOF_GAP_LINES}"
  --steps "${STEPS}"
  --batch-size "${BATCH_SIZE}"
  --seq-len "${SEQ_LEN}"
  --d-model "${D_MODEL}"
  --layers "${LAYERS}"
  --slots "${SLOTS}"
  --topk "${TOPK}"
  --chunk "${CHUNK}"
  --local-heads "${LOCAL_HEADS}"
  --copy-window "${COPY_WINDOW}"
  --lr "${LR}"
  --device cuda
  --dtype bf16
  --tokenizer-backend "${TOKENIZER_BACKEND}"
  --qwen-tokenizer-path "${QWEN_TOKENIZER_PATH}"
  --bytepatch-vocab-path "${BYTEPATCH_VOCAB_PATH}"
  --ablation-margin-weight "${AB_MARGIN_WEIGHT}"
  --memory-logit-weight "${MEMORY_LOGIT_WEIGHT}"
  --memory-read-bias "${MEMORY_READ_BIAS}"
  --save-every "${SAVE_EVERY}"
)
if [[ "${USE_SLOTS}" == "0" ]]; then
  TRAIN_ARGS+=(--disable-slots)
fi
if [[ "${USE_LOCAL_ATTENTION}" == "0" ]]; then
  TRAIN_ARGS+=(--disable-local-attention)
fi
if [[ "${RAW_ANSWER_ONLY}" == "1" ]]; then
  TRAIN_ARGS+=(--raw-answer-only)
fi
if [[ -n "${INIT_FROM}" ]]; then
  TRAIN_ARGS+=(--init-from "${INIT_FROM}")
fi
if [[ -n "${RAW_DATA}" ]]; then
  TRAIN_ARGS+=(--raw-data "${RAW_DATA_ARGS[@]}")
fi
if [[ -n "${QA_DATA}" ]]; then
  TRAIN_ARGS+=(--qa-data "${QA_DATA_ARGS[@]}")
fi
if [[ -n "${CHAT_DATA}" ]]; then
  TRAIN_ARGS+=(--chat-data "${CHAT_DATA_ARGS[@]}")
fi
if [[ -n "${MEMORY_DATA}" ]]; then
  TRAIN_ARGS+=(--memory-data "${MEMORY_DATA_ARGS[@]}")
fi

python -m luma.train "${TRAIN_ARGS[@]}"

if [[ "${RUN_GENERATE}" == "1" ]]; then
  python -m luma.generate \
    --ckpt "${RUN_DIR}/model.pt" \
    --prompt $'Memory page:\nMina owns the blue key.\nMina should go to seoul.\nQuestion: What object belongs to Mina?\nAnswer:' \
    --max-new 96 \
    --temperature 0.8 | tee "${RUN_DIR}/decode_probe.txt" || true
fi

if [[ "${RUN_MEMORY_EVAL}" == "1" ]]; then
  python -m luma.eval_memory \
    --ckpt "${RUN_DIR}/model.pt" \
    --cases "${EVAL_CASES}" \
    --max-new "${EVAL_MAX_NEW}" \
    --dtype bf16 \
    --out "${RUN_DIR}/memory_ablation_eval.json" \
    --compare-ablations | tee "${RUN_DIR}/memory_ablation_eval.stdout.json" || true
fi

echo "luma_checkpoint=${RUN_DIR}/model.pt"
