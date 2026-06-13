#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODE="${MODE:-mimo-r4-tiny}"
TOKENIZER="${TOKENIZER:-llama31}"
SEQ_LEN="${SEQ_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-4}"
BASE_STEPS="${BASE_STEPS:-200}"
CURRICULUM_STEPS="${CURRICULUM_STEPS:-0}"
SFT_STEPS="${SFT_STEPS:-200}"
BASE_LR="${BASE_LR:-2e-4}"
CURRICULUM_LR="${CURRICULUM_LR:-1e-4}"
SFT_LR="${SFT_LR:-1e-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
OPTIMIZER="${OPTIMIZER:-adamw}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-0}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
RUN_DIR="${RUN_DIR:-runs/mamba3_kr_scientific}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/train.log}"
PROGRAMMATIC_DATA="${PROGRAMMATIC_DATA:-data/mamba3_programmatic_curriculum.jsonl}"
PROGRAMMATIC_RECORDS="${PROGRAMMATIC_RECORDS:-5000}"
PROGRAMMATIC_REGENERATE="${PROGRAMMATIC_REGENERATE:-0}"
CURRICULUM_MAX_RECORDS="${CURRICULUM_MAX_RECORDS:-0}"
SFT_NATURAL_MAX_RECORDS="${SFT_NATURAL_MAX_RECORDS:-0}"
SFT_PROGRAMMATIC_MAX_RECORDS="${SFT_PROGRAMMATIC_MAX_RECORDS:-0}"
SFT_NATURAL_FORMAT="${SFT_NATURAL_FORMAT:-answer}"
SFT_PROGRAMMATIC_FORMAT="${SFT_PROGRAMMATIC_FORMAT:-qa}"
CUDA_GRAPH="${CUDA_GRAPH:-1}"

cd "${ROOT}"
mkdir -p data/splits "${RUN_DIR}"
exec > >(tee "${LOG_FILE}") 2>&1

echo "== scientific run metadata =="
python - <<'PY'
import json, os, platform, subprocess, time
payload = {
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "hostname": platform.node(),
    "mode": os.environ.get("MODE", "mimo-r4-tiny"),
    "tokenizer": os.environ.get("TOKENIZER", "llama31"),
    "seq_len": os.environ.get("SEQ_LEN", "128"),
    "batch_size": os.environ.get("BATCH_SIZE", "4"),
    "base_steps": os.environ.get("BASE_STEPS", "200"),
    "curriculum_steps": os.environ.get("CURRICULUM_STEPS", "0"),
    "sft_steps": os.environ.get("SFT_STEPS", "200"),
    "grad_accum_steps": os.environ.get("GRAD_ACCUM_STEPS", "1"),
    "optimizer": os.environ.get("OPTIMIZER", "adamw"),
    "deepspeed_config": os.environ.get("DEEPSPEED_CONFIG", ""),
    "activation_checkpointing": os.environ.get("ACTIVATION_CHECKPOINTING", "0"),
    "programmatic_data": os.environ.get("PROGRAMMATIC_DATA", "data/mamba3_programmatic_curriculum.jsonl"),
    "programmatic_records": os.environ.get("PROGRAMMATIC_RECORDS", "5000"),
    "curriculum_max_records": os.environ.get("CURRICULUM_MAX_RECORDS", "0"),
    "sft_natural_max_records": os.environ.get("SFT_NATURAL_MAX_RECORDS", "0"),
    "sft_programmatic_max_records": os.environ.get("SFT_PROGRAMMATIC_MAX_RECORDS", "0"),
    "sft_natural_format": os.environ.get("SFT_NATURAL_FORMAT", "answer"),
    "sft_programmatic_format": os.environ.get("SFT_PROGRAMMATIC_FORMAT", "qa"),
    "cuda_graph": os.environ.get("CUDA_GRAPH", "1"),
}
try:
    payload["git_head"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    payload["git_head"] = "unavailable"
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

echo "== split governed base data =="
python scripts/mamba3_make_splits.py \
  --inputs data/governed_fineweb_edu_sample.jsonl data/governed_dclm_sample.jsonl \
  --train-out data/splits/base_train.jsonl \
  --valid-out data/splits/base_valid.jsonl \
  --valid-ratio 0.02 \
  --seed 1337

echo "== split instruction SFT data =="
if [[ "${PROGRAMMATIC_REGENERATE}" == "1" || "${PROGRAMMATIC_REGENERATE}" == "true" || ! -f "${PROGRAMMATIC_DATA}" ]]; then
  echo "== generate no-teacher programmatic curriculum =="
  python scripts/mamba3_generate_programmatic_curriculum.py \
    --out "${PROGRAMMATIC_DATA}" \
    --records "${PROGRAMMATIC_RECORDS}" \
    --seed 20260613
fi

python scripts/mamba3_build_sft_mix.py \
  --natural-inputs data/english_instruction_bootstrap.txt data/english_completion_bootstrap.txt \
  --programmatic-inputs "${PROGRAMMATIC_DATA}" \
  --train-out data/splits/sft_train.txt \
  --valid-out data/splits/sft_valid.txt \
  --natural-max-records "${SFT_NATURAL_MAX_RECORDS}" \
  --programmatic-max-records "${SFT_PROGRAMMATIC_MAX_RECORDS}" \
  --natural-format "${SFT_NATURAL_FORMAT}" \
  --programmatic-format "${SFT_PROGRAMMATIC_FORMAT}" \
  --valid-ratio 0.10 \
  --seed 2026

if [[ "${CURRICULUM_STEPS}" != "0" ]]; then
  echo "== split separated recall/copy curriculum data =="
  python scripts/mamba3_make_splits.py \
    --inputs "${PROGRAMMATIC_DATA}" \
    --train-out data/splits/curriculum_train.txt \
    --valid-out data/splits/curriculum_valid.txt \
    --valid-ratio 0.10 \
    --max-records "${CURRICULUM_MAX_RECORDS}" \
    --seed 2027
fi

BASE_CKPT="${RUN_DIR}/base.pt"
CURRICULUM_CKPT="${RUN_DIR}/curriculum.pt"
SFT_CKPT="${RUN_DIR}/sft.pt"

echo "== base pretrain canary from scratch =="
CKPT_FLAG=()
if [[ "${ACTIVATION_CHECKPOINTING}" == "1" || "${ACTIVATION_CHECKPOINTING}" == "true" ]]; then
  CKPT_FLAG=(--activation-checkpointing)
fi
DS_FLAG=()
if [[ -n "${DEEPSPEED_CONFIG}" ]]; then
  DS_FLAG=(--deepspeed-config "${DEEPSPEED_CONFIG}")
fi
GRAPH_FLAG=()
if [[ "${CUDA_GRAPH}" == "1" || "${CUDA_GRAPH}" == "true" ]]; then
  GRAPH_FLAG=(--cuda-graph)
fi

python -m mamba3_kr.cli train-packed \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --steps "${BASE_STEPS}" \
  --lr "${BASE_LR}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --optimizer "${OPTIMIZER}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --data data/splits/base_train.jsonl \
  --checkpoint "${BASE_CKPT}" \
  --save-every 50 \
  --no-resume \
  "${DS_FLAG[@]}" \
  "${CKPT_FLAG[@]}"

echo "== base validation loss =="
python -m mamba3_kr.cli eval-loss \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --data data/splits/base_valid.jsonl \
  --checkpoint "${BASE_CKPT}" \
  --batches 8

if [[ "${CURRICULUM_STEPS}" != "0" ]]; then
  echo "== separated recall/copy curriculum from base checkpoint =="
  cp "${BASE_CKPT}" "${CURRICULUM_CKPT}"
  python -m mamba3_kr.cli train-packed \
    --mode "${MODE}" \
    --tokenizer "${TOKENIZER}" \
    --seq-len "${SEQ_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --steps "${CURRICULUM_STEPS}" \
    --lr "${CURRICULUM_LR}" \
    --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
    --optimizer "${OPTIMIZER}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --data data/splits/curriculum_train.txt \
    --checkpoint "${CURRICULUM_CKPT}" \
    --save-every 50 \
    --no-resume \
    "${DS_FLAG[@]}" \
    "${CKPT_FLAG[@]}"

  echo "== curriculum validation loss =="
  python -m mamba3_kr.cli eval-loss \
    --mode "${MODE}" \
    --tokenizer "${TOKENIZER}" \
    --seq-len "${SEQ_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --data data/splits/curriculum_valid.txt \
    --checkpoint "${CURRICULUM_CKPT}" \
    --batches 8

  echo "== curriculum exact-match probe =="
  python scripts/mamba3_eval_programmatic.py \
    --mode "${MODE}" \
    --tokenizer "${TOKENIZER}" \
    --checkpoint "${CURRICULUM_CKPT}" \
    --data "${PROGRAMMATIC_DATA}" \
    --limit 32 \
    --seq-len "${SEQ_LEN}" \
    --max-new 16 \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --top-k 1 \
    --top-p 0 \
    --temperature 1.0 \
    --repetition-penalty 1.0 || true
fi

echo "== answer-only SFT from latest checkpoint =="
if [[ "${CURRICULUM_STEPS}" != "0" ]]; then
  cp "${CURRICULUM_CKPT}" "${SFT_CKPT}"
else
  cp "${BASE_CKPT}" "${SFT_CKPT}"
fi
python -m mamba3_kr.cli train-packed \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --steps "${SFT_STEPS}" \
  --lr "${SFT_LR}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --optimizer "${OPTIMIZER}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --data data/splits/sft_train.txt \
  --checkpoint "${SFT_CKPT}" \
  --save-every 50 \
  --no-resume \
  "${DS_FLAG[@]}" \
  "${CKPT_FLAG[@]}"

echo "== SFT validation loss =="
python -m mamba3_kr.cli eval-loss \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --data data/splits/sft_valid.txt \
  --checkpoint "${SFT_CKPT}" \
  --batches 8

if [[ "${CURRICULUM_STEPS}" != "0" ]]; then
  echo "== post-SFT exact-match probe =="
  python scripts/mamba3_eval_programmatic.py \
    --mode "${MODE}" \
    --tokenizer "${TOKENIZER}" \
    --checkpoint "${SFT_CKPT}" \
    --data "${PROGRAMMATIC_DATA}" \
    --limit 32 \
    --seq-len "${SEQ_LEN}" \
    --max-new 16 \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --top-k 1 \
    --top-p 0 \
    --temperature 1.0 \
    --repetition-penalty 1.0 || true
fi

echo "== SFT English I/O gate =="
python -m mamba3_kr.cli eval-english \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${SFT_CKPT}" \
  --max-new 32 \
  --seq-len "${SEQ_LEN}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --top-k 1 \
  --top-p 0 \
  --temperature 1.0 \
  --repetition-penalty 1.0 \
  "${GRAPH_FLAG[@]}"

echo "== SFT quality promotion gate =="
python -m mamba3_kr.cli quality-gate \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${SFT_CKPT}" \
  --max-new 32 \
  --seq-len "${SEQ_LEN}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --top-k 1 \
  --top-p 0 \
  --temperature 1.0 \
  --repetition-penalty 1.0 \
  "${GRAPH_FLAG[@]}"

echo "== SFT decode speed gate =="
python -m mamba3_kr.cli bench-decode \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${SFT_CKPT}" \
  --prompt "The main idea is" \
  --max-new 64 \
  --seq-len "${SEQ_LEN}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --top-k 1 \
  --top-p 0 \
  --temperature 1.0 \
  --repetition-penalty 1.0 \
  "${GRAPH_FLAG[@]}" \
  --repeats 3
