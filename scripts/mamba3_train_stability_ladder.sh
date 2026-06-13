#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TOKENIZER="${TOKENIZER:-llama31}"
SEQ_LEN="${SEQ_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
BASE_STEPS="${BASE_STEPS:-20}"
SFT_STEPS="${SFT_STEPS:-20}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
OPTIMIZER="${OPTIMIZER:-adamw8bit}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
RUN_ROOT="${RUN_ROOT:-runs/mamba3_stability_ladder}"
PROGRAMMATIC_DATA="${PROGRAMMATIC_DATA:-data/mamba3_programmatic_curriculum_ladder.jsonl}"
PROGRAMMATIC_RECORDS="${PROGRAMMATIC_RECORDS:-1200}"
SFT_PROGRAMMATIC_MAX_RECORDS="${SFT_PROGRAMMATIC_MAX_RECORDS:-0}"
SFT_NATURAL_MAX_RECORDS="${SFT_NATURAL_MAX_RECORDS:-0}"
SFT_NATURAL_FORMAT="${SFT_NATURAL_FORMAT:-answer}"
SFT_PROGRAMMATIC_FORMAT="${SFT_PROGRAMMATIC_FORMAT:-qa}"
CUDA_GRAPH="${CUDA_GRAPH:-1}"

cd "${ROOT}"
mkdir -p "${RUN_ROOT}"

if [[ -n "${MODES_CSV:-}" ]]; then
  IFS=',' read -r -a MODES <<< "${MODES_CSV}"
else
  MODES=(
    "mamba3-siso-hybrid-95m"
    "mamba3-siso-hybrid-0.3b"
    "siso"
    "mimo-r2-fast-tiny"
    "mimo-r4-fast-tiny"
  )
fi

python scripts/mamba3_generate_programmatic_curriculum.py \
  --out "${PROGRAMMATIC_DATA}" \
  --records "${PROGRAMMATIC_RECORDS}" \
  --seed 20260613

for mode in "${MODES[@]}"; do
  run_dir="${RUN_ROOT}/${mode}"
  echo "== ladder stage ${mode} =="
  MODE="${mode}" \
  TOKENIZER="${TOKENIZER}" \
  SEQ_LEN="${SEQ_LEN}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  BASE_STEPS="${BASE_STEPS}" \
  SFT_STEPS="${SFT_STEPS}" \
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS}" \
  OPTIMIZER="${OPTIMIZER}" \
  DEVICE="${DEVICE}" \
  DTYPE="${DTYPE}" \
  RUN_DIR="${run_dir}" \
  PROGRAMMATIC_DATA="${PROGRAMMATIC_DATA}" \
  SFT_PROGRAMMATIC_MAX_RECORDS="${SFT_PROGRAMMATIC_MAX_RECORDS}" \
  SFT_NATURAL_MAX_RECORDS="${SFT_NATURAL_MAX_RECORDS}" \
  SFT_NATURAL_FORMAT="${SFT_NATURAL_FORMAT}" \
  SFT_PROGRAMMATIC_FORMAT="${SFT_PROGRAMMATIC_FORMAT}" \
  CUDA_GRAPH="${CUDA_GRAPH}" \
    scripts/mamba3_train_scientific_tiny.sh || {
      echo "stage_failed=${mode}"
      continue
    }
done

echo "ladder_done=${RUN_ROOT}"
