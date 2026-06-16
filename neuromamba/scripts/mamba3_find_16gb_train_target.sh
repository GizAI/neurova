#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TOKENIZER="${TOKENIZER:-llama31}"
SEQ_LEN="${SEQ_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
OUT="${OUT:-neuromamba/runs/mamba3_16gb_target_probe.jsonl}"

cd "${ROOT}"
mkdir -p "$(dirname "${OUT}")"
: > "${OUT}"

MODES=(
  "mamba3-siso-hybrid-95m"
  "mamba3-siso-hybrid-0.3b"
  "mamba3-siso-hybrid-0.7b"
  "mamba3-siso-hybrid-1.3b"
  "mamba3-siso-hybrid-2b"
  "mimo-r2-fast-tiny"
  "mimo-r4-fast-tiny"
)

for mode in "${MODES[@]}"; do
  echo "== probe ${mode} =="
  python -m neuromamba.cli model-info \
    --mode "${mode}" \
    --tokenizer "${TOKENIZER}" \
    --device "${DEVICE}" \
    | tee -a "${OUT}" || true

  python -m neuromamba.cli probe-kernel \
    --mode "${mode}" \
    --tokenizer "${TOKENIZER}" \
    --checkpoint "neuromamba/runs/_missing_${mode}.pt" \
    --seq-len "${SEQ_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --data luma/data/english_completion_bootstrap.txt luma/data/english_instruction_bootstrap.txt \
    | tee -a "${OUT}" || true
done

echo "probe_log=${OUT}"
