#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODE="${MODE:-mimo-r4-tiny}"
TOKENIZER="${TOKENIZER:-llama31}"
CHECKPOINT="${CHECKPOINT:-neuromamba/runs/mamba3_tiny/model_mimo_r4_speak.pt}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
OUT="${OUT:-neuromamba/runs/mamba3_context_limits.jsonl}"
SEQ_LENS_CSV="${SEQ_LENS_CSV:-512,1024,2048,4096,8192,16384}"
MAX_NEW="${MAX_NEW:-64}"

cd "${ROOT}"
mkdir -p "$(dirname "${OUT}")"
: > "${OUT}"

IFS=',' read -r -a SEQ_LENS <<< "${SEQ_LENS_CSV}"

for seq_len in "${SEQ_LENS[@]}"; do
  echo "== context probe seq_len=${seq_len} =="
  if python -m neuromamba.cli smoke \
      --mode "${MODE}" \
      --tokenizer "${TOKENIZER}" \
      --checkpoint "${CHECKPOINT}" \
      --seq-len "${seq_len}" \
      --batch-size 1 \
      --device "${DEVICE}" \
      --dtype "${DTYPE}" \
      | tee -a "${OUT}"; then
    :
  else
    printf '{"mode":"%s","seq_len":%s,"ok":false,"stage":"smoke"}\n' "${MODE}" "${seq_len}" | tee -a "${OUT}"
    break
  fi
done

echo "== long decode smoke =="
python -m neuromamba.cli bench-decode \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --prompt "Write a concise technical answer about why long context matters." \
  --max-new "${MAX_NEW}" \
  --seq-len "${SEQ_LENS[${#SEQ_LENS[@]}-1]}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --top-k 1 \
  --top-p 0 \
  --temperature 1.0 \
  --repetition-penalty 1.0 \
  --repeats 1 \
  | tee -a "${OUT}" || true

echo "context_probe=${OUT}"
