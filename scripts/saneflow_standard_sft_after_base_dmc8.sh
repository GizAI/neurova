#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BASE_RUN="runs/saneflow_fineweb_edu_base_v3_100m_muon_mem"
BASE_CKPT="$BASE_RUN/model.pt"
SPEAK_RUN="runs/saneflow_speak_base_v1_100m"
SFT_RUN="runs/saneflow_chatml_sft_v9_assistant"
CURRENT_RUN="runs/saneflow_current"

while [[ ! -f "$BASE_CKPT" ]]; do
  echo "{\"waiting_for_base\":\"$BASE_CKPT\"}" >&2
  sleep 60
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python scripts/saneflow_run.py wait-train dmc8-speak-base-v1 --build-speak

python scripts/saneflow_run.py wait-train dmc8-chatml-sft-v9 --build-chatml

mkdir -p "$CURRENT_RUN"
ln -sfn "$(pwd)/$SFT_RUN/model.pt" "$CURRENT_RUN/model.pt"
python scripts/saneflow_generate.py \
  --ckpt "$SFT_RUN/model.pt" \
  --prompt "Who are you?" \
  --chatml --max-new 80 --context 384 \
  --temperature 0.65 --top-k 40 --top-p 0.9 --repetition-penalty 1.08 --no-repeat-ngram-size 4 \
  --decode cache --device cuda --dtype bf16 \
  > "$SFT_RUN/final_smoke_who_are_you.txt"
