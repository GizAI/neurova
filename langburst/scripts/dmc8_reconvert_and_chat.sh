#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/home/user/models/Qwen3.6-27B}"
BITS="${BITS:-4}"
QB_DIR="${QB_DIR:-/home/user/models/Qwen3.6-27B-qb${BITS}}"
ENV_NAME="${ENV_NAME:-langburst}"
WORKDIR="${WORKDIR:-/home/user/workspace/neurova/langburst}"
LOG_DIR="${LOG_DIR:-/home/user/logs}"
GROUP_SIZE="${GROUP_SIZE:-128}"
RECENT_WINDOW="${RECENT_WINDOW:-${LANGBURST_CONTEXT_WINDOW:-32768}}"
PROMPT="${PROMPT:-안녕. 너는 누구야? 한국어로 짧게 답해.}"

mkdir -p "$LOG_DIR"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
cd "$WORKDIR"
source "$WORKDIR/scripts/langburst_cuda_env.sh"
export PYTHONPATH="$WORKDIR"
export LANGBURST_QUANT_BLOCK_ROWS="${LANGBURST_QUANT_BLOCK_ROWS:-256}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

LANGBURST_REQUIRE_CUDA_EXT=1 pip install -v --no-build-isolation -e .
langburst-doctor --require-cuda

rm -rf "${QB_DIR}.tmp"
python -m langburst.quantize "$MODEL_DIR" "${QB_DIR}.tmp" --bits "$BITS" --group-size "$GROUP_SIZE"
rm -rf "$QB_DIR"
mv "${QB_DIR}.tmp" "$QB_DIR"

langburst-audit "$QB_DIR" --hf-model "$MODEL_DIR"
langburst-chat --hf-model "$MODEL_DIR" --qb-model "$QB_DIR" --device cuda --recent-window "$RECENT_WINDOW" --max-new-tokens 96 --temperature 0 --prompt "$PROMPT" --stream
