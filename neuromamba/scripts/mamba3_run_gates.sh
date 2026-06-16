#!/usr/bin/env bash
set -euo pipefail

MODE_TINY="${MODE_TINY:-mimo-r4-tiny}"
MODE_TARGET="${MODE_TARGET:-mimo-r4-1.5b}"
TOKENIZER="${TOKENIZER:-llama31}"
CHECKPOINT="${CHECKPOINT:-neuromamba/runs/mamba3_tiny/model_mimo_r4_speak.pt}"
SEQ_LEN="${SEQ_LEN:-128}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"

echo "== contract: target ${MODE_TARGET} =="
python -m neuromamba.cli check-contract \
  --mode "${MODE_TARGET}" \
  --tokenizer "${TOKENIZER}" \
  --device "${DEVICE}"

echo "== model-info: target ${MODE_TARGET} =="
python -m neuromamba.cli model-info \
  --mode "${MODE_TARGET}" \
  --tokenizer "${TOKENIZER}" \
  --device "${DEVICE}" \
  | grep -E 'effective_mlp_hidden_dim|d_intermediate|d_state|headdim|mimo_rank|mlp_multiple_of|estimated_parameters'

echo "== kernel probe: trainable tiny =="
python -m neuromamba.cli probe-kernel \
  --mode "${MODE_TINY}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size 1 \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --data luma/data/english_completion_bootstrap.txt luma/data/english_instruction_bootstrap.txt

echo "== decode benchmark: trainable tiny =="
python -m neuromamba.cli bench-decode \
  --mode "${MODE_TINY}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --prompt "The main idea is" \
  --max-new 64 \
  --seq-len "${SEQ_LEN}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --top-k 1 \
  --top-p 0 \
  --temperature 1.0 \
  --repetition-penalty 1.0 \
  --cuda-graph \
  --repeats 3

echo "== decode parity: eager vs CUDA graph =="
python -m neuromamba.cli decode-parity \
  --mode "${MODE_TINY}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --prompt "The main idea is" \
  --max-new 24 \
  --seq-len "${SEQ_LEN}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --top-k 1 \
  --top-p 0 \
  --temperature 1.0 \
  --repetition-penalty 1.0 \
  --cuda-graph

echo "== recurrent state roundtrip =="
python -m neuromamba.cli state-roundtrip \
  --mode "${MODE_TINY}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --text "Stateful Mamba-3 memory should continue from a saved recurrent state without replaying the whole document." \
  --state-in neuromamba/runs/mamba3_tiny/gate_state.pt \
  --seq-len "${SEQ_LEN}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}"

echo "== English I/O smoke =="
python -m neuromamba.cli eval-english \
  --mode "${MODE_TINY}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --max-new 32 \
  --seq-len "${SEQ_LEN}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --top-k 1 \
  --top-p 0 \
  --temperature 1.0 \
  --repetition-penalty 1.0 \
  --cuda-graph
