#!/usr/bin/env bash
set -euo pipefail

MODE_TINY="${MODE_TINY:-mimo-r4-tiny}"
MODE_TARGET="${MODE_TARGET:-mimo-r4-1.5b}"
TOKENIZER="${TOKENIZER:-llama31}"
CHECKPOINT="${CHECKPOINT:-runs/mamba3_kr_tiny/model_mimo_r4_speak.pt}"
SEQ_LEN="${SEQ_LEN:-128}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"

echo "== contract: target ${MODE_TARGET} =="
python -m mamba3_kr.cli check-contract \
  --mode "${MODE_TARGET}" \
  --tokenizer "${TOKENIZER}" \
  --device "${DEVICE}"

echo "== model-info: target ${MODE_TARGET} =="
python -m mamba3_kr.cli model-info \
  --mode "${MODE_TARGET}" \
  --tokenizer "${TOKENIZER}" \
  --device "${DEVICE}" \
  | grep -E 'effective_mlp_hidden_dim|d_intermediate|d_state|headdim|mimo_rank|mlp_multiple_of|estimated_parameters'

echo "== kernel probe: trainable tiny =="
python -m mamba3_kr.cli probe-kernel \
  --mode "${MODE_TINY}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size 1 \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --data data/english_completion_bootstrap.txt data/english_instruction_bootstrap.txt

echo "== decode benchmark: trainable tiny =="
python -m mamba3_kr.cli bench-decode \
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
python -m mamba3_kr.cli decode-parity \
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
python -m mamba3_kr.cli state-roundtrip \
  --mode "${MODE_TINY}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --text "Stateful Mamba-3 memory should continue from a saved recurrent state without replaying the whole document." \
  --state-in runs/mamba3_kr_tiny/gate_state.pt \
  --seq-len "${SEQ_LEN}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}"

echo "== English I/O smoke =="
python -m mamba3_kr.cli eval-english \
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
