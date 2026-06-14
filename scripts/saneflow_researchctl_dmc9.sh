#!/usr/bin/env bash
set -euo pipefail

cd "${ROOT:-$HOME/workspace/neurova}"
PYTHON="${SANEFLOW_PYTHON:-/home/user/miniconda3/envs/saneflow/bin/python}"

cmd="${1:-status}"
case "$cmd" in
  status)
    "$PYTHON" scripts/saneflow_run.py status \
      dmc9-dense-0.3b-v1 \
      dmc9-dense-deepthin-0.3b-v1 \
      dmc9-dense-0.3b-v2-en-cont \
      dmc9-dense-deepthin-0.3b-v2-en-cont
    echo
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader
    ;;
  start-auto|auto)
    if pgrep -af "bash scripts/saneflow_autoresearch_loop.sh dmc9" | grep -v pgrep >/dev/null; then
      echo "saneflow autoresearch dmc9 already running"
    else
      mkdir -p runs/saneflow_autoresearch
      [[ -s runs/saneflow_autoresearch/dmc9.out ]] && mv runs/saneflow_autoresearch/dmc9.out "runs/saneflow_autoresearch/dmc9.$(date +%Y%m%d_%H%M%S).out" || true
      nohup bash scripts/saneflow_autoresearch_loop.sh dmc9 \
        > runs/saneflow_autoresearch/dmc9.out 2>&1 &
      echo "started saneflow autoresearch dmc9 pid=$!"
    fi
    ;;
  restart-auto)
    pkill -f "bash scripts/saneflow_autoresearch_loop.sh dmc9" || true
    mkdir -p runs/saneflow_autoresearch
    [[ -s runs/saneflow_autoresearch/dmc9.out ]] && mv runs/saneflow_autoresearch/dmc9.out "runs/saneflow_autoresearch/dmc9.$(date +%Y%m%d_%H%M%S).out" || true
    nohup bash scripts/saneflow_autoresearch_loop.sh dmc9 \
      > runs/saneflow_autoresearch/dmc9.out 2>&1 &
    echo "restarted saneflow autoresearch dmc9 pid=$!"
    ;;
  eval-latest)
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" "$PYTHON" scripts/saneflow_reasoning_gate.py \
      --ckpt runs/neurova_dense_0_3b_v1/latest.pt \
      --out runs/neurova_dense_0_3b_v1/reasoning_gate_latest.json \
      --device cuda --dtype bf16 --max-new "${MAX_NEW:-48}"
    ;;
  prepare-v2-en)
    mkdir -p runs/data_prep
    if [[ -f runs/data_prep/pretrain_v2_en.pid ]] && kill -0 "$(cat runs/data_prep/pretrain_v2_en.pid)" 2>/dev/null; then
      echo "pretrain v2 en prep already running pid=$(cat runs/data_prep/pretrain_v2_en.pid)"
    else
      nohup "$PYTHON" scripts/saneflow_prepare_pretrain_v2.py \
        --recipe configs/saneflow_pretrain_sources_v2.json \
        > runs/data_prep/pretrain_v2_en.out 2>&1 &
      echo "$!" > runs/data_prep/pretrain_v2_en.pid
      echo "started pretrain v2 en prep pid=$!"
    fi
    tail -40 runs/data_prep/pretrain_v2_en.out 2>/dev/null || true
    ;;
  doremi-v2-en)
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" "$PYTHON" scripts/saneflow_doremi_pipeline.py \
      --recipe configs/saneflow_practical_pretrain_mix_v2_en.json \
      --out runs/doremi_proxy_practical_v2_en \
      --reference-out runs/doremi_reference_practical_v2_en \
      --steps "${DOREMI_PROXY_STEPS:-200}" \
      --reference-steps "${DOREMI_REFERENCE_STEPS:-300}" \
      --seq-len "${DOREMI_PROXY_SEQ_LEN:-512}" \
      --batch-size "${DOREMI_PROXY_BATCH_SIZE:-1}" \
      --tokenizer-path tokenizers/neurova_spm_unigram_64k \
      --tf32 \
      --activation-checkpointing
    ;;
  switch-v2-en)
    if [[ ! -s data/corpus/mixes/saneflow_practical_pretrain_v2_en/train.jsonl || ! -s data/corpus/mixes/saneflow_practical_pretrain_v2_en/valid.jsonl ]]; then
      echo "v2 en mix is not ready; run prepare-v2-en first" >&2
      exit 1
    fi
    pkill -f "scripts/saneflow_train.py --out runs/neurova_dense_0_3b_v1" || true
    pkill -f "scripts/saneflow_train.py --out runs/neurova_dense_deepthin_0_3b_v1" || true
    if [[ ! -f data/corpus/mixes/saneflow_practical_pretrain_v2_en/doremi_ratios.json ]]; then
      CUDA_VISIBLE_DEVICES="${DOREMI_CUDA_VISIBLE_DEVICES:-1}" "$PYTHON" scripts/saneflow_doremi_pipeline.py \
        --recipe configs/saneflow_practical_pretrain_mix_v2_en.json \
        --out runs/doremi_proxy_practical_v2_en \
        --reference-out runs/doremi_reference_practical_v2_en \
        --steps "${DOREMI_PROXY_STEPS:-200}" \
        --reference-steps "${DOREMI_REFERENCE_STEPS:-300}" \
        --seq-len "${DOREMI_PROXY_SEQ_LEN:-512}" \
        --batch-size "${DOREMI_PROXY_BATCH_SIZE:-1}" \
        --tokenizer-path tokenizers/neurova_spm_unigram_64k \
        --tf32 \
        --activation-checkpointing
    fi
    "$PYTHON" scripts/saneflow_run.py start dmc9-dense-0.3b-v2-en-cont
    "$PYTHON" scripts/saneflow_run.py start dmc9-dense-deepthin-0.3b-v2-en-cont
    ;;
  *)
    echo "usage: $0 {status|start-auto|restart-auto|eval-latest|prepare-v2-en|doremi-v2-en|switch-v2-en}" >&2
    exit 2
    ;;
esac
