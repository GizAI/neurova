#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"

echo "== processes =="
pgrep -af "luma.train|luma_research_pipeline_dmc8" || true

echo
echo "== gpu =="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader || true

echo
echo "== runs =="
for run in \
  luma/runs/luma_stage1_qwen_natural_pre_v1 \
  luma/runs/luma_stage2_qwen_chat_sft_v1 \
  luma/runs/luma_stage3_qwen_slotproof_v1
do
  echo "-- ${run}"
  if [[ -f "${run}/train_log.jsonl" ]]; then
    tail -n 1 "${run}/train_log.jsonl"
  else
    echo "no train_log.jsonl"
  fi
  ls -lh "${run}/latest.pt" "${run}/model.pt" "${run}/natural_sanity.json" "${run}/chat_sanity.json" "${run}/memory_ablation_eval.json" "${run}/gate_summary.json" 2>/dev/null || true
done

echo
echo "== pipeline =="
tail -n 10 luma/runs/luma_research_pipeline_dmc8.log 2>/dev/null || true
