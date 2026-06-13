#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-runs/mamba3_stability_ladder}"
TOKENIZER="${TOKENIZER:-llama31}"
SEQ_LEN="${SEQ_LEN:-128}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
PROGRAMMATIC_DATA="${PROGRAMMATIC_DATA:-data/mamba3_programmatic_curriculum.jsonl}"
OUT="${OUT:-${RUN_ROOT}/summary.jsonl}"
CUDA_GRAPH="${CUDA_GRAPH:-1}"

cd "${ROOT}"
mkdir -p "${RUN_ROOT}"
: > "${OUT}"

if [[ -n "${MODES_CSV:-}" ]]; then
  IFS=',' read -r -a MODES <<< "${MODES_CSV}"
else
  MODES=(
    "siso"
    "mimo-r2"
    "mimo-r2-attn-tiny"
    "mamba3-recall-r2-tiny"
    "mimo-r4-tiny"
    "mimo-r4-attn-tiny"
    "mamba3-recall-r4-tiny"
  )
fi

GRAPH_FLAG=()
if [[ "${CUDA_GRAPH}" == "1" || "${CUDA_GRAPH}" == "true" ]]; then
  GRAPH_FLAG=(--cuda-graph)
fi

for mode in "${MODES[@]}"; do
  ckpt="${RUN_ROOT}/${mode}/sft.pt"
  if [[ ! -f "${ckpt}" ]]; then
    printf '{"mode":"%s","checkpoint":"%s","exists":false}\n' "${mode}" "${ckpt}" | tee -a "${OUT}"
    continue
  fi
  echo "== collect ${mode} =="
  python scripts/mamba3_eval_programmatic.py \
    --mode "${mode}" \
    --tokenizer "${TOKENIZER}" \
    --checkpoint "${ckpt}" \
    --data "${PROGRAMMATIC_DATA}" \
    --limit 32 \
    --seq-len "${SEQ_LEN}" \
    --max-new 12 \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --top-k 1 \
    --top-p 0 \
    --temperature 1.0 \
    --repetition-penalty 1.0 \
    > "${RUN_ROOT}/${mode}/programmatic_eval.json" || true

  python -m mamba3_kr.cli quality-gate \
    --mode "${mode}" \
    --tokenizer "${TOKENIZER}" \
    --checkpoint "${ckpt}" \
    --max-new 32 \
    --seq-len "${SEQ_LEN}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --top-k 1 \
    --top-p 0 \
    --temperature 1.0 \
    --repetition-penalty 1.0 \
    "${GRAPH_FLAG[@]}" \
    > "${RUN_ROOT}/${mode}/quality_gate.json" || true

  python - <<PY | tee -a "${OUT}"
import json
from pathlib import Path
mode = ${mode@Q}
root = Path(${RUN_ROOT@Q}) / mode
payload = {"mode": mode, "checkpoint": str(root / "sft.pt"), "exists": True}
for name in ["programmatic_eval", "quality_gate"]:
    path = root / f"{name}.json"
    if path.exists():
        try:
            payload[name] = json.loads(path.read_text())
        except Exception as exc:
            payload[name] = {"ok": False, "error": str(exc)}
print(json.dumps(payload, ensure_ascii=False))
PY
done

echo "summary=${OUT}"
