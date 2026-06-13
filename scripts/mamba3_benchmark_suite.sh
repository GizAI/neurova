#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"

MODE="${MODE:-mamba3-siso-fast-0.3b-ds128}"
TOKENIZER="${TOKENIZER:-llama31}"
CHECKPOINT="${CHECKPOINT:-runs/mamba3_siso_fast_0_3b_ds128_v3/model.pt}"
SEQ_LEN="${SEQ_LEN:-128}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
OUT_DIR="${OUT_DIR:-runs/mamba3_benchmarks/$(date -u +%Y%m%dT%H%M%SZ)}"
MMLU_LIMIT="${MMLU_LIMIT:-100}"
MMLU_REDUX_LIMIT="${MMLU_REDUX_LIMIT:-100}"
MMLU_SUBJECT="${MMLU_SUBJECT:-all}"
MMLU_REDUX_FILTER="${MMLU_REDUX_FILTER:-ok}"

mkdir -p "${OUT_DIR}"

echo "== benchmark suite =="
echo "mode=${MODE}"
echo "checkpoint=${CHECKPOINT}"
echo "out_dir=${OUT_DIR}"

python scripts/mamba3_eval_mcq_bench.py \
  --suite smoke \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --seq-len "${SEQ_LEN}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --out "${OUT_DIR}/mcq_smoke.json" \
  | tee "${OUT_DIR}/mcq_smoke.stdout"

python scripts/mamba3_eval_mcq_bench.py \
  --suite mmlu \
  --mmlu-subject "${MMLU_SUBJECT}" \
  --limit "${MMLU_LIMIT}" \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --seq-len "${SEQ_LEN}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --out "${OUT_DIR}/mmlu_sample.json" \
  | tee "${OUT_DIR}/mmlu_sample.stdout" || true

python scripts/mamba3_eval_mcq_bench.py \
  --suite mmlu_redux \
  --mmlu-subject "${MMLU_SUBJECT}" \
  --redux-filter "${MMLU_REDUX_FILTER}" \
  --limit "${MMLU_REDUX_LIMIT}" \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --seq-len "${SEQ_LEN}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --out "${OUT_DIR}/mmlu_redux_sample.json" \
  | tee "${OUT_DIR}/mmlu_redux_sample.stdout" || true

if [[ -f /tmp/neurova_state_all_heldout.jsonl ]]; then
  python scripts/mamba3_eval_programmatic.py \
    --mode "${MODE}" \
    --tokenizer "${TOKENIZER}" \
    --checkpoint "${CHECKPOINT}" \
    --data /tmp/neurova_state_all_heldout.jsonl \
    --seq-len "${SEQ_LEN}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --max-new 32 \
    | tee "${OUT_DIR}/programmatic_heldout.stdout" || true
fi

python - "${OUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
summary = {"out_dir": str(out), "metrics": {}}
for name in ("mcq_smoke", "mmlu_sample", "mmlu_redux_sample", "programmatic_heldout"):
    path = out / f"{name}.json"
    if not path.exists():
        stdout_path = out / f"{name}.stdout"
        if stdout_path.exists():
            raw = stdout_path.read_text(encoding="utf-8", errors="replace")
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end >= start:
                path.write_text(raw[start:end + 1], encoding="utf-8")
    if not path.exists():
        continue
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        summary["metrics"][name] = {"error": repr(exc)}
        continue
    summary["metrics"][name] = {
        "accuracy": data.get("accuracy"),
        "choice_accuracy": data.get("choice_accuracy"),
        "letter_accuracy": data.get("letter_accuracy"),
        "correct": data.get("correct"),
        "total": data.get("total"),
    }
summary_path = out / "summary.json"
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
latest = out.parent / "latest_summary.json"
latest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
