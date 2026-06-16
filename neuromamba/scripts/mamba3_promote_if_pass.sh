#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MODE="${MODE:-mimo-r4-tiny}"
TOKENIZER="${TOKENIZER:-llama31}"
CHECKPOINT="${CHECKPOINT:-neuromamba/runs/mamba3_tiny/model_mimo_r4_speak.pt}"
SEQ_LEN="${SEQ_LEN:-128}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
TARGET_DIR="${TARGET_DIR:-neuromamba/runs/mamba3_current}"
PROMOTE_LOG="${PROMOTE_LOG:-${TARGET_DIR}/promotion.log}"

cd "${ROOT}"
mkdir -p "${TARGET_DIR}"
exec > >(tee "${PROMOTE_LOG}") 2>&1

echo "== promote candidate =="
python - <<'PY'
import json, os, platform, subprocess, time
payload = {
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "hostname": platform.node(),
    "mode": os.environ.get("MODE", "mimo-r4-tiny"),
    "tokenizer": os.environ.get("TOKENIZER", "llama31"),
    "checkpoint": os.environ.get("CHECKPOINT", "neuromamba/runs/mamba3_tiny/model_mimo_r4_speak.pt"),
}
try:
    payload["git_head"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
except Exception:
    payload["git_head"] = "unavailable"
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

python -m neuromamba.cli check-contract \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --device "${DEVICE}" \
  || exit 1

if [[ "${MODE}" == "mimo-r4-tiny" ]]; then
  echo "mimo-r4-tiny is not an official fast recurrent runtime candidate; use safe serving only or train an official-shape preset" >&2
  exit 1
fi

python -m neuromamba.cli probe-kernel \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size 1 \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --data luma/data/english_completion_bootstrap.txt luma/data/english_instruction_bootstrap.txt

python -m neuromamba.cli quality-gate \
  --mode "${MODE}" \
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

python -m neuromamba.cli decode-parity \
  --mode "${MODE}" \
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

python neuromamba/scripts/mamba3_recurrent_parity.py \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --prompt "The main idea is" \
  --steps 12 \
  --seq-len "${SEQ_LEN}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}"

python -m neuromamba.cli state-roundtrip \
  --mode "${MODE}" \
  --tokenizer "${TOKENIZER}" \
  --checkpoint "${CHECKPOINT}" \
  --text "Stateful Mamba-3 memory should continue from a saved recurrent state without replaying the whole document." \
  --state-in "${TARGET_DIR}/promotion_state.pt" \
  --seq-len "${SEQ_LEN}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}"

python -m neuromamba.cli bench-decode \
  --mode "${MODE}" \
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

cp "${CHECKPOINT}" "${TARGET_DIR}/model.pt"
cat > "${TARGET_DIR}/metadata.json" <<EOF
{
  "mode": "${MODE}",
  "tokenizer": "${TOKENIZER}",
  "source_checkpoint": "${CHECKPOINT}",
  "seq_len": ${SEQ_LEN},
  "dtype": "${DTYPE}",
  "promotion_log": "${PROMOTE_LOG}"
}
EOF

echo "promoted=${TARGET_DIR}/model.pt"
