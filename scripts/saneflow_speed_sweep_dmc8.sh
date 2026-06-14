#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TRAIN_DATA="${TRAIN_DATA:-data/corpus/sources/fineweb_edu_sample10bt/train.jsonl}"
VALID_DATA="${VALID_DATA:-data/corpus/sources/fineweb_edu_sample10bt/valid.jsonl}"
TOKENIZER="${TOKENIZER:-tokenizers/saneflow_fineweb_edu_16k}"
OUT_ROOT="${OUT_ROOT:-runs/saneflow_speed_sweep}"
mkdir -p "$OUT_ROOT"

run_case() {
  local name="$1"; shift
  local out="$OUT_ROOT/$name"
  rm -rf "$out"
  mkdir -p "$out"
  echo "== $name =="
  set +e
  /usr/bin/time -f "elapsed_sec=%e" -o "$out/time.txt" \
    python scripts/saneflow_train.py \
      --out "$out" \
      --train-data "$TRAIN_DATA" \
      --valid-data "$VALID_DATA" \
      --tokenizer-path "$TOKENIZER" \
      --steps "${SWEEP_STEPS:-120}" \
      --save-every 0 \
      --log-every 20 \
      --warmup-steps 30 \
      --device cuda \
      --dtype bf16 \
      --tf32 \
      "$@"
  local rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    echo "{\"run\":\"$out\",\"failed\":true,\"returncode\":$rc}"
    return 0
  fi
  python - "$out" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
rows = [json.loads(x) for x in (out / "train_log.jsonl").read_text().splitlines() if x.startswith("{")]
elapsed = float((out / "time.txt").read_text().strip().split("=")[1])
last = rows[-1]
first = rows[0]
print(json.dumps({
    "run": str(out),
    "elapsed_sec": elapsed,
    "last_step": last.get("step"),
    "loss0": first.get("loss"),
    "loss": last.get("loss"),
    "valid_loss": last.get("valid_loss"),
}, indent=2))
PY
}

run_case muon_b64_s384_d512 \
  --optimizer muon --muon-lr 0.02 --dataset-device cuda --fused-adamw \
  --batch-size 64 --seq-len 384 --d-model 512 --layers 10 --heads 8 --d-ff 1536 --state-mixer-version v2

run_case adamw_b96_s384_d512 \
  --optimizer adamw --dataset-device cuda --fused-adamw \
  --batch-size 96 --seq-len 384 --d-model 512 --layers 10 --heads 8 --d-ff 1536 --state-mixer-version v2

run_case adamw_b64_s384_d512 \
  --optimizer adamw --dataset-device cuda --fused-adamw \
  --batch-size 64 --seq-len 384 --d-model 512 --layers 10 --heads 8 --d-ff 1536 --state-mixer-version v2

run_case ademamix_b64_s384_d512 \
  --optimizer ademamix --dataset-device cuda --fused-adamw \
  --batch-size 64 --seq-len 384 --d-model 512 --layers 10 --heads 8 --d-ff 1536 --state-mixer-version v2

run_case muon_b64_s512_d384 \
  --optimizer muon --muon-lr 0.02 --dataset-device cuda --fused-adamw \
  --batch-size 64 --seq-len 512 --d-model 384 --layers 8 --heads 6 --d-ff 1152 --state-mixer-version v2
