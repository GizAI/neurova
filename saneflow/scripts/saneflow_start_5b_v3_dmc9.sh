#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${SANEFLOW_PYTHON:-/home/user/miniconda3/envs/saneflow/bin/python}"
LOG_DIR="$ROOT/saneflow/runs/orchestration"
LOG="$LOG_DIR/saneflow_5b_v3_dmc9.out"
PREP_RECIPE="saneflow/configs/saneflow_pretrain_sources_v3_5b_en.json"
MAIN_PROFILE="dmc9-dense-wide-0.3b-5b-v3-en"
OLD_03_OUT="saneflow/runs/neurova_dense_wide_0_3b_v2_en_original"
V3_TRAIN="$ROOT/saneflow/data/corpus/mixes/saneflow_practical_pretrain_v3_5b_en/train.jsonl"
V3_VALID="$ROOT/saneflow/data/corpus/mixes/saneflow_practical_pretrain_v3_5b_en/valid.jsonl"

mkdir -p "$LOG_DIR"
exec >>"$LOG" 2>&1

echo "=== $(date -Is) start 5B v3 orchestration ==="
cd "$ROOT"

echo "prepare v3 data: $PREP_RECIPE"
"$PY" saneflow/scripts/saneflow_prepare_pretrain_v2.py --recipe "$PREP_RECIPE"

test -s "$V3_TRAIN"
test -s "$V3_VALID"
echo "v3 train bytes=$(stat -c%s "$V3_TRAIN") valid bytes=$(stat -c%s "$V3_VALID")"

echo "stop old 0.3B v2 run only: $OLD_03_OUT"
mapfile -t old_pids < <(pgrep -f "saneflow/scripts/saneflow_train.py --out ${OLD_03_OUT}" || true)
for pid in "${old_pids[@]}"; do
  if [[ -n "$pid" ]]; then
    echo "kill old 0.3B pid=$pid"
    kill "$pid" || true
  fi
done
sleep 3
for pid in "${old_pids[@]}"; do
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "kill -9 old 0.3B pid=$pid"
    kill -9 "$pid" || true
  fi
done

echo "start $MAIN_PROFILE"
"$PY" saneflow/scripts/saneflow_run.py start "$MAIN_PROFILE"

echo "=== $(date -Is) done 5B v3 orchestration ==="
