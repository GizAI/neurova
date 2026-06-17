#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_DATASET="${HF_DATASET:-IkJun1/korean-qa-dataset}"
export HF_SPLIT="${HF_SPLIT:-train}"
export MAX_PAIRS="${MAX_PAIRS:-30000}"
export MAX_CHAR_PER_PAIR="${MAX_CHAR_PER_PAIR:-7000}"
export RANDOM_SEED="${RANDOM_SEED:-7}"
export PAIRS_OUT="${PAIRS_OUT:-${BASE_DIR}/seed_pairs.jsonl}"

if ! python3 - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("datasets") else 1)
PY
then
  echo "python 패키지 'datasets' 가 필요합니다. pip install datasets 로 설치하세요." >&2
  exit 1
fi

if [ -f "$PAIRS_OUT" ]; then
  BACKUP="${PAIRS_OUT}.backup.$(date +%Y%m%d-%H%M%S).jsonl"
  cp "$PAIRS_OUT" "$BACKUP"
  echo "기존 파일 백업: $BACKUP"
fi

python3 - <<'PY'
import json
import random
from pathlib import Path
from datasets import load_dataset

import os

DATASET = os.environ["HF_DATASET"]
SPLIT = os.environ["HF_SPLIT"]
MAX_PAIRS = int(os.environ["MAX_PAIRS"])
MAX_CHAR_PER_PAIR = int(os.environ["MAX_CHAR_PER_PAIR"])
RANDOM_SEED = int(os.environ["RANDOM_SEED"])
OUT = Path(os.environ["PAIRS_OUT"])

random.seed(RANDOM_SEED)
ds = load_dataset(DATASET, split=SPLIT)

rows = []
seen = set()
for row in ds:
    source = str(row.get("prompt", "")).strip()
    target = str(row.get("response", "")).strip()
    if not source or not target:
        continue
    if source.startswith("Human:"):
        source = source.split(":", 1)[1].strip()
    if target.startswith("GPT:"):
        target = target.split(":", 1)[1].strip()
    if not source or not target:
        continue
    if len(source) > MAX_CHAR_PER_PAIR or len(target) > MAX_CHAR_PER_PAIR:
        continue
    key = (source, target)
    if key in seen:
        continue
    seen.add(key)
    rows.append({"source": source, "target": target})

random.shuffle(rows)
if MAX_PAIRS > 0:
    rows = rows[:MAX_PAIRS]

OUT.write_text(
    "\n".join(json.dumps({"source": r["source"], "target": r["target"]}, ensure_ascii=False) for r in rows) + "\n",
    encoding="utf-8",
)
print(f"written={len(rows)} dataset={DATASET} split={SPLIT} out={OUT}")
PY

echo "완료: $PAIRS_OUT"
