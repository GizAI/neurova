#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RUN_DIR="${RUN_DIR:-neuromamba/runs/mamba3_clean_doc_base_moe900_v1}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/until_gate}"
SUMMARY_JSONL="${SUMMARY_JSONL:-${LOG_DIR}/summary.jsonl}"

TRAIN_DATA="${TRAIN_DATA:-neuromamba/data/splits/base_doc_cont_v3_train.jsonl}"
VALID_DATA="${VALID_DATA:-neuromamba/data/splits/base_doc_cont_v3_valid.jsonl}"
TARGET_LOSS="${TARGET_LOSS:-5.0}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-16}"
MAX_REPEATED_WORD_RUN="${MAX_REPEATED_WORD_RUN:-12}"
MIN_DISTINCT_WORDS="${MIN_DISTINCT_WORDS:-8}"
PROMPT="${PROMPT:-<doc source=\"probe\" domain=\"science\"> The main idea is}"
MAX_ROUNDS="${MAX_ROUNDS:-4}"
STEPS_PER_ROUND="${STEPS_PER_ROUND:-500}"
LR="${LR:-1.5e-5}"
SAVE_EVERY="${SAVE_EVERY:-250}"
EVAL_BATCHES="${EVAL_BATCHES:-32}"
SHUFFLE_TEXTS="${SHUFFLE_TEXTS:-1}"
DATA_SEED_BASE="${DATA_SEED_BASE:-424200}"

cd "${ROOT}"
mkdir -p "${LOG_DIR}"

echo "== clean doc until-gate =="
echo "target_loss=${TARGET_LOSS} min_new_tokens=${MIN_NEW_TOKENS} max_repeated_word_run=${MAX_REPEATED_WORD_RUN} min_distinct_words=${MIN_DISTINCT_WORDS} max_rounds=${MAX_ROUNDS} steps_per_round=${STEPS_PER_ROUND}"
echo "summary=${SUMMARY_JSONL}"
python neuromamba/scripts/mamba3_validate_doc_continuation_split.py "${TRAIN_DATA}" "${VALID_DATA}"

for round in $(seq 1 "${MAX_ROUNDS}"); do
  echo "== round ${round}/${MAX_ROUNDS}: continuation =="
  TRAIN_DATA="${TRAIN_DATA}" \
  VALID_DATA="${VALID_DATA}" \
  STEPS="${STEPS_PER_ROUND}" \
  SAVE_EVERY="${SAVE_EVERY}" \
  LR="${LR}" \
  EVAL_BATCHES="${EVAL_BATCHES}" \
  SHUFFLE_TEXTS="${SHUFFLE_TEXTS}" \
  DATA_SEED="$((DATA_SEED_BASE + round))" \
    neuromamba/scripts/mamba3_continue_clean_doc_base.sh

  python - "${RUN_DIR}" "${LOG_DIR}" "${SUMMARY_JSONL}" "${round}" "${TARGET_LOSS}" "${MIN_NEW_TOKENS}" "${MAX_REPEATED_WORD_RUN}" "${MIN_DISTINCT_WORDS}" "${PROMPT}" <<'PY'
import glob
import json
import re
import sys
import time
from pathlib import Path

run_dir, log_dir, summary_path, round_no, target_loss, min_new_tokens, max_repeated_word_run, min_distinct_words, prompt = sys.argv[1:]
target_loss = float(target_loss)
min_new_tokens = int(min_new_tokens)
max_repeated_word_run = int(max_repeated_word_run)
min_distinct_words = int(min_distinct_words)
eval_files = sorted(glob.glob(str(Path(run_dir) / "continuations" / "*_eval.json")))
decode_files = sorted(glob.glob(str(Path(run_dir) / "continuations" / "*_decode.txt")))
if not eval_files or not decode_files:
    raise SystemExit("missing continuation eval/decode artifacts")
eval_path = Path(eval_files[-1])
decode_path = Path(decode_files[-1])
eval_payload = json.loads(eval_path.read_text(encoding="utf-8"))
decode_text = decode_path.read_text(encoding="utf-8")
decode_payload = {}
json_start = decode_text.find("{")
generated_text = decode_text[:json_start].strip() if json_start >= 0 else decode_text.strip()
continuation_text = generated_text[len(prompt):].strip() if generated_text.startswith(prompt) else generated_text
for match in re.finditer(r"\{[\s\S]*?\}", decode_text):
    try:
        decode_payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        pass
loss = float(eval_payload["loss"])
new_tokens = int(decode_payload.get("new_tokens", 0))
words = re.findall(r"[A-Za-z][A-Za-z']*", continuation_text.lower())
distinct_words = len(set(words))
longest_run = 0
current_run = 0
previous = None
for word in words:
    if word == previous:
        current_run += 1
    else:
        previous = word
        current_run = 1
    longest_run = max(longest_run, current_run)
collapsed = longest_run > max_repeated_word_run or (new_tokens >= min_new_tokens and distinct_words < min_distinct_words)
passed = loss <= target_loss and new_tokens >= min_new_tokens and not collapsed
payload = {
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "round": int(round_no),
    "loss": loss,
    "new_tokens": new_tokens,
    "distinct_words": distinct_words,
    "longest_repeated_word_run": longest_run,
    "repetition_scope": "continuation_only",
    "max_repeated_word_run": max_repeated_word_run,
    "min_distinct_words": min_distinct_words,
    "collapsed": collapsed,
    "target_loss": target_loss,
    "min_new_tokens": min_new_tokens,
    "passed": passed,
    "eval_json": str(eval_path),
    "decode_log": str(decode_path),
}
Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
with Path(summary_path).open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
Path(log_dir, "latest_round.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
PY

  if python - "${LOG_DIR}/latest_round.json" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if payload.get("passed") else 1)
PY
  then
    echo "gate passed"
    exit 0
  fi
done

echo "gate not passed after ${MAX_ROUNDS} rounds"
exit 1
