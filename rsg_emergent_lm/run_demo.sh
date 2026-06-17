#!/usr/bin/env bash
set -euo pipefail
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOK=${1:-}

if [ -z "${TOK}" ]; then
  if [ -f "/mnt/data/tokenizer.json" ]; then
    TOK="/mnt/data/tokenizer.json"
  elif [ -f "${BASE_DIR}/tokenizer.json" ]; then
    TOK="${BASE_DIR}/tokenizer.json"
  elif [ -f "${BASE_DIR}/../luma/tokenizers/qwen35/tokenizer.json" ]; then
    TOK="${BASE_DIR}/../luma/tokenizers/qwen35/tokenizer.json"
  elif [ -f "/home/user/workspace/neurova/luma/tokenizers/qwen35/tokenizer.json" ]; then
    TOK="/home/user/workspace/neurova/luma/tokenizers/qwen35/tokenizer.json"
  else
    echo "tokenizer 파일을 찾지 못했습니다. 사용법: ./run_demo.sh [tokenizer.json]" >&2
    exit 1
  fi
fi

python3 emergent_rsg_lm.py --tokenizer "$TOK" generate --pairs seed_pairs.jsonl --prompt '오늘 일정 수립해줘' --out demo_exemplar_today.json >/dev/null
python3 compose_rsg.py --tokenizer "$TOK" --pairs seed_pairs.jsonl --prompt '오늘 일정 수립해줘' --out demo_compose_today.json >/dev/null
python3 emergent_rsg_lm.py --tokenizer "$TOK" generate --pairs seed_pairs.jsonl --prompt '자기소개해라' --out demo_self_intro.json >/dev/null
python3 emergent_rsg_lm.py --tokenizer "$TOK" generate --pairs seed_pairs.jsonl --prompt '토크나이저 구조가 뭐야?' --out demo_tokenizer_question.json >/dev/null
python3 emergent_rsg_lm.py --tokenizer "$TOK" generate --pairs seed_pairs.jsonl --prompt '하드코딩 없이 언어 능력을 만들려면?' --out demo_no_hardcoding_question.json >/dev/null
python3 - <<'PY'
import json, pathlib
files=['demo_exemplar_today.json','demo_compose_today.json','demo_self_intro.json','demo_tokenizer_question.json','demo_no_hardcoding_question.json']
out={}
for f in files:
    data=json.load(open(f,encoding='utf-8'))
    out[f]=data['result']
pathlib.Path('demo_all_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
PY
