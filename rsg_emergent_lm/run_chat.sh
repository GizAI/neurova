#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKENS_FILE="${RSG_TOKENIZER:-}"
PAIRS_FILE="${RSG_PAIRS:-${BASE_DIR}/seed_pairs.jsonl}"
CACHE_DIR="${RSG_CACHE_DIR:-${HOME}/.cache/neurova_rsg}"
USER_ARGS=("$@")
HAS_TOKENIZER=false
HAS_PAIRS=false
USER_CHAT_ARGS=()

for ((i=0; i<${#USER_ARGS[@]}; i++)); do
  case "${USER_ARGS[i]}" in
    --tokenizer)
      HAS_TOKENIZER=true
      TOKENS_FILE="${USER_ARGS[i+1]}"
      ((i+=1))
      ;;
    --tokenizer=*)
      HAS_TOKENIZER=true
      TOKENS_FILE="${USER_ARGS[i]#*=}"
      ;;
    --pairs)
      HAS_PAIRS=true
      PAIRS_FILE="${USER_ARGS[i+1]}"
      ((i+=1))
      ;;
    --pairs=*)
      HAS_PAIRS=true
      PAIRS_FILE="${USER_ARGS[i]#*=}"
      ;;
    *)
      USER_CHAT_ARGS+=("${USER_ARGS[i]}")
      ;;
  esac
done

if [ -z "$TOKENS_FILE" ]; then
  if [ -f "/mnt/data/tokenizer.json" ]; then
    TOKENS_FILE="/mnt/data/tokenizer.json"
  elif [ -f "${BASE_DIR}/tokenizer.json" ]; then
    TOKENS_FILE="${BASE_DIR}/tokenizer.json"
  elif [ -f "${BASE_DIR}/../luma/tokenizers/qwen35/tokenizer.json" ]; then
    TOKENS_FILE="${BASE_DIR}/../luma/tokenizers/qwen35/tokenizer.json"
  elif [ -f "/home/user/workspace/neurova/luma/tokenizers/qwen35/tokenizer.json" ]; then
    TOKENS_FILE="/home/user/workspace/neurova/luma/tokenizers/qwen35/tokenizer.json"
  else
    echo "tokenizer 파일을 찾지 못했습니다. --tokenizer 또는 RSG_TOKENIZER로 지정하세요." >&2
    exit 1
  fi
fi

if [ ! -f "$PAIRS_FILE" ]; then
  echo "seed pairs 파일이 없습니다: $PAIRS_FILE" >&2
  exit 1
fi

echo "RSG chat 시작"
echo "  tokenizer: $TOKENS_FILE"
echo "  pairs: $PAIRS_FILE"
echo

CMD_ARGS=()
if [ -n "$TOKENS_FILE" ]; then
  CMD_ARGS+=(--tokenizer "$TOKENS_FILE")
fi

cd "$BASE_DIR"
python3 "${BASE_DIR}/emergent_rsg_lm.py" \
  "${CMD_ARGS[@]}" \
  chat \
  --pairs "$PAIRS_FILE" \
  --cache-dir "$CACHE_DIR" \
  "${USER_CHAT_ARGS[@]}"
