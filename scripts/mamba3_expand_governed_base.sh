#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MAX_DOCS="${MAX_DOCS:-10000}"
MAX_BYTES="${MAX_BYTES:-250000000}"
MIN_CHARS="${MIN_CHARS:-300}"
CLEAN_RECORDS="${CLEAN_RECORDS:-4000}"
TECH_RECORDS="${TECH_RECORDS:-1200}"
TRAIN_OUT="${TRAIN_OUT:-data/splits/base_expanded_train.txt}"
VALID_OUT="${VALID_OUT:-data/splits/base_expanded_valid.txt}"
VALID_RATIO="${VALID_RATIO:-0.02}"
SEED="${SEED:-4242}"
STRATIFIED_SPLIT="${STRATIFIED_SPLIT:-1}"
DOC_CONTINUATION="${DOC_CONTINUATION:-1}"
DOC_CORPUS_OUT="${DOC_CORPUS_OUT:-data/base_doc_continuation_v1.jsonl}"

cd "${ROOT}"

echo "== expand governed corpus =="
MAX_DOCS="${MAX_DOCS}" \
MAX_BYTES="${MAX_BYTES}" \
MIN_CHARS="${MIN_CHARS}" \
  scripts/mamba3_prepare_corpora.sh

echo "== deterministic clean English supplement =="
python scripts/mamba3_generate_clean_english_sft.py \
  --out data/clean_english_sft_expanded_v1.jsonl \
  --records "${CLEAN_RECORDS}" \
  --seed 20260614

SPLIT_INPUTS=(
  data/governed_fineweb_edu_sample.jsonl
  data/governed_dclm_sample.jsonl
  data/governed_open_web_math_sample.jsonl
  data/governed_arxiv_abstracts_sample.jsonl
)

if (( TECH_RECORDS > 0 )); then
  echo "== deterministic technical supplement =="
  python scripts/mamba3_generate_technical_bootstrap.py \
    --out data/technical_bootstrap_v1.jsonl \
    --records "${TECH_RECORDS}" \
    --seed 20260614
  scripts/mamba3_validate_governance.py data/technical_bootstrap_v1.jsonl
  SPLIT_INPUTS+=(data/technical_bootstrap_v1.jsonl)
else
  echo "== deterministic technical supplement skipped =="
  rm -f data/technical_bootstrap_v1.jsonl
fi

if [[ "${DOC_CONTINUATION}" != "0" && "${DOC_CONTINUATION}" != "false" ]]; then
  echo "== build document-continuation corpus =="
  python scripts/mamba3_build_doc_continuation_corpus.py \
    --inputs "${SPLIT_INPUTS[@]}" \
    --out "${DOC_CORPUS_OUT}" \
    --min-chars "${MIN_CHARS}"
  SPLIT_INPUTS=("${DOC_CORPUS_OUT}")
else
  SPLIT_INPUTS+=(data/clean_english_sft_expanded_v1.jsonl)
fi

echo "== build expanded base split =="
split_script="scripts/mamba3_make_source_stratified_splits.py"
if [[ "${STRATIFIED_SPLIT}" == "0" || "${STRATIFIED_SPLIT}" == "false" ]]; then
  split_script="scripts/mamba3_make_splits.py"
fi
python "${split_script}" \
  --inputs "${SPLIT_INPUTS[@]}" \
  --train-out "${TRAIN_OUT}" \
  --valid-out "${VALID_OUT}" \
  --valid-ratio "${VALID_RATIO}" \
  --seed "${SEED}"

scripts/mamba3_corpus_manifest.py data/*.txt data/*.jsonl --out data/mamba3_corpus_manifest.json
