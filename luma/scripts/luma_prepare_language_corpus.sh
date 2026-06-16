#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONFIG="${CONFIG:-${ROOT}/neuromamba/configs/mamba3_corpus_sources.json}"
MAX_DOCS="${MAX_DOCS:-3000}"
MAX_BYTES="${MAX_BYTES:-200000000}"
MIN_CHARS="${MIN_CHARS:-300}"
OUT="${OUT:-luma/data/luma_stage_doc_cont_v1.jsonl}"

cd "${ROOT}"

MAX_DOCS="${MAX_DOCS}" \
MAX_BYTES="${MAX_BYTES}" \
MIN_CHARS="${MIN_CHARS}" \
CONFIG="${CONFIG}" \
neuromamba/scripts/mamba3_prepare_corpora.sh

inputs=()
for path in \
  neuromamba/data/governed_fineweb_edu_sample.jsonl \
  neuromamba/data/governed_dclm_sample.jsonl \
  neuromamba/data/governed_open_web_math_sample.jsonl \
  neuromamba/data/governed_arxiv_abstracts_sample.jsonl \
  luma/data/english_bootstrap.txt
do
  if [[ -f "${path}" ]]; then
    inputs+=("${path}")
  fi
done

if [[ "${#inputs[@]}" -eq 0 ]]; then
  echo "no language corpus inputs found" >&2
  exit 2
fi

python neuromamba/scripts/mamba3_build_doc_continuation_corpus.py \
  --inputs "${inputs[@]}" \
  --out "${OUT}" \
  --min-chars "${MIN_CHARS}"

python -m luma.analyze_training_data \
  "${OUT}" \
  luma/data/luma_stage_natural_speak_raw_v2.jsonl \
  --out luma/data/luma_language_corpus_analysis_v1.json

echo "luma_language_corpus=${OUT}"
