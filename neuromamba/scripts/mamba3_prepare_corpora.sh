#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONFIG="${CONFIG:-${ROOT}/neuromamba/configs/mamba3_corpus_sources.json}"
MAX_DOCS="${MAX_DOCS:-}"
MAX_BYTES="${MAX_BYTES:-}"
MIN_CHARS="${MIN_CHARS:-}"

cd "${ROOT}"

python -m pip install -q datasets huggingface_hub pyarrow zstandard

python - "$CONFIG" "$MAX_DOCS" "$MAX_BYTES" "$MIN_CHARS" <<'PY' | while IFS=$'\t' read -r dataset config_name split text_field out source license domain quality max_docs max_bytes min_chars trust_remote_code; do
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
override_docs, override_bytes, override_min_chars = sys.argv[2:5]
budget = config["default_budget"]
for source in config["sources"]:
    if source.get("status") != "enabled":
        continue
    print(
        source["dataset"],
        source.get("config") or "__none__",
        source.get("split", "train"),
        source.get("text_field", "text"),
        source["out"],
        source["dataset"],
        source["license"],
        source["domain"],
        str(source.get("quality_score", 0.7)),
        override_docs or str(budget["max_docs_per_source"]),
        override_bytes or str(budget["max_bytes_per_source"]),
        override_min_chars or str(budget["min_chars"]),
        "1" if source.get("trust_remote_code") else "0",
        sep="\t",
    )
PY
  echo "== download ${source} -> ${out} =="
  config_args=()
  if [[ "${config_name}" != "__none__" ]]; then
    config_args+=(--config "${config_name}")
  fi
  trust_args=()
  if [[ "${trust_remote_code}" == "1" ]]; then
    trust_args+=(--trust-remote-code)
  fi
  python neuromamba/scripts/mamba3_download_streaming_corpus.py \
    --dataset "${dataset}" \
    "${config_args[@]}" \
    --split "${split}" \
    --text-field "${text_field}" \
    --out "${out}" \
    --source "${source}" \
    --license "${license}" \
    --domain "${domain}" \
    --quality-score "${quality}" \
    --max-docs "${max_docs}" \
    --max-bytes "${max_bytes}" \
    --min-chars "${min_chars}" \
    "${trust_args[@]}"
  neuromamba/scripts/mamba3_validate_governance.py "${out}"
done

neuromamba/scripts/mamba3_corpus_manifest.py neuromamba/data/*.txt neuromamba/data/*.jsonl --out neuromamba/data/mamba3_corpus_manifest.json
