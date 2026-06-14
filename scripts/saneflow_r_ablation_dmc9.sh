#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

profiles=(
  dmc9-r-a-delta-only
  dmc9-r-b-delta-sparse-attn
  dmc9-r-c-delta-thought-late
  dmc9-r-d-delta-landmark
  dmc9-r-e-full-lite
)

for profile in "${profiles[@]}"; do
  echo "== ${profile} =="
  python scripts/saneflow_run.py train "${profile}"
done

python scripts/saneflow_r_ablation_report.py
