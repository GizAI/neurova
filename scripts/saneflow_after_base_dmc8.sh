#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec bash scripts/saneflow_standard_sft_after_base_dmc8.sh
