#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec python scripts/saneflow_run.py train dmc8-base-100m
