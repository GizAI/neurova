#!/usr/bin/env bash
set -euo pipefail

echo "mamba3_train_governed_tiny.sh is deprecated because it mixed raw web data into an instruction checkpoint."
echo "Running the staged scientific pipeline instead: base pretrain -> validation -> instruction SFT -> eval."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/mamba3_train_scientific_tiny.sh"
