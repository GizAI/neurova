#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"

CONTROL_DIR="${CONTROL_DIR:-neuromamba/runs/mamba3_current/apply_watcher}"
CHECKPOINT="${CHECKPOINT:-neuromamba/runs/mamba3_current/model.pt}"
INTERVAL="${INTERVAL:-120}"
mkdir -p "${CONTROL_DIR}"

mtime_of() {
  stat -c '%Y' "$1" 2>/dev/null || echo 0
}

while true; do
  if [[ -f "${CHECKPOINT}" ]]; then
    mtime="$(mtime_of "${CHECKPOINT}")"
    last="$(cat "${CONTROL_DIR}/last_applied_mtime" 2>/dev/null || echo 0)"
    if [[ "${mtime}" != "0" && "${mtime}" != "${last}" ]]; then
      echo "apply_current_checkpoint mtime=${mtime} checkpoint=${CHECKPOINT}"
      NEUROVA_MAMBA3_CHECKPOINT="${CHECKPOINT}" \
      NEUROVA_MAMBA3_MODE="${NEUROVA_MAMBA3_MODE:-mamba3-siso-fast-0.3b-ds128}" \
      NEUROVA_MAMBA3_TOKENIZER="${NEUROVA_MAMBA3_TOKENIZER:-llama31}" \
      NEUROVA_MAMBA3_SEQ="${NEUROVA_MAMBA3_SEQ:-128}" \
      NEUROVA_MAMBA3_DECODE_MODE="${NEUROVA_MAMBA3_DECODE_MODE:-cache}" \
      NEUROVA_MAMBA3_CUDA_GRAPH="${NEUROVA_MAMBA3_CUDA_GRAPH:-1}" \
      neuromamba/scripts/mamba3_chat_serverctl.sh restart || true
      printf '%s\n' "${mtime}" > "${CONTROL_DIR}/last_applied_mtime"
      date -u +%Y-%m-%dT%H:%M:%SZ > "${CONTROL_DIR}/last_applied_at"
    fi
  fi
  sleep "${INTERVAL}"
done
