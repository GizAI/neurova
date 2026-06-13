#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/mamba3_clean_doc_base_moe24_v1}"
LONG_DIR="${LONG_DIR:-${RUN_DIR}/long_blocks}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG="${LOG:-${LONG_DIR}/${STAMP}_100m.log}"
PIDFILE="${PIDFILE:-${LONG_DIR}/${STAMP}_100m.pid}"
CMDFILE="${CMDFILE:-${LONG_DIR}/${STAMP}_100m.cmd}"

mkdir -p "${LONG_DIR}"

if pgrep -f "mamba3_kr.cli train-packed --mode mimo-r4-moe-2.4b" >/dev/null; then
  echo "A mimo-r4-moe-2.4b train-packed process is already running:" >&2
  ps -eo pid,etime,cmd | grep "mamba3_kr.cli train-packed --mode mimo-r4-moe-2.4b" | grep -v grep >&2
  exit 2
fi

cat > "${CMDFILE}" <<EOF
cd "${ROOT}"
source "\${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate mamba3_siso
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TARGET_TOKENS="${TARGET_TOKENS:-100000000}"
export LR="${LR:-8e-6}"
export EVAL_BATCHES="${EVAL_BATCHES:-32}"
export DATA_SEED_BASE="${DATA_SEED_BASE:-997000}"
export MAX_TEXT_CHARS="${MAX_TEXT_CHARS:-65536}"
export MAX_TEXT_TOKENS="${MAX_TEXT_TOKENS:-120000}"
exec scripts/mamba3_train_max_moe_100m_block.sh
EOF

setsid bash "${CMDFILE}" > "${LOG}" 2>&1 < /dev/null &
launcher_pid=$!
sleep 3

train_pid="$(pgrep -f "mamba3_kr.cli train-packed --mode mimo-r4-moe-2.4b" | tail -n 1 || true)"
if [[ -z "${train_pid}" ]]; then
  echo "Training did not start. Last log lines:" >&2
  tail -n 80 "${LOG}" >&2 || true
  exit 1
fi

echo "${train_pid}" > "${PIDFILE}"
echo "launcher_pid=${launcher_pid}"
echo "train_pid=${train_pid}"
echo "log=${LOG}"
echo "pidfile=${PIDFILE}"
echo "cmdfile=${CMDFILE}"
