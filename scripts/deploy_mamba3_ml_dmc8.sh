#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-ml-dmc8}"
ROOT="${ROOT:-/home/user/workspace/neurova}"
ENV_NAME="${ENV_NAME:-mamba3_siso}"
MODE="${MODE:-mimo-r4-tiny}"
SEQ_LEN="${SEQ_LEN:-512}"
STEPS="${STEPS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DATA_PATH="${DATA_PATH:-data/english_bootstrap.txt}"
TOKENIZER="${TOKENIZER:-llama31}"
CHECKPOINT="${CHECKPOINT:-runs/mamba3_kr_tiny/model_mimo_r4_llama31.pt}"

echo "=== Mamba3-KR deploy to ${HOST} ==="

rsync -az --delete \
  "${ROOT}/mamba3_kr" \
  "${ROOT}/mamba" \
  "${ROOT}/data" \
  "${ROOT}/papers" \
  "${ROOT}/scripts" \
  "${HOST}:${ROOT}/"

ssh "${HOST}" \
  "ROOT='${ROOT}' ENV_NAME='${ENV_NAME}' MODE='${MODE}' SEQ_LEN='${SEQ_LEN}' STEPS='${STEPS}' BATCH_SIZE='${BATCH_SIZE}' DATA_PATH='${DATA_PATH}' TOKENIZER='${TOKENIZER}' CHECKPOINT='${CHECKPOINT}' bash -s" <<'REMOTE'
set -euo pipefail
cd "${ROOT}"
source ~/miniconda3/etc/profile.d/conda.sh
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.11
fi
conda activate "${ENV_NAME}"
python -m pip install -U pip wheel 'setuptools<82' ninja packaging
if python - <<'PY'
try:
    import torch
except Exception:
    raise SystemExit(1)
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  :
else
  python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
  python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
PY
fi
python -m pip install einops triton transformers numpy
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.9 MAMBA_FORCE_BUILD=TRUE \
  python -m pip install --no-cache-dir --force-reinstall --no-deps -e ./mamba --no-build-isolation
python -m pip install --no-deps tilelang==0.1.8 apache-tvm-ffi==0.1.9 torch-c-dlpack-ext cloudpickle ml-dtypes psutil z3-solver
python -m mamba3_kr.cli smoke --mode "${MODE}" --tokenizer "${TOKENIZER}" --seq-len "${SEQ_LEN}" --batch-size "${BATCH_SIZE}" --device cuda --checkpoint "${CHECKPOINT}"
python -m mamba3_kr.cli train-tiny --mode "${MODE}" --tokenizer "${TOKENIZER}" --seq-len "${SEQ_LEN}" --batch-size "${BATCH_SIZE}" --steps "${STEPS}" --device cuda --data "${DATA_PATH}" --checkpoint "${CHECKPOINT}"
python -m mamba3_kr.cli check-contract --mode "${MODE}" --tokenizer "${TOKENIZER}" --device cuda
python -m mamba3_kr.cli state-prefill --mode "${MODE}" --tokenizer "${TOKENIZER}" --seq-len "${SEQ_LEN}" --device cuda --state-out runs/mamba3_kr_tiny/state_mimo_r4_llama31.pt --checkpoint "${CHECKPOINT}" --text 'Mamba-3 MIMO-R4 recurrent state checkpoint for English-first training.'
python -m mamba3_kr.cli diagnose-decode --mode "${MODE}" --tokenizer "${TOKENIZER}" --seq-len "${SEQ_LEN}" --device cuda --checkpoint "${CHECKPOINT}" --prompt 'The main idea is'
python -m mamba3_kr.cli bench-decode --mode "${MODE}" --tokenizer "${TOKENIZER}" --seq-len "${SEQ_LEN}" --device cuda --checkpoint "${CHECKPOINT}" --prompt 'The main idea is' --max-new 64 --cuda-graph --top-k 1 --top-p 0 --temperature 1.0 --repetition-penalty 1.0
REMOTE
