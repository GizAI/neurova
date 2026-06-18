#!/usr/bin/env bash
# Canonical LangBurst CUDA toolchain selector.
#
# The langburst conda environment uses a PyTorch CUDA 13 wheel. On ml-dmc8 the
# system /usr/bin/nvcc is CUDA 12, so builds and servers must prefer the CUDA 13
# toolchain shipped inside the conda environment.
set -euo pipefail

PY_SITE="$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
CONDA_CUDA_ROOT="${PY_SITE}/nvidia/cu13"
TORCH_LIB_ROOT="${PY_SITE}/torch/lib"

if [[ ! -x "${CONDA_CUDA_ROOT}/bin/nvcc" ]]; then
  echo "LangBurst CUDA 13 nvcc not found under ${CONDA_CUDA_ROOT}" >&2
  exit 1
fi

export CUDA_HOME="${CONDA_CUDA_ROOT}"
export CUDACXX="${CONDA_CUDA_ROOT}/bin/nvcc"
export PATH="${CONDA_CUDA_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${TORCH_LIB_ROOT}:${CONDA_CUDA_ROOT}/lib:${CONDA_CUDA_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
export LANGBURST_REQUIRE_CUDA_EXT="${LANGBURST_REQUIRE_CUDA_EXT:-1}"
