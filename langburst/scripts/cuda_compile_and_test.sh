#!/usr/bin/env bash
set -euo pipefail

# End-to-end RTX 4080/4090 validation path:
#   1. hard-require CUDA/nvcc
#   2. compile langburst_cuda for SM89
#   3. run kernel parity tests
#   4. run microbenchmarks

cd "$(dirname "$0")/.."
export LANGBURST_REQUIRE_CUDA_EXT=1
export LANGBURST_CUDA_ARCH_LIST="${LANGBURST_CUDA_ARCH_LIST:-8.9}"

python -m langburst.doctor --require-cuda
python -m pip install -U pip wheel setuptools ninja pytest
python -m pip install -v --no-build-isolation -e .
python - <<'PY'
import langburst_cuda
print('langburst_cuda loaded:', langburst_cuda.__doc__)
PY
pytest -q \
  tests/test_quant_lowbit_cpu.py \
  tests/test_gdn_reference_cpu.py \
  tests/test_v05_runtime_cpu.py \
  tests/test_gdn_parity_cuda.py \
  tests/test_lowbit_gemv_cuda.py \
  tests/test_sampling_cuda.py \
  tests/test_v05_cuda_kernels.py
python benchmarks/bench_kernels.py --iters "${LANGBURST_BENCH_ITERS:-1000}"
