#!/usr/bin/env bash
set -euo pipefail

cd /home/user/workspace/neurova/langburst
source ~/miniconda3/etc/profile.d/conda.sh
conda activate langburst
source ./scripts/langburst_cuda_env.sh

LOCK_FILE="${LANGBURST_BUILD_LOCK_FILE:-/tmp/langburst_cuda_build.lock}"
LOG_FILE="${LANGBURST_BUILD_LOG_FILE:-/tmp/langburst_cuda_build.log}"
BUILD_ROOT="${LANGBURST_BUILD_ROOT:-/tmp/langburst_cuda_build_${USER}_$$}"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another LangBurst CUDA build is already running; waiting on $LOCK_FILE" >&2
  flock 9
fi

rm -rf "$BUILD_ROOT"
mkdir -p "$BUILD_ROOT/temp" "$BUILD_ROOT/lib"
trap 'rm -rf "$BUILD_ROOT"' EXIT
rm -f langburst_cuda*.so
MAX_JOBS="${MAX_JOBS:-1}" LANGBURST_REQUIRE_CUDA_EXT=1 \
  python setup.py build_ext --inplace \
    --build-temp "$BUILD_ROOT/temp" \
    --build-lib "$BUILD_ROOT/lib" \
    2>&1 | tee "$LOG_FILE"

python - <<'PY'
from langburst.ops import cuda_ops

ops = cuda_ops()
required = [
    "attention_decode_fp16_gated",
    "gdn_recurrent_ab_batch_norm_gate",
    "lowbit_marlin_gemm_silu_packed_out",
    "lowbit_marlin_gemm_argmax_out",
    "rmsnorm_qwen_pair_cat",
    "rmsnorm_qwen_rope",
]
missing = [name for name in required if not hasattr(ops, name)]
if missing:
    raise SystemExit(f"missing CUDA ops after build: {missing}")
print("LangBurst CUDA build ok")
for name in required:
    print(f"  {name}=true")
PY
