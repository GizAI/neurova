from __future__ import annotations

import os
import shutil
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

ROOT = Path(__file__).parent


def _cuda_arch_flags() -> list[str]:
    """Return nvcc -gencode flags.

    Default is SM89 for RTX 4080/4090 Ada. Override with:
      QWENBURST_CUDA_ARCH_LIST="8.9;9.0"
    """
    raw = os.environ.get("QWENBURST_CUDA_ARCH_LIST", "8.9")
    flags: list[str] = []
    for item in raw.replace(",", ";").split(";"):
        item = item.strip()
        if not item:
            continue
        major_minor = item.replace("sm_", "").replace("compute_", "").replace(".", "")
        flags.append(f"-gencode=arch=compute_{major_minor},code=sm_{major_minor}")
    return flags or ["-gencode=arch=compute_89,code=sm_89"]


def get_extensions():
    if os.environ.get("QWENBURST_SKIP_CUDA_EXT", "0") == "1":
        return []

    # Keep CPU-only editing/testing usable. On the real RTX box, set
    # QWENBURST_REQUIRE_CUDA_EXT=1 to fail hard if CUDA_HOME/nvcc is missing.
    require_cuda = os.environ.get("QWENBURST_REQUIRE_CUDA_EXT", "0") == "1"
    nvcc = shutil.which("nvcc")
    if CUDA_HOME is None or nvcc is None:
        msg = (
            "QwenBurst CUDA extension not built because CUDA_HOME/nvcc was not found. "
            "Set QWENBURST_REQUIRE_CUDA_EXT=1 on the RTX 4080 box to force a hard failure."
        )
        if require_cuda:
            raise RuntimeError(msg)
        print("[qwenburst] " + msg)
        return []

    sources = [
        "csrc/qwenburst_ext.cpp",
        "csrc/lowbit_gemv.cu",
        "csrc/marlin_cuda_kernel.cu",
        "csrc/rmsnorm.cu",
        "csrc/gdn_recurrent.cu",
        "csrc/attention_decode.cu",
        "csrc/sampling.cu",
    ]
    nvcc_args = [
        "-O3",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "-std=c++17",
        "-lineinfo",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-Xptxas=-v",
    ] + _cuda_arch_flags()
    return [
        CUDAExtension(
            name="qwenburst_cuda",
            sources=sources,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": nvcc_args,
            },
        )
    ]


setup(
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
