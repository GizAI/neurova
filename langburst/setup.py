from __future__ import annotations

import os
import shutil
import site
import sys
from pathlib import Path
from setuptools import setup


def _prefer_conda_cuda_toolchain() -> None:
    """Prefer the CUDA toolkit bundled with the active conda environment.

    ml-dmc8 has /usr/bin/nvcc from CUDA 12 while the langburst PyTorch wheel is
    CUDA 13. Letting torch extensions discover /usr/bin/nvcc creates a hard
    version mismatch. If the CUDA 13 package exists in the active environment,
    make it the default before importing torch's CUDA_HOME helper.
    """

    if os.environ.get("LANGBURST_NO_CONDA_CUDA_PIN", "0") == "1":
        return
    root = Path(sys.prefix) / "lib/python3.11/site-packages/nvidia/cu13"
    nvcc = root / "bin/nvcc"
    if not nvcc.is_file():
        return
    os.environ.setdefault("CUDA_HOME", str(root))
    os.environ.setdefault("CUDACXX", str(nvcc))
    os.environ["PATH"] = f"{root / 'bin'}:{os.environ.get('PATH', '')}"


_prefer_conda_cuda_toolchain()

from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

ROOT = Path(__file__).parent


def _cuda_arch_flags() -> list[str]:
    """Return nvcc -gencode flags.

    Default is SM89 for RTX 4080/4090 Ada. Override with:
      LANGBURST_CUDA_ARCH_LIST="8.9;9.0"
    """
    raw = os.environ.get("LANGBURST_CUDA_ARCH_LIST", "8.9")
    flags: list[str] = []
    for item in raw.replace(",", ";").split(";"):
        item = item.strip()
        if not item:
            continue
        major_minor = item.replace("sm_", "").replace("compute_", "").replace(".", "")
        flags.append(f"-gencode=arch=compute_{major_minor},code=sm_{major_minor}")
    return flags or ["-gencode=arch=compute_89,code=sm_89"]


def _cuda_include_dirs() -> list[str]:
    """Return CUDA include dirs for conda/pip CUDA layouts.

    PyTorch wheels often bring CUDA headers through nvidia-* packages under
    site-packages instead of a classic /usr/local/cuda tree. nvcc can compile
    torch extensions in that environment, but transitive CUB/Thrust includes
    may fail on headers like cusparse.h unless those package include dirs are
    added explicitly.
    """
    candidates: list[Path] = []
    if CUDA_HOME:
        candidates.append(Path(CUDA_HOME) / "include")
    for root in site.getsitepackages() + [site.getusersitepackages()]:
        if not root:
            continue
        candidates.extend(Path(root).glob("nvidia/*/include"))
    seen: set[str] = set()
    include_dirs: list[str] = []
    for path in candidates:
        if not path.is_dir():
            continue
        resolved = str(path.resolve())
        if resolved not in seen:
            include_dirs.append(resolved)
            seen.add(resolved)
    return include_dirs


def get_extensions():
    if os.environ.get("LANGBURST_SKIP_CUDA_EXT", "0") == "1":
        return []

    # Keep CPU-only editing/testing usable. On the real RTX box, set
    # LANGBURST_REQUIRE_CUDA_EXT=1 to fail hard if CUDA_HOME/nvcc is missing.
    require_cuda = os.environ.get("LANGBURST_REQUIRE_CUDA_EXT", "0") == "1"
    nvcc = shutil.which("nvcc")
    if CUDA_HOME is None or nvcc is None:
        msg = (
            "LangBurst CUDA extension not built because CUDA_HOME/nvcc was not found. "
            "Set LANGBURST_REQUIRE_CUDA_EXT=1 on the RTX 4080 box to force a hard failure."
        )
        if require_cuda:
            raise RuntimeError(msg)
        print("[langburst] " + msg)
        return []

    sources = [
        "csrc/langburst_ext.cpp",
        "csrc/lowbit_gemv.cu",
        "csrc/marlin_cuda_kernel.cu",
        "csrc/marlin_mlp_stream.cu",
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
            name="langburst_cuda",
            sources=sources,
            include_dirs=_cuda_include_dirs(),
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
