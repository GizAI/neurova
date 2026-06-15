from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


def _cmd_output(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=10).strip()
    except Exception as exc:
        return f"unavailable ({type(exc).__name__}: {exc})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check whether this machine can build/run QwenBurst CUDA kernels")
    ap.add_argument("--require-cuda", action="store_true", help="return non-zero when CUDA/nvcc/GPU is missing")
    args = ap.parse_args(argv)

    import torch

    nvcc = shutil.which("nvcc")
    ok = True
    print("QwenBurst CUDA doctor")
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', '<unset>')}")
    print(f"nvcc: {nvcc or '<missing>'}")
    if nvcc:
        print(_cmd_output([nvcc, "--version"]).splitlines()[-1])
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    print(f"device_count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"device {i}: {torch.cuda.get_device_name(i)} capability={torch.cuda.get_device_capability(i)}")
    else:
        ok = False
    if nvcc is None:
        ok = False
    if args.require_cuda and not ok:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
