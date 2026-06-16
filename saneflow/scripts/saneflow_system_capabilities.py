#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch


def main() -> None:
    cuda = torch.cuda.is_available()
    report = {
        "torch": torch.__version__,
        "cuda": cuda,
        "device": torch.cuda.get_device_name(0) if cuda else "",
        "cuda_capability": torch.cuda.get_device_capability(0) if cuda else None,
        "bf16_supported": bool(cuda and torch.cuda.is_bf16_supported()),
        "float8_dtype_present": bool(hasattr(torch, "float8_e4m3fn")),
        "sdpa_flash_enabled": bool(cuda and torch.backends.cuda.flash_sdp_enabled()),
        "sdpa_mem_efficient_enabled": bool(cuda and torch.backends.cuda.mem_efficient_sdp_enabled()),
        "sdpa_math_enabled": bool(cuda and torch.backends.cuda.math_sdp_enabled()),
        "packages": {
            "flash_attn": importlib.util.find_spec("flash_attn") is not None,
            "deepspeed": importlib.util.find_spec("deepspeed") is not None,
            "galore_torch": importlib.util.find_spec("galore_torch") is not None,
            "bitsandbytes": importlib.util.find_spec("bitsandbytes") is not None,
        },
        "fsdp_available": False,
        "zero_available": False,
        "active_policy": {
            "attention": "torch.scaled_dot_product_attention with Flash/mem-efficient backends when PyTorch dispatch allows it",
            "dtype": "bf16 for active training; fp8 requires a separate smoke test and scaling recipe before use",
            "optimizer": "Muon for active dense profile; built-in GaLoreAdamW is available only as a memory-reduction experiment",
            "checkpointing": "activation checkpointing enabled on dense profile",
            "fsdp_zero": "single-GPU runs do not use FSDP/ZeRO; ZeRO requires deepspeed package and a multi-process launcher",
        },
    }
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP  # noqa: F401

        report["fsdp_available"] = True
    except Exception:
        report["fsdp_available"] = False
    report["zero_available"] = bool(report["packages"]["deepspeed"])
    out = Path("saneflow/runs/saneflow_system_capabilities.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
