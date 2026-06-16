#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuromamba.cli import setup_perf, torch_dtype
from neuromamba.model import build_model, preset_config
from neuromamba.tokenizer import build_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize a candidate checkpoint from shape-compatible weights.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--tokenizer", default="llama31")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args()

    setup_perf(args.device)
    tokenizer = build_tokenizer(args.tokenizer)
    cfg = preset_config(args.mode, vocab_size=tokenizer.vocab_size)
    model = build_model(cfg, device=args.device, dtype=torch_dtype(args.dtype))

    source = torch.load(args.source, map_location=args.device)
    source_state = source.get("model", source)
    target_state = model.state_dict()
    copied = {}
    skipped = {}
    for key, value in source_state.items():
        target = target_state.get(key)
        if target is not None and tuple(target.shape) == tuple(value.shape):
            copied[key] = value.to(device=target.device, dtype=target.dtype)
        else:
            skipped[key] = {
                "source": list(value.shape) if hasattr(value, "shape") else None,
                "target": list(target.shape) if target is not None and hasattr(target, "shape") else None,
            }
    missing, unexpected = model.load_state_dict(copied, strict=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": None,
            "mode": args.mode,
            "tokenizer": args.tokenizer,
            "step": int(source.get("step", 0)) if isinstance(source, dict) else 0,
            "losses": [],
            "transplant": {
                "source": str(args.source),
                "copied_tensors": len(copied),
                "skipped_tensors": len(skipped),
                "missing_tensors": len(missing),
                "unexpected_tensors": len(unexpected),
            },
        },
        args.out,
    )
    print(json.dumps({
        "out": str(args.out),
        "mode": args.mode,
        "source": str(args.source),
        "copied_tensors": len(copied),
        "skipped_tensors": len(skipped),
        "missing_tensors": len(missing),
        "unexpected_tensors": len(unexpected),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
