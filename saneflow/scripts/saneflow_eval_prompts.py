#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


PROMPTS = [
    "Once upon a time",
    "Lily found a small",
    "The robot went to the garden because",
    "Explain what a computer is in simple words:",
    "Write one sentence about the moon:",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run fixed generation probes for SaneFlow checkpoints.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--max-new", type=int, default=80)
    p.add_argument("--context", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--decode", choices=["full", "cache"], default="cache")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for prompt in PROMPTS:
        cmd = [
            sys.executable,
            str(ROOT / "saneflow/scripts/saneflow_generate.py"),
            "--ckpt",
            args.ckpt,
            "--prompt",
            prompt,
            "--max-new",
            str(args.max_new),
            "--context",
            str(args.context),
            "--temperature",
            str(args.temperature),
            "--top-k",
            str(args.top_k),
            "--top-p",
            str(args.top_p),
            "--decode",
            args.decode,
            "--device",
            args.device,
            "--dtype",
            args.dtype,
        ]
        result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        text = result.stdout.strip()
        rows.append({"prompt": prompt, "output": text, "returncode": result.returncode})
        print(f"=== PROMPT: {prompt} ===")
        print(text)
        print()
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
