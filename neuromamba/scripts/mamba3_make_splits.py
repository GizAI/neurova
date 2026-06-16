#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def read_lines(paths: list[Path]) -> list[str]:
    lines: list[str] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.strip():
                    lines.append(normalize_record_line(line))
    return lines


def normalize_record_line(line: str) -> str:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return line
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for key in ("text", "content", "prompt", "input", "output", "response", "completion"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return line


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as fh:
            for text in lines:
                fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        return
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic Mamba-3 train/valid splits.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--valid-out", type=Path, required=True)
    parser.add_argument("--valid-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()

    lines = read_lines(args.inputs)
    rng = random.Random(args.seed)
    rng.shuffle(lines)
    if args.max_records > 0:
        lines = lines[: args.max_records]
    valid_count = max(1, int(len(lines) * args.valid_ratio)) if len(lines) > 1 else 0
    valid = lines[:valid_count]
    train = lines[valid_count:]
    write_lines(args.train_out, train)
    write_lines(args.valid_out, valid)
    print({
        "inputs": [str(p) for p in args.inputs],
        "train_out": str(args.train_out),
        "valid_out": str(args.valid_out),
        "train_records": len(train),
        "valid_records": len(valid),
        "seed": args.seed,
    })


if __name__ == "__main__":
    main()
