#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def normalize_record(obj: Any, fallback_source: str) -> tuple[str, str] | None:
    if isinstance(obj, str):
        text = obj.strip()
        return (text, fallback_source) if text else None
    if not isinstance(obj, dict):
        return None
    text = ""
    for key in ("text", "content", "prompt", "input", "output", "response", "completion"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            break
    if not text:
        return None
    source = obj.get("source") or obj.get("dataset") or obj.get("domain") or fallback_source
    return text, str(source)


def read_grouped(paths: list[Path]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        fallback_source = path.stem
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    item = (raw, fallback_source)
                else:
                    item = normalize_record(obj, fallback_source)
                if item is None:
                    continue
                text, source = item
                grouped[source].append(text)
    return grouped


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as fh:
            for text in lines:
                fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        return
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic source-stratified Mamba-3 train/valid splits.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--valid-out", type=Path, required=True)
    parser.add_argument("--valid-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    grouped = read_grouped(args.inputs)
    train: list[str] = []
    valid: list[str] = []
    source_counts: dict[str, dict[str, int]] = {}

    for source in sorted(grouped):
        lines = grouped[source]
        rng.shuffle(lines)
        if args.max_records > 0 and len(lines) > args.max_records:
            lines = lines[: args.max_records]
        valid_count = max(1, int(len(lines) * args.valid_ratio)) if len(lines) > 1 else 0
        valid_part = lines[:valid_count]
        train_part = lines[valid_count:]
        valid.extend(valid_part)
        train.extend(train_part)
        source_counts[source] = {
            "train": len(train_part),
            "valid": len(valid_part),
            "total": len(lines),
        }

    rng.shuffle(train)
    rng.shuffle(valid)
    write_lines(args.train_out, train)
    write_lines(args.valid_out, valid)
    print({
        "inputs": [str(p) for p in args.inputs],
        "train_out": str(args.train_out),
        "valid_out": str(args.valid_out),
        "train_records": len(train),
        "valid_records": len(valid),
        "seed": args.seed,
        "valid_ratio": args.valid_ratio,
        "sources": source_counts,
    })


if __name__ == "__main__":
    main()
