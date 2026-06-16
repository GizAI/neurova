#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def read_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = str(row.get("text", "")).strip()
                if "\nAnswer:" not in text and " Answer:" not in text:
                    continue
                if row.get("benchmark_contamination_flag") is True:
                    continue
                key = str(row.get("dedup_hash") or text.casefold())
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge no-cheat teacher MCQ JSONL files with dedup.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=20260614)
    args = parser.parse_args()

    rows = read_rows(args.inputs)
    rng = random.Random(args.seed)
    deepseek = [r for r in rows if str(r.get("teacher_model", "")).startswith("deepseek")]
    other = [r for r in rows if r not in deepseek]
    rng.shuffle(deepseek)
    rng.shuffle(other)
    # Keep frontier teacher rows first, then fill with deterministic no-cheat coverage.
    mixed = (deepseek + other)[: args.max_records]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in mixed:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(args.out),
        "records": len(mixed),
        "deepseek_records": len(deepseek[: args.max_records]),
        "input_records": len(rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
