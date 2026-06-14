#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download a clean TinyStories subset for SaneFlow.")
    p.add_argument("--out-dir", default="data/saneflow/tinystories")
    p.add_argument("--train-records", type=int, default=200000)
    p.add_argument("--valid-records", type=int, default=2000)
    return p.parse_args()


def write_split(split_name: str, out_path: Path, limit: int) -> int:
    ds = load_dataset("roneneldan/TinyStories", split=split_name, streaming=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            text = (row.get("text") or "").strip()
            if len(text) < 80:
                continue
            if "Answer:" in text or "Question:" in text:
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1
            if count >= limit:
                break
    return count


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_count = write_split("train", out / "train.jsonl", args.train_records)
    valid_count = write_split("validation", out / "valid.jsonl", args.valid_records)
    manifest = {
        "source": "roneneldan/TinyStories",
        "train_records": train_count,
        "valid_records": valid_count,
        "format": "jsonl text field, raw continuation",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
