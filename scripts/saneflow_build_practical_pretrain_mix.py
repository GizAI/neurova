#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def digest(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8", errors="ignore")).hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                text = obj.get("text", "")
                if isinstance(text, str) and text.strip():
                    rows.append(obj)
    return rows


def sample_split(recipe: dict[str, Any], split: str, target: int, rng: random.Random) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    reports = []
    for source in recipe["sources"]:
        path = Path(source[split])
        rows = read_rows(path)
        rng.shuffle(rows)
        want = int(round(target * float(source["ratio"])))
        taken = 0
        for row in rows:
            text = row["text"].strip()
            key = row.get("dedup_hash") or digest(text)
            if key in seen:
                continue
            seen.add(key)
            selected.append({
                "text": text,
                "source": row.get("source") or source["name"],
                "dataset": row.get("dataset") or source["name"],
                "language": row.get("language", ""),
                "domain": row.get("domain", ""),
                "dedup_hash": key,
            })
            taken += 1
            if taken >= want:
                break
        reports.append({
            "source": source["name"],
            "split": split,
            "path": str(path),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
            "available": len(rows),
            "wanted": want,
            "taken": taken,
            "missing_or_empty": not path.exists() or (path.exists() and path.stat().st_size == 0),
        })

    if len(selected) < target:
        remainder = []
        for source in recipe["sources"]:
            remainder.extend(read_rows(Path(source[split])))
        rng.shuffle(remainder)
        for row in remainder:
            text = row.get("text", "").strip()
            if not text:
                continue
            key = row.get("dedup_hash") or digest(text)
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
            if len(selected) >= target:
                break
    rng.shuffle(selected)
    return selected[:target], reports


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a ratio-controlled practical base pretrain mix.")
    parser.add_argument("--recipe", type=Path, default=Path("configs/saneflow_practical_pretrain_mix.json"))
    parser.add_argument("--seed", type=int, default=20260614)
    args = parser.parse_args()

    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    train, train_reports = sample_split(recipe, "train", int(recipe["train_records"]), rng)
    valid, valid_reports = sample_split(recipe, "valid", int(recipe["valid_records"]), rng)
    write_jsonl(Path(recipe["train_out"]), train)
    write_jsonl(Path(recipe["valid_out"]), valid)
    manifest = {
        "name": recipe["name"],
        "train": recipe["train_out"],
        "valid": recipe["valid_out"],
        "train_records": len(train),
        "valid_records": len(valid),
        "train_counts": Counter(row.get("dataset", "") for row in train),
        "valid_counts": Counter(row.get("dataset", "") for row in valid),
        "reports": train_reports + valid_reports,
    }
    out = Path(recipe["manifest_out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=dict) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
