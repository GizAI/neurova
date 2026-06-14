#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def sha256_head(path: Path, limit: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(limit))
    return h.hexdigest()


def audit_jsonl(path: Path, sample_rows: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
    }
    if not path.exists() or path.stat().st_size == 0:
        return out
    sources: Counter[str] = Counter()
    datasets: Counter[str] = Counter()
    keys: Counter[str] = Counter()
    lengths = []
    examples = []
    rows = 0
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            rows += 1
            keys.update(obj.keys())
            sources[str(obj.get("source", ""))] += 1
            datasets[str(obj.get("dataset", ""))] += 1
            text = str(obj.get("text") or "")
            lengths.append(len(text))
            if len(examples) < 3:
                examples.append({"keys": sorted(obj.keys()), "source": obj.get("source", ""), "text": text[:500]})
            if rows >= sample_rows:
                break
    out.update({
        "sample_rows": rows,
        "sha256_head": sha256_head(path),
        "avg_chars_sample": round(sum(lengths) / max(1, len(lengths)), 2),
        "min_chars_sample": min(lengths) if lengths else 0,
        "max_chars_sample": max(lengths) if lengths else 0,
        "keys": keys.most_common(20),
        "sources": sources.most_common(20),
        "datasets": datasets.most_common(20),
        "examples": examples,
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SaneFlow local corpus files.")
    parser.add_argument("--out", type=Path, default=Path("runs/saneflow_data_audit/audit.json"))
    parser.add_argument("--sample-rows", type=int, default=20000)
    parser.add_argument("--paths", nargs="*", default=[])
    args = parser.parse_args()

    paths = [Path(p) for p in args.paths]
    if not paths:
        roots = [Path("data/corpus/sources"), Path("data/corpus/mixes"), Path("data/corpus/sft_sources")]
        for root in roots:
            if root.exists():
                paths.extend(sorted(root.rglob("*.jsonl")))
                paths.extend(sorted(root.rglob("*.json")))
    rows = [audit_jsonl(path, args.sample_rows) for path in paths]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"files": rows}
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "files": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
