from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any


SUSPECT_TERMS = [
    "claim",
    "Explanation",
    "\nA",
    "\nB",
    "ced",
    "har",
    "patient",
    "relation",
    "object",
    "vector",
    "AX-888",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                obj = {"text": line}
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def text_of(row: dict[str, Any]) -> str:
    value = row.get("text")
    if isinstance(value, str):
        return value
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def repeated_ngrams(text: str, n: int = 4) -> int:
    tokens = re.findall(r"\S+", text)
    if len(tokens) < n * 2:
        return 0
    grams = [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = collections.Counter(grams)
    return max(counts.values(), default=0)


def analyze(path: Path, top: int) -> dict[str, Any]:
    rows = read_jsonl(path)
    lengths = [len(text_of(row)) for row in rows]
    term_hits = {term: 0 for term in SUSPECT_TERMS}
    repeat_hist: collections.Counter[int] = collections.Counter()
    source_counts: collections.Counter[str] = collections.Counter()
    role_counts: collections.Counter[str] = collections.Counter()
    samples: dict[str, list[str]] = {term: [] for term in SUSPECT_TERMS}
    for row in rows:
        text = text_of(row)
        source_counts[str(row.get("source", ""))] += 1
        role_counts[str(row.get("role", ""))] += 1
        repeat_hist[repeated_ngrams(text)] += 1
        for term in SUSPECT_TERMS:
            if term in text:
                term_hits[term] += 1
                if len(samples[term]) < 3:
                    samples[term].append(text[:240].replace("\n", "\\n"))
    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    return {
        "path": str(path),
        "records": len(rows),
        "avg_chars": round(avg_len, 2),
        "max_chars": max(lengths, default=0),
        "suspect_hits": term_hits,
        "repeat4_hist": dict(sorted(repeat_hist.items())),
        "top_sources": source_counts.most_common(top),
        "top_roles": role_counts.most_common(top),
        "samples": {term: vals for term, vals in samples.items() if vals},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze LUMA dataset contamination and repetition risk.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()
    payload = {"datasets": [analyze(path, args.top) for path in args.paths]}
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
