#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_text(line: str, path: Path, line_no: int) -> str:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{line_no}: expected JSONL record") from exc
    if not isinstance(obj, dict) or not isinstance(obj.get("text"), str):
        raise ValueError(f"{path}:{line_no}: expected object with text string")
    return obj["text"]


def validate(path: Path, sample_limit: int) -> dict:
    if path.suffix != ".jsonl":
        raise ValueError(f"{path}: doc-continuation splits must be .jsonl, not {path.suffix or '<none>'}")
    total = 0
    bad = []
    newline_counts = []
    char_counts = []
    source_counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                text = load_text(line, path, line_no)
            except ValueError as exc:
                bad.append(str(exc))
                continue
            if not text.startswith("<doc "):
                bad.append(f"{path}:{line_no}: text does not start with <doc")
            if "</doc>" not in text:
                bad.append(f"{path}:{line_no}: missing </doc>")
            if text.count("<doc ") != 1:
                bad.append(f"{path}:{line_no}: expected exactly one <doc marker")
            if text.count("</doc>") != 1:
                bad.append(f"{path}:{line_no}: expected exactly one </doc> marker")
            newline_counts.append(text.count("\n"))
            char_counts.append(len(text))
            marker = text.split(">", 1)[0]
            source = "unknown"
            for part in marker.split():
                if part.startswith("source="):
                    source = part.split("=", 1)[1].strip('"')
                    break
            source_counts[source] = source_counts.get(source, 0) + 1
            if len(bad) >= sample_limit:
                break
    if total == 0:
        bad.append(f"{path}: no records")
    return {
        "path": str(path),
        "records": total,
        "ok": not bad,
        "bad_samples": bad[:sample_limit],
        "min_chars": min(char_counts) if char_counts else 0,
        "max_chars": max(char_counts) if char_counts else 0,
        "avg_chars": round(sum(char_counts) / len(char_counts), 2) if char_counts else 0,
        "min_newlines": min(newline_counts) if newline_counts else 0,
        "max_newlines": max(newline_counts) if newline_counts else 0,
        "avg_newlines": round(sum(newline_counts) / len(newline_counts), 2) if newline_counts else 0,
        "sources": source_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Mamba-3 raw document-continuation JSONL splits.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--sample-limit", type=int, default=10)
    args = parser.parse_args()
    reports = [validate(path, args.sample_limit) for path in args.paths]
    payload = {"ok": all(report["ok"] for report in reports), "reports": reports}
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
