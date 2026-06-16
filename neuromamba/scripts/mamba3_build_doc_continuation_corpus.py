#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any


TEXT_KEYS = ("text", "content", "body", "article", "abstract")
INSTRUCTION_MARKERS = (
    "Instruction:",
    "Question:",
    "Task:",
    "\nAnswer:",
    "User:",
    "Assistant:",
    "<|user|>",
    "<|assistant|>",
)


def compact_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def text_from_record(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return ""
    for key in TEXT_KEYS:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def should_skip(text: str, *, skip_instruction_like: bool, min_chars: int) -> bool:
    if len(text) < min_chars:
        return True
    if skip_instruction_like and any(marker in text for marker in INSTRUCTION_MARKERS):
        return True
    return False


def source_for(obj: Any, fallback: str) -> str:
    if isinstance(obj, dict):
        for key in ("source", "dataset", "repo", "license"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return fallback


def domain_for(obj: Any) -> str:
    if isinstance(obj, dict):
        value = obj.get("domain")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "document"


def wrap_document(text: str, source: str, domain: str) -> str:
    source = html.escape(source, quote=True)
    domain = html.escape(domain, quote=True)
    return f'<doc source="{source}" domain="{domain}">\n{text}\n</doc>'


def iter_records(path: Path):
    fallback = path.stem
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    obj = raw
                yield obj, fallback
    else:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                raw = line.strip()
                if raw:
                    yield raw, fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Build raw document-continuation corpus for early Mamba-3 pretraining.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--allow-instruction-like", action="store_true")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_short_or_instruction = 0
    per_source: dict[str, int] = {}
    with args.out.open("w", encoding="utf-8") as out:
        for path in args.inputs:
            if not path.exists():
                raise FileNotFoundError(path)
            for obj, fallback in iter_records(path):
                text = compact_text(text_from_record(obj))
                if should_skip(
                    text,
                    skip_instruction_like=not args.allow_instruction_like,
                    min_chars=args.min_chars,
                ):
                    skipped_short_or_instruction += 1
                    continue
                source = source_for(obj, fallback)
                domain = domain_for(obj)
                row = {
                    "text": wrap_document(text, source, domain),
                    "source": source,
                    "domain": domain,
                    "format": "doc-continuation-v1",
                }
                out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                written += 1
                per_source[source] = per_source.get(source, 0) + 1
                if args.max_records > 0 and written >= args.max_records:
                    break
            if args.max_records > 0 and written >= args.max_records:
                break

    print(json.dumps({
        "out": str(args.out),
        "written": written,
        "skipped_short_or_instruction": skipped_short_or_instruction,
        "min_chars": args.min_chars,
        "format": "doc-continuation-v1",
        "sources": per_source,
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
