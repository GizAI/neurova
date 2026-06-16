#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def stable_hash(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def clean_text(text: str, *, min_chars: int, max_chars: int) -> str | None:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = text.strip()
    if len(text) < min_chars or len(text) > max_chars:
        return None
    if "<|im_start|>" in text or "<|im_end|>" in text:
        return None
    alpha = sum(ch.isalpha() for ch in text)
    if alpha < max(40, int(len(text) * 0.30)):
        return None
    return text


def get_field(row: dict[str, Any], field: str) -> str:
    if field in row:
        return as_text(row[field])
    for fallback in ("text", "content", "raw_content", "document"):
        if fallback in row:
            return as_text(row[fallback])
    return ""


def write_record(fh, *, text: str, source: dict[str, Any], dedup_hash: str) -> int:
    record = {
        "text": text,
        "source": f"{source['dataset']}/{source.get('config') or 'default'}/{source.get('split', 'train')}",
        "dataset": source["name"],
        "language": source.get("language", ""),
        "domain": source.get("domain", "web"),
        "dedup_hash": dedup_hash,
    }
    encoded = json.dumps(record, ensure_ascii=False)
    fh.write(encoded + "\n")
    return len(encoded.encode("utf-8"))


def load_stream(source: dict[str, Any]):
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "path": source["dataset"],
        "split": source.get("split", "train"),
        "streaming": True,
    }
    if source.get("config"):
        kwargs["name"] = source["config"]
    ds = load_dataset(**kwargs)
    text_field = source.get("text_field", "text")
    keep = [col for col in getattr(ds, "column_names", []) if col == text_field or col in {"text", "content", "raw_content", "document"}]
    if keep:
        try:
            ds = ds.select_columns(keep)
        except Exception:
            pass
    return ds


def download_source(source: dict[str, Any], *, min_chars: int, max_chars: int, valid_every: int) -> dict[str, Any]:
    out_dir = Path(source["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "valid.jsonl"
    manifest_path = out_dir / "manifest.json"
    max_train_bytes = int(source["max_train_bytes"])
    max_valid_bytes = int(source["max_valid_bytes"])
    text_field = source.get("text_field", "text")

    seen: set[str] = set()
    train_bytes = valid_bytes = 0
    train_rows = valid_rows = 0
    scanned = accepted = 0

    ds = load_stream(source)
    with train_path.open("w", encoding="utf-8") as train_f, valid_path.open("w", encoding="utf-8") as valid_f:
        for row in ds:
            if train_bytes >= max_train_bytes and valid_bytes >= max_valid_bytes:
                break
            scanned += 1
            try:
                text = clean_text(get_field(row, text_field), min_chars=min_chars, max_chars=max_chars)
            except Exception:
                continue
            if not text:
                continue
            digest = stable_hash(text)
            if digest in seen:
                continue
            seen.add(digest)
            accepted += 1
            if accepted % valid_every == 0 and valid_bytes < max_valid_bytes:
                valid_bytes += write_record(valid_f, text=text, source=source, dedup_hash=digest)
                valid_rows += 1
            elif train_bytes < max_train_bytes:
                train_bytes += write_record(train_f, text=text, source=source, dedup_hash=digest)
                train_rows += 1

    manifest = {
        "name": source["name"],
        "dataset": source["dataset"],
        "config": source.get("config"),
        "split": source.get("split", "train"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train": {
            "path": str(train_path),
            "rows": train_rows,
            "bytes": train_path.stat().st_size,
            "sha256": sha256_file(train_path),
        },
        "valid": {
            "path": str(valid_path),
            "rows": valid_rows,
            "bytes": valid_path.stat().st_size,
            "sha256": sha256_file(valid_path),
        },
        "filters": {
            "min_chars": min_chars,
            "max_chars": max_chars,
            "dedup": "normalized_text_sha256",
            "valid_every": valid_every,
        },
        "scanned": scanned,
        "accepted": accepted,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download bounded practical pretrain sources for SaneFlow.")
    parser.add_argument("--recipe", type=Path, default=Path("saneflow/configs/saneflow_practical_pretrain_sources.json"))
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument("--valid-every", type=int, default=97)
    args = parser.parse_args()

    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    selected = set(args.only or [])
    manifests = []
    for source in recipe["sources"]:
        if not source.get("enabled", True):
            continue
        if selected and source["name"] not in selected:
            continue
        print(f"== {source['name']} ==", flush=True)
        manifest = download_source(source, min_chars=args.min_chars, max_chars=args.max_chars, valid_every=args.valid_every)
        manifests.append(manifest)
        print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)

    out = Path("saneflow/data/corpus/sources/practical_pretrain_manifest_v1.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"recipe": str(args.recipe), "sources": manifests}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
