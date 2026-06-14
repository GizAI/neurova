#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
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


def write_record(fh, text: str, source: str, dedup_hash: str) -> int:
    record = {
        "text": text,
        "source": source,
        "language": "en",
        "domain": "educational_web",
        "dedup_hash": dedup_hash,
    }
    encoded = json.dumps(record, ensure_ascii=False)
    fh.write(encoded + "\n")
    return len(encoded.encode("utf-8"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download a bounded FineWeb-Edu subset for SaneFlow.")
    p.add_argument("--out-dir", type=Path, default=Path("data/corpus/sources/fineweb_edu_sample10bt"))
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--config", default="sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--max-train-bytes", type=int, default=200_000_000)
    p.add_argument("--max-valid-bytes", type=int, default=5_000_000)
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--valid-every", type=int, default=97)
    p.add_argument("--trust-remote-code", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.jsonl"
    valid_path = args.out_dir / "valid.jsonl"
    manifest_path = args.out_dir / "manifest.json"

    load_kwargs = {
        "path": args.dataset,
        "name": args.config,
        "split": args.split,
        "streaming": True,
        "trust_remote_code": args.trust_remote_code,
    }
    ds = load_dataset(**load_kwargs)

    source = f"{args.dataset}/{args.config}/{args.split}"
    seen: set[str] = set()
    train_bytes = 0
    valid_bytes = 0
    train_rows = 0
    valid_rows = 0
    scanned = 0

    with train_path.open("w", encoding="utf-8") as train_f, valid_path.open("w", encoding="utf-8") as valid_f:
        for row in ds:
            if train_bytes >= args.max_train_bytes and valid_bytes >= args.max_valid_bytes:
                break
            text = as_text(row.get(args.text_field)).strip()
            if len(text) < args.min_chars:
                continue
            digest = stable_hash(text)
            if digest in seen:
                continue
            seen.add(digest)
            scanned += 1
            if scanned % args.valid_every == 0 and valid_bytes < args.max_valid_bytes:
                valid_bytes += write_record(valid_f, text, source, digest)
                valid_rows += 1
            elif train_bytes < args.max_train_bytes:
                train_bytes += write_record(train_f, text, source, digest)
                train_rows += 1

    manifest = {
        "name": "fineweb_edu_sample10bt",
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
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
            "min_chars": args.min_chars,
            "dedup": "normalized_text_sha256",
            "valid_every": args.valid_every,
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
