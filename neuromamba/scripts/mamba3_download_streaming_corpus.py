#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import date
from pathlib import Path


def stable_hash(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


def as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download/stream governed corpus shards for Mamba-3.")
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset id")
    parser.add_argument("--config", default=None, help="Optional dataset config/name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", default=None)
    parser.add_argument("--license", default="terms_required")
    parser.add_argument("--domain", default="web")
    parser.add_argument("--quality-score", type=float, default=0.7)
    parser.add_argument("--max-docs", type=int, default=10000)
    parser.add_argument("--max-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    from datasets import load_dataset

    load_kwargs = {
        "path": args.dataset,
        "split": args.split,
        "streaming": True,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.config:
        load_kwargs["name"] = args.config
    dataset = load_dataset(**load_kwargs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    source = args.source or (args.dataset if args.config is None else f"{args.dataset}/{args.config}")
    total_bytes = 0
    written = 0
    seen_hashes: set[str] = set()
    today = date.today().isoformat()
    with args.out.open("w", encoding="utf-8") as fh:
        for row in dataset:
            text = as_text(row.get(args.text_field)).strip()
            if len(text) < args.min_chars:
                continue
            digest = stable_hash(text)
            if digest in seen_hashes:
                continue
            encoded_size = len(text.encode("utf-8", errors="ignore"))
            if total_bytes + encoded_size > args.max_bytes:
                break
            seen_hashes.add(digest)
            record = {
                "text": text,
                "source": source,
                "license": args.license,
                "language": "en",
                "domain": args.domain,
                "quality_score": args.quality_score,
                "toxicity_score": 0.0,
                "pii_score": 0.0,
                "dedup_hash": digest,
                "benchmark_contamination_flag": False,
                "teacher_model": "none",
                "generation_date": today,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_bytes += encoded_size
            written += 1
            if written >= args.max_docs:
                break
    print(json.dumps({
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "out": str(args.out),
        "documents": written,
        "text_bytes": total_bytes,
    }, ensure_ascii=False, indent=2))
    sys.stdout.flush()
    # Some streaming backends keep retry/finalizer threads alive and can abort
    # during interpreter teardown after all requested records are already written.
    os._exit(0)


if __name__ == "__main__":
    main()
