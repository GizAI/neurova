#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def estimate_tokens(text: str) -> int:
    # Conservative planning estimate for English BPE corpora.
    return max(1, int(len(text) / 4))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Mamba-3 corpus manifest with rough token estimates.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("neuromamba/data/mamba3_corpus_manifest.json"))
    parser.add_argument("--minimum-tokens", type=int, default=100_000_000_000)
    args = parser.parse_args()

    entries = []
    total_bytes = 0
    total_tokens = 0
    for path in args.paths:
        files = sorted(path.rglob("*")) if path.is_dir() else [path]
        for file in files:
            if not file.is_file():
                continue
            text = file.read_text(encoding="utf-8", errors="ignore")
            tokens = estimate_tokens(text)
            size = file.stat().st_size
            total_bytes += size
            total_tokens += tokens
            entries.append({
                "path": str(file),
                "bytes": size,
                "estimated_tokens": tokens,
            })

    manifest = {
        "entries": entries,
        "summary": {
            "files": len(entries),
            "bytes": total_bytes,
            "estimated_tokens": total_tokens,
            "minimum_target_tokens": args.minimum_tokens,
            "target_fraction": total_tokens / args.minimum_tokens if args.minimum_tokens else None,
            "status": "ready_for_real_pretraining" if total_tokens >= args.minimum_tokens else "bootstrap_only",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
