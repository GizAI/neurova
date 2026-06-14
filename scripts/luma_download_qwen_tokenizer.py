#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download tokenizer-only files for the LUMA Qwen BBPE front-end.")
    parser.add_argument("--repo", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--out", default="tokenizers/qwen35")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo,
        local_dir=str(out),
        allow_patterns=[
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "*.tiktoken",
        ],
    )
    meta = {"repo": args.repo, "path": str(path)}
    tokenizer_json = out / "tokenizer.json"
    if tokenizer_json.exists():
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_json))
        meta["vocab_size"] = tokenizer.get_vocab_size()
    (out / "neurova_tokenizer_source.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
