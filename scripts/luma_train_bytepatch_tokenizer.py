#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from luma.tokenizer import AdaptiveBytePatchTokenizer, learn_bytepatch_vocab, write_bytepatch_vocab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn a corpus-adaptive byte-latent patch vocabulary for LUMA.")
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--out", default="tokenizers/luma_bytepatch/bytepatch_vocab.json")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-patches", type=int, default=8192)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=12)
    return parser.parse_args()


def read_text_record(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
        for key in ("text", "content", "completion", "answer", "prompt"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value
        if isinstance(obj.get("messages"), list):
            parts = []
            for msg in obj["messages"]:
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    parts.append(msg["content"])
            return "\n".join(parts)
    return stripped


def load_texts(paths: list[str], max_records: int) -> list[str]:
    texts: list[str] = []
    for item in paths:
        path = Path(item)
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                text = read_text_record(line)
                if not text:
                    continue
                texts.append(text)
                if max_records and len(texts) >= max_records:
                    return texts
    return texts


def main() -> None:
    args = parse_args()
    texts = load_texts(args.data, args.max_records)
    if not texts:
        raise SystemExit("no text records loaded")
    patches = learn_bytepatch_vocab(
        texts,
        max_patches=args.max_patches,
        min_count=args.min_count,
        min_len=args.min_len,
        max_len=args.max_len,
    )
    write_bytepatch_vocab(args.out, patches, source=",".join(args.data))

    base = AdaptiveBytePatchTokenizer(vocab_path="__missing_builtin_only__.json")
    learned = AdaptiveBytePatchTokenizer(vocab_path=args.out)
    total_bytes = sum(len(text.encode("utf-8", errors="replace")) for text in texts)
    base_tokens = sum(len(base.encode(text, add_bos=False, add_eos=False)) for text in texts)
    learned_tokens = sum(len(learned.encode(text, add_bos=False, add_eos=False)) for text in texts)
    report = {
        "records": len(texts),
        "bytes": total_bytes,
        "learned_patches": len(patches),
        "base_tokens": base_tokens,
        "learned_tokens": learned_tokens,
        "base_bytes_per_token": total_bytes / max(base_tokens, 1),
        "learned_bytes_per_token": total_bytes / max(learned_tokens, 1),
        "token_reduction": 1.0 - learned_tokens / max(base_tokens, 1),
        "out": args.out,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
