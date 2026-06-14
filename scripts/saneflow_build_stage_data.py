#!/usr/bin/env python3
"""Build deterministic SaneFlow stage datasets without transformers.

The output format is JSONL with a `text` field plus lightweight provenance.
This script intentionally keeps Stage C as raw continuation data. ChatML and
Answer-heavy templates are filtered out unless the caller explicitly passes a
plain QA source with a small ratio.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


BAD_PATTERNS = (
    "<|im_start|>",
    "<|im_end|>",
    "Instruction:",
    "Answer:",
    "Question:",
    "A.",
    "B.",
    "C.",
    "D.",
)


def read_jsonl_texts(path: Path, limit: int | None = None) -> list[str]:
    texts: list[str] = []
    if not path or not path.exists():
        return texts
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
            if limit and len(texts) >= limit:
                break
    return texts


def split_plain_text(path: Path, min_chars: int, max_chars: int) -> list[str]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = raw.replace("\r\n", "\n")
    paras = [p.strip() for p in re.split(r"\n\s*\n+", raw) if p.strip()]
    chunks: list[str] = []
    for para in paras:
        para = re.sub(r"\s+", " ", para).strip()
        if len(para) < min_chars:
            continue
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para)
        buf = ""
        for sent in sentences:
            cand = (buf + " " + sent).strip() if buf else sent
            if len(cand) <= max_chars:
                buf = cand
            else:
                if len(buf) >= min_chars:
                    chunks.append(buf)
                buf = sent
        if len(buf) >= min_chars:
            chunks.append(buf[:max_chars])
    return chunks


def clean_raw_continuation(text: str, min_chars: int, max_chars: int) -> str | None:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < min_chars or len(text) > max_chars:
        return None
    if any(p in text for p in BAD_PATTERNS):
        return None
    alpha = sum(ch.isalpha() for ch in text)
    if alpha < max(20, int(len(text) * 0.35)):
        return None
    return text


def sample_records(
    rng: random.Random,
    pool: list[str],
    n: int,
    source: str,
    role: str,
) -> list[dict[str, str]]:
    if n <= 0 or not pool:
        return []
    out = []
    for _ in range(n):
        out.append({"text": rng.choice(pool), "source": source, "role": role})
    return out


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_split(args: argparse.Namespace, split: str) -> list[dict[str, str]]:
    rng = random.Random(args.seed + (0 if split == "train" else 1))

    story_path = Path(args.tinystories_train if split == "train" else args.tinystories_valid)
    stories = read_jsonl_texts(story_path, args.max_story_source_records)
    stories = [
        t for t in stories
        if clean_raw_continuation(t, args.min_chars, args.max_chars) is not None
    ]

    prose_pool: list[str] = []
    for p in args.raw_prose:
        prose_pool.extend(split_plain_text(Path(p), args.min_chars, args.max_chars))
    for p in args.raw_jsonl:
        for text in read_jsonl_texts(Path(p), args.max_raw_jsonl_records):
            clean = clean_raw_continuation(text, args.min_chars, args.max_chars)
            if clean:
                prose_pool.append(clean)

    qa_pool: list[str] = []
    for p in args.plain_qa:
        for text in read_jsonl_texts(Path(p), args.max_plain_qa_records):
            if "<|im_" in text:
                continue
            if "Question:" in text and "Answer:" in text:
                qa_pool.append(re.sub(r"\s+", " ", text).strip()[: args.max_chars])

    total = args.train_records if split == "train" else args.valid_records
    story_n = int(total * args.story_ratio)
    prose_n = int(total * args.prose_ratio)
    qa_n = max(0, total - story_n - prose_n)
    if args.qa_ratio >= 0:
        qa_n = int(total * args.qa_ratio)
        story_n = int(total * args.story_ratio)
        prose_n = max(0, total - story_n - qa_n)

    records = []
    records.extend(sample_records(rng, stories, story_n, "tinystories", "raw_continuation"))
    records.extend(sample_records(rng, prose_pool, prose_n, "filtered_raw_prose", "raw_continuation"))
    records.extend(sample_records(rng, qa_pool, qa_n, "plain_qa_small", "plain_qa"))
    rng.shuffle(records)
    return records[:total]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tinystories-train", required=True)
    parser.add_argument("--tinystories-valid", required=True)
    parser.add_argument("--raw-prose", action="append", default=[])
    parser.add_argument("--raw-jsonl", action="append", default=[])
    parser.add_argument("--plain-qa", action="append", default=[])
    parser.add_argument("--train-records", type=int, default=50000)
    parser.add_argument("--valid-records", type=int, default=2000)
    parser.add_argument("--story-ratio", type=float, default=0.78)
    parser.add_argument("--prose-ratio", type=float, default=0.20)
    parser.add_argument("--qa-ratio", type=float, default=0.02)
    parser.add_argument("--min-chars", type=int, default=120)
    parser.add_argument("--max-chars", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--max-story-source-records", type=int, default=0)
    parser.add_argument("--max-raw-jsonl-records", type=int, default=200000)
    parser.add_argument("--max-plain-qa-records", type=int, default=20000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    train = build_split(args, "train")
    valid = build_split(args, "valid")
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)

    manifest = {
        "name": out_dir.name,
        "train_records": len(train),
        "valid_records": len(valid),
        "ratios": {
            "story": args.story_ratio,
            "prose": args.prose_ratio,
            "qa": args.qa_ratio,
        },
        "sources": {
            "tinystories_train": args.tinystories_train,
            "tinystories_valid": args.tinystories_valid,
            "raw_prose": args.raw_prose,
            "raw_jsonl": args.raw_jsonl,
            "plain_qa": args.plain_qa,
        },
        "policy": "Stage C raw/explanation continuation; no ChatML; QA capped small.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
