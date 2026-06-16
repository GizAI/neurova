#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def strip_answer_marker(text: str) -> str:
    marker = "Answer:"
    if marker in text:
        return text.rsplit(marker, 1)[1].strip()
    return text.strip()


def normalize_record(obj, record_format: str) -> str:
    if isinstance(obj, str):
        text = obj.strip()
    if isinstance(obj, dict):
        for key in ("text", "content", "prompt", "input", "output", "response", "completion"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                text = value.strip()
                break
        else:
            text = ""
        if isinstance(obj.get("messages"), list):
            parts = []
            for message in obj["messages"]:
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    role = message.get("role", "user")
                    parts.append(f"{role}: {message['content'].strip()}")
            if parts:
                text = "\n".join(parts)
        if not text:
            text = json.dumps(obj, ensure_ascii=False)
    if record_format == "answer":
        return strip_answer_marker(text)
    if record_format == "qa":
        return text.replace("Instruction:", "Question:", 1).strip()
    return text.strip()


def read_records(paths: list[Path], record_format: str) -> list[str]:
    records: list[str] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    text = normalize_record(line, record_format)
                else:
                    text = normalize_record(obj, record_format)
                if text:
                    records.append(text)
    return records


def sample(records: list[str], limit: int, rng: random.Random, zero_means_all: bool) -> list[str]:
    if limit == 0 and not zero_means_all:
        return []
    if limit <= 0 or len(records) <= limit:
        return list(records)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    return [records[i] for i in indices[:limit]]


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic natural/programmatic SFT mix.")
    parser.add_argument("--natural-inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--programmatic-inputs", nargs="*", type=Path, default=[])
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--valid-out", type=Path, required=True)
    parser.add_argument("--natural-max-records", type=int, default=0)
    parser.add_argument("--programmatic-max-records", type=int, default=400)
    parser.add_argument("--natural-format", choices=["raw", "answer", "qa"], default="answer")
    parser.add_argument("--programmatic-format", choices=["raw", "answer", "qa"], default="qa")
    parser.add_argument("--valid-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    natural = sample(
        read_records(args.natural_inputs, args.natural_format),
        args.natural_max_records,
        rng,
        zero_means_all=True,
    )
    programmatic = (
        sample(
            read_records(args.programmatic_inputs, args.programmatic_format),
            args.programmatic_max_records,
            rng,
            zero_means_all=False,
        )
        if args.programmatic_inputs
        else []
    )
    tagged = [(text, "natural") for text in natural] + [(text, "programmatic") for text in programmatic]
    rng.shuffle(tagged)
    lines = [text for text, _ in tagged]

    valid_count = max(1, int(len(lines) * args.valid_ratio)) if len(lines) > 1 else 0
    valid = lines[:valid_count]
    train = lines[valid_count:]
    write_lines(args.train_out, train)
    write_lines(args.valid_out, valid)
    print(json.dumps({
        "train_out": str(args.train_out),
        "valid_out": str(args.valid_out),
        "train_records": len(train),
        "valid_records": len(valid),
        "natural_records": len(natural),
        "programmatic_records": len(programmatic),
        "programmatic_share": round(len(programmatic) / max(1, len(natural) + len(programmatic)), 4),
        "natural_format": args.natural_format,
        "programmatic_format": args.programmatic_format,
        "seed": args.seed,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
