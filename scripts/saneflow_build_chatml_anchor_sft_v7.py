#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from saneflow_build_short_sft_v4 import SYSTEM, chatml, write_jsonl


def row(user: str, assistant: str, source: str, idx: int) -> dict[str, Any]:
    text = chatml(user, assistant, SYSTEM)
    # Keep intentional duplicates for anchor oversampling by salting the hash.
    key = f"{idx}:{source}:{text}"
    return {"text": text, "source": source, "dedup_hash": hashlib.sha256(key.encode("utf-8")).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build anchor-heavy ChatML SFT for immediate dialogue recovery.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/corpus/mixes/saneflow_chatml_anchor_sft_v7"))
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--valid-size", type=int, default=800)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    idx = 0

    anchors = [
        ("hi", "Hello. I am Neurova. How can I help?"),
        ("hello", "Hello. I am Neurova. What would you like to do?"),
        ("hey", "Hi. I am Neurova. How can I help?"),
        ("Who are you?", "I am Neurova, a local assistant that answers clearly and directly."),
        ("What is your name?", "My name is Neurova."),
        ("Introduce yourself.", "I am Neurova, a concise local assistant."),
        ("What can you do?", "I can answer questions, explain ideas, and help with simple reasoning."),
        ("What is a computer?", "A computer is a machine that stores and processes information by following instructions."),
        ("What is machine learning?", "Machine learning is a way for computers to learn patterns from data."),
        ("Explain why the sky looks blue.", "The sky looks blue because air scatters blue sunlight more strongly than many other colors."),
        ("What should you do if you do not know?", "I should say that I do not know instead of guessing."),
        ("Answer briefly: what is water?", "Water is a liquid made of hydrogen and oxygen."),
        ("Say one clear sentence about the moon.", "The moon is Earth's natural satellite."),
    ]
    for repeat in range(700):
        for user, assistant in anchors:
            rows.append(row(user, assistant, "anchor-v7-dialogue", idx))
            idx += 1

    names = ["Alice", "Ben", "Cara", "Dina", "Evan", "Liam", "Mina", "Noah", "Omar", "Rina", "Tom", "Sam", "Leo"]
    objects = ["apples", "books", "coins", "pencils", "marbles", "cards", "oranges", "stickers"]
    for _ in range(5000):
        name = rng.choice(names)
        obj = rng.choice(objects)
        a = rng.randint(1, 30)
        b = rng.randint(1, 30)
        rows.append(row(f"If {name} has {a} {obj} and gets {b} more, how many {obj} does {name} have?", str(a + b), "anchor-v7-arithmetic", idx))
        idx += 1
        x, y = rng.sample(range(1, 100), 2)
        rows.append(row(f"Which is larger, {x} or {y}?", str(max(x, y)), "anchor-v7-comparison", idx))
        idx += 1
        code = f"{rng.choice(['AX','BK','CX','NV','QZ'])}-{rng.randint(100,999)}"
        rows.append(row(f"Remember this code: {code}. What is the code?", code, "anchor-v7-copy", idx))
        idx += 1

    rng.shuffle(rows)
    valid_size = min(args.valid_size, len(rows) // 10)
    valid = rows[:valid_size]
    train = rows[valid_size:]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "valid.jsonl", valid)
    manifest = {
        "name": "saneflow_chatml_anchor_sft_v7",
        "format": "chatml",
        "loss_mode": "chatml_assistant",
        "goal": "force stable basic dialogue and short QA after ChatML format adaptation",
        "train_records": len(train),
        "valid_records": len(valid),
        "sources": {
            "dialogue_anchor_repeats": 700,
            "arithmetic_comparison_copy_samples": 5000,
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
