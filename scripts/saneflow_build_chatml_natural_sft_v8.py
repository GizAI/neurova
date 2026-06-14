#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from saneflow_build_short_sft_v4 import SYSTEM, add_from_jsonl, chatml, write_jsonl


def make_row(user: str, assistant: str, source: str) -> dict[str, Any]:
    text = chatml(user, assistant, SYSTEM)
    key = " ".join(text.split())
    return {"text": text, "source": source, "dedup_hash": hashlib.sha256(key.encode("utf-8")).hexdigest()}


def add_many(rows: list[dict[str, Any]], pairs: list[tuple[str, str]], source: str, repeats: int) -> None:
    for _ in range(repeats):
        for user, assistant in pairs:
            rows.append(make_row(user, assistant, source))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build natural ChatML SFT v8 for SaneFlow.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/corpus/mixes/saneflow_chatml_natural_sft_v8"))
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--valid-size", type=int, default=1000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []

    dialogue = [
        ("hi", "Hello. I am Neurova. How can I help?"),
        ("hello", "Hello. I am Neurova. What would you like to do?"),
        ("Who are you?", "I am Neurova, a concise local assistant that answers questions directly."),
        ("What is your name?", "My name is Neurova."),
        ("Introduce yourself briefly.", "I am Neurova, a local assistant for short explanations and simple reasoning."),
        ("What can you do?", "I can answer questions, explain ideas, summarize text, and help with simple reasoning."),
        ("What should you do if you do not know?", "I should say that I do not know instead of inventing an answer."),
        ("Please answer briefly.", "Understood. I will keep the answer brief and direct."),
    ]
    explanations = [
        ("What is a computer?", "A computer is a machine that stores and processes information by following instructions."),
        ("What is machine learning?", "Machine learning is a way for computers to learn patterns from data and use them for predictions or decisions."),
        ("What is water?", "Water is a clear liquid made of hydrogen and oxygen, and living things need it to survive."),
        ("Explain why the sky looks blue.", "The sky looks blue because air scatters blue light from the sun more than many other colors."),
        ("What is the moon?", "The moon is Earth's natural satellite. It reflects sunlight and orbits Earth."),
        ("What is a book?", "A book is a collection of written pages that stores stories, explanations, or information."),
        ("What is a city?", "A city is a large human settlement with homes, roads, services, and many connected activities."),
    ]
    add_many(rows, dialogue, "natural-v8-dialogue", repeats=220)
    add_many(rows, explanations, "natural-v8-explanation", repeats=180)

    reports.append(
        add_from_jsonl(
            rows,
            Path("data/luma_clean_chatml_sft_v1.jsonl"),
            "local-clean",
            limit=5000,
            max_user=300,
            max_assistant=420,
            seed=args.seed + 1,
        )
    )

    names = ["Alice", "Ben", "Cara", "Dina", "Evan", "Liam", "Mina", "Noah", "Omar", "Rina", "Tom", "Sam", "Leo"]
    objects = ["apples", "books", "coins", "pencils", "marbles", "cards", "oranges", "stickers"]
    for _ in range(2500):
        name = rng.choice(names)
        obj = rng.choice(objects)
        a = rng.randint(1, 30)
        b = rng.randint(1, 30)
        rows.append(make_row(f"If {name} has {a} {obj} and gets {b} more, how many {obj} does {name} have?", f"{name} has {a + b} {obj}.", "natural-v8-arithmetic"))
        x, y = rng.sample(range(1, 100), 2)
        rows.append(make_row(f"Which is larger, {x} or {y}?", f"{max(x, y)} is larger.", "natural-v8-comparison"))
    for _ in range(900):
        code = f"{rng.choice(['AX','BK','CX','NV','QZ'])}-{rng.randint(100,999)}"
        rows.append(make_row(f"Remember this code: {code}. What is the code?", f"The code is {code}.", "natural-v8-copy"))

    # Keep deliberate oversampling. This dataset is not a broad corpus; it is a
    # short SFT curriculum that must repeatedly show stable dialogue openings and
    # natural sentence-shaped answers. Exact dedup here removes the anchors and
    # recreates the short-answer collapse that v7 exposed.
    rng.shuffle(rows)
    valid_size = min(args.valid_size, max(1, len(rows) // 10))
    valid = rows[:valid_size]
    train = rows[valid_size:]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "valid.jsonl", valid)
    manifest = {
        "name": "saneflow_chatml_natural_sft_v8",
        "format": "chatml",
        "loss_mode": "chatml_assistant",
        "dataset_layout": "sample",
        "goal": "natural dialogue first; copy/math are phrased as sentences to avoid single-token collapse",
        "train_records": len(train),
        "valid_records": len(valid),
        "reports": reports,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
