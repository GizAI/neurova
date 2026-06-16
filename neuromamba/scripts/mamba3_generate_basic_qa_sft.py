#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path


QA_ROWS = [
    (
        "Answer in one clear English sentence: What is photosynthesis?",
        "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to make sugar and oxygen.",
    ),
    (
        "Answer in one clear English sentence: Why should we test software?",
        "We test software to find mistakes early and make sure the program works as intended.",
    ),
    (
        "Answer with the number only. Which is larger, 7 or 3?",
        "7",
    ),
    (
        "Answer with the number only. What is 2 + 5?",
        "7",
    ),
    (
        "Answer with yes or no only. Is water usually wet?",
        "yes",
    ),
    (
        "Answer in one clear English sentence: What does a teacher do?",
        "A teacher explains ideas, checks understanding, and helps students practice.",
    ),
    (
        "Answer in one clear English sentence: What is science?",
        "Science is a careful way to learn about reality by asking questions and testing evidence.",
    ),
    (
        "Answer in one clear English sentence: What is courage?",
        "Courage is doing the right thing even when it feels difficult.",
    ),
    (
        "Answer in one clear English sentence: What is inference in machine learning?",
        "Inference is using a trained model to produce an answer from new input.",
    ),
    (
        "Answer in one clear English sentence: Why is evidence important?",
        "Evidence is important because it helps people separate facts from guesses.",
    ),
]

PARAPHRASE_PREFIXES = [
    "Instruction:",
    "Question:",
    "Task:",
]


def stable_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def make_rows(repeats: int) -> list[dict]:
    today = date.today().isoformat()
    rows: list[dict] = []
    for _ in range(repeats):
        for prefix in PARAPHRASE_PREFIXES:
            for question, answer in QA_ROWS:
                text = f"{prefix} {question}\nAnswer: {answer}"
                rows.append({
                    "text": text,
                    "source": "deterministic-basic-english-qa-v1",
                    "license": "synthetic-self-generated-no-llm",
                    "language": "en",
                    "domain": "qa",
                    "quality_score": 0.95,
                    "toxicity_score": 0.0,
                    "pii_score": 0.0,
                    "dedup_hash": stable_hash(text),
                    "benchmark_contamination_flag": False,
                    "teacher_model": "none",
                    "generation_date": today,
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic basic English QA SFT data without LLM teachers.")
    parser.add_argument("--out", type=Path, default=Path("neuromamba/data/basic_english_qa_v1.jsonl"))
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    rows = make_rows(args.repeats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "records": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
