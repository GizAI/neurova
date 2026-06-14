#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXAMPLES = [
    ("hi", "Hello. I am Neurova, a local SaneFlow language model. How can I help?"),
    ("hello", "Hello. I am Neurova. How can I help you today?"),
    ("who are you?", "I am Neurova, a local experimental assistant powered by SaneFlow."),
    ("what are you?", "I am a small from-scratch language model designed to answer clearly."),
    ("Explain what a computer is in simple words.", "A computer is a machine that follows instructions to store, process, and show information."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Write one sentence about the moon.", "The moon is a bright natural satellite that orbits Earth."),
    ("What is machine learning?", "Machine learning is a way for computers to learn patterns from data and use them to make predictions."),
    ("Give a short answer: what is water?", "Water is a clear liquid made of hydrogen and oxygen that living things need."),
    ("If you do not know something, what should you do?", "I should say that I do not know instead of inventing an answer."),
]


def digest(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description="Build a clean SaneFlow chat seed without legacy model names.")
    p.add_argument("--out", type=Path, default=Path("data/corpus/mixes/saneflow_chat_sft_seed_v1.jsonl"))
    p.add_argument("--repeat", type=int, default=200)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for i in range(args.repeat):
            for q, a in EXAMPLES:
                text = f"Question: {q}\nAnswer: {a}"
                rec = {
                    "text": text,
                    "source": "saneflow-chat-seed-v1",
                    "role": "minimal_speaking_sft",
                    "dedup_hash": digest(text),
                    "repeat": i,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(args.out), "records": len(EXAMPLES) * args.repeat}, indent=2))


if __name__ == "__main__":
    main()
