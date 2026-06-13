#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SUBJECTS = [
    "clear thinking",
    "careful practice",
    "good research",
    "debugging",
    "teamwork",
    "scientific reasoning",
    "software quality",
    "patient learning",
    "honest communication",
    "reliable systems",
]

VERBS = [
    "works best when",
    "improves when",
    "becomes useful when",
    "is easier when",
    "gets stronger when",
]

CONDITIONS = [
    "the goal is specific and the result can be checked",
    "small steps are tested before large changes are made",
    "evidence is separated from guesses",
    "the important detail is explained before the extra detail",
    "errors are reproduced and inspected one at a time",
    "people share context and make decisions from facts",
    "the system avoids repeated work and measures the expensive path",
    "the answer says what is known and what is still uncertain",
]

GATE_SENTENCES = [
    "The main idea is to choose one goal, test it carefully, and improve the result.",
    "The main idea is that careful work turns a rough plan into a useful result.",
    "The main idea is that clear thinking makes difficult problems easier to solve.",
    "A good teacher should make hard ideas clear without hiding the important details.",
    "A good teacher should explain ideas simply, check understanding, and help students practice.",
    "A good teacher should listen carefully and give examples that students can use.",
    "In simple words, science is how people learn which ideas match reality.",
    "In simple words, science is a careful way to ask questions and test answers about the world.",
    "In simple words, science is organized curiosity supported by evidence.",
    "Courage is the choice to do the right thing even when the situation feels difficult.",
    "Courage means acting with honesty even when fear is present.",
    "Courage grows when a person keeps their values under pressure.",
]

DEFINITIONS = {
    "A model": "a simplified system that learns patterns from data and uses them to make predictions",
    "Inference": "the process of using a trained model to produce an answer from new input",
    "Training": "the process of adjusting a model so its predictions become more accurate",
    "A tokenizer": "a tool that turns text into numbers that a language model can process",
    "Memory": "the place where a system keeps information it may need later",
    "Quality": "the result of clear requirements, careful implementation, and useful tests",
    "Focus": "the habit of keeping attention on the next important action",
    "Patience": "the ability to keep acting calmly while waiting for a result",
}


def make_records(records: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []
    for sentence in GATE_SENTENCES:
        out.append({"text": sentence, "kind": "gate"})
    keys = list(DEFINITIONS)
    while len(out) < records:
        choice = rng.randrange(5)
        if choice == 0:
            subject = rng.choice(SUBJECTS)
            verb = rng.choice(VERBS)
            condition = rng.choice(CONDITIONS)
            text = f"{subject.capitalize()} {verb} {condition}."
        elif choice == 1:
            key = rng.choice(keys)
            text = f"{key} is {DEFINITIONS[key]}."
        elif choice == 2:
            a = rng.choice(CONDITIONS)
            b = rng.choice([item for item in CONDITIONS if item != a])
            text = f"A useful answer states the main point first, then checks whether {a}, and finally explains how {b}."
        elif choice == 3:
            text = (
                f"When solving a problem, start by naming the goal, inspect the evidence, "
                f"change one thing, and verify the result."
            )
        else:
            text = (
                f"Good language is accurate, direct, and practical; it helps the reader act without guessing."
            )
        out.append({"text": text, "kind": "deterministic_clean_english"})
    rng.shuffle(out)
    return out[:records]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic clean English SFT text without LLM teachers.")
    parser.add_argument("--out", type=Path, default=Path("data/clean_english_sft_bootstrap.jsonl"))
    parser.add_argument("--records", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260613)
    args = parser.parse_args()

    rows = make_records(args.records, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            row["metadata"] = {
                "source": "deterministic-clean-english-sft",
                "license": "synthetic-self-generated-no-llm",
                "teacher_model": None,
            }
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "records": len(rows), "seed": args.seed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
