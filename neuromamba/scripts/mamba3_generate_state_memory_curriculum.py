#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import date
from pathlib import Path


NAMES = [
    "Ada", "Byron", "Clara", "Dijkstra", "Emmy", "Faraday", "Grace", "Hopper",
    "Iris", "Jules", "Kepler", "Lena", "Minsky", "Noether", "Turing", "Vera",
]
COLORS = ["red", "blue", "green", "yellow", "purple", "silver", "black", "white"]
TOOLS = ["compiler", "database", "router", "tokenizer", "kernel", "checkpoint", "verifier", "scheduler"]
COUNTRIES = ["Korea", "Japan", "France", "Canada", "Brazil", "India", "Germany", "Kenya"]


def stable_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def record(text: str, domain: str) -> dict:
    return {
        "text": text,
        "source": "deterministic-neurova-state-memory-v1",
        "license": "synthetic-self-generated-no-llm",
        "language": "en",
        "domain": domain,
        "quality_score": 0.97,
        "toxicity_score": 0.0,
        "pii_score": 0.0,
        "dedup_hash": stable_hash(text),
        "benchmark_contamination_flag": False,
        "teacher_model": "none",
        "generation_date": date.today().isoformat(),
    }


def phonebook(rng: random.Random) -> tuple[str, str]:
    pairs = rng.sample(NAMES, 5)
    facts = []
    answer_name = rng.choice(pairs)
    answer_color = rng.choice(COLORS)
    for name in pairs:
        color = answer_color if name == answer_name else rng.choice([c for c in COLORS if c != answer_color])
        facts.append(f"{name}'s code color is {color}.")
    rng.shuffle(facts)
    prompt = "Read the facts and answer only the requested value. " + " ".join(facts) + f" What is {answer_name}'s code color?"
    answer = answer_color
    return prompt, answer


def json_field(rng: random.Random) -> tuple[str, str]:
    owner = rng.choice(NAMES)
    tool = rng.choice(TOOLS)
    status = rng.choice(["ready", "blocked", "running", "paused"])
    payload = {"owner": owner, "tool": tool, "status": status, "priority": rng.randint(1, 9)}
    field = rng.choice(["owner", "tool", "status", "priority"])
    prompt = f"Given this JSON object, return the {field} field only: {json.dumps(payload, sort_keys=True)}"
    return prompt, str(payload[field])


def copy_span(rng: random.Random) -> tuple[str, str]:
    key = f"NX-{rng.randint(1000, 9999)}-{rng.choice(['ALPHA', 'BETA', 'DELTA', 'SIGMA'])}"
    distractors = [f"noise-{rng.randint(100, 999)}" for _ in range(5)]
    rng.shuffle(distractors)
    prompt = "Copy the exact access key from this sentence: " + " ".join(distractors[:2]) + f" access_key={key} " + " ".join(distractors[2:])
    return prompt, key


def state_summary(rng: random.Random) -> tuple[str, str]:
    person = rng.choice(NAMES)
    country = rng.choice(COUNTRIES)
    tool = rng.choice(TOOLS)
    prompt = (
        f"State memory update: {person} is working in {country}. "
        f"{person} is using the {tool}. Summarize the memory in one sentence."
    )
    answer = f"{person} is working in {country} and using the {tool}."
    return prompt, answer


def route_mode(rng: random.Random) -> tuple[str, str]:
    task, answer = rng.choice([
        ("The user asks for a definition.", "definition"),
        ("The user asks to copy an exact key.", "copy"),
        ("The user asks about missing evidence.", "uncertain"),
        ("The user asks for JSON field extraction.", "extraction"),
    ])
    prompt = f"Choose the correct route label only: {task} Labels: definition, copy, uncertain, extraction."
    return prompt, answer


def build_rows(target: int, seed: int, tasks: list[str] | None = None) -> list[dict]:
    rng = random.Random(seed)
    builders = [
        ("phonebook", phonebook),
        ("json_field", json_field),
        ("copy_span", copy_span),
        ("state_summary", state_summary),
        ("route_mode", route_mode),
    ]
    if tasks:
        wanted = set(tasks)
        builders = [(name, builder) for name, builder in builders if name in wanted]
        if not builders:
            raise ValueError(f"no known tasks selected: {sorted(wanted)}")
    rows: list[dict] = []
    seen: set[str] = set()
    attempts = 0
    while len(rows) < target and attempts < target * 20:
        attempts += 1
        domain, builder = rng.choice(builders)
        prompt, answer = builder(rng)
        template = rng.choice([
            "Instruction: {prompt}\nAnswer: {answer}",
            "Question: {prompt}\nAnswer: {answer}",
            "User: {prompt}\nAssistant: {answer}",
        ])
        text = template.format(prompt=prompt, answer=answer)
        key = stable_hash(text)
        if key in seen:
            continue
        seen.add(key)
        rows.append(record(text, domain))
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("neuromamba/data/neurova_state_memory_curriculum_v1.jsonl"))
    parser.add_argument("--records", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--tasks", default="", help="Optional comma-separated task subset.")
    args = parser.parse_args()

    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    rows = build_rows(args.records, args.seed, tasks or None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "records": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
