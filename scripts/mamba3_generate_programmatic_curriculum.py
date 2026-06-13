#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import string
from pathlib import Path


WORDS = [
    "amber", "brisk", "cedar", "delta", "ember", "fable", "glade", "harbor",
    "ion", "juniper", "keystone", "lumen", "mosaic", "nova", "onyx", "prairie",
    "quartz", "raven", "signal", "thicket", "umbra", "vector", "willow", "xenon",
    "yarrow", "zenith",
]


def rand_word(rng: random.Random) -> str:
    return rng.choice(WORDS) + rng.choice(string.ascii_lowercase)


def record(task: str, prompt: str, answer: str, **meta) -> dict:
    return {
        "text": f"Instruction: {prompt.strip()}\nAnswer: {answer.strip()}",
        "task": task,
        "answer": answer,
        "metadata": {
            "source": "programmatic",
            "license": "synthetic-self-generated-no-llm",
            "teacher_model": None,
            **meta,
        },
    }


def gen_copy(rng: random.Random) -> dict:
    items = [rand_word(rng) for _ in range(rng.randint(4, 10))]
    prompt = "Copy this sequence exactly, preserving spaces: " + " ".join(items)
    return record("copy", prompt, " ".join(items), length=len(items))


def gen_copy_easy(rng: random.Random) -> dict:
    items = [rand_word(rng) for _ in range(rng.randint(1, 3))]
    prompt = "Copy this sequence exactly: " + " ".join(items)
    return record("copy", prompt, " ".join(items), length=len(items), difficulty="easy")


def gen_phonebook(rng: random.Random) -> dict:
    pairs = [(rand_word(rng), f"{rng.randint(100, 999)}-{rng.randint(1000, 9999)}") for _ in range(rng.randint(4, 9))]
    key, value = rng.choice(pairs)
    book = "; ".join(f"{name}: {phone}" for name, phone in pairs)
    prompt = f"Phonebook: {book}. What is the phone number for {key}?"
    return record("phonebook_lookup", prompt, value, entries=len(pairs))


def gen_phonebook_easy(rng: random.Random) -> dict:
    pairs = [(rand_word(rng), f"{rng.randint(100, 999)}-{rng.randint(1000, 9999)}") for _ in range(2)]
    key, value = rng.choice(pairs)
    book = "; ".join(f"{name}: {phone}" for name, phone in pairs)
    prompt = f"Phonebook: {book}. Number for {key}?"
    return record("phonebook_lookup", prompt, value, entries=len(pairs), difficulty="easy")


def gen_joint_recall(rng: random.Random) -> dict:
    pairs = [(rand_word(rng), rng.choice(["red", "blue", "green", "silver", "black"])) for _ in range(rng.randint(5, 10))]
    left = rng.choice(pairs)
    right = rng.choice([item for item in pairs if item != left])
    facts = ". ".join(f"{name} has color {color}" for name, color in pairs)
    answer = "yes" if left[1] == right[1] else "no"
    prompt = f"{facts}. Do {left[0]} and {right[0]} have the same color? Answer yes or no."
    return record("joint_recall", prompt, answer, entries=len(pairs))


def gen_mc_routing(rng: random.Random) -> dict:
    a, b, c, d = [rng.randint(1, 30) for _ in range(4)]
    options = {"A": a, "B": b, "C": c, "D": d}
    correct = max(options, key=options.get)
    prompt = f"Choose the option with the largest number. A={a}, B={b}, C={c}, D={d}. Return only A, B, C, or D."
    return record("multiple_choice_routing", prompt, correct, options=options)


def gen_json_field(rng: random.Random) -> dict:
    obj = {
        "id": rng.randint(1000, 9999),
        "city": rng.choice(["Seoul", "Boston", "Nairobi", "Oslo", "Lima"]),
        "score": rng.randint(1, 100),
        "tag": rand_word(rng),
    }
    key = rng.choice(list(obj.keys()))
    prompt = f"Given JSON {json.dumps(obj, sort_keys=True)}, return the value of field `{key}` only."
    return record("json_field_extraction", prompt, str(obj[key]), key=key)


def gen_json_field_easy(rng: random.Random) -> dict:
    obj = {
        "id": rng.randint(10, 99),
        "tag": rand_word(rng),
    }
    key = rng.choice(list(obj.keys()))
    prompt = f"Given JSON {json.dumps(obj, sort_keys=True)}, return `{key}` only."
    return record("json_field_extraction", prompt, str(obj[key]), key=key, difficulty="easy")


def gen_code_trace(rng: random.Random) -> dict:
    x = rng.randint(1, 20)
    y = rng.randint(1, 20)
    z = x * 2 + y
    code = f"x = {x}\ny = {y}\nz = x * 2 + y"
    prompt = f"Trace this Python code and return z only:\n{code}"
    return record("code_variable_tracing", prompt, str(z), x=x, y=y)


def gen_arithmetic(rng: random.Random) -> dict:
    a, b, c = rng.randint(1, 50), rng.randint(1, 50), rng.randint(1, 20)
    answer = a + b * c
    prompt = f"Compute exactly: {a} + {b} * {c}. Return only the integer."
    return record("arithmetic", prompt, str(answer), a=a, b=b, c=c)


def gen_needle(rng: random.Random) -> dict:
    key = rand_word(rng).upper()
    value = rand_word(rng) + "-" + str(rng.randint(100, 999))
    distractors = [f"note {i}: {rand_word(rng)}" for i in range(rng.randint(12, 30))]
    insert_at = rng.randint(0, len(distractors))
    distractors.insert(insert_at, f"SECRET {key} = {value}")
    prompt = " ".join(distractors) + f" What is the value for SECRET {key}?"
    return record("needle_in_haystack", prompt, value, distractors=len(distractors) - 1)


def gen_needle_easy(rng: random.Random) -> dict:
    key = rand_word(rng).upper()
    value = rand_word(rng) + "-" + str(rng.randint(10, 99))
    distractors = [f"note {i}: {rand_word(rng)}" for i in range(4)]
    insert_at = rng.randint(0, len(distractors))
    distractors.insert(insert_at, f"SECRET {key} = {value}")
    prompt = " ".join(distractors) + f" Value for SECRET {key}?"
    return record("needle_in_haystack", prompt, value, distractors=len(distractors) - 1, difficulty="easy")


GENERATORS = [
    gen_copy,
    gen_phonebook,
    gen_joint_recall,
    gen_mc_routing,
    gen_json_field,
    gen_code_trace,
    gen_arithmetic,
    gen_needle,
]

GENERATOR_BY_TASK = {
    fn(random.Random(0))["task"]: fn
    for fn in GENERATORS
}

EASY_GENERATOR_BY_TASK = {
    "copy": gen_copy_easy,
    "phonebook_lookup": gen_phonebook_easy,
    "json_field_extraction": gen_json_field_easy,
    "needle_in_haystack": gen_needle_easy,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic no-teacher Mamba-3 weakness-correction curriculum.")
    parser.add_argument("--out", type=Path, default=Path("data/mamba3_programmatic_curriculum.jsonl"))
    parser.add_argument("--records", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument(
        "--tasks",
        default="",
        help="Optional comma-separated task filter, e.g. copy,phonebook_lookup,json_field_extraction,needle_in_haystack.",
    )
    parser.add_argument("--difficulty", choices=["normal", "easy"], default="normal")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    generators = GENERATORS
    if args.tasks.strip():
        task_names = [item.strip() for item in args.tasks.split(",") if item.strip()]
        missing = [name for name in task_names if name not in GENERATOR_BY_TASK]
        if missing:
            raise SystemExit(f"unknown tasks: {', '.join(missing)}")
        source = EASY_GENERATOR_BY_TASK if args.difficulty == "easy" else GENERATOR_BY_TASK
        generators = [source.get(name, GENERATOR_BY_TASK[name]) for name in task_names]
    counts: dict[str, int] = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for idx in range(args.records):
            fn = generators[idx % len(generators)]
            item = fn(rng)
            counts[item["task"]] = counts.get(item["task"], 0) + 1
            fh.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "records": args.records, "seed": args.seed, "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
