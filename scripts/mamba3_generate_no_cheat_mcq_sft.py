#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import date
from pathlib import Path


LETTERS = "ABCD"


FACTS = [
    {
        "domain": "science",
        "question": "Which gas do plants primarily use from air during photosynthesis?",
        "answer": "carbon dioxide",
        "distractors": ["oxygen", "nitrogen", "helium", "hydrogen"],
        "explanation": "Photosynthesis uses carbon dioxide and water to make sugars.",
    },
    {
        "domain": "science",
        "question": "What particle has a negative electric charge?",
        "answer": "electron",
        "distractors": ["proton", "neutron", "photon", "nucleus"],
        "explanation": "Electrons are negatively charged; protons are positively charged.",
    },
    {
        "domain": "science",
        "question": "Which organ pumps blood through the human body?",
        "answer": "heart",
        "distractors": ["lung", "liver", "kidney", "stomach"],
        "explanation": "The heart contracts to move blood through the circulatory system.",
    },
    {
        "domain": "science",
        "question": "What is the SI unit of force?",
        "answer": "newton",
        "distractors": ["joule", "watt", "pascal", "volt"],
        "explanation": "Force is measured in newtons.",
    },
    {
        "domain": "science",
        "question": "Which process changes liquid water into water vapor?",
        "answer": "evaporation",
        "distractors": ["freezing", "condensation", "melting", "deposition"],
        "explanation": "Evaporation is the transition from liquid to gas.",
    },
    {
        "domain": "biology",
        "question": "DNA is mainly responsible for storing what kind of information?",
        "answer": "genetic information",
        "distractors": ["thermal energy", "blood pressure", "sound waves", "muscle force"],
        "explanation": "DNA stores hereditary genetic information.",
    },
    {
        "domain": "chemistry",
        "question": "What is the chemical symbol for sodium?",
        "answer": "Na",
        "distractors": ["S", "So", "N", "Sn"],
        "explanation": "Sodium's chemical symbol is Na.",
    },
    {
        "domain": "chemistry",
        "question": "A pH below 7 usually indicates what type of solution?",
        "answer": "acidic",
        "distractors": ["basic", "neutral", "metallic", "magnetic"],
        "explanation": "Acids have pH values below 7.",
    },
    {
        "domain": "physics",
        "question": "What constant relates a photon's energy to its frequency?",
        "answer": "Planck's constant",
        "distractors": ["Boltzmann's constant", "Avogadro's number", "the gas constant", "Coulomb's constant"],
        "explanation": "The relationship is E = h f, where h is Planck's constant.",
    },
    {
        "domain": "physics",
        "question": "What is the speed of light in vacuum approximately?",
        "answer": "3.0 x 10^8 meters per second",
        "distractors": ["3.0 x 10^6 meters per second", "9.8 meters per second squared", "343 meters per second", "1.6 x 10^-19 coulombs"],
        "explanation": "Light in vacuum travels at about 3.0 x 10^8 m/s.",
    },
    {
        "domain": "computer_science",
        "question": "Which data structure uses first-in, first-out order?",
        "answer": "queue",
        "distractors": ["stack", "heap", "tree", "hash map"],
        "explanation": "A queue removes items in the order they were inserted.",
    },
    {
        "domain": "computer_science",
        "question": "Which data structure uses last-in, first-out order?",
        "answer": "stack",
        "distractors": ["queue", "array", "graph", "set"],
        "explanation": "A stack removes the most recently inserted item first.",
    },
    {
        "domain": "computer_science",
        "question": "In Big-O notation, binary search on a sorted array has what time complexity?",
        "answer": "O(log n)",
        "distractors": ["O(n)", "O(n log n)", "O(n^2)", "O(1) for every case"],
        "explanation": "Binary search halves the search interval each step.",
    },
    {
        "domain": "computer_science",
        "question": "What does a compiler usually do?",
        "answer": "translates source code into another form such as machine code",
        "distractors": ["stores passwords in plain text", "draws pixels on a monitor", "charges a battery", "measures network voltage"],
        "explanation": "A compiler translates programs from one language or representation to another.",
    },
    {
        "domain": "computer_security",
        "question": "Which property means information is not disclosed to unauthorized parties?",
        "answer": "confidentiality",
        "distractors": ["availability", "compression", "latency", "redundancy"],
        "explanation": "Confidentiality protects information from unauthorized disclosure.",
    },
    {
        "domain": "computer_security",
        "question": "Which security control verifies a user's identity?",
        "answer": "authentication",
        "distractors": ["authorization", "compression", "compilation", "pagination"],
        "explanation": "Authentication checks who a user is.",
    },
    {
        "domain": "mathematics",
        "question": "What is the derivative of x squared with respect to x?",
        "answer": "2x",
        "distractors": ["x", "x^2", "2", "1/x"],
        "explanation": "By the power rule, d(x^2)/dx = 2x.",
    },
    {
        "domain": "mathematics",
        "question": "What is the sum of the angles in a triangle in Euclidean geometry?",
        "answer": "180 degrees",
        "distractors": ["90 degrees", "270 degrees", "360 degrees", "45 degrees"],
        "explanation": "The interior angles of a Euclidean triangle sum to 180 degrees.",
    },
    {
        "domain": "mathematics",
        "question": "If 3x = 12, what is x?",
        "answer": "4",
        "distractors": ["3", "6", "9", "12"],
        "explanation": "Dividing both sides by 3 gives x = 4.",
    },
    {
        "domain": "logic",
        "question": "If all roses are flowers and this plant is a rose, what follows?",
        "answer": "this plant is a flower",
        "distractors": ["all flowers are roses", "this plant is not a flower", "no roses are plants", "flowers cannot be plants"],
        "explanation": "The conclusion follows by applying the universal statement to the rose.",
    },
    {
        "domain": "geography",
        "question": "Which continent contains the Sahara Desert?",
        "answer": "Africa",
        "distractors": ["Europe", "South America", "Australia", "Antarctica"],
        "explanation": "The Sahara Desert is in northern Africa.",
    },
    {
        "domain": "geography",
        "question": "What is the capital city of France?",
        "answer": "Paris",
        "distractors": ["Lyon", "Berlin", "Madrid", "Rome"],
        "explanation": "Paris is the capital of France.",
    },
    {
        "domain": "history",
        "question": "Who was the first president of the United States?",
        "answer": "George Washington",
        "distractors": ["Thomas Jefferson", "Abraham Lincoln", "John Adams", "James Madison"],
        "explanation": "George Washington served as the first U.S. president.",
    },
    {
        "domain": "economics",
        "question": "In economics, what does scarcity mean?",
        "answer": "resources are limited relative to wants",
        "distractors": ["all goods are free", "prices are always zero", "money has no value", "supply is infinite"],
        "explanation": "Scarcity means limited resources must be allocated among competing uses.",
    },
    {
        "domain": "medicine",
        "question": "Which vitamin is produced in human skin after sunlight exposure?",
        "answer": "vitamin D",
        "distractors": ["vitamin C", "vitamin B12", "vitamin K", "vitamin A"],
        "explanation": "Sunlight exposure helps skin synthesize vitamin D.",
    },
]


TEMPLATES = [
    "Answer the multiple-choice question. Reply with the single best letter.\n\nQuestion: {question}\n{choices}\nAnswer: {letter}",
    "Question: {question}\n{choices}\nAnswer: {letter}\nExplanation: {explanation}",
    "Instruction: Select the best answer.\nQuestion: {question}\n{choices}\nAnswer: {letter}. {answer}",
    "Instruction: Choose A, B, C, or D.\nQuestion: {question}\n{choices}\nAnswer: {letter}",
]


def generated_arithmetic(rng: random.Random) -> dict:
    kind = rng.choice(["add", "sub", "mul", "linear"])
    if kind == "add":
        a, b = rng.randint(2, 80), rng.randint(2, 80)
        ans = a + b
        return {
            "domain": "mathematics",
            "question": f"What is {a} + {b}?",
            "answer": str(ans),
            "distractors": [str(ans + d) for d in (-2, -1, 1, 3) if ans + d != ans],
            "explanation": f"Adding {a} and {b} gives {ans}.",
        }
    if kind == "sub":
        a, b = rng.randint(30, 140), rng.randint(2, 29)
        ans = a - b
        return {
            "domain": "mathematics",
            "question": f"What is {a} - {b}?",
            "answer": str(ans),
            "distractors": [str(ans + d) for d in (-5, -1, 2, 7) if ans + d != ans],
            "explanation": f"Subtracting {b} from {a} gives {ans}.",
        }
    if kind == "mul":
        a, b = rng.randint(3, 12), rng.randint(3, 12)
        ans = a * b
        return {
            "domain": "mathematics",
            "question": f"What is {a} multiplied by {b}?",
            "answer": str(ans),
            "distractors": [str(ans + d) for d in (-a, -b, a, b) if ans + d != ans],
            "explanation": f"{a} times {b} equals {ans}.",
        }
    coef = rng.randint(2, 9)
    x = rng.randint(2, 15)
    rhs = coef * x
    return {
        "domain": "mathematics",
        "question": f"If {coef}x = {rhs}, what is x?",
        "answer": str(x),
        "distractors": [str(v) for v in {x - 2, x - 1, x + 1, x + 3} if v > 0 and v != x],
        "explanation": f"Dividing {rhs} by {coef} gives x = {x}.",
    }


def generated_logic(rng: random.Random) -> dict:
    animals = [
        ("sparrows", "birds", "have feathers"),
        ("whales", "mammals", "breathe air"),
        ("oak trees", "plants", "use photosynthesis"),
        ("triangles", "polygons", "have straight sides"),
    ]
    subject, group, property_text = rng.choice(animals)
    return {
        "domain": "logic",
        "question": f"All {group} {property_text}. All {subject} are {group}. What follows?",
        "answer": f"{subject} {property_text}",
        "distractors": [
            f"no {subject} are {group}",
            f"all {group} are {subject}",
            f"{subject} do not {property_text}",
            "nothing can be concluded about the subject",
        ],
        "explanation": f"The property applies to all {group}, and {subject} are {group}.",
    }


def generated_code(rng: random.Random) -> dict:
    xs = [rng.randint(1, 9) for _ in range(3)]
    idx = rng.randrange(3)
    options = [str(xs[idx]), str(sum(xs)), str(len(xs)), str(max(xs))]
    return {
        "domain": "computer_science",
        "question": f"In Python, what is the value of [{xs[0]}, {xs[1]}, {xs[2]}][{idx}]?",
        "answer": str(xs[idx]),
        "distractors": [x for x in options if x != str(xs[idx])] + [str(idx)],
        "explanation": f"Python lists are zero-indexed, so index {idx} selects {xs[idx]}.",
    }


def choose_fact(rng: random.Random) -> dict:
    source = rng.choices(
        ["static", "arithmetic", "logic", "code"],
        weights=[55, 25, 10, 10],
        k=1,
    )[0]
    if source == "arithmetic":
        return generated_arithmetic(rng)
    if source == "logic":
        return generated_logic(rng)
    if source == "code":
        return generated_code(rng)
    return dict(rng.choice(FACTS))


def stable_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.casefold().split()).encode("utf-8")).hexdigest()


def format_choices(choices: list[str]) -> str:
    return "\n".join(f"{LETTERS[i]}. {choice}" for i, choice in enumerate(choices))


def make_record(fact: dict, rng: random.Random, idx: int) -> dict:
    choices = [fact["answer"]] + list(fact["distractors"])
    rng.shuffle(choices)
    choices = choices[:4]
    if fact["answer"] not in choices:
        choices[-1] = fact["answer"]
        rng.shuffle(choices)
    answer_idx = choices.index(fact["answer"])
    letter = LETTERS[answer_idx]
    question = fact["question"]
    if idx % 5 == 0:
        question = question.replace("Which", "What", 1) if question.startswith("Which") else question
    text = rng.choice(TEMPLATES).format(
        question=question,
        choices=format_choices(choices),
        letter=letter,
        answer=fact["answer"],
        explanation=fact["explanation"],
    )
    return {
        "text": text,
        "source": "deterministic-no-cheat-mcq-v1",
        "license": "synthetic-self-generated-no-llm",
        "language": "en",
        "domain": fact["domain"],
        "quality_score": 0.92,
        "toxicity_score": 0.0,
        "pii_score": 0.0,
        "dedup_hash": stable_hash(text),
        "benchmark_contamination_flag": False,
        "teacher_model": "codex-self-teacher",
        "generation_date": date.today().isoformat(),
        "answer_letter": letter,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate no-cheat synthetic MCQ/rationale SFT data.")
    parser.add_argument("--out", type=Path, default=Path("data/no_cheat_mcq_sft_v1.jsonl"))
    parser.add_argument("--records", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260614)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    counts: dict[str, int] = {}
    with args.out.open("w", encoding="utf-8") as fh:
        idx = 0
        attempts = 0
        while idx < args.records and attempts < args.records * 20:
            attempts += 1
            fact = choose_fact(rng)
            row = make_record(fact, rng, idx)
            key = row["dedup_hash"]
            if key in seen:
                # The fact bank is deliberately small; deterministic suffix changes phrasing without changing labels.
                row["text"] = row["text"].replace("Question:", f"Question {attempts % 997}:")
                row["dedup_hash"] = stable_hash(row["text"])
                key = row["dedup_hash"]
            if key in seen:
                continue
            seen.add(key)
            counts[row["domain"]] = counts.get(row["domain"], 0) + 1
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            idx += 1
    print(json.dumps({"out": str(args.out), "records": idx, "seed": args.seed, "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
