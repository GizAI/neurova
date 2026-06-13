from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


WORDS = [
    "amber", "brisk", "cedar", "delta", "ember", "fable", "glade", "harbor",
    "ion", "juniper", "keystone", "lumen", "mosaic", "nova", "onyx", "prairie",
    "quartz", "raven", "signal", "thicket", "umbra", "vector", "willow", "zenith",
]


def rand_word(rng: random.Random) -> str:
    return rng.choice(WORDS) + rng.choice("abcdefghijklmnopqrstuvwxyz")


def emit(task: str, instruction: str, answer: str, **meta) -> dict:
    return {
        "task": task,
        "answer": answer,
        "text": f"Instruction: {instruction.strip()}\nAnswer: {answer.strip()}",
        "metadata": {"source": "luma-memory-curriculum", **meta},
    }


def gen_copy(rng: random.Random) -> dict:
    items = [rand_word(rng) for _ in range(rng.randint(1, 8))]
    answer = " ".join(items)
    return emit("copy", f"Copy this sequence exactly: {answer}", answer, length=len(items))


def gen_phonebook(rng: random.Random) -> dict:
    pairs = [(rand_word(rng), f"{rng.randint(100, 999)}-{rng.randint(1000, 9999)}") for _ in range(rng.randint(2, 8))]
    key, value = rng.choice(pairs)
    book = "; ".join(f"{name}: {phone}" for name, phone in pairs)
    return emit("phonebook", f"Phonebook: {book}. What is the phone number for {key}?", value, entries=len(pairs))


def gen_json_field(rng: random.Random) -> dict:
    obj = {
        "city": rng.choice(["Seoul", "Boston", "Nairobi", "Oslo", "Lima"]),
        "score": rng.randint(1, 100),
        "tag": rand_word(rng),
        "code": f"{rng.choice(['AX', 'LM', 'NV'])}-{rng.randint(100, 999)}",
    }
    key = rng.choice(list(obj.keys()))
    return emit(
        "json_field",
        f"Given JSON {json.dumps(obj, sort_keys=True)}, return the value of field `{key}` only.",
        str(obj[key]),
        key=key,
    )


def gen_recall(rng: random.Random) -> dict:
    name = rng.choice(["Mina", "Joon", "Ara", "Noah", "Yuna", "Sora"])
    obj = rng.choice(["blue key", "red notebook", "silver coin", "green map", "black card"])
    place = rng.choice(["seoul", "busan", "lab7", "mars room", "quiet library"])
    color = rng.choice(["cyan", "amber", "violet", "white", "orange"])
    facts = [
        f"{name} owns the {obj}",
        f"{name} should go to {place}",
        f"{name}'s color is {color}",
        f"{rand_word(rng)} owns a distractor object",
    ]
    rng.shuffle(facts)
    question, answer = rng.choice(
        [
            (f"What object belongs to {name}?", obj),
            (f"Where should {name} go?", place),
            (f"What is {name}'s color?", color),
        ]
    )
    return emit("recall", f"Facts: {'; '.join(facts)}. {question}", answer)


def gen_update(rng: random.Random) -> dict:
    name = rng.choice(["Mina", "Joon", "Ara", "Noah"])
    old = rng.choice(["blue key", "red notebook", "silver coin"])
    new = rng.choice([item for item in ["green map", "black card", "white coin"] if item != old])
    return emit("update", f"{name} first owns the {old}. Later, {name} owns the {new}. What does {name} own now?", new)


def gen_protect(rng: random.Random) -> dict:
    name = rng.choice(["Mina", "Joon", "Ara", "Noah"])
    real = rng.choice(["blue key", "red notebook", "silver coin"])
    fake = rng.choice([item for item in ["green map", "black card", "white coin"] if item != real])
    return emit(
        "protect",
        f"Protected fact: {name} owns the {real}. Do not overwrite it. A later rumor says {name} owns the {fake}. What is the protected object?",
        real,
    )


GENERATORS = [gen_copy, gen_phonebook, gen_json_field, gen_recall, gen_update, gen_protect]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a synthetic memory-first LUMA curriculum.")
    p.add_argument("--out", required=True)
    p.add_argument("--records", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=20260614)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for _ in range(args.records):
            row = rng.choice(GENERATORS)(rng)
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"out": str(out), "records": args.records}, ensure_ascii=False))


if __name__ == "__main__":
    main()
