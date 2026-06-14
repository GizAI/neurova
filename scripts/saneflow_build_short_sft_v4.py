#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


SYSTEM = (
    "You are Neurova, a concise local assistant. Answer the user's request directly. "
    "Use one or two clear sentences unless the user asks for a list. If the answer is unknown, say you do not know."
)


def digest(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def chatml(user: str, assistant: str, system: str = SYSTEM) -> str:
    return (
        f"<|im_start|>system\n{system}\n<|im_end|>\n"
        f"<|im_start|>user\n{user.strip()}\n<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant.strip()}\n<|im_end|>"
    )


def parse_single_turn(text: str) -> tuple[str, str] | None:
    user_marker = "<|im_start|>user\n"
    assistant_marker = "<|im_start|>assistant\n"
    end_marker = "<|im_end|>"
    users = list(re.finditer(re.escape(user_marker), text))
    assistants = list(re.finditer(re.escape(assistant_marker), text))
    if len(users) != 1 or len(assistants) != 1:
        return None
    u0 = users[0].end()
    u1 = text.find(end_marker, u0)
    a0 = assistants[0].end()
    a1 = text.find(end_marker, a0)
    if u1 < 0 or a1 < 0 or not (u0 < u1 < a0 < a1):
        return None
    user = text[u0:u1].strip()
    assistant = text[a0:a1].strip()
    if not user or not assistant:
        return None
    return user, assistant


def is_good_short_pair(user: str, assistant: str, *, max_user: int, max_assistant: int) -> bool:
    if len(user) > max_user or len(assistant) > max_assistant:
        return False
    bad = ["```", "<think", "</think", "###", "Question:", "Instruction:"]
    if any(marker in assistant for marker in bad):
        return False
    # Avoid huge lists and code-heavy answers in the first speaking SFT pass.
    if assistant.count("\n") > 5:
        return False
    if len(re.findall(r"\b\d+\.", assistant)) > 4:
        return False
    if assistant.count("{") + assistant.count("}") > 8:
        return False
    if not re.search(r"[A-Za-z가-힣0-9]", assistant):
        return False
    return True


def add(rows: list[dict[str, Any]], user: str, assistant: str, source: str) -> None:
    text = chatml(user, assistant)
    rows.append({"text": text, "source": source, "dedup_hash": digest(text)})


def add_programmatic(rows: list[dict[str, Any]], seed: int) -> None:
    rng = random.Random(seed)
    basics = [
        ("hi", "Hello. I am Neurova. How can I help?"),
        ("hello", "Hello. I am Neurova. What would you like to do?"),
        ("Who are you?", "I am Neurova, a small local assistant trained to answer clearly."),
        ("What is a computer?", "A computer is a machine that stores and processes information by following instructions."),
        ("What is machine learning?", "Machine learning is a way for computers to learn patterns from data and use them to make predictions."),
        ("What is water?", "Water is a clear liquid made of hydrogen and oxygen that living things need."),
        ("Write one sentence about the moon.", "The moon is Earth's natural satellite and reflects sunlight at night."),
        ("Explain why the sky looks blue.", "The sky looks blue because air scatters blue light from the sun more than many other colors."),
        ("If you do not know an answer, what should you do?", "I should say that I do not know instead of inventing an answer."),
    ]
    for q, a in basics:
        for _ in range(40):
            add(rows, q, a, "short-v4-basic")

    names = ["Alice", "Ben", "Cara", "Dina", "Evan", "Liam", "Mina", "Noah", "Omar", "Rina", "Tom", "Sam", "Leo"]
    objects = ["apples", "books", "coins", "pencils", "marbles", "cards", "oranges", "stickers"]
    for _ in range(3000):
        name = rng.choice(names)
        obj = rng.choice(objects)
        a = rng.randint(1, 30)
        b = rng.randint(1, 30)
        add(rows, f"If {name} has {a} {obj} and gets {b} more, how many {obj} does {name} have?", str(a + b), "short-v4-arithmetic")
        x, y = rng.sample(range(1, 100), 2)
        add(rows, f"Which is larger, {x} or {y}?", str(max(x, y)), "short-v4-comparison")
        old, mid, young = rng.sample(names, 3)
        add(rows, f"If {old} is older than {mid}, and {mid} is older than {young}, who is the youngest?", young, "short-v4-ordering")
        code = f"{rng.choice(['AX','BK','CX','NV','QZ'])}-{rng.randint(100,999)}"
        add(rows, f"Remember this code: {code}. What is the code?", code, "short-v4-copy")


def add_from_jsonl(rows: list[dict[str, Any]], path: Path, source_prefix: str, *, limit: int, max_user: int, max_assistant: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
    rng.shuffle(items)
    added = 0
    scanned = 0
    for row in items:
        scanned += 1
        text = str(row.get("text", ""))
        pair = parse_single_turn(text)
        if not pair:
            continue
        user, assistant = pair
        assistant = assistant.replace("LUMA", "Neurova")
        user = user.replace("LUMA", "Neurova")
        if not is_good_short_pair(user, assistant, max_user=max_user, max_assistant=max_assistant):
            continue
        add(rows, user, assistant, f"{source_prefix}:{row.get('source','unknown')}")
        added += 1
        if added >= limit:
            break
    return {"path": str(path), "scanned": scanned, "added": added}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Build short, stable, same-day SaneFlow SFT v4.")
    p.add_argument("--out-dir", type=Path, default=Path("data/corpus/mixes/saneflow_short_sft_v4"))
    p.add_argument("--seed", type=int, default=20260614)
    p.add_argument("--valid-size", type=int, default=1000)
    args = p.parse_args()

    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    add_programmatic(rows, args.seed)
    reports.append({"source": "programmatic", "added": len(rows)})
    reports.append(add_from_jsonl(rows, Path("data/corpus/mixes/saneflow_open_sft_v2/train.jsonl"), "open-short", limit=8000, max_user=700, max_assistant=700, seed=args.seed + 1))
    reports.append(add_from_jsonl(rows, Path("data/luma_clean_chatml_sft_v1.jsonl"), "local-clean", limit=5000, max_user=300, max_assistant=400, seed=args.seed + 2))
    reports.append(add_from_jsonl(rows, Path("data/luma_stage_chatml_reasoning_v2.jsonl"), "local-mcq", limit=500, max_user=700, max_assistant=500, seed=args.seed + 3))
    reports.append(add_from_jsonl(rows, Path("data/luma_stage_chatml_memory_v2.jsonl"), "local-memory", limit=2500, max_user=700, max_assistant=500, seed=args.seed + 4))
    reports.append(add_from_jsonl(rows, Path("data/luma_stage_chatml_slotproof_v2.jsonl"), "local-slotproof", limit=1200, max_user=900, max_assistant=300, seed=args.seed + 5))

    dedup: dict[str, dict[str, Any]] = {}
    for row in rows:
        dedup.setdefault(row["dedup_hash"], row)
    rows = list(dedup.values())
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    valid_size = min(args.valid_size, max(1, len(rows) // 10))
    valid = rows[:valid_size]
    train = rows[valid_size:]
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "valid.jsonl", valid)
    manifest = {
        "name": "saneflow_short_sft_v4",
        "goal": "same-day stable dialogue and basic QA, not broad reasoning",
        "train_records": len(train),
        "valid_records": len(valid),
        "reports": reports,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
