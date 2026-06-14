#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


def digest(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def parse_chatml(text: str) -> tuple[str, str] | None:
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
    if u1 < 0 or a1 < 0:
        return None
    user = text[u0:u1].strip()
    assistant = text[a0:a1].strip()
    if not user or not assistant:
        return None
    return user, assistant


def plain(user: str, assistant: str) -> str:
    return f"User: {user.strip()}\nAssistant: {assistant.strip()}"


def keep_pair(user: str, assistant: str) -> bool:
    noisy_user = ["Teacher:", "Student:", "Options:", "Let's think:", "Choose A, B, C, or D."]
    noisy_assistant = ["Let's think:", "So the final answer", "```"]
    if any(x in user for x in noisy_user):
        return False
    if any(x in assistant for x in noisy_assistant):
        return False
    if assistant.count("\n") > 4:
        return False
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Convert short ChatML SFT to plain User/Assistant SFT.")
    p.add_argument("--input-dir", type=Path, default=Path("data/corpus/mixes/saneflow_short_sft_v4"))
    p.add_argument("--out-dir", type=Path, default=Path("data/corpus/mixes/saneflow_plain_sft_v5"))
    p.add_argument("--seed", type=int, default=20260614)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"name": "saneflow_plain_sft_v5", "source": str(args.input_dir), "splits": {}}
    for split in ["train", "valid"]:
        rows: list[dict[str, str]] = []
        src = args.input_dir / f"{split}.jsonl"
        for line in src.open(encoding="utf-8"):
            row = json.loads(line)
            pair = parse_chatml(str(row.get("text", "")))
            if not pair:
                continue
            user, assistant = pair
            if not keep_pair(user, assistant):
                continue
            text = plain(user, assistant)
            rows.append({"text": text, "source": "plain-v5:" + str(row.get("source", "unknown")), "dedup_hash": digest(text)})
        random.Random(args.seed + (0 if split == "train" else 1)).shuffle(rows)
        out = args.out_dir / f"{split}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["splits"][split] = {"records": len(rows), "path": str(out)}
    (args.out_dir / "manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
