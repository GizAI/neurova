#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from saneflow_build_short_sft_v4 import (
    add,
    add_from_jsonl,
    add_programmatic,
    digest,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean ChatML SFT data for same-day SaneFlow dialogue recovery.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/corpus/mixes/saneflow_chatml_clean_sft_v6"))
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--valid-size", type=int, default=1000)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    add_programmatic(rows, args.seed)
    reports.append({"source": "programmatic", "added": len(rows)})

    # Keep this pass intentionally narrow. The broad open SFT mixture is useful
    # later, but it currently injects long NLI/MCQ scaffolding and harms basic
    # chat recovery on an undertrained base.
    reports.append(
        add_from_jsonl(
            rows,
            Path("data/luma_clean_chatml_sft_v1.jsonl"),
            "local-clean",
            limit=5000,
            max_user=260,
            max_assistant=360,
            seed=args.seed + 1,
        )
    )
    reports.append(
        add_from_jsonl(
            rows,
            Path("data/luma_stage_chatml_memory_v2.jsonl"),
            "local-memory-short",
            limit=2500,
            max_user=420,
            max_assistant=220,
            seed=args.seed + 2,
        )
    )
    reports.append(
        add_from_jsonl(
            rows,
            Path("data/luma_stage_chatml_reasoning_v2.jsonl"),
            "local-reasoning-short",
            limit=600,
            max_user=420,
            max_assistant=180,
            seed=args.seed + 3,
        )
    )

    # Extra identity/behavior anchors. These are duplicated with varied phrasings
    # on purpose; the target is basic dialogue stability, not broad benchmark
    # coverage.
    anchors = [
        ("What is your name?", "My name is Neurova."),
        ("Introduce yourself briefly.", "I am Neurova, a local assistant that gives short, direct answers."),
        ("Say hello in one sentence.", "Hello. I am Neurova, and I am ready to help."),
        ("What should you do if you are unsure?", "I should say that I am unsure instead of guessing."),
        ("Answer with only the number: 2 plus 3.", "5"),
        ("Answer with only the larger number: 9 or 12.", "12"),
        ("What is the capital of France?", "Paris is the capital of France."),
        ("What language are we using now?", "We are using English in this answer."),
    ]
    for idx in range(160):
        for user, assistant in anchors:
            add(rows, user, assistant, f"clean-v6-anchor-{idx % 8}")

    dedup: dict[str, dict[str, Any]] = {}
    for row in rows:
        row["dedup_hash"] = row.get("dedup_hash") or digest(row["text"])
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
        "name": "saneflow_chatml_clean_sft_v6",
        "format": "chatml",
        "loss_mode": "chatml_assistant",
        "goal": "recover basic instruction dialogue with atomic ChatML special tokens",
        "train_records": len(train),
        "valid_records": len(valid),
        "reports": reports,
        "excluded": ["broad open SFT mixture until base chat quality is stable"],
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
