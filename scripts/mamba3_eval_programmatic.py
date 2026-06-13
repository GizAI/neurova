#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mamba3_kr.cli import _fast_decode_once, load_or_new, normalize_runtime_flags, setup_perf


def answer_prompt(text: str) -> tuple[str, str]:
    for marker in ("\nAnswer:", "\nAssistant:", "\nA:"):
        if marker in text:
            prompt, answer = text.split(marker, 1)
            return prompt + marker, answer.strip()
    raise ValueError("record text must contain an answer marker")


def normalize(text: str) -> str:
    return " ".join(text.strip().split()).casefold()


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact-match eval for no-teacher Mamba-3 programmatic curriculum.")
    parser.add_argument("--data", type=Path, default=Path("data/mamba3_programmatic_curriculum.jsonl"))
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--mode", default="mimo-r4-tiny")
    parser.add_argument("--tokenizer", default="llama31")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/mamba3_kr_tiny/model_mimo_r4_speak.pt"))
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-new", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--min-accuracy", type=float, default=0.0)
    args = parser.parse_args()

    args.cmd = "eval-programmatic"
    args.safe_decode = False
    args.nan_check = False
    setup_perf(args.device)
    normalize_runtime_flags(args)
    model, tokenizer, _ = load_or_new(args)

    total = 0
    correct = 0
    by_task: dict[str, dict[str, int]] = {}
    examples = []
    with args.data.open("r", encoding="utf-8") as fh:
        for line in fh:
            if total >= args.limit:
                break
            item = json.loads(line)
            prompt, answer = answer_prompt(item["text"])
            result = _fast_decode_once(model, tokenizer, args, prompt)
            output = result["text"]
            generated = output[len(prompt):] if output.startswith(prompt) else output
            ok = normalize(generated).startswith(normalize(answer))
            task = item.get("task") or item.get("domain", "unknown")
            by_task.setdefault(task, {"correct": 0, "total": 0})
            by_task[task]["total"] += 1
            by_task[task]["correct"] += int(ok)
            total += 1
            correct += int(ok)
            if len(examples) < 8:
                examples.append({
                    "task": task,
                    "ok": ok,
                    "prompt": prompt,
                    "expected": answer,
                    "generated": generated.strip(),
                })

    summary = {
        "ok": total > 0,
        "mode": args.mode,
        "tokenizer": args.tokenizer,
        "checkpoint": str(args.checkpoint),
        "data": str(args.data),
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "min_accuracy": args.min_accuracy,
        "by_task": {
            task: {
                **counts,
                "accuracy": round(counts["correct"] / counts["total"], 4) if counts["total"] else 0.0,
            }
            for task, counts in sorted(by_task.items())
        },
        "examples": examples,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if total == 0:
        raise SystemExit(1)
    if summary["accuracy"] < args.min_accuracy:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
