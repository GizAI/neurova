#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuromamba.cli import load_or_new, setup_perf  # noqa: E402


SMOKE_ITEMS = [
    {
        "task": "mmlu_like_science",
        "question": "Which gas do plants primarily absorb from the atmosphere during photosynthesis?",
        "choices": ["Oxygen", "Carbon dioxide", "Nitrogen", "Hydrogen"],
        "answer": 1,
    },
    {
        "task": "mmlu_like_history",
        "question": "Who was the first president of the United States?",
        "choices": ["George Washington", "Abraham Lincoln", "Thomas Jefferson", "John Adams"],
        "answer": 0,
    },
    {
        "task": "mmlu_like_math",
        "question": "If x + 3 = 7, what is x?",
        "choices": ["2", "3", "4", "10"],
        "answer": 2,
    },
    {
        "task": "mmlu_like_logic",
        "question": "All mammals are warm-blooded. Whales are mammals. What follows?",
        "choices": ["Whales are fish", "Whales are warm-blooded", "Mammals are whales", "Whales are cold-blooded"],
        "answer": 1,
    },
    {
        "task": "arc_like",
        "question": "A metal spoon feels colder than a wooden spoon at the same room temperature because metal",
        "choices": ["has less mass", "conducts heat better", "is always colder", "contains more air"],
        "answer": 1,
    },
    {
        "task": "arc_like",
        "question": "Which object is most likely to be attracted by a magnet?",
        "choices": ["Iron nail", "Plastic cup", "Wood pencil", "Glass bottle"],
        "answer": 0,
    },
    {
        "task": "gsm_like",
        "question": "A bag has 3 red balls and 2 blue balls. How many balls are in the bag?",
        "choices": ["1", "2", "5", "6"],
        "answer": 2,
    },
    {
        "task": "gsm_like",
        "question": "Tom buys 4 apples and then buys 3 more. How many apples does he have?",
        "choices": ["1", "3", "7", "12"],
        "answer": 2,
    },
    {
        "task": "code_like",
        "question": "In Python, what does len([1, 2, 3]) return?",
        "choices": ["1", "2", "3", "4"],
        "answer": 2,
    },
    {
        "task": "json_like",
        "question": "In JSON, which value type is written without quotes?",
        "choices": ["string", "number", "object key", "plain word with spaces"],
        "answer": 1,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multiple-choice benchmark gate for local Mamba-3 checkpoints.")
    parser.add_argument("--mode", default="mamba3-siso-fast-0.3b-ds128")
    parser.add_argument("--tokenizer", default="llama31")
    parser.add_argument("--checkpoint", type=Path, default=Path("neuromamba/runs/mamba3_siso_fast_0_3b_ds128_v3/model.pt"))
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--suite", choices=["smoke", "mmlu", "mmlu_redux"], default="smoke")
    parser.add_argument("--mmlu-subject", default="all")
    parser.add_argument("--mmlu-redux-dataset", default="edinburgh-dawg/mmlu-redux-2.0")
    parser.add_argument(
        "--redux-filter",
        choices=["ok", "corrected", "all"],
        default="ok",
        help="ok evaluates only Redux rows marked ok; corrected also uses numeric corrected labels where provided.",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("neuromamba/runs/mamba3_benchmarks/latest_mcq.json"))
    parser.add_argument("--show", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument(
        "--score-mode",
        choices=["choice_text", "letter", "both"],
        default="both",
        help="choice_text is less sensitive to A/B/C/D prior; both records both metrics.",
    )
    return parser.parse_args()


def format_prompt(item: dict[str, Any]) -> str:
    letters = "ABCD"
    choices = "\n".join(f"{letters[i]}. {choice}" for i, choice in enumerate(item["choices"]))
    return (
        "Answer the multiple-choice question. Reply with the single best letter.\n\n"
        f"Question: {item['question']}\n"
        f"{choices}\n"
        "Answer:"
    )


def score_target(model, tokenizer, prompt: str, target: str, seq_len: int, device: str) -> float:
    prompt_ids = tokenizer.encode(prompt, add_eos=False)
    target_ids = tokenizer.encode(target, add_eos=False)
    if not prompt_ids or not target_ids:
        return -math.inf
    ids = prompt_ids + target_ids[:-1]
    ids = ids[-seq_len:]
    x = torch.tensor([ids], device=device, dtype=torch.long)
    with torch.no_grad():
      logits = model(x).logits[0].float()
    prompt_kept = min(len(prompt_ids), len(ids))
    start = max(0, prompt_kept - 1)
    total = 0.0
    usable = min(len(target_ids), logits.shape[0] - start)
    if usable <= 0:
        return -math.inf
    for j in range(usable):
        logp = F.log_softmax(logits[start + j], dim=-1)
        total += float(logp[target_ids[j]].item())
    return total / usable


def shuffle_item(item: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    pairs = list(enumerate(item["choices"]))
    rng.shuffle(pairs)
    new_answer = next(i for i, (old_i, _) in enumerate(pairs) if old_i == int(item["answer"]))
    return {
        **item,
        "choices": [choice for _, choice in pairs],
        "answer": new_answer,
        "original_answer": int(item["answer"]),
    }


def load_smoke_items(limit: int) -> list[dict[str, Any]]:
    return SMOKE_ITEMS[:limit]


def load_mmlu_items(subject: str, limit: int) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise SystemExit(f"datasets is not available: {exc}") from exc

    configs = [subject] if subject != "all" else ["all"]
    rows: list[dict[str, Any]] = []
    last_error: Exception | None = None
    for config in configs:
        for split in ("test", "validation", "dev"):
            try:
                ds = load_dataset("cais/mmlu", config, split=split)
                for row in ds:
                    choices = row.get("choices") or row.get("options")
                    answer = row.get("answer")
                    question = row.get("question")
                    if question is None or choices is None or answer is None:
                        continue
                    if isinstance(answer, str):
                        answer = "ABCD".find(answer.strip().upper())
                    if not isinstance(answer, int) or answer < 0:
                        continue
                    rows.append({
                        "task": f"mmlu:{config}",
                        "question": str(question),
                        "choices": [str(x) for x in choices[:4]],
                        "answer": int(answer),
                    })
                    if len(rows) >= limit:
                        return rows
                if rows:
                    return rows
            except Exception as exc:
                last_error = exc
                continue
    raise SystemExit(f"failed to load MMLU from HuggingFace: {last_error}")


def _parse_mcq_answer(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    letter = "ABCD".find(text.upper())
    if letter >= 0:
        return letter
    return None


def load_mmlu_redux_items(dataset_name: str, subject: str, redux_filter: str, limit: int) -> list[dict[str, Any]]:
    try:
        from datasets import get_dataset_config_names, load_dataset
    except Exception as exc:
        raise SystemExit(f"datasets is not available: {exc}") from exc

    try:
        configs = [subject] if subject != "all" else list(get_dataset_config_names(dataset_name))
    except Exception as exc:
        raise SystemExit(f"failed to list MMLU-Redux configs from {dataset_name}: {exc}") from exc

    rows: list[dict[str, Any]] = []
    last_error: Exception | None = None
    for config in configs:
        for split in ("test", "validation", "dev"):
            try:
                ds = load_dataset(dataset_name, config, split=split)
            except Exception as exc:
                last_error = exc
                continue
            for row in ds:
                error_type = str(row.get("error_type") or "").strip()
                if redux_filter == "ok" and error_type != "ok":
                    continue
                choices = row.get("choices")
                question = row.get("question")
                if question is None or not isinstance(choices, list) or len(choices) < 2:
                    continue
                answer = _parse_mcq_answer(row.get("answer"))
                corrected = _parse_mcq_answer(row.get("correct_answer"))
                if redux_filter == "corrected" and corrected is not None:
                    answer = corrected
                if answer is None or answer < 0 or answer >= min(4, len(choices)):
                    continue
                rows.append({
                    "task": f"mmlu_redux:{config}",
                    "question": str(question),
                    "choices": [str(x) for x in choices[:4]],
                    "answer": int(answer),
                    "redux_error_type": error_type,
                })
                if len(rows) >= limit:
                    return rows
            if rows or subject != "all":
                break
    if not rows:
        raise SystemExit(f"failed to load usable MMLU-Redux rows from {dataset_name}: {last_error}")
    return rows


def main() -> None:
    args = parse_args()
    setup_perf(args.device)
    model_args = SimpleNamespace(
        cmd="eval-mcq",
        mode=args.mode,
        tokenizer=args.tokenizer,
        checkpoint=args.checkpoint,
        seq_len=args.seq_len,
        batch_size=1,
        device=args.device,
        dtype=args.dtype,
        activation_checkpointing=False,
    )
    model, tokenizer, _ = load_or_new(model_args)
    model.eval()

    rng = random.Random(args.seed)
    if args.suite == "smoke":
        items = load_smoke_items(args.limit)
    elif args.suite == "mmlu":
        items = load_mmlu_items(args.mmlu_subject, args.limit)
    else:
        items = load_mmlu_redux_items(args.mmlu_redux_dataset, args.mmlu_subject, args.redux_filter, args.limit)
        rng.shuffle(items)
        items = items[: args.limit]
    items = [shuffle_item(item, rng) for item in items]
    letters = "ABCD"
    results = []
    started = time.time()
    correct_letter = 0
    correct_choice = 0
    by_task: dict[str, dict[str, int]] = {}
    for idx, item in enumerate(items):
        prompt = format_prompt(item)
        letter_scores = [
            score_target(model, tokenizer, prompt, f" {letter}", args.seq_len, args.device)
            for letter in letters[: len(item["choices"])]
        ]
        choice_scores = [
            score_target(model, tokenizer, prompt, f" {choice}", args.seq_len, args.device)
            for choice in item["choices"]
        ]
        pred_letter = int(max(range(len(letter_scores)), key=lambda i: letter_scores[i]))
        pred_choice = int(max(range(len(choice_scores)), key=lambda i: choice_scores[i]))
        ok_letter = pred_letter == int(item["answer"])
        ok_choice = pred_choice == int(item["answer"])
        correct_letter += int(ok_letter)
        correct_choice += int(ok_choice)
        task = item.get("task", "unknown")
        by_task.setdefault(task, {"letter_correct": 0, "choice_correct": 0, "total": 0})
        by_task[task]["letter_correct"] += int(ok_letter)
        by_task[task]["choice_correct"] += int(ok_choice)
        by_task[task]["total"] += 1
        if idx < args.show:
            results.append({
                "task": task,
                "question": item["question"],
                "letter_prediction": letters[pred_letter],
                "choice_prediction": letters[pred_choice],
                "answer": letters[int(item["answer"])],
                "letter_ok": ok_letter,
                "choice_ok": ok_choice,
                "letter_scores": {letters[i]: round(letter_scores[i], 4) for i in range(len(letter_scores))},
                "choice_scores": {letters[i]: round(choice_scores[i], 4) for i in range(len(choice_scores))},
                "choices": {letters[i]: item["choices"][i] for i in range(len(item["choices"]))},
            })

    total = len(items)
    primary_correct = correct_choice if args.score_mode in {"choice_text", "both"} else correct_letter
    payload = {
        "suite": args.suite,
        "mmlu_subject": args.mmlu_subject,
        "mmlu_redux_dataset": args.mmlu_redux_dataset if args.suite == "mmlu_redux" else None,
        "redux_filter": args.redux_filter if args.suite == "mmlu_redux" else None,
        "score_mode": args.score_mode,
        "seed": args.seed,
        "mode": args.mode,
        "checkpoint": str(args.checkpoint),
        "seq_len": args.seq_len,
        "total": total,
        "correct": primary_correct,
        "accuracy": round(primary_correct / total, 6) if total else 0.0,
        "letter_correct": correct_letter,
        "letter_accuracy": round(correct_letter / total, 6) if total else 0.0,
        "choice_correct": correct_choice,
        "choice_accuracy": round(correct_choice / total, 6) if total else 0.0,
        "elapsed_sec": round(time.time() - started, 3),
        "by_task": {
            task: {
                **vals,
                "letter_accuracy": round(vals["letter_correct"] / vals["total"], 6) if vals["total"] else 0.0,
                "choice_accuracy": round(vals["choice_correct"] / vals["total"], 6) if vals["total"] else 0.0,
            }
            for task, vals in sorted(by_task.items())
        },
        "examples": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
