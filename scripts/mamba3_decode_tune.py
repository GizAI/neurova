#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mamba3_kr.cli import _fast_decode_once, load_or_new, normalize_runtime_flags, setup_perf


STOP_MARKERS = [
    "\nQuestion:",
    "\nInstruction:",
    "\nUser:",
    "\nAssistant:",
    "\nAnswer in",
]


@dataclass(frozen=True)
class PromptCase:
    name: str
    prompt: str
    required_any: tuple[str, ...]
    banned: tuple[str, ...] = ()


PROMPTS = [
    PromptCase(
        "identity",
        "Who are you?",
        ("neurova", "assistant", "model"),
        ("photosynthesis", "sky blue", "korea"),
    ),
    PromptCase(
        "ml",
        "What is machine learning?",
        ("data", "model", "learn", "pattern", "prediction"),
        ("korea", "neurova"),
    ),
    PromptCase(
        "honesty",
        "What should you do if you do not know?",
        ("say", "know", "uncertain", "clearly", "invent"),
        ("photosynthesis",),
    ),
    PromptCase(
        "simple_math",
        "What is 2 + 5?",
        ("7", "seven"),
        ("korea", "photosynthesis"),
    ),
    PromptCase(
        "science",
        "Answer in one sentence: why is the sky blue?",
        ("light", "blue", "scatter", "atmosphere"),
        ("neurova", "korea"),
    ),
]


CONFIGS = [
    {
        "name": "quality_argmax_full",
        "safe_decode": True,
        "cuda_graph": False,
        "top_k": 1,
        "top_p": 1.0,
        "temperature": 1.0,
        "repetition_penalty": 1.0,
    },
    {
        "name": "fast_argmax_graph",
        "safe_decode": False,
        "cuda_graph": True,
        "top_k": 1,
        "top_p": 1.0,
        "temperature": 1.0,
        "repetition_penalty": 1.0,
    },
    {
        "name": "fast_low_entropy_graph",
        "safe_decode": False,
        "cuda_graph": True,
        "top_k": 8,
        "top_p": 0.8,
        "temperature": 0.6,
        "repetition_penalty": 1.05,
    },
    {
        "name": "fast_default_graph",
        "safe_decode": False,
        "cuda_graph": True,
        "top_k": 40,
        "top_p": 0.9,
        "temperature": 0.8,
        "repetition_penalty": 1.15,
    },
]


def chat_prompt(text: str) -> str:
    return f"Instruction: {text.strip()}\nAnswer:"


def clean_answer(text: str, prompt: str) -> str:
    answer = text[len(prompt):] if text.startswith(prompt) else text
    answer = answer.strip()
    answer = re.sub(r"^\?+\s*", "", answer)
    answer = re.sub(r"^(Answer:|A:)\s*", "", answer, flags=re.IGNORECASE)
    for marker in STOP_MARKERS:
        pos = answer.find(marker)
        if pos >= 0:
            answer = answer[:pos]
    return re.sub(r"\s+", " ", answer.strip())


def repeat_metrics(text: str) -> dict:
    words = [part.strip(".,;:!?()[]{}\"'").lower() for part in text.split()]
    words = [word for word in words if word]
    longest_run = 0
    current_run = 0
    previous = None
    for word in words:
        if word == previous:
            current_run += 1
        else:
            current_run = 1
            previous = word
        longest_run = max(longest_run, current_run)
    distinct = len(set(words))
    return {
        "words": len(words),
        "distinct_words": distinct,
        "longest_repeated_word_run": longest_run,
    }


def score_answer(case: PromptCase, answer: str) -> dict:
    lowered = answer.casefold()
    required_hits = [word for word in case.required_any if word.casefold() in lowered]
    banned_hits = [word for word in case.banned if word.casefold() in lowered]
    repeats = repeat_metrics(answer)
    has_text = repeats["words"] >= 2
    no_collapse = repeats["longest_repeated_word_run"] <= 3 and repeats["distinct_words"] >= min(3, repeats["words"])
    has_signal = bool(required_hits)
    ok = has_text and no_collapse and has_signal and not banned_hits
    return {
        "ok": ok,
        "required_hits": required_hits,
        "banned_hits": banned_hits,
        **repeats,
    }


@torch.inference_mode()
def run_config(model, tokenizer, base_args: argparse.Namespace, cfg: dict) -> dict:
    args = SimpleNamespace(**vars(base_args))
    for key, value in cfg.items():
        if key != "name":
            setattr(args, key, value)
    normalize_runtime_flags(args)
    rows = []
    for case in PROMPTS:
        prompt = chat_prompt(case.prompt)
        result = _fast_decode_once(model, tokenizer, args, prompt)
        raw_text = result.pop("text")
        answer = clean_answer(raw_text, prompt)
        score = score_answer(case, answer)
        rows.append(
            {
                "case": case.name,
                "input": case.prompt,
                "answer": answer,
                "score": score,
                "metrics": result,
            }
        )
    ok_count = sum(1 for row in rows if row["score"]["ok"])
    speeds = [
        float(row["metrics"]["new_tokens_per_sec"])
        for row in rows
        if row["metrics"].get("new_tokens_per_sec") is not None
    ]
    total_new = sum(int(row["metrics"].get("new_tokens", 0)) for row in rows)
    avg_speed = sum(speeds) / max(1, len(speeds))
    return {
        "name": cfg["name"],
        "config": {key: value for key, value in cfg.items() if key != "name"},
        "ok_count": ok_count,
        "total_cases": len(rows),
        "quality_rate": ok_count / max(1, len(rows)),
        "avg_new_tokens_per_sec": round(avg_speed, 2),
        "total_new_tokens": total_new,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="mimo-r4-tiny")
    parser.add_argument("--tokenizer", default="llama31")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/mamba3_neurova_speak_v1/sft.pt"))
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-new", type=int, default=24)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--nan-check", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("runs/mamba3_neurova_speak_v1/decode_tune/latest.json"))
    args = parser.parse_args()

    setup_perf(args.device)
    args.cmd = "fast-generate"
    model, tokenizer, _ = load_or_new(args)
    model.eval()

    # Warm up kernels before scoring measured runs.
    warm_args = SimpleNamespace(**vars(args))
    warm_args.safe_decode = True
    warm_args.cuda_graph = False
    warm_args.top_k = 1
    warm_args.top_p = 1.0
    warm_args.temperature = 1.0
    warm_args.repetition_penalty = 1.0
    _fast_decode_once(model, tokenizer, warm_args, chat_prompt("Can you help me?"))

    results = [run_config(model, tokenizer, args, cfg) for cfg in CONFIGS]
    best = max(
        results,
        key=lambda item: (
            item["ok_count"],
            item["quality_rate"],
            item["avg_new_tokens_per_sec"],
        ),
    )
    payload = {
        "mode": args.mode,
        "tokenizer": args.tokenizer,
        "checkpoint": str(args.checkpoint),
        "seq_len": args.seq_len,
        "max_new": args.max_new,
        "best": {
            "name": best["name"],
            "ok_count": best["ok_count"],
            "total_cases": best["total_cases"],
            "avg_new_tokens_per_sec": best["avg_new_tokens_per_sec"],
            "config": best["config"],
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
