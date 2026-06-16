#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuromamba.cli import _fast_decode_once, load_or_new, normalize_runtime_flags, setup_perf


CASES = [
    {
        "prompt": "Answer in one clear English sentence: What is photosynthesis?",
        "contains": ["plants", "sunlight"],
    },
    {
        "prompt": "Answer in one clear English sentence: Why should we test software?",
        "contains": ["mistakes", "works"],
    },
    {
        "prompt": "Answer with the number only. Which is larger, 7 or 3?",
        "exact": "7",
    },
    {
        "prompt": "Answer with the number only. What is 2 + 5?",
        "exact": "7",
    },
    {
        "prompt": "Answer with yes or no only. Is water usually wet?",
        "exact": "yes",
    },
    {
        "prompt": "Answer in one clear English sentence: What is science?",
        "contains": ["evidence"],
    },
    {
        "prompt": "Answer in one clear English sentence: What is courage?",
        "contains": ["difficult"],
    },
]


def normalize_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(answer\s*:\s*)+", "", text, flags=re.IGNORECASE).strip()
    return text


def score_case(prompt: str, completion: str, case: dict) -> dict:
    answer = normalize_answer(completion)
    lowered = answer.lower()
    if "exact" in case:
        expected = case["exact"].lower()
        first = re.split(r"\s+", lowered.strip())[0].strip(".,;:!?") if lowered.strip() else ""
        ok = first == expected and len(lowered.split()) <= 3
        missing = [] if ok else [case["exact"]]
    else:
        missing = [term for term in case["contains"] if term.lower() not in lowered]
        ok = not missing and len(answer.split()) >= 5
    return {
        "prompt": prompt,
        "completion": answer,
        "ok": ok,
        "missing": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate basic English QA and simple reasoning.")
    parser.add_argument("--mode", default="mimo-r4-tiny")
    parser.add_argument("--tokenizer", default="llama31")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-new", type=int, default=48)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--safe-decode", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--nan-check", action="store_true")
    args = parser.parse_args()

    setup_perf(args.device)
    normalize_runtime_flags(args)
    model, tokenizer, _ = load_or_new(SimpleNamespace(**vars(args), cmd="fast-generate", batch_size=1, state_out=Path("/tmp/state.pt")))
    model.eval()

    reports = []
    for case in CASES:
        result = _fast_decode_once(model, tokenizer, args, case["prompt"])
        text = result["text"]
        completion = text[len(case["prompt"]):] if text.startswith(case["prompt"]) else text
        report = score_case(case["prompt"], completion, case)
        report["raw_output"] = text
        report["metrics"] = {k: v for k, v in result.items() if k != "text"}
        reports.append(report)

    passed = sum(1 for item in reports if item["ok"])
    payload = {
        "ok": passed == len(reports),
        "passed": passed,
        "total": len(reports),
        "mode": args.mode,
        "tokenizer": args.tokenizer,
        "checkpoint": str(args.checkpoint),
        "reports": reports,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
