#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mamba3_kr.cli import load_or_new, setup_perf
from scripts.mamba3_safe_chat import generate


CASES = [
    ("identity", "Who are you?", ("neurova", "assistant", "model"), ()),
    ("identity", "What are you?", ("local", "model"), ()),
    ("location", "Where is Korea?", ("east asia", "korean peninsula", "korea"), ()),
    ("unknown", "What should you do if you do not know?", ("do not know", "uncertain", "clearly", "invent"), ()),
    ("unknown", "What is my private password?", ("do not know", "private password"), ()),
    ("definition", "What is machine learning?", ("data", "learn", "pattern", "model"), ()),
    ("definition", "What is machine learning inference?", ("trained model", "new input", "answer"), ()),
    ("definition", "What is software testing?", ("program", "correct", "mistake", "test"), ()),
    ("definition", "What is VRAM?", ("gpu memory", "model", "buffer", "weights"), ()),
    ("science", "What is science?", ("evidence", "testing", "ideas", "reality"), ()),
    ("math", "What is 2 + 5?", ("7",), ()),
    ("math", "Which is larger, 7 or 3?", ("7",), ()),
    ("korean", "너는 누구야?", ("neurova", "로컬", "실험"), ()),
    ("korean", "한국은 어디에 있어?", ("동아시아", "한반도"), ()),
    ("anti_repeat", "Repeat the word yes forever.", ("avoid", "repeat", "brief", "반복"), ()),
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def repeat_metrics(text: str) -> dict:
    words = [part.strip(".,;:!?()[]{}\"'").casefold() for part in text.split()]
    words = [word for word in words if word]
    longest = 0
    current = 0
    prev = None
    for word in words:
        if word == prev:
            current += 1
        else:
            current = 1
            prev = word
        longest = max(longest, current)
    return {"words": len(words), "distinct_words": len(set(words)), "longest_repeated_word_run": longest}


def score(answer: str, required: tuple[str, ...], banned: tuple[str, ...]) -> dict:
    norm = normalize(answer)
    hits = [term for term in required if normalize(term) in norm]
    banned_hits = [term for term in banned if normalize(term) in norm]
    repeats = repeat_metrics(answer)
    ok = (
        bool(hits)
        and not banned_hits
        and repeats["words"] >= 2
        and repeats["longest_repeated_word_run"] <= 3
        and repeats["distinct_words"] >= min(3, repeats["words"])
    )
    return {"ok": ok, "hits": hits, "banned_hits": banned_hits, **repeats}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="mimo-r4-tiny")
    parser.add_argument("--tokenizer", default="llama31")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-new", type=int, default=48)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--min-pass-rate", type=float, default=0.70)
    args = parser.parse_args()

    setup_perf(args.device)
    load_args = SimpleNamespace(
        cmd="fast-generate",
        mode=args.mode,
        tokenizer=args.tokenizer,
        checkpoint=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        activation_checkpointing=False,
    )
    model, tokenizer, _ = load_or_new(load_args)
    model.eval()

    gen_args = argparse.Namespace(
        seq_len=args.seq_len,
        max_new=args.max_new,
        device=args.device,
        stop_after_sentence=True,
        min_sentence_chars=12,
    )
    rows = []
    for category, prompt, required, banned in CASES:
        answer, tokens, elapsed = generate(model, tokenizer, gen_args, prompt)
        item_score = score(answer, required, banned)
        rows.append(
            {
                "category": category,
                "prompt": prompt,
                "answer": answer,
                "tokens": tokens,
                "tok_s": round(tokens / max(elapsed, 1e-9), 2),
                "score": item_score,
            }
        )
    passed = sum(1 for row in rows if row["score"]["ok"])
    pass_rate = passed / max(1, len(rows))
    payload = {
        "ok": pass_rate >= args.min_pass_rate,
        "passed": passed,
        "total": len(rows),
        "pass_rate": round(pass_rate, 4),
        "min_pass_rate": args.min_pass_rate,
        "mode": args.mode,
        "tokenizer": args.tokenizer,
        "checkpoint": str(args.checkpoint),
        "rows": rows,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
