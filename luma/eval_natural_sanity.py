from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.luma_chat import generate_stream, load_model


DEFAULT_PROMPTS = [
    ("Question: hi\nAnswer:", ["hello", "hi", "안녕"]),
    ("Question: who are you?\nAnswer:", ["luma", "assistant", "model"]),
    ("Question: what is machine learning?\nAnswer:", ["data", "learn", "model", "pattern"]),
    ("Question: 한국어로 짧게 인공지능이 뭐야?\nAnswer:", ["인공지능", "컴퓨터", "언어", "예측"]),
    ("Question: Give JSON with key ok true.\nAnswer:", ["ok", "true"]),
]


def repetition_score(text: str, n: int = 4) -> int:
    chars = [ch for ch in text if not ch.isspace()]
    if len(chars) < n:
        return 0
    grams = Counter(tuple(chars[idx : idx + n]) for idx in range(len(chars) - n + 1))
    return max(grams.values(), default=0)


def has_word(text: str) -> bool:
    return bool(re.search(r"[A-Za-z가-힣0-9]{2,}", text))


def first_answer(text: str) -> str:
    clean = text.strip()
    stops = ["\nQuestion:", "\nAnswer:", "<|im_end|>", "<|im_start|>"]
    cut = len(clean)
    for stop in stops:
        pos = clean.find(stop)
        if pos > 0:
            cut = min(cut, pos)
    return clean[:cut].strip()


def passed(text: str, required: list[str]) -> bool:
    clean = first_answer(text)
    if len(clean) < 2:
        return False
    if "\ufffd" in clean:
        return False
    if repetition_score(clean) >= 5:
        return False
    if not has_word(clean):
        return False
    lower = clean.lower()
    return any(item.lower() in lower for item in required)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate plain natural QA completion before ChatML SFT.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--max-new", type=int, default=80)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    parser.add_argument("--ablation", default="normal")
    parser.add_argument("--prompt", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model(args)
    prompt_specs = [(prompt, []) for prompt in args.prompt] if args.prompt else DEFAULT_PROMPTS
    rows = []
    for prompt, required in prompt_specs:
        gen_args = SimpleNamespace(
            ckpt=args.ckpt,
            max_new=args.max_new,
            context=args.context,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=1.08,
            no_repeat_ngram=4,
            greedy=False,
            device=args.device,
            dtype=args.dtype,
            ablation=args.ablation,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            text, tokens, elapsed, _ = generate_stream(model, tokenizer, prompt, gen_args)
        answer = first_answer(text)
        ok = passed(text, required) if required else len(answer) >= 2 and repetition_score(answer) < 5
        rows.append(
            {
                "prompt": prompt,
                "response": text.strip(),
                "first_answer": answer,
                "tokens": tokens,
                "tok_per_sec": round(tokens / max(elapsed, 1e-9), 2),
                "replacement_char": "\ufffd" in text,
                "repeat4_max": repetition_score(answer),
                "pass": ok,
            }
        )
    summary = {
        "ckpt": args.ckpt,
        "passed": sum(1 for row in rows if row["pass"]),
        "total": len(rows),
        "pass_rate": round(sum(1 for row in rows if row["pass"]) / max(1, len(rows)), 4),
        "rows": rows,
    }
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
