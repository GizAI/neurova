from __future__ import annotations

import argparse
import contextlib
import io
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.luma_chat import format_prompt, generate_stream, load_model


DEFAULT_PROMPTS = [
    "hi",
    "who are you?",
    "what is machine learning?",
    "한국어로 짧게 인공지능이 뭐야?",
    "Give JSON with key ok true.",
    "If you do not know the answer, what should you do?",
]


def repetition_score(text: str, n: int = 4) -> int:
    chars = [ch for ch in text if not ch.isspace()]
    if len(chars) < n:
        return 0
    grams = Counter(tuple(chars[idx : idx + n]) for idx in range(len(chars) - n + 1))
    return max(grams.values(), default=0)


def passed(prompt: str, text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 2:
        return False
    if "\ufffd" in stripped:
        return False
    if repetition_score(stripped) >= 5:
        return False
    lower = stripped.lower()
    if prompt == "hi":
        return any(word in lower for word in ["hello", "hi", "luma", "안녕"])
    if "who are you" in prompt:
        return any(word in lower for word in ["luma", "assistant", "model"])
    if "machine learning" in prompt:
        return any(word in lower for word in ["data", "learn", "model", "pattern"])
    if "json" in prompt.lower():
        return "ok" in lower and "true" in lower
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed chat sanity probes and write JSON metrics.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--max-new", type=int, default=80)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
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
    prompts = args.prompt or DEFAULT_PROMPTS
    rows = []
    for prompt in prompts:
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
            system="You are LUMA, a concise helpful assistant. Answer directly.",
            ablation=args.ablation,
        )
        chat_prompt = format_prompt(gen_args.system, [], prompt)
        with contextlib.redirect_stdout(io.StringIO()):
            text, tokens, elapsed, _ = generate_stream(model, tokenizer, chat_prompt, gen_args)
        rows.append(
            {
                "prompt": prompt,
                "response": text.strip(),
                "tokens": tokens,
                "tok_per_sec": round(tokens / max(elapsed, 1e-9), 2),
                "replacement_char": "\ufffd" in text,
                "repeat4_max": repetition_score(text),
                "pass": passed(prompt, text),
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
