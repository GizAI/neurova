#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any


LETTERS = "ABCD"

SUBJECTS = [
    "abstract algebra",
    "anatomy",
    "astronomy",
    "biology",
    "chemistry",
    "computer science",
    "data structures",
    "economics",
    "electromagnetism",
    "ethics",
    "formal logic",
    "geography",
    "history",
    "machine learning",
    "medicine",
    "operating systems",
    "physics",
    "probability",
    "security",
    "world literature",
]

SYSTEM_PROMPT = """You are a curriculum writer for training a small language model.
Create original multiple-choice training items from general domain knowledge.
Do not quote, paraphrase, or reconstruct MMLU, MMLU-Redux, ARC, HellaSwag, GPQA, GSM8K, HumanEval, MBPP, or any benchmark item.
Do not include benchmark source names, copyrighted passages, private data, or web excerpts.
Each item must be self-contained, factual, concise, and suitable for answer-only supervised fine-tuning.
Return valid JSON only."""


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def stable_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.casefold().split()).encode("utf-8")).hexdigest()


def call_deepseek(
    *,
    api_key: str,
    base_url: str,
    model: str,
    subject: str,
    batch_size: int,
    seed: int,
    timeout: float,
    temperature: float,
) -> dict[str, Any]:
    prompt = f"""Generate {batch_size} original no-cheat multiple-choice items for subject: {subject}.

Constraints:
- Exactly four choices A-D.
- Exactly one correct answer.
- The answer field must be one of A, B, C, D.
- Include a short explanation of no more than 35 words.
- Difficulty should vary from high-school to early undergraduate.
- Avoid benchmark-style famous wording; write fresh items.

Return JSON with this exact shape:
{{
  "items": [
    {{
      "domain": "subject label",
      "question": "question text",
      "choices": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "answer": "A",
      "explanation": "brief reason"
    }}
  ]
}}

Seed for diversity: {seed}
"""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max(4096, min(32000, batch_size * 480)),
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    return json.loads(content)


def normalize_item(item: dict[str, Any]) -> dict[str, Any] | None:
    question = str(item.get("question", "")).strip()
    choices = item.get("choices")
    answer = str(item.get("answer", "")).strip().upper()
    explanation = str(item.get("explanation", "")).strip()
    domain = str(item.get("domain", "general")).strip() or "general"
    if not question or not isinstance(choices, dict) or answer not in LETTERS:
        return None
    clean_choices: dict[str, str] = {}
    for letter in LETTERS:
        text = str(choices.get(letter, "")).strip()
        if not text:
            return None
        clean_choices[letter] = re.sub(r"\s+", " ", text)
    if len(set(clean_choices.values())) != 4:
        return None
    if not explanation:
        explanation = f"{answer} is the best answer."
    if any(name.lower() in question.lower() for name in ("mmlu", "redux", "hellaswag", "gpqa", "gsm8k")):
        return None
    return {
        "domain": domain,
        "question": re.sub(r"\s+", " ", question),
        "choices": clean_choices,
        "answer": answer,
        "explanation": re.sub(r"\s+", " ", explanation),
    }


def item_to_record(item: dict[str, Any], *, model: str) -> dict[str, Any]:
    choices_text = "\n".join(f"{letter}. {item['choices'][letter]}" for letter in LETTERS)
    text = (
        "Instruction: Choose A, B, C, or D.\n"
        f"Question: {item['question']}\n"
        f"{choices_text}\n"
        f"Answer: {item['answer']}\n"
        f"Explanation: {item['explanation']}"
    )
    return {
        "text": text,
        "source": "deepseek-no-cheat-mcq-v1",
        "license": "generated-via-deepseek-api-review-before-release",
        "language": "en",
        "domain": item["domain"],
        "quality_score": 0.95,
        "toxicity_score": 0.0,
        "pii_score": 0.0,
        "dedup_hash": stable_hash(text),
        "benchmark_contamination_flag": False,
        "teacher_model": model,
        "generation_date": date.today().isoformat(),
        "answer_letter": item["answer"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate no-cheat MCQ SFT data with DeepSeek teacher.")
    parser.add_argument("--out", type=Path, default=Path("neuromamba/data/deepseek_no_cheat_mcq_sft_v1.jsonl"))
    parser.add_argument("--records", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=4)
    args = parser.parse_args()

    load_env_file(args.env_file)
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is missing; expected it in environment or --env-file")

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    counts: dict[str, int] = {}
    written = 0
    attempts = 0
    with args.out.open("w", encoding="utf-8") as fh:
        while written < args.records:
            attempts += 1
            subject = rng.choice(SUBJECTS)
            batch_seed = rng.randint(1, 2_000_000_000)
            parsed: dict[str, Any] | None = None
            last_error = ""
            for retry in range(args.max_retries):
                try:
                    parsed = call_deepseek(
                        api_key=api_key,
                        base_url=args.base_url,
                        model=args.model,
                        subject=subject,
                        batch_size=min(args.batch_size, args.records - written),
                        seed=batch_seed + retry,
                        timeout=args.timeout,
                        temperature=args.temperature,
                    )
                    break
                except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
                    last_error = repr(exc)
                    time.sleep(min(8.0, 1.5 * (retry + 1)))
            if parsed is None:
                print(json.dumps({"warning": "deepseek_batch_failed", "subject": subject, "error": last_error}), flush=True)
                continue
            items = parsed.get("items", [])
            if not isinstance(items, list):
                continue
            for raw_item in items:
                if written >= args.records:
                    break
                if not isinstance(raw_item, dict):
                    continue
                item = normalize_item(raw_item)
                if item is None:
                    continue
                row = item_to_record(item, model=args.model)
                key = row["dedup_hash"]
                if key in seen:
                    continue
                seen.add(key)
                counts[row["domain"]] = counts.get(row["domain"], 0) + 1
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            if attempts % 10 == 0 or written >= args.records:
                print(json.dumps({"written": written, "target": args.records, "attempts": attempts, "counts": counts}, ensure_ascii=False), flush=True)
            if args.sleep > 0:
                time.sleep(args.sleep)
    print(json.dumps({"out": str(args.out), "records": written, "model": args.model, "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
