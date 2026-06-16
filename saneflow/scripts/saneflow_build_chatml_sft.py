#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saneflow.chat_format import DEFAULT_SYSTEM, IM_END, IM_START, format_chatml_pair


BENCHMARK_PATTERNS = re.compile(r"\b(GSM8K|MMLU|HumanEval|MBPP|AIME\s*(202[0-9])?|MATH benchmark)\b", re.I)


def digest(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def near_key(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.lower())
    normalized = re.sub(r"\d+", "#", normalized)
    normalized = re.sub(r"[^a-z가-힣# ]+", "", normalized)
    return hashlib.sha1(normalized[:1800].encode("utf-8")).hexdigest()


def canonicalize_text(text: str) -> str:
    text = text.strip()
    text = text.replace("You are LUMA", "You are Neurova")
    text = text.replace("I am LUMA", "I am Neurova")
    text = text.replace("LUMA", "Neurova")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def is_chatml(text: str) -> bool:
    return IM_START in text and IM_END in text and f"{IM_START}assistant" in text


def clip_chatml(text: str, max_chars: int) -> str | None:
    text = canonicalize_text(text)
    if len(text) <= max_chars:
        return text
    if text.count(f"{IM_START}assistant") != 1:
        return None
    prefix, rest = text.split(f"{IM_START}assistant\n", 1)
    answer = rest.split(IM_END, 1)[0].strip()
    budget = max_chars - len(prefix) - len(f"{IM_START}assistant\n") - len(f"\n{IM_END}")
    if budget < 80:
        return None
    answer = answer[:budget].rsplit(" ", 1)[0].strip()
    return f"{prefix}{IM_START}assistant\n{answer}\n{IM_END}"


def normalize_role(role: str) -> str:
    role = str(role).lower().strip()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant", "model"}:
        return "assistant"
    if role == "system":
        return "system"
    return "user"


def messages_to_chatml(messages: list[dict[str, str]], system: str = DEFAULT_SYSTEM) -> str | None:
    out: list[str] = []
    if not any(m.get("role") == "system" for m in messages):
        out.append(f"{IM_START}system\n{system.strip()}\n{IM_END}")
    for msg in messages:
        role = normalize_role(msg.get("role", "user"))
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        out.append(f"{IM_START}{role}\n{content}\n{IM_END}")
    text = "\n".join(out)
    return text if f"{IM_START}assistant" in text else None


def row_to_chatml(row: dict[str, Any]) -> str | None:
    text = str(row.get("text") or "").strip()
    if is_chatml(text):
        return canonicalize_text(text)
    match = re.match(r"Instruction:\s*(.*?)\s*Answer:\s*(.*)\s*$", text, flags=re.DOTALL)
    if match:
        return format_chatml_pair(match.group(1).strip(), match.group(2).strip())
    if "<usr>" in text and "<bot>" in text:
        user = text.split("<usr>", 1)[1].split("<bot>", 1)[0].strip()
        assistant = text.split("<bot>", 1)[1].strip()
        return format_chatml_pair(user, assistant)
    if "messages" in row and isinstance(row["messages"], list):
        return messages_to_chatml([{"role": m.get("role", "user"), "content": m.get("content", "")} for m in row["messages"]])
    if "conversations" in row and isinstance(row["conversations"], list):
        messages = [{"role": m.get("from", "user"), "content": m.get("value", "")} for m in row["conversations"]]
        return messages_to_chatml(messages, system=str(row.get("system") or DEFAULT_SYSTEM))
    if "input" in row and "output" in row:
        user = str(row.get("input") or "").strip()
        assistant = str(row.get("output") or "").strip()
        return format_chatml_pair(user, assistant) if user and assistant else None
    task = str(row.get("task") or "").strip()
    answer = str(row.get("answer") or "").strip()
    if text and answer:
        user = text.split("Answer:", 1)[0].strip()
        if user.startswith("Instruction:"):
            user = user[len("Instruction:") :].strip()
        return format_chatml_pair(user, answer)
    if task and answer:
        return format_chatml_pair(task, answer)
    return None


def bucket_from_source(source: str, default: str) -> str:
    s = source.lower()
    if "tulu3" in s:
        return "tulu3"
    if "smoltalk2" in s:
        return "smoltalk2"
    if "opencode" in s:
        return "code"
    if any(x in s for x in ("openmath", "openthought", "openr1", "stratos", "limo", "deepseek", "slotproof", "memory", "reasoning")):
        return "reasoning"
    if any(x in s for x in ("tool", "apigen", "xlam")):
        return "tool"
    if any(x in s for x in ("korean", "kit19", "ko", "luma_clean", "dialogue")):
        return "korean"
    if any(x in s for x in ("guard", "safety", "polyguard", "wildguard")):
        return "safety"
    return default


def append_candidate(
    pools: dict[str, list[dict[str, Any]]],
    *,
    text: str | None,
    source: str,
    bucket: str,
    max_chars: int,
) -> bool:
    if not text:
        return False
    if BENCHMARK_PATTERNS.search(text):
        return False
    text = clip_chatml(text, max_chars)
    if not text or not is_chatml(text):
        return False
    pools[bucket].append({"text": text, "source": source, "dedup_hash": digest(text), "near_hash": near_key(text), "bucket": bucket})
    return True


def iter_json_file(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".json":
        txt = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not txt:
            return
        if txt[0] == "[":
            for row in json.loads(txt):
                yield row
            return
        yield json.loads(txt)
        return
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_source_rows(spec: dict[str, Any]) -> Iterable[tuple[dict[str, Any], str]]:
    typ = spec["type"]
    if typ == "jsonl":
        for path_s in spec.get("paths", []):
            path = Path(path_s)
            if not path.exists():
                continue
            for row in iter_json_file(path):
                yield row, str(row.get("source") or spec["name"])
        return

    raise ValueError(f"unknown source type: {typ}")


def source_row_to_chatml(spec: dict[str, Any], row: dict[str, Any]) -> str | None:
    typ = spec["type"]
    if typ == "jsonl":
        return row_to_chatml(row)
    return row_to_chatml(row)


def add_synthetic(pools: dict[str, list[dict[str, Any]]], seed: int, max_chars: int) -> None:
    rng = random.Random(seed)
    pairs = [
        ("hi", "Hello. I am Neurova, a local assistant. I can answer short questions and explain simple ideas clearly.", "tulu3"),
        ("Who are you?", "I am Neurova, a concise local assistant designed for direct answers, simple reasoning, and short explanations.", "tulu3"),
        ("What can you do?", "I can explain concepts, answer basic questions, summarize short text, help with simple math, and say when I do not know.", "tulu3"),
        ("What should you do if you do not know?", "I should say that I do not know. I should not invent facts or pretend to be certain.", "safety"),
        ("Explain photosynthesis in simple words.", "Photosynthesis is the process plants use to make food. They use sunlight, water, and carbon dioxide to make sugar and release oxygen.", "reasoning"),
        ("Why do airplanes fly?", "Airplanes fly because their wings create lift as air moves around them, and their engines provide thrust to keep them moving forward.", "reasoning"),
        ("Name three planets other than Earth.", "Three planets other than Earth are Mars, Jupiter, and Venus.", "reasoning"),
        ("Is Paris in Germany? Answer briefly.", "No. Paris is in France, not Germany.", "reasoning"),
        ("What is a computer?", "A computer is an electronic machine that stores and processes information by following instructions called programs.", "reasoning"),
        ("Explain why the sky looks blue.", "The sky looks blue because air scatters blue light from the sun more strongly than many other colors.", "reasoning"),
        ("Translate to Korean: I like learning new things.", "나는 새로운 것을 배우는 것을 좋아합니다.", "korean"),
        ("한국어로 짧게 자기소개해봐.", "안녕하세요. 저는 Neurova입니다. 간단하고 명확하게 답하는 로컬 어시스턴트입니다.", "korean"),
        ("Write a Python function that adds two numbers.", "```python\ndef add(a, b):\n    return a + b\n```", "code"),
    ]
    for _ in range(20):
        for user, answer, bucket in pairs:
            append_candidate(pools, text=format_chatml_pair(user, answer), source=f"synthetic-{bucket}", bucket=bucket, max_chars=max_chars)
    for _ in range(1600):
        a, b = rng.randint(1, 90), rng.randint(1, 90)
        append_candidate(
            pools,
            text=format_chatml_pair(f"What is {a} plus {b}?", f"{a} plus {b} is {a + b}."),
            source="synthetic-arithmetic",
            bucket="reasoning",
            max_chars=max_chars,
        )
    codes = ["AX", "BK", "NV", "QZ", "LM", "RF"]
    for _ in range(1200):
        code = f"{rng.choice(codes)}-{rng.randint(100, 999)}"
        append_candidate(
            pools,
            text=format_chatml_pair(f"Remember this code: {code}. What is the code?", f"The code is {code}."),
            source="synthetic-copy",
            bucket="reasoning",
            max_chars=max_chars,
        )


def build(recipe: dict[str, Any], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    max_chars_cfg = recipe["max_chars"]
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reports: list[dict[str, Any]] = []
    add_synthetic(pools, seed, int(max_chars_cfg["default"]))

    for idx, spec in enumerate(recipe["sources"]):
        scanned = added = 0
        limit = int(spec.get("limit") or 10**9)
        default_bucket = spec.get("bucket", "reasoning")
        max_chars = int(max_chars_cfg.get(default_bucket, max_chars_cfg["default"]))
        try:
            rows = list(iter_source_rows(spec))
        except Exception as exc:
            reports.append({"source": spec["name"], "error": str(exc)[:500], "added": 0})
            continue
        rng.shuffle(rows)
        for row, row_source in rows:
            scanned += 1
            source = f"{spec['name']}:{row_source}" if row_source != spec["name"] else spec["name"]
            bucket = bucket_from_source(source, default_bucket)
            text = source_row_to_chatml(spec, row)
            if append_candidate(pools, text=text, source=source, bucket=bucket, max_chars=int(max_chars_cfg.get(bucket, max_chars_cfg["default"]))):
                added += 1
                if added >= limit:
                    break
        reports.append({"source": spec["name"], "type": spec["type"], "scanned": scanned, "added": added, "bucket": default_bucket})

    selected: list[dict[str, Any]] = []
    seen_exact: set[str] = set()
    seen_near: set[str] = set()
    target_records = int(recipe["target_records"])
    for bucket, ratio in recipe["buckets"].items():
        want = int(round(target_records * float(ratio)))
        rows = pools.get(bucket, [])
        rng.shuffle(rows)
        taken = 0
        for row in rows:
            if row["dedup_hash"] in seen_exact or row["near_hash"] in seen_near:
                continue
            seen_exact.add(row["dedup_hash"])
            seen_near.add(row["near_hash"])
            selected.append(row)
            taken += 1
            if taken >= want:
                break
        reports.append({"bucket": bucket, "wanted": want, "available": len(rows), "selected": taken})

    if len(selected) < target_records:
        remainder = [r for rows in pools.values() for r in rows]
        rng.shuffle(remainder)
        for row in remainder:
            if row["dedup_hash"] in seen_exact or row["near_hash"] in seen_near:
                continue
            seen_exact.add(row["dedup_hash"])
            seen_near.add(row["near_hash"])
            selected.append(row)
            if len(selected) >= target_records:
                break

    rng.shuffle(selected)
    valid_count = min(int(recipe["valid_records"]), max(1, len(selected) // 10))
    valid = selected[:valid_count]
    train = selected[valid_count:]
    manifest = {
        "name": recipe["name"],
        "format": recipe["format"],
        "loss_mode": recipe["loss_mode"],
        "target_records": target_records,
        "train_records": len(train),
        "valid_records": len(valid),
        "train_bucket_counts": Counter(r["bucket"] for r in train),
        "valid_bucket_counts": Counter(r["bucket"] for r in valid),
        "reports": reports,
    }
    return train, valid, manifest


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            clean = {k: row[k] for k in ("text", "source", "dedup_hash", "bucket")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical ratio-controlled ChatML SFT data for SaneFlow.")
    parser.add_argument("--recipe", type=Path, default=Path("saneflow/configs/saneflow_chatml_sft_recipe.json"))
    parser.add_argument("--train-out", type=Path, default=Path("saneflow/data/corpus/mixes/saneflow_chatml_sft_train_v1.jsonl"))
    parser.add_argument("--valid-out", type=Path, default=Path("saneflow/data/corpus/mixes/saneflow_chatml_sft_valid_v1.jsonl"))
    parser.add_argument("--manifest-out", type=Path, default=Path("saneflow/data/corpus/mixes/saneflow_chatml_sft_manifest_v1.json"))
    parser.add_argument("--seed", type=int, default=20260614)
    args = parser.parse_args()

    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    train, valid, manifest = build(recipe, args.seed)
    write_jsonl(args.train_out, train)
    write_jsonl(args.valid_out, valid)
    manifest["train"] = str(args.train_out)
    manifest["valid"] = str(args.valid_out)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=dict), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
